#!/usr/bin/env python
"""H3: the cycle-model sweeps that answer the Gate 1 report's open questions (`docs/gate1-report.md` §4) with
numbers (`docs/phase5-cycle-model-task.md` §3). One parameter at a time, on the relevant family, under the
phase-5 config-C timing descriptor (`CONFIG_C_PHASE5`) unless stated; the v1 KV260 descriptor is run for the
baseline table. Phase 6 (ISA v0.1.4 §11: rings, striping, broadcast, cores) is `model/sweeps_mlp.py`.

    python model/sweeps.py                          # all sweeps -> model/cycle_results.json, model/cycle_results.md
    python model/sweeps.py --only baseline,inp      # a subset

Needs the TileLang interpreter (the programs are emitted from the kernels; see compiler/prototype/PINNED.md).
Every run is a `CycleMachine` run: the functional result is checked against the fixture each time, so a sweep
can never change numerics (task §5).
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from dataclasses import asdict, replace

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "compiler", "prototype"))
sys.path.insert(0, os.path.join(HERE, "..", "sim"))
sys.path.insert(0, os.path.join(HERE, "..", "ref"))

import common  # noqa: E402
import gemm_kernel  # noqa: E402
import kda_kernel  # noqa: E402
import kda_kernel_roundtrip  # noqa: E402
import mla_kernel  # noqa: E402
import moe_data  # noqa: E402
import moe_kernel  # noqa: E402
import overlap_report  # noqa: E402
from cut import lower_to_tile_tir  # noqa: E402
from fa_kernel import build_fa_kernel  # noqa: E402
from tcta import DeviceConfig, Fault  # noqa: E402
from tcta.cycle import CycleDeadlock, CycleMachine  # noqa: E402
from tcta.isa import SP  # noqa: E402
from tcta.timing import CONFIG_C_PHASE5, KV260_PLAN, DeviceTiming  # noqa: E402

# the phase-5/6 tables were made with the planning descriptor of the KV260 core; the calibrated `KV260`
# (phase 7, docs/calibration-report.md) is compared against the RTL by rtl/test/calibrate.py
KV260 = KV260_PLAN

# These are the phase-5 sweeps: they run on the phase-5 descriptor (3-deep rings, one request per channel, one
# reader per bank, FIFO 16, WGT 16) so that `python model/sweeps.py` reproduces the phase-5 tables and the
# `module0` / `module0_striped` keys. The v0.1.4 descriptor and its sweeps are `model/sweeps_mlp.py` (phase 6).
CONFIG_C = CONFIG_C_PHASE5
from tile_tir_to_program import tile_tir_to_program  # noqa: E402

FIX = os.path.join(HERE, "..", "fixtures")
TOL = 2.0 ** -7


def fixture(name):
    return dict(np.load(os.path.join(FIX, name, "data.npz")))


# --------------------------------------------------------------------------
# families: program + functional device config + DRAM loader + fixture check
# --------------------------------------------------------------------------

class Family:
    def __init__(self, name, prog, rep, cfg, dram_bytes, load, check, units, unit_name):
        self.name, self.prog, self.rep, self.cfg, self.dram_bytes = name, prog, rep, cfg, dram_bytes
        self.load, self.check, self.units, self.unit_name = load, check, units, unit_name


_cache: dict = {}


def fam_gemm():
    fx = fixture("gemm_partial")
    cut = lower_to_tile_tir(gemm_kernel.build_gemm_kernel(), scope_map=gemm_kernel.SCOPE_MAP)
    prog, rep = tile_tir_to_program(cut, bx=0, dram=gemm_kernel.DRAM)
    return Family("GEMM", prog, rep, None, gemm_kernel.DRAM_BYTES, lambda m: gemm_kernel.load_dram(m, fx),
                  lambda m: common.rowmax_err(gemm_kernel.read_c(m, fx), fx["C"]), 1, "program")


FA_DRAM = {"Q": 0, "K": 1 << 19, "V": 1 << 20, "Output": 1 << 21}


def fam_fa():
    d = fixture("flash_attn_gqa")
    heads = d["Q"].shape[0]
    G = heads // d["K"].shape[0]
    kv = [h // G for h in range(heads)]

    def load(m):
        m.dram_write(FA_DRAM["Q"], common.bf16_bytes(d["Q"]).ravel())
        m.dram_write(FA_DRAM["K"], common.bf16_bytes(d["K"][kv]).ravel())
        m.dram_write(FA_DRAM["V"], common.bf16_bytes(d["V"][kv]).ravel())

    def check(m):
        seq, dim, blk = d["O"].shape[1], d["O"].shape[2], 64
        addr = FA_DRAM["Output"]
        bits = m.dram[addr: addr + blk * dim * 2].view(np.uint16).astype(np.uint32) << 16
        return common.rowmax_err(bits.view(np.float32).reshape(blk, dim), d["O"][0, :blk])

    cut = lower_to_tile_tir(build_fa_kernel(), cut="inject_pipeline")
    prog, rep = tile_tir_to_program(cut, bx=0, by=0, dram=FA_DRAM)
    return Family("FA", prog, rep, None, 4 << 20, load, check, 4, "KV tile")


def fam_mla(inp_slots=2, kv_unroll=1, cfg=None):
    """`kv_unroll` (phase 6): K ring of 5 * kv_unroll WGT slots (4 at kv_unroll 1)."""
    fx = fixture("mla_decode_absorbed")
    if cfg is None:
        cfg = DeviceConfig(slots={SP.INP: max(2, inp_slots), SP.WGT: max(4, 5 * kv_unroll), SP.ACC: 6, SP.VEC: 16, SP.STATE: 1})
    cut = lower_to_tile_tir(mla_kernel.build_mla_kernel(inp_slots=inp_slots, kv_unroll=kv_unroll), scope_map=mla_kernel.SCOPE_MAP)
    prog, rep = tile_tir_to_program(cut, bx=0, dram=mla_kernel.DRAM, cfg=cfg)
    O = fx["O"].reshape(-1, fx["O"].shape[-1])
    name = f"MLA(INP {inp_slots})" if kv_unroll == 1 else f"MLA(INP {inp_slots}, K ring {5 * kv_unroll})"
    return Family(name, prog, rep, cfg, mla_kernel.DRAM_BYTES, lambda m: mla_kernel.load_dram(m, fx),
                  lambda m: common.rowmax_err(mla_kernel.read_o(m, fx).reshape(-1, O.shape[-1]), O), 16, "KV tile")


def fam_kda(roundtrip=False):
    fx = fixture("kda_chunk64")
    mod = kda_kernel_roundtrip if roundtrip else kda_kernel
    cut = lower_to_tile_tir(mod.build_kda_kernel(), scope_map=mod.SCOPE_MAP)
    prog, rep = tile_tir_to_program(cut, bx=0, dram=mod.DRAM, cfg=kda_kernel.CONFIG_C)

    def check(m):
        o, S = mod.read_outputs(m, fx)
        return max([common.rowmax_err(o, fx["o"])] + [common.rowmax_err(S[c], fx["S_chunks"][c]) for c in range(S.shape[0])])

    return Family("KDA (v0.1.2 round trips)" if roundtrip else "KDA", prog, rep, kda_kernel.CONFIG_C, kda_kernel.DRAM_BYTES,
                  lambda m: mod.load_dram(m, fx), check, 4, "chunk")


def fam_moe(num_stages=3, prefetch=True, weight_format="f4.b32", flat_w13=False, cfg=None, expert_base=0, n_experts=None):
    """Phase 6: `flat_w13` (one 16-tile [W1|W3] stream, rings to 16; needs INP >= 4), `expert_base`/`n_experts`
    (the expert subset one core streams; the check covers the parts of the tokens routed to that subset)."""
    fx = fixture("moe_grouped")
    idx, w, dst = moe_data.pad_lists(fx)
    E = moe_kernel.E
    n_experts = E if n_experts is None else n_experts
    cut = lower_to_tile_tir(moe_kernel.build_moe_persistent_kernel(weight_format=weight_format, num_stages=num_stages, prefetch=prefetch,
                                                                   flat_w13=flat_w13, expert_base=expert_base, n_experts=n_experts),
                            scope_map=moe_kernel.SCOPE_MAP)
    prog, rep = tile_tir_to_program(cut, bx=0, dram=moe_data.DRAM, cfg=cfg)
    B, D, K = moe_kernel.B, moe_kernel.D, moe_kernel.TOPK
    subset = np.arange(expert_base, expert_base + n_experts)
    tiles = int(sum(-(-int(c) // moe_kernel.BLOCK_M) for c in fx["counts"][subset]))
    sel = np.isin(fx["topk_idx"], subset)                   # [B, k]: the parts this program produces

    def check(m):
        parts = moe_data.read_parts(m, B, K, D)
        return common.rowmax_err(parts[sel], fx["y_parts"][sel])

    name = f"MoE (ring {num_stages}{'' if prefetch else ', no prefetch'}{', flat' if flat_w13 else ''}"
    if n_experts != E:
        name += f", experts {expert_base}..{expert_base + n_experts - 1}"
    f = Family(name + ")", prog, rep, cfg, moe_data.DRAM_BYTES,
               lambda m: moe_data.load_dram(m, fx, idx, w, dst, weight_format), check, tiles, "token tile")
    f.counts = fx["counts"][subset]
    f.sel = sel
    return f


# --------------------------------------------------------------------------
# one run
# --------------------------------------------------------------------------

def run(fam: Family, timing: DeviceTiming, cfg: DeviceConfig | None = None) -> dict:
    cfg = cfg if cfg is not None else fam.cfg
    m = CycleMachine(timing=timing, cfg=cfg, dram_bytes=fam.dram_bytes)
    fam.load(m)
    t0 = time.time()
    try:
        st = m.run(fam.prog.instrs)
    except CycleDeadlock as e:
        return dict(family=fam.name, timing=timing.name, deadlock=str(e.heads), cycles=None)
    except Fault as e:
        return dict(family=fam.name, timing=timing.name, fault=str(e), cycles=None)
    t = m.timing_report
    err = fam.check(m)
    assert err < TOL, (fam.name, err)
    out = dict(family=fam.name, timing=timing.name, ok=st["ok"], err=float(err), cycles=t.cycles, us=t.us,
               per_unit=t.cycles / fam.units, units=fam.units, unit_name=fam.unit_name,
               eta_mfu=t.eta_mfu, eta_bw=t.eta_bw, macs=t.macs, dram_read=t.dram_read_bytes, dram_write=t.dram_write_bytes,
               per_queue={k: {kk: vv for kk, vv in v.items() if kk not in ("first", "last")} for k, v in t.per_queue.items()},
               edges=t.edges, channels=t.channels, banks=t.banks, fe_loop_wait=t.fe_loop_wait, wall_s=time.time() - t0,
               mean_load2d=statistics.mean([r.end - r.start for r in t.recs if r.op == "LOAD.2D"] or [0]),
               mean_loadidx=statistics.mean([r.end - r.start for r in t.recs if r.op == "LOAD.IDX"] or [0]))
    out["_timing"] = t
    out["_machine"] = m
    return out


def clean(r):
    return {k: v for k, v in r.items() if not k.startswith("_")}


def pct(a, b):
    return f"{100.0 * (a - b) / b:+.1f}%" if b else "—"


def busy(r, q):
    return r["per_queue"].get(q, {}).get("busy", 0)


# --------------------------------------------------------------------------
# sweeps
# --------------------------------------------------------------------------

def sweep_baseline(fams):
    rows = []
    for fam in fams:
        for T in (KV260, CONFIG_C):
            r = run(fam, T)
            rows.append(clean(r))
    md = ["| family | timing | cycles | µs | per " + "unit" + " (cycles) | η_mfu | η_bw | MMA busy | VEC busy | LOAD busy | token-stall MMA | fifo-blocked | bank-stall |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        if r.get("cycles") is None:
            md.append(f"| {r['family']} | {r['timing']} | deadlock | | | | | | | | | | |")
            continue
        fb = sum(e["push_blocked_cycles"] for e in r["edges"].values())
        bs = sum(r["banks"].values())
        md.append(f"| {r['family']} | {r['timing']} | {r['cycles']} | {r['us']:.1f} | {r['per_unit']:.0f} / {r['unit_name']} | {r['eta_mfu']:.3f} | {r['eta_bw']:.3f} | "
                  f"{busy(r, 'MMA')} | {busy(r, 'VEC')} | {busy(r, 'LOAD')} | {r['per_queue'].get('MMA', {}).get('stall_token', 0)} | {fb} | {bs} |")
    return rows, "\n".join(md)


def sweep_inp(_):
    rows = []
    for n in (2, 3, 4, 5, 6):
        fam = fam_mla(inp_slots=n)
        T = replace(CONFIG_C, banks={**CONFIG_C.banks, SP.INP: max(CONFIG_C.banks[SP.INP], n)})
        r = run(fam, T)
        rows.append(clean(r) | dict(inp_slots=n, n_instrs=fam.rep.n_instrs, loads_per_tile=sum(1 for i in fam.prog.instrs if i.op == "LOAD.2D") ))
    base = rows[0]["cycles"]
    md = ["| INP slots | program | cycles | per KV tile | vs INP 2 | DRAM read (KB) | LOAD busy | MMA token-stall | η_mfu |", "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['inp_slots']} | {r['n_instrs']} instrs | {r['cycles']} | {r['per_unit']:.0f} | {pct(r['cycles'], base)} | {r['dram_read'] / 1024:.0f} | "
                  f"{busy(r, 'LOAD')} | {r['per_queue']['MMA']['stall_token']} | {r['eta_mfu']:.3f} |")
    return rows, "\n".join(md)


def boundary_stats(fam, r):
    """MoE expert-boundary numbers in cycles from the Gantt trace (overlap_report.boundaries on the cycle trace)."""
    t = r["_timing"]
    trace = t.step_trace(include_sync=True)
    rows = overlap_report.boundaries(fam.prog, trace, fam.counts, fam.rep)
    ov = [x[3] for x in rows]; idle = [x[6] for x in rows]; gap = [x[7] for x in rows]; ahead = [x[4] for x in rows]
    return dict(boundaries=len(rows), overlap_median=statistics.median(ov), overlap_min=min(ov), overlap_max=max(ov),
                load_idle_median=statistics.median(idle), mma_gap_median=statistics.median(gap), mma_gap_max=max(gap),
                loads_ahead_median=statistics.median(ahead))


def sweep_wgt_ring(_):
    rows = []
    for stages in (2, 3, 4):
        fam = fam_moe(num_stages=stages)
        r = run(fam, CONFIG_C)
        b = boundary_stats(fam, r)
        rows.append(clean(r) | b | dict(ring=stages, n_instrs=fam.rep.n_instrs))
    base = [x for x in rows if x["ring"] == 3][0]["cycles"]
    md = ["| WGT ring | program | cycles | per token tile | vs ring 3 | MMA busy | LOAD busy | MMA token-stall | boundary: MMA gap (median / max cycles) | LOAD idle at boundary (median) | e+1 LOADs before e's last MMA (median) |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['ring']} | {r['n_instrs']} | {r['cycles']} | {r['per_unit']:.0f} | {pct(r['cycles'], base)} | {busy(r, 'MMA')} | {busy(r, 'LOAD')} | "
                  f"{r['per_queue']['MMA']['stall_token']} | {r['mma_gap_median']:.0f} / {r['mma_gap_max']} | {r['load_idle_median']:.0f} | {r['loads_ahead_median']:.0f} |")
    return rows, "\n".join(md)


def sweep_fifo(fams):
    rows = []
    for fam in fams:
        for depth in (4, 8, 16):
            T = replace(CONFIG_C, fifo_depth=depth)
            cfg = replace(fam.cfg, fifo_depth=depth) if fam.cfg is not None else DeviceConfig(fifo_depth=depth)
            r = run(fam, T, cfg)
            hot = {}
            if r.get("cycles") is not None:
                hot = {k: v["push_blocked_cycles"] for k, v in r["edges"].items() if v["push_blocked_cycles"]}
                lvl = {k: v["max_level"] for k, v in r["edges"].items()}
            rows.append(clean(r) | dict(depth=depth, hot_edges=hot, levels=lvl if r.get("cycles") is not None else {}))
    md = ["| family | FIFO depth | cycles | PUSH blocked (cycles, per edge) | max dynamic level per edge |", "|---|---|---|---|---|"]
    for r in rows:
        if r.get("cycles") is None:
            md.append(f"| {r['family']} | {r['depth']} | **deadlock** | {r.get('deadlock', r.get('fault'))} | |")
        else:
            md.append(f"| {r['family']} | {r['depth']} | {r['cycles']} | {r['hot_edges'] or '—'} | {r['levels']} |")
    return rows, "\n".join(md)


def sweep_gather(_):
    rows = []
    for fam in (fam_moe(), fam_kda()):
        for base in (KV260, CONFIG_C):
            for mode in ("row", "burst64"):
                T = replace(base, row_gather_cost=mode)
                if fam.cfg is not None and base.fifo_depth != fam.cfg.fifo_depth:
                    T = replace(T, fifo_depth=fam.cfg.fifo_depth)
                r = run(fam, T)
                rows.append(clean(r) | dict(mode=mode, base=base.name))
    md = ["| family | timing | gather cost | cycles | per unit | LOAD busy | LOAD token-stall | η_bw |", "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['family']} | {r['base']} | {r['mode']} | {r['cycles']} | {r['per_unit']:.0f} / {r['unit_name']} | {busy(r, 'LOAD')} | "
                  f"{r['per_queue']['LOAD']['stall_token']} | {r['eta_bw']:.3f} |")
    return rows, "\n".join(md)


def sweep_vec(_):
    rows = []
    for fam in (fam_mla(), fam_fa()):
        for lanes in (16, 32, 64, 128):
            for fused in (False, True):
                T = replace(CONFIG_C, vec_lanes=lanes, sfu_lanes=max(8, lanes // 8), sfu_fused_sub_sum=fused)
                r = run(fam, T)
                rows.append(clean(r) | dict(vec_lanes=lanes, sfu_lanes=T.sfu_lanes, fused=fused))
    md = ["| family | VEC lanes | SFU lanes | fused exp2-sum | cycles | per KV tile | MMA busy / arrays | VEC busy | VEC share of tile | MMA token-stall |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        mma = busy(r, "MMA") / CONFIG_C.n_arrays
        md.append(f"| {r['family']} | {r['vec_lanes']} | {r['sfu_lanes']} | {r['fused']} | {r['cycles']} | {r['per_unit']:.0f} | {mma:.0f} | {busy(r, 'VEC')} | "
                  f"{100 * busy(r, 'VEC') / r['cycles']:.0f}% | {r['per_queue']['MMA']['stall_token']} |")
    return rows, "\n".join(md)


def sweep_latency(_):
    rows = []
    fams = [fam_kda(), fam_kda(roundtrip=True), fam_moe(), fam_moe(prefetch=False)]
    for fam in fams:
        for lat in (100, 200, 300, 400, 600):
            T = replace(CONFIG_C, dram_latency_cycles=lat)
            r = run(fam, T)
            extra = {}
            if hasattr(fam, "counts"):
                extra = dict(fe_loop_wait=r["fe_loop_wait"])
            rows.append(clean(r) | dict(latency=lat) | extra)
    md = ["| family | dram_latency | cycles | per unit | vs latency 300 | LOAD busy | LOAD token-stall | STORE busy | FE wait on VCOUNT→LOOP |", "|---|---|---|---|---|---|---|---|---|"]
    for fam in fams:
        base = [x for x in rows if x["family"] == fam.name and x["latency"] == 300][0]["cycles"]
        for r in [x for x in rows if x["family"] == fam.name]:
            md.append(f"| {r['family']} | {r['latency']} | {r['cycles']} | {r['per_unit']:.0f} / {r['unit_name']} | {pct(r['cycles'], base)} | {busy(r, 'LOAD')} | "
                      f"{r['per_queue']['LOAD']['stall_token']} | {busy(r, 'STORE')} | {r.get('fe_loop_wait', 0)} |")
    return rows, "\n".join(md)


def sweep_channels(_):
    """What-if beyond the task's rule set: a LOAD.2D / STORE.2D request striped over k channels (address
    interleave) instead of occupying one channel. Under the one-channel rule a 32 KB weight tile transfers for
    1024 cycles at 32 B/cycle while the MMA that consumes it takes 256; this sweep prices the alternative."""
    rows = []
    for fam in (fam_moe(), fam_mla(), fam_fa(), fam_kda()):
        for k in (1, 2, 4, 8):
            T = replace(CONFIG_C, request_channels=k)
            r = run(fam, T)
            rows.append(clean(r) | dict(request_channels=k))
    md = ["| family | channels per LOAD.2D | cycles | per unit | vs 1 channel | η_bw | η_mfu | LOAD busy (channel-cycles) | mean LOAD.2D duration | MMA token-stall | FE wait on VCOUNT→LOOP |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        base = [x for x in rows if x["family"] == r["family"] and x["request_channels"] == 1][0]["cycles"]
        md.append(f"| {r['family']} | {r['request_channels']} | {r['cycles']} | {r['per_unit']:.0f} / {r['unit_name']} | {pct(r['cycles'], base)} | "
                  f"{r['eta_bw']:.3f} | {r['eta_mfu']:.3f} | {busy(r, 'LOAD')} | {r['mean_load2d']:.0f} | {r['per_queue']['MMA']['stall_token']} | {r['fe_loop_wait']} |")
    return rows, "\n".join(md)


def sweep_ports(_):
    """What-if beyond the task's rule set: read ports per scratchpad bank 1 / 2 / 4. Under one reader per bank the
    four arrays of config C serialize on a shared A operand (the MoE token tile is read by all 24 MMAs of a tile,
    the MLA Q tile by the 5 QK^T MMAs)."""
    rows = []
    for fam in (fam_moe(), fam_mla(), fam_kda()):
        for rd in (1, 2, 4):
            T = replace(CONFIG_C, ports_per_bank=(rd, 1))
            r = run(fam, T)
            rows.append(clean(r) | dict(readers=rd))
    md = ["| family | read ports per bank | cycles | per unit | vs 1 reader | η_mfu | MMA bank-stall | VEC bank-stall | LOAD/STORE bank-stall | top banks |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        base = [x for x in rows if x["family"] == r["family"] and x["readers"] == 1][0]["cycles"]
        pq = r["per_queue"]
        top = dict(sorted(r["banks"].items(), key=lambda x: -x[1])[:3])
        md.append(f"| {r['family']} | {r['readers']} | {r['cycles']} | {r['per_unit']:.0f} / {r['unit_name']} | {pct(r['cycles'], base)} | {r['eta_mfu']:.3f} | "
                  f"{pq.get('MMA', {}).get('stall_bank', 0)} | {pq.get('VEC', {}).get('stall_bank', 0)} | {pq.get('LOAD', {}).get('stall_bank', 0) + pq.get('STORE', {}).get('stall_bank', 0)} | {top} |")
    return rows, "\n".join(md)


def module0_hook(rows, source="model/sweeps.py baseline, config-C timing (model/cycle_results.json)"):
    """eta_bw / eta_mfu for module0.py from the config-C baseline: the MoE decode tile (weights + gathers; the
    bandwidth-bound op of decode) and the MLA tile."""
    def pick(name):
        return next(r for r in rows if r["family"].startswith(name) and r["timing"] == CONFIG_C.name)
    moe, mla = pick("MoE"), pick("MLA")
    return dict(eta_bw_moe=moe["eta_bw"], eta_mfu_moe=moe["eta_mfu"], eta_bw_mla=mla["eta_bw"], eta_mfu_mla=mla["eta_mfu"],
                moe_cycles_per_tile=moe["per_unit"], mla_cycles_per_tile=mla["per_unit"], source=source)


SWEEPS = {
    "baseline": ("Baseline: the five families under both descriptors", sweep_baseline),
    "inp": ("INP slots 2→6: MLA Q re-streaming (config C)", sweep_inp),
    "wgt": ("WGT ring 2→4: MoE expert-boundary overlap in cycles (config C)", sweep_wgt_ring),
    "fifo": ("FIFO depth 4 / 8 / 16: which edges stall (config C; the functional run at the same depth)", sweep_fifo),
    "gather": ("row vs burst64 gather cost: MoE tile, KDA chunk", sweep_gather),
    "vec": ("VEC lanes 16→128 and SFU exp2-sum fusion: MLA and FA tile time vs MMA time (config C)", sweep_vec),
    "latency": ("DRAM latency 100→600: KDA before/after the v0.1.3 round-trip removal, MoE count round trip (config C)", sweep_latency),
    "channels": ("What-if: a LOAD.2D striped over 1 / 2 / 4 / 8 channels (the task's rule is 1; config C)", sweep_channels),
    "ports": ("What-if: 1 / 2 / 4 read ports per scratchpad bank (the task's rule is 1; config C)", sweep_ports),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=",".join(SWEEPS))
    ap.add_argument("--out", default=os.path.join(HERE, "cycle_results.json"))
    ap.add_argument("--md", default=os.path.join(HERE, "cycle_results.md"))
    args = ap.parse_args()
    t0 = time.time()
    fams = [fam_gemm(), fam_fa(), fam_mla(), fam_kda(), fam_moe()]
    results, mds = {"timings": {"KV260": asdict(KV260), "CONFIG_C": asdict(CONFIG_C)}}, []
    for key in args.only.split(","):
        title, fn = SWEEPS[key]
        print(f"== {title}", flush=True)
        rows, md = fn(fams)
        results[key] = rows
        mds.append(f"## {title}\n\n{md}\n")
        print(md, flush=True)
    if "baseline" in results:
        results["module0"] = module0_hook(results["baseline"])
        print("module0:", results["module0"])
    if "channels" in results:
        results["module0_striped"] = module0_hook([r for r in results["channels"] if r["request_channels"] == 8],
                                                  source="model/sweeps.py channels sweep, config-C timing with request_channels=8 (what-if)")
        print("module0_striped:", results["module0_striped"])
    results["wall_s"] = time.time() - t0

    def default(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if hasattr(o, "value"):
            return o.value
        return str(o)

    merged = {}
    if os.path.exists(args.out):                 # keep the phase-6 keys (mlp_*, module0_recommended, module0_1core)
        with open(args.out) as fh:
            merged = json.load(fh)
    merged.update(results)
    with open(args.out, "w") as fh:
        json.dump(merged, fh, indent=1, default=default)
    with open(args.md, "w") as fh:
        fh.write("# Cycle-model sweep tables (generated by model/sweeps.py)\n\nCycles at the descriptor's clock; "
                 "KV260 = 200 MHz, config C = 1 GHz. `per unit` divides the program's cycles by its tiles/chunks.\n\n" + "\n".join(mds))
    print(f"wrote {args.out}, {args.md} in {results['wall_s']:.0f}s")


if __name__ == "__main__":
    main()
