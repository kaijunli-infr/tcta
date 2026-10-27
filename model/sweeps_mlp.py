#!/usr/bin/env python
"""Phase 6 (`docs/phase6-mlp-task.md`): memory-level parallelism sweeps on config C after ISA v0.1.4 §11.

    python model/sweeps_mlp.py                      # all -> merged into model/cycle_results.json, tables in model/mlp_results.md
    python model/sweeps_mlp.py --only ring,cores    # a subset

I1 ring depth (MoE token tile, MLA KV tile), I2 cores sanity (GEMM copies, a saturating stream), I3 core sweep
(MoE with disjoint expert subsets per core, MLA copies), I4 striping x cores, I5 the Module 0 keys. Every run
checks its functional result against the fixture (`sweeps.run` / per-core checks), so no sweep changes numerics.
Descriptors: `CONFIG_C` (one stream, the four 64x64 arrays as one MMA unit) and `CONFIG_C_CORE` (one array per
core, channels shared), both with the §1 decisions (FIFO 8, INP 5, WGT 32, striping 2, broadcast); rings deeper
than the FIFO need the FIFO raised (finding, §I1), so the ring and core sweeps also run at FIFO 32.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import replace

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from sweeps import TOL, clean, fam_gemm, fam_mla, fam_moe, pct, run  # noqa: E402
from tcta import DeviceConfig, Fault  # noqa: E402
from tcta.cycle import CycleDeadlock, run_multicore  # noqa: E402
from tcta.isa import DEVICE_CONFIG_C, SP  # noqa: E402
from tcta.timing import CONFIG_C, CONFIG_C_CORE, CONFIG_C_PHASE5  # noqa: E402

CHIP_ARRAYS = 4
MOE_RING_KNEE = 12
MLA_UNROLL_KNEE = 2       # K ring 10


def dev(fifo):
    return replace(DEVICE_CONFIG_C, fifo_depth=fifo)


def one(fam, T, cfg=None):
    """A single-stream run as a result row (deadlock / fault rows carry `cycles=None`)."""
    r = run(fam, T, cfg)
    if r.get("cycles") is None:
        return clean(r) | dict(n_instrs=fam.rep.n_instrs, static_levels=fam.rep.max_fifo_level)
    t = r["_timing"]
    mean, p90 = t.inflight_bytes()
    return clean(r) | dict(n_instrs=fam.rep.n_instrs, inflight_mean=mean, inflight_p90=p90, channel_util=t.channel_util,
                           stall_arbiter=t.stall_arbiter, stall_arbiter_gather=t.stall_arbiter_gather, gather_bound=t.gather_bound,
                           static_levels=fam.rep.max_fifo_level, mma_load_level=t.edges.get("MMA->LOAD", {}).get("max_level", 0),
                           push_blocked=sum(e["push_blocked_cycles"] for e in t.edges.values()))


def multi(fams, T, label):
    """fams on len(fams) cores; -> result row with per-core numbers; every core's result checked."""
    t0 = time.time()
    try:
        mt, ms = run_multicore(T, [dict(instrs=f.prog.instrs, load=f.load, cfg=f.cfg, dram_bytes=f.dram_bytes) for f in fams])
    except CycleDeadlock as e:
        return dict(label=label, cores=len(fams), deadlock=str(e.heads), cycles=None)
    except Fault as e:
        return dict(label=label, cores=len(fams), fault=str(e), cycles=None)
    errs = [float(f.check(m)) for f, m in zip(fams, ms)]
    assert max(errs) < TOL, (label, errs)
    units = sum(f.units for f in fams)
    mean, p90 = mt.inflight_bytes()
    return dict(label=label, cores=len(fams), ok=all(m.status["ok"] for m in ms), errs=errs, cycles=mt.cycles, us=mt.us,
                units=units, unit_name=fams[0].unit_name, per_unit=mt.cycles / units,
                eta_bw=mt.eta_bw, eta_mfu=mt.eta_mfu, eta_mfu_chip=mt.eta_mfu_chip(CHIP_ARRAYS), channel_util=mt.channel_util,
                inflight_mean=mean, inflight_p90=p90, dram_read=mt.dram_read_bytes, dram_write=mt.dram_write_bytes, macs=mt.macs,
                core_cycles=[c.cycles for c in mt.cores], core_arbiter=[c.stall_arbiter for c in mt.cores],
                core_arbiter_gather=[c.stall_arbiter_gather for c in mt.cores],
                core_gather_wait_per_burst=[c.stall_arbiter_gather / max(1, c.gather_bound["requests"]) for c in mt.cores],
                core_gather_bound=[c.gather_bound for c in mt.cores],
                core_loads=[c.per_queue.get("LOAD", {}).get("n", 0) for c in mt.cores],
                mma_load_level=max(c.edges.get("MMA->LOAD", {}).get("max_level", 0) for c in mt.cores),
                channels=mt.channels, wall_s=time.time() - t0)


def kb(x):
    return f"{x / 1024:.0f}"


# --------------------------------------------------------------------------
# I1: ring depth
# --------------------------------------------------------------------------

def sweep_ring(_):
    rows = []
    # reference rows: the phase-5 descriptor and the v0.1.4 descriptor on the phase-3 program form (ring 3)
    f = fam_moe(num_stages=3, cfg=dev(16))
    rows.append(one(f, CONFIG_C_PHASE5, dev(16)) | dict(family="MoE", form="phase-3 form, phase-5 descriptor", ring=3, fifo=16))
    f = fam_moe(num_stages=3, cfg=dev(8))
    rows.append(one(f, CONFIG_C) | dict(family="MoE", form="phase-3 form", ring=3, fifo=8))
    for fifo in (8, 32):
        for st in (3, 6, 12, 16):
            f = fam_moe(num_stages=st, flat_w13=True, cfg=dev(fifo))
            rows.append(one(f, replace(CONFIG_C, fifo_depth=fifo)) | dict(family="MoE", form="flat [W1|W3] stream", ring=st, fifo=fifo))
    f = fam_mla(inp_slots=5, cfg=dev(16))
    rows.append(one(f, CONFIG_C_PHASE5, dev(16)) | dict(family="MLA", form="INP 5, phase-5 descriptor", ring=4, fifo=16))
    for fifo in (8, 32):
        for u in (1, 2, 4):
            f = fam_mla(inp_slots=5, kv_unroll=u, cfg=dev(fifo))
            rows.append(one(f, replace(CONFIG_C, fifo_depth=fifo)) | dict(family="MLA", form=f"INP 5, K ring over {u} KV tile(s)", ring=4 if u == 1 else 5 * u, fifo=fifo))
    md = ["| family | program form | WGT ring | FIFO | cycles | per unit | η_bw | η_mfu | channel util | bytes in flight mean / p90 (KB) | MMA→LOAD level (dyn) | PUSH blocked | bank-stall |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        if r.get("cycles") is None:
            why = "deadlock" if r.get("deadlock") else "functional TOKEN_TIMEOUT (ring tokens exceed the FIFO)"
            md.append(f"| {r['family']} | {r['form']} | {r['ring']} | {r['fifo']} | **{why}** | | | | | | static MMA→LOAD {r['static_levels'].get('MMA->LOAD')} | | |")
            continue
        md.append(f"| {r['family']} | {r['form']} | {r['ring']} | {r['fifo']} | {r['cycles']} | {r['per_unit']:.0f} / {r['unit_name']} | {r['eta_bw']:.3f} | {r['eta_mfu']:.3f} | "
                  f"{r['channel_util']:.3f} | {kb(r['inflight_mean'])} / {kb(r['inflight_p90'])} | {r['mma_load_level']} | {r['push_blocked']} | {sum(r['banks'].values())} |")
    return rows, "\n".join(md)


# --------------------------------------------------------------------------
# I2: cores sanity
# --------------------------------------------------------------------------

def sweep_cores_sanity(_):
    from tcta import ACC, INP, WGT, Machine, Program, Q, Region
    from tcta import dtypes as dt
    rng = np.random.default_rng(0)

    def put_bf16(m, addr, arr):
        arr = np.asarray(arr, np.float32)
        bits = (dt.to_bf16(arr).view(np.uint32) >> 16).astype(np.uint16)
        m.dram_write(addr, bits)
        return Region(addr, "bf16", arr.shape, (arr.shape[1] * 2, 2))

    A = rng.standard_normal((64, 128)); B = rng.standard_normal((128, 128))

    def load(m):
        put_bf16(m, 0, A); put_bf16(m, 1 << 16, B)

    g = fam_gemm()
    cfg = DeviceConfig(slots={SP.INP: 2, SP.WGT: 16, SP.ACC: 6, SP.VEC: 16, SP.STATE: 1})
    m = Machine(cfg=cfg)
    rb = put_bf16(m, 1 << 16, B)
    p = Program()
    for i in range(16):
        p.load2d(WGT(i), rb, 128, 128)
    p.finish()
    rows = []
    for label, instrs, ld, c_cfg, T in (("GEMM (2 LOAD, 1 MMA, 1 STORE; 8 write channels)", g.prog.instrs, g.load, None, replace(CONFIG_C_CORE, write_channels=8)),
                                        ("weight stream (16 x 32 KB LOAD.2D, no tokens)", p.instrs, load, cfg, CONFIG_C_CORE)):
        base = None
        for c in (1, 2, 4):
            mt, ms = run_multicore(T, [dict(instrs=instrs, load=ld, cfg=c_cfg, dram_bytes=g.dram_bytes)] * c)
            base = base or mt.cycles
            rows.append(dict(label=label, cores=c, cycles=mt.cycles, ratio=mt.cycles / base, channel_util=mt.channel_util,
                             core_cycles=[t.cycles for t in mt.cores], arbiter=[t.stall_arbiter for t in mt.cores]))
    md = ["| program | cores | cycles | vs 1 core | channel util | per-core cycles | per-core arbiter wait |", "|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['label']} | {r['cores']} | {r['cycles']} | {r['ratio']:.2f}x | {r['channel_util']:.3f} | {r['core_cycles']} | {r['arbiter']} |")
    return rows, "\n".join(md)


# --------------------------------------------------------------------------
# I3: core sweep
# --------------------------------------------------------------------------

def moe_cores(c, stages=MOE_RING_KNEE, fifo=8, striping=2, T=None):
    E = 32
    fams = [fam_moe(num_stages=stages, flat_w13=True, cfg=dev(fifo), expert_base=i * (E // c), n_experts=E // c) for i in range(c)]
    T = replace(T or CONFIG_C_CORE, fifo_depth=fifo, request_channels=striping)
    return multi(fams, T, f"MoE ring {stages}, {c} core(s) x {E // c} experts, striping {striping}")


def sweep_cores(_):
    rows = []
    for c in (1, 2, 4):
        rows.append(moe_cores(c) | dict(family="MoE", arrays_per_core=1, ring=MOE_RING_KNEE))
    # the same silicon as one core with a 4-array MMA unit
    f = fam_moe(num_stages=MOE_RING_KNEE, flat_w13=True, cfg=dev(8))
    r = multi([f], CONFIG_C, f"MoE ring {MOE_RING_KNEE}, 1 core x 4 arrays")
    rows.append(r | dict(family="MoE", arrays_per_core=4, ring=MOE_RING_KNEE))
    for c in (1, 2, 4):
        fams = [fam_mla(inp_slots=5, kv_unroll=MLA_UNROLL_KNEE, cfg=dev(8)) for _ in range(c)]
        rows.append(multi(fams, CONFIG_C_CORE, f"MLA K ring 10, {c} core(s) x 1 head group") | dict(family="MLA", arrays_per_core=1, ring=10))
    f = fam_mla(inp_slots=5, kv_unroll=MLA_UNROLL_KNEE, cfg=dev(8))
    rows.append(multi([f], CONFIG_C, "MLA K ring 10, 1 core x 4 arrays") | dict(family="MLA", arrays_per_core=4, ring=10))
    md = ["| run | cycles | per unit | η_bw | η_mfu (chip, 4 arrays) | channel util | bytes in flight mean / p90 (KB) | per-core cycles | per-core arbiter wait (tile requests, cycles) | per-core gather wait (cycles per burst) |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        if r.get("cycles") is None:
            md.append(f"| {r['label']} | **{r.get('deadlock') or r.get('fault')}** | | | | | | | | |")
            continue
        md.append(f"| {r['label']} | {r['cycles']} | {r['per_unit']:.0f} / {r['unit_name']} | {r['eta_bw']:.3f} | {r['eta_mfu_chip']:.3f} | {r['channel_util']:.3f} | "
                  f"{kb(r['inflight_mean'])} / {kb(r['inflight_p90'])} | {r['core_cycles']} | {r['core_arbiter']} | {[round(x, 1) for x in r['core_gather_wait_per_burst']]} |")
    return rows, "\n".join(md)


# --------------------------------------------------------------------------
# I4: striping x cores
# --------------------------------------------------------------------------

def sweep_striping(_):
    rows = []
    for striping in (1, 2, 8):
        for c in (1, 2, 4):
            rows.append(moe_cores(c, striping=striping) | dict(striping=striping))
    md = ["| request_channels | cores | cycles | per token tile | η_bw | η_mfu (chip) | channel util | bytes in flight mean (KB) | vs striping 1 |", "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        base = [x for x in rows if x["striping"] == 1 and x["cores"] == r["cores"]][0]["cycles"]
        md.append(f"| {r['striping']} | {r['cores']} | {r['cycles']} | {r['per_unit']:.0f} | {r['eta_bw']:.3f} | {r['eta_mfu_chip']:.3f} | {r['channel_util']:.3f} | {kb(r['inflight_mean'])} | {pct(r['cycles'], base)} |")
    return rows, "\n".join(md)


# --------------------------------------------------------------------------
# I5: Module 0 keys
# --------------------------------------------------------------------------

def module0_keys(results):
    cores = results.get("cores", [])

    def pick(family, arrays, c):
        return next(r for r in cores if r["family"] == family and r["arrays_per_core"] == arrays and r["cores"] == c)
    moe4, mla4 = pick("MoE", 1, 4), pick("MLA", 1, 4)
    moe1, mla1 = pick("MoE", 4, 1), pick("MLA", 4, 1)
    rec = dict(eta_bw_moe=moe4["eta_bw"], eta_mfu_moe=moe4["eta_mfu_chip"], eta_bw_mla=mla4["eta_bw"], eta_mfu_mla=mla4["eta_mfu_chip"],
               moe_cycles_per_tile=moe4["per_unit"], mla_cycles_per_tile=mla4["per_unit"],
               source="model/sweeps_mlp.py cores sweep: 4 cores x 1 array, WGT ring 12 (MLA K ring 10), striping 2, broadcast, FIFO 8")
    one_core = dict(eta_bw_moe=moe1["eta_bw"], eta_mfu_moe=moe1["eta_mfu_chip"], eta_bw_mla=mla1["eta_bw"], eta_mfu_mla=mla1["eta_mfu_chip"],
                    moe_cycles_per_tile=moe1["per_unit"], mla_cycles_per_tile=mla1["per_unit"],
                    source="model/sweeps_mlp.py cores sweep: 1 core x 4-array MMA unit, WGT ring 12 (MLA K ring 10), striping 2, broadcast, FIFO 8")
    return rec, one_core


SWEEPS = {
    "ring": ("I1: WGT ring depth (config C, §1 decisions; FIFO 8 and 32)", sweep_ring),
    "sanity": ("I2: cores against shared channels (GEMM copies, a saturating weight stream)", sweep_cores_sanity),
    "cores": ("I3: MoE on 1 / 2 / 4 cores (disjoint expert subsets), MLA copies; the 4-array single core", sweep_cores),
    "striping": ("I4: request striping x cores (MoE, ring 12)", sweep_striping),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=",".join(SWEEPS))
    ap.add_argument("--out", default=os.path.join(HERE, "cycle_results.json"))
    ap.add_argument("--md", default=os.path.join(HERE, "mlp_results.md"))
    args = ap.parse_args()
    t0 = time.time()
    results, mds = {}, []
    for key in args.only.split(","):
        title, fn = SWEEPS[key]
        print(f"== {title}", flush=True)
        rows, md = fn(None)
        results[key] = rows
        mds.append(f"## {title}\n\n{md}\n")
        print(md, flush=True)
    if "cores" in results:
        rec, one_core = module0_keys(results)
        results["module0_recommended"], results["module0_1core"] = rec, one_core
        print("module0_recommended:", rec)
        print("module0_1core:", one_core)
    results["mlp_wall_s"] = time.time() - t0

    def default(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if hasattr(o, "value"):
            return o.value
        return str(o)

    merged = {}
    if os.path.exists(args.out):
        with open(args.out) as fh:
            merged = json.load(fh)
    merged.update({("mlp_" + k if k in SWEEPS else k): v for k, v in results.items()})
    with open(args.out, "w") as fh:
        json.dump(merged, fh, indent=1, default=default)
    with open(args.md, "w") as fh:
        fh.write("# Phase 6 sweep tables (generated by model/sweeps_mlp.py)\n\nConfig C at 1 GHz after ISA v0.1.4 §11 (FIFO 8 unless stated, "
                 "INP 5, WGT 32, request striping 2, operand broadcast). `CONFIG_C` = one stream with the four 64x64 arrays as one MMA unit; "
                 "`CONFIG_C_CORE` = one array per core, channels shared. η_mfu (chip) charges all four arrays.\n\n" + "\n".join(mds))
    print(f"wrote {args.out} (keys mlp_*, module0_recommended, module0_1core), {args.md} in {results['mlp_wall_s']:.0f}s")


if __name__ == "__main__":
    main()
