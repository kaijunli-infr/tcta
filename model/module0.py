#!/usr/bin/env python3
"""Module 0: workload + chip + fleet TCO analytical model (the formulas of the
"Module 0" doc as code). No dependencies beyond the standard library.

    python model/module0.py                 # default report: V4.1-Flash on config C @ 22FDX
    python model/module0.py --sweep         # array x channels, batch, alpha, node sweeps
    python model/module0.py --json          # machine-readable

Every number that is an approximation is in the `*_APPROX` tables with a
comment; replace them with PDK/vendor values as they arrive. Model parameters
come from the published V4.1-Flash config.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import asdict, dataclass, field, replace

# ============================================================================
# 1. Workload
# ============================================================================

@dataclass
class Workload:
    name: str = "DeepSeek-V4.1-Flash"
    L: int = 40                 # layers run in decode
    L_pre: int = 20             # layers run in prefill (encoder)
    d: int = 5120
    E: int = 384
    k: int = 6
    d_ff: int = 2304
    n_h: int = 64
    hd: int = 512
    rope: int = 64
    q_lora: int = 1280
    o_lora: int = 1024
    o_groups: int = 8
    topk: int = 512
    idx_heads: int = 32
    idx_dim: int = 128
    kv_bytes_per_token: int = 890        # published
    kv_entry_bytes: int = 384            # f4.r16 row (Appendix A)
    vocab: int = 129280
    b_exp: float = 0.5 + 1 / 32          # MXFP4 bytes/param
    b_attn: float = 1.0 + 1 / 32         # FP8 block-scaled bytes/param
    full_index_layers: int = 4           # layers that scan the whole compressed index
    cand_index_layers: int = 4           # layers restricted to the 2048x8 candidate pool
    attn_layers: int = 38                # layers doing sparse attention

    # ---- derived ----
    @property
    def expert_params_per_layer(self): return self.E * 3 * self.d * self.d_ff
    @property
    def shared_params_per_layer(self): return 3 * self.d * self.d_ff
    @property
    def attn_params_per_layer(self):
        dk = self.hd + self.rope
        return (self.d * self.q_lora + self.q_lora * self.n_h * dk
                + self.o_groups * (self.n_h // self.o_groups * self.hd) * self.o_lora
                + self.o_groups * self.o_lora * self.d + self.d * dk
                + self.d * self.idx_heads * self.idx_dim + self.d * self.idx_dim + self.d * self.E)
    @property
    def active_params_per_layer(self):
        return (self.k + 1) * self.shared_params_per_layer + self.attn_params_per_layer
    @property
    def P_act_dec(self): return self.active_params_per_layer * self.L
    @property
    def P_act_pre(self): return self.active_params_per_layer * self.L_pre
    @property
    def W_exp_bytes(self): return self.expert_params_per_layer * self.L * self.b_exp
    @property
    def W_attn_bytes(self): return self.attn_params_per_layer * self.L * self.b_attn
    @property
    def W_emb_bytes(self): return 2 * self.vocab * self.d * 2
    @property
    def W_total_bytes(self): return self.W_exp_bytes + self.W_attn_bytes + self.W_emb_bytes

    def experts_touched(self, B: int) -> float:
        return self.E * (1 - (1 - self.k / self.E) ** B)

    def flop_per_token_decode(self, ctx: int) -> float:
        dk = self.hd + self.rope
        attn = 2 * self.n_h * self.topk * (dk + self.hd) * self.attn_layers
        idx = 2 * self.idx_heads * self.idx_dim * (2048 * 8) * self.cand_index_layers \
            + 2 * self.idx_heads * self.idx_dim * (ctx / 2) * self.full_index_layers
        return 2 * self.P_act_dec + attn + idx

    def flop_per_token_prefill(self) -> float:
        dk = self.hd + self.rope
        attn = 2 * self.n_h * self.topk * (dk + self.hd) * (self.L_pre - 2)
        return 2 * self.P_act_pre + attn

    def kv_read_per_token(self, ctx: int) -> float:
        """Bytes of KV + index read per decode token at context ctx."""
        full = self.full_index_layers * (ctx / 2) * self.idx_dim * 1.0
        cand = self.cand_index_layers * (2048 * 8) * self.idx_dim * 1.0
        sparse = self.attn_layers * self.topk * self.kv_entry_bytes
        return full + cand + sparse

    def bytes_per_step(self, B: int, ctx: int) -> float:
        frac = self.experts_touched(B) / self.E
        return self.W_exp_bytes * frac + self.W_attn_bytes + B * self.kv_read_per_token(ctx)


# ============================================================================
# 2. Chip and node
# ============================================================================

# Approximate planning densities. Sources: SKY130 11.26 um^2/GE and Intel16 0.33 um^2/GE
# (Chipstitch, arXiv 2512.10089); SRAM macro densities and clocks are estimates.
NODES_APPROX = {
    "130nm-open":  dict(ge_per_mm2=89e3,  sram_mb_per_mm2=0.10, f_ghz=0.20, eur_per_mm2=1500,  min_mm2=0.8,
                        phy_mm2_per_ch=0.0, max_mem="DDR3 ~10 GB/s", lpddr5x=False, hbm=False,
                        pj_per_mac=8.0, src="IHP SG13 open-source rate; SkyWater ChipIgnite ~$975/mm2"),
    "130nm-comm":  dict(ge_per_mm2=100e3, sram_mb_per_mm2=0.40, f_ghz=0.30, eur_per_mm2=1760,  min_mm2=12,
                        phy_mm2_per_ch=0.0, max_mem="DDR3", lpddr5x=False, hbm=False,
                        pj_per_mac=6.0, src="GF 130BCDlite, Europractice 2026"),
    "22FDX":       dict(ge_per_mm2=2.4e6, sram_mb_per_mm2=4.5,  f_ghz=1.0,  eur_per_mm2=17820, min_mm2=4,
                        phy_mm2_per_ch=2.0, max_mem="LPDDR5X", lpddr5x=True, hbm=False,
                        pj_per_mac=1.2, src="GF 22FDX, Europractice 2026"),
    "12LP+":       dict(ge_per_mm2=3.0e6, sram_mb_per_mm2=8.0,  f_ghz=1.4,  eur_per_mm2=27500, min_mm2=4,
                        phy_mm2_per_ch=2.0, max_mem="LPDDR5X / HBM2e", lpddr5x=True, hbm=True,
                        pj_per_mac=0.8, src="GF 12LP+, Europractice 2026; TSMC16 ~$23k/mm2 (MUSE)"),
}

MEM_APPROX = {
    # per channel: bandwidth GB/s, capacity GB, pJ/bit (incl. DRAM), USD per channel's DRAM
    "LPDDR5X-8533 x32": dict(gbps=34.1, gb=8,  pj_bit=5.0, usd=30),
    "HBM2e stack":      dict(gbps=460,  gb=8,  pj_bit=3.5, usd=250),
    "HBM3 stack":       dict(gbps=820,  gb=24, pj_bit=3.0, usd=600),
    "DDR3 x64":         dict(gbps=10.6, gb=4,  pj_bit=20.0, usd=15),
}

GE_PER_MAC_BF16 = 2500   # BF16 x BF16, FP32 accumulate, pipelined
GE_PER_MAC_FP8 = 700     # reserved for MMA.MX v2


@dataclass
class Chip:
    name: str
    node: str
    n_mac: int
    sram_mb: float               # scratchpads
    mem_kind: str
    mem_channels: int
    eta_mfu: float = 0.5
    eta_bw: float = 0.7
    ctrl_mm2: float = 3.0
    pad_mm2: float = 4.0
    routing_overhead: float = 1.15
    chip_usd_volume: float = 0.0  # die at volume; 0 -> estimated from area

    @property
    def nd(self): return NODES_APPROX[self.node]
    @property
    def mem(self): return MEM_APPROX[self.mem_kind]
    @property
    def f_ghz(self): return self.nd["f_ghz"]
    @property
    def tflops(self): return 2 * self.n_mac * self.f_ghz * 1e9 / 1e12
    @property
    def bw_gbps(self): return self.mem_channels * self.mem["gbps"]
    @property
    def cap_gb(self): return self.mem_channels * self.mem["gb"]

    def area(self) -> dict:
        logic = self.n_mac * GE_PER_MAC_BF16 / self.nd["ge_per_mm2"]
        sram = self.sram_mb * 8 / self.nd["sram_mb_per_mm2"]
        phy = self.mem_channels * self.nd["phy_mm2_per_ch"]
        core = logic + sram + phy + self.ctrl_mm2
        die = core * self.routing_overhead + self.pad_mm2
        return dict(logic=logic, sram=sram, phy=phy, ctrl=self.ctrl_mm2, die=die)

    def shuttle_eur(self) -> float:
        a = max(self.area()["die"], self.nd["min_mm2"])
        return a * self.nd["eur_per_mm2"]

    def power_w(self, mfu: float, bw_util: float) -> float:
        array = self.n_mac * self.f_ghz * 1e9 * self.nd["pj_per_mac"] * 1e-12 * mfu
        mem = self.bw_gbps * 1e9 * 8 * self.mem["pj_bit"] * 1e-12 * bw_util
        static = 0.15 * (array + mem) + 2.0
        return array + mem + static

    def die_usd(self) -> float:
        if self.chip_usd_volume:
            return self.chip_usd_volume
        # crude: wafer $ / (dies per 300mm wafer * yield). Node wafer prices approximate.
        wafer = {"130nm-open": 1500, "130nm-comm": 2000, "22FDX": 4000, "12LP+": 6000}[self.node]
        die = self.area()["die"]
        dies = 70000 / die * 0.85
        return wafer / dies / 0.8


# ============================================================================
# 3. Replica sizing and TCO
# ============================================================================

@dataclass
class Serving:
    B: int = 256                 # tokens per decode step per replica
    ctx: int = 131072
    tpot_ms: float = 30.0
    alpha: float = 0.0           # DSpark acceptance; tau = 1 + 5*alpha
    kv_sequences: int = 256      # concurrent sequences resident (capacity term)
    utilization: float = 0.5     # fleet utilization over life
    years: float = 3.0
    usd_per_kwh: float = 0.10
    pue: float = 1.3
    host_usd: float = 5000.0     # per replica
    board_usd_per_chip: float = 60.0
    nre_eur: float = 0.0         # amortized over replicas if > 0
    replicas: int = 100

    @property
    def tau(self): return 1 + 5 * self.alpha


def size_replica(w: Workload, c: Chip, s: Serving) -> dict:
    bytes_step = w.bytes_per_step(s.B, s.ctx)
    flop_step = s.B * w.flop_per_token_decode(s.ctx)
    T = s.tpot_ms * 1e-3 * s.tau                      # step time budget
    N_bw = bytes_step / (c.bw_gbps * 1e9 * c.eta_bw * T)
    N_fl = flop_step / (c.tflops * 1e12 * c.eta_mfu * T)
    cap_need = w.W_total_bytes + s.kv_sequences * s.ctx * w.kv_bytes_per_token
    N_cap = cap_need / (c.cap_gb * 1e9)
    N = math.ceil(max(N_bw, N_fl, N_cap))
    # actual step time with N chips at batch B
    t_bw = bytes_step / (N * c.bw_gbps * 1e9 * c.eta_bw)
    t_fl = flop_step / (N * c.tflops * 1e12 * c.eta_mfu)
    t_step = max(t_bw, t_fl)
    B_star = N * c.tflops * 1e12 * c.eta_mfu * t_bw / w.flop_per_token_decode(s.ctx)
    thr = s.B * s.tau / t_step
    mfu = flop_step / (N * c.tflops * 1e12 * t_step)
    # throughput if we run at B* instead (bytes/step nearly constant above ~256)
    Bs = max(int(B_star), s.B)
    bytes_bs = w.bytes_per_step(Bs, s.ctx)
    t_bs = max(bytes_bs / (N * c.bw_gbps * 1e9 * c.eta_bw), Bs * w.flop_per_token_decode(s.ctx) / (N * c.tflops * 1e12 * c.eta_mfu))
    thr_bs = Bs * s.tau / t_bs
    ttft_128k = 131072 * w.flop_per_token_prefill() / (N * c.tflops * 1e12 * c.eta_mfu)
    return dict(N=N, N_bw=N_bw, N_flop=N_fl, N_cap=N_cap, bytes_per_step_GB=bytes_step / 1e9,
                t_step_ms=t_step * 1e3, tpot_ms=t_step * 1e3 / s.tau, bound="bandwidth" if t_bw >= t_fl else "compute",
                mfu=mfu, B_star=B_star, tok_s=thr, tok_s_at_Bstar=thr_bs, B_used_for_Bstar=Bs,
                ttft_128k_s=ttft_128k)


def tco(w: Workload, c: Chip, s: Serving, sz: dict) -> dict:
    N = sz["N"]
    mem_usd = c.mem_channels * c.mem["usd"]
    capex = N * (c.die_usd() + mem_usd + s.board_usd_per_chip) + s.host_usd
    nre = (s.nre_eur * 1.08 / s.replicas) if s.nre_eur else 0.0
    p_chip = c.power_w(mfu=sz["mfu"], bw_util=c.eta_bw)
    p_replica = N * p_chip + 200.0
    energy_usd = p_replica / 1000 * 8760 * s.years * s.pue * s.usd_per_kwh
    total = capex + nre + energy_usd
    life_s = s.utilization * s.years * 365 * 86400
    return dict(N=N, capex_usd=capex, nre_amortized_usd=nre, power_w=p_replica, energy_usd=energy_usd,
                tco_usd=total, usd_per_mtoken=total / (sz["tok_s"] * life_s) * 1e6,
                usd_per_mtoken_at_Bstar=total / (sz["tok_s_at_Bstar"] * life_s) * 1e6,
                die_usd=c.die_usd(), mem_usd_per_chip=mem_usd, p_chip_w=p_chip)


# ============================================================================
# 4. Presets and reporting
# ============================================================================

PRESETS = {
    "A-v1-sky130": Chip("A: v1 16x16 / 368KB", "130nm-open", 256, 0.36, "DDR3 x64", 1),
    "B-22fdx":     Chip("B: 32x32 / 2MB", "22FDX", 1024, 2.0, "LPDDR5X-8533 x32", 2),
    "C-22fdx-4ch": Chip("C: 4x64x64 / 16MB / 4ch", "22FDX", 16384, 16.0, "LPDDR5X-8533 x32", 4),
    "C-22fdx-8ch": Chip("C: 4x64x64 / 16MB / 8ch", "22FDX", 16384, 16.0, "LPDDR5X-8533 x32", 8),
    "C-12lp-hbm":  Chip("C: 4x64x64 / 16MB / 2xHBM2e", "12LP+", 16384, 16.0, "HBM2e stack", 2),
    "H100-ref":    Chip("H100-class reference", "12LP+", 0, 0, "HBM3 stack", 5, eta_mfu=0.5, eta_bw=0.7, chip_usd_volume=25000),
}
PRESETS["H100-ref"].n_mac = int(990e12 / 2 / 1.4e9)   # ~990 TFLOPS dense BF16 equivalent


def fmt(v):
    if isinstance(v, float):
        if abs(v) >= 1e5: return f"{v:,.0f}"
        if abs(v) >= 10: return f"{v:,.1f}"
        return f"{v:.3g}"
    return str(v)


def report(w: Workload, c: Chip, s: Serving, as_json=False) -> dict:
    sz = size_replica(w, c, s); t = tco(w, c, s, sz); a = c.area()
    out = dict(chip=c.name, node=c.node, area_mm2=a, shuttle_eur=c.shuttle_eur(), tflops=c.tflops,
               bw_gbps=c.bw_gbps, cap_gb=c.cap_gb, sizing=sz, tco=t)
    if as_json:
        return out
    print(f"\n=== {c.name} @ {c.node} | {w.name} | B={s.B} ctx={s.ctx} TPOT<={s.tpot_ms} ms tau={s.tau:.2f} ===")
    print(f"die {a['die']:.1f} mm2 (logic {a['logic']:.1f}, sram {a['sram']:.1f}, phy {a['phy']:.1f})  "
          f"shuttle {c.shuttle_eur()/1e3:.0f} kEUR  peak {c.tflops:.1f} TFLOPS  mem {c.bw_gbps:.0f} GB/s / {c.cap_gb} GB")
    print(f"replica: N={sz['N']} (bw {sz['N_bw']:.1f}, flop {sz['N_flop']:.1f}, cap {sz['N_cap']:.1f}) "
          f"{sz['bound']}-bound  bytes/step {sz['bytes_per_step_GB']:.0f} GB  step {sz['t_step_ms']:.1f} ms  "
          f"TPOT {sz['tpot_ms']:.1f} ms  MFU {sz['mfu']*100:.0f}%")
    print(f"throughput {sz['tok_s']:,.0f} tok/s at B={s.B}; B*={sz['B_star']:.0f} -> {sz['tok_s_at_Bstar']:,.0f} tok/s; "
          f"TTFT(128K prefill on this replica) {sz['ttft_128k_s']:.1f} s")
    print(f"TCO/replica ${t['tco_usd']:,.0f} (capex ${t['capex_usd']:,.0f}, energy ${t['energy_usd']:,.0f}, "
          f"NRE ${t['nre_amortized_usd']:,.0f})  {t['power_w']/1e3:.1f} kW  -> ${t['usd_per_mtoken']:.3f}/Mtoken at B={s.B}, "
          f"${t['usd_per_mtoken_at_Bstar']:.3f} at B*={sz['B_star']:.0f} (only if the scheduler can fill that batch)")
    return out


def sweeps(w: Workload, s: Serving):
    print("\n--- Sweep 1: array size x LPDDR5X channels (22FDX), N chips / $ per Mtoken ---")
    print(f"{'MACs':>8} " + " ".join(f"{ch:>20}ch" for ch in (2, 4, 8, 12)))
    for n_mac in (4096, 8192, 16384, 32768):
        row = []
        for ch in (2, 4, 8, 12):
            c = Chip("sw", "22FDX", n_mac, 16.0, "LPDDR5X-8533 x32", ch)
            sz = size_replica(w, c, s); t = tco(w, c, s, sz)
            row.append(f"N={sz['N']:>3} ${t['usd_per_mtoken']:.3f}/{t['usd_per_mtoken_at_Bstar']:.3f}")
        print(f"{n_mac:>8} " + " ".join(f"{r:>22}" for r in row))
    print("  ($/Mtoken at B=%d / at B*; chips per replica is set by total bandwidth, so fewer channels per chip "
          "means more chips and more idle array)" % s.B)

    c = PRESETS["C-22fdx-8ch"]
    print("\n--- Sweep 2: batch B (C-22fdx-8ch) ---")
    for B in (64, 128, 256, 512, 1024, 2048):
        sz = size_replica(w, c, replace(s, B=B)); t = tco(w, c, replace(s, B=B), sz)
        print(f"B={B:>5}: N={sz['N']:>3} bytes/step {sz['bytes_per_step_GB']:.0f} GB MFU {sz['mfu']*100:4.0f}% "
              f"tok/s {sz['tok_s']:>9,.0f}  ${t['usd_per_mtoken']:.3f}/Mtok")

    print("\n--- Sweep 3: DSpark acceptance alpha (C-22fdx-8ch, B=256) ---")
    for a in (0.0, 0.2, 0.4, 0.6):
        s2 = replace(s, alpha=a); sz = size_replica(w, c, s2); t = tco(w, c, s2, sz)
        print(f"alpha={a:.1f} tau={s2.tau:.1f}: N={sz['N']:>3} tok/s {sz['tok_s_at_Bstar']:>9,.0f} ${t['usd_per_mtoken']:.3f}/Mtok")

    print("\n--- Sweep 4: eta_bw (gather efficiency) ---")
    for e in (0.4, 0.55, 0.7, 0.85):
        c2 = replace(c, eta_bw=e); sz = size_replica(w, c2, s); t = tco(w, c2, s, sz)
        print(f"eta_bw={e:.2f}: N={sz['N']:>3} ${t['usd_per_mtoken']:.3f}/Mtok")

    print("\n--- Sweep 5: node (same microarchitecture, 8 memory channels or 2 HBM2e) ---")
    for key in ("C-22fdx-8ch", "C-12lp-hbm", "H100-ref"):
        report(w, PRESETS[key], s)


CYCLE_RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cycle_results.json")


def apply_cycle_model(path: str = CYCLE_RESULTS, verbose: bool = True, key: str = "module0") -> dict | None:
    """H4 hook (phase 5): replace the eta_bw / eta_mfu planning placeholders (0.7 / 0.5) of the config-C presets
    with the cycle model's measured values (`model/sweeps.py` -> `cycle_results.json`, key "module0"):
    eta_bw from the MoE decode tile (weight stream + token gathers, the bandwidth-bound op of decode), eta_mfu
    from the MoE tile's array utilization; the MLA tile's values are reported alongside. Returns the values
    applied, or None when the results file is absent (placeholders stay). `key` selects the result set:
    "module0" is the task's DMA rule (one LOAD.2D request occupies one channel), "module0_striped" the what-if
    where a request is striped over the 8 channels (`DeviceTiming.request_channels`)."""
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        m0 = json.load(fh).get(key)
    if not m0:
        return None
    eta_bw, eta_mfu = float(m0["eta_bw_moe"]), float(m0["eta_mfu_moe"])
    for name, chip in PRESETS.items():
        if chip.node in ("22FDX", "12LP+") and name.startswith("C-"):
            PRESETS[name] = replace(chip, eta_bw=eta_bw, eta_mfu=eta_mfu)
    if verbose:
        print(f"[cycle model:{key}] config-C presets: eta_bw={eta_bw:.3f} eta_mfu={eta_mfu:.3f} (MoE decode tile, config-C timing); "
              f"MLA tile: eta_bw={float(m0['eta_bw_mla']):.3f} eta_mfu={float(m0['eta_mfu_mla']):.3f}; source {m0.get('source')}")
    return dict(eta_bw=eta_bw, eta_mfu=eta_mfu, mla=dict(eta_bw=float(m0["eta_bw_mla"]), eta_mfu=float(m0["eta_mfu_mla"])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chip", default="C-22fdx-8ch", choices=list(PRESETS))
    ap.add_argument("--B", type=int, default=256)
    ap.add_argument("--ctx", type=int, default=131072)
    ap.add_argument("--tpot", type=float, default=30.0)
    ap.add_argument("--alpha", type=float, default=0.0)
    ap.add_argument("--nre-eur", type=float, default=1.2e6)
    ap.add_argument("--replicas", type=int, default=100)
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--cycle-results", default=CYCLE_RESULTS, help="cycle-model results (model/sweeps.py); eta_bw/eta_mfu of the config-C presets")
    ap.add_argument("--placeholders", action="store_true", help="keep the 0.7 / 0.5 planning placeholders")
    ap.add_argument("--cycle-key", default="module0", choices=["module0", "module0_striped"],
                    help="which cycle-model result set: the task's one-channel-per-request rule, or LOAD.2D striped over 8 channels")
    args = ap.parse_args()
    if not args.placeholders:
        apply_cycle_model(args.cycle_results, verbose=not args.json, key=args.cycle_key)
    w = Workload()
    s = Serving(B=args.B, ctx=args.ctx, tpot_ms=args.tpot, alpha=args.alpha, nre_eur=args.nre_eur, replicas=args.replicas)
    if args.json:
        print(json.dumps({"workload": {k: v for k, v in asdict(w).items()},
                          "derived": dict(P_act_dec=w.P_act_dec, P_act_pre=w.P_act_pre, W_total_GB=w.W_total_bytes / 1e9),
                          "report": report(w, PRESETS[args.chip], s, as_json=True)}, indent=2, default=str))
        return
    print(f"{w.name}: active {w.P_act_dec/1e9:.1f}B decode / {w.P_act_pre/1e9:.1f}B prefill; resident weights {w.W_total_bytes/1e9:.0f} GB; "
          f"decode {w.flop_per_token_decode(s.ctx)/1e9:.1f} GFLOP/token; KV read {w.kv_read_per_token(s.ctx)/1e6:.0f} MB/token @ ctx {s.ctx}")
    report(w, PRESETS[args.chip], s)
    if args.sweep:
        sweeps(w, s)


if __name__ == "__main__":
    main()
