# Task: phase 8 — v1 core and Gate 2 closure (full text, supersedes the amendment)

Status at issue (2026-10-29): K0 (fit) is done through K0.4 at head 9d34f96:
the core is 208 k LUT-eq by Yosys against a 99.5 k fit budget, formal passes,
every program is within 0.1 % of the cycle model. This file is the complete
phase-8 task; the earlier amendment is folded into §0 and the K-list below.
Spec: `isa/isa-v0.1.md` with errata E1–E6 (attached; replace the repo copy).

Gate 2 in the plan is "all four kernels on Verilator, cycle model within
10 %". Phase 7 got GEMM and FA there; K0 made the core 6× smaller; this
phase makes it fit, makes it fast, runs the remaining families, and closes
Gate 2 with a recalibrated model and a config-C projection.

## 0. Order of work and the fit decision

1. **Vivado first, when the x86 machine is available**
   (`rtl/synth/vivado_ooc.tcl`). It decides the lane width; every Yosys
   number is an upper bound (no MUXF7/F8, no cross-module optimisation).
   Do not block on it: K1–K3 proceed on the 16-lane core.
2. **L1–L4 now** (no speed cost, −43 k by estimate), each its own commit
   with the per-module Yosys number before/after:
   - L1 instruction queues in LUT-RAM (−13 k)
   - L2 scratchpad clients per the ISA §1 access table (INP/WGT 3, ACC 5,
     VEC 4, STATE 3; today every pad takes all 7) (−15 k)
   - L3 MMA writeback: 8 adders over two cycles, 4 converter lanes (−7.5 k;
     +1 cycle per accumulated row; fold into K3's RMW redesign)
   - L4 operand buffers in RAMB18 (−8 k)
3. **Lane width.** Decision rule, applied when the Vivado number exists:
   16-lane core with L1–L4 if Vivado ≤ 99.5 k LUT-eq; otherwise the
   **8-lane configuration** (`BANKS = MAC_COLS = VEC_LANES = 8`) for the
   KV260 build only. The 16-lane core stays the planning machine
   (`tcta.timing.KV260`) and the config-C basis. If Vivado is still
   unavailable when K5 starts, build K5 on the 16-lane core (Verilator does
   not care) and leave the lane decision to the FPGA phase.
4. Block RAM: ACC + STATE to URAM (16 URAMs) if Vivado does not merge the
   ports; state it in the fit report either way.

## 1. Deliverables

| # | Deliverable | Definition of done |
|---|---|---|
| K1 | DMA row pipelining + the two formats | `dma_load`/`dma_store` keep ≥ 2 rows in flight (`DMA_ROWS_IN_FLIGHT`); tile loads reach ≥ 14 B/cycle on the 64×128 BF16 tile at the harness's 24-cycle latency (K0: 5.1). `STORE.2D f4.r16` (quantize on egress, Appendix A) and block-scaled `LOAD.IDX` (`f4.r16` KV rows; `f4.b32` rows with their scale slice) implemented, closing E6's two gaps. The `gather` program extended to cover both. |
| K2 | VEC bank map (E2) | VEC 2-D slots banked by row; `[64, 1]` ops cost `⌈M/16⌉ + overhead`, not `M`; FA's per-KV-tile VEC chain drops from ~4.6 k toward ~4.2 k cycles. |
| K3 | SFU slice pipelining + accumulate RMW | SFU: slices in flight so an op costs ~4 cycles per slice, not 6. MMA accumulate writeback: old-value read pipelined with the previous row's write (with L3's 8 adders over two cycles), removing the K ≤ 64 writeback bound (FA PV +14 %). |
| K4 | Front end (optional) | One stream word per cycle fetch/decode (K0: 3). Do it if K1–K3 land early. |
| K5 | MoE, MLA, KDA on Verilator | Per-expert MoE (f4.b32), MoE combine, router, MLA decode (wide ACC), KDA chunk (v0.1.3 form: `VCVT dst=WGT`, `MMA.MASKED mul`, STATE `row0`). Each bit-exact to the `fixed48` simulator where the simulator is deterministic (gathers, scatters, dequant, VTOPK, VCOUNT, LOOP r), within §9 on SFU paths; fixture tolerances met; token-check mode on. |
| K6 | Recalibration | `calibrate.py` extended to every new op (`LOAD.IDX`/`STORE.IDX` per-row and index-slice costs, `VTOPK`, `VGATE`, `VMASK`, `VSEL`, `VCVT dst=WGT`, `MMA.MASKED`, wide ops, `row0`, `LOOP r, step`) and to the K1–K4 structures as new `rtl_rules`; the K0 constants they change are superseded, not kept as options. **All five families within 10 %** (target 2 %). Rerun with every commit (K0's lesson). |
| K7 | Config-C projection | Re-derive `CONFIG_C` by **scaling the measured v1 structure** (same FSM costs, 64-wide datapath, 64 VEC lanes, 16 SFU lanes, striping 2, 300-cycle latency), not by reverting to planning rules. Rerun the phase-6 sweeps and Module 0; report what moved against the phase-6 numbers (58 chips, $0.096/Mtoken). |
| K8 | Gate 2 report + verdict | `docs/gate2-report.md`: per family RTL vs model; K0–K4 before/after; fit table (Yosys, and Vivado when available) with the lane decision; formal results; remaining differences; the calibration list that survives into the FPGA phase (memory latency, clock). Verdict Phase 8: PASS per §3. |

## 2. Rules

- Errata E1–E6 are the only spec changes; anything else the RTL needs is a
  new erratum sent back (E7+), not a silent fix. K1's formats implement
  existing §4 text.
- The programs do not change. If a family's program must change to pass on
  RTL, stop and report.
- Each of L1–L4 and K1–K4 is a separate commit with its before/after on the
  micro-programs (`dma_copy`, `vec`, `mma`, `loop`, `gather`) and a
  `calibrate.py` rerun.
- The cycle model is calibrated per structure, never tuned per program.
- `acc_mode="fixed48"` is the RTL's conformance reference; kernels are
  still judged against the PyTorch fixtures with the §9 tolerance.

## 3. Decision rule

- **PASS**: L1–L4, K1–K3, K5, K6, K8; formal clean; five families within
  10 %; tile loads ≥ 14 B/cycle; FA PV no longer writeback-bound; a fit
  statement with the lane decision (Vivado-backed, or Yosys-backed with the
  8-lane configuration named).
- **PASS-WITH-CAVEATS**: K4 or K7 skipped, one family between 10 and 20 %
  with the cause named, or the lane decision deferred to the FPGA phase
  for lack of a Vivado run.
- **FAIL**: a formal counterexample unresolved, a program change required,
  or a family off by more than 20 %.

## 4. Not in scope

Multi-core on RTL, COMM, AXI/DDR, the KV260 board, Vivado timing closure
(the x86 machine), ASIC flow.
