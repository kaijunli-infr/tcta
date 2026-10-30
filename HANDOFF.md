# Handoff: TileLang cut-off task, phases 1–7 and phase 8 K0, L1–L5, K1–K3 (2026-10-30)

For the session steering `docs/tilelang-lowering-cutoff-task.md`, its
addendum, `docs/tilelang-phase2-moe-task.md`, `docs/tilelang-phase3-task.md`,
`docs/tilelang-phase4-task.md`, `docs/phase5-cycle-model-task.md`,
`docs/phase6-mlp-task.md`, `docs/phase7-rtl-core-task.md` and
`docs/phase8-amendment-fit.md` (K0: fit) and `docs/phase8-gate2-task.md`
(the full phase-8 text: L1–L4, K1–K8). All deliverables of phases 1–7,
phase 8 K0.1–K0.4, L1–L3 and K1 exist (L4 analysed and not taken); the
whole repo is green (`pytest tests -q`: **96 passed**; `make -C rtl/test
all`: **12/12 PASS**, also with `FORMAL=1`, lint clean). The ISA is the main
session's full file with errata E1–E7 (§16; E1 and E5 were assigned here,
E2–E4, E6 and E7 by the main session; E7 makes the scratchpad access table
normative hardware). **This repo is canonical**; since 2026-10-29 the main
session sends whole files (ISA, task), not deltas. Remotes:
`github.com/kaijunli-infr/tcta` (origin, private) and
`github.com/datacrunch-research/tcta` (`datacrunch`, private, org team
`silicon` reads).

**Phase 1: PASS. Phase 2: PASS. Phase 3: PASS. Phase 4: PASS-WITH-CAVEATS
(closed by v0.1.3, Gate 1 closed). Phase 5: PASS. Phase 6: PASS. Phase 7:
PASS-WITH-CAVEATS** — GEMM bit-exact and FA within §9 on the RTL under
Verilator with the compiler's programs unchanged, the three negative tests
fault with the simulator's codes, the cycle model is within 0.3 % of the RTL
after calibration (but only with rule changes: the v0 core is row-serial in
its DMA and VEC vectors, report below), the SymbiYosys proofs pass; but the
Yosys synthesis estimate is 1.30 M LUTs against the KV260's 117 k (11×): the
byte-addressed STORE row buffer, the 256 binary32 accumulators of the array
and the unshared VEC arithmetic. Fmax is still open (Vivado, x86).
**Phase 8 K0 (fit): K0.1–K0.3 done, K0.4 FAIL on fit** — the three fixes
take the core from 1.30 M to 208 k LUT-equivalents (Yosys), 1.78× the K26
and 2.1× the fit budget with 15 % headroom; the 8×8 fallback alone does not
fit either (the lane-proportional logic outside the array dominates); see
the Phase 8 K0 verdict section and `docs/phase8-k0-fit-report.md`.
**Phase 8 L1–L3 done, L4 not taken** — 208 k → 158 k LUT-eq (135 % of the
K26, 1.59× the fit budget) at +1.4 % on FA; the 8×8 fallback is no longer
ruled out (needs Vivado ≥ 19 % under Yosys); see the Phase 8 L1–L4 verdict
section. **Phase 8 K1 done** — row-pipelined DMA engines (64×128 BF16 tile
3205 → 1127 cycles, 14.5 B/cycle; GEMM 11144 → 5917, FA 39919 → 32109),
`STORE.2D f4.r16` and block-scaled `LOAD.IDX` bit-exact; core 162.5 k
LUT-eq (139 % of the K26, +4 k for the `f4.r16` quantiser); see the Phase 8
K1 verdict section. **L5 measured** (the 8×8 array on the 16-lane core,
`TCTA_MAC8`, default off): −17.4 k LUT-eq (core 145.1 k, 124 % of the K26)
for FA +72 % / GEMM +101 %; not a middle ground, the §0.3 lane rule stands
(Phase 8 L5 section; fit report §4c). **Phase 8 K2 done** — erratum E2,
the VEC pad's 2-D slots banked by row: `[64, 1]` ops 70–134 → 9–14 cycles,
FA 32109 → 26049 (−18.9 %); row access to `[M, N > 1]` 2-D VEC tiles is
element-serial (one finding for the main session); core 167.7 k LUT-eq
(143 %, +5.2 k). **Phase 8 K3 done** — the SFU's slices pipelined (4 cycles
per slice, was 6) and the MMA writeback pipelined through its port (an
accumulated row 2 cycles, was 5): FA 26049 → 21593 (−17.1 %), its PV
accumulate 1436 → 1150 and no longer writeback-bound; two harness programs
added (`mma_mask`, `sfu`); core 168.5 k LUT-eq (144 %, +0.8 k). Next: K4–K8
of `docs/phase8-gate2-task.md`; the Vivado run (x86) decides the lane width
and can report both arrays (`vivado_ooc.tcl` with `TCTA_MAC8`).

## Phase 8 K3 in one paragraph

The VEC unit issues the next SFU slice's read two lane steps before the
stepping slice ends, so it returns in the last step and the four SFU lanes
never idle: every SFU kind costs 4 cycles per slice (VEXP2.SUM 64×64 1874 →
1052, `[64, 1]` VEXP2 29 → 23; VGATE decay 6, two operand reads); the
sigmoid chain's lane use is gated on the kind and VEXP2.SUM's partial add
keeps its own tag. The MMA writeback prefetches the next row's first operand
(old accumulator, else mask) while the current row adds and takes the next
row in the write cycle, every read landing in one register: streaming rows
cost 2 cycles plain and accumulate, 3 masked, 6 masked accumulate, so FA's
PV accumulate is the plain product plus 2 (1150 vs 1148; 1436 before). Two
programs join the harness — `mma_mask` (bit-exact) and `sfu` (every SFU kind,
a row-relative 2⁻¹⁰ check) — and the cycle model replays the writeback FSM
state by state and carries the SFU constants; every MMA and SFU row is exact,
also on the 8×8 build.

## Phase 8 K2 in one paragraph

The VEC pad's 2-D slots are banked by row (`bank = row mod 16`, word = slot
base + column × 4 + row / 16); the VEC unit reads a `[M, 1]` vector operand
as one column slice per 16 rows and runs elementwise ops whose tiles are all
2-D VEC tiles in column-slice order (`col_req`, `req.cs`; a `[64, 1]` op is 4
slices); row reductions write their result as one element of a column slice.
A row-slice request into the row-banked range (every DMA access to a 2-D VEC
tile, VGATE / VSEL / VMASK with a VEC tile) is served by the pad element by
element on port A (granted with the last element, the assembled slice the
cycle after; a changed request aborts the burst), so such accesses cost one
cycle per element: `dma_copy`'s `[64, 64]` VEC round trip 2228 → 8396. Both
arbiter proofs pass (a second job on the VEC pad). The cycle model's VEC and
DMA rules carry the column mode, the per-16-row vector read and the serial
cost; every VEC row of the seven programs is exact except VCOUNT 1×8 (13 vs
14), and `loop` shows pad port contention the rules do not model (−7.4 %).

## Phase 8 K1 in one paragraph

Each DMA engine became a front and a back end around the K0.1 row buffer
turned into a 128-beat ring with eight row records (`DMA_ROWS_IN_FLIGHT`):
LOAD allocates a row, issues its beats and moves on while the responses
land in the ring and a writer dequantises complete rows a slice per cycle;
STORE reads a slice per cycle into the ring while a writer streams complete
rows as beats. A row costs `max(beats + 1, slices + 1)`; the latency is paid
once per instruction. `LOAD.IDX` on `f8.b32` / `f4.b32` takes the row's
scale slice from the grid at `idx / 32` (simulator too); `STORE.2D f4.r16`
quantises on egress bit-exact to the simulator without a divider (an
integer comparison per E2M1 midpoint; four lanes, cold path, +4 k LUTs).
The `gather` program covers both; the cycle model's DMA rule is the
two-stage structure and reproduces every DMA row of the seven programs
(two rows within 1–2 cycles). E7 (ISA §16) landed in the same commit.

## Phase 8 L1–L4 in one paragraph

L1 (e88d321): each queue's instruction buffer is a two-head LUT-RAM
(−12.9 k); the simulator faults MMA K = 0 / K > 128 (E6). L2 (f4afb57):
`tcta_pkg::SP_READERS / SP_WRITERS` carry the ISA access table into the
scratchpads (arbiter on the union, write muxes on the writers, response
path on the readers, one word address per request): pads 40.8 k → 14.5 k,
glue 8.4 k → 3.6 k (−31 k, twice the estimate). L3 (a8244a2): the MMA
writeback adds a row on 8 adders in two halves (−5.75 k, +1 cycle per
accumulated row; the 4-lane converter was measured at +15.6 % on FA's
K = 64 products and left at 8 lanes). L4: the operand buffers are a corner
turn; in block RAM they need 64 RAMB18, not 16, for ≈ −3 k — not taken
(`docs/phase8-k0-fit-report.md` §4b). Every commit: lint, 10/10,
`calibrate.py` within 0.1 %, pytest, sby arbiter PASS.

## Phase 8 K0 in one paragraph

K0.1 (71e9c4d): beat-banked LUT-RAM row buffers in both DMA engines
(`dma_store` 768 k → 7 k LUTs), a `gather` program (tenth program) that
found a pre-existing index-slice bug (every gathered row used index 0).
K0.2 (5055321, follow-up aba825d): one ALU per VEC lane and a shared row
reducer (`vec_unit` 123 k → 54 k); the reducer pacing regression (+3 cycles
per 4-slice row) was found by recalibration and fixed. K0.3 (0d2735f):
erratum E5 — the array accumulates in a 48-bit fixed-point window placed by
each block's exponent maxima, two DSP48E2 per cell, operands block-buffered
in LUT-RAM, one rounding to binary32 per element; the simulator's
`acc_mode="fixed48"` (`DEVICE_KV260`) is bit-exact to the RTL (`mma_unit`
290 k → 52 k LUTs, 624 DSP); the cycle model replays the unit cycle by cycle
and matches every MMA shape exactly. K0.4: `docs/phase8-k0-fit-report.md`
(per-module table, LUT-RAM counted, block RAM, levers L1–L7, the fallback
statement: 8 lanes throughout is what fits with headroom; Vivado first).

## Phase 7 in one paragraph

One core of ISA v0.1.5 in SystemVerilog (`rtl/src/`, 20 modules: front end
with a 4-deep loop stack and extension-word apply, five in-order queues over
token FIFOs, five column-banked scratchpads with a two-port seven-client
arbiter, row-at-a-time LOAD/STORE engines with dequant on ingress, a 16×16
output-stationary systolic array with a result FIFO and a mask/accumulate
writeback, a slice-serial VEC pipe with a 4-lane SFU, a debug token checker),
fed by the §12 binary stream (`sim/tcta/encode.py`) and run under Verilator
5.052 (`rtl/test/`) against the simulator's own DRAM images and expectations.
Ten programs: dma_copy, mma (plain / accumulate / A-from-ACC), vec, gemm, fa,
gather (K0.1), loop (front-end micro-test) and the three negative tests of
`test_sim.py`.
Calibration (`rtl/test/calibrate.py`, `docs/calibration-report.md`): the
measured core differs from the plan in structure — DMA row-serial (5.1 B/cycle
on a BF16 tile, plan 16), column vectors M cycles (plan ⌈M/16⌉), SFU 6 cycles
per slice (plan 4) but VEXP2.SUM fused, front end 3 cycles per word with
PUSH/POP words counted, 16-deep instruction queues, PUSH visible one cycle
later, MMA drain 56/88 (plan 8), accumulate writeback-bound at K ≤ 64 — and
`tcta.timing.KV260` now carries that structure (`rtl_rules`) with every
constant measured; `KV260_PLAN` keeps the planning descriptor for the
phase-5/6 tables. Formal: SymbiYosys BMC depth 24 passes on the token FIFO
(6 assertions) and the bank arbiter (18), and the same properties run as
assertions on all programs. Synthesis: a per-module Yosys
`synth_xilinx` estimate (`rtl/synth/yosys_ooc.py`,
`rtl/synth/yosys_xck26_estimate.md`) puts the core at 1.30 M LUTs, 79 k FFs,
546 DSPs, scratchpads in block RAM; the KV260 has 117 k LUTs. Vivado timing
needs an x86 machine (`rtl/synth/vivado_ooc.tcl`).

## What to read first

1. `docs/tilelang-cutoff-verdict.md`, Phase 8 K0 section (fit verdict,
   eight findings) and Phase 7 section (verdict per §3, the numbers in one
   table, eight findings, the calibration pointer); `docs/phase8-k0-fit-report.md`.
2. `docs/calibration-report.md`: what was measured parameter by parameter
   (§1), what still differs (§2), what it says about the plan (§4).
3. `rtl/README.md` (the core as built, parameters, numerics) and
   `rtl/test/README.md` (programs, vector format, the negative-test mapping).
4. Phase 6: `docs/phase6-mlp-report.md`, the Phase 6 verdict section,
   `docs/cycle-model-report.md` (the planning rules), `model/mlp_results.md`,
   `compiler/prototype/PINNED.md` (phase 6 and 7 notes).

## Deliverables (phase 7; earlier phases unchanged, see the verdict)

| # | Path | Status |
|---|---|---|
| J1 | `sim/tcta/encode.py`, ISA §12, `tests/test_encode.py`; `rtl/gen_pkg.py` → `rtl/src/tcta_isa_pkg.sv` | done (c771593) |
| J2 | `rtl/src/*.sv`, `rtl/src/fp/*.sv`, `rtl/test/fp/` (FP library vs NumPy: 0 mismatches / 200 000) | done, lint clean |
| J3 | `rtl/test/{tb_top.sv,tb_main.cpp,Makefile,compare.py,gen_vectors.py}`, `vectors/<program>/` | done |
| J4 | GEMM bit-exact, FA within §9 (4.03e-3 ≤ 7.81e-3), negative tests TOKEN_LEAK / TOKEN_TIMEOUT / UNTOKENED_READ (`TOKEN_CHECK=1`) | PASS |
| J5 | `sim/tcta/timing.py` (`KV260` measured, `KV260_PLAN`), `sim/tcta/cycle.py` (`rtl_rules`), `rtl/test/calibrate.py`, `docs/calibration-report.md`, `tests/test_cycle.py` (+1) | GEMM −0.0 %, FA +0.3 % |
| J6 | `rtl/formal/{token_fifo,bank_arbiter}.sby`, properties under `ifdef FORMAL`; `make all FORMAL=1` passes | PASS (sby BMC depth 24, both jobs) |
| J7 | verdict Phase 7 section; `rtl/synth/yosys_ooc.py` → `yosys_xck26_estimate.md`, `yosys_xck26.ys`, `vivado_ooc.tcl`, `vivado_ooc.xdc` | estimate done: 11× the KV260 LUTs (FAIL on fit); Fmax pending Vivado |
| E1 | `isa/isa-v0.1.md` §16, `sim/tcta/isa.py` (`sum_lanes`), `sim/tcta/machine.py` (`_mma_sum`, `_row_sum`), `tests/test_sim.py` | done |
| K0.1 | `rtl/src/dma_store.sv`, `dma_load.sv`, `dequant.sv`; `gather` program (`gen_vectors.py`, `vectors/gather/`) | done (71e9c4d): 768 k → 7 k, 47 k → 18 k LUT |
| K0.2 | `rtl/src/vec_unit.sv`, `sfu.sv`; cycle model `vec_reduce_extra`, `sfu_sum_row_extra` | done (5055321, aba825d): 123 k → 54 k LUT |
| K0.3 | `rtl/src/mma_unit.sv`, `mma_opbuf.sv`, `mma_cell.sv`, `fp/fix_to_f32.sv`, `tcta_pkg.sv` (FIX_*); `sim/tcta/machine.py` (`_mma_sum_fixed48`), `isa.py` (`acc_mode`); ISA §9 + E5; `cycle.py::_rtl_mma_cycles`; `rtl/synth/yosys_ooc.py` (DSP48E2 stub, `mma_opbuf_a`) | done (0d2735f): 290 k → 52 k LUT, bit-exact |
| K0.4 | `docs/phase8-k0-fit-report.md`, `rtl/synth/yosys_xck26_estimate.md` | done: 208 k LUT-eq vs 117 k (FAIL on fit; fallback stated) |
| L1 | `rtl/src/queue.sv`; `sim/tcta/machine.py` (E6 K check), `tests/test_sim.py`; `rtl/synth/yosys_ooc.py` (LUT-eq columns) | done (e88d321): queues 15.9 k → 3.0 k LUT-eq |
| L2 | `rtl/src/tcta_pkg.sv` (`SP_READERS / SP_WRITERS`, one address), `scratchpad.sv`, `tcta_core.sv`; `rtl/formal/bank_arbiter.sby` | done (f4afb57): pads 40.8 k → 14.5 k, glue 8.4 k → 3.6 k |
| L3 | `rtl/src/mma_unit.sv` (`W_ADD`, `ADD_LANES`); `sim/tcta/timing.py` (`mma_cv_row_cycles`, wb extras), `cycle.py`, `tests/test_cycle.py` | done (a8244a2): `mma_unit` 51.8 k → 46.0 k; FA +1.4 % |
| L4 | `docs/phase8-k0-fit-report.md` §4b | not taken: 64 RAMB18 for ≈ −3 k; block RAM is the tighter resource |
| L5 | `tcta_pkg.sv` (`TCTA_MAC8`), `mma_unit.sv`, `mma_opbuf.sv`; `make all MAC8=1`; `yosys_ooc.py --define/--md`; `rtl/synth/yosys_xck26_estimate_mac8.md`; `timing.KV260_L5`; `calibrate.py --timing=` | measured, not taken: −17 445 LUT-eq (est. −35 k), MMA 3–4×, FA +72 % |
| K3 | `rtl/src/vec_unit.sv` (`sfu_can_take`, `sfu_load`, `sum_tag`), `mma_unit.sv` (`pf_d`, `pf_out`, `pf_v`, `take_next`, the prefetch states); `rtl/test/gen_vectors.py` (`mma_mask`, `sfu`), `compare.py` (`f32tol`); `sim/tcta/timing.py` (`sfu_fill_cycles`, `sfu_vec_read_cycles`; `mma_wb_*` dropped), `cycle.py` (writeback FSM replay, SFU rules), `tests/test_cycle.py` | done: FA −17.1 %, SFU 4 per slice, accumulate rows 2; `vec_unit` 56.1 k → 55.9 k, `mma_unit` 53.2 k → 54.1 k |
| K2 | `rtl/src/tcta_pkg.sv` (`col_req`, `vec_col_addr`, `req.cs`), `scratchpad.sv` (serial path), `vec_unit.sv` (`cs_mode`, `rowvec_q`), `dma_store.sv` (read lane masks); `rtl/formal/bank_arbiter_vec.sby`; `sim/tcta/cycle.py` (`_vec2d`, VEC / DMA rules) | done: FA −18.9 %, `[64, 1]` ops 9–14 cycles; VEC pad 5.0 k → 8.0 k, `vec_unit` 53.9 k → 56.1 k |
| K1 | `rtl/src/dma_load.sv`, `dma_store.sv`, `tcta_pkg.sv` (`DMA_ROWS_IN_FLIGHT`); `sim/tcta/machine.py` (`LOAD.IDX` b32, `_row_source` phase), `dtypes.py` (`col_phase`), `cycle.py::_rtl_dma_cycles`, `timing.py`; `gen_vectors.py` gather; `isa/isa-v0.1.md` E7 | done: 14.5 B/cycle (≥ 14), FA −19.6 %, GEMM −46.9 %; `dma_store` 7.4 k → 11.4 k (f4.r16) |

## Key numbers (v1 KV260 core, RTL cycles, 24-cycle harness memory latency)

- GEMM 5917 (planning model 5633, calibrated 5914); FA 21593 (27856;
  21584); dma_copy 8396 (a `[64, 64]` tile through a 2-D VEC slot, E2
  element-serial), mma 23138, mma_mask 16281, vec 14610, sfu 18898, loop 1031
  (model 955: pad contention), gather 23089, all within 0.2 % after
  calibration except loop (after K2: FA 26049, mma 23230; after K1: FA
  32109, vec 14910, dma_copy 2228, loop 489, gather 22683;
  after L3: GEMM 11144, FA 39919, mma 30512, vec 22317; K0: FA 39371; phase
  7: GEMM 11291, FA 38543).
- LOAD.2D: 5 + 24 + rows · max(beats + 1, slices + 1) + fill + 1 (64×128
  BF16 → 1127, 14.5 B/cycle; K0 row-serial 3205); STORE.2D: 4 + rows ·
  max(slices + 1, beats + 1) + fill + 1 (64×128 f32 → 2126; K0 3140);
  STORE.2D f4.r16 9 cycles per slice (64×128 → 5069); LOAD.IDX bf16 128
  rows 2239, f4.r16 1195, f4.b32 827.
- MMA 64×128×128: 4284 (accumulate 4286; L3: 4332) = first fill 131 + 32
  blocks × 128 + drain 57 (59); MMA.MASKED 4302, mask-mul accumulate 4350;
  GEMM's MMA 2571 (P = 100); FA QKᵀ 1148, PV accumulate 1150 (K3: rows
  pipelined through the writeback port, 2 cycles per accumulated row; L3:
  1436, writeback-bound at P = 64).
- VEC (K3): SFU 4 cycles per slice (VEXP2 / VRCP 64×128 2055, VLOG2 /
  VSIGMOID 64×64 1031, VEXP2 64×64 with a vector 1039, VEXP2.SUM 64×64 1052
  / 64×128 2076, `[64, 1]` VEXP2 23 / with a vector 31; VGATE decay 64×64
  1543 — two reads per slice).
- VEC (K2): `[64, 1]` op 9–14 (VFILL 9, imm 10, with a vector 14, SFU 29;
  K1: 70–134); with a column-vector operand `[64, 128]` 522 (K1: 582);
  VEXP2.SUM 64×64 1874 (K1: 1994); row access to a `[M, N > 1]` 2-D VEC
  tile one cycle per element (LOAD / STORE 64×64 f32 4207 / 4182).
- VEC (K1): `[64, 128]` op 518; with a column-vector operand 582; `[64, 1]` op
  70 (134 with a vector operand); VEXP2 `[64, 1]` 389; VEXP2.SUM `[64, 64]`
  1994 (plan 3072); row reductions +4 (shared reducer).
- Front end: 3 cycles per word (PUSH/POP too), extension word 2 + 1 per
  entry (+1 per `_mod`), LOOP/ENDLOOP 2, LOOP r 1 + ⌈R/step⌉ after VCOUNT.
- Synthesis after K3: core 158 578 LUT + 1 235 RAM32M16 = 168 458 LUT-eq (144 % of the K26) / 43 199 FF / 622 DSP; `mma_unit` 53 199 → 54 124 (the writeback's landing register and muxes), `vec_unit` 56 102 → 55 936.
- Synthesis after K2: core 157 819 LUT + 1 235 RAM32M16 = 167 699 LUT-eq (143 % of the K26) / 42 639 FF / 622 DSP; `scratchpad_VEC` 5 019 → 8 021 (the serial path), `vec_unit` 53 901 → 56 102.
- Synthesis with the 8×8 array (`TCTA_MAC8`, L5): core 145 051 LUT-eq (124 %), `mma_unit` 35 754 (153 DSP); MMA 64×128×128 16 547 cycles, GEMM 11 875, FA 55 269.
- Synthesis (Yosys, cells only, after K1): core 152 624 LUT + 1 234
  RAM32M16 (= 162 496 LUT-eq, 139 % of the K26) / 41 550 FF / 622 DSP;
  `mma_unit` 46 031 + 896 RAM32M16, `vec_unit` 53 581, `dma_load` 17 865 +
  70 RAM32M16, `dma_store` 11 034 + 48 RAM32M16 (7 057 before the f4.r16
  quantiser), pads 14 506, glue 3 592, queues 5 × (316 + 36 RAM32M16). After
  L1–L3 it was 148 864 LUT + 1 196 RAM32M16 (= 158 432 LUT-eq). After K0 it was 200 006 LUT + 1 016
  RAM32M16 (= 208 134 LUT-eq) / 59 878 FF / 624 DSP; `mma_unit` 51 781 +
  896 RAM32M16 (cell 7 LUT + 2 DSP), `vec_unit` 53 581, scratchpads 40 789
  LUT + block RAM (4.4 Mbit, 122 RAMB36 minimum, Yosys uses 2×), `dma_load`
  18 082, queues 15 920, `dma_store` 7 057. Phase 7: 1 296 593 LUT.
  KV260: 117 120 LUT, 234 240 FF, 1 248 DSP, 144 RAMB36, 64 URAM.

## Decisions taken while implementing (phase 7)

- SystemVerilog over Chisel (Kevin); both synthesis routes (Kevin); the OSS
  CAD Suite binaries for Yosys/sby after the distro package proved broken
  (Kevin).
- Scratchpad banks rewritten for block-RAM inference (one process per port,
  a read register per bank and port, `no_rw_check`) after the single-process
  form synthesized to LUT-RAM; behavior unchanged (nine programs, arbiter
  proof). The synthesis estimate is per module with blackboxed children
  because yosys-slang uniquifies every instance and the whole core needs
  > 16 GB; `dma_store` is mapped by a short abc script (default abc9 did not
  finish in an hour on its row buffer).
- Numerics: erratum E1 fixes the MMA k-sum (sequential from +0, products
  rounded first, mask, then + old) and the row-sum order (16 lane partials,
  binary tree) in the ISA and the simulator so that the RTL is bit-exact
  without an FMA-free re-derivation per op; `f32_max` follows `np.maximum`'s
  +0 on a ±0 tie.
- The token checker (`TOKEN_CHECK=1` debug twin) is fed by granted
  scratchpad accesses *and* by the operand tiles an instruction names at
  issue, because the simulator checks the reference, not the bytes (an MMA on
  a zero-shaped untokened tile reads no data).
- Programs ran unchanged; the simulator-level deviations the RTL settled
  (BAD_QUALIFIER for `LOAD.IDX` on block-scaled formats and `STORE.2D
  f4.r16`, BAD_SHAPE for `K = 0`, BAD_OPCODE for SEND/RECV) are recorded,
  not spec changes.
- Calibration entered the RTL's structure as `DeviceTiming.rtl_rules` with
  measured constants rather than fitting the planning rules' values; the
  planning descriptor is `KV260_PLAN` and `model/sweeps.py` stays on it so
  the phase-5 tables reproduce. The `mma` vector program grew the accumulate
  and A-from-ACC forms and a `loop` program was added to measure the front
  end; `tb_top.sv` traces front-end dispatches (`FE DISP`).
- Scratchpad ports in the model are the as-built ones (2 readers + 1 writer
  per bank, banks shared by all slots); the `mma` program's STORE of a tile
  the array is reading showed the planning (1, 1) ports serialize what the
  RTL overlaps.

## Environment

- No GPU, aarch64 (Arch), 16 GB. Interpreter: `~/Projects/tilelang-cutoff/.venv/bin/python`
  (tilelang 0.1.15, Python 3.14). Verilator 5.052 (pacman). Yosys 0.54-3
  from pacman does not load (built against boost 1.89, boost-libs is 1.92);
  `sby` and `yosys-slang` are not packaged. Formal and synthesis use the
  prebuilt OSS CAD Suite (linux-arm64 2026-10-26: Yosys 0.70+221, sby,
  yices, slang) at `~/.local/opt/oss-cad-suite/bin` (prepend to `PATH`).
  Vivado: x86 only.
- Run everything: `python -m pytest tests -q` (~65 s); `make -C rtl/test
  all` (~10 min incl. two Verilator builds); `make -C rtl/test lint`;
  `python rtl/test/calibrate.py`.
- Regenerate: `python rtl/test/gen_vectors.py` (vectors from the simulator;
  `make -C rtl/test vectors`), `python rtl/gen_pkg.py` (after an `encode.py`
  change), `python ref/kernels.py fixtures`, `python compiler/prototype/dump_programs.py`,
  `python model/sweeps.py` (phase-5 tables, `KV260_PLAN`), `python model/sweeps_mlp.py`.
- Formal: `cd rtl/formal && sby -f token_fifo.sby && sby -f bank_arbiter.sby`
  (~5 s). Synthesis estimate: `python3 rtl/synth/yosys_ooc.py -j 2` (~20 min;
  `--only <jobs>` reruns a subset; `mma_unit` and `vec_unit` are the long
  jobs); `vivado -mode batch -source rtl/synth/vivado_ooc.tcl` on x86 for timing.

## Not done / out of scope

Fmax (Vivado on x86). Making the core fit the KV260 beyond L1–L3 and K1
(162.5 k LUT-eq): the lane-width decision of `docs/phase8-gate2-task.md`
§0.3 from the Vivado number (16-lane core needs ≥ 39 % under Yosys, the 8×8
array on it ≥ 31 % measured, the 8-lane configuration fits by Yosys), L6–L7
of the fit report §4 are estimates, L4 is analysed and not taken (§4b), L5
measured and not taken (§4c). Phase 8 K3–K8 of `docs/phase8-gate2-task.md` not started:
MoE/MLA/KDA on the RTL (mechanical through `gen_vectors.py`; they measure
VTOPK, VGATE/VMASK/VSEL, which the calibration still takes from the FSMs,
and KDA's gating tile would measure the E2 serial path), the RTL items the
calibration surfaced (SFU slice pipelining and the pipelined accumulate
writeback K3, the front end K4),
the `f4.r16` egress trim (K8 list), COMM, multi-core, AXI/DDR, the board,
power, ASIC synthesis. Open for the main session from phase 6: the
arbiter's grant policy (convoy), `sfu_lanes`, request-burst queueing at 4
cores.
