# Handoff: TileLang cut-off task, phases 1–7 and phase 8 K0 (2026-10-29)

For the session steering `docs/tilelang-lowering-cutoff-task.md`, its
addendum, `docs/tilelang-phase2-moe-task.md`, `docs/tilelang-phase3-task.md`,
`docs/tilelang-phase4-task.md`, `docs/phase5-cycle-model-task.md`,
`docs/phase6-mlp-task.md`, `docs/phase7-rtl-core-task.md` and
`docs/phase8-amendment-fit.md` (K0: fit). All deliverables of phases 1–7 and
phase 8 K0.1–K0.4 exist; the whole repo is green (`pytest tests -q`: **94
passed**; `make -C rtl/test all`: **10/10 PASS**, lint clean). The ISA changed only where the main session sent deltas
(v0.1.5 = Gate 1 frozen, §12 normative) plus erratum E1 (§16, the MMA k-sum
and row-sum orders, which the simulator now implements). **This repo is
canonical**: the main session sends spec and doc deltas only, no `sim/`
copies. Remote: `github.com/kaijunli-infr/tcta` (private).

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

## Key numbers (v1 KV260 core, RTL cycles, 24-cycle harness memory latency)

- GEMM 11144 (planning model 5633, calibrated 11141); FA 39371 (27856;
  39370); dma_copy 4560, mma 30480, vec 22317, loop 814, all within 0.1 %
  after calibration (phase 7: GEMM 11291, FA 38543).
- LOAD.2D row: 24 + beats + slices + 2 (64×128 BF16 → 3205, 5.1 B/cycle);
  STORE.2D row: 2·slices + beats + 1 (64×128 f32 → 3140).
- MMA 64×128×128: 4284 (accumulate 4316) = first fill 131 + 32 blocks × 128
  + drain 57 (89); GEMM's MMA 2571 (P = 100); FA QKᵀ 1148, PV accumulate
  1299 (writeback-bound at P = 64).
- VEC: `[64, 128]` op 518; with a column-vector operand 582; `[64, 1]` op
  70 (134 with a vector operand); VEXP2 `[64, 1]` 389; VEXP2.SUM `[64, 64]`
  1994 (plan 3072); row reductions +4 (shared reducer).
- Front end: 3 cycles per word (PUSH/POP too), extension word 2 + 1 per
  entry (+1 per `_mod`), LOOP/ENDLOOP 2, LOOP r 1 + ⌈R/step⌉ after VCOUNT.
- Synthesis (Yosys, cells only, after K0): core 200 006 LUT + 1 016
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

Fmax (Vivado on x86). Making the core fit the KV260 beyond K0: the levers
L1–L7 of `docs/phase8-k0-fit-report.md` §4 (queues in LUT-RAM, scratchpad
clients per the ISA access table, writeback adders, operand buffers in
RAMB18; the lane-width decision — 8×8 array / 8 lanes — for the main
session; Vivado first). The K1–K8 task text was not delivered to this repo.
Phase 8: MoE/MLA/KDA on the RTL (mechanical through
`gen_vectors.py`; they measure the gather path, VTOPK, VGATE/VMASK/VSEL and
the dequant formats the calibration takes from the FSMs), the RTL items the
calibration surfaced (row prefetch / pipelined DMA, column-vector layout, SFU
slice pipelining, pipelined accumulate writeback), COMM, multi-core, AXI/DDR,
the board, power, ASIC synthesis. Open for the main session from phase 6: the arbiter's grant
policy (convoy), `sfu_lanes`, request-burst queueing at 4 cores.
