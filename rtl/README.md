# TCTA core v0 (phase 7): one core of ISA v0.1.5 in SystemVerilog

Task: `docs/phase7-rtl-core-task.md`. Language: SystemVerilog (decided over the
task's Chisel default; same module boundaries as the Gemmini-style structure the
task names). Scope: one core, the v1 KV260 descriptor (§11), fed by the §12
binary stream, run under Verilator against the Python simulator's programs.
Config C is a parameter set of the same design, not a second design. Status:
Phase 7 section of `docs/tilelang-cutoff-verdict.md`; timing:
`docs/calibration-report.md`.

```
rtl/
  gen_pkg.py  encode.py -> src/tcta_isa_pkg.sv (opcodes, flags, qualifiers, edges, incr field positions)
  src/        the core: tcta_isa_pkg.sv (generated), tcta_pkg.sv (parameters, records, descriptor and address
              helpers), tcta_core.sv, front_end.sv, queue.sv, token_fifo.sv, token_net.sv, scratchpad.sv (column
              banks + the two-port arbiter), tile_meta.sv, token_check.sv (debug twin of the simulator's token
              checks, TOKEN_CHECK=1), dma_load.sv, dma_store.sv, dequant.sv, mma_unit.sv, mma_pe.sv, vec_unit.sv,
              sfu.sv (tables + lane), fp/f32_add.sv, fp/f32_mul.sv, fp/f32_misc.sv
  test/       Verilator harness (tb_top.sv, tb_main.cpp, Makefile, compare.py), gen_vectors.py, calibrate.py,
              vectors/<program>/, fp/ (FP library unit test); see test/README.md
  formal/     SymbiYosys jobs (token_fifo.sby, bank_arbiter.sby, depth 24); the properties live in token_fifo.sv
              and scratchpad.sv under `ifdef FORMAL (also compiled as runtime assertions by `make all FORMAL=1`)
  synth/      Yosys synth_xilinx estimate: yosys_ooc.py (per module, children blackboxed; writes
              yosys_xck26_estimate.md), yosys_xck26.ys (whole design, needs > 16 GB), and the Vivado out-of-context
              scripts for the KV260 part (vivado_ooc.tcl, vivado_ooc.xdc: 200 MHz)
```

## 1. Parameters (`tcta_pkg.sv`)

| Parameter | v1 KV260 | config C (per core) | Meaning |
|---|---|---|---|
| `T_M, T_K, T_N, T_V, T_VR, T_S` | 64, 128, 128, 64, 512, 128 | same | tile shapes (§2.2) |
| `N_INP, N_WGT, N_ACC, N_VEC, N_STATE` | 2, 4, 6, 16, 1 | 5, 32, 16, 64, 2 | slots (ACC: 2 + one wide = 6 indices) |
| `VEC_ROW_SLOT0` | 8 | 32 | VEC slots ≥ this are row tiles `[1, T_VR]` |
| `FIFO_DEPTH` | 8 | 8 | tokens per edge |
| `MAC_ROWS, MAC_COLS` | 16, 16 | 64, 64 | systolic array |
| `VEC_LANES, SFU_LANES` | 16, 4 | 64, 16 | VEC pipe width, SFU pipe width |
| `BANKS` | 16 | 64 | column banks per scratchpad (= MAC_COLS = VEC_LANES) |
| `DMA_BYTES` | 16 | 32 | bytes per beat on the memory port |
| `ROWBUF_BYTES` | 2048 | 2048 | DMA row buffer (one tile row at most) |
| `N_REGS` | 16 | 16 | front-end registers |
| `IQ_DEPTH` | 16 | 16 | dispatched-but-unissued instructions per queue |
| `LOOP_DEPTH` (front end) | 4 | 4 | nested LOOP levels |
| `N_DESC` | 256 | 256 | descriptor table entries (§12) |

## 2. Microarchitecture (as built; the cycle costs are the measured ones of `docs/calibration-report.md`)

**Front end** (`front_end.sv`). Reads the §12 stream one 128-bit word per
cycle, decodes (fetch, decode, dispatch: 3 cycles per word, PUSH/POP words
included), fetches and applies the 0–3 extension words (2 cycles per word plus
one per loop-variant entry; a `_mod` entry is a modulo formed by repeated
subtraction, one more cycle), keeps a `LOOP_DEPTH`-deep loop stack with
per-level iteration counters (static count, or `LOOP r` with
`⌈R[r]/step⌉` formed by repeated subtraction after every `VCOUNT` writing
`R[r]` has completed; zero skips the body), handles `SET*` and `FINISH`, and
dispatches into the five queues' instruction buffers (`IQ_DEPTH`; a full
buffer stalls the front end and everything behind that instruction). LOOP and
ENDLOOP cost 2 cycles each.

**Queues** (`queue.sv`, five instances). In order, one instruction in
flight. Each cycle a queue retires every leading `PUSH`/`POP` whose edge can
proceed and, in the same cycle, issues the next work instruction to its unit
if the unit is free (ISA §8.5: syncs cost no issue slot, measured); the
PUSHes that follow a work instruction retire in the cycle the unit reports
DONE, and the consumer's POP sees the token the cycle after. The next work
instruction issues the cycle after DONE. A core-level idle counter raises
TOKEN_TIMEOUT when no queue progresses for `t_deadlock` cycles.

**Token network** (`token_net.sv`): one `token_fifo` (an up/down counter
with full/empty; tokens carry no payload) per edge of §8.1; PUSH on an unknown
edge is BAD_EDGE. At FINISH a non-empty FIFO is TOKEN_LEAK. Formal target
J6 (`ifdef FORMAL`: the count tracks pushes − pops, never exceeds DEPTH, a
push into a non-full FIFO is never dropped, a pop from a non-empty one never
refused).

**Scratchpads** (`scratchpad.sv`, five instances; INP/WGT 16-bit, the others
32-bit). Sixteen column banks (`bank = column mod BANKS`, address = slot
row-base + row + column-group), shared by all slots of the scratchpad. Each
bank has two ports; the arbiter grants at most two of the seven clients per
cycle, at most one of them writing, fixed priority MMA A, MMA B (always
granted, so the array never stalls), MMA writeback, VEC read, VEC write, LOAD,
STORE; a request is one whole row slice (all banks, one element each) and a
losing client retries. The MMA's skew makes every operand fetch hit 16
distinct banks. Each bank is written for block-RAM inference (one process per port, a
read register per bank and port, `no_rw_check`; J7). Properties (`ifdef FORMAL`): never two writers, at most one
port per client, the highest-priority requester always granted, two pure
reads at the top both granted. Per-slot metadata (`rows`, `cols`, `pready`;
`tile_meta.sv`, three write ports) is what the units read for operand
shapes, as the simulator's `TileState.valid`.

**DMA** (`dma_load.sv`, `dma_store.sv`, `dequant.sv`), over the testbench
memory model (fixed latency, `DMA_BYTES` per beat, any alignment). Row at a
time: LOAD computes the row address (2D/3D strides folded by the assembler;
`LOAD.IDX` reads the index slice every 16 rows, skips −1 rows with a zero fill,
faults IDX_RANGE / IDX_SHORT), issues the row's beats (plus two scale beats
for block-scaled formats), waits for them, then writes one 16-element slice
per cycle through `dequant_elem` (`bf16 / f32 / i32 / f8.t / f8.b32 / f4.b32
/ f4.r16`, RNE to BF16 for INP/WGT destinations) — `24 + beats + slices + 2`
cycles per row at the harness's latency. STORE reads one slice per two
cycles, converts (`f32`, `i32` = rint, `bf16` RNE), packs a byte row buffer
and writes it as beats with byte enables (`2·slices + beats + 1` per row).
The next row's fetch does not overlap the current row (the main gap to the
plan: 5 B/cycle instead of 16 on a 64×128 BF16 tile).

**MMA unit** (`mma_unit.sv`, `mma_opbuf.sv`, `mma_cell.sv`, `fp/fix_to_f32.sv`;
phase 8 K0.3). Output-stationary 16×16 systolic array of `mma_cell`, each
cell two DSP48E2 multiply-accumulators (signed 24×18, 48-bit sums, erratum
E5's two B levels) plus sticky NaN/inf flags; the FP32 cell `mma_pe` remains
under the `ACC_F32` parameter. Operands are block-buffered: an A fill reads
one 16-row block (`16·⌈K/16⌉` slice reads) and a B fill one block's 16
columns (`K` reads for `MMA`, `16·⌈K/16⌉` for `MMA.T`) into one of four
LUT-RAM buffer halves each (`mma_opbuf`; the B buffer is 16 per-lane RAMs
behind an XOR permutation so a column slice scatters and a row streams
without a crossbar); the fill also records every row's / column's largest
exponent. Fills run back to back while a half is free; a half is ready two
cycles after its last read and is released `K + 20` cycles after the last
block start that used it. A product `[M, K] × [K, N]` is `⌈M/16⌉ · ⌈N/16⌉`
blocks, row-block major; blocks start every `P = max(K, 16)` cycles when
both halves are ready and the 32-row result FIFO has room for one. Row i of
a block streams `a[i0+i][k]` shifted into the block's window from cycle
`T + 1 + i`, column j streams `b[k][j0+j]` from `T + 1 + j`; cell `(i, j)`
finishes at `T + K + 5 + i + j` and a one-hot collector per lane writes
`{flags, 57-bit sum}` into the FIFO (row i lands `K + 21 + i` cycles after
the block start). A converter (8 `fix_to_f32` lanes, 2 cycles per row)
rounds each row once to binary32 with the block's exponent maxima; the
writeback FSM takes one converted row at a time: mask (`MMA.MASKED`: a mask
slice read, then one add cycle or two multiply cycles on 8 multipliers),
`+ old` (accumulate: a read of the destination slice, 2 more cycles), the
write — 2 cycles per row plain, 4 accumulate, 5 masked add, 6 masked mul;
padding rows of the last row block cost the converter only. `wide` and
`row0` are address arithmetic there; A from ACC reads the BF16 payload
written by `VCVT` in place. Faults: K = 0, K > 128 or a bad scratchpad
BAD_SHAPE, a bad mask mode BAD_QUALIFIER. `docs/calibration-report.md` §1.2
has the measured cycles; `sim/tcta/cycle.py::_rtl_mma_cycles` replays this
schedule cycle by cycle.

**VEC unit** (`vec_unit.sv`, `sfu.sv`). One in-order pipe over row slices:
a micro-sequencer issues one scratchpad read per cycle (the row-vector
operand once per row, then each slice operand), a tag queue (2) and a 4-deep
write FIFO decouple the response stage, which computes when a slice's last
operand returns (binary32 RNE, the simulator's operand orders). A column
vector of M elements is M slices (one element per row), a row-vector operand
one read per row; per instruction ~6 cycles of setup, drain and metadata.
SFU ops (`VEXP2[.SUM] [vec]`, `VRCP`, `VLOG2`, `VSIGMOID`, `VGATE decay`)
hold one slice and pass it through `SFU_LANES` lanes in `L / SFU_LANES`
steps (6 cycles per slice; `VEXP2.SUM` is one fused pass: subtract, exp2,
running sum, 7 per slice + 2 per row). Reductions (`VROWMAX`, `VROWSUM`)
keep 16 lane partials and add them as a binary tree (E1). `VTOPK` is k
selection passes over one row (stable), `VCOUNT` a counting pass that writes
a front-end register, `VSEL`, `VMASK causal/band/tile`, `VFILL imm`, `VCVT`
(BF16 RNE into INP/WGT, in place P-ready, FP32 move with `row0`), `VCOPY`,
`VSCALE_ACC` (wide aware). Non-SFU ops are bit-exact with the simulator.

**FP library** (`fp/`): `f32_add` (IEEE binary32 RNE, subnormals, inf/NaN as
NumPy, explicit sticky mask), `f32_mul #(MA, MB)` (exact product of
truncated-mantissa operands, so a BF16×BF16 product is exact unless
subnormal), `f32_max` (NaN propagates; equal values return +0 for a ±0 tie,
as `np.maximum`), `f32_to_bf16` (RNE; every NaN to 0x7fc0), `f32_to_i32_rne`,
`i32_to_f32`, `f32_to_int_trunc`; `sfu_tables` + `sfu_lane` (exp2 / rcp /
log2 / sigmoid, 25-bit tables with linear interpolation, §9: ≤ 2 ulp BF16).
`test/fp/check_fp.py` compares 200 000 random operand pairs with NumPy
float32: 0 mismatches.

**COMM**: `SEND`/`RECV` decode to a stub that raises BAD_OPCODE in v0.

**Status** (§10): the harness's `status.txt` carries `ok <cycle>` or `fault
<code> <queue> <pc> <cycle>`; fault priority token-check > units > queues >
front end > TOKEN_TIMEOUT > TOKEN_LEAK at finish. UNTOKENED_READ /
UNTOKENED_WRITE are raised only by `token_check.sv` (`TOKEN_CHECK=1`), which
watches every granted scratchpad access and the operand tiles named at issue
against the token FIFO history, so the negative test faults at the same
instruction as the simulator. VALID_MISMATCH and ACC_NOT_CONVERTED are
simulator-only.

## 3. Numerics the RTL and the simulator share (ISA §9, errata E1 and E5)

- MMA (`acc_mode="fixed48"`, erratum E5, `DEVICE_KV260`): per output element
  the K products are summed exactly in a 48-bit window placed by the block's
  exponent maxima `ea_max(i)`, `eb_max(j)` (A significands shifted to
  `ma · 2^(15 − da)`, truncated below the window; B significands in two
  levels `mb · 2^(9(l+1) − db)` for `9l < db ≤ 9(l+1)`, level 1 keeping the
  tail truncated), `(acc0 · 2^9 + acc1) · 2^(ea_max + eb_max − 301)` rounded
  once to binary32 (RNE, subnormals, overflow), NaN / ±inf as sticky flags.
  Bit-exact between `sim/tcta/machine.py::_mma_sum_fixed48` and
  `mma_cell.sv` + `fix_to_f32.sv` on every harness program (GEMM, the MMA
  forms, the gather test). Then `S' = S ∘ mask` (MMA.MASKED) and
  `fl(old + S')` if accumulate, in binary32 as E1. Measured against E1's
  sequential binary32 sums (`tests/test_sim.py`, 64×128×128): Gaussian data
  max 0.5 ulp, outlier row / column 0.5 ulp, A spanning 2^±20 mean 216 ulp
  (7.7e-7 of the `max|A row| · max|B col|` bound, binary32's own sequential
  error 6.1e-7), softmax·V mean 52 ulp (2.15e-7 vs 1.77e-7); a cancelling
  row `[1e8, −1e8, 1, 1, 3e-5] · 1` gives 0 where binary32 gives 2.00003
  (both within 2^-23 of the bound); specials, subnormal and overflow results
  identical to binary32.
- MMA (`acc_mode="f32"`, erratum E1, the `ACC_F32` array of `mma_pe`):
  `S = fl(…fl(fl(0 + a0·b0) + a1·b1)… + a_{K−1}·b_{K−1})`, products rounded
  to binary32 first (exact unless subnormal), then mask and `+ old` as above.
- Row sums: 16 lane partials, lane l over columns `l, l+16, …` in order, then
  a binary tree over lanes.
- Elementwise ops: binary32 RNE, one rounding per op, same operand order as
  the simulator (`x − v`, `x · v`, `x + v`, `max(x, v)`, `−x`, `acc · vec[r]`,
  `acc ∘ g`).
- BF16 rounding: RNE on the binary32 bit pattern (`dtypes.to_bf16`), every
  NaN to the canonical quiet NaN.
- SFU: within 2 ulp BF16 of the double result; FA is checked against the
  fixture with the §9 tolerance (RTL row-max error 4.03e-3, equal to the
  simulator's own, tolerance 7.81e-3), 138 of its output bytes differ from
  the simulator in the last bits.

## 4. Testbench

`test/README.md`: the ten programs (six positive, the front-end
micro-test, three negative), the vector directory, the memory model, the
trace format (`cycle queue SYNC|ISSUE|DONE op pc`, `cycle FE DISP op pc`),
`compare.py`'s checks and `calibrate.py`. `make -C rtl/test all` runs
everything (`neg_untokened` with `TOKEN_CHECK=1`); `make all FORMAL=1`
compiles the formal properties as runtime assertions; `make lint` is clean
under `-Wall`.

Known deviations from the simulator, all recorded in the verdict:
`STORE.2D f4.r16` and `LOAD.IDX` on block-scaled formats are BAD_QUALIFIER;
`MMA` with `K = 0` is BAD_SHAPE; SEND/RECV are BAD_OPCODE (COMM stub); the
memory port takes 16-byte beats at any alignment (a real AXI front adds the
alignment logic outside the core).

## 5. Toolchain

Verilator 5.052 (pacman). The formal and synthesis flows need Yosys with
`yosys-smtbmc`, the `slang` frontend plugin (the packages use typed
localparams the native parser rejects) and SymbiYosys; the distro `yosys
0.54-3` does not load on this machine (linked against boost 1.89, installed
boost-libs 1.92) and `sby` is not packaged, so both ran from the OSS CAD Suite
(linux-arm64 2026-10-26, Yosys 0.70+221) under `~/.local/opt/oss-cad-suite/bin`:

```
export PATH=~/.local/opt/oss-cad-suite/bin:$PATH
cd rtl/formal && sby -f token_fifo.sby && sby -f bank_arbiter.sby   # BMC depth 24, both PASS
python rtl/synth/yosys_ooc.py -j 4                                  # from the repo root -> rtl/synth/yosys_xck26_estimate.md
```

The whole-design script (`yosys_xck26.ys`) needs more memory than this machine
has: yosys-slang flattens the design, and with `--keep-hierarchy` it emits one
module per instance (256 `mma_pe`, 387 `f32_add`), 11 GB before synthesis
starts and swapping in ABC. `yosys_ooc.py` synthesizes each distinct module
(and parameterization) once with the other modules as blackbox stubs generated
from their headers, and sums the hierarchy from the instance counts.

The BMC starts from an arbitrary state, so the FIFO properties assume reset
in the first cycle and check from the second (`f_past_valid`); the arbiter's
are purely combinational. Vivado runs on x86: `vivado -mode batch -source
rtl/synth/vivado_ooc.tcl` from the repo root writes
`rtl/synth/vivado/{utilization,timing}.rpt`.
