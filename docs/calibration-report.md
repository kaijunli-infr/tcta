# Calibration report (phase 7, J5): the cycle model against the RTL

Task: `docs/phase7-rtl-core-task.md` J5. Code: `sim/tcta/timing.py` (`KV260`
now carries the measured values under `rtl_rules=True`; `KV260_PLAN` is the
phase-5 planning descriptor of the same core), `sim/tcta/cycle.py` (the
`rtl_rules` cost functions `_rtl_dma_cycles`, `_rtl_vec_cycles`,
`_rtl_mma_cycles`, the front-end word costs, the instruction-queue depth, the
PUSH visibility), `rtl/test/calibrate.py` (pairs the Verilator trace with the
model per instruction and prints every table below), `tests/test_cycle.py::
test_kv260_calibrated_reproduces_rtl_durations`. RTL: `rtl/` (SystemVerilog,
Verilator 5.052); harness memory model: fixed read latency 24 cycles
(`+MEMLAT`), 16 bytes per cycle, one read and one write port.

Rule of the task (§2): a planning value the RTL pins down is *replaced* by the
measured one, never tuned. Where the RTL's structure is not the planning rule
(row-serial DMA, slice-serial VEC, a 3-cycle front end), the structure is
entered as a rule (`rtl_rules`) with its constants measured; the planning
rules stay available as `KV260_PLAN` and are unchanged for `CONFIG_C`.

## 0. Result

| program | RTL cycles | model, planning (`KV260_PLAN`) | model, calibrated (`KV260`) |
|---|---|---|---|
| dma_copy (LOAD f32 64×64, STORE) | 4560 | 2248 (−50.7 %) | 4557 (−0.1 %) |
| mma (2 LOAD, 4 MMA forms, VCVT, 2 STORE) | 30480 | 24280 (−20.3 %) | 30478 (−0.0 %) |
| vec (the FA VEC chain on one tile) | 22317 | 14056 (−37.0 %) | 22314 (−0.0 %) |
| **gemm** (compiler GEMM, 50×100·100×96) | **11144** | 5633 (−49.5 %) | **11141 (−0.0 %)** |
| **fa** (compiler FA, 4 KV tiles of 64) | **39371** | 27856 (−29.2 %) | **39370 (−0.0 %)** |
| loop (front-end micro-test) | 814 | 1096 (+34.6 %) | 815 (+0.1 %) |

Re-measured after phase 8 K0.2 / K0.3 (the shared VEC reducer, the
fixed-point MMA unit, 2026-10-29): every MMA shape matches the RTL exactly
(§1.2) and the reductions within 1 cycle (§1.3). Phase-7 numbers: mma 29952,
vec 22177, gemm 11291, fa 38543 (model +0.3 %).

GEMM and FA are within 10 % (the J5 bar), within 0.3 % in fact, but only
after the rule changes of §1: with the planning rules the model
underestimates the v0 core by 28–50 %. Per the decision rule of the task
(§3) that is **PASS-WITH-CAVEATS** for J5; the caveat is that the v0 core is
not the planned machine in three places (DMA, VEC vectors, front end), which
§4 quantifies and §5 turns into the RTL work that would make it so.

## 1. What was measured, parameter by parameter

The RTL trace (`rtl_trace.txt`: `cycle queue ISSUE|DONE|SYNC op pc`, plus
`cycle FE DISP op pc` for every front-end dispatch) gives per instruction the
issue → done duration; shapes come from the simulator's recording. Every
constant below was read off that trace and then confirmed against the FSM
that produces it (file:line in the right column).

### 1.1 DMA (`dma_load.sv`, `dma_store.sv`)

| | planning (`KV260_PLAN`) | measured (`KV260`) |
|---|---|---|
| structure | one request per LOAD.2D/STORE.2D: `dram_latency_cycles` once, then `bytes / 16` cycles | **row-serial**: one row at a time, the next row's fetch starts after the current row is written (`S_ROW → S_FETCH → S_WAIT → S_DEQ → S_ROW`) |
| LOAD.2D row | — | `24 (latency) + ⌈row_bytes/16⌉ beats + ⌈cols/16⌉ slice writes + 2` |
| STORE.2D row | — | `2·⌈cols/16⌉ (slice read + convert) + ⌈row_bytes/16⌉ beats + 1` |
| per instruction | 0 | LOAD 5 (descriptor, setup, metadata, done), STORE 4 |
| `dram_latency_cycles` | 100 | 24 = the harness's `+MEMLAT` (the plan's fixed-latency port assumption; not a measured LPDDR) |
| `bytes_per_cycle_per_channel` | 16 | 16 (one beat per cycle, confirmed) |
| channels | 1 read + 1 write | same |

Measured rows (RTL cycles; the formula reproduces every one exactly):

| instruction | rows × row bytes | RTL | per row | formula |
|---|---|---|---|---|
| LOAD.2D f32 64×64 | 64 × 256 | 2949 | 46 | 24 + 16 + 4 + 2 |
| LOAD.2D bf16 64×128 | 64 × 256 | 3205 | 50 | 24 + 16 + 8 + 2 |
| LOAD.2D bf16 128×128 | 128 × 256 | 6405 | 50 | |
| LOAD.2D bf16 50×100 (GEMM A) | 50 × 200 | 2305 | 46 | 24 + 13 + 7 + 2 |
| LOAD.2D bf16 100×96 (GEMM B) | 100 × 192 | 4405 | 44 | 24 + 12 + 6 + 2 |
| LOAD.2D bf16 64×64 (FA K, V tiles) | 64 × 128 | 2437 | 38 | 24 + 8 + 4 + 2 |
| STORE.2D f32 64×64 | 64 × 256 | 1604 | 25 | 2·4 + 16 + 1 |
| STORE.2D f32 64×128 | 64 × 512 | 3140 | 49 | 2·8 + 32 + 1 |
| STORE.2D f32 64×1 | 64 × 4 | 260 | 4 | 2·1 + 1 + 1 |
| STORE.2D f32 50×96 | 50 × 384 | 1854 | 37 | 2·6 + 24 + 1 |
| STORE.2D bf16 64×64 (FA O) | 64 × 128 | 1092 | 17 | 2·4 + 8 + 1 |

Effective bandwidth: 5.1 B/cycle on a 64×128 BF16 tile load, 3.4 B/cycle on
the FA K/V tiles, 10.4 B/cycle on a 64×128 f32 store, against the planned 16
B/cycle. The row engine spends the 24-cycle memory latency on every row; a
one-row prefetch would make a row cost `max(latency, beats + slices + 2)`
(26 → 38 instead of 50 → 38 for the FA tile, roughly 1.6–2×), full
pipelining of rows would make it beats-bound (16 B/cycle, the plan). Gathers
(`LOAD.IDX`, `STORE.IDX`) follow the same per-row cost plus 2 cycles per 16
rows for the index slice and 1 per skipped row (from the FSM; no gather
program ran on RTL in this phase).

### 1.2 MMA (`mma_unit.sv`, phase 8 K0.3 unit)

| | planning | measured |
|---|---|---|
| occupancy | `⌈M/16⌉·⌈N/16⌉·K` | blocks start every `P = max(K, 16)` once their operand halves are filled: identical for K ≥ 16 (phase 7's unit needed `ceil16(K)`, K = 100 cost 112) |
| operand fills | — | A: `16·⌈K/16⌉` slice reads per row block; B: `K` reads per block (`MMA.T`: `16·⌈K/16⌉`); four buffer halves each, filled back to back ahead of the array, a half ready 2 cycles after its last read and free `K + 20` cycles after the last block start that used it. The first block starts `max(16·⌈K/16⌉, K) + 3` cycles after issue; the fills otherwise hide behind `P` except `MMA.T` with K not a multiple of 16 (fill-bound at `16·⌈K/16⌉` per block) |
| drain / `mma_latency` | 8 | row i of a block lands in the result FIFO `K + 21 + i` cycles after its start (`mma_result_latency = 21`); a converter takes 2 cycles per row and the writeback 2 cycles per row plain (`+2` accumulate: `W_OLD, W_OLD_WAIT`; `+3` masked add, `+4` masked mul); padding rows of the last row block cost the converter only: drain **57** plain on a full last block (`K + 21 + 2·16 + 4`), **89** accumulate |
| result FIFO | — | 32 rows: a block starts only when ≤ 16 committed rows wait (`mma_rf_rows = 32`) |
| next MMA | may start when the array is free (the drain overlaps) | the queue holds the next instruction until DONE (+1 cycle); no overlap |
| A from ACC | — | same cost as A from INP (two read ports per bank) |

| instruction | RTL | calibrated model | planning model |
|---|---|---|---|
| MMA 64×128×128 (32 blocks, P 128) | 4284 | 4284 | 4104 |
| MMA accumulate 64×128×128 | 4316 | 4316 | 4104 |
| MMA, A from ACC, plain / accumulate | 4284 / 4316 | 4284 / 4316 | 4104 |
| MMA 50×100×96 (GEMM: 24 blocks, P 100) | 2571 | 2571 | 2408 |
| MMA.T 64×64×64 (FA QKᵀ: 16 blocks, P 64) | 1148 | 1148 | 1032 |
| MMA accumulate 64×64×64 (FA PV) | 1299 | 1299 | 1032 |

`_rtl_mma_cycles` replays the unit cycle by cycle (fill engines, block
scheduler, FIFO, converter, writeback FSM) with the constants above and
reproduces every measured shape exactly. The 64×128×128 product costs 132
cycles more than phase 7's unit (the B fill of the first block, `K + 3`),
GEMM 147 fewer (P = 100 instead of 112), FA's QKᵀ 68 more (`16·4` fill) and
its PV accumulate 68 more: at P = 64 each block's 16 rows take 64 cycles to
write back, so the FIFO rule (≤ 16 committed rows) lets a block start only
every ~73 cycles (+14 %), as in phase 7, plus the first fill.

### 1.3 VEC / SFU (`vec_unit.sv`, `sfu.sv`)

| | planning | measured |
|---|---|---|
| elementwise op on an `[M, N]` tile | `⌈M·N/16⌉` | `M · ⌈N/16⌉` slices + **6** (`vec_op_overhead_cycles`; 5 for VFILL, 7 for VROWMAX/VROWSUM) |
| op on a column vector `[M, 1]` | `⌈M/16⌉` (4 for 64) | **M** (64): the lanes span columns; a column vector is one element per row |
| op with a row-vector operand (`vec=`, `VSCALE_ACC`) | same as without | **+1 per row** (the vector element is read before the row's slices) |
| SFU op (VEXP2, VRCP, VLOG2, VSIGMOID) | `⌈M·N/4⌉` (4 cycles per slice) | **6 cycles per slice** (`sfu_slice_cycles`: issue, return, 4 lane steps; one slice in flight), the row-vector read overlapped |
| VEXP2.SUM | 3 passes unfused (12 per slice) | **fused, one pass**: 7 per slice + 3 per row (`sfu_fused_sub_sum = True`, `sfu_sum_slice_extra = 1`, `sfu_sum_row_extra = 1`: the partial add on the lane ALUs since K0.2) |
| row reductions (VROWMAX, VROWSUM, VEXP2.SUM) | — | **+4 at the end** (`vec_reduce_extra`): the shared four-cycle reducer of K0.2 (8 adders, level 0 in the fire cycle); rows of ≥ 4 slices stream without a stall since the last slice is issued when the reducer frees by the time it returns |

| instruction | RTL | calibrated | planning |
|---|---|---|---|
| VADD/VMUL/VNEG/VCOPY/VCVT imm, 64×128 | 518 | 518 | 512 |
| VROWMAX / VROWSUM 64×128 | 523 | 522 | 512 |
| VSUB vec / VSCALE_ACC 64×128 | 582 | 582 | 512 |
| VFILL 64×1 | 69 | 70 | 4 |
| VMAX / VADD / VSUB with a column-vector operand, 64×1 | 134 | 134 | 4 |
| VMUL imm / VCOPY 64×1 | 70 | 70 | 4 |
| VEXP2 64×1 (vec subtrahend) / VRCP 64×1 | 389 | 390 | 16 |
| VEXP2.SUM 64×64 | 1994 | 1994 | 3072 |
| VROWMAX / VMUL / VCVT / VFILL 64×64 | 267 / 262 / 262 / 261 | 266 / 262 / 262 / 262 | 256 |
| VSCALE_ACC 64×64 | 326 | 326 | 256 |

### 1.4 Front end, queues and tokens (`front_end.sv`, `queue.sv`, `token_net.sv`)

Measured on the `loop` micro-program (static nested loops, `VCOUNT → LOOP r`
with R = 6, loop-variant `row0` on LOAD/STORE) and on FA's `FE DISP` lines:

| | planning | measured |
|---|---|---|
| dispatch, plain word | 1 cycle; PUSH / POP free | **3 cycles per word** (fetch, decode, dispatch), **PUSH and POP words too** (`fe_word_cycles = fe_sync_word_cycles = 3`) |
| extension word | — | **2 + 1 per entry + 1 per `_mod` entry** (`fe_ext_word_cycles`, `fe_ext_entry_cycles`, `fe_mod_entry_cycles`): FA's `MMA.T {b_mod}` 7, `LOAD.2D {region.base, dst_mod}` 8, `LOAD.2D {row0}` 6 |
| LOOP entry / ENDLOOP | `loop_overhead_cycles` 2 each | **2 each**, loop exit 2 (unchanged) |
| LOOP r after its VCOUNT | 4 | **1 + ⌈R/step⌉** (`vcount_to_loop_latency = 1`, `loop_r_count_cycles = 1`; the count is formed by repeated subtraction): 7 for R = 6 |
| instruction queue depth | unbounded | **16** (`iq_depth`): the front end stalls on a full queue and everything behind it waits (FA: VCVT at pc 42 dispatched at cycle 7320 behind 16 queued VEC ops) |
| PUSH → POP visibility | 0 | **1 cycle** (`push_visible_cycles`): the PUSH retires in the producer's DONE cycle, the consumer's POP and its ISSUE are in the next |
| PUSH / POP issue slot (§8.5) | free | confirmed free at the queue: `9615 MMA SYNC POP 5` and `9615 MMA ISSUE MMA 6` in the same cycle; trailing PUSHes retire in the DONE cycle |
| next instruction of a queue | may issue the cycle after the previous one *started* | the cycle after the previous one *completed* (one instruction in flight per queue) |

### 1.5 Scratchpad ports (`scratchpad.sv`), as built

| | planning | as built |
|---|---|---|
| banks | slot → bank (`banks` = INP 2, WGT 4, ACC 4, VEC 2, STATE 1), 1 reader + 1 writer per bank | 16 column banks per scratchpad shared by all slots; **two ports per bank, at most one of them a write** (`ports_per_bank = (2, 1)`, `banks` = 1 per scratchpad); 7 clients, fixed priority MMA A, MMA B (always granted), MMA writeback, VEC read, VEC write, LOAD, STORE |

This is what let the `mma` program's STORE of ACC(0) run concurrently with
an MMA reading ACC(0) as its A operand (3154 cycles instead of 3140: 14 lost
to arbitration); with the planning ports the model serialized the two.

### 1.6 Not changed

`f_mhz` 200 (the plan's clock, still a target: the J7 Yosys estimate gives
cell counts, not timing, and the Vivado run is pending an x86 machine),
`mac_rows/cols` 16, `vec_lanes` 16, `sfu_lanes` 4, `fifo_depth` 8, one read
and one write channel, `request_channels` 1, `operand_broadcast` False (one
array), COMM link values (not built).

## 2. What still differs (after calibration)

- **Row reductions with fewer than 4 slices per row** (N ≤ 48): the shared
  reducer (K0.2) then stalls the last slice of every row until it is free
  (up to 3 cycles per row); not in the model (no program measures it; FA's
  and the micro-programs' reductions have 4 or 8 slices). K0.2's first rule
  blocked while the reducer was busy and cost 3 cycles per row at 4 slices
  (`VROWMAX 64×64` 456): fixed in the K0.2 follow-up.
- **Per-instruction overheads ±1 cycle**: VFILL 5 (model 6), VROWMAX/VROWSUM
  7 (model 6), VCOUNT 6 (model 7); the FINISH detection adds 3 cycles after
  the last DONE that the model does not count (every total is 3–5 cycles
  under the RTL, except FA).
- **Port arbitration is cycle-granular in the RTL, instruction-granular in
  the model**: two instructions on the same scratchpad with a third reader or
  a second writer interleave in the RTL and serialize in the model (not
  exercised by GEMM/FA beyond the 14 cycles above).
- **Instruction-queue occupancy** is counted when a head is processed, one
  cycle coarser than the RTL's issue.
- **Formulas from the FSMs, not from a run**: `LOAD.IDX`, `STORE.IDX`
  (per-row cost plus the index-slice reads), `VTOPK` (k passes), `VGATE`,
  `VMASK`, `VSEL`, the dequant formats (same slice rate, more beats). Phase 8
  (MoE/MLA/KDA on RTL) measures them.
- **Memory latency** is the harness's 24 cycles; a real LPDDR port changes
  the DMA numbers of §1.1 proportionally (24 of a row's 38–50 cycles).
- **`mma_latency` (planning) is unused under `rtl_rules`**; the MMA cost is
  the replay of §1.2.

## 3. How to reproduce

```
make -C rtl/test all                      # the nine programs (neg_untokened with TOKEN_CHECK=1), writes rtl_trace.txt
python rtl/test/calibrate.py              # per-instruction tables and the totals above
python -m pytest tests/test_cycle.py -q   # planning rules on KV260_PLAN / CONFIG_C, the measured durations on KV260
```

`model/sweeps.py` keeps running on `KV260_PLAN` so the phase-5 tables
reproduce; the calibrated descriptor is `tcta.timing.KV260`.

## 4. What the measurements say about the plan

1. **The v0 DMA is 3–5× slower than planned** on tile loads (5.1 and 3.4
   B/cycle vs 16) because it is row-serial. GEMM on the KV260 is 11291 cycles
   of which 6710 are the two loads; FA's nine K/V/Q loads are 21933 of its
   38543 cycles on the LOAD queue. This is the single largest gap and the
   first RTL item for phase 8 (§1.1: a one-row prefetch roughly halves it,
   row pipelining removes it).
2. **Column-vector VEC ops cost M cycles, not ⌈M/16⌉**: a consequence of
   the §11 column banking (a column vector lives in one bank), not of the
   VEC pipe. FA's per-KV-tile VEC chain is 4581 RTL cycles against 4156
   planned, the vector ops (VMAX, VADD, VSUB, VMUL on `[64, 1]`, VEXP2 and
   VRCP on `[64, 1]`) accounting for ~1500 of it. The fused VEXP2.SUM (1927
   vs 3072 planned) hides most of that in FA's total.
3. **The SFU is 6 cycles per slice, not 4**: one slice in flight through
   the 4-lane pipe. Pipelining slices (the lanes are idle 2 of 6 cycles) is
   a −33 % on every SFU op; VEXP2.SUM is 42 % of FA's VEC time on RTL.
4. **The front end is 3 cycles per word and queues PUSH/POP words**; with
   16-deep instruction queues it is never on GEMM's or FA's critical path
   (FA's front end finishes dispatching at cycle 36278 of 38543 only because
   the VEC queue was full), but a 1-word-per-cycle fetch is a cheap 3×.
5. **The accumulate writeback binds at K ≤ 64** (FA PV +14 %): 4 cycles per
   row. A read-modify-write that pipelines the old-value read with the
   previous row's write would make it 2 and remove the bound.

Everything else the plan assumed about the core (block period, bank-parallel
operand fetch, free syncs at the queue, token FIFO behaviour, the E1
numerics) is what the RTL does.
