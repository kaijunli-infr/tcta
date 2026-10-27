# Task: phase 5 — the cycle model

Prerequisites: Gate 1 closed (`docs/gate1-report.md`). ISA/simulator are
**v0.1.3** (sync per §0). This phase builds the timing layer the plan calls
the cycle model and uses it to answer the open items in the Gate 1 report §4.
Same environment; TileLang is no longer on the critical path, the inputs are
the five emitted programs.

## 0. Sync (half a day)

Unpack `isa/`, `sim/`, `tests/test_sim.py`. v0.1.3 responds to the Gate 1
findings:

| Finding | Change |
|---|---|
| Only LOAD writes WGT/STATE → 13 DRAM round trips per KDA chunk | `VCVT dst=WGT/INP` (ACC→operand move); `VCVT dst=STATE/ACC, row0` (f32 move at a row offset) |
| Additive `MMA.MASKED` wrong for the delta rule | `mask_mode="mul"`; the mask may live in ACC (full `[T_M, T_N]`) |
| 128-row state does not fit the 64-row write path | `MMA ... row0=64` into STATE; the tile's valid rows grow to the union |
| `VNEG` unusable | fixed |
| `VCOPY` missing | added (VEC→VEC, ACC→ACC) |

Then re-emit KDA with the round trips replaced by `VCVT dst=WGT`, the
multiplicative masks by `MMA.MASKED mul`, and the two-tile state + mirror by
one STATE tile with `row0` writes. Expect the 13 STORE/LOAD pairs per chunk
and the 64 KB state reload to disappear; checkpoints must still match. Update
the Gate 1 report's KDA row. `pytest tests -q` green before §1.

## 1. What the cycle model is

A timing layer **on top of the functional simulator**, not a second
simulator: the same `Program` runs, the same token checks apply, and each
executed instruction is given a start and end cycle from a device model.
Correctness stays the functional simulator's job; the cycle model only
answers "how long" and "who waited for whom".

Architecture: a discrete-event scheduler over the five queues. Each queue
is in-order; an instruction issues when (a) it is at the head of its queue,
(b) its POPs are satisfied (token available, with the FIFO level tracked
over time), (c) its execution unit is free, and (d) its scratchpad bank ports
are free. PUSH is free and takes effect at the producing instruction's
completion. Latencies and throughputs come from a `DeviceTiming` descriptor
(§2); nothing is hard-coded.

Outputs per run: total cycles; per queue busy / stalled-on-token /
stalled-on-unit / stalled-on-bank cycles; per edge the max FIFO level and
the cycles any PUSH was blocked by a full FIFO; per DMA channel utilization;
a Gantt-style trace (cycle, queue, pc, start, end) that `overlap_report.py`
can consume in place of the step trace.

## 2. `DeviceTiming` (the parameters; all sweepable)

| Group | Parameters | v1 KV260 defaults (planning values) | Config C defaults |
|---|---|---|---|
| Clock | `f_mhz` | 200 | 1000 |
| Array | `mac_rows × mac_cols`, `mma_issue_cycles(M,K,N)` = ⌈M/rows⌉·⌈N/cols⌉·K + pipeline latency | 16×16, latency 8 | 4 arrays of 64×64, latency 12 |
| VEC | `vec_lanes`, per-op cycles = ⌈M·N / lanes⌉ (row ops: ⌈M/lanes⌉), `vcvt`, `vgate` same rate | 16 | 64 |
| SFU | `sfu_lanes`, `exp2/rcp/log2/sigmoid` cycles = ⌈M·N / lanes⌉; **fusion flag** `sfu_fused_sub_sum` (true: `VEXP2.SUM t, vec` costs one pass; false: three) | 4, false | 8, true |
| DMA | `read_channels`, `write_channels`, `bytes_per_cycle_per_channel`, `dram_latency_cycles`, `row_gather_cost` = {`row`: one request per row, `burst64`: one per 64 B} , `max_outstanding_rows` | 1/1, 16 B, 100, row, 16 | 8/2, 32 B, 300, burst64, 256 |
| Dequant | `dequant_bytes_per_cycle` per channel for `f4.b32`/`f8.*`/`f4.r16` | 16 | 32 |
| Scratchpads | per scratchpad: `banks`, `ports`, bank-conflict rule (slot → bank = slot mod banks; a bank serves one reader + one writer per cycle) | INP 2/WGT 4/ACC 4/VEC 2 | 8/16/8/4 |
| FIFOs | `depth` per edge (one number or a dict) | 4 | 16 |
| Front end | `loop_overhead_cycles`, `vcount_to_loop_latency` | 2, 4 | 2, 4 |
| COMM | `link_bytes_per_cycle`, `link_latency_cycles` | emulated | 32 B, 2000 |

Put the two descriptors in `sim/tcta/timing.py` as dataclasses with these
exact names so sweeps can `replace()` them. Document every default as a
planning value.

## 3. Deliverables

| # | Deliverable | Definition of done |
|---|---|---|
| H1 | `sim/tcta/cycle.py` | `CycleMachine(Machine)` or a wrapper: runs a `Program`, returns the functional result plus a `Timing` report (§1 outputs). Token checks unchanged. |
| H2 | `tests/test_cycle.py` | Sanity: (a) a single `MMA` costs exactly `mma_issue_cycles`; (b) two independent MMAs on different banks pipeline, on the same bank serialize; (c) a LOAD→MMA token dependency gives start(MMA) ≥ end(LOAD); (d) FIFO depth 1 on LOAD→MMA serializes the FA ring, depth ≥ 2 does not; (e) a `LOAD.IDX` of 64 rows under `row` gather costs 64 requests, under `burst64` costs ⌈row_bytes/64⌉·64 |
| H3 | `model/sweeps.py` + `docs/cycle-model-report.md` | The Gate 1 §4 questions answered with numbers, each as a one-parameter sweep on the relevant family under **config C**: INP slots 2→6 (MLA Q re-streaming); WGT ring 2→4 (MoE boundary overlap, in cycles now); FIFO depth 4/8/16 (all families: which stall); `row` vs `burst64` gather (MoE tile time, KDA chunk time); `vec_lanes` 16→64 and `sfu_fused_sub_sum` (MLA and FA tile time vs MMA time: the VEC-bound question from the plan review); `dram_latency` 100→600 (KDA before/after the v0.1.3 round-trip removal, MoE count round trip) |
| H4 | `model/module0.py` hook | `eta_bw` and `eta_mfu` for the MoE decode tile and the MLA tile read from H3's results instead of the 0.7/0.5 placeholders; rerun the headline config and record the change in `model/README.md` |
| H5 | Phase 5 section in the verdict + the updated Gate 1 report §4 | One paragraph per question with the number and the recommended v1/config-C value; and the explicit list of parameters that must be calibrated against Verilator later (array latency, DMA bytes/cycle, dram_latency) versus those the model fixes by construction |

## 4. Modelling rules (so the numbers mean the same thing everywhere)

- One cycle = one clock at `f_mhz`; report cycles and also µs at the clock.
- An MMA occupies the array for its issue cycles; its result is visible
  (PUSH fires) at issue + latency. Back-to-back MMAs into the same ACC with
  `accumulate` have no extra dependency cost (accumulator forwarding).
- A LOAD occupies one read channel for ⌈bytes / bytes_per_cycle⌉ cycles
  after `dram_latency`; multiple outstanding LOADs on one channel pipeline
  their latency (latency is paid once per idle→busy transition), which is
  what rings buy. Dequant is a per-channel rate limit, not extra latency.
- `LOAD.IDX`: `row` cost = rows × (latency amortized as above + row_bytes /
  bytes_per_cycle); `burst64` cost = same with requests per 64 B, limited by
  `max_outstanding_rows`. `-1` rows cost nothing.
- STORE mirrors LOAD on write channels. A STORE→LOAD round trip through DRAM
  pays write completion + `dram_latency` + read.
- VEC/SFU ops are single-unit, in order, cost by element count; `VCVT`
  into INP/WGT also takes a write port on that scratchpad.
- Scratchpad port conflicts stall the later instruction by one cycle per
  conflict; report them separately so bank counts can be sized.
- The front end dispatches one instruction per cycle; `LOOP r` waits for
  `VCOUNT` + `vcount_to_loop_latency`.

## 5. Decision rule

PASS: H1–H5 delivered and every sweep in H3 reproduces the functional result
(cycle model must never change numerics). There is no FAIL condition on the
numbers themselves; a surprising number is a finding. The only FAIL is a
cycle model whose dependencies disagree with the functional simulator's
token checks (an instruction that starts before a token it popped was
pushed), which H2(c) guards.

## 6. Not in scope

Power, area, RTL, Verilator, multi-chip timing beyond the COMM link
parameters, changes to the ISA. Findings go to the main session.
