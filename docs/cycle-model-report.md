# Cycle model report (phase 5, H3/H5)

Task: `docs/phase5-cycle-model-task.md`. Code: `sim/tcta/timing.py`
(`DeviceTiming`, the `KV260` and `CONFIG_C` descriptors), `sim/tcta/cycle.py`
(`CycleMachine`), `tests/test_cycle.py` (H2), `model/sweeps.py` (H3; writes
`model/cycle_results.json` and the tables in `model/cycle_results.md`),
`model/module0.py` (H4 hook). ISA/simulator v0.1.3; the inputs are the five
programs emitted by `compiler/prototype/dump_programs.py` on v0.1.3 plus two
sweep variants kept beside them (`program_kda_roundtrip.txt`, the v0.1.2 KDA
form; `program_mla_inp6.txt`, Q resident). Every number below comes from a
`CycleMachine` run whose functional result was checked against the fixture
(`model/sweeps.py::run` asserts it), so no sweep changed numerics (task §5).

Every timing parameter is a planning value (§6 says which must be calibrated).
Cycles are at the descriptor's clock: KV260 200 MHz, config C 1 GHz.

## 1. The model as implemented

A post-mortem replay. `CycleMachine.run` is `Machine.run` with a recording
front end: it keeps the dispatch order (including which `LOOP r` took effect
and which `VCOUNT` wrote its register) and every executed instruction with the
operand shapes the functional simulator saw (`valid` rows/cols, bytes, live
gather rows). The timing pass then walks the per-queue streams with a
discrete-event loop that always advances the earliest-ready head (front end,
then LOAD, MMA, VEC, STORE, COMM). Rules, in the order they are applied to a
work instruction:

| Step | Rule |
|---|---|
| dispatch | the front end issues one instruction per cycle; PUSH/POP are free (ISA §8.5); `LOOP`/`ENDLOOP` cost `loop_overhead_cycles`; `LOOP r` waits until the `VCOUNT` that wrote R[r] has completed plus `vcount_to_loop_latency` |
| in order | an instruction issues no earlier than its queue's previous instruction issued (+1) |
| tokens | each POP ahead of it in the stream is consumed when the queue reaches it and the producer's PUSH has fired; the PUSH fires at the producer's completion (LOAD: last byte landed; MMA: result visible at occupancy + `mma_latency`; VEC: last element) |
| FIFO | a PUSH into a full FIFO (`fifo_depth` tokens not yet popped) is accepted when the consumer pops; only then does it hold its queue. A PUSH never otherwise delays its queue: the LOAD queue issues the next LOAD while the previous one is in flight (a scoreboard fires the token) |
| unit | MMA: the earliest-free of `n_arrays` arrays, occupied for ⌈M/rows⌉·⌈N/cols⌉·K cycles; VEC/SFU: one in-order pipe, ⌈M·N/lanes⌉ cycles per op (vectors ⌈len/lanes⌉; SFU ops at `sfu_lanes`; `VEXP2.SUM` one pass fused, three unfused; `VEXP2` with a row subtrahend one or two; `VTOPK` two); LOAD/STORE: the DMA engine, see below |
| ports | slot → bank = slot mod `banks[sp]`; a bank has `ports_per_bank` = (readers, writers); an instruction holds its ports for its streaming duration (MMA: the array occupancy, so a back-to-back accumulate into the same ACC forwards per task §4; VEC: its cycles; LOAD/STORE: the transfer; a gather holds its index vector's read port only while it issues requests). A later instruction waits for a free port; the wait is charged to the bank(s) that imposed it |
| DMA | one request per LOAD.2D/STORE.2D; `LOAD.IDX`/`STORE.IDX` issue one request per live row (`row`) or per 64-byte burst (`burst64`), one per cycle, at most `max_outstanding_rows` in flight; `-1` rows cost nothing. A request goes to the least-busy channel of its direction and starts transferring at max(channel free, issue + `dram_latency_cycles`), so requests behind a busy channel overlap their latency with the transfer in front; it occupies the channel ⌈bytes / bytes_per_cycle⌉ cycles (dequant formats at min(`bytes_per_cycle`, `dequant_bytes_per_cycle`)) |
| COMM | ⌈bytes / link_bytes_per_cycle⌉ + `link_latency_cycles` (never exercised by the five families) |

Reported per run (`Timing`): cycles and µs; per queue busy (unit cycles),
`stall_token` (cycles the head waited for a token with its unit free),
`wait_unit` (back-to-back serialization on its own unit), `stall_bank`,
`stall_fifo`; per edge pushes, pops, the maximum number of tokens waiting in
the FIFO at any cycle (dynamic level) and the cycles a PUSH was blocked; per
channel busy cycles; per bank the conflict cycles charged to it; the Gantt
trace (`Timing.trace()`), and `step_trace()` in the `Machine.trace` shape so
`overlap_report.py`'s boundary analysis runs on cycles (the WGT-ring sweep uses
it). `eta_mfu` = MACs / (peak MACs per cycle × cycles); `eta_bw` = (DRAM bytes
read + written) / (read-channel peak × cycles).

Two what-if parameters sit outside the task's rule set and default to the
task's rules: `request_channels` (a LOAD.2D/STORE.2D striped over k channels,
gather rows round-robin over k channels; 1 = one request on one channel) and
`ports_per_bank` readers > 1. They exist because the sweeps showed the rules
themselves set the config-C numbers (§4.7, §4.8).

Approximations to keep in mind when reading the numbers:

- Queues are replayed in ready-time order, not global time order. A bank-port
  hold that starts later than the cycle being considered does not block; the
  later-starting instruction is never delayed in return. Port conflicts are
  therefore a lower bound.
- Dispatch is modelled as free-running one per cycle with no instruction
  buffer limit: a queue can hold arbitrarily many dispatched instructions.
  The "dispatch → start" gap is reported, not limited.
- The MMA's input operands are read for the whole occupancy; the array is
  assumed to stream A and B from the scratchpads (no operand buffer).
- A gather's issue rate is one request per cycle regardless of channel count.
- `CycleDeadlock`: when the functional run succeeded (at its own FIFO depth)
  but the modelled FIFO depth leaves no head able to proceed, the timing pass
  raises instead of returning a number (KDA at depth 4, §4.3).

## 2. Sanity tests (H2, `tests/test_cycle.py`, 16 tests)

(a) one MMA costs exactly `mma_issue_cycles` on both descriptors; (b) two
independent MMAs on two arrays start one cycle apart on different ACC banks and
serialize on the shared bank's write port (the wait is charged to that bank
only); (c) for every popped token of the MLA and KDA programs, the consumer
starts at or after the producer's completion, and pops never exceed pushes
per edge; (d) a producer that runs ahead is blocked at FIFO depth 1 and free
at depth ≥ its run-ahead, and the FA LOAD→MMA ring never holds more than one
token at any depth (the 2-deep ring bounds it, not the FIFO); (e) a 64-row
`LOAD.IDX` costs 64 requests under `row`, ⌈row_bytes/64⌉·64 under `burst64`,
and `-1` rows cost nothing; plus: the cycle machine leaves the same status and
DRAM as `Machine`, a modelled deadlock is reported, and the striping what-if
shortens the transfer without changing channel-cycles.

## 3. Baseline: the five families under both descriptors

| family | KV260 cycles (µs) | per unit | η_mfu | η_bw | config C cycles (µs) | per unit | η_mfu | η_bw |
|---|---|---|---|---|---|---|---|---|
| GEMM (partial tile) | 5633 (28.2) | program | 0.333 | 0.537 | 2013 (2.0) | program | 0.015 | 0.094 |
| FA (4 KV tiles, one head) | 27856 (139.3) | 6964 / KV tile | 0.294 | 0.184 | 4709 (4.7) | 1177 / KV tile | 0.027 | 0.068 |
| MLA decode (16 KV tiles, INP 2) | 412652 (2063) | 25791 / KV tile | 0.675 | 0.377 | 73653 (73.7) | 4603 / KV tile | 0.059 | 0.132 |
| KDA (4 chunks, config-C slots) | deadlock at FIFO 4 (§4.3); 321303 at depth 16 | 80326 / chunk | 0.056 | 0.256 | 60704 (60.7) | 15176 / chunk | 0.048 | 0.085 |
| MoE persistent (32 token tiles, ring 3) | 3325990 (16630) | 103937 / token tile | 0.946 | 0.304 | 433718 (433.7) | 13554 / token tile | 0.113 | 0.146 |

The two descriptors are bound by different things. The KV260 (one 16×16
array, one 16 B/cycle channel) is array-bound on MoE (η_mfu 0.95) and MLA
(0.68); FA and the GEMM are small enough that the DMA latency shows. Config C
(four 64×64 arrays, eight 32 B/cycle channels, 300-cycle latency) is bound by
the LOAD chain on every family: a 16 KB tile takes 300 + 512 cycles on its
channel and a 32 KB tile 300 + 1024, while the MMA that consumes it takes
140–268 cycles on one of four arrays. With 2- to 3-deep rings the LOAD queue's
head waits for a ring token almost all the time (MoE LOAD `stall_token`
430399 of 433718 cycles; MLA 66406 of 73653) and the number of transfers in
flight is 0 or 1 for most of the run (MoE: 0 for 44 % of cycles, ≥ 3 for 10
%). The arrays sit at 5–11 % and the channels at 13–15 % of peak. The
per-family sections say which parameter moves this.

Bank conflicts under the one-reader rule: MoE 89792 stall cycles charged,
32640 each on INP banks 0 and 1 (the token tile that all 24 MMAs of a tile
read: the four arrays serialize on the A operand's read port, §4.8), the rest
on ACC banks; MLA 6096 on ACC bank 0 (the five QKᵀ chunks accumulating into
one ACC, sequential by nature); KDA 5151 spread over ACC/VEC banks.

## 4. The Gate 1 §4 questions, one sweep each (config C unless stated)

### 4.1 INP slots 2→6: MLA Q re-streaming

| INP slots | Q chunks resident | cycles / KV tile | vs INP 2 | DRAM read per 64-row step | η_mfu |
|---|---|---|---|---|---|
| 2 | 0 (re-streamed) | 4603 | — | 2304 KB | 0.059 |
| 3 | 1 (rope) | 4100 | −10.9 % | 2184 KB | 0.066 |
| 4 | 2 | 4103 | −10.9 % | 1944 KB | 0.066 |
| 5 | 5 (all) | 3665 | −20.4 % | 1224 KB | 0.074 |
| 6 | 5 | 3665 | −20.4 % | 1224 KB | 0.074 |

The saving comes in two steps. The third slot keeps the rope chunk resident,
which removes the first LOAD of every tile from the head of the LOAD chain;
the fourth slot removes a load that was already overlapped and buys nothing in
cycles (only bytes). The fifth slot makes Q fully resident: the loop body
drops from 10 to 5 LOAD.2D (`dumps/program_mla_inp6.txt`), DRAM traffic halves
and the tile is 20 % shorter. **Recommended: INP 5 for config C** (decode
attention with a 64-row stacked Q of 4.5 tiles); a sixth slot is idle. On the
KV260 MLA is array-bound (η_mfu 0.68) and INP depth is not the lever.

### 4.2 WGT ring 2→4: MoE expert-boundary overlap in cycles

| WGT ring | cycles / token tile | vs ring 3 | MMA token-stall | boundary MMA gap median / max | e+1 LOADs issued before e's last MMA (median) |
|---|---|---|---|---|---|
| 2 | 14925 | +10.1 % | 440567 | 1235 / 1619 | 2 |
| 3 | 13554 | — | 349590 | 1235 / 1619 | 3 |
| 4 | 11586 | −14.5 % | 245860 | 840 / 840 | 4 |

The issue-slot analysis of phase 4 said ring 4 "adds prologue only"; in
cycles it is worth 14.5 %, because a weight tile load (f4.b32, 8 KB + scales:
300 + ~234 cycles, 534 mean) is twice as long as the MMA that consumes it
(268). Keeping the array fed needs ⌈534 / 268⌉ = 2 loads in flight plus the
tile being consumed plus one being pushed back, i.e. four slots. Ring 4 is
also the first depth at which the boundary gap is the same at every
boundary (840 / 840). **Recommended: WGT ring 4 under config C** (16 WGT
banks, no bank cost); ring 3 on the KV260, where the array is the bottleneck.

### 4.3 FIFO depth 4 / 8 / 16: which edges stall

| family | depth 4 | depth 8 | depth 16 | deepest dynamic level (edge) |
|---|---|---|---|---|
| GEMM | 2013 | 2013 | 2013 | 0 |
| FA | 4709 | 4709 | 4709 | 1 (LOAD→MMA, MMA→VEC) |
| MLA | 73653, MMA→LOAD PUSH blocked 4 cycles | 73653 | 73653 | 6 (MMA→LOAD), 4 (MMA→VEC) |
| KDA | **deadlock** | 60704 | 60704 | 4 (LOAD→VEC), 3 (STORE→MMA) |
| MoE | 433718 | 433718 | 433718 | 3 (MMA→LOAD) |

Depth 4 deadlocks KDA: the program's static level on several edges is 5
(the Gate 1 report's "FIFO max 8" counts program order), so a producer's fifth
PUSH blocks its queue while the consumer that would pop is itself behind a
token from that queue. The functional run at config C's depth 8 completes;
the timing pass at depth 4 finds no head that can proceed and reports it
(`CycleDeadlock`: LOAD waits on MMA, MMA on VEC, VEC's PUSH to LOAD is full).
Nothing else changes between 4, 8 and 16: the dynamic level (tokens waiting in
the FIFO at any cycle) never exceeds 6, and on most edges the consumer is
already waiting when the token arrives (level 0). **Recommended: depth 8 on
every edge, for both descriptors**; the v1 default of 4 is unsafe for KDA, 16
is never approached.

### 4.4 row vs burst64 gather cost: MoE tile, KDA chunk

| family | KV260 `row` | KV260 `burst64` | config C `row` | config C `burst64` |
|---|---|---|---|---|
| MoE | 103937 / tile | 104032 (+0.1 %) | 13554 / tile | 13554 (0 %) |
| KDA | 80326 / chunk (depth 16) | 84562 (+5.3 %) | 15097 / chunk | 15176 (+0.5 %) |

At config C the two cost models are the same: a 768-byte row is 24 cycles on
a 32 B/cycle channel whether it is one request or twelve 64-byte bursts, and
256 outstanding requests cover the latency either way. On the KV260 `burst64`
with 16 outstanding bursts keeps only 1 KB in flight, which costs KDA 5 %.
What decides the MoE tile is not the request granularity but where the
requests go: under the task's rule every request of one gather queues on one
channel, so a 64-row × 768-byte `LOAD.IDX` takes 1836 cycles (768 bursts × 2
cycles + 300) and the four gathers of a tile are 7344 of its 13554 cycles. If
the rows are spread over the channels (§4.7) a gather is 1181 cycles and is
bound by the one-request-per-cycle issue rate. **Recommended: `row` with
max_outstanding_rows ≥ 16 on the KV260; at config C the granularity is
immaterial and the gather needs address-interleaved channels and an issue
rate above one request per cycle to stop being the tile's long pole.**

### 4.5 vec_lanes 16→128 and sfu_fused_sub_sum: MLA and FA tile time vs MMA time

| family | lanes / SFU | unfused | fused | VEC share of tile (fused) | MMA busy per array per tile |
|---|---|---|---|---|---|
| MLA | 16 / 8 | 7865 | 6841 | 53 % | 272 |
| MLA | 32 / 8 | 6373 | 5349 | 39 % | 272 |
| MLA | 64 / 8 (config C) | 5627 | 4603 | 28 % | 272 |
| MLA | 128 / 16 | 4487 | 3975 | 16 % | 272 |
| FA | 16 / 8 | 3047 | 2023 | 85 % | 32 |
| FA | 32 / 8 | 2483 | 1459 | 77 % | 32 |
| FA | 64 / 8 (config C) | 2201 | 1177 | 70 % | 32 |
| FA | 128 / 16 | 1293 | 860 | 48 % | 32 |

The fused `VEXP2.SUM` is worth 18 % on the MLA tile and 47 % on the FA tile
at 64 lanes; lanes 16 → 64 are worth 41 % (MLA) and 42 % (FA) unfused. At
config C's 64 lanes + fusion the MLA tile spends 28 % of its time in VEC and
is then bound by the LOAD chain (§3), so more lanes buy 14 % at most. FA at
64-row tiles is VEC-bound at every lane count: its eight MMAs occupy an array
for 32 cycles per tile against 822 VEC cycles, and even 128 lanes leave VEC at
48 % of the tile. **Recommended: 64 lanes with the fused SFU for config C
(the plan's values); attention prefill needs 128-row tiles or 128 lanes to
make the 64×64 array matter.** On the KV260 (16 lanes, unfused) FA is 62 % VEC
and MLA 24 %.

### 4.6 dram_latency 100→600: KDA before/after the round-trip removal, MoE count round trip

| family | 100 | 200 | 300 | 400 | 600 | slope (cycles per unit per latency cycle) |
|---|---|---|---|---|---|---|
| KDA v0.1.3 (0 round trips) | 14083 | 14626 | 15176 | 15726 | 17332 | 6.5 / chunk |
| KDA v0.1.2 (13 round trips + state reload) | 21685 | 24110 | 26535 | 28960 | 34170 | 25.0 / chunk |
| MoE ring 3, count prefetch | 12538 | 13044 | 13554 | 14063 | 16041 | 7.0 / tile |
| MoE ring 3, no prefetch | 13413 | 13926 | 14498 | 15104 | 17275 | 7.7 / tile |

The v0.1.3 KDA chunk is 43 % shorter than the v0.1.2 form at the config-C
latency (15176 vs 26535) and four times less latency-sensitive: each DRAM
round trip (STORE completion + latency + LOAD) sat on the solve's critical
path. The count round trip: the prefetch form (`VCOUNT` for expert e+1
issued during expert e) is 7.0 % faster than the single-register form at
latency 300 (13554 vs 14498 per tile) and, by the probe that lets `LOOP r`
dispatch without waiting for `VCOUNT`, within 0.1 % of a front end that knew
the counts in advance; the single-register form is 6.6 % behind that bound. The
"FE wait on VCOUNT→LOOP" column of `cycle_results.md` (≈ the whole run) is the
front end idling at `LOOP r` while the queues drain, not a cost: the dispatch
is far ahead of the units in every MoE run (dispatch → start gaps of 10–30 k
cycles). **Recommended: keep the prefetch form; `vcount_to_loop_latency` is
not a lever at these latencies.**

### 4.7 What-if: a LOAD.2D striped over k channels (`request_channels`)

| family | 1 channel (task rule) | 2 | 4 | 8 | η_bw at 1 → 8 |
|---|---|---|---|---|---|
| MoE | 13554 / tile | 10838 (−20.0 %) | 10738 (−20.8 %) | 10873 (−19.8 %) | 0.146 → 0.181 |
| MLA | 4603 / tile | 3835 (−16.7 %) | 3515 (−23.6 %) | 3483 (−24.3 %) | 0.132 → 0.175 |
| KDA | 15176 / chunk | 12612 (−16.9 %) | 12612 | 12612 | 0.085 → 0.102 |
| FA | 1177 / tile | 1113 (−5.4 %) | 1113 | 1112 | 0.068 → 0.072 |

Striping a tile request over two channels halves its transfer and takes 17–20
% off three families; beyond two channels the chain is latency-bound (300
cycles of the 300 + 128 a 16 KB tile then costs) and the rings would have to
deepen for more. Even at 8 channels the single in-order LOAD stream keeps the
channels at 18 % of peak: the rule the task set (one request, one channel) is
not what limits η_bw; the number of outstanding tile requests is (Little's law
at 300 cycles and 256 B/cycle asks for ~77 KB in flight; a 3-deep ring of 8 KB
tiles holds 24 KB).

### 4.8 What-if: read ports per bank (`ports_per_bank`)

| family | 1 reader (task rule) | 2 | 4 | what the conflicts were |
|---|---|---|---|---|
| MoE | 13554 / tile, MMA bank-stall 81632 | 13298 (−1.9 %) | 13044 (−3.8 %), 16320 | INP banks 0/1: the token tile read by all 24 MMAs; the four arrays serialize on it |
| MLA | 4603 / tile, 6096 | 4348 (−5.5 %) | 4222 (−8.3 %), 0 | ACC bank 0: the five QKᵀ chunks accumulating into one ACC. This gain is not realizable: accumulation into one tile is sequential; the lever is a K-split with a reduction (compiler) |
| KDA | 15176 / chunk, 4068 | 14969 (−1.4 %) | 14969 | ACC banks (accumulations), VEC bank 0 |

A second read port (or an A-operand broadcast to the four arrays) is worth 2–4
% on MoE and nothing elsewhere; the config-C bank counts (INP 8, WGT 16, ACC
8, VEC 4) are not a bottleneck for these programs.

## 5. Module 0 hook (H4)

`model/module0.py` replaces the `eta_bw` = 0.7 / `eta_mfu` = 0.5 placeholders
of the config-C presets with the MoE decode tile's values from
`cycle_results.json` (key `module0`: the task's DMA rule; `--cycle-key
module0_striped` for §4.7's 8-channel what-if; `--placeholders` for the old
numbers). Headline config C @ 22FDX, 8 × LPDDR5X, B = 256, 30 ms:

| η source | η_bw | η_mfu | N chips / replica | $/Mtoken at B = 256 | at B* |
|---|---|---|---|---|---|
| placeholders | 0.70 | 0.50 | 53 | 0.093 | 0.038 |
| cycle model, task rule | 0.146 | 0.113 | 255 | 0.249 | 0.093 |
| cycle model, striped 8 channels | 0.181 | 0.141 | 204 | 0.210 | 0.079 |

The MLA tile's values (η_bw 0.132 / η_mfu 0.059; striped 0.175 / 0.078) are
printed alongside. Read §4.7 before using these: they are the efficiency of
one in-order program on a device whose channels it cannot keep busy. Whether a
config-C chip runs one such program or several against the same channels is a
device question the descriptor does not answer (§6, finding for the main
session).

## 6. Calibration: what Verilator must supply, what the model fixes

**Calibrate against RTL/Verilator** (every one of these is a planning value in
`timing.py`):

| parameter | model assumption | leverage seen in §4 |
|---|---|---|
| request → channel mapping | one request on one channel (`request_channels` = 1) | 17–24 % on three families (§4.7); first thing to settle |
| `dram_latency_cycles` | constant per request, no queueing in the controller | 6.5–7.7 cycles per unit per latency cycle (§4.6) |
| `bytes_per_cycle_per_channel`, `dequant_bytes_per_cycle` | flat rate once the data arrives | sets every tile load (512–1024 cycles at config C) |
| gather issue rate, `max_outstanding_rows` | one request per cycle, fixed window | bounds the striped gather (§4.4) |
| `mma_latency`, array streaming rate | occupancy ⌈M/rows⌉⌈N/cols⌉K at one K-step per cycle; A/B streamed from the scratchpads for the whole occupancy | small at config C (arrays 5–11 % busy); decisive on the KV260 |
| VEC/SFU per-op cycles | ⌈elements/lanes⌉ × passes (1/2/3 for the exp2 forms, 2 for VTOPK); `VCVT dst=WGT|INP` at the VEC rate | 41–47 % on the attention tiles (§4.5) |
| bank ports and hold duration | (1, 1) per bank, held for the streaming duration | 2–4 % (§4.8) |
| `vcount_to_loop_latency`, `loop_overhead_cycles`, dispatch rate | 4, 2, one per cycle, unbounded queue buffers | not visible (front end always ahead) |
| token (PUSH) timing | fires at the producer's completion with no sync latency | every chain in §3 |
| COMM link | priced, never exercised | — |

**Fixed by construction** (the ISA's semantics; the model would be wrong to
vary them): in-order issue per queue; a POP consumed when its queue reaches it;
a PUSH ordered per edge after its producer; FIFO depth as the number of
unpopped tokens, a full FIFO holding the producer's queue; one DMA request per
LOAD.2D/STORE.2D and per live gather row or burst, `-1` rows free; no
speculation past a `LOOP r` whose count is unknown; back-to-back accumulate
forwarding; the dispatch order and operand shapes of the functional run.

## 7. Findings (also in the verdict, Phase 5)

1. **Two rules outside the task's set decide the config-C numbers:** how a
   request maps to channels (§4.7) and whether the four arrays can share an
   operand (§4.8). The descriptor should carry both.
2. **A single in-order program keeps 8 channels at 13–18 % of peak** (§3,
   §4.7). Module 0's 0.7 placeholder is 4–5× the measured η_bw; N per
   replica goes from 53 to 204–255 chips unless a chip runs several
   programs against its channels or the rings deepen to ~10 outstanding
   tiles.
3. **FIFO depth 4 deadlocks KDA** (§4.3); 8 is enough for all five families.
4. **WGT ring 4, INP 5** are the config-C values for MoE and MLA decode
   (§4.1, §4.2), contrary to the issue-slot estimate for the ring.
5. **The MoE tile at config C is gather-bound** under the one-channel rule:
   four 1836-cycle gathers in a 13554-cycle tile (§4.4).
6. **FA at 64-row tiles is VEC-bound at any lane count** on config C; the
   64×64 array is 2.7 % busy (§4.5).
7. **The v0.1.3 round-trip removal is worth 43 % of the KDA chunk** at the
   config-C latency and cuts its latency slope fourfold (§4.6).
8. **The count prefetch is worth 7 %** on the MoE tile and matches an
   oracle front end (§4.6).
