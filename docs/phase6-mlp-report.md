# Phase 6 report: memory-level parallelism (rings, cores, striping) and the η Module 0 uses

Task: `docs/phase6-mlp-task.md`. ISA v0.1.4 (§11 only: the config-C device
descriptor). Code: `sim/tcta/timing.py` (`CONFIG_C`, `CONFIG_C_CORE`,
`CONFIG_C_PHASE5`; `request_channels`, `operand_broadcast`, `cores`),
`sim/tcta/cycle.py` (`run_multicore`, `MultiTiming`, the arbiter,
`stall_arbiter`), `sim/tcta/isa.py` (`DEVICE_CONFIG_C`: INP 5 / WGT 32 / ACC
16 / VEC 64 / STATE 2, FIFO 8), `compiler/prototype/moe_kernel.py`
(`flat_w13`, `expert_base` / `n_experts`), `compiler/prototype/mla_kernel.py`
(`kv_unroll`), `model/sweeps_mlp.py` (I1–I5; tables in `model/mlp_results.md`,
keys `mlp_*`, `module0_recommended`, `module0_1core` in
`model/cycle_results.json`), `model/module0.py` (consumes
`module0_recommended` by default), `tests/test_cycle.py` (+4 tests: operand
broadcast, idle-channel copies, saturated channels, multi-core == single-core).
Every row below is a `CycleMachine` / `run_multicore` run whose functional
result (every core's) was checked against the fixture; no sweep changed
numerics. Cycles at 1 GHz. η_mfu is always charged to the whole chip (four
64×64 arrays), so a 1-core and a 4-core run are comparable.

## 0. The result in one paragraph

Phase 5's η_bw 0.146 was a 3-deep ring on a descriptor without striping or
operand broadcast, run as one program. Applying the §1 decisions to the same
program gives 0.198; letting the compiler stream `[W1|W3]` as one 16-tile ring
instead of two 8-tile rings gives 0.230; deepening that ring to 12 gives
0.304, where it flattens (ring 16: 0.292) because the token tile is then bound
by its own serial chains (VEC/SFU 3331 cycles and 6144 array-cycles of
accumulation per tile), not by bytes in flight. The second lever is cores: the
same silicon as 4 cores × one 64×64 array, each streaming 8 of the 32 experts,
gives η_bw **0.644** and η_mfu **0.502** (channels 52 % busy); 2 cores give
0.410. Striping no longer pays once cores supply the parallelism (±0.4 % at 4
cores; 8-way striping costs 27 %). Module 0 with the recommended configuration
(4 cores × 1 array, WGT ring 12, striping 2, broadcast, FIFO 8) sizes a
replica at **58 chips and $0.096 per Mtoken** (B = 256) against 255 chips and
$0.249 with the phase-5 values and 53 / $0.093 with the planning placeholders.
The descriptor decision is **4 cores × 1 array**: on MLA, whose KV tile is a
serial softmax chain, four independent streams are 2.9× a 4-array unit fed by
one stream (758 vs 2177 cycles per KV tile chip-wide), and on MoE 2.1×.

## 1. §1 decisions as applied

| Item | Applied as |
|---|---|
| FIFO 8 | `DeviceConfig.fifo_depth = 8`, `DeviceTiming.fifo_depth = 8` on both descriptors; the ring sweep also runs at 32 to show where 8 binds (§2) |
| INP 5 | `DEVICE_CONFIG_C.slots[INP] = 5`; MLA `inp_slots=5` (Q resident) in every MLA row |
| `request_channels` 2 | `CONFIG_C.request_channels = 2` (least-busy two channels per tile request; gather rows round-robin over two) |
| `operand_broadcast` true | `CONFIG_C.operand_broadcast = True`: an MMA's A-operand port hold is keyed by the A slot, so the arrays of one core reading one A tile share the port (`test_operand_broadcast_lets_two_arrays_read_one_a_tile`) |
| WGT 32 | `DEVICE_CONFIG_C.slots[WGT] = 32`, one bank per slot (`banks[WGT] = 32`) |
| MLA K-split | not done; recorded in §7 |
| `max_outstanding_rows` | 512 ≥ a striped gather's demand (one burst per cycle × (300 latency + 2 transfer) ≈ 302 in flight). Which bound each gather hit is counted (`Timing.gather_bound`): at 1 core the window never binds; at 4 cores 16 % of bursts hit the window and 80 % a busy channel (§4) |

## 2. I1: ring depth (one stream, `CONFIG_C`, four arrays as one MMA unit)

The compiler sets the ring from `num_stages` (`T.Pipelined`). A ring cannot be
deeper than the trip count of the loop it pipelines, and a ring restarts at
every outer iteration (ISA §15), so the phase-3 MoE form, which streams W1 and
W3 as two 8-tile loops per token tile, caps at ring 8. The ring sweep
therefore uses `build_moe_persistent_kernel(flat_w13=True)`: x resident in two
INP slots (one per K chunk), `[W1|W3]` streamed as one 16-tile loop, which
allows rings up to 16 within a token tile (24 is unreachable: the W2 stream is
8 tiles and the ring restarts at the W2 loop). MLA's K ring spans `kv_unroll`
KV tiles (5 slots each): 4 (`u = 1`, the phase-5 form), 10, 20.

| family | program form | WGT ring | FIFO | cycles per tile | η_bw | η_mfu | channel util | bytes in flight mean / p90 (KB) | MMA→LOAD dynamic level | PUSH blocked (cycles) |
|---|---|---|---|---|---|---|---|---|---|---|
| MoE | phase-3 form, phase-5 descriptor | 3 | 16 | 13554 | 0.146 | 0.113 | 0.117 | 35 / 70 | 3 | 0 |
| MoE | phase-3 form, §1 decisions | 3 | 8 | 9974 | 0.198 | 0.154 | 0.159 | 30 / 70 | 3 | 0 |
| MoE | flat `[W1\|W3]` | 3 | 8 | 8561 | 0.230 | 0.179 | 0.186 | 36 / 181 | 3 | 0 |
| MoE | flat `[W1\|W3]` | 6 | 8 | 7116 | 0.277 | 0.216 | 0.223 | 43 / 184 | 6 | 0 |
| MoE | flat `[W1\|W3]` | **12** | 8 | **6481** | **0.304** | **0.237** | 0.245 | 50 / 210 | 8 (held) | 2061 |
| MoE | flat `[W1\|W3]` | 16 | 8 | 6750 | 0.292 | 0.228 | 0.235 | 48 / 206 | 8 (held) | 40065 |
| MoE | flat `[W1\|W3]` | 12 | 32 | 6481 | 0.304 | 0.237 | 0.245 | 50 / 210 | 8 | 0 |
| MoE | flat `[W1\|W3]` | 16 | 32 | 6746 | 0.292 | 0.228 | 0.236 | 48 / 206 | 16 | 0 |
| MLA | INP 5, phase-5 descriptor | 4 | 16 | 3665 | 0.092 | 0.074 | 0.083 | 16 / 56 | 8 | 0 |
| MLA | INP 5, §1 decisions | 4 | 8 | 2772 | 0.122 | 0.098 | 0.110 | 15 / 56 | 8 | 0 |
| MLA | K ring over 2 KV tiles | **10** | 8 | **2177** | **0.155** | **0.125** | 0.141 | 21 / 72 | 8 (held) | 79 |
| MLA | K ring over 4 KV tiles | 20 | 8 | functional TOKEN_TIMEOUT: static MMA→LOAD level 36 | | | | | | |
| MLA | K ring over 2 KV tiles | 10 | 32 | 2177 | 0.155 | 0.125 | 0.141 | 21 / 72 | 18 | 0 |
| MLA | K ring over 4 KV tiles | 20 | 32 | 2177 | 0.155 | 0.125 | 0.141 | 25 / 72 | 32 | 14 |

Reading it:

- **The knee is ring 12 for MoE and ring 10 for MLA**, and it is not the
  Little's-law knee. §11.1 predicts ≥ 12 slots to hold 77 KB in flight; the
  ring-12 tile holds 50 KB on average (p90 210 KB during the ring refill) and
  the channels are 25 % busy. What flattens the curve is the tile's own
  serial work: per token tile the VEC/SFU pipe is busy 3331 cycles (VSIGMOID
  2048 of them at 8 SFU lanes), the MMAs are 6144 array-cycles of
  accumulation chains, the stores 1536 write-channel cycles; at 6481 cycles
  per tile the LOAD queue still waits 5804 cycles per tile for release tokens
  (`stall_token`), i.e. for the consumers. One in-order stream cannot push
  η_bw past ~0.30 on this tile because it cannot consume faster, not because
  it cannot fetch.
- **Ring 16 is slightly worse than 12** (+4 %): with the ring equal to the
  trip count there is no slot reuse within a token tile, the whole W13 stream
  is issued at the tile start and the next tile's loads wait for the first
  releases; the convoy costs more than the extra bytes in flight return.
- **FIFO 8 is enough at the knee.** Ring 12 holds the MMA→LOAD FIFO at its
  depth for 2061 cycles over the run (0 cost: 6481 at both depths); ring 16
  holds it 40065 cycles (+4 cycles per tile). MLA's K ring over 4 KV tiles
  needs 36 release tokens in flight on MMA→LOAD (two per K slot: scores and
  PV) and times out in the *functional* simulator at FIFO 8; at FIFO 32 it
  runs and gains nothing over ring 10 (2177 both). Rule for the compiler: a
  ring of depth d with r release tokens per slot needs FIFO ≥ r·(d − 1) on
  that edge, else the FIFO sets the bytes in flight. The §1 decision (8)
  stands for the recommended rings.
- **MLA is chain-bound harder than MoE**: 2177 cycles per KV tile against
  1088 array-cycles of MMA and 1300 of VEC; the scores → softmax → PV chain
  is serial per KV tile and the K ring only removes the load latency from it
  (−21 % from ring 4 to 10).

## 3. I2: the multi-core model and its sanity tests

`run_multicore(timing, programs)` replays c programs on c `_Core`s (own
queues, dispatch, scratchpads, token FIFOs, arrays, VEC pipe) against shared
read and write channel vectors. Policy: requests reach the channels in
arrival order (all cores' heads are replayed in nondecreasing ready-time
order), requests ready in the same cycle rotate across cores, a striped
request takes k channel slots, a gather's bursts queue individually. A
request's wait is max(0, channel busy − (issue + latency)), reported per core
as `stall_arbiter` (LOAD.2D / STORE.2D) and `stall_arbiter_gather`; it
includes queueing behind the core's own requests, so a single-core run also
reports it. No inter-core tokens; `operand_broadcast` applies within a core.

| program | cores | chip cycles | vs 1 core | channel util | per-core cycles | per-core arbiter wait |
|---|---|---|---|---|---|---|
| GEMM (2 LOAD, 1 MMA, 1 STORE), 8 write channels | 1 / 2 / 4 | 1413 / 1413 / 1569 | 1.00 / 1.00 / 1.11× | 0.08 / 0.16 / 0.29 | equal | 0 / 0 / 156 |
| weight stream, 16 × 32 KB LOAD.2D, no tokens | 1 / 2 / 4 | 2351 / 4397 / 8492 | 1.00 / 1.87 / 3.61× | 0.87 / 0.93 / 0.97 | equal | 12192 / 28560 / 61320 |

Copies of an idle-channel program cost ~1× (the task's test; with the
config-C write channels (2) the GEMM's 32 KB f32 stores already contend at 2
copies, hence 8 write channels in that row); copies of a program that
saturates the channels approach c× with equal per-core waits. The fourth test
checks every core's functional result against the single-core run (the task's
FAIL condition) and the phase-5 H2(c) check (no popped token before its
push) still runs on every family.

## 4. I3: cores (`CONFIG_C_CORE`: one 64×64 array per core, channels shared)

MoE: core i streams experts `8i .. 8i+7` (`expert_base`, `n_experts`), the
token tiles whose top-k hit that subset are checked against the fixture.
MLA: c independent copies (one head group per core). WGT ring 12 (MoE) / K
ring 10 (MLA), striping 2, broadcast, FIFO 8.

| run | chip cycles | per tile (chip) | η_bw | η_mfu (chip) | channel util | bytes in flight mean / p90 (KB) | per-core cycles | per-core arbiter wait, tile requests | gather wait per burst |
|---|---|---|---|---|---|---|---|---|---|
| MoE 1 core × 1 array, 32 experts | 298058 | 9314 | 0.212 | 0.165 | 0.171 | 34 / 185 | 298058 | 72430 | 0 |
| MoE 2 cores × 16 experts | 153995 | 4812 | 0.410 | 0.319 | 0.330 | 85 / 236 | 153995, 151208 | 97197, 45136 | 287, 119 |
| **MoE 4 cores × 8 experts** | **98002** | **3063** | **0.644** | **0.502** | 0.519 | 239 / 676 | 98002, 97330, 96970, 96658 | 170604, 217690, 209290, 219207 | 943, 850, 900, 831 |
| MoE 1 core × 4-array unit (I1 knee) | 207386 | 6481 | 0.304 | 0.237 | 0.245 | 50 / 210 | 207386 | 109201 | 0 |
| MLA 1 core × 1 array | 40972 | 2561 | 0.132 | 0.106 | 0.119 | 18 / 56 | 40972 | 4364 | – |
| MLA 2 cores | 43402 | 1356 | 0.249 | 0.201 | 0.226 | 38 / 112 | 41354, 43402 | 12092, 14140 | – |
| **MLA 4 cores** | **48522** | **758** | **0.446** | **0.359** | 0.404 | 103 / 256 | 42378, 44426, 46474, 48522 | 43189, 45237, 47285, 49333 | – |
| MLA 1 core × 4-array unit (I1 knee) | 34831 | 2177 | 0.155 | 0.125 | 0.141 | 21 / 72 | 34831 | 6860 | – |

Reading it:

- **Cores scale the MoE tile almost linearly**: 9314 → 4812 → 3063 cycles per
  tile chip-wide (1.94×, 3.04×), because each core's stream is bound by its
  own chains and the channels have room (52 % at 4 cores). Bytes in flight
  rise from 34 to 239 KB mean (p90 676 KB), well past the 77 KB Little's-law
  figure, which is why the fourth core still pays: the mean hides that the
  four rings refill in bursts.
- **At 4 cores the chip is arbitration-bound, not bandwidth-bound**: a tile
  request waits ~850 cycles on average for a channel (≈ 217 K cycles over
  ~250 requests per core) and a gather burst ~900 cycles, while the channels
  are 52 % busy. Four 12-deep rings issue 48 requests of 128 channel-cycles
  (8 KB f4 tiles, striped 2) within a few cycles of each other on 4 channel
  pairs; the queue is 12 deep and the last request waits ~1500 cycles. The
  gather window (512 bursts) binds on 16 % of bursts, a busy channel on 80 %
  (`gather_bound`); at 1 core neither binds. The headroom from 0.64 to the
  channel limit is a scheduling question (staggered rings, a wider
  interleave), not more slots.
- **Fairness.** MoE: per-core completion within 1.4 % (96658–98002) with
  arbiter waits within 1.28× of each other (core 0 streams experts 0–7, whose
  token counts differ); arrival order is fair enough for unequal programs.
  MLA: four *identical* lock-step copies finish in a staircase, 42378 /
  44426 / 46474 / 48522, exactly 2048 cycles (128 per KV tile) apart per
  core index. The core granted first in the first round gets its token first,
  issues its next KV tile first and arrives first for the rest of the run; a
  rotating tie-break (implemented) does not change it because after the first
  round there are no ties. Arrival order has a convoy mode on identical
  programs; the chip time is the last core's (+18 % over one core alone).
  A per-core round-robin grant, or staggered starts, would remove it;
  recorded for the main session (§7).
- **The 4-array single core is the wrong shape for these tiles.** MoE: 6481
  cycles per tile with four arrays vs 9314 with one; the three extra arrays
  buy 30 % because the 24 MMAs per tile form accumulation chains and the SFU
  pipe is serial. MLA: 2177 vs 2561, 15 %: the softmax chain uses one array
  at a time. Four cores give 3063 and 758 instead: 2.1× and 2.9× the
  4-array unit on the same silicon.

## 5. I4: striping × cores (MoE, ring 12, FIFO 8, broadcast)

| `request_channels` | 1 core | 2 cores | 4 cores |
|---|---|---|---|
| 1 | 9873 / tile, η_bw 0.200 | 5095, 0.387 | 3051, **0.647** |
| 2 | 9314 (−5.7 %), 0.212 | 4812 (−5.6 %), 0.410 | 3063 (+0.4 %), 0.644 |
| 8 | 9403 (−4.8 %), 0.210 | 5631 (+10.5 %), 0.350 | 3862 (+26.6 %), 0.511 |

Striping pays while one or two streams cannot fill the channels (−5.7 %, −5.6
%) and stops paying when four do (+0.4 %, noise); 8-way striping hurts as soon
as there is a second core because every request then occupies every channel
and the arbiter serializes the cores. Decision: keep `request_channels = 2`
(the §1 default; it costs nothing at 4 cores and helps every single-stream
program on the chip), and never 8.

## 6. I5: Module 0

`model/module0.py` reads `module0_recommended` by default (`--cycle-key`
selects `module0_1core`, `module0` (phase 5), `module0_striped`). C @ 22FDX,
8 × LPDDR5X, B = 256, 30 ms TPOT:

| η source | η_bw | η_mfu | N chips / replica | $/Mtoken at B = 256 | at B* = 751 | MFU |
|---|---|---|---|---|---|---|
| planning placeholders | 0.70 | 0.50 | 53 | 0.093 | 0.038 | – |
| phase 5: one program, ring 3, task's DMA rule | 0.146 | 0.113 | 255 | 0.249 | 0.093 | 4 % |
| phase 5 what-if: striped over 8 channels | 0.181 | 0.141 | 204 | 0.210 | 0.079 | 5 % |
| phase 6: 1 core × 4-array unit, ring 12, striping 2, broadcast | 0.304 | 0.237 | 122 | 0.146 | 0.055 | 8 % |
| **phase 6: 4 cores × 1 array, ring 12, striping 2, broadcast (recommended)** | **0.644** | **0.502** | **58** | **0.096** | **0.036** | 17 % |

Node comparison (`--sweep`, sweep 5), before → after: C @ 22FDX 8 ch N 255 →
**58**; C @ 12LP+ with 2 × HBM2e N 76 → **21** (MFU 9 → 41 %); the
H100-class reference stays at 4. The batch sweep moves accordingly (B = 1024:
N 347 → 79, $0.082 → $0.030); the η_bw sensitivity sweep is unchanged by
construction. The MLA tile's values (η_bw 0.446 / η_mfu 0.359 at 4 cores) are
printed alongside; Module 0 still keys on the MoE tile.

The recommended configuration lands within 10 % of the planning placeholders
(58 vs 53 chips, $0.096 vs $0.093), with the remaining gap being η_bw 0.64 vs
0.70. A chip with these four cores is bandwidth-bound at B = 256 (bw 57.4 vs
flop 19.6 chips), so η_bw is still the number to calibrate first.

## 7. I6: descriptor decision and the η_bw argument

**Config C = 4 cores × one 64×64 array**, sharing 8 read + 2 write channels
through an arrival-order arbiter, `request_channels` 2, `operand_broadcast`
within a core (moot with one array; keep the flag for a core that grows a
second array), FIFO 8 on every edge. Per core: **INP 5** (MLA's Q resident;
MoE's flat form needs 2 x slots + the gather destinations), **WGT 32** (MoE
at the knee uses 12 for `[W1|W3]` + 8 for W2 = 20; MLA 10; 24 would do, 32
is the §11 headroom and costs 256 KB), ACC 8 + 2 wide, VEC 64, STATE 2 as in
§11. The choice over "1 core × 4-array MMA unit" is 2.1× (MoE) and 2.9× (MLA)
on the same silicon, because these decode tiles are serial chains (softmax,
accumulation, SFU) that one in-order stream cannot spread over four arrays,
while four streams hide each other's chains against the channels.

What moved η_bw from 0.146 to 0.644 (MoE decode tile, config C), by lever:

| step | lever | η_bw | Δ | why |
|---|---|---|---|---|
| 0 | phase 5: one program, ring 3, one request per channel, one reader per bank | 0.146 | | 24 KB in flight against 77 KB needed; 89792 bank-stall cycles |
| 1 | §1 descriptor: striping 2, operand broadcast, FIFO 8, WGT 32 banks (same program) | 0.198 | +0.052 | a tile moves at 64 B/cycle; A-operand reads no longer serialize (bank stalls 26544) |
| 2 | one `[W1\|W3]` ring instead of two 8-tile rings that restart per matrix | 0.230 | +0.032 | the ring no longer re-primes at the W3 boundary |
| 3 | ring 3 → 12 (Little's law) | 0.304 | +0.074 | 50 KB in flight; the stream is now bound by its consumers (VEC 3331, 6144 array-cycles per tile), ring 16 goes back to 0.292 |
| 4 | 4 cores × 1 array, 8 experts each, instead of 1 core × 4 arrays | **0.644** | +0.340 | four chains overlap; channels 52 % busy, requests queue ~850 cycles in bursts |
| – | striping at 4 cores | 0.647 (k = 1) / 0.644 (k = 2) / 0.511 (k = 8) | 0 | parallelism now comes from cores |

Two thirds of the gain is cores, one quarter the ring, the rest the
descriptor rules phase 5 flagged. The ring lever flattens where the task
expected it to (where the ring exceeds BW × latency) but for a different
reason: the stream runs out of independent work before it runs out of
slots.

Findings for the main session (nothing in `sim/` semantics changed; `isa/`
unchanged beyond the supplied §11):

1. **Descriptor: 4 cores × 1 array** (above), INP 5, WGT 32 (≥ 24), FIFO 8,
   striping 2, never 8.
2. **Arrival-order arbitration has a convoy mode** on identical lock-step
   programs (MLA copies: +4.8 % per core index, chip +18 % over one core
   alone); a rotating tie-break does not remove it. Consider a round-robin
   grant across cores with pending requests, or stagger program starts.
3. **At 4 cores the chip is request-queue-bound before it is
   bandwidth-bound** (52 % channel utilization, ~850-cycle waits): the four
   rings refill in bursts. The window of 512 outstanding gather bursts binds
   on 16 % of bursts at 4 cores; 256 (§11's number) would bind more. Keep
   512.
4. **FIFO ≥ r·(d − 1) for a ring of depth d with r release tokens per slot.**
   MLA's K ring over 4 KV tiles needs 36 on MMA→LOAD and times out
   functionally at 8; it also gains nothing over ring 10. The compiler should
   check the static MMA→LOAD level against the FIFO when it picks a ring.
5. **Rings are per loop**: a ring deeper than the pipelined loop's trip count
   issues everything up front and restarts at the next outer iteration (ring
   16 = trip count is slower than 12). To reach a Little's-law depth the
   compiler must flatten the streams it pipelines (`flat_w13`), which is a
   program-shape decision, not a slot count.
6. **MLA K-split reduction** (two arrays on two K halves with a reduction),
   recorded not done: with 4 cores × 1 array the per-core question is moot;
   it would matter again for a core with more than one array.
7. **The MoE tile at the knee is SFU-bound**: VSIGMOID is 2048 of the 3331
   VEC cycles per tile at 8 SFU lanes; 16 lanes would take ~1000 cycles off
   a 6481-cycle single-stream tile. Worth a row in the calibration list
   (`sfu_lanes`).

## 8. Reproduce

```
python model/sweeps_mlp.py                 # ~100 s: I1–I5 -> model/mlp_results.md, keys in model/cycle_results.json
python model/module0.py                    # headline on module0_recommended
python model/module0.py --cycle-key module0 --sweep      # the phase-5 numbers, node comparison
python compiler/prototype/dump_programs.py # + program_moe_flat_ring12.txt, program_mla_kv2.txt
python -m pytest tests -q                  # 84 passed
```

`python model/sweeps.py` still reproduces the phase-5 tables: it runs on
`CONFIG_C_PHASE5` and merges into `cycle_results.json` without touching the
phase-6 keys.
