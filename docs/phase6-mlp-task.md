# Task: phase 6 — memory-level parallelism: rings, cores, and the real η_bw

Prerequisites: phase 5 (`docs/cycle-model-report.md`). ISA **v0.1.4** changes
only §11 (device descriptor); no simulator semantics change. The repo in the
phase-5 tarball is canonical; the main session now sends spec and doc deltas
only.

## 0. Why

Phase 5's headline, η_bw 0.146 for one in-order program on config C, is a
property of a 3-deep ring, not of the machine: Little's law at 300 cycles ×
256 B/cycle asks for ~77 KB in flight, and 3 × 8 KB tiles hold 24 KB. The
ISA already has the two levers (ring slots are a descriptor number; cores
are independent streams sharing channels) and §11 now states both with
numbers. This phase measures them and replaces the η values that Module 0
consumes. The decisions it informs are the ones Module 0 is most sensitive
to: chips per replica and $/Mtoken.

## 1. Decisions taken (apply, do not re-derive)

| Item | Value | From |
|---|---|---|
| FIFO depth | 8 on every edge, both descriptors | §4.3 |
| INP slots, config C | 5 | §4.1 |
| `request_channels` default | 2 | §4.7 (knee; latency-bound beyond) |
| A-operand sharing | descriptor flag `operand_broadcast`, default **true** for config C (the four 64×64 arrays of one core read one A tile) | §4.8 |
| WGT slots, config C | 32 (Little's law, §11.1) | §3, §4.7 |
| MLA K-split reduction | not now; compiler item, recorded | §4.8 |

Also fix in the descriptor: `max_outstanding_rows` must be ≥ the striped
gather's demand, or the gather sweep is capped by the window, not the
channels; report which bound was hit.

## 2. Deliverables

| # | Deliverable | Definition of done |
|---|---|---|
| I1 | Ring-depth sweep | Under config C with the §1 values, MoE token tile and MLA KV tile versus WGT ring depth 3, 6, 12, 24 (the compiler sets the ring from the slot budget; `num_stages` in the kernel or the D4 slot allocator, whichever is simpler). Report cycles per tile, bytes in flight (mean, p90), channel utilization, η_bw, η_mfu. Expect η_bw to rise toward the striping limit and then flatten where the ring exceeds BW × latency. |
| I2 | Multi-core timing | `CycleMachine` runs `c` programs concurrently (one per core: own queues, scratchpads, FIFOs) against the shared channels through an arbiter (round-robin per request; record the policy). Test: `c` copies of the GEMM program on `c` cores take ~1× the time of one when channels are idle, and approach `c×` when a single program already saturates the channels. |
| I3 | Core sweep | MoE persistent program on 1, 2, 4 cores, each core streaming a disjoint expert subset (E/c experts), ring depth from I1's knee. Report aggregate channel utilization, η_bw and η_mfu per chip, and the arbiter's fairness (per-core wait). Same for MLA with 4 independent heads per core. |
| I4 | Striping × cores | The 2-D table (request_channels 1/2 × cores 1/2/4) for MoE; identify whether striping still pays once cores supply the parallelism. |
| I5 | Module 0 update | `cycle_results.json` gains keys for the recommended configuration (ring at knee, 4 cores, striping 2, broadcast on); `module0.py` consumes them by default; rerun the headline and the node comparison; update `model/README.md` and the Module 0 doc's η rows. State N chips per replica and $/Mtoken before and after. |
| I6 | Descriptor decision + Phase 6 section in the verdict | Recommend config C as either "4 cores × 1 array" or "1 core × 4-array MMA unit" from I3 vs I1 (same silicon, different control); recommend WGT/INP slot counts; and list what moved η_bw from 0.15 to its final value, by lever, so the report reads as an argument and not a table. |

## 3. Modelling rules added for cores

- Cores are independent programs: separate `Machine` state, separate token
  FIFOs, no shared scratchpads. The only shared resources are the DMA
  channels (and the write channels for STORE). Functional results of each
  program are checked against the fixture as before.
- The arbiter serves requests in arrival order across cores, one request
  per channel per cycle; a request striped over `k` channels occupies `k`
  channel slots. Record per-core issue→grant wait as a stall class
  (`stall_arbiter`).
- Per-core MMA units, VEC, SFU: one set each; `operand_broadcast` applies
  within a core only.
- No inter-core tokens. If a program needs another core's result, it is two
  programs with a DRAM hand-off; do not add an edge.

## 4. Decision rule

PASS: I1–I6 delivered, every run reproduces its fixture, and I6 states a
descriptor with numbers. The number itself is not graded. A FAIL is a timing
dependency violation (the H2(c) check) or a multi-core run whose per-core
functional result differs from the single-core run.

## 5. Not in scope

COMM between cores or chips, power, Verilator, ISA changes beyond §11,
kernel changes other than ring depth and the expert subset per core.
