# Overlap report (F3): what happens at an expert boundary of the persistent MoE program

Program: `compiler/prototype/moe_kernel.py::build_moe_persistent_kernel()` (F2),
all 32 experts of `fixtures/moe_grouped` in one tile program, f4.b32 weights,
weight ring depth 3 (`num_stages=3`, the default). Cut at #19, mapped by
`tile_tir_to_program.py`, run on the **v0.1.2** simulator with the step trace
(`Machine.trace = []`, the one simulator change of phase 3, re-applied to the
v0.1.2 `machine.py`: a `_trace` helper, no semantic effect). Produced by

    python compiler/prototype/overlap_report.py --markdown --stages 3
    python compiler/prototype/overlap_report.py --stages 2|4 [--weights bf16] [--no-prefetch]

The script also runs the four combine programs (F1) on the same machine and
checks `y` against the fixture, so every table below comes from a run whose
output is known to be right.

This is the phase-4 §0 rewrite of the phase-3 report. Two things changed
underneath it: the program is now the nested form the task asked for
(`LOOP n=16 { count; LOOP r=0, step=64; count; LOOP r=1, step=64 }`, 527
instructions, per-level `incr`), and `PUSH`/`POP` no longer cost an issue
slot (ISA §8.5 free sync). The second change inverts the critical queue and
makes the phase-3 criterion ("the first gather of `e+1` issues before the last
MMA of `e`") the wrong question; the sections below say why and what replaces it.

## Method

The simulator issues at most one *work* instruction per queue per step; a
queue drains the `PUSH`/`POP` in front of its next work instruction in the
same step (`stats["sync"]` counts them, `stats["instr"]` the work). There is
no cycle model, so a step is an issue slot of the slowest queue, not a cycle.
The trace records `(step, queue, op, pc)` for every executed instruction.
Each pc is tagged with its innermost loop (`prog.tags`); in the nested
program the `n`-th execution of a body pc of `LOOP r=0` belongs to the `n`-th
token tile of experts 0, 2, 4, … (and `LOOP r=1` to the odd experts), so the
trace splits by expert without marker instructions. Per boundary `e → e+1`
the report takes

- **LOAD idle at the boundary**: steps between the last LOAD-queue work
  instruction of `e` and the first of `e+1` on which the LOAD queue issued
  no expert work. This is the criterion now: the LOAD queue is the critical
  queue (next section), so any step it loses at a boundary is a step the
  program loses.
- **MMA gap at the boundary**: steps from the last MMA of `e` to the first
  MMA of `e+1`, against the intra-expert cadence (median 1).
- **token slack**: the step at which the first gather of `e+1` had all its
  token dependencies satisfied (the latest push consumed by the pops the
  LOAD queue executed between `e`'s last load and that gather, hoisted pops
  before the `LOOP` included), subtracted from the last MMA of `e`.
  Positive: the gather was not waiting on `e`.
- the phase-3 columns, kept for comparison: the step of the last MMA of `e`,
  the step of the first `LOAD.IDX` of `e+1`, their difference ("overlap"),
  MMAs of `e` executed at or after that gather, LOAD instructions of `e+1`
  issued before `e`'s last MMA.

Empty experts are skipped (the fixture has none: counts 39–59, one 64-token
tile each; `test_zero_token_expert_in_persistent_program` blanks one expert
and checks the remaining 29 boundaries).

## The critical queue moved

Per queue over the run (ring 3, f4.b32, 1092 steps): LOAD busy on 1025 steps
(897 `LOAD.2D` + 128 `LOAD.IDX`), MMA 768, VEC 481, STORE 128. In phase 3
the MMA queue was busy on ~92 % of steps because its 2311 `PUSH`/`POP`
cost slots; with free sync it is busy on 70 %, and the LOAD queue, which has
32 work instructions per token tile against the MMA queue's 24, is busy on
94 %. The program is LOAD-issue-bound end to end: 1092 steps for 1025 LOAD
work instructions + 17 count loads, i.e. ~50 steps of prologue, boundaries
and drain in total.

Under this accounting the phase-3 criterion cannot hold and should not: the
first gather of `e+1` is the 33rd LOAD instruction after the first of `e`
and the last MMA of `e` is the 24th MMA, both queues issuing one per step,
so the gather trails the last MMA by construction (by exactly one step, see
the table). "Overlap" in the phase-3 sense measured how far the LOAD queue
ran *ahead* of an MMA queue that was slowed by its own token traffic; it was
the ring depth minus one because every extra ring slot let LOAD issue one
more fill before waiting for a release. With equal issue costs the ring
depth no longer buys anything at the boundary (ring 2/3/4: LOAD idle 1 at
all 31 boundaries, MMA gap 3/3/4), and the step totals order 2 < 3 < 4
because a deeper ring only adds prologue fills.

## Result

| weights | ring | prefetch | instrs | steps | LOAD idle at boundary | MMA gap at boundary | first gather of e+1 minus last MMA of e | token slack |
|---|---|---|---|---|---|---|---|---|
| f4.b32 | 2 | yes | 525 | 1091 | 1 | 3 | +2 | 5 |
| f4.b32 | **3** | yes | 527 | **1092** | **1** | **3** | **+1** | **5** |
| f4.b32 | 4 | yes | 529 | 1125 | 1 | 4 | 0 | 5 |
| bf16 | 3 | yes | 527 | 1060 | 1 | 2 | +1 | 5 |
| f4.b32 | 3 | no | 274 | 1153 | 3 | 5 | +3 | 5 |
| f4.b32 | 2 | no | 272 | 1121 | 2 | 4 | +3 | 5 |
| bf16 | 3 | no | 274 | 1121 | 3 | 4 | +3 | 5 |

Every row is the same at all 31 boundaries (min = max in every column), as
it must be for 32 identical one-tile experts. Numerics are identical for every
configuration (`Yp` vs `y_parts` rowmax error 3.092e-3, end-to-end `y`
1.519e-3 against `tol_rowmax` 7.8e-3; the `y` error dropped from phase 3's
2.149e-3 because `LOAD.IDX` now keeps f32 into ACC).

**The boundary costs the LOAD queue one step**, the `LOAD.2D` of the next
count's token list (work of no expert), and nothing else: the first gather of
`e+1` issues on the step after that load, its tokens were satisfied 5 steps
before `e`'s last MMA (the slack column), and the MMA queue restarts 3 steps
after its last MMA of `e` (gather, first fill, first MMA, each one issue slot
on a different queue: the `LOAD.IDX → LOAD.2D → MMA` chain, 2 for bf16 whose
fills carry no scales). Nothing on any queue waits for the previous expert
at a boundary. Under the §3 decision rule of the phase-3 task read on the
critical queue, this is the best a boundary can do short of removing the
count load.

### Per boundary, f4.b32, ring 3, prefetch

All 31 rows read `overlap −1, MMAs of e outstanding 0, e+1 LOAD instrs
before e's last MMA 0, LOAD idle 1, MMA gap 3, token slack 5`; the boundaries
are 34 steps apart (one tile = 32 LOAD instructions + the count load + one
step of ramp), 36 at the ring-3 prologue. The full table is the `--markdown`
output; first and last rows:

| boundary | last MMA of e | first LOAD.IDX of e+1 | overlap | MMAs of e outstanding | e+1 LOAD instrs before e's last MMA | LOAD idle | MMA gap | token slack |
|---|---|---|---|---|---|---|---|---|
| 0 → 1 | 36 | 37 | −1 | 0 | 0 | 1 | 3 | 5 |
| 1 → 2 | 70 | 71 | −1 | 0 | 0 | 1 | 3 | 5 |
| … | | | | | | | | |
| 29 → 30 | 1022 | 1023 | −1 | 0 | 0 | 1 | 3 | 5 |
| 30 → 31 | 1056 | 1057 | −1 | 0 | 0 | 1 | 3 | 5 |

## What the §0 re-emission found: a 2–3 step LOAD bubble, and its cause

The first nested program did **not** show this. It idled the LOAD queue 2–3
steps at every boundary (ring 3: 1138 steps instead of 1092; 1 idle step at
even boundaries, 3 at odd ones), with a negative token slack at the odd
boundaries: the first gather of `e+1` had waited for a token pushed *after*
`e`'s last MMA. The trace showed the chain: `LOAD.2D` of the next count's
token list → `VCOUNT` on VEC (one step later than possible, VEC was
finishing `e`'s last `VSCALE_ACC`) → a `LOAD POP(from=VEC)` that D4 had
placed in front of the inner `LOOP r` → the loop's first `LOAD.2D`. A
LOAD → VEC → LOAD round trip sat between two experts on the critical queue.

Why the pop was there is a D4 rule, and the rule is right: the two counts of
an outer iteration shared one VEC buffer (`idx_all`, and after the `VCOUNT`
collapse also the `mask` fragment), so the second count's load needed the
first `VCOUNT`'s release, pushed before `LOOP r=0`. A dynamic loop may run
zero times, so a dependency that crosses it must be satisfied in front of it
(a zero-trip `LOOP r` has to be transparent), and the hoisted pop then waits
for the `VCOUNT`. Giving each count register its own token-list and mask
buffers (`idx_a`/`idx_b`, `mask_a`/`mask_b`) moves each release to the load
two sections later, whose pop is satisfied long before; the hoisted pops in
front of the inner loops now consume tokens that were pushed by the previous
expert's early VEC work, and the bubble is the one count load. Same cause,
smaller effect, was already fixed on the way: the combine-weight column `w_s`
read by the four `VSCALE_ACC`s released VEC→LOAD only after the tile's last
MMA; copying it once into a VEC-owned vector (`w_c`) moves the release to the
start of the tile.

The lesson for the token model: **anonymous per-edge FIFOs plus
zero-trip-transparent loops force every release that is pushed before a
`LOOP` and consumed after it to be popped at the loop entry**, and a pop at
the loop entry serialises the entering queue on the pushing queue's progress.
Whether a dependency crosses a loop is decided by buffer reuse, which the
kernel author controls (two buffers instead of one), so this is an authoring
rule rather than an ISA problem; it is recorded in the expressivity notes
and in `PINNED.md`.

## Is the count prefetch worth its program size?

The task expected ~270 instructions for `LOOP n=E { count; LOOP r }`. The
emitted program has 527 because the prefetch of expert `e+1`'s count into the
other of two count registers needs two token tiles per outer iteration
(`LOOP n=E/2 { count → R1; LOOP r=0; count → R0; LOOP r=1 }`). The table has
both forms (`prefetch=False` is the single-register form, 274 instructions,
`LOOP n=32 { count; LOOP r=0 }`): without the prefetch the `LOOP r` waits on
`reg_pending` for its own `VCOUNT`, the LOAD queue idles 3 steps per boundary
instead of 1, and the run takes 1153 steps instead of 1092 (5.6 %). The
prefetch stays the default; the single-register program is kept as the
reference point the task asked for (`dumps/program_moe_persistent_noprefetch.txt`,
`test_persistent_without_prefetch`).

## What produces the boundary behaviour

Nothing in the instruction stream is pipelined across experts: the program is
the TileLang nested loop, transcribed. Three things make the boundary cheap:

1. **Rings with release tokens.** The token-tile buffer `x_s[kk % 2]` (2 INP
   slots), the weight ring `W_s[· % 3]` (3 WGT slots) and the output tile
   `out[no % 2]` (2 ACC slots) are rewritten only after the consumer of the
   previous occupant has pushed its release (MMA→LOAD after each GEMM,
   STORE→MMA after each `STORE.IDX`). A load into a slot pops exactly the
   release of that slot's previous occupant.
2. **Level-invariant token placement (D4 phase B).** Pops are placed where
   the FIFO level says the matching push is reachable, so a `LOOP` body ends
   without draining anything (`body_end_pops = 0`); what is popped in front
   of a `LOOP` (2 hoisted pops, 10 primes in the whole program) is consumed
   from pushes the previous expert made early.
3. **Count prefetch into the other register, with its own buffers.** The
   `LOOP r` header never waits on VEC and no pop in front of it waits on the
   `VCOUNT`.

The two phase-3 kernel changes about the tail of an expert (double-buffered
output tile, one flattened W2 stream) are still needed: both removed
four-queue round trips per N-tile that would otherwise sit on the LOAD queue
as well.

## On the step totals

The totals are issue-slot counts of the critical queue: 1092 steps for 1025
LOAD work instructions (ring 3, f4.b32). They are a proxy for issue
bandwidth, **not cycles**: a `LOAD.2D` of a 128×128 f4 tile with scales, a
`LOAD.IDX` gather of 64 rows and a 64×128×128 GEMM each take one step here
and will not take the same time on hardware. What the proxy does say is which
queue the instruction mix saturates (LOAD, 32 instructions per tile against
MMA's 24: the 4 gathers, 24 fills, 4 scale loads of the f4 path per tile),
and that the dependency structure costs nothing at a boundary. How much of
the LOAD issue stream is hidden behind the GEMMs is the cycle model's
question (gather efficiency and fill bandwidth are on its open-items list in
the gate-1 report). Static FIFO levels never exceed the depth 4 except
VEC→MMA (level 6: the SwiGLU pushes four P-ready/out-rewritten tokens before
the first is popped); depth is a device parameter (config C: 16) and the
simulator stalls the producer, the run is clean (`ok`, FIFOs empty at FINISH).
