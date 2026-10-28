# TileLang cut-off prototype: verdict (D6)

Task: `docs/tilelang-lowering-cutoff-task.md` + addendum. Versions:
`compiler/prototype/PINNED.md` (tilelang 0.1.15, bundled TVM 0.25.dev0).
Target config used for D3: **sm_80** (`{"kind": "cuda", "arch": "sm_80"}`,
fast-math on). sm_90 with `tl.disable_warp_specialized=True` gives identical
IR and is selectable with `arch="sm_90"`.

## Verdict: PASS

Against §6 as amended by the addendum:

- D3 (`compiler/prototype/cut.py`) is produced by TileLang's own passes.
  `CUDAPassPipelineBodyPrologue` runs unmodified; a `PassInstrument` captures
  the module after `tl.InjectSoftwarePipeline` (#19 on sm_80) and aborts the
  rest. **No pass before the cut is skipped.** The only additions are the four
  allowed rewrites: (a) scope rewrite, (c) shell strip, (d) event rewrite;
  (b) pass skipping was not needed.
- The body reads no `threadIdx.*`; `bx`/`by` are int32 PrimFunc params.
  Asserted by `tests/test_tilelang_cut.py` (no `launch_thread`, `threadIdx`,
  `blockIdx`, `ptx_`, `mbarrier`, `tma`, `cp_async`, `mma`; every buffer in a
  `tcta.*` or `global` scope; `#fill_issued == #copies into tcta.wgt`).
- D5 is green: `tile_tir_to_program` → `Machine.run` with token checking,
  all 8 heads × 4 query tiles, matches `fixtures/flash_attn_gqa`
  (regenerated with `causal=False`) with max rowmax error **5.05e-3**
  (2^-7.6) against `tol_rowmax` = 2^-7. No `UNTOKENED_*`/`TOKEN_*` fault.
  Whole repo: 22 passed, 1 skipped (pre-existing fixture-dir skip).

## The three load-bearing passes

- **`tl.LayoutInference` (#21): REJECT, cut before it.** It assigns
  lane-dependent MMA fragment layouts to `acc_s`/`acc_s_cast`/`acc_o`
  (element (i,j) → thread `i//16*32 + i%8*4 + j%8//2`, 32 values per thread
  over 128 threads) and 4-way replicated row fragments to the vectors; it is
  annotation-only (tile ops survive), target-independent between sm_80 and
  sm_90 apart from the thread range, and nothing in it is reusable for a
  tile ISA.
- **`tl.LowerTileOp` (#23): REJECT, cut before it.** `T.gemm` does not
  survive as a call: it becomes `ptx_ldmatrix` + `ptx_mma` per-thread loops
  (sm_80) or `wgmma` + TMA (sm_90), `T.copy` becomes `ptx_cp_async`,
  `T.reduce` becomes `tl::AllReduce` with named barriers; everything it does
  is what `tile_tir_to_program` does instead, one tile op → one ISA
  instruction.
- **The pipeline pass pair (#18 `PipelinePlanning`, #19
  `InjectSoftwarePipeline`): ACCEPT and CUT-HERE.** #18 only annotates the
  loop with a stage/order plan; #19 materialises exactly the schedule the
  ISA wants (K/V → `[2,64,64]` rings indexed `k % 2`, 2-fill prologue,
  steady-state loop of `n−2` with the refill of bank `k%2` placed after the
  gemm that read it, epilogue) and its only GPU-specific residue is
  `ptx_commit_group` / `ptx_wait_group(n)`, which map one-to-one onto the
  token insertion points (`fill_issued` → `PUSH LOAD→MMA`; `fill_wait(n)` →
  `c − n − p` POPs).

## What was skipped or replaced

- **Skipped before the cut:** nothing. (Passes #1–#19 all run; #14–#16, the
  warp-specialisation passes, are not on the sm_80 path at all, and are
  disabled by `tl.disable_warp_specialized` on sm_90.)
- **Replaced (everything after #19):** `LayoutInference`,
  `ReducerPlanAndMaterialize`, `LowerTileOp`, and the remaining CUDA
  pipeline (#24–#74: buffer flattening, vectorisation, storage rewrite,
  thread sync, shared-memory merging, host/device split, packed API, kernel
  launch) are replaced by `tile_tir_to_program.py`: Phase A tensorises each
  tile op to one ISA instruction with a fixed slot allocation, Phase B
  inserts tokens, Phase C emits `LOOP`/`ENDLOOP` with `incr` for the
  loop-variant DRAM base and ring slot.
- **Rewritten:** `ptx_commit_group`/`ptx_wait_group` → `tcta.fill_issued` /
  `tcta.fill_wait(n)` externs; the `no_implicit_async_commit_wait` copy
  annotation dropped; the six `thread_extent` AttrStmts removed and `bx`,
  `by` promoted to params; `tl.launch_threads` annotation removed; the two
  wrapper blocks' `reads`/`writes` cleared (they referenced the loop var `k`
  outside its loop; TileLang's own output).

## Scope rewrite order (addendum §D3 step 2)

Three orders were tested:

| Where the `tcta.*` scopes are set | Result |
|---|---|
| In the kernel source (`T.alloc_fragment(..., scope="tcta.acc")`) | **Rejected at trace time** by TileLang's Python builder: `tilelang/language/reduce_op.py:171` `ValueError: Invalid buffer scopes: tcta.acc and tcta.vec` (`T.reduce_max` accepts only shared/fragment pairs). |
| On the frontend PrimFunc, before pass #1 (`rewrite_scopes` then passes 1–19) | **Accepted by every pass, but #18/#19 then do nothing**: the output is the plain 4-iteration serial loop, `K_shared` stays `[64,64]`, no ring, no `ptx_*`. `PipelinePlanning`/`InjectSoftwarePipeline` select the buffers to multi-version by `shared`/`shared.dyn` scope, so foreign scopes silently disable pipelining. |
| After #19 (what `lower_to_tile_tir` does) | Works; the ring re-declaration `data=K_shared.data` is handled by giving each old data Var one new Var shared by all Buffers that aliased it. |

So the scope rewrite **had to go after #19**, and for a stronger reason than
the addendum anticipated: TileLang's pipeline pass is keyed on CUDA scope
names. A real `tcta` target would have to teach `PipelinePlanning` which
scopes are ring-able, or run scope inference after pipelining as done here.

## fill_wait counts

All eight `fill_wait` sites resolved to **exactly 1 pop, all ≥ 0**
(`MappingReport.fill_wait_pops`: body `(n=3, c=4..7, p=0..3)`, epilogue
`(n=3,2,1,0, c=8, p=4..7)`). The event model and the FIFO model agree; the
`c − n − p` rule is static at every program point as the addendum predicted,
including in the loop body (counts carried over as if the body executed
`extent` times).

## Finding during D5: pops must be ordered per edge like their pushes

The first D5 run faulted `UNTOKENED_WRITE` at the epilogue's second `MMA.T`.
Cause: tokens on an edge are a FIFO count, not tile-tagged. On MMA→VEC the
producer pushes "S ready (k=3)" after "O ready (k=2)", but the dependency
analysis had placed VEC's pop for "O ready (k=2)" (before `VSCALE_ACC` of
k=3) *after* its pop for "S ready (k=3)" (before `VROWMAX` of k=3); the S pop
consumed the O token, VEC ran ahead and read stale `ACC[0]`, and the checker
caught the overwrite. The hand-written schedule avoids this by popping
"O ready" at the end of every iteration. The fix in Phase B: dependency
tokens are collected as (push, pop) pairs and, per edge, any pop that would
overtake an earlier-pushed token's pop is moved up to just before it (moving
a pop earlier is always safe). One pop moved (`MappingReport.moved_pops=1`).
This ordering constraint is a rule the general token-insertion pass must own.
(Phase 2 rebuilt the pass around it: pops are now emitted cumulatively per edge,
see "What the token pass became" below; the FA token counts are unchanged.)

## Token counts vs the hand schedule (addendum §D4 Phase B assertion)

| Edge | hand (`test_flash_attention_inner_loop`, nk=4) | this prototype |
|---|---|---|
| LOAD→MMA | 9 / 9 | 9 / 9 |
| MMA→VEC | 8 / 8 | 8 / 8 |
| VEC→MMA | 8 / 8 | 8 / 8 |
| VEC→STORE | 1 / 1 | 1 / 1 |
| MMA→LOAD | 12 / 12 | **8 / 8** |

MMA→LOAD differs by design: the hand schedule pre-releases `2·num_stages`
bank tokens before the loop and drains them after; the software-pipelined
TIR's prologue fills go into never-read banks, so under the addendum's rule
("pop before every fill of a slot that has been read") they need no pop, the
4 epilogue gemms release banks nobody refills, and those 4 are drained before
`FINISH`. `test_token_counts_per_edge` pins both values.

## Assumptions that leak

1. **Scopes by name, not inferred** (`cut.SCOPE_MAP`). Inference is the real
   compiler's job; the §3 table was applied verbatim.
2. **TileLang fp16 ≙ ISA bf16.** Same byte size and strides; the simulator
   rounds to bf16 on LOAD. The output is stored as bf16 because `Output` is
   fp16 in the kernel; this costs about half of the 2^-7 budget (the hand
   schedule stores f32).
3. **Pattern pass, not a compiler.** `tile_tir_to_program` recognises
   exactly the six `T.parallel` body shapes, the fill-before-gemm
   (`accumulate=False`), fill-before-rowmax (dropped), and exp2+rowsum
   (`VEXP2.SUM`) idioms of this kernel; anything else raises.
4. **Fixed slot allocation** in order of appearance (INP[0]; WGT[0..1] K,
   WGT[2..3] V; ACC[0] S/P, ACC[1] O; VEC[0..4] + one temp VEC[5]).
   `acc_s_cast` aliases `acc_s` (in-place `VCVT`).
5. **`-inf` → `-1e30`**, and `exp2(x) = 0` for `x ≤ −127` in the simulator.
6. **One `(bx, by)` per `Machine.run`**; GQA handled by replicating K/V per
   query head in DRAM (`K_virtual[h] = K[h // 4]`), not by stacking.
7. **The ring index must be `k % n` with base slot 0** (the simulator's
   `_mod` increment adds `i % n` to the base slot). TileLang emits exactly
   `k % 2` / `(k + 2) % 2`.
8. **Cross-iteration dependencies get their pop at the end of the body**, and
   dependencies from before the loop hoist their pop before `LOOP`; both are
   the hand schedule's conventions, hard-coded in Phase B.
9. The cut PrimFunc still carries the CUDA `target` attr and the stale
   enclosing-block `reads`/`writes` mentioning `k`; D4 ignores both.
10. The frontend's `T.Kernel` shell exists in every cut; the shell strip is
    the ~30-line pass the amended §6 allows.

## ISA v0.2 candidates (recorded, not added)

- `VCOPY vec, vec`: the `T.copy(scores_max → scores_max_prev)` maps to
  `VFILL 0 + VADD` today (2 instructions per iteration).
- Fused scale-then-exp2 for the rescale vector: `scores_scale = exp2(s·(prev −
  max))` costs `VFILL, VADD, VSUB, VMUL, VEXP2` (5 instructions); the matrix
  path already has `VEXP2.SUM t, vec`.

## What the real compiler owns after this result

Scope inference (which TileLang buffer is INP/WGT/ACC/VEC), a target hook so
`PipelinePlanning` multi-versions `tcta.wgt` the way it does `shared.dyn`,
the event→token rewrite generalised beyond this FA schedule (including the
per-edge pop-ordering rule found in D5), and everything `LayoutInference` and
`LowerTileOp` do for the GPU, which is replaced by tensorise-to-ISA.

---

# Phase 2: the MoE decode kernel (E6)

Task: `docs/tilelang-phase2-moe-task.md`. Kernel `compiler/prototype/moe_kernel.py`
(one tile program per expert, plus a router program per token), expressivity
record `docs/tilelang-moe-expressivity.md` (E2), cut PrimFuncs
`compiler/prototype/dumps/cut_moe.py` (f4.b32, the normative weights since
phase 3) / `cut_moe_bf16.py` / `cut_router.py` (E3), programs
`dumps/program_moe_e0.txt` / `program_moe_bf16_e0.txt` /
`program_router_t0.txt` (E4), tests `tests/test_tilelang_moe.py` (E5).
The numbers in this section are the phase-2 ones; the phase-3 kernel
(`LOOP r, step`, `STORE.IDX`, rings for overlap) is measured in the Phase 3
section below.

## Verdict: PASS

Against §4 of the phase-2 task:

- **All three features have a TileLang authoring form; the only extern is
  `VTOPK`.** Gather: elementwise `if_then_else` over an index fragment (TileLang
  has no tile-level gather; `T.copy` in `T.Parallel` is rejected). Dynamic
  count: `T.serial(n[0])` with `n` computed as `reduce_sum(heads >= 0)`, which
  is `VCOUNT`'s definition. MXFP4: payload + scales copies and an explicit
  dequant `T.Parallel` loop in the producer stage of a manually staged
  `T.Pipelined`. None of these is an extern.
- **The cut is at #19** (`tl.InjectSoftwarePipeline`) for all three kernels, with
  the same four rewrites as FA (`cut.lower_to_tile_tir(..., scope_map=...)`); no
  pass before it is skipped.
- **E5 is green**, token checking on: 11 tests. Every expert (32 × bf16, 32 ×
  f4.b32) matches float64 math on the weights the device sees, max rowmax error
  **4.15e-3** (bf16, 2^-7.9) and **3.72e-3** (f4.b32) against `tol_rowmax` =
  7.8e-3; the bf16 result scatter-added in NumPy with the shared expert
  reproduces the fixture's `y` within tolerance. `VTOPK` reproduces `topk_idx`
  for all 256 tokens. A zero-token expert (`heads = -1`) runs zero iterations:
  no MMA, no gather, no DRAM write, FIFOs empty at FINISH. A synthetic expert
  with 109 tokens runs two `LOOP R0` iterations with the DRAM bases advancing
  per iteration. Whole repo: 33 passed, 1 skipped (pre-existing).

Caveats that do not change the verdict under §4, in order of weight:

1. **`ceil(count/64)` is not computable on the device.** `VCOUNT` counts rows,
   `LOOP` has no step, there is no ALU (§7). The kernel takes a host-prepared
   `heads[e, t]` row (first token of each tile, `-1` if empty) and counts it.
   The host already prepares `idx_lists`; this is one more strided view of it.
   ISA v0.2 candidate: `VCOUNT.TILES` (count ⌈n/T_M⌉) or `LOOP r, step`.
2. **Dequant-on-load changes the pipeline stage of the dequant.** TileLang's
   automatic planning puts the dequant with the gemm; after folding it into the
   load, that schedule refills the ring slot before the gemm reads it. The first
   f4 run was numerically wrong with a clean token trace. Manual
   `order/stage` puts the dequant with the copies and fixes it. A `tcta` target
   would need PipelinePlanning to treat "dequant of a just-copied tile" as
   producer work.
3. **`T.Pipelined` over a dynamic extent is accepted but predicated** (`if 0 <
   n`, `T.where`, `serial(n − 2)`): no ISA form. The token-tile loop is
   `T.serial`; only the static weight streams inside are pipelined, so the
   gather of the next token tile does not overlap the current tile's gemms.
4. **`VGATE acc, vec, mul` with an ACC operand** is the only elementwise
   ACC×ACC multiply (`VMUL` is a row broadcast); the ISA text says `vec ∈ VEC`,
   the simulator accepts ACC. The SwiGLU gate needs it 4× per token tile.
5. **The fixture has bf16 expert weights**; the MXFP4 image is produced by the
   test (power-of-two block scales, Appendix A) and checked against the
   simulator's own dequant rule. The f4 path is therefore tested against a
   derived reference, not against `y`.

### Caveats closed by the ISA v0.1.1 sync (phase 3, §0)

| # | Caveat / gap | Closed by | Where |
|---|---|---|---|
| 1 | `ceil(count/64)` not computable on the device; host `heads` row | `LOOP r, step=64` runs ⌈R/64⌉ iterations; the kernel writes `T.serial(T.ceildiv(n[0], 64))`, which survives #19 as `(n[0] + 63) // 64` and D4 maps to `LOOP R, step=64` | `moe_kernel.py`, `tile_tir_to_program._dynamic_extent` |
| 4 | `VGATE` with an ACC operand undocumented | ISA v0.1.1 admits `src ∈ VEC ∪ ACC` | `isa/isa-v0.1.md` |
| 5 | fixture had bf16 weights; f4 path checked against a derived reference | `moe_grouped` carries `W*_payload`/`W*_scales` as the normative values plus `y_parts`; the f4 **and** bf16 programs are checked against `y_parts` (both to 3.09e-3) and `y` end to end (2.15e-3) | `moe_data.expert_weights`, `test_every_expert_matches_y_parts` |
| gap | `STORE.IDX` not in the simulator; NumPy scatter-add | `STORE.IDX` implemented; the kernel scatters each output N-tile to `Yp[B·k, d]` at `dst_rows` and the F1 combine kernel reduces it on the device | `moe_kernel.py`, `tile_tir_to_program._parallel` (`scatter`) |
| gap | no `i32` store | `STORE.2D`/`STORE.IDX` take `i32`; the router stores `topk_idx` as int32 | `build_router_kernel`, `test_router_vtopk_matches_topk_idx` |
| gap | `asm.Program.vgate` uncallable | renamed keyword `gate_op`; D4 emits through `asm` again | `tile_tir_to_program._emit_op` |
| gap | tile state kept across `Machine.run` | `Machine.reset()`; the tests run 32 expert programs, or 4 combine programs, on one machine | `test_every_expert_matches_y_parts`, `overlap_report.run_combine` |

Caveats 2 (dequant stage must be placed manually) and 3 (`T.Pipelined` over
a dynamic extent is predicated) are TileLang properties and remain open; the
phase-3 kernel keeps the manual `order/stage` lists and keeps the token-tile
loop a plain `T.serial`.

## Simulator and ISA gaps exposed (not changed, per §5)

All four were fixed by the v0.1.1 sync (table above); the original record:

- `STORE.IDX` is in the ISA, not in the simulator (expected by the task); the
  combine is a per-expert `STORE.2D f32` plus a NumPy scatter-add.
- `STORE.2D` has no `i32` qualifier (§4): the router's int32 index row is
  stored as f32.
- `asm.Program.vgate(acc, vec, op)` cannot be called — its `op` keyword collides
  with `_emit(self, op, ...)`; D4 builds the `Instr` directly.
- `Machine` keeps tile reader/writer state across `run` calls, so a second
  program on the same machine faults `UNTOKENED_WRITE` on tiles the first one
  read. Correct for one kernel; a per-kernel reset is missing.

## Token graph (per expert program, one token tile)

| Edge | pushes / pops | from |
|---|---|---|
| LOAD→MMA | 28 / 28 | 4 gathers + 24 weight fills (`fill_issued`) |
| MMA→LOAD | 24 / 24 | bank release after every MMA (ring refills + INP refill) |
| MMA→VEC | 8 / 8 | h1/h3 ready ×4, out ready ×4 |
| VEC→MMA | 8 / 8 | P-ready ×2, out rewritten ×4, h overwritten next tile ×2 |
| VEC→STORE / STORE→MMA / STORE→VEC | 4 / 4 each | output tile per N-tile |
| LOAD→VEC | 2 / 2 | heads row (pre-loop), combine weights |
| VEC→LOAD | 1 / 1 | combine-weight column rewritten next tile |

241 instructions (86 LOAD, 96 MMA, 40 VEC, 16 STORE, 3 control), 136 (bf16)
/ 133 (f4) simulator steps per token tile; `fill_wait` resolved statically at
all 48 sites (bf16: 1 pop each; f4 manual stages: `wait(0)` after each fill
→ 1–2, `wait(1)` before the gemm → 0); body token-balanced on every edge,
nothing hoisted before `LOOP`, nothing drained after it — the property that
makes `LOOP R0 = 0` legal.

## What the token pass became

The D5 finding (pops must follow push order per edge) is now the rule the pass
is built on: every consumer site states which push it needs and pops are
emitted cumulatively per edge, so the order is right by construction; a
dependency on the previous iteration pops at the end of the iteration that
pushed (the hand schedule's convention), a dependency on a pre-loop write pops
before `LOOP`, excess body pushes are popped at body end so the body is
balanced for any trip count, and the `fill_wait(n)` events are requirements
like any other (`fill_wait_pops` still reports the `c − n − p` fill count per
site). The FA program is unchanged in token counts (9/9, 8/8, 8/8, 8/8, 1/1),
instruction count (128) and test results; its cross-iteration pops moved to
body end. The addendum's "pop before the fill of a read slot" and the `moved_pops`
repair are subsumed.

## TileLang constructs with no tile-level form

Tile-level gather, top-k, dequant-on-load, a pipelined loop over a runtime
count, and `ceil` of a runtime count on the device (see E2 §7). Everything
else the kernel needs — dynamic `T.serial`, nested static `T.Pipelined` with
rings and events, sliced gemm operands `H[nn, :, :]`, elementwise ACC×ACC,
`T.sigmoid`, in-place convert, `T.reduce_sum` into a scalar used as a loop
extent — survives #19 intact.

# Phase 3: persistent expert loop, combine, overlap (F5)

Task: `docs/tilelang-phase3-task.md`. ISA/simulator v0.1.1 (synced per §0).
Kernels: `compiler/prototype/moe_kernel.py` — `build_moe_persistent_kernel()`
(F2), `build_moe_combine_kernel()` (F1), `build_moe_expert_kernel()` (the
phase-2 per-expert program, updated); both phase-3 kernels live in that module
rather than in the task's `moe_persistent.py` / `moe_combine_kernel.py`
because F2 is the phase-2 token tile wrapped in an expert loop and shares its
macros. Cut PrimFuncs `dumps/cut_moe_persistent.py`, `cut_moe_combine.py`;
programs `dumps/program_moe_persistent.txt`, `program_moe_combine_t0.txt`;
overlap evidence `docs/overlap-report.md` (F3) from
`compiler/prototype/overlap_report.py`; tests `tests/test_tilelang_moe.py`
(F4, 18 tests). Whole repo: **42 passed**.

## Verdict: PASS

Against §3 of the phase-3 task:

- **F1 and F2 are authored in TileLang and cut at #19.** The persistent
  kernel is a static `for ee in T.serial(E // 2)` around two dynamic
  `for t in T.serial(T.ceildiv(n[0], 64))` token-tile loops (two, so that the
  count of expert `e+1` is prefetched into the other count register while `e`
  runs). The nested dynamic loop survives every pass up to #19 unchanged
  (`test_persistent_nested_dynamic_loop_survives`): no expert-loop wrapper
  outside TileLang was needed, so the PASS-WITH-CAVEATS case of §3 does not
  apply. The combine kernel is four `T.Pipelined`-free loops: gather `k`
  partial rows per token (`LOAD.IDX` into ACC), add in ACC (`VGATE add`), add
  the shared expert, `STORE.2D`.
- **F3 shows overlap at every boundary.** With the default weight ring depth 3,
  the first `LOAD.IDX` of expert `e+1` is issued 3–4 steps before the last MMA
  of expert `e` at all 31 boundaries, with two MMAs of `e` still queued and
  two LOAD instructions of `e+1` (gather and first weight fill) already issued.
  Ring 4 gives 9–10 steps / 3 MMAs; ring 2 gives none (the gather issues on
  the same step as the last MMA). Blanking two experts
  (`test_zero_token_expert_in_persistent_program`) leaves 29 overlapped
  boundaries and no DRAM write for the blanked ones.
- **F4: `y` matches end to end.** Persistent program → `Yp[B·k, d]` →
  four combine programs → `y`, rowmax error 2.15e-3 (`tol_rowmax` 7.8e-3);
  `Yp` against `y_parts` 3.09e-3 for both weight formats (the bf16 program
  sees the dequantized f4 values, so the two agree exactly). Token checking
  on; every program finishes `ok` with empty FIFOs.

One caveat that does not change the verdict (F2 is in TileLang and F3 holds)
but that the main session should know:

- **The emitted program unrolls the expert loop.** The task asked for a
  static outer `LOOP n=E` with `LOOP r, step=64` inside. The simulator applies
  an instruction's `incr` to the innermost enclosing loop only, so an outer
  hardware `LOOP` cannot advance the per-expert operands of instructions
  inside the inner `LOOP r` (weight bases, index rows, the count register).
  D4 therefore unrolls `ee` and emits 32 `LOOP R, step=64` sections
  (alternating `R0`/`R1`), 8025 instructions for 32 experts. The TileLang
  source is the nested loop; only the last mapping step flattens it. An ISA
  v0.2 candidate is per-level `incr` (or an outer-loop-indexed base
  register); until then the persistent program's size is linear in E, which is
  fine for E = 32 and not for a real model.

## Does a decoupled machine need software pipelining in the instruction stream?

No — and F3 is the measurement. The persistent program has no prologue,
epilogue or cross-expert interleaving: each expert's section is the straight
transcription of the TileLang token tile, and the sections are back to back.
Overlap between the gather of expert `e+1` and the GEMMs of expert `e` comes
from the rings (2 INP slots for the token tile, 3 WGT slots for the weight
stream, 2 ACC slots for the output tile) plus release tokens on every slot,
and from a token placement rule that never drains a `LOOP` body: a load into
a slot pops exactly the release of that slot's previous occupant, so the
in-order LOAD queue leaves an expert carrying only the pops that gate its
next instruction and runs ahead as far as the rings allow. The depth of the
overlap is then a resource number, not a schedule: ring depth N keeps N−1
GEMMs of `e` in flight when `e+1`'s tokens start arriving (1, 2, 3 for N = 2,
3, 4 in the report), and raising it costs a WGT slot, not an instruction.
What TileLang's software pipelining (#19) still contributes is the *ring
itself* — the `W_s[m % 3]` indexing and the `fill` placed N−1 iterations
ahead — which on this machine is a buffer-allocation decision rather than a
schedule; its prologue/epilogue copies are just the first fills and the last
GEMMs, with the tokens doing the waiting. Two things did have to be in the
instruction stream, and both are about the *tail* of an expert rather than
the head: the output tile must be double-buffered (otherwise the first GEMM of
each N-tile waits for the scale → `STORE.IDX` round trip of the previous one,
a serialisation no token rule can hide), and the W2 phase must be one
flattened stream so the ring is not re-primed per N-tile (TileLang restarts a
`T.Pipelined` ring per loop). So: no pipelining of instructions, yes to
buffers sized for the lead you want, and keep single-buffered round trips
off the tail of a loop body.

## Numbers

| Program | instrs | sim steps | notes |
|---|---|---|---|
| persistent, f4.b32, ring 3 | 8025 | 3352 | 32 experts, 768 MMAs, 128 gathers, 128 scatters; 32 `LOOP R, step 64` |
| persistent, f4.b32, ring 2 / 4 | 8023 / 8027 | 3252 / 3195 | overlap 0 / 9–10 steps |
| persistent, bf16, ring 3 | 8025 | 3255 | overlap 3–4 steps |
| per-expert program (E4 updated), f4.b32 / bf16 | 271 | 124 / 120 | one token tile; `fill_wait` 48 sites, f4 {0,1,3} pops, bf16 1 |
| combine, one 64-token tile | 61 | 154 | per N-tile 6 `LOAD.IDX`, 7 `VGATE add`, 1 `STORE.2D` (`LOOP n=4`); 4 programs for B = 256 |
| router, one token | 8 | — | `VTOPK` + `STORE.2D i32` |

Token graph of the persistent program (pushes = pops on every edge):
LOAD→MMA 896, MMA→LOAD 771, MMA→VEC 256, VEC→MMA 258, VEC→STORE 160,
STORE→MMA 130, STORE→VEC 130, LOAD→VEC 97, VEC→LOAD 97; 10 primed before
the first `LOOP`, 30 hoisted pops at loop entries, **0 popped at body end**,
11 drained before FINISH. The step totals are issue-slot counts in which
every `PUSH`/`POP` costs a slot (the MMA queue runs 3079 instructions for 768
GEMMs), so they are not a performance figure; see the report's last section.

## What changed in D4 (`tile_tir_to_program.py`)

- **Level-invariant token placement** replaces phase 2's cumulative scheme.
  Loop bodies are analysed twice (two passes of the timeline) so that every
  steady-state need is classified as "this iteration's push" or "the previous
  iteration's push"; the FIFO level at an iteration boundary is then fixed per
  edge and pops are placed where that level makes the matching push reachable.
  Dependencies on pre-loop pushes are hoisted before `LOOP` with priming
  pushes if the entry level is short; primes are drained before FINISH. The
  consequence that F3 needed: a body never pops at its end, so the queues run
  into the next iteration and the next expert. On FA this adds one
  MMA→VEC push/pop pair (9/9 instead of 8/8; the O rescale's dependency on
  the previous PV GEMM is primed before `LOOP` and drained after, instead of
  popped at the end of the pushing iteration), 130 instructions instead of
  128, identical numerics.
- Dynamic extents `(n[0] + 63) // 64` → `LOOP R, step=64`; static loops that
  contain a dynamic loop are unrolled (the `incr` limitation above); a
  constant-condition `if` after substitution is resolved statically (the
  guarded `fill`/scale/scatter of the flattened W2 stream).
- Scatter `if dst_v[i] >= 0: Yp[dst_v[i], c + j] = out[...]` → `STORE.IDX`
  (the region base carries the column offset since `STORE.IDX` has no `col0`);
  gathers are allowed into INP or ACC; `VGATE add` for ACC+ACC; VEC→VEC row
  copy → `VSEL dst, a, a, mask` (a copy idiom; `VCOPY` stays a v0.2 candidate).
- `MappingReport` grew `loops`, `loop_pcs`, `max_fifo_level`, `primed_pushes`,
  `hoisted_pops`, `body_end_pops`; `Program.tags` maps pc → loop id, which is
  how the overlap report attributes trace entries to experts.

## Findings for the main session (simulator / ISA, not changed per §4)

1. `incr` applies to the innermost loop only → the persistent program is
   unrolled over experts (above). v0.2 candidate: per-level `incr`.
2. No LOAD→STORE or STORE→LOAD edge in `EDGES`: a tile written by `LOAD` may
   only have VEC readers, so the scatter's index row has to be relayed
   `LOAD → VEC (VSEL copy) → STORE.IDX`. Either the edge or a `STORE.IDX`
   that takes its index from a register/INP tile would save a VEC
   instruction and two tokens per tile.
3. `LOAD.IDX` rounds gathered rows to bf16 even when the destination is ACC
   (f32). The combine gathers f32 partials, so `y` carries bf16 rounding of
   the partials (1.8e-3 against the fixture's own `y_parts`; within
   tolerance, but a `LOAD.IDX f32` would remove it).
4. `PUSH`/`POP` occupy an issue slot of their queue in the step model, so the
   MMA queue issues on ~92 % of steps while executing 25 % GEMMs. Whether a
   wait/release should be a field of the work instruction is a hardware
   question; the simulator's accounting makes step totals useless as a
   throughput proxy until it is answered or a cycle model exists.
5. Static FIFO level 6 on VEC→MMA with depth 4 (the SwiGLU pushes four
   P-ready/out-rewritten tokens before the first pop): the producer stalls,
   the run is clean. D4 reports the per-edge maximum instead of asserting it.
6. `Machine.reset()` keeps `stats` (documented); the tests compare deltas.

## Findings for TileLang (recorded, GPU-relevant)

1. **A guarded producer in stage 0 is multi-buffered as if unconditional.**
   Putting the token gather inside the pipelined weight loop as
   `if nn == 0: x_s[...] = ...` made InjectSoftwarePipeline allocate
   `x_s[2, 2, 64, 128]` and read ring slot `m % 2` that the guarded gather
   never wrote: wrong numerics with a clean token trace. The gather is kept
   outside the pipelined loop (per K-tile, 2-slot INP ring by hand).
2. Manual `T.Pipelined(order=, stage=)` lists need one entry per *statement*,
   counting statements inside `if` guards (PipelinePlanning asserts
   `annotations.size() == original_stmt_count`).
3. Buffers allocated outside the body macro (e.g. in a helper object) lose
   their names (`buffer_N`) and the scope rewrite cannot find them; allocate
   named buffers inside the `@T.macro` body and pass them as parameters.
4. `T.ceildiv(n[0], 64)` and a guarded scatter inside `T.Parallel` survive
   #19 verbatim; a static `T.serial` around dynamic `T.serial` loops survives;
   `T.Pipelined` rings restart per loop (hence the flattened W2 stream).

# Phase 4, §0: ISA v0.1.2 sync and the F3 restatement

Task: `docs/tilelang-phase4-task.md` §0. The tarball's `isa/`, `sim/`,
`tests/test_sim.py` replaced the v0.1.1 ones; the phase-3 trace list was
re-applied to the new `machine.py` as a `_trace` helper (approved, no
semantic effect). Whole repo after §0: **45 passed** (`pytest tests -q`).

## Phase-3 findings closed by v0.1.2

| # | Phase-3 finding | v0.1.2 | What changed on this side |
|---|---|---|---|
| 1 | `incr` innermost-only → expert loop unrolled (8,025 instrs) | per-level `incr` (`field@L`, `region.scales_base`) | D4 fits operands as affine functions of all loop variables and emits one key per level; the expert loop is a hardware `LOOP n=16`; 527 instructions |
| 2 | no LOAD↔STORE edge → `VSEL` relay for the scatter row | `(LOAD, STORE)`, `(STORE, LOAD)` in `EDGES` | `STORE.IDX` reads the loaded `dst_s` row (`dst_v` and its copy removed); LOAD→STORE 32/32, STORE→LOAD 33/33 per program |
| 3 | `LOAD.IDX` rounds to bf16 into ACC | rounds only into INP/WGT | combine `y` 1.519e-3 (was 2.149e-3 end to end) |
| 4 | `PUSH`/`POP` cost an issue slot | free sync (§8.5), `stats["sync"]` | step totals are an issue-slot proxy of the critical queue; F3 restated below |
| 5 | FIFO level 6 > 4 on VEC→MMA | device parameter (config C: 16) | reported, not asserted, unchanged |
| 6 | `reset()` keeps stats | documented | tests compare deltas, unchanged |

## §0.2: the nested program

`LOOP n=16 { count → R1; LOOP r=0, step=64 { tile }; count → R0; LOOP r=1,
step=64 { tile } }`, 527 instructions (`dumps/program_moe_persistent.txt`),
numerics unchanged (`Yp` 3.092e-3 both formats, `y` 1.519e-3). It is 527 and
not the ~270 the task expected because the count prefetch into the other
register needs two tiles per outer iteration. The single-register form
`LOOP n=32 { count; LOOP r=0 }` is `build_moe_persistent_kernel(prefetch=False)`:
274 instructions, same numerics, 1153 steps instead of 1092 (its `LOOP r`
waits for its own `VCOUNT`; `test_persistent_without_prefetch`,
`dumps/program_moe_persistent_noprefetch.txt`). The idx table carries a `-1`
pad row at `E` so the last prefetch reads row `E` instead of clamping (D4's
affine fit rejects `T.min(e + 1, E - 1)`).

## §0.3: F3 under free sync

Full account in `docs/overlap-report.md`. Summary:

- **The LOAD queue is the critical queue** (32 work instructions per tile
  against MMA's 24; busy on 1025 of 1092 steps). The phase-3 criterion
  "first gather of `e+1` before the last MMA of `e`" measured how far LOAD ran
  ahead of an MMA queue slowed by its own `PUSH`/`POP` slots; with equal
  issue costs the gather trails the last MMA by construction (one step at
  ring 3) and the ring depth no longer changes the boundary (ring 2/3/4: same
  one idle LOAD step). **F3 cannot be "unchanged"**; the phase-3 overlap was
  an artefact of the slot accounting.
- **Restated criterion, on the critical queue:** LOAD-idle steps at the
  boundary, MMA gap at the boundary, and the token slack of the first gather.
  Result at all 31 boundaries, every ring depth: LOAD idle 1 (the next
  count's `LOAD.2D`, work of no expert), MMA gap 3 (the gather → fill → MMA
  chain; 2 for bf16), token slack 5 (the gather's dependencies were met five
  steps before `e`'s last MMA). Nothing waits for the previous expert.
  `test_overlap_at_every_expert_boundary` and the zero-token test assert
  these.
- **Found and fixed on the way:** the first nested program idled LOAD 2–3
  steps per boundary (1138 steps). Cause: the two counts of an outer
  iteration shared their `idx_all`/`mask` fragments, so the second count's
  load depended on the first `VCOUNT`'s release across the inner `LOOP r`;
  D4 correctly pops a dependency that crosses a dynamic loop in front of it
  (a zero-trip loop must be transparent), and that pop waited for the
  `VCOUNT`. One buffer pair per count register removes it. Smaller instance
  of the same thing: the combine-weight column released VEC→LOAD only after
  the tile's last `VSCALE_ACC`; a VEC-owned copy (`w_c`) moves the release to
  the tile's start.

## Numbers (v0.1.2, free sync)

| Program | instrs | steps | LOAD / MMA / VEC / STORE busy | notes |
|---|---|---|---|---|
| persistent, f4.b32, ring 3 (default) | 527 | 1092 | 1025 / 768 / 481 / 128 | 2402 work, 5528 sync; 31/31 boundaries idle 1, gap 3 |
| persistent, f4.b32, ring 2 / 4 | 525 / 529 | 1091 / 1125 | same | boundary unchanged; ring 4 adds prologue |
| persistent, bf16, ring 3 | 527 | 1060 | same | MMA gap 2 |
| persistent, no prefetch, f4.b32 / bf16 | 274 | 1153 / 1121 | | LOAD idle 3 per boundary |
| per-expert program, f4.b32 / bf16 | 270 | — | | one token tile |
| combine, one 64-token tile | 61 | | | f32 gathers |
| router, one token | 8 | | | |

Token graph of the persistent program (pushes = pops on every edge): LOAD→MMA
896, MMA→LOAD 771, MMA→VEC 256, VEC→MMA 258, VEC→STORE 128, STORE→MMA 130,
STORE→VEC 130, LOAD→VEC 65, VEC→LOAD 82, LOAD→STORE 32, STORE→LOAD 33; 10
primed before the first `LOOP`, 2 hoisted pops at loop entries, **0 popped
at body end**, 11 drained before FINISH.

## Findings for the main session (v0.1.2; nothing changed in `sim/`)

1. **Loop-entry pops are where cross-queue latency surfaces.** Anonymous
   per-edge FIFOs + zero-trip-transparent `LOOP r` mean every release pushed
   before a loop and consumed after it is popped at the loop entry, and the
   entering queue then waits for the pushing queue's progress. Kernel-side
   remedy: never reuse a buffer across a dynamic loop between different
   queues (one buffer per in-flight count). ISA-side candidates if this
   recurs in attention-style kernels: a count instruction on the LOAD queue
   (`LOAD.COUNT reg, region`, keeping the trip-count round trip inside one
   queue), or tagged tokens so that a far consumer's release need not be
   drained at the loop entry.
2. **Issue-slot totals say which queue saturates, not how long it takes.**
   The LOAD queue's 32 instructions per tile (4 gathers, 24 fills, 4 scale
   loads for f4) make the MoE tile LOAD-issue-bound at equal costs; the
   cycle model decides whether gather and fill bandwidth hide behind the
   GEMMs. Open item for the gate-1 report.
3. **The count prefetch is worth 5.6 %** of the persistent program's steps
   at equal issue costs, and it doubles the program body. If the cycle model
   makes the `VCOUNT` round trip cheap relative to a tile, the
   single-register form is the one to keep.

# Phase 4: MLA decode, KDA chunkwise, Gate 1 closure (G5)

Task: `docs/tilelang-phase4-task.md`. ISA/simulator v0.1.2 (§0 above).
Kernels: `compiler/prototype/mla_kernel.py` (G1), `kda_kernel.py` (G2),
`gemm_kernel.py` (the GEMM family, for the Gate 1 report). Cut PrimFuncs
`dumps/cut_mla.py`, `cut_kda.py`, `cut_gemm.py`; programs `dumps/program_mla.txt`
(123 instructions), `program_kda.txt` (521), `program_gemm.txt` (13).
Expressivity record `docs/tilelang-mla-kda-expressivity.md` (G3); tests
`tests/test_tilelang_mla_kda.py` (G4, 13 tests); Gate 1 report
`docs/gate1-report.md`. Whole repo: **58 passed**.

## Verdict: PASS-WITH-CAVEATS

Against §3 of the phase-4 task:

- **G1 (MLA) is a PASS with no caveat.** The 64-row stacked Q tile, the 576-deep
  reduction as one 64-deep rope chunk plus four 128-deep chunks, the resident K
  chunks doubling as V, the online softmax and the `[64, 512]` output in the
  wide accumulator are all TileLang; the `acc_o` fragment survives #19 as one
  buffer with sliced gemm destinations, D4 maps it to four consecutive ACC
  slots with `VSCALE_ACC wide` and `STORE.2D wide`. `O` matches the fixture at
  2.15e-3 (`tol_rowmax` 7.8e-3) with token checking on. The one note is a
  device parameter: Q (4.5 tiles) does not fit the 2-slot INP scratchpad and is
  re-streamed at every KV tile (10 LOAD.2D per tile instead of 5).
- **G2 (KDA) is a PASS-WITH-CAVEATS: no extern, but five mapping idioms that
  were not in D4's pattern set**, each named in G3 §11: (1) computed B operands
  go through a DRAM scratch round trip (`STORE.2D` → `LOAD.2D`, the v0.1.2
  edges), because only LOAD writes WGT/STATE; (2) the triangular masks are
  multiplicative 0/1 VEC tiles applied with `VGATE mul` — `MMA.MASKED`'s
  additive −inf mask would poison the Neumann products and is not used; (3) U,
  R and the decayed Kᵀ are fed as bf16 hi/lo pairs, because the plain bf16
  schedule misses the tolerance (8.2e-3 on `o`); (4) the resident state is two
  64-row ACC tiles plus an f32 STATE mirror reloaded from the checkpoint,
  because the array and the VEC ops write 64 rows and STATE has no row-offset
  destination; (5) live-range slot sharing and the ISA §11 config-C descriptor
  (the KV260 descriptor cannot hold the chunk's working set). With these, every
  `S_chunks[c]` checkpoint matches (1.46 / 1.36 / 1.15 / 1.27e-3) and `o`
  matches (5.99e-3) — the same figures the NumPy model with the ISA's roundings
  predicts. `ref/kernels.py::kda_recurrent` was not changed: the chunk algebra
  reproduces it exactly in float64.
- **G4 is green** (13 tests: cut structure, program shape, token graphs, the
  config-C requirement, numerics for MLA, KDA and the GEMM).
- **Neither FAIL condition applies.** The wide accumulator and the resident
  state survive #19 (asserted on the cut PrimFuncs), and the checkpoints match.

## MLA in numbers

| | |
|---|---|
| program | 123 instructions, `LOOP n=16` over KV tiles (chunk loops unrolled) |
| per KV tile | 10 LOAD.2D (5 Q, 5 K), 5 `MMA.T` (QKᵀ) + 4 `MMA` (PV into the four slots), 18 VEC |
| run | 451 steps, 601 work instructions, 854 free syncs, 144 MMAs; VEC busy 296 steps, LOAD 160, MMA 144 |
| tokens | LOAD→MMA 160, MMA→LOAD 150, MMA→VEC 84, VEC→MMA 32, VEC→STORE 1 (pushes = pops); 10 primed, 6 drained, 0 at body end; FIFO max MMA→LOAD 6, MMA→VEC 5 |
| numerics | `O` rowmax 2.15e-3 (tol 7.8e-3) |
| DRAM | 2.36 MB read (K 1.18 MB + Q re-streamed 16×), 128 KB written |

## KDA in numbers

| | |
|---|---|
| program | 521 instructions, `LOOP n=4` over chunks (solve and state-half loops unrolled), config C |
| per chunk | 27 LOAD (8 gathers, 19 LOAD.2D incl. 13 round-trip reloads and the STATE reload), 27 MMA, 72 VEC, 14 STORE |
| run | 335 steps, 566 work, 1374 free syncs, 108 MMAs; VEC busy 290 steps, LOAD 112, MMA 108, STORE 56 |
| tokens | 12 edges, 688 tokens (pushes = pops); STORE→LOAD 49, LOAD→STORE 42; 15 primed, 3 hoisted, 16 drained, 0 at body end; FIFO max MMA→LOAD 8 |
| slots | INP 3, WGT 8, ACC 8 (14 buffers share), VEC 5 + 2 rows, STATE 1 |
| numerics | `S_chunks` 1.46 / 1.36 / 1.15 / 1.27e-3; `o` 5.99e-3 (tol 7.8e-3); plain bf16 would be 8.2e-3 |
| DRAM | 1.77 MB read, 1.71 MB written (1.3 MB of it scratch round trips) |

## Findings for the main session (v0.1.2; nothing changed in `sim/`)

1. **Only LOAD writes WGT/STATE, so every device-computed B operand costs a
   DRAM round trip.** KDA has 13 per chunk (K̄, the five Neumann squares, the
   hi/lo pairs of R and U). ISA v0.2 candidate: an ACC→WGT move (`VCVT
   dst=WGT`) or an MMA B operand from ACC. This is the largest single cost in
   the KDA program and sits on the solve's critical path.
2. **`MMA.MASKED` (additive −inf) is the softmax mask, not the triangular
   mask.** The delta rule needs zeros. `VGATE mul` with a 0/1 VEC tile does
   it in one VEC instruction; a multiplicative variant of `MMA.MASKED` would
   fuse it. The simulator's `VGATE` accepts a 2-D VEC source; ISA §5.3 should
   say so.
3. **bf16 array inputs are marginal for the delta rule.** Plain bf16 misses
   the tolerance on `o`; hi/lo pairs on three operands recover it at +7 MMAs
   per chunk (27 vs 20). Whether the array should offer a bf16×2 / f32 operand
   mode, or the reference should define those products in bf16, is a Gate 1
   open item.
4. **The 128-row state does not fit the 64-row write path.** MMA and VEC write
   64 rows; STATE has no row-offset destination, so the state is two ACC tiles
   and STATE is a read-only mirror reloaded per chunk (64 KB). Candidates: a
   row-offset destination for MMA/VEC on STATE, or 128-row state ops.
5. **No transposed A operand.** The state update contracts over tokens
   (`Kᵀ U`), so the host provides `kt` and `gt` layouts. A `LOAD.2D transpose`
   or an `MMA` A-transpose qualifier would remove them.
6. **`_op_VNEG` cannot execute.** It goes through `_binop`, which reads
   `imm` when no `vec` is given; the assembler's `vneg(src)` emits neither, so
   any program using `VNEG` faults with `KeyError: 'imm'`. D4 emits `VMUL
   imm=-1.0` instead. A one-line simulator fix (`fn(x, None)` path for
   unary ops).
7. **INP depth.** The MLA Q tile is 4.5 INP tiles; with 2 slots it is
   re-streamed (2× DRAM reads). The MoE tile is LOAD-issue-bound already; the
   cycle model should price INP 5–6 for decode shapes.
8. **Device descriptor.** KDA needs config C (INP 3, WGT 8, ACC 8 after
   sharing, VEC 7, STATE 1, FIFO 8); the default descriptor fails at slot
   allocation. The report lists the per-family maxima.

## Findings for TileLang (recorded)

- `from __future__ import annotations` breaks `@T.prim_func` signatures whose
  shapes are closure variables (annotations are evaluated by name).
- 3-D buffers must be indexed on every axis in region operands.
- Column slices of a fragment are legal gemm destinations and survive #19 as
  regions with a column offset — the whole of the wide-accumulator story rests
  on this.
- `T.copy` between two fragments of different dtypes is a `Cast` on the
  store; a copy whose source is read later must not be mapped in place (D4
  now checks the later reads).

## ISA v0.2 candidates added by this phase

ACC→WGT move (or MMA B from ACC); multiplicative `MMA.MASKED`; bf16×2 / f32
operand mode; row-offset STATE/ACC destinations or 128-row state ops;
transposed A load; `VNEG` fix (simulator). The consolidated list across all
phases is in `docs/gate1-report.md`.

# Phase 5: the cycle model (H5)

Task: `docs/phase5-cycle-model-task.md`. ISA/simulator v0.1.3 (§0: the Gate 1
freeze candidate). Deliverables: `sim/tcta/timing.py` + `sim/tcta/cycle.py`
(H1), `tests/test_cycle.py` (H2, 16 tests), `model/sweeps.py` +
`docs/cycle-model-report.md` + `model/cycle_results.{json,md}` (H3),
`model/module0.py` hook + `model/README.md` (H4), this section and the Gate 1
report §4 (H5). Whole repo: **80 passed** (`pytest tests -q`). Nothing in
`isa/` changed; `sim/` gained two new modules and no change to `machine.py`
beyond the phase-3 trace list.

## §0: v0.1.3 sync

KDA re-emitted on v0.1.3: 521 → 352 instructions, 13 DRAM round trips + a 64 KB
state reload per chunk → none, 230 KB of DRAM reads per chunk (inputs only),
checkpoints and `o` unchanged (1.46 / 1.36 / 1.15 / 1.27e-3, 5.99e-3). D4 gained
`VCVT dst=INP|WGT`, `VCVT dst=STATE, row0`, `VCVT to_bf16` (the ACC→ACC cast is
now explicit: v0.1.3 made `VCVT dst=ACC` an f32 move, which KDA's hi/lo split
first missed at 8.17e-3), `VCOPY`, `VNEG`, and the `MMA` + `VGATE mul` →
`MMA.MASKED mode=mul` fusion. The fusion does not fire in KDA: the masked
products are `MMA.T` (transposed B) and `MMA.MASKED` has no transposed form
(§3 of the Gate 1 report). MLA 123 → 122, FA 130 → 127, MoE 527 → 525 via
`VCOPY`. The v0.1.2 KDA authoring is kept as `kda_kernel_roundtrip.py` (and
`dumps/program_kda_roundtrip.txt`) for the before/after sweep; MLA gained
`build_mla_kernel(inp_slots=n)` for the INP sweep (`dumps/program_mla_inp6.txt`).

## Verdict: PASS

Against §5 of the task: H1–H5 are delivered; every sweep row is a
`CycleMachine` run whose functional result was checked against the fixture
(`model/sweeps.py::run` asserts it; 150 runs, no numerics change); and the one
FAIL condition does not occur: `test_popped_tokens_were_pushed_first_on_every_family`
checks every popped token of the MLA and KDA programs for start(consumer) ≥
completion(producer). The sanity tests (a)–(e) of H2 pass on both descriptors.

## The answers, in one table (config C; details in `docs/cycle-model-report.md` §4)

| Question | Number | Recommended value |
|---|---|---|
| INP slots (MLA Q re-streaming) | tile 4603 → 4100 (3 slots) → 3665 cycles (5 slots, −20 %); DRAM 2304 → 1224 KB per step | **INP 5** at config C; KV260 array-bound, not a lever |
| WGT ring (MoE boundary) | tile 14925 (2) / 13554 (3) / 11586 (4, −14.5 %) | **WGT ring 4** at config C, 3 on the KV260 |
| FIFO depth | 4 deadlocks KDA; MLA loses 4 cycles; others identical at 4/8/16; deepest dynamic level 6 | **8 on every edge**, both descriptors |
| `row` vs `burst64` | 0–0.5 % at config C; +5.3 % on KDA at the KV260 (16 outstanding bursts) | `row` with ≥ 16 outstanding rows on v1; immaterial at config C |
| VEC lanes / SFU fusion | MLA tile 7865 (16, unfused) → 4603 (64, fused) → 3975 (128); FA 3047 → 1177 → 860; fusion −18 % / −47 % | **64 lanes + fused** (config C); FA needs 128-row tiles |
| DRAM latency, KDA before/after | per chunk 26535 → 15176 at 300 (−43 %); slope 25.0 → 6.5 cycles per latency cycle | the v0.1.3 moves stay |
| DRAM latency, MoE count round trip | prefetch 13554 vs 14498 per tile (−7 %), = oracle front end | keep the prefetch form |

## Findings for the main session (nothing changed in `isa/`; `sim/` only gained the cycle model)

1. **Two device rules the task left implicit decide the config-C numbers,
   and the descriptor should carry them.** (a) How a request maps to
   channels: one request on one channel (the rule as written) vs a tile
   striped over two channels is MoE −20 %, MLA −17 %, KDA −17 %, FA −5 %.
   (b) Whether the four arrays can read one operand at once: under one read
   port per bank they serialize on the MoE token tile (32640 stall cycles per
   INP bank, −3.8 % with four ports or a broadcast). Both are
   `DeviceTiming` what-ifs (`request_channels`, `ports_per_bank`) defaulting
   to the task's rules.
2. **One in-order program keeps the eight channels at 13–18 % of peak**
   (η_bw of the MoE decode tile 0.146, 0.181 striped), because a 2–3-deep
   ring of 8–16 KB tiles holds 24–48 KB in flight against the ~77 KB that
   300 cycles × 256 B/cycle needs. Module 0 with these values sizes a replica
   at 255 (204) chips instead of 53 and $0.249 ($0.210) per Mtoken instead of
   $0.093. The device question behind it: does a config-C chip run one program
   against its channels or several? The one-program value is what the model
   records (`model/README.md`).
3. **FIFO depth 4 (the v1 default) deadlocks KDA.** Several edges of the
   program reach a static level of 5; the modelled deadlock is reported as
   `CycleDeadlock` with each queue's blocked head. Depth 8 covers all five
   families (deepest dynamic level 6).
4. **The MoE tile at config C is gather-bound** under the one-channel rule:
   four `LOAD.IDX` of 64 rows × 768 B at 1836 cycles each in a 13554-cycle
   tile. With rows spread over channels a gather is 1181 cycles, bound by the
   one-request-per-cycle issue rate; that rate is the next DMA parameter to
   fix.
5. **FA at 64-row tiles is VEC-bound at every lane count on config C** (VEC
   48 % of the tile even at 128 lanes; the 64×64 array is 2.7 % busy).
   Attention prefill needs 128-row tiles for this array.
6. **The issue-slot estimate for the WGT ring was wrong in cycles**: ring 4
   is −14.5 % at config C because a weight tile load (534 cycles) is twice
   the MMA that consumes it (268). Issue-slot counts cannot see this.
7. **`MMA.MASKED` has no transposed-B form** (carried from §0): KDA's masked
   products are `MMA.T`, so the v0.1.3 `mode=mul` is unused there and the
   `VGATE mul` masks remain (2 per chunk).
8. **Sim detail**: `VCVT` with `row0 == 0` resets the destination's valid
   rows to (M, N) while `row0 > 0` grows them to the union, so the row-0 half
   must be written first; D4 orders the two `vcvt_state` writes accordingly.

## Calibration list (task H5)

Calibrate against Verilator: request → channel mapping; `dram_latency_cycles`
(constant in the model); `bytes_per_cycle_per_channel` and the dequant rate;
the gather issue rate and `max_outstanding_rows`; `mma_latency` and the
array's operand streaming (A/B from the scratchpads for the whole occupancy);
VEC/SFU per-op cycles and pass counts; bank port counts and hold durations;
`vcount_to_loop_latency`, loop overhead, dispatch rate and queue buffer
depths; the PUSH/scoreboard latency; the COMM link. Fixed by construction
(ISA semantics): in-order queues, POP consumed at the queue, PUSH ordered per
edge at the producer's completion, FIFO depth as unpopped tokens with a full
FIFO holding the producer, one request per LOAD.2D/STORE.2D and per live
gather row, `-1` rows free, no speculation past `LOOP r`, accumulate
forwarding, the functional run's dispatch order and operand shapes. The full
table with the leverage each parameter showed is `docs/cycle-model-report.md`
§6.

# Phase 6: memory-level parallelism (rings, cores, the real η_bw)

Task: `docs/phase6-mlp-task.md`. ISA v0.1.4 (§11 only). Deliverables: I1
ring sweep, I2 multi-core `CycleMachine` (`run_multicore`, arbiter,
`stall_arbiter`; 4 new tests), I3 core sweep, I4 striping × cores, I5 Module
0 keys (`module0_recommended`, consumed by default), I6 this section and
`docs/phase6-mlp-report.md`. Whole repo: **84 passed**. Code touched:
`sim/tcta/{timing,cycle,isa}.py`, `compiler/prototype/{moe,mla,kda}_kernel.py`
(ring depth, expert subset, K-ring span only), `model/{sweeps_mlp,module0,sweeps}.py`.
The repo in this session is canonical; the spec delta (§11) was applied as
received.

## Verdict: PASS

I1–I6 delivered; every one of the 39 timed runs (16 single-stream, 23
through `run_multicore`) reproduced its fixture, the multi-core runs per core; the
H2(c) dependency check still passes on every family; no multi-core per-core
result differs from the single-core run (`test_multicore_*`). I6 states a
descriptor with numbers (below).

## The answers, in one table (config C, cycles at 1 GHz; `docs/phase6-mlp-report.md`)

| Question | Number | Decision |
|---|---|---|
| I1 ring depth, MoE token tile | 9974 (ring 3, §1 descriptor) → 8561 (flat `[W1\|W3]`, 3) → 7116 (6) → **6481 (12)** → 6750 (16); η_bw 0.198 → 0.304; 50 KB in flight at the knee, channels 25 % | **WGT ring 12**; the knee is the tile's serial VEC/SFU and accumulation chains, not Little's law |
| I1 ring depth, MLA KV tile | 2772 (K ring 4) → **2177 (10)** → 2177 (20, needs FIFO 36); η_bw 0.122 → 0.155 | **K ring over 2 KV tiles (10 slots)** |
| FIFO at the knee | ring 12 holds MMA→LOAD at depth 8 for 2061 cycles, 0 cost; MLA ring 20 times out functionally at 8 | **FIFO 8 stands**; rule FIFO ≥ r·(d − 1) |
| I2 cores sanity | GEMM copies 1.00× / 1.00× / 1.11× (idle channels); 16 × 32 KB stream 1.87× / 3.61× (channels 87 → 97 %) | arrival-order arbiter, ties rotate; recorded |
| I3 MoE on 1 / 2 / 4 cores (E/c experts) | per tile 9314 / 4812 / **3063**; η_bw 0.212 / 0.410 / **0.644**; η_mfu 0.165 / 0.319 / **0.502**; channels 17 / 33 / 52 %; per-core completion within 1.4 % | 4 cores |
| I3 MLA copies on 1 / 2 / 4 cores | per tile 2561 / 1356 / **758**; η_bw 0.132 / 0.249 / **0.446**; per-core times 42378 … 48522 (convoy, +4.8 % per core index) | 4 cores; arbiter finding |
| I3 vs I1: 4 cores × 1 array vs 1 core × 4-array unit | MoE 3063 vs 6481 (2.1×); MLA 758 vs 2177 (2.9×) | **4 cores × 1 array** |
| I4 striping × cores (MoE) | k = 2: −5.7 % (1 core), −5.6 % (2), +0.4 % (4); k = 8: −4.8 %, +10.5 %, +26.6 % | keep **2**; never 8 |
| I5 Module 0 | η_bw 0.146 → **0.644**, η_mfu 0.113 → **0.502**; N 255 → **58** chips per replica; $0.249 → **$0.096** per Mtoken (B* $0.093 → $0.036); 12LP+/HBM2e N 76 → 21 | `module0_recommended` is the default |
| I6 slots per core | INP 5 (MLA Q resident), WGT 32 (knee uses 20 for MoE, 10 for MLA; ≥ 24), ACC 8 + 2, VEC 64, STATE 2 | as §11 |

η_bw by lever (MoE tile): 0.146 (phase 5) → 0.198 (striping 2 + broadcast +
FIFO 8 + WGT banks) → 0.230 (one `[W1|W3]` ring) → 0.304 (ring 12) → **0.644**
(4 cores). Two thirds of the gain is cores, a quarter the ring.

## Findings for the main session

1. **Descriptor decision: config C is 4 cores × one 64×64 array** (I3 vs I1
   above), 8 + 2 shared channels, striping 2, FIFO 8, INP 5 / WGT 32 per
   core. The 4-array single core buys 30 % (MoE) and 15 % (MLA) over one
   array because these decode tiles are serial chains; four streams buy
   2.1× and 2.9×.
2. **The ring knee is the consumer, not Little's law**: the MoE token tile at
   ring 12 has 50 KB in flight and 25 % channel utilization yet gains nothing
   from ring 16, because per tile the SFU is busy 3331 cycles (VSIGMOID 2048
   at 8 lanes) and the arrays run 6144 array-cycles of accumulation chains.
   `sfu_lanes` 8 → 16 is the next single-stream lever (~−1000 cycles per
   6481-cycle tile).
3. **Arrival-order arbitration has a convoy mode on identical lock-step
   programs**: four MLA copies finish 128 cycles per KV tile apart per core
   index (+18 % chip time over one core alone); a rotating tie-break does not
   remove it. A round-robin grant across cores with pending requests, or
   staggered starts, would.
4. **At 4 cores the chip is request-queue-bound before bandwidth-bound**:
   channels 52 % busy, tile requests wait ~850 cycles because the four
   12-deep rings refill in bursts. The gather window binds on 16 % of bursts
   at 512 outstanding (never at 1 core); keep 512, not §11's 256.
5. **FIFO ≥ r·(d − 1) for a ring of depth d with r release tokens per slot**:
   MLA's K ring over 4 KV tiles (36 tokens on MMA→LOAD) times out
   functionally at FIFO 8 and gains nothing at 32. The compiler should check
   the static edge level against the FIFO when it sizes a ring.
6. **Rings are per pipelined loop**: a ring deeper than the trip count issues
   everything up front and restarts at the next outer iteration (ring 16 =
   trip count is slower than 12). Reaching Little's-law depth needs the
   compiler to flatten the streams it pipelines (`flat_w13`), a program-shape
   decision.
7. **Striping is a single-stream lever**: −5.7 % at 1–2 cores, 0 at 4, and
   8-way striping costs 27 % at 4 cores because every request then occupies
   every channel.
8. **MLA K-split reduction** not done (recorded): moot for a one-array core.

## Calibration additions (to the phase-5 list)

The arbiter's grant policy and per-request overhead; the channel interleave
granularity behind `request_channels`; the gather window (512); `sfu_lanes`
(now on the MoE critical path); the per-core dispatch and FIFO costs with
four streams.

# Phase 7: RTL core v0 and the Verilator conformance harness (toward Gate 2)

Task: `docs/phase7-rtl-core-task.md`. ISA v0.1.5 (Gate 1 frozen) plus
erratum E1 (§16: the MMA k-sum and row-sum orders the array and the VEC
reductions use; the simulator implements it since this phase). Decisions
applied: **SystemVerilog** (Kevin's override of the Chisel default, same
module boundaries); synthesis **both** ways (a Yosys `synth_xilinx` estimate
here, Vivado out-of-context scripts for an x86 machine). Deliverables: J1
`sim/tcta/encode.py` + §12 normative (commit c771593); J2 `rtl/src/` (one
core, 20 modules, lint-clean under `-Wall`); J3 `rtl/test/` (Verilator
harness, nine programs, per-instruction trace in the cycle model's shape); J4
conformance; J5 `docs/calibration-report.md` and the calibrated `KV260`
descriptor; J6 `rtl/formal/` properties and jobs; J7 this section,
`rtl/synth/`. Whole repo: **92 passed**; `make -C rtl/test all`: 9/9 PASS,
also with the formal properties compiled as runtime assertions (`FORMAL=1`).
Code touched outside `rtl/`: `sim/tcta/{isa,machine}.py` (E1 orders),
`sim/tcta/{timing,cycle}.py` (the `rtl_rules` cost model, `KV260` /
`KV260_PLAN`), `tests/test_{sim,cycle}.py`, `model/sweeps.py` (pinned to
`KV260_PLAN`), `isa/isa-v0.1.md` (§16 E1).

## Verdict: PASS-WITH-CAVEATS (J6 and the synthesis half of J7 pending the toolchain)

Per §3 of the task:

- **J1–J4 PASS.** GEMM's DRAM output is bit-exact with the simulator (17
  pages); FA is within §9 (row-max error 4.03e-3 against the fixture, equal
  to the simulator's own, tolerance 2⁻⁷); every instruction event of the
  nine programs retires in the simulator's multiset; the three negative
  tests fault with the simulator's codes (TOKEN_LEAK at FINISH,
  TOKEN_TIMEOUT at cycle 206 with `t_deadlock` 200, UNTOKENED_READ at MMA
  pc 4 under `TOKEN_CHECK=1`). **No program changed to run on RTL**, so no
  erratum beyond E1 was needed; the compiler's GEMM and FA streams ran as
  encoded. The FP library is bit-exact with NumPy float32 on 200 000 random
  pairs per op.
- **J5 PASS-WITH-CAVEATS.** Cycle model vs RTL: GEMM 11286 vs 11291
  (−0.0 %), FA 38642 vs 38543 (+0.3 %), the four micro-programs within
  0.1 %. The 10 % bar is met only after `DeviceTiming` rule changes
  (`rtl_rules`): row-serial DMA, the array's block schedule with per-row
  writeback and no drain overlap, the slice-serial VEC pipe, a 3-cycle front
  end with 16-deep instruction queues, one cycle of PUSH visibility, and the
  as-built scratchpad ports (2 readers + 1 writer per bank, all slots
  sharing the banks). With the planning rules the model is 28–50 % under
  the RTL. Every constant is measured (report §1); the planning descriptor
  survives as `KV260_PLAN`.
- **J6 not concluded (toolchain).** The properties exist (`token_fifo.sv`,
  `scratchpad.sv` under `ifdef FORMAL`), the SymbiYosys jobs are written
  (depth 24, yices), but the distro `yosys 0.54-3` does not load (boost ABI)
  and `sby` is not installed; the prebuilt OSS CAD Suite on the machine was
  not to be executed. Evidence short of a proof: all nine programs run with
  the properties as runtime assertions (`make all FORMAL=1`), no violation
  in 157 000 simulated cycles of seven-client arbitration and token traffic.
  A FAIL condition (a counterexample on `TokenFifo`) has therefore not been
  ruled out formally; nothing points at one.
- **J7 synthesis not run.** `rtl/synth/yosys_xck26.ys` needs the same
  Yosys plus `yosys-slang`; `rtl/synth/vivado_ooc.tcl` + `.xdc` (part
  `xck26-sfvc784-2LV-c`, 200 MHz) are ready for an x86 machine with Vivado.
  Fmax and LUT/DSP/BRAM/URAM against the budget are open; the design has
  single-cycle binary32 add/multiply paths (mma_pe, the VEC lanes) and
  32-entry combinational SFU interpolation, which is where 200 MHz will be
  decided.

## The numbers in one table (v1 KV260 core, RTL cycles at the harness's 24-cycle memory latency)

| | RTL | planning model | calibrated model | what the gap is |
|---|---|---|---|---|
| GEMM 50×100·100×96 (compiler program) | 11291 | 5633 (−50 %) | 11286 | loads 6710 of 11291: row-serial DMA at 5 B/cycle |
| FA, 4 KV tiles of 64 (compiler program) | 38543 | 27856 (−28 %) | 38642 | nine K/V/Q loads 21933 cycles on the LOAD queue; VEC chain 4581 per tile (plan 4156) |
| LOAD.2D 64×128 BF16 (16 KB) | 3205 | 1124 | 3205 | 50 cycles per row = 24 latency + 16 beats + 8 slices + 2 |
| STORE.2D 64×128 f32 (32 KB) | 3140 | 2148 | 3140 | 49 per row = 2·8 slice reads + 32 beats + 1 |
| MMA 64×128×128 / accumulate | 4152 / 4184 | 4104 | 4152 / 4182 | 32 blocks × 128 + drain 56 / 88 |
| MMA accumulate 64×64×64 (FA PV) | 1231 | 1032 | 1257 | writeback-bound: 4 cycles per row at P = 64 |
| VEC op on `[64, 128]` / with a column-vector operand | 518 / 582 | 512 | 518 / 582 | 1 slice per cycle; +1 read per row for the vector |
| VEC op on a `[64, 1]` vector / with a vector operand | 70 / 134 | 4 | 70 / 134 | a column vector is one element per row (column banking) |
| VEXP2 `[64, 1]` / VEXP2.SUM `[64, 64]` | 389 / 1927 | 16 / 3072 | 390 / 1926 | SFU 6 cycles per slice, one in flight; the sum is fused |
| front end | 3 cycles per word, PUSH/POP words too; ext word 2 + entries; LOOP/ENDLOOP 2; LOOP r 1 + ⌈R/step⌉ after VCOUNT | 1 / free / 2 / 4 | as measured | never on GEMM's or FA's critical path |

## Findings for the main session

1. **The v0 DMA engine is row-serial**: every row pays the memory latency,
   so a tile load runs at 3.4–5.1 B/cycle against the planned 16. It is the
   whole GEMM gap and most of FA's. Phase-8 RTL item one: prefetch the next
   row while the current one waits (≈ 1.6–2×), or pipeline rows fully
   (beats-bound, the plan). The cycle model's planning DMA rule is right
   for that machine, not for this one.
2. **Column vectors cost M cycles on this banking.** §11's `bank = column
   mod 16` puts a `[M, 1]` vector in one bank, so the lanes cannot help; FA's
   per-row statistics (VROWMAX result, the running max, the row sums, their
   exps and reciprocals) are 70–134 cycles per op instead of 4. ISA v0.2
   candidate: hold per-row statistics as row vectors (`[1, T_VR]`, which the
   VEC row slots already are) with the reductions writing transposed, or a
   column-mode slot map (row mod 16) for the `[M, 1]` slots.
3. **The SFU is 6 cycles per slice, not 4**: one slice in flight through the
   4-lane pipe. Pipelining slices is −33 % on every SFU op; VEXP2.SUM is 42 %
   of FA's VEC time on the RTL. The fused VEXP2.SUM (one pass) is what the
   plan should assume for `sfu_fused_sub_sum` on the v1 core.
4. **§8.5 holds at the queue, not at the front end**: PUSH and POP take no
   issue slot (a queue retires its leading syncs in the ISSUE cycle of the
   next work instruction, measured), but each is a stream word the front end
   spends 3 cycles on. Worth one sentence in §8.5 ("sync instructions cost no
   issue slot; their dispatch cost is implementation-defined").
5. **Accumulate MMAs with K ≤ 64 are writeback-bound** (+14 % on FA's PV):
   the read-modify-write takes 4 cycles per result row against the array's
   4 cycles per row at P = 64. Pipelining the old-value read with the
   previous row's write makes it 2.
6. **The token checker needs issue-time operand events.** The simulator
   raises UNTOKENED_READ when an instruction *names* an untokened tile; the
   RTL's MMA, seeing a zero-shaped WGT tile, never read its data, so a
   checker fed by scratchpad accesses alone saw nothing. `token_check.sv`
   now also takes the operand tiles at issue; this is a statement about §8.4
   worth making explicit: the check is on the operand reference, not on the
   bytes.
7. **ISA details the RTL settled** (recorded, no spec change): `LOAD.IDX` on
   block-scaled formats and `STORE.2D f4.r16` are BAD_QUALIFIER in v0; `MMA`
   with `K = 0` is BAD_SHAPE; SEND/RECV are BAD_OPCODE (COMM stub); the
   `np.maximum` ±0 tie rule (`+0`) is what `f32_max` implements, so VROWMAX
   is bit-exact.
8. **Toolchain**: formal and synthesis wait on a working Yosys (the distro
   package is broken against boost 1.92; `sby` is not packaged). The jobs
   and scripts are ready; `make all FORMAL=1` is the simulation evidence.

## Calibration (what the plan should now carry for the v1 core)

`tcta.timing.KV260` is the measured core; `docs/calibration-report.md` §1
lists every value with the trace it came from and §2 what still differs
(the accumulate handshake, ±1-cycle per-instruction overheads, cycle- vs
instruction-granular port arbitration, the gather and VTOPK formulas taken
from the FSMs, the harness's 24-cycle memory latency). `f_mhz` stays 200
until synthesis says otherwise. Phase 8 (MoE/MLA/KDA on RTL) measures the
gather path, VTOPK, VGATE/VMASK/VSEL and the dequant formats.
