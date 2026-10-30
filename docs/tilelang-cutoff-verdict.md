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

## Verdict: PASS-WITH-CAVEATS (conformance and calibration pass; the v0 core as built does not fit the KV260)

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
- **J6 PASS.** SymbiYosys BMC (smtbmc + yices, OSS CAD Suite Yosys
  0.70+221), depth 24 ≥ the required bound, both jobs `DONE (PASS)`:
  `token_fifo` (DEPTH 8; 6 assertions: count = pushes − pops, count ≤
  DEPTH, no accepted push when full / pop when empty, no dropped push into a
  non-full FIFO, no refused pop from a non-empty one) and `bank_arbiter`
  (`scratchpad` with WORDS 16, 18 assertions: one writer per cycle, at most
  one port per client, the highest-priority requester granted, two pure
  reads both granted). Two harness-side fixes were needed, no RTL change: the
  FIFO's first run failed at step 0 on the unconstrained initial flops (no
  reset had been applied), so the properties now assume reset in the first
  cycle and check from the second (`f_past_valid`, the usual idiom); the
  arbiter job reads the sources through the `slang` plugin because Yosys's
  own parser stops at the packages. The same properties still pass as
  runtime assertions on the nine programs (`make all FORMAL=1`).
- **J7 synthesis: estimate done, FAIL against the KV260 budget.** Yosys
  `synth_xilinx` (UltraScale+ cells, no placement, no timing), per module
  with the children blackboxed and the hierarchy summed
  (`rtl/synth/yosys_ooc.py` → `rtl/synth/yosys_xck26_estimate.md`; the
  whole-design script needs more than this machine's 16 GB): **1.30 M LUTs
  against 117 k (11×)**, 79 k FFs (34 %), 546 DSP48E2 (44 %), scratchpads in
  block RAM at 2× the 122-RAMB36 bit-minimum because Yosys does not merge a
  port's read with its write (144 on the part; Vivado merges). Where the LUTs
  are: `dma_store` 768 k (59 %: the byte-addressed 2 KB row buffer is 16 k
  flops each behind a 64-source write mux), `mma_unit` 290 k (22 %: 256 PEs
  × 929 LUTs, of which a single-cycle binary32 adder is 643 and the BF16
  multiplier 253 + 1 DSP), `vec_unit` 123 k (95 binary32 adders, 36
  multipliers, 47 max units and the four SFU lanes, each 4.8 k), `dma_load`
  47 k (the beat-indexed row buffer and its read window, 16 dequant paths),
  the five scratchpads 41 k of port muxing, the five queues 16 k. Even with
  `dma_store` rewritten around a RAM, the arithmetic alone (MMA + VEC, 413 k)
  is 3.5× the fabric: a 16×16 output-stationary array with binary32
  accumulators in LUTs does not fit a K26. No Fmax: Yosys gives no timing and
  Vivado (`rtl/synth/vivado_ooc.tcl`, part `xck26-sfvc784-2LV-c`, 200 MHz)
  needs an x86 machine; the single-cycle f32 add (643 LUTs, 37 CARRY8) is the
  path that will decide it. One RTL change came out of this: the scratchpad
  bank was rewritten for block-RAM inference (one process per port, a read
  register per bank and port, `no_rw_check`; the earlier form mapped to
  7 680 LUT-RAM cells + 34 k LUTs for the ACC pad alone); all nine programs
  and the arbiter proof pass unchanged on it.

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
8. **The v0 core is ~11× the KV260's LUTs; three structures account for it.**
   (a) `dma_store`'s byte-addressed row buffer (759 k LUTs): the slice
   converters write up to 64 bytes per cycle at an arbitrary byte offset into
   2 048 byte registers. The fix is a beat-organized buffer written one
   aligned 16-byte beat per cycle (as `dma_load` already does, 28 k) or
   streaming the converted slice straight to the bus. (b) The array: 256 ×
   (binary32 adder 643 + BF16 multiplier 253 + 33) = 238 k. UltraScale+ has
   no floating-point DSP, so an f32 accumulator costs ~650 LUTs per PE; the
   options are a smaller array (8×8 = 60 k), BF16/fixed-point accumulation
   with periodic f32 conversion, or a 2-cycle pipelined adder shared by two
   PEs at half the rate. This is a plan-level decision: `mac_rows/cols` 16
   with f32 accumulate was the phase-5 assumption. (c) `vec_unit` keeps a
   separate adder, subtractor, multiplier, max and gate path per lane (95
   adders, 61 k LUTs): one shared adder and multiplier per lane behind an
   opcode mux is 16 of each (~20 k). Independent of the fit, the scratchpads (4.4 Mbit) are 85 %
   of the part's block RAM; the two larger ones (WGT 1 Mbit, ACC 1.5 Mbit)
   belong in URAM.
9. **Toolchain**: the distro Yosys is broken against boost 1.92 and `sby`
   is not packaged; formal and the Yosys estimate ran from the OSS CAD Suite
   (`~/.local/opt/oss-cad-suite/bin`, Kevin's go-ahead). The SystemVerilog
   sources need the `slang` frontend in Yosys (packages with typed
   localparams); yosys-slang flattens by default and uniquifies per instance
   with `--keep-hierarchy`, so the whole core needs > 16 GB and the estimate
   is per module (`yosys_ooc.py`), with `dma_store` mapped by a short abc
   script because the default abc9 call did not finish in an hour on it.

## Calibration (what the plan should now carry for the v1 core)

`tcta.timing.KV260` is the measured core; `docs/calibration-report.md` §1
lists every value with the trace it came from and §2 what still differs
(the accumulate handshake, ±1-cycle per-instruction overheads, cycle- vs
instruction-granular port arbitration, the gather and VTOPK formulas taken
from the FSMs, the harness's 24-cycle memory latency). `f_mhz` stays 200 as
a target: Yosys gives no timing and the Vivado run is pending an x86
machine, so the cycle counts above are not yet seconds. Phase 8 (MoE/MLA/KDA
on RTL) measures the gather path, VTOPK, VGATE/VMASK/VSEL and the dequant
formats.

## Synthesis estimate (Yosys synth_xilinx, UltraScale+ cells, per module, no timing)

| | LUT | FF | DSP48E2 | RAMB36 (Yosys / bit-minimum) |
|---|---|---|---|---|
| **tcta_core** | **1 296 593** | 78 685 | 546 | 299 / 122 |
| `dma_store` (byte row buffer) | 767 643 | 17 532 | 26 | |
| `mma_unit` (256 PEs × 929 + mask/accumulate 32 adders, 16 multipliers) | 290 180 | 32 299 | 342 | |
| `vec_unit` (16 lanes, 4 SFU lanes) | 122 507 | 4 519 | 110 | |
| `dma_load` | 46 758 | 1 669 | 65 | |
| scratchpads INP/WGT/ACC/VEC/STATE (port muxing) | 40 789 | 225 | | 299 / 122 |
| queues ×5, front end, tile_meta, token net, core glue | 28 716 | 22 441 | 3 | |
| KV260 (XCK26) | 117 120 | 234 240 | 1 248 | 144 (+ 64 URAM) |

Leaf cells: `f32_add` 643 LUT / 37 CARRY8, `f32_mul` 541 + 2 DSP, the BF16
`f32_mul` of the PE 253 + 1 DSP, `f32_max` 123, `sfu_lane` 4 847 + 9 DSP,
`mma_pe` 929 + 100 FF, `queue` 3 184 + 3 969 FF, `token_fifo` 9 + 4 FF.

---

# Phase 8 K0 (fit): verdict (2026-10-29)

Per `docs/phase8-amendment-fit.md`: three fit fixes, each its own commit
with the per-module Yosys number before / after, then the fit report.

## Verdict: K0.1–K0.3 DONE, K0.4 FAIL on fit (2.1× the fit budget; 6.2× better than phase 7)

- **K0.1 `dma_store` / `dma_load` row buffers** (71e9c4d): beat-banked
  LUT-RAM written one converted slice per cycle, read one beat per cycle.
  `dma_store` 767 643 → 7 057 LUT (target < 5 k for the engine: the 2 096
  own LUTs meet it; the rest is the 16 f32→i32 converters of i32 stores),
  `dma_load` 46 758 → 18 082. A **pre-existing bug** came out with the new
  `gather` program (LOAD.IDX f32 with −1 rows, bf16 and f4.r16 gathers as
  MMA operands, STORE.IDX, i32 STORE.2D): both engines read the index slice
  with the stale `idx_cg` (−1 on the first row), so every gathered row used
  index 0; none of the nine phase-7 programs touched the IDX paths. Ten
  programs since.
- **K0.2 `vec_unit` one ALU per lane** (5055321 + follow-up aba825d):
  one adder, one multiplier, one BF16 rounder per lane, op-muxed; max is
  the adder's subtract path; a shared four-cycle row reducer on 8 adders;
  the sigmoid chain and VGATE decay on the lane ALUs during the SFU steps.
  122 507 → 53 581 LUT (target < 25 k **not met**: 24 adders at 643 LUTs
  and 16 multipliers at 541 are 24 k by themselves, the operand muxes 19 k).
  The reducer's first pacing rule blocked a row's last slice while the
  reducer was busy and cost 3 cycles per row on 4-slice rows — found only
  when the calibration was rerun for K0.3 (`VROWMAX 64×64` 456 instead of
  263, FA −2.6 % vs the model); the follow-up issues the last slice when
  the reducer frees by the time it returns (267, the +4 being the reducer's
  levels; `VEXP2.SUM` +1 per row for the partial add on the lane ALUs).
  Lesson recorded: rerun `calibrate.py` with every K0 commit, not only
  the harness.
- **K0.3 `mma_unit` accumulator: option B taken** (0d2735f), bit-exact
  within the day, so the decision rule did not fall back to A. The window
  is not the per-cell running window the amendment sketched (that needs a
  shifter and a renormalisation lag in every cell and truncates operands
  product by product; measured 28 ulp mean) but a **per-block exponent
  bound**: the operand fill records the largest exponent of every A row and
  B column, each A significand is shifted once per row (`ma · 2^(15−da)`)
  and each B significand once per column into two 9-binade levels, so a
  cell is two DSP48E2 signed 24×18 MACs (`P = (first ? 0 : P) + M`) with
  48-bit sums and ~7 LUTs of sticky NaN/inf flags, and the K products of an
  output are summed exactly; one rounding to binary32 per element in a
  converter (8 lanes, 2 cycles per row) with the block's maxima as the
  exponent. The simulator's `acc_mode="fixed48"` (`DEVICE_KV260`) models it
  exactly; the harness is bit-exact to it on GEMM, the MMA forms and the
  gather test, FA within §9 against the fixture (row-max error 4.03e-3,
  unchanged). **Erratum E5** added to `isa/isa-v0.1.md` §16 with the §9
  note (the format is implementation-defined within §9's tolerance; the
  simulator exposes both modes). Accuracy against E1's sequential binary32
  sums (`tests/test_sim.py`, 64×128×128): 0.5 ulp on same-magnitude data,
  bounded by `2^-23 · 128 · max|A row| · max|B col|` in general (A spanning
  2^±20: mean 216 ulp = 7.7e-7 of the bound vs binary32's own 6.1e-7;
  softmax·V 52 ulp), specials / subnormals / overflow identical. Cost:
  `mma_unit` 290 180 → 51 781 LUT + 896 RAM32M16, DSP 342 → 537 (2 per
  cell), FF 32 k → 29.6 k. Timing: operands are block-buffered, so the
  first block starts `max(16·⌈K/16⌉, K) + 3` cycles after issue and blocks
  go every `max(K, 16)` (no `ceil16`): MMA 64×128×128 4 152 → 4 284 (+3 %),
  GEMM 50×100×96 2 718 → 2 571 (−5 %), FA QKᵀ 1 080 → 1 148, PV 1 231 →
  1 299; the cycle model replays the unit cycle by cycle and matches every
  shape exactly (`docs/calibration-report.md` §1.2).
- **K0.4 fit report** (`docs/phase8-k0-fit-report.md`): **200 006 LUT +
  1 016 RAM32M16 = 208 134 LUT-eq, 178 % of the K26's 117 k and 2.1× the
  99.5 k fit budget (15 % headroom)**; 59 878 FF (26 %), 624 DSP (50 %),
  block RAM 2× the bit-minimum as Yosys maps it (122 RAMB36 minimum; port
  merging or URAM for ACC/STATE makes it fit). Where it is: `mma_unit` 59 k
  (28 %: 26 k of own logic — lane collectors, shifters, writeback — 10 k of
  f32 adders, 5 k converters, 4.4 k + 7 k LUT-RAM of operand buffers; the
  256 cells are 1.8 k), `vec_unit` 54 k (26 %), scratchpad port muxing 41 k
  (20 %), `dma_load` 18 k, the five instruction queues 16 k, `dma_store`
  7 k. **The fallback as stated in the amendment (8×8 array) does not fit
  either**: the array's cells are 1 % of the core; what scales with the
  array is the collectors and operand buffers (−35 k), while VEC lanes,
  scratchpad ports, DMA converters and queues are sized by `BANKS = 16`.
  Report §4 lists the levers: L1–L4 (queues in LUT-RAM, scratchpad clients
  per the ISA access table, 8 writeback adders / 4 converter lanes, operand
  buffers in RAMB18: −43 k, no speed cost) bring the 16-lane core to
  ~165 k; with the 8×8 array ~130 k; the configuration that fits with
  headroom is 8 lanes throughout (`BANKS = 8`: ≈ 105–110 k before Vivado's
  typical 20–35 % gain over Yosys). Recommendation: the x86 Vivado run
  first, L1–L4 in any case, then decide the lane width.

## The numbers in one table (RTL cycles, harness memory latency 24)

| | phase 7 | after K0 | calibrated model | note |
|---|---|---|---|---|
| GEMM 50×100·100×96 | 11 291 | 11 144 | 11 141 | MMA 2 718 → 2 571 (P = 100, not 112) |
| FA, 4 KV tiles of 64 | 38 543 | 39 371 | 39 370 | MMA.T / PV +68 each (operand fills), reductions +4, VEXP2.SUM +1 per row |
| MMA 64×128×128 / accumulate | 4 152 / 4 184 | 4 284 / 4 316 | 4 284 / 4 316 | first fill K + 3, then 32 × 128 + drain 57 / 89 |
| VROWMAX 64×128 / 64×64 | 519 / 263 | 523 / 267 | 522 / 266 | shared reducer |
| VEXP2.SUM 64×64 | 1 927 | 1 994 | 1 994 | partial add on the lane ALUs |
| core LUT-eq (Yosys) | 1 297 865 | 208 134 | | KV260 117 120; fit budget 99 552 |

## Findings for the main session

1. **The fit target needs a lane-width decision, not only the array
   decision.** With `BANKS = MAC_COLS = VEC_LANES = 16` fixed in
   `tcta_pkg.sv`, three quarters of the remaining LUTs are lane-proportional
   logic outside the array. An 8×8 array is not a parameter change today
   (writeback half-slices, B buffer lanes); an 8-lane core is the
   configuration that fits with headroom by the Yosys estimate. The
   16-lane core remains the planning machine; the KV260 is the flow proof.
   *Update after phase 8 L1–L2 (measured −44 k, 164 k LUT-eq):* the 8×8
   array on the 16-lane core projects to ~114 k with L3–L4, inside Vivado's
   usual gain over Yosys, so the fallback is no longer ruled out; the lane
   decision waits for the Vivado number (`docs/phase8-k0-fit-report.md` §4a).
2. **Vivado is the decisive number.** Yosys's `synth_xilinx` here runs
   with `-nowidelut`, no cross-module optimization and no LUT combining;
   its mux-heavy results (collectors, scratchpad ports, operand muxes) are
   the kind Vivado maps 20–35 % smaller. Until the x86 run exists every
   fit statement above is an upper bound.
3. **Erratum E5 as written.** The accumulation format is
   implementation-defined within §9; the simulator exposes `acc_mode` f32
   (E1, ASIC) and fixed48 (v1 FPGA). The RTL's conformance reference is
   the fixed48 simulator (bit-exact); kernels are still judged against the
   PyTorch fixtures with the §9 tolerance (FA passes with the same row-max
   error as the f32 simulator). Note for the ISA copy in this repo: §16
   holds E1 and E5 only; E2–E4 were never delivered here, the number
   follows the amendment.
4. **The fixed48 window is a block-level bound, not a running window.**
   Design notes for the ASIC option or a later revision: a running window
   per cell needs a barrel shifter and a renormalisation lag per cell and
   loses up to 28 ulp mean on mixed-magnitude data; truncating products
   instead of operands fixes the accuracy but is not expressible in a
   DSP48E2; one B level gives 2^-17 relative error (mean 28 ulp); two
   levels at 2 DSPs per cell give ≤ 2^-23 of the bound; combining the two
   levels inside the DSP needs a 17-bit relation the ports do not allow. A
   known case: a row that cancels exactly in the window (`[1e8, −1e8, 1, 1,
   3e-5] · 1`) returns 0 where binary32's sequential sum returns 2.00003;
   both are within the bound, neither is the exact 2.00003.
5. **MMA timing changed shape.** Block-buffered operands add a first-block
   fill (`max(16·⌈K/16⌉, K) + 3` cycles) but remove `ceil16(K)` from the
   block period; `MMA.T` with K not a multiple of 16 is fill-bound at
   `16·⌈K/16⌉` per block; K > 128 faults (BAD_SHAPE: the window's headroom
   assumes ≤ 128 products). The planning rule `mma_occupancy` is unchanged.
6. **A pre-existing gather bug and a reducer regression were both found
   by coverage that phase 7 lacked** (no IDX program; no calibration rerun
   per commit). The `gather` program and `calibrate.py` are now part of
   every K0 commit's check list.
7. **Toolchain.** Yosys infers DSP48E2 for the multiply but leaves the
   accumulate and its clear in LUTs (48 LUT3 + 12 CARRY per cell), so the
   cell instantiates the primitive explicitly under `TCTA_DSP48E2` with a
   blackbox stub generated from Yosys's `cells_xtra.v`; the behavioural
   model is used by Verilator. yosys-slang's distributed-RAM mapping is a
   32 × 14-bit SDP model, so 16-bit lanes take two `RAM32M16` each and a
   16-port read replicates the RAM 16× (the A buffer's 304 cells); LUT-RAM
   is reported in its own column and must be counted as 8 LUTs per cell.
8. **Not delivered to this repo**: the K1–K8 task text ("as written" in
   the amendment refers to a phase-8 task document this session never
   received; only the amendment arrived). Fmax still waits for Vivado.


# Phase 8 L1–L4 (fit levers): verdict (2026-10-29)

Per `docs/phase8-gate2-task.md` §0.2 (the full phase-8 text, which also
closed K0): L1–L4 first, each its own commit with the per-module Yosys
number before / after, a `calibrate.py` rerun per commit, no speed cost
except L3's stated +1 cycle per accumulated row; the go message added the
simulator's E6 K check to L1 and a re-run of the "8×8 does not fit" line
after L2.

## Verdict: L1–L3 DONE (−49.7 k, 208 k → 158 k LUT-eq), L4 NOT TAKEN (analysed: no net win)

| step | commit | module(s) before → after (LUT) | core LUT-eq | of the K26 | of the fit budget | timing |
|---|---|---|---|---|---|---|
| after K0 | 9d34f96 | | 208 134 | 178 % | 209 % | |
| L1 queues in LUT-RAM | e88d321 | `queue` ×5: 3 184 + 3 969 FF → 316 + 36 RAM32M16 + 112 FF | 195 234 | 167 % | 196 % | unchanged |
| L2 scratchpad clients per the access table | f4afb57 | pads 40 789 → 14 506; glue 8 361 → 3 592 | 164 182 | 140 % | 165 % | unchanged |
| L3 writeback on 8 adders | a8244a2 | `mma_unit` 51 781 → 46 031 | 158 432 | 135 % | 159 % | +1 cycle per accumulated / mask-added row: FA +1.4 % |
| L4 operand buffers in block RAM | — | not built | | | | |

- **L1** (e88d321): the 16 × 241-bit instruction buffer of each queue is a
  distributed RAM with two asynchronous read heads and no reset; −12.9 k,
  at the estimate. The simulator now faults MMA with K = 0, or K > 128
  under `fixed48`, as the RTL's `FIX_K_MAX` check (erratum E6; test added).
- **L2** (f4afb57): `tcta_pkg::SP_READERS / SP_WRITERS` encode the ISA's
  access table (§1, §2.2, §5): 12 reading and 12 writing (pad, client)
  pairs of the 35; a pad's arbiter covers their union, its write-data muxes
  only the writers, its response path (and the core's response mux) only
  the readers; the request record carries one word address instead of 16
  identical per-bank addresses; a request outside the table is a simulation
  error. The bank-arbiter proof runs with all seven clients enabled
  (PASS). **−31.1 k against the −15 k estimate**: the estimate counted
  clients, the table separates readers from writers.
- **L3** (a8244a2): the mask op and the old-accumulator add each take a
  row in two 8-lane halves (new `W_ADD` state): 16 → 8 `f32_add`, −5.75 k.
  Cost as stated: +1 cycle per accumulated or mask-added row (MMA
  accumulate 4 316 → 4 332, FA PV 1 299 → 1 436, FA 39 371 → 39 919). The
  4-lane converter (−2.5 k) was built and measured: at P = 64 the converter
  saturates (16 rows × 4 cycles) and FA's QKᵀ went 1 148 → 1 327 (+15.6 %),
  outside the stated trade (P < 64); the converter keeps 8 lanes and the
  2.5 k is listed as an option. The cycle model (`mma_cv_row_cycles`,
  `mma_wb_acc_extra` 3, `mma_wb_mask_extra` 4) is exact on every shape.
- **L4** (`docs/phase8-k0-fit-report.md` §4b): the K0 estimate (16 RAMB18,
  −8 k) sized the buffers by capacity. They are a corner turn (whole row
  slices written, one element per row read, each row at its own k): a
  block-RAM version needs lane-major memories split by buffer-half parity,
  64 RAMB18 (32 RAMB36-eq) plus ≈ 8 k LUTs of lane permutation, for a net
  ≈ −3 to −4 k LUT-eq — on a part whose block RAM (122 RAMB36 minimum for
  the pads, 144 available, 299 as Yosys maps them) is the tighter resource.
  Not taken; reopened only if ACC / STATE move to URAM (task §0.4) and
  Vivado's block-RAM count leaves ≥ 32 spare.

Checks per commit: lint, `make -C rtl/test all` 10/10 (also `FORMAL=1`
after L3), `calibrate.py` within 0.1 % on every program, `pytest tests -q`
95 passed, `sby` bank arbiter PASS.

## Findings for the main session

1. **158 k LUT-eq after L1–L3** (135 % of the K26, 1.59× the fit budget):
   the 16-lane core with L1–L3 fits only if Vivado maps ≥ 37 % under Yosys;
   the 8×8 array on the 16-lane core (L5, −35 k by the K0 estimate, not
   re-measured) projects to ~123 k and needs ≥ 19 %, inside the usual
   20–35 %; the 8-lane configuration fits by Yosys alone. The decision rule
   of task §0.3 stands; the Vivado number decides it.
2. **Where the 158 k is now**: `mma_unit` 46 k (own 25.4 k: collectors,
   shifters, writeback; 5.1 k adders, 4.3 k multipliers, 5.0 k converters,
   1.8 k cells, 4.4 k + 7.2 k LUT-RAM-eq operand buffers), `vec_unit`
   53.6 k (now the largest module: 24 adders + 16 multipliers 24 k, operand
   muxes 19 k), `dma_load` 18 k, pads 14.5 k, `dma_store` 7 k, glue 3.6 k,
   queues 3 k, front end + meta + tokens 4.4 k. The next levers of the same
   kind are in `vec_unit` (operand muxes) and `dma_load` (16 dequantizers).
3. **The access table is now hardware.** A new instruction that reads or
   writes a pad outside ISA §1 / §2.2 / §5 needs `SP_READERS / SP_WRITERS`
   extended (one line) or it is a simulation error; the arbiter proof
   covers the full seven-client case.
4. **L3's converter lesson for K3**: the writeback pipeline is at its
   throughput limit at P = 64 (FA's K); K3's RMW pipelining should take the
   whole row through the pipeline (old read ‖ previous write, adder halves
   ‖ next row) rather than shorten states, and the 4-lane converter should
   stay out until P < 64 shapes stop mattering.
5. **Estimates of mux-heavy logic err both ways**: L2 came in at 2× its
   estimate (the readers / writers split), L4 at a loss (ports, not bits).
   The per-module `--only` flow (2–4 min per module) makes measuring
   cheaper than estimating; the fit report's remaining L5–L7 are still
   estimates.

# Phase 8 K1 (DMA row pipelining and the two formats): verdict (2026-10-29)

Per `docs/phase8-gate2-task.md` §1 K1: `dma_load` / `dma_store` keep ≥ 2
rows in flight (`DMA_ROWS_IN_FLIGHT`), tile loads reach ≥ 14 B/cycle on the
64×128 BF16 tile at the harness's 24-cycle latency (K0: 5.1); `STORE.2D
f4.r16` (quantise on egress, Appendix A) and block-scaled `LOAD.IDX`
(`f4.r16` KV rows; `f4.b32` rows with their scale slice) implemented,
closing E6's two gaps; the `gather` program extended to cover both. The
main session's E7 (the access table is normative hardware) is in the same
commit, as asked.

## Verdict: K1 DONE — 14.5 B/cycle (≥ 14), both formats bit-exact, FA −19.6 %

| | K0 (row-serial) | K1 (row-pipelined) |
|---|---|---|
| LOAD.2D bf16 64×128 | 3205 cycles, 5.1 B/cycle | **1127, 14.5 B/cycle** |
| LOAD.2D bf16 64×64 (FA K/V) | 2437, 3.4 | 611, 13.4 |
| STORE.2D f32 64×128 | 3140, 10.4 | 2126, 15.4 |
| GEMM 50×100·100×96 | 11144 | **5917 (−46.9 %)** |
| FA (4 KV tiles) | 39919 | **32109 (−19.6 %)** |
| mma / vec / dma_copy / loop | 30512 / 22317 / 4560 / 814 | 23230 / 14910 / 2228 / 489 |
| gather | 27364 (K0.1, old program) | 22683 (K1 program: + f4.b32 gather, 3 f4.r16 stores) |
| `dma_load` / `dma_store` (Yosys) | 18 402 / 7 377 LUT-eq | 18 425 / 11 418 LUT-eq |
| core | 158 432 LUT-eq (135 % of the K26) | 162 496 (139 %) |

- **Structure** (`rtl/src/dma_load.sv`, `dma_store.sv`): each engine is a
  front and a back end around the K0.1 row buffer turned into a ring (128
  beats, LUT-RAM, rows at any beat offset) with eight row records. LOAD's
  front resolves a row (index, address, range check), allocates its record
  and ring space, issues its beats at one per cycle and moves on; the
  responses land in the ring in order (scale beats in a per-record side
  memory); the writer takes complete records (a cumulative beat count) and
  writes one dequantised slice per cycle, zero slices for −1 rows. STORE's
  front reads one slice per cycle (the response is in the ring the next
  cycle) and the writer streams complete rows as beats. A row costs
  `max(beats + 1, slices + 1)` (LOAD) / `max(slices + 1, beats + 1)` (STORE)
  in steady state; the memory latency is paid once per instruction. Rows
  in flight: 8 records, or the ring (8 × 256 B rows, 4 × 512 B, 1 × 2048 B);
  three rows cover the 24-cycle latency on 256 B rows, so the engine is
  beat-bound on every family tile. The slot/ring limit binds only on rows
  shorter than ~4 beats (`loop`'s 1×8 f32 load: 35 cycles, model 36).
- **The two formats.** `LOAD.IDX` on `f8.b32` / `f4.b32` takes the row's
  scale codes from the grid at row `idx / 32` (the simulator gained the
  same, `machine._op_LOAD_IDX`); scale codes now index from the row's
  column 0 for every block-scaled format (Appendix A), so any `col0` works —
  the simulator's `LOAD.2D` on `*.b32` at a `col0` that is not a multiple of
  32 followed the window instead of the row before (unreachable by the
  compiler's programs, which tile at 128; now Appendix-A-exact in both).
  `STORE.2D f4.r16` quantises on egress bit-exact to
  `dtypes.encode_f4_r16_row`: the block scale is the E4M3 nearest of
  RN(amax / 6) with the simulator's tie rule (ties to the smaller
  magnitude; 1.0 for an all-zero or NaN block; 448 at the top; the divide
  by 3 is an exact 26-bit integer division), the codes are the E2M1 nearest
  of RN(x / scale) decided without a divider: with scale = S · 2^e (S the
  4-bit E4M3 significand) and midpoints M_t / 4, RN(x / scale) > M_t / 4 iff
  X · 2^(ex − e − 124) > M_t · S · (2^24 + 1), one shifter and seven 34-bit
  comparators per lane; NaN / zero → 0, inf and x / 0 → ±0 with the sign
  from `x < 0`. Four lanes, nine cycles per slice, a one-beat assembler for
  the nibbles, the scale bytes and the zero padding to 64 B.
- **Cost.** `dma_load` unchanged (18.4 k: the ring and the records replace
  the one-row buffer and its wait states); `dma_store` +4.0 k for the
  `f4.r16` path (the quantiser lanes, the amax tree, the assembler, the
  scale-byte registers). Core 158.4 k → 162.5 k LUT-eq (139 % of the K26).
  The path is cold on every family; trimming it (two lanes, sharing the
  `f32_to_bf16` / `f32_to_i32` converters with the plain path) is a K8 item.
- **Cycle model** (`sim/tcta/cycle.py::_rtl_dma_cycles`, `timing.py`):
  the two-stage structure with `dma_load_row_overhead` 1,
  `dma_store_row_overhead` 1, `dma_tail_cycles` 1, `dma_rows_in_flight` 8,
  `dma_store_r16_slice_cycles` 9, `dma_idx_slice_cycles` 3 (the K0
  `dma_store_slice_cycles` is gone); gathers charge the column window, the
  index slices and the skipped rows on the front and the first index slice
  before the back end starts. Every DMA row of the seven programs is
  reproduced exactly except two: `LOAD.IDX f4.r16` 1195 vs 1193 and the
  one-row 1×8 load 35 vs 36. Totals within 0.2 % on every program; the
  planning model is now within 5 % on GEMM and 13 % on FA.
- **Bugs found on the way** (both by the bit-exact harness, both fixed
  before the commit): a SystemVerilog width rule — `int'(a − b)` on 8-bit
  ring positions widens the operands before the subtraction, so the
  occupancy went negative at the first wrap and a row was allocated into
  space still being written (rows 4–10 of a 64×128 f32 store held rows
  8–14); and the `f4.r16` lane thresholds built as `{ms, 25'd0}` (2^25, not
  2^24), which halved every code while the scale bytes stayed right.

Checks: lint clean; `make -C rtl/test all` 10/10 PASS (also `FORMAL=1`);
`calibrate.py` within 0.2 % on all seven programs; `pytest tests -q` 95
passed (the duration test updated to the K1 numbers); the bank-arbiter
proof is untouched by K1 (no scratchpad change).

## Findings for the main session

1. **The DMA is no longer the gap.** With rows pipelined the planning
   model's DMA rule (one request, latency once, 16 B/cycle) is within one
   cycle per row of the RTL; GEMM is 5917 cycles (planned 5633), FA 32109
   (planned 27856, the rest is the VEC vectors and the SFU: K2 / K3).
2. **Latency is now paid once per instruction**, so the harness's 24-cycle
   memory model matters less: a real LPDDR latency (say 100–200 cycles) adds
   that once per LOAD, not per row, as long as rows in flight × beats per
   row ≥ latency: 8 records × 16 beats = 128 cycles of cover on 256 B rows.
   For a 200-cycle memory the records would have to go to 16 (the ring to
   4 KB) to stay beat-bound; a parameter, not a redesign.
3. **The access table held** (E7): the new engines added no (pad, client)
   pair; the LOAD client still writes INP / WGT / ACC / VEC and reads VEC,
   the STORE client reads ACC / VEC.
4. **`f4.r16` egress costs 4 k LUTs for a cold path**; before K8, decide
   whether the KV-write path stays four lanes or drops to two (≈ −1.5 k).
   `LOAD.IDX` on block-scaled rows cost nothing: the scale-grid address is
   the 2-D one with `idx` in place of `row0 + i`.
5. **The width rule bit once and is worth a lint rule**: a sized cast
   around a difference of narrow counters (`int'(a − b)`) silently widens
   the operands. Both engines now size the difference first; the pattern
   does not occur elsewhere in `rtl/src` (grep `int'(` over subtractions).

# Phase 8 L5 (the 8×8 array on the 16-lane core): measurement (2026-10-29)

Requested by the main session with the K1 go: "measure L5 (the 8×8 array on
the 16-lane core) before the Vivado run". Built as a define (`TCTA_MAC8`,
default off: the 16×16 array stays the planning machine), measured under
Yosys and on the harness, modelled in `tcta.timing.KV260_L5`; the K0 fit
report gains §4c.

## Verdict: L5 saves 17.4 k LUT-eq (−10.7 %), not 35 k, for FA +72 % and GEMM +101 % — not a middle ground

| | 16×16 | 8×8 |
|---|---|---|
| `mma_unit` | 53 199 LUT-eq, 537 DSP | 35 754, 153 DSP (−17 445) |
| core | 162 496 LUT-eq (139 % of the K26) | **145 051 (124 %)** |
| Vivado gain needed for the 99.5 k budget | 39 % | 31 % |
| MMA 64×128×128 | 4 284 | 16 547 |
| GEMM / FA programs | 5 917 / 32 109 | 11 875 (+101 %) / 55 269 (+72 %) |

- **What changed** (`rtl/src/mma_unit.sv`, `tcta_pkg.sv`, `mma_opbuf.sv`):
  with `MAC_COLS = 8` and `BANKS = 16` a block covers half a slice. The B
  fill takes lanes `j0 % 16 .. +7` of each WGT slice (registered offset
  alongside the fill pipeline; the exponent maxima follow it), the
  converter is one pass that places the row at those banks, the writeback
  works on the adder half the row sits in (one mask pass; old + result
  straight into the write data: 4 cycles per accumulated row instead of 5)
  and writes it with the half's bank enables; the result FIFO holds two
  blocks (`2 · MAC_ROWS` rows), blocks start every `max(K, MAC_ROWS)`
  cycles; the lane and row indices are sized from the parameters. The
  16×16 build is unchanged (identical traces, lint clean in both builds).
- **Measured**: `make all MAC8=1` 10/10 PASS bit-exact (the fixed-point
  window is per row and per column of the product, not per block, so the
  array size cannot change a result); `rtl/synth/yosys_xck26_estimate_mac8.md`
  (`yosys_ooc.py --define TCTA_MAC8 --out rtl/synth/ooc_mac8 --md …`, only
  `mma_unit` and the two operand buffers re-synthesised). `calibrate.py
  --timing=KV260_L5` reproduces every MMA shape of mma / gemm / fa / vec
  exactly and the program totals within 0.0 %; `KV260_L5` differs from
  `KV260` only in `mac_rows/cols` 8, `mma_rf_rows` 16,
  `mma_cv_row_cycles` 1, `mma_wb_acc_extra` 2, `mma_wb_mask_extra` 3,
  `mma_wb_mask_mul_extra` 3, `mma_release_latency` 12, `mma_result_latency`
  13 (= K + MAC_ROWS + 4, K + MAC_COLS + 5; the model's block period is now
  `max(K, mac_rows)`).
- **Why half the estimate**: the cells are DSP-based (7 LUTs each), so the
  192 removed cells are 1.3 k LUTs; the saving is the array's control
  (−11.1 k) and half the operand buffers (−5.0 k). Everything else
  lane-proportional (VEC 53.6 k, DMA 29.8 k, pads 14.5 k) is sized by
  `BANKS` and stays. The 8×8 array runs at full occupancy (blocks back to
  back at `K` cycles), so the 3–4× MMA cost is the MAC count, not a
  structural loss; the family programs grow less (FA +72 %, GEMM +101 %)
  because their DMA and VEC parts do not change.

Checks: lint clean (16×16 and 8×8); default `make run PROG=mma` identical
trace; `make all MAC8=1` 10/10 PASS; the committed vector traces are the
16-lane ones (the 8×8 traces were restored after the measurement); `pytest
tests -q` 96 passed (a new test pins the 8×8 program totals to `KV260_L5`).

## Findings for the main session

1. **L5 is not a fit lever worth its speed**: −10.7 % of the core for FA
   +72 % and GEMM +101 %. It moves the Vivado requirement from 39 % to 31 %,
   both at or beyond the usual 20–35 % gain. The §0.3 rule stands as
   written: the 16-lane core if Vivado ≤ 99.5 k, otherwise the 8-lane
   configuration (`BANKS = 8`, ≈ 80 k, fits by Yosys alone). The define
   stays in the tree so the Vivado run can report both arrays.
2. **The array is cheap, the lanes are not.** `mma_unit`'s 256 cells are
   1.8 k LUTs + 512 DSPs; its 53 k is control, buffers and the writeback.
   The 109 k outside it scales with `BANKS`: the lane width is the fit
   decision, as the K0 report said, and the measurement now backs it.
3. **The 8×8 B fill reads each WGT slice twice** (both column halves of a
   slice are separate blocks). It costs nothing today (fill = block period =
   `K`) but would under a two-array or wider-slice variant; filling both
   halves per read is a one-cycle change to the fill engine if L5 is ever
   taken.
4. **DSP budget**: the 8×8 build uses 238 of 1 248 DSP48E2 (19 %), the 16×16
   622 (50 %). Neither is the constraint; block RAM (192 RAMB36 for 144,
   L4's finding) still is for both.

# Phase 8 K2 (erratum E2: the VEC pad banked by row): verdict (2026-10-29)

Per `docs/phase8-gate2-task.md` §1 K2: VEC 2-D slots banked by row; `[64, 1]`
ops cost `⌈M/16⌉ + overhead`, not `M`; FA's per-KV-tile VEC chain drops from
~4.6 k toward ~4.2 k cycles.

## Verdict: K2 DONE — `[64, 1]` ops 70–134 → 9–14 cycles, FA −18.9 %, at +5.2 k LUT-eq and one ISA consequence to decide

| | K1 | K2 |
|---|---|---|
| VFILL / VMUL imm / VADD vec / VEXP2 on `[64, 1]` | 69 / 70 / 134 / 389 | **9 / 10 / 14 / 29** |
| VSUB with a column-vector operand, 64×128 | 582 | 522 |
| VEXP2.SUM 64×64 | 1994 | 1874 |
| FA (4 KV tiles) | 32109 | **26049 (−18.9 %)**; planning model 27856 (+6.9 %) |
| vec program | 14910 | 14610 |
| dma_copy (`[64, 64]` f32 through a 2-D VEC slot) | 2228 | 8396 (element-serial) |
| `scratchpad_VEC` / `vec_unit` (Yosys) | 5 019 / 53 901 LUT-eq | 8 021 / 56 102 |
| core | 162 496 LUT-eq (139 % of the K26) | 167 699 (143 %) |

- **Bank map** (`tcta_pkg.sv`, `scratchpad.sv`): element (row, col) of a 2-D
  VEC slot lives in bank `row mod 16` at word `slot base + col · 4 + row /
  16` (the slot's word range is unchanged). A *column-slice* request
  (`col_req`, `req.cs`: rows `rg·16 .. +15` of one column) is one word in
  every bank, like any other request. A *row-slice* request into that range
  finds its elements in one bank at one word per column, so the pad serves
  it element by element on port A over its enabled lanes (`ben` now names
  the lanes wanted on a read too; 0 = all), grants it with the last element
  and returns the assembled slice the cycle after; port B keeps serving the
  other clients and never a second writer. A client that presents a
  different request mid-burst (dma_load's index read preempts its writer)
  aborts the burst, which restarts when the request returns (writes are
  idempotent). Row slots (`[1, 512]`, slots ≥ 8) are untouched.
- **VEC unit** (`vec_unit.sv`): the `[M, 1]` vector operand is read as one
  column slice per 16 rows into a 16-lane register (lane = row mod 16; K1
  read it once per row); an elementwise kind whose tile operands are all
  2-D VEC tiles (VSUB/VMUL/VADD/VMAX/VNEG/VCOPY/VSEL, the four SFU kinds,
  VFILL) runs in column-slice order — the sequencer's "row" is the row
  group, its "slice" the column, so a `[64, 1]` op is 4 slices; row
  reductions write their result as one element of a column slice. The
  mixed-shape kinds (VGATE / VSEL / VMASK with an ACC tile and a VEC tile,
  VCOUNT on a 2-D slot) keep row order and pay the serial path on the VEC
  side. `dma_store` passes lane masks on its reads so a `[64, 1]` store
  stays one cycle per row.
- **Formal**: the arbiter properties were restated for the burst (the
  top requester holds port A and is granted unless serial; two pure reads
  both granted unless one is serial; a burst is one client, granted only
  with its last element, never beside a port-B write) and proved on the
  default pad (`bank_arbiter.sby`) and on the VEC pad (`bank_arbiter_vec.sby`,
  `SP_ID = 3`), BMC depth 24, both PASS. A first version failed on an
  unreachable initial state (the burst's client index at 7 of 7 clients);
  the index is now range-guarded.
- **Cycle model** (`cycle.py`): `_vec2d` on the tile refs; the VEC rule
  gains the column mode (`N · ⌈M/16⌉` slices, VFILL and SFU one less, the
  vector read `⌈M/16⌉`), the serial cost (`lanes − 1` per row-ordered slice
  of a 2-D VEC tile) and the reductions' +1; the DMA rule costs a 2-D VEC
  destination / source `cols + 1` per row (f4.r16: `8 + lanes` per slice)
  and charges the back end for a gather's skipped rows. Every DMA and VEC
  row of the seven programs is exact except VCOUNT 1×8 on a 2-D slot (13
  vs 14); totals within 0.1 % except `loop` (−7.4 %: the DMA's serial bursts
  and the VEC ops share the VEC pad, and port contention is not in the
  per-instruction rules).
- **Cost**: the pad's serial path is 3.0 k LUTs (the per-lane element
  muxes on the write and the read side and the 512-FF assembly register; a
  first version at 4.3 k selected the whole request from all seven clients),
  the VEC unit +2.2 k (the 16-lane vector register's per-row select, the
  column-request forms). Core 162 496 → 167 699 LUT-eq (143 % of the K26).

Checks: lint clean; `make -C rtl/test all` 10/10 PASS bit-exact (the K1
traces for INP / WGT / ACC are unchanged: mma 23230, gemm 5917); both sby
jobs PASS; `calibrate.py` as above; `pytest tests -q` 96 passed (the
duration test pins VSUB 522 and VROWMAX 523).

## Findings for the main session

1. **E2 makes row-shaped access to a `[M, N > 1]` 2-D VEC tile one cycle per
   element.** A row slice of a row-banked slot is 16 words of one bank; no
   single-address bank map serves both a row slice and a column slice of
   the same tile. The ISA lets such tiles be loaded and stored (`LOAD.2D` /
   `STORE.2D` into VEC) and used by `VGATE acc, src ∈ VEC` (KDA's per-channel
   gating), `VSEL` and `VMASK tile`; the `[64, 1]` statistics the erratum
   targets are unaffected (one element per row either way). Three ways out,
   for the main session to pick before K5 measures KDA: (a) as built — the
   pad's serial path, correct everywhere, 16× slower on those accesses and
   3.0 k LUTs; (b) a skewed map `bank = (row + col) mod 16` with per-bank
   addresses and lane rotators — both shapes at full rate, ≈ 5 k LUTs, and
   a departure from E2's text; (c) restrict 2-D VEC tiles to column access
   (an erratum: `[M, N > 1]` elementwise tiles live in ACC, `VGATE src ∈
   ACC` exists) and drop the serial path to a single-element translation
   (≈ −2.4 k of K2's cost).
2. **The planning model now overestimates FA** (27856 vs 26049, +6.9 %):
   its SFU rule (4 cycles per slice, `⌈M·N/4⌉`) is slower than the fused
   VEXP2.SUM at 7 per slice over 16 lanes, and its row-vector and reduction
   costs are coarser. GEMM stays 4.8 % under. After K3 (SFU 4 per slice)
   the two should meet; worth re-deriving `KV260_PLAN`'s VEC rule from the
   K2 structure at K7.
3. **Pad port contention is a model gap**: a serial DMA burst holds the VEC
   pad's port A for up to 16 cycles per slice, and a VEC op on the same pad
   runs at a third of its rate meanwhile (`loop`: VADD 22 → 31, LOAD 4×16
   103 → 166). Only the `loop` micro-test does this; if (a) stands, the
   replay's port-hold model should learn burst holds before K5.
4. **The protocol hazard** found on the way — a client swapping its request
   while the pad is mid-burst — is closed in the pad (burst abort), but it
   says the request/grant protocol now has state: a request must be held
   until granted, which every client does, and the formal jobs assume
   nothing about it (the properties hold under arbitrary request
   sequences).

## Verdict: K3 DONE — SFU 4 cycles per slice (was 6), MMA accumulate rows 2 cycles (was 5), FA −17.1 %, PV no longer writeback-bound

| | K2 | K3 |
|---|---|---|
| SFU op per slice (VEXP2 / VRCP 64×128: 512 slices) | 6 (model) | **4**: 2055 = 7 + 512·4 |
| VEXP2 / VRCP on `[64, 1]` | 29 | **23** (31 with a column-vector subtrahend) |
| VEXP2.SUM 64×64 (FA) | 1874 | **1052** (−43.9 %); 64×128: 2076 |
| VLOG2 / VSIGMOID 64×64; VEXP2 64×64 with a vector; VGATE decay 64×64 | unmeasured (model 1542 / 1542 / 518) | 1031 / 1039 / **1543** (decay: two operand reads, 6 per slice) |
| MMA accumulate 64×64×64 (FA PV) | 1436 | **1150** (the plain MMA.T 1148) |
| MMA accumulate 64×128×128 | 4332 | 4286 (plain 4284) |
| MMA.MASKED add / mask-mul accumulate 64×128×128 | 4348 / 4892 | 4302 / 4350 |
| FA (4 KV tiles) | 26049 | **21593 (−17.1 %)**; planning model 27856 (+29.0 %) |
| mma program | 23230 | 23138 |
| 8×8 array (`TCTA_MAC8`): MMA 64×128×128 plain / accumulate | 16547 / 16563 | 16540 / 16549 |
| `vec_unit` / `mma_unit` (Yosys, own LUT-eq) | 21 340 / 26 023 | 21 174 / 26 948 |
| core | 167 699 LUT-eq (143 % of the K26) | 168 458 (144 %, +759) |

- **SFU slices in flight** (`vec_unit.sv`): the sequencer issues the next
  slice's read once the stepping slice is within two lane steps of its end
  (`sfu_can_take`), so the read returns in the last step at the earliest,
  when that step's lanes go straight into the write and the returning slice
  loads behind it; the pipe therefore costs the 4 lane steps per slice and
  one return cycle per instruction. Three hazards closed on the way: the
  sigmoid chain's use of lanes 0–3 (x · −log2 e, 1 + u) is now gated on the
  kind, so a returning VEXP2 slice's `x − vec[r]` and VEXP2.SUM's partial add
  keep the adders; VEXP2.SUM's partial add (the cycle after the last step,
  on all 16 adders) must not coincide with a return, so its last step does
  not grant and `!exp_sum_pending` still gates the issue; and the partial add
  keeps its own tag (`sum_tag`) since `sfu_tag` already names the next
  slice. The write FIFO reserves a slot for the stepping slice and the
  issued one. VGATE decay's second operand read (gate then acc) cannot
  overlap the steps — one read in flight — so it costs 6 per slice.
- **Writeback pipelined through its port** (`mma_unit.sv`), per the L1–L4
  verdict's finding 4 (whole rows, the 8 adders and the 8-lane converter
  kept): a row's first operand read — the old accumulator when accumulating,
  else the mask — is prefetched for the row still at the converter output
  (its descriptors are known one cycle before `cv_v`, in the previous row's
  `W_ADD`) while the current row adds (`W_MCALC`, `W_MCALC2`, `W_ADD`: the
  port is free and no read is outstanding), and the write cycle takes the
  next row. Every read lands in one 512-FF register (`pf_d`); a row's old
  value is copied to `old_d` for the second add half (in `W_ADD`) or before
  the mask read overwrites `pf_d` (in `W_MASK`), its mask to `mask_d` for the
  second mask half. Order per row: old, mask, mask op, add, write; from idle
  the first read goes out with the take. Streaming rows cost 2 cycles plain
  and accumulate (read old n+1 ‖ add n, write n), 3 masked, 6 masked
  accumulate (L3: 2 / 5 / 6 / 8); a lone accumulated row 4 (was 5). At P =
  64 the accumulate's 16 rows now take 32 of the 64 cycles: FA's PV is the
  plain product plus 2.
- **Harness**: two programs join `make all` (12/12): `mma_mask` (MMA.MASKED
  add, mask-mul accumulate; bit-exact — no family program emits MMA.MASKED
  and the masked path was unexercised) and `sfu` (VEXP2 / VRCP / VEXP2.SUM
  vec on 64×128, VLOG2 / VSIGMOID / VEXP2 vec / VGATE decay on 64×64, VEXP2
  vec on `[64, 1]`: every SFU kind in row order plus the column-order vector
  case, checked by a new `f32tol` row-relative check at 2⁻¹⁰ against the
  simulator; measured ≤ 8e-5 — a first version chained exp2 through a
  row-sum argument of 77 and failed at 2.8e-3, error propagation, not the
  pipe). `calibrate.py` runs them by default.
- **Cycle model**: `sfu_slice_cycles` 6 → 4, `sfu_fill_cycles` 1 (the first
  return), `sfu_vec_read_cycles` 2 (an issue slot and a return per vector or
  second operand read; VEXP2.SUM's vector read 4), `sfu_sum_slice_extra` and
  `sfu_sum_row_extra` 0; VGATE decay is an SFU kind with two reads. The MMA
  replay drops the four `mma_wb_*` constants and replays the writeback FSM
  state by state (prefetch, landing register, take in the write cycle); it
  reproduces every MMA row — plain, accumulate, masked add, masked
  accumulate, A from ACC, the FA shapes, and the 8×8 build's 16540 / 16549
  (`KV260_L5` re-measured with `make run PROG=mma MAC8=1`). Every SFU row of
  `sfu` and `fa` is exact.
- **Cost**: `mma_unit` +925 LUT-eq (the landing register's 512 FFs, the
  adder-input muxes that pick `pf_d` or the copy, the request-descriptor
  mux between the converter's and the writeback's row), `vec_unit` −166 (the
  kind-gated sigmoid chain). Core 167 699 → 168 458 LUT-eq (144 % of the
  K26). The scratchpad is untouched.

Checks: lint clean; `make -C rtl/test all` 12/12 PASS (also `FORMAL=1`: the
pad's arbiter properties as runtime assertions under the new MMA read
pattern; the two sby jobs are unaffected, `scratchpad.sv` did not change);
`calibrate.py`: every MMA and SFU row exact, totals within 0.1 % except
`loop` (−7.4 %, K2's port-contention gap) and `sfu` (+0.2 %: a 64×1 f32 LOAD
into a 2-D VEC slot, 251 vs 288); `pytest tests -q` 96 passed (the duration
tests pin MMA accumulate 4286 and the 8×8 build's 16540 / 16549).

## Findings for the main session

1. **FA's bound moved**: with the PV accumulate at the plain product's cost
   and the SFU at 4 per slice, FA is 21593 cycles for 4 KV tiles: the MMAs
   are 9.2 k (43 %), the per-tile VEC chain 9.6 k (45 %: VEXP2.SUM 4.2 k,
   VSCALE_ACC 1.3 k, VROWMAX 1.1 k, VCVT 1.0 k, VMUL 1.0 k), the loads
   2.5 k. The next VEC lever is the chain itself (fusing the `VSUB vec →
   VMUL → VEXP2.SUM` and `VSCALE_ACC → VCVT` pairs would be K6 territory),
   not the SFU.
2. **The planning descriptor is now 29 % over FA**: `KV260_PLAN`'s unfused
   SFU rule (three passes, 3072 per VEXP2.SUM 64×64 against 1052) is the
   slow one, after K1–K3 made the DMA, the vectors and the SFU the planned
   machine. K7 should re-derive the planning VEC / SFU rules from the K3
   structure (`sfu_fused_sub_sum`, 4 per slice) so the planning and
   calibrated models meet on FA as they do on GEMM (−4.8 %).
3. **VGATE decay costs 6 per slice, not 4**: its gate and accumulator reads
   are issued one at a time (the tag queue allows one SFU read in flight).
   KDA's gating is this kind; allowing two reads in flight for the two-operand
   SFU kind is a small sequencer change (K8 list) if K5 shows KDA caring.
4. **Masked accumulate rows cost 6 cycles** (old read, mask read, two mask
   halves, add, write): at P ≤ 96 a masked accumulate would be
   writeback-bound again. No family program uses `MMA.MASKED` with
   accumulate (MLA / KDA mask with VGATE); if one appears, the mask read can
   be prefetched too (a second landing register, ≈ 0.5 k).
5. **A DMA model gap surfaced by `sfu`**: a 64×1 f32 LOAD into a 2-D VEC slot
   is 251 cycles against the rule's 288 — the K1 rows-in-flight bound
   (a row holds its record for latency + fetch + write) over-charges
   one-column rows by ~0.5 cycle per row. Not K3's; noted for K7.
6. **The prefetch assumes a one-cycle pad response** for its 2-cycle rhythm;
   a slower response (the VEC pad's serial path, were an accumulator ever a
   2-D VEC tile) only lengthens the row (`W_OLD_WAIT` / `W_MASK_WAIT` wait
   for it), never corrupts it: the taken row owns the outstanding read.
