# TileLang 0.1.15 lowering of the FA kernel: pass table (D1)

Kernel: `compiler/prototype/fa_kernel.py` (batch 1, heads 8, seq 256, dim 64,
64×64 tiles, `T.Pipelined(num_stages=2)`, fp16 in / fp32 accumulate).
Dumps (D2): `compiler/prototype/dumps/<variant>/NN_<pass>.py`, produced by
`compiler/prototype/dump_passes.py`. Versions: `compiler/prototype/PINNED.md`.

| Variant dir | Target | Pass config | Passes |
|---|---|---|---|
| `sm_90/` | `{"kind":"cuda","arch":"sm_90"}` | fast-math | 79 |
| `sm_80/` | `{"kind":"cuda","arch":"sm_80"}` | fast-math | 74 |
| `sm_90_nows/` | sm_90 | + `tl.disable_warp_specialized` | 74, IR identical to `sm_80/` apart from the target attr |
| `sm_80_noasync/` | sm_80 | + `tl.enable_async_copy=False` | 74, **no effect** on this kernel (InjectSoftwarePipeline output identical; still 6 `ptx_cp_async` after LowerTileOp) |
| `sm_90_nows_noasync/` | sm_90 | both | 74, same as `sm_90_nows/` |

Each dir also holds `00_frontend.py` and `passes.tsv` (n, pass, changed).
"Changed" means `mod.script()` differs from the previous dump.

## Summary

| | sm_90 (default) | sm_80, or sm_90 + `disable_warp_specialized` |
|---|---|---|
| **CUT-HERE** | #15 `tl.MaterializeWSSchedule` (IR = #10 `tl.Simplify`) | #18 `tl.PipelinePlanning` |
| First REJECT | #16 `tl.ProducerConsumerWarpSpecialized` | #19 `tl.InjectSoftwarePipeline` |
| What the cut PrimFunc looks like | frontend + target attr + `T.launch_thread` shell + `bz` folded to 0; loop still `T.serial(4, annotations={"num_stages": 2})` | same, plus the loop carries TileLang's stage plan: `software_pipeline_stage = [0, 2×12, 0, 2]` (K and V copies stage 0, everything else stage 2), `software_pipeline_order`, `software_pipeline_async_producers`, `tl_pipelined_num_stages: 2` |

**The three deciding passes:**
- **LayoutInference: REJECT.** It assigns lane-dependent MMA fragment layouts
  to `acc_s`, `acc_s_cast` and `acc_o` (element (i,j) → thread
  `i//16*32 + i%8*4 + j%8//2`, 32 values per thread over 128 threads), and
  4-way replicated row fragments to the `scores_*` and `logsum` vectors. It
  also gives Q/K/V_shared XOR-swizzled layouts. The layouts are annotations
  only (`layout_map` block attrs); tile ops are unchanged by this pass. The
  layouts are the same on sm_80 and sm_90; on sm_90 only the thread range
  moves to 128–256 (the consumer warpgroup).
- **LowerTileOp: REJECT.** No tile op survives, and `T.gemm` is not kept as a
  `tl.gemm` call. On sm_80 it becomes `ptx_ldmatrix` + `ptx_mma` over
  per-thread unrolled loops, `T.copy` becomes `ptx_cp_async`, and
  `T.reduce` becomes `tl::AllReduce` with `NamedBarrier`. On sm_90 it is
  `ptx_wgmma_ss` / `ptx_wgmma_rs` with descriptors, plus `tma_load` and
  `mbarrier_expect_tx`. Fragments become per-thread `local` arrays ([32], [2]).
- **The pipeline pass(es): split.** `PipelinePlanning` is ACCEPT: it only
  replaces the `num_stages` annotation with a stage/order plan and leaves the
  body alone. `InjectSoftwarePipeline` is REJECT, but narrowly: tile ops
  survive and it builds exactly the 2-deep K/V ring the ISA wants
  (`K_shared`/`V_shared` → `[2,64,64]` indexed `k % 2`, 2-fill prologue, loop
  of 2 + epilogue), but it inserts `T.ptx_commit_group()` /
  `T.ptx_wait_group(3)` and tags copies `no_implicit_async_commit_wait`.
  `tl.enable_async_copy=False` does not stop this. On sm_90 neither pass
  does anything: `ProducerConsumerWarpSpecialized` (#16) has already turned
  `T.Pipelined` into a producer/consumer split with mbarriers and TMA.

## Findings that differ from the task's expected shape

1. **The frontend already has thread axes.** `T.Kernel(4, 8, 1, threads=128)`
   emits `T.thread_binding(..., thread="blockIdx.{x,y,z}")` loops and
   `T.launch_thread_idx(0..2)` before any pass runs (`00_frontend.py`).
   `tx/ty/tz` are never read by the tile body before #19/#23; `bx, by` are
   used only as tile coordinates in copy offsets. So D3's assertion (no
   `threadIdx`/`blockIdx`/`launch_thread`) cannot hold at any cut, including
   `00_frontend.py`, without a small shell-stripping rewrite. That rewrite is
   not on §6's list of allowed changes (scope rewrite, skipping passes).
2. **Order.** In 0.1.15, warp specialization (#14–16, sm_90 only) and the
   pipeline passes (#18–19 on sm_80) run **before** LayoutInference and
   LowerTileOp, not after them. `LetInline` does not run (only when
   `tl.force_let_inline` is set). `ThreadPartialSync` does not exist.
3. **sm_90 vs sm_80.** The sm_90 path cuts earlier (warp specialization at
   #16). With `tl.disable_warp_specialized` it is identical to sm_80, so that
   is the path to use for D3: it gets `PipelinePlanning`'s stage plan for free.

## Verdict rules applied

- **ACCEPT**: still tile level (tile calls `T.copy`/`T.gemm`/`T.fill`/`T.reduce`,
  or plain `T.parallel` loops over whole tiles), buffers whole, nothing new
  reads a thread index.
- **CUT-HERE**: last ACCEPT before the first REJECT.
- **REJECT**: introduces thread bindings or `T.launch_thread`, warp
  specialisation, `cp.async`/`mbarrier`/`tma`/`mma`/`wgmma`, lane-dependent
  fragment layouts, or `ptx_*`. Every pass after the first REJECT is REJECT
  ("after cut"); the reason column says what it adds.
- † **Judgement call on #5 `MaterializeKernelLaunch`**: it rewrites the
  frontend's existing `blockIdx` loops / `launch_thread_idx` into
  `T.launch_thread` statements. It adds no thread axis that wasn't there and
  leaves the body untouched, so it is marked ACCEPT. Read strictly ("introduces
  `T.launch_thread`"), it is REJECT and the cut moves to #4 `BindTarget`,
  i.e. effectively the frontend. Either way, finding 1 applies.

## Pass table

`#90` / `#80` are the dump numbers in `sm_90/` and `sm_80/` (`—` = pass not run
on that target). "no change" = IR identical to the previous dump.

| #90 | #80 | Pass | What it introduced (IR diff) | sm_90 | sm_80 | Reason |
|---|---|---|---|---|---|---|
| 00 | 00 | *frontend* | `T.Kernel` shell (blockIdx loops, `launch_thread_idx`), tile ops, `T.serial(4, annotations={"num_stages": 2})` | ACCEPT | ACCEPT | tile ops on whole buffers; shell see finding 1 |
| 1 | 1 | `pass_fn` (NestedLoopChecker) | no change | ACCEPT | ACCEPT | analysis only |
| 2 | 2 | `pass_fn` (ParallelLocalIndexChecker) | no change | ACCEPT | ACCEPT | analysis only |
| 3 | 3 | `pass_fn` (FragmentLoopChecker) | no change | ACCEPT | ACCEPT | analysis only |
| 4 | 4 | `tirx.BindTarget` | `target` func attr | ACCEPT | ACCEPT | attribute only |
| 5 | 5 | `tl.MaterializeKernelLaunch` | shell → `T.launch_thread("blockIdx.x/y/z")`, `T.launch_thread("threadIdx.x", 128)`; body unchanged | ACCEPT† | ACCEPT† | re-expresses existing shell, see † |
| 6 | 6 | `tl.AnnotateDeviceBoundTmaCopies` | no change | ACCEPT | ACCEPT | |
| 7 | 7 | `tl.AddWrapperForSingleBufStore` | no change | ACCEPT | ACCEPT | |
| 8 | 8 | `tl.LegalizeNegativeIndex` | no change | ACCEPT | ACCEPT | |
| 9 | 9 | `tl.InjectAssumes` | no change | ACCEPT | ACCEPT | |
| 10 | 10 | `tl.Simplify` | folds `bz` → 0 in copy offsets (blockIdx.z extent 1) | ACCEPT | ACCEPT | index simplification |
| 11 | 11 | `tl.CanonicalizeLegacyReducer` | no change | ACCEPT | ACCEPT | |
| 12 | 12 | `tl.VerifyReducerEpoch` | no change | ACCEPT | ACCEPT | |
| 13 | 13 | `tl.VerifyBufferInit` | no change | ACCEPT | ACCEPT | |
| 14 | — | `tl.AutoWarpSpecialization` | no change (opt-in, unset) | ACCEPT | — | |
| 15 | — | `tl.MaterializeWSSchedule` | no change (no user schedule) | **CUT-HERE** | — | last ACCEPT on sm_90 |
| 16 | — | `tl.ProducerConsumerWarpSpecialized` | threads 128→256; `if tx < 128` producer / else consumer; 9 mbarriers (`shared.barrier`); K/V copies → `T.tma_copy`; `mbarrier_wait_parity`, `ptx_arrive_barrier`; K/V_shared → `[2,64,64]`; `T.Pipelined` consumed (loop becomes `range(4)`) | **REJECT** | — | first REJECT on sm_90: warp specialisation, mbarrier, TMA |
| 17 | 14 | `tl.LowerBlackwell2SM` | no change | REJECT | ACCEPT | sm_90: after cut |
| 18 | 15 | `tl.IfStmtBinding` | no change | REJECT | ACCEPT | sm_90: after cut |
| 19 | 16 | `tl.UnrollLoop` | no change | REJECT | ACCEPT | sm_90: after cut |
| 20 | 17 | `tl.Simplify` | sm_90: simplifies mbarrier parity exprs; sm_80: no change | REJECT | ACCEPT | sm_90: after cut |
| 21 | 18 | `tl.PipelinePlanning` | sm_80: `num_stages` → `software_pipeline_stage/order/async_producers/async_producer_groups/async_stages`, `tl_pipelined_num_stages`; body unchanged. sm_90: no change | REJECT | **CUT-HERE** | sm_80: loop annotation only, last ACCEPT |
| 22 | 19 | `tl.InjectSoftwarePipeline` | sm_80: K/V_shared → `[2,64,64]` ring (`k % 2`), 2-fill prologue, loop 4→2 + epilogue, `T.ptx_commit_group()`, `T.ptx_wait_group(3)`, copies tagged `no_implicit_async_commit_wait`. sm_90: no change | REJECT | **REJECT** | first REJECT on sm_80: `ptx_*` (tile ops otherwise intact) |
| 23 | 20 | `tl.Simplify` | sm_80: folds prologue constants | REJECT | REJECT | after cut |
| 24 | 21 | `tl.LayoutInference` | `layout_map` attrs: MMA fragments for acc_s/acc_s_cast/acc_o, replicated row fragments for vectors, swizzled shared layouts | REJECT | REJECT | lane-dependent fragment layouts |
| 25 | 22 | `tl.ReducerPlanAndMaterialize` | no change | REJECT | REJECT | after cut |
| 26 | 23 | `tl.LowerTileOp` | all tile ops lowered: sm_80 `ptx_cp_async`/`ptx_ldmatrix`/`ptx_mma`/`tl::AllReduce`/`NamedBarrier`; sm_90 `tma_load`/`mbarrier_expect_tx`/`ptx_wgmma_ss`/`ptx_wgmma_rs`; fragments → per-thread `local` arrays | REJECT | REJECT | thread-level code, MMA intrinsics |
| 27 | 24 | `tl.VerifyReducerConsumed` | no change | REJECT | REJECT | after cut |
| 28 | 25 | `tl.LowerL2Persistent` | no change | REJECT | REJECT | after cut |
| 29 | 26 | `tl.DecoupleTypeCast` | separate `local` cast buffer for the fp32→fp16 output store | REJECT | REJECT | after cut |
| 30 | 27 | `tl.LegalizeVectorizedLoop` | no change | REJECT | REJECT | after cut |
| 31 | 28 | `tl.LegalizeSafeMemoryAccess` | no change | REJECT | REJECT | after cut |
| 32 | 29 | `tl.LowerAccessPtr` | sm_80: pointer metadata → `tvm_access_ptr`; sm_90: no change | REJECT | REJECT | after cut |
| 33 | 30 | `tl.Simplify` | index simplification | REJECT | REJECT | after cut |
| 34 | 31 | `tl.HoistNonRestrictParams` | no change | REJECT | REJECT | after cut |
| 35 | 32 | `tl.LowerSharedTmem` | no change | REJECT | REJECT | after cut |
| 36 | 33 | `tl.PlanAndUpdateBufferAllocationLocation` | moves allocations to their use sites | REJECT | REJECT | after cut |
| 37 | 34 | `tl.LowerSharedBarrier` | sm_90: mbarrier init (`ptx_init_barrier_thread_count` under `tl_shuffle_elect`); sm_80: no change | REJECT | REJECT | after cut; mbarrier |
| 38 | — | `tl.FuseMBarrierArriveExpectTx` | `ptx_arrive_barrier_expect_tx` | REJECT | — | after cut; mbarrier |
| 39 | 35 | `tl.HoistGlobalBufferAllocations` | no change | REJECT | REJECT | after cut |
| 40 | 36 | `tl.LowerOpaqueBlock` | blocks removed, `launch_thread` at function top | REJECT | REJECT | after cut |
| 41 | 37 | `tl.Simplify` | no change | REJECT | REJECT | after cut |
| 42 | 38 | `tirx.NarrowDataType` | sm_90: index dtype narrowing; sm_80: no change | REJECT | REJECT | after cut |
| 43 | 39 | `tl.FlattenBuffer` | buffers flattened to 1-D | REJECT | REJECT | after cut |
| 44 | 40 | `tl.ConfigIndexBitwidth` | no change | REJECT | REJECT | after cut |
| 45 | 41 | `tirx.Simplify` | sm_90: simplification; sm_80: no change | REJECT | REJECT | after cut |
| 46 | 42 | `tl.VectorizeLoop` | vectorized loads, `T.Broadcast` fills | REJECT | REJECT | after cut |
| 47 | 43 | `tir.StorageRewrite` | buffer reuse / renaming | REJECT | REJECT | after cut |
| 48 | 44 | `tl.LoopUnswitching` | no change | REJECT | REJECT | after cut |
| 49 | 45 | `tl.UnrollLoop` | no change | REJECT | REJECT | after cut |
| 50 | 46 | `s_tir.RenormalizeSplitPattern` | index renormalisation | REJECT | REJECT | after cut |
| 51 | 47 | `tirx.Simplify` | index simplification | REJECT | REJECT | after cut |
| 52 | 48 | `tirx.RemoveNoOp` | no change | REJECT | REJECT | after cut |
| 53 | 49 | `s_tir.HoistIfThenElse` | no change | REJECT | REJECT | after cut |
| 54 | 50 | `tirx.VerifyMemory` | no change | REJECT | REJECT | after cut |
| 55 | 51 | `tirx.AnnotateEntryFunc` | entry-func attr | REJECT | REJECT | after cut |
| 56 | 52 | `s_tir.InferFragment` | no change | REJECT | REJECT | after cut |
| 57 | 53 | `tl.LowerThreadAllreduce` | no change | REJECT | REJECT | after cut |
| 58 | 54 | `tl.LowerLDGSTG` | no change | REJECT | REJECT | after cut |
| 59 | 55 | `tl.LowerHopperIntrin` | sm_90: TMA descriptors (`grid_constant` params); sm_80: attr only | REJECT | REJECT | after cut; TMA |
| 60 | 56 | `tl.AnnotateDeviceRegions` | device-region target attr | REJECT | REJECT | after cut |
| 61 | 57 | `tl.SplitHostDevice` | separate `main_kernel` device function | REJECT | REJECT | after cut |
| 62 | 58 | `tl.MarkCudaSyncCalls` | no change | REJECT | REJECT | after cut |
| 63 | 59 | `tl.AnnotateReadOnlyParams` | read-only param attrs | REJECT | REJECT | after cut |
| 64 | 60 | `tl.MergeSharedMemoryAllocations` | one 40960-byte `shared.dyn` buffer with byte offsets | REJECT | REJECT | after cut |
| 65 | 61 | `tl.InjectFenceProxy` | no change | REJECT | REJECT | after cut |
| 66 | 62 | `tl.ThreadSync("shared")` | sm_80: `tvm_storage_sync("shared")`; sm_90: no change | REJECT | REJECT | after cut |
| 67 | 63 | `tl.ThreadSync("shared.dyn")` | `tvm_storage_sync("shared.dyn")` | REJECT | REJECT | after cut |
| 68 | 64 | `tl.InjectTcgen05Fence` | no change | REJECT | REJECT | after cut |
| 69 | 65 | `tl.MergeIfStmt` | no change | REJECT | REJECT | after cut |
| 70 | — | `tl.AnnotateWarpGroupRegAlloc` | `set_max_nreg(24, 0)` / `set_max_nreg(240, 1)` | REJECT | — | after cut; warp specialisation |
| 71 | 66 | `tl.MakePackedAPI` | host packed-API wrapper with arg checks | REJECT | REJECT | after cut |
| 72 | 67 | `tl.Simplify` | simplification | REJECT | REJECT | after cut |
| 73 | 68 | `tl.LowerDeviceKernelLaunch` | `call_packed("main_kernel", …, 4, 8, 1, 128 or 256, 1, 1, 40960)` | REJECT | REJECT | after cut |
| 74 | 69 | `tl.PersistThreadblock` | no change | REJECT | REJECT | after cut |
| 75 | 70 | `tirx.Filter` (host) | host functions only | REJECT | REJECT | after cut |
| 76 | 71 | `tirx.Filter` (device) | device functions only | REJECT | REJECT | after cut |
| 77 | 72 | `tl.LowerIntrin` | intrinsic lowering (shift/and index math) | REJECT | REJECT | after cut |
| 78 | 73 | `tirx.Simplify` | simplification | REJECT | REJECT | after cut |
| 79 | 74 | `pass_fn` (HoistBroadcastValues) | hoists broadcast values into lets | REJECT | REJECT | after cut |

4. **The frontend rejects `tcta.*` scopes before any pass runs.** Setting
   `scope="tcta.acc"` / `"tcta.vec"` in `T.alloc_fragment` makes
   `T.reduce_max` fail at trace time (`tilelang/language/reduce_op.py:171`,
   `ValueError: Invalid buffer scopes: tcta.acc and tcta.vec`): the builder
   only accepts shared/fragment pairs. Set on the frontend PrimFunc instead
   (after the builder, before pass #1), every pass accepts the scopes but
   **#18/#19 then do nothing**: `PipelinePlanning`/`InjectSoftwarePipeline`
   pick the buffers to multi-version by `shared`/`shared.dyn` scope, so the
   loop stays a plain 4-iteration serial loop with no ring. The scope rewrite
   therefore has to run after #19 (see the verdict).

## Outcome (D3–D6)

Per the addendum the cut moved to **#19 `InjectSoftwarePipeline`** (sm_80, or
sm_90 with `tl.disable_warp_specialized`), with its `ptx_commit_group` /
`ptx_wait_group` rewritten to abstract fill events; #18 is the fallback.
`compiler/prototype/cut.py` runs TileLang's own prologue and captures the
module after the cut pass; `compiler/prototype/dumps/cut_inject_pipeline.py`
is the result. The end-to-end run on the simulator is green; see
`docs/tilelang-cutoff-verdict.md`.
