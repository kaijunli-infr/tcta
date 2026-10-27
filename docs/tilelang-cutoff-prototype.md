# TileLang cut-off prototype: FA kernel to tile-level TIR (3 weeks)

**Question to answer:** can TileLang's lowering be stopped at the tile level,
with buffers in named scratchpad scopes and no thread axes, so that the TIR
can be tensorized to the ISA intrinsics? Pass/fail by the end of week 3.

**Pass criterion:** the FA kernel from `tests/test_sim.py` (one Q tile,
2-stage K/V ring, online softmax) is produced *from a TileLang source* as a
TensorIR PrimFunc that (a) contains only tile-level ops on buffers scoped
`INP/WGT/ACC/VEC`, (b) has no `threadIdx`/`blockIdx` binding, and (c) can be
walked by a ~200-line Python pass into the `Program` DSL and run on the
simulator against `fixtures/flash_attn_gqa`. Fail: any of (a)–(c) needs
changes inside TileLang's core passes rather than a target descriptor and
our own passes.

## Week 1: find the cut

1. Install TileLang from source (pin the commit in `third_party/TILELANG_COMMIT`),
   with its bundled TVM. Build with `USE_CUDA=OFF` if it allows it; the prototype
   does not need a GPU.
2. Write the FA kernel in TileLang the standard way (`T.Kernel`, `T.alloc_shared`,
   `T.alloc_fragment`, `T.copy`, `T.gemm`, `T.reduce_max`, `T.exp2`, `T.Pipelined`).
   Keep it to Bq = 64, Bk = 64, d = 64, causal off.
3. Dump the IR **after every pass**. TileLang exposes the pass pipeline through
   `tilelang.engine.lower` / `tilelang.transform.*`; register a `tvm.instrument.PassInstrument`
   that writes `pass_<n>_<name>.py` with `mod.script()`. The point of week 1 is
   the list of passes with, for each, whether it introduces anything we cannot
   accept: thread bindings, `T.launch_thread`, fragment layouts, warp-specialization
   annotations, CUDA-specific intrinsics (`tl.*mma*`, `cp.async`, `mbarrier`).
4. Identify the **last pass before the first unacceptable one**. That is the cut
   candidate. Expected: somewhere after `LowerTileOp`/`LayoutInference` and before
   `ThreadBinding`/`WarpSpecialized`/`LowerHopperIntrin`. If layout inference
   already assumes MMA fragment layouts for `T.gemm` operands, that is the first
   real obstacle; note whether it is gated on target.

Deliverable: `docs/tilelang-passes.md`, a table of passes with verdicts, and the
cut candidate.

## Week 2: make the cut

5. Add a target descriptor. Try, in this order:
   - a new `Target` kind (`tcta`) with no thread extents, so passes that query
     `target.max_num_threads` get 1 or skip;
   - if TileLang keys behavior on `target.kind.name in ("cuda", "hip")`, the
     `tcta` kind should fall through the "generic" path; find every such check.
6. Buffer scopes: map `T.alloc_shared` → `WGT`/`INP` by role (operand B vs A)
   and `T.alloc_fragment` → `ACC`/`VEC`. In the prototype this can be an
   explicit annotation in the kernel (`scope="tcta.wgt"`) rather than inference.
   TVM accepts arbitrary scope strings; the question is whether TileLang's
   lowering rejects unknown scopes or passes them through.
7. Run the pipeline up to the cut with the `tcta` target. Expected failures and
   their fixes, in order of likelihood:
   - `LayoutInference` wants an MMA layout for `T.gemm` operands → register a
     trivial row-major layout for the `tcta` target, or skip the pass.
   - `T.Pipelined` lowering inserts `cp.async`/mbarrier → stop before it and
     keep the loop annotated with `num_stages`; the stage assignment is what
     our token-insertion pass consumes.
   - `T.reduce_max` / `T.exp2` lower to per-thread loops → stop before `LowerTileOp`
     for those ops, or re-raise them from the lowered loops (harder; counts
     against the prototype if required).
8. Freeze the cut as a function: `tilelang_to_tile_tir(prim_func, target="tcta") -> PrimFunc`.

Deliverable: the cut function and the FA PrimFunc at the cut, checked in under
`compiler/prototype/`.

## Week 3: prove it lands on the simulator

9. Write `tile_tir_to_program.py`: a TVM `PrimFunc` visitor that maps
   - `T.copy(global→WGT/INP)` → `LOAD.2D` with the loop-variant base;
   - `T.gemm` → `MMA`/`MMA.T` (transpose from the layout flags);
   - `T.reduce_max`/`reduce_sum` → `VROWMAX`/`VROWSUM`;
   - `T.exp2` with a broadcast subtract → `VEXP2 t, vec`;
   - the `num_stages` annotation → the ring-buffer token pattern from
     `sim/README.md` (pre-release, pop-before-fill, push-after-use, drain).
   It needs no cleverness: the FA schedule is fixed, so the pass can be
   pattern-based. Token insertion for the general case is the real compiler's
   job later.
10. Run it: TileLang source → cut → Program → `Machine.run` → compare to
    `fixtures/flash_attn_gqa` at `tol_rowmax`. Token-check faults here mean the
    pattern pass is wrong, not TileLang.
11. Write the verdict in `docs/tilelang-cutoff-verdict.md`: pass/fail, which
    TileLang passes were skipped or replaced, which assumptions leak, and the
    estimated cost of carrying the cut forward versus writing the four kernels
    directly in TensorIR with our intrinsics.

## Decision rule

- **Pass**: TileLang stays the authoring language; compiler stages 1–2 of the
  plan are "TileLang + cut + scope legalization". Keep the TileLang commit pinned.
- **Fail**: write the four kernels in TensorIR directly against `TensorIntrin`
  descriptions of the ISA (one intrinsic per instruction), keep `T.Pipelined`'s
  idea as a loop annotation of our own. The simulator, fixtures and token
  patterns are unchanged either way, which is why this prototype can run in
  parallel with the simulator work.

## What to send back

The pass table (week 1), the cut function (week 2), and the FA PrimFunc at the
cut (week 2). With those I can write the `tile_tir_to_program` pass and the
scope-legalization pass; without the dumps I would be guessing at TileLang
internals that change between commits.
