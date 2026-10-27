# Task: cut TileLang's lowering off at the tile level (FA kernel)

Handoff instruction for a fresh session. Read this whole file before writing code.

## 0. What this is and is not

**"Cut-off" here means cutting the compiler's lowering pipeline off at the tile
level.** It does NOT mean attention with a distance cut-off or a sliding window.
A previous session built a sliding-window kernel (`tilelang-cutoff/`); that work
is a useful starting point (see §2) but its band mask, `skip_tiles` and `cutoff`
parameter are out of scope. Do not extend them.

Goal: show that a FlashAttention-2 kernel authored in TileLang can be stopped
part-way through TileLang's lowering as a TensorIR `PrimFunc` that contains only
**tile-level operations on buffers in named scratchpad scopes, with no thread
axes**, and that this PrimFunc can be mapped by a small pass onto a
tile-level ISA and executed on a Python simulator against a fixed reference.

The ISA, simulator and fixtures already exist (repo `tcta`, tarball
`tcta-v0.1.tar.gz`): `isa/isa-v0.1.md`, `sim/tcta/`, `tests/test_sim.py`
(contains the hand-written FA schedule `test_flash_attention_inner_loop`),
`ref/kernels.py` (`fixtures/flash_attn_gqa`). You do not change the ISA or the
simulator; if you believe you must, stop and report why.

## 1. Deliverables (in this order; each is useful alone)

| # | Deliverable | Where | Definition of done |
|---|---|---|---|
| D1 | Pass table | `docs/tilelang-passes.md` | One row per pass in TileLang's lowering of the FA kernel, with: pass name, what it introduced (as seen in the IR diff), verdict ACCEPT / CUT-HERE / REJECT, one-line reason |
| D2 | IR dumps | `compiler/prototype/dumps/NN_<pass>.py` | `mod.script()` after every pass, numbered; plus `00_frontend.py` (before any pass) |
| D3 | The cut | `compiler/prototype/cut.py` exposing `lower_to_tile_tir(prim_func) -> tvm.tir.PrimFunc` | Returns the PrimFunc at the CUT-HERE point; a pytest asserts it has no `threadIdx`/`blockIdx`/`T.launch_thread`, no `tl.*mma*`, `cp.async`, `mbarrier`, `tma` intrinsics, and that every buffer carries one of the scopes in §3 |
| D4 | Mapping pass | `compiler/prototype/tile_tir_to_program.py` | Walks the D3 PrimFunc and emits a `tcta.Program` (the simulator's DSL) |
| D5 | End-to-end test | `tests/test_tilelang_cut.py` | TileLang source → D3 → D4 → `Machine.run` matches `fixtures/flash_attn_gqa` within `tol_rowmax = 2^-7`, with token checking on |
| D6 | Verdict | `docs/tilelang-cutoff-verdict.md` | PASS / FAIL per §6, plus the list of TileLang passes skipped or replaced and the assumptions that leak |

Deliver D1+D2 as soon as they exist, before starting D3. They are the part that
cannot be done without a TileLang install, and the rest can be planned from them.

## 2. Environment and starting point

- Linux, no GPU, no CUDA toolkit needed. `tilelang==0.1.15` from PyPI (aarch64
  or x86_64 wheel) with its bundled TVM; `numpy`, `pytest`; `torch` CPU-only is
  optional (only `tilelang-cutoff/`'s own tests use it).
- Pin: record `tilelang.__version__` and the TVM commit in
  `compiler/prototype/PINNED.md`. Everything in this task is version-specific.
- Starting kernel: `tilelang-cutoff/cutoff_attn/kernel.py::build_cutoff_attention`
  with `backend="cuda"`. Copy it to `compiler/prototype/fa_kernel.py` and **reduce it**:
  remove `cutoff`, `causal`, the band mask loop, `skip_tiles`, `TilesVisited`, the
  CPU backend branches, and the `one_trip` workaround. The KV loop becomes
  `for k in T.Pipelined(0, n_kv_tiles, num_stages=2)`. Keep: `T.copy`, `T.gemm`
  (`transpose_B=True` for QKᵀ), `T.fill`, `T.reduce_max`, `T.reduce_sum`, `T.exp2`
  with the `LOG2E` scale, the `scores_scale` rescale of `acc_o`, `T.copy(acc_s, acc_s_cast)`.
- Shapes for the whole task: `batch=1, heads=8, seq_len=256, dim=64, block_M=64,
  block_N=64`, fp16 operands, fp32 accumulate, `sm_scale = 1/sqrt(dim)`. These
  match `fixtures/flash_attn_gqa` (treat each query head against its KV head
  `h // 4`; GQA stacking is NOT required for this task, run one head at a time).

## 3. Scopes and the ISA you are mapping to

Buffers in the cut PrimFunc must carry exactly one of these storage scopes:

| Scope string | ISA scratchpad | Holds in the FA kernel |
|---|---|---|
| `tcta.inp` | INP | Q tile (A operand of QKᵀ) |
| `tcta.wgt` | WGT | K tile, V tile (B operands) |
| `tcta.acc` | ACC | `acc_s` (S, then P in place), `acc_o` (O) |
| `tcta.vec` | VEC | `scores_max`, `scores_max_prev`, `scores_scale`, `scores_sum`, `logsum` |
| `global` | DRAM | Q, K, V, Output |

In the prototype it is acceptable to set these scopes **explicitly in the kernel
source** (e.g. `T.alloc_shared(..., scope="tcta.wgt")` if the API allows a scope,
otherwise by a small TIR pass that rewrites `shared`/`shared.dyn`/`local.fragment`
to the table above by buffer name). Inferring them is the real compiler's job.
Note that `acc_s_cast` disappears: the ISA converts P in place (`VCVT ... inplace`,
ISA §6.3), so a copy `acc_s -> acc_s_cast` maps to one `VCVT` on the same ACC tile.

Target instructions (ISA v0.1, `sim/tcta/asm.py`): `LOAD.2D`, `MMA`, `MMA.T`,
`VFILL`, `VROWMAX`, `VROWSUM`, `VMAX`, `VSUB`, `VMUL`, `VADD`, `VEXP2 t, vec`
(fused subtract), `VEXP2.SUM`, `VSCALE_ACC`, `VCVT inplace`, `VRCP`, `STORE.2D`,
`PUSH`/`POP`, `LOOP`/`ENDLOOP`, `FINISH`. The hand-written target schedule you
are reproducing is `test_flash_attention_inner_loop` in `tests/test_sim.py`;
read it first, it defines the token pattern (§5).

## 4. How to do D1–D3 (the diagnostic)

1. Get the pass pipeline. TileLang 0.1.15 lowers through `tilelang.engine.lower.lower`
   (used by `tilelang-cutoff/kernel.py::cuda_source`). Read that function and the
   phase functions it calls (names like `LowerAndLegalize`, `OptimizeForTarget`);
   they build `tvm.transform.Sequential` pipelines of `tilelang.transform.*` and
   `tvm.tir.transform.*` passes.
2. Instrument, don't guess. Register a `tvm.instrument.PassInstrument` subclass
   with `run_after_pass(mod, info)` that writes `dumps/NN_<info.name>.py` with
   `mod.script()`. Use `tvm.transform.PassContext(instruments=[...], config=...)`
   around the `lower` call, with `target = tvm.target.Target({"kind":"cuda","arch":"sm_90"})`
   first (it is the path whose pass list you need to see) and then `sm_80`.
   If `lower` runs phases in separate `PassContext`s, instrument each.
3. For each dump, diff against the previous one and classify:
   - **ACCEPT**: still tile-level (`T.copy`/`tl.*` tile calls or plain loops over
     whole tiles, buffers whole, no thread index).
   - **CUT-HERE**: the last ACCEPT before the first REJECT.
   - **REJECT**: introduces any of: `threadIdx`/`blockIdx` bindings or
     `T.launch_thread`; warp-specialisation annotations or producer/consumer
     branches; `cp.async`, `mbarrier`, `tma`, `wgmma`/`mma` intrinsics; fragment
     layouts (`local.fragment` index maps that depend on lane id); `ptx_*`.
   Expected shape of the answer (verify, do not assume): frontend → `LetInline`/
   `AddWrapperForSingleBufStore`/`LayoutInference`/`LowerTileOp` ... →
   `LegalizeSafeMemoryAccess` ... → `Pipeline`/`InjectSoftwarePipeline` →
   `WarpSpecialized`/`LowerHopperIntrin`/`ThreadSync` → `ThreadPartialSync` → codegen.
   Three passes decide the task: **LayoutInference** (does it force MMA fragment
   layouts onto `acc_s`/`acc_o`?), **LowerTileOp** (does it turn `T.gemm` into a
   thread loop or keep a `tl.gemm` call?), and **the pipeline pass** (does
   `T.Pipelined` survive as a loop annotation, or become cp.async/mbarrier code?).
4. Build `lower_to_tile_tir` by running only the passes up to CUT-HERE, with the
   scope rewrite of §3 applied before any pass that reads scopes. If a pass before
   the cut rejects an unknown scope, note it in D1 and apply the rewrite after that
   pass instead; the cut PrimFunc must still carry the §3 scopes.
5. If `LayoutInference` or `LowerTileOp` are REJECT, try skipping them (construct
   the `Sequential` without them). If the later ACCEPT passes then fail, the cut
   moves earlier; the earliest legal cut is the frontend PrimFunc itself
   (`00_frontend.py`), which is still a valid D3 as long as D4 can consume it.
   Record which option you took.

The pass table is the deliverable even if nothing after it works: it tells us
what the real compiler has to own.

## 5. How to do D4–D5 (the mapping)

`tile_tir_to_program.py` is a pattern pass for one fixed schedule; it does not
need to be general. Use `tvm.tir.stmt_functor.post_order_visit` or
`ir_transform`. Mapping rules:

| Tile TIR | Program |
|---|---|
| `T.copy(global Q[...], Q_inp)` before the loop | `load2d(INP(0), rq, 64, 64)` then `push(LOAD, MMA)` |
| `T.copy(global K[k*bN...], K_wgt)` inside the loop | `load2d(WGT(0), rk, 64, 64, incr={"region.base": bN*dim*2, "dst_mod": 2})` + push; the `V` copy to `WGT(2)` likewise |
| `T.gemm(Q_inp, K_wgt, acc_s, transpose_B=True)` | `mma_t(ACC(0), INP(0), WGT(0), incr={"b_mod": 2})` |
| `T.fill(scores_max, -inf)` | `vfill(VEC(..), imm=-1e30, rows_valid=64, cols_valid=1)` (the simulator's `-inf` handling is via `VEXP2` returning 0 for x ≤ −127; use a large finite negative) |
| `T.reduce_max(acc_s, scores_max, dim=1)` | `vrowmax(VEC(m_tile), ACC(0))` |
| `scores_max[i] = max(scores_max[i], prev[i])` | `vmax` |
| `scores_scale[i] = exp2(prev*scale - max*scale)` | `vsub` + `vexp2`, with the `scale` multiply hoisted: multiply `ACC(0)` once by `scale` (`vmul(ACC(0), imm=scale)`) right after `mma_t`, so all later exps are of already-scaled values |
| `acc_s[i,j] = exp2(acc_s*scale - max*scale)` + `T.reduce_sum(acc_s, scores_sum)` | one `vexp2_sum(ACC(0), VEC(sum), vec=VEC(m_new))` |
| `logsum = logsum*scale + sum` | `vmul` + `vadd` |
| `T.copy(acc_s, acc_s_cast)` | `vcvt(ACC(0), ACC(0), inplace=True)` |
| `acc_o[i,j] *= scores_scale[i]` | `vscale_acc(ACC(1), VEC(scale))` |
| `T.gemm(acc_s_cast, V_wgt, acc_o)` | `mma(ACC(1), ACC(0), WGT(2), accumulate=True, incr={"b_mod": 2})` |
| `acc_o /= logsum`; `T.copy(acc_o, Output)` | `vrcp` + `vscale_acc` + `store2d` |

Tokens: do not derive them from the TIR. Emit the fixed pattern from
`sim/README.md` ("Writing a kernel") keyed on `num_stages`: pre-release
`num_stages*2` MMA→LOAD tokens, LOAD pops two before each K/V fill, MMA pushes
two releases after the PV GEMM, LOAD drains `num_stages*2` after the loop;
LOAD→MMA on every fill; MMA→VEC after `mma_t` and after PV; VEC→MMA twice before
PV (P ready, O rescaled). Copy the exact sequence from
`test_flash_attention_inner_loop` and parametrize only the stage count. General
token insertion is explicitly not part of this task.

D5 then is: build the kernel, `lower_to_tile_tir`, `tile_tir_to_program`, load the
fixture's Q/K/V into the simulator's DRAM (see `tests/test_ref.py` for the
helpers), run, compare head by head with `common.rowmax_err`. Any
`UNTOKENED_*`/`TOKEN_*` fault means the pattern emission is wrong, not TileLang.

## 6. Decision rule for D6

- **PASS**: D3 is produced by TileLang's own passes up to a cut point, with at
  most (a) a scope-rewrite pass and (b) skipping of named passes; D5 is green.
- **PASS-WITH-CAVEATS**: the cut is the frontend PrimFunc (`00_frontend.py`),
  i.e. nothing of TileLang's lowering is reused but authoring in TileLang works
  and D5 is green. State which TileLang passes would have to be re-implemented.
- **FAIL**: D3 needs changes inside TileLang's C++ or Python passes, or the
  frontend PrimFunc itself contains thread or layout assumptions that D4 cannot
  undo.

Write the verdict with the three load-bearing passes (§4.3) named and their
behaviour described in one sentence each. That paragraph is what the compiler
plan will be rewritten from.

## 7. What not to do

- No GPU work, no nvcc, no CUDA codegen beyond what the pass dumps produce for free.
- No changes to the ISA or the simulator. No GQA stacking, no causal mask, no
  sliding window, no autotuning, no second kernel.
- No general token-insertion or scheduling logic. Fixed pattern only.
- Do not spend more than one day fighting a single TileLang pass; record it as
  REJECT with the error text and move the cut earlier.
