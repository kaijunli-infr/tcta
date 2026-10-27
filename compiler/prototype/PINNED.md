# Pinned versions

Everything under `compiler/prototype/` is specific to these versions.

| Component | Version |
|---|---|
| tilelang | 0.1.15 (PyPI wheel `cp39-abi3-manylinux_2_34_aarch64`) |
| tilelang wheel sha256 | `77343b47643eaab07d8c1f8314c6c33e4992cb07883207796c900b719626524f` (`tilelang-0.1.15-cp39-abi3-manylinux_2_34_aarch64.whl`, via `pip download tilelang==0.1.15 --no-deps`, 2026-10-26). This is the pin, per the addendum (decision 5). |
| bundled TVM | `tvm.__version__` = `0.25.dev0`; commit **not recorded in the wheel** (no git hash in `libtvm_compiler.so`, `version.py` or the dist-info) |
| Python | 3.14.7 |
| torch | 2.14.1 CPU (not used by this task) |
| Host | aarch64 Arch Linux, no GPU, no CUDA toolkit |

The interpreter used is `~/Projects/tilelang-cutoff/.venv/bin/python` (the venv
from the earlier sliding-window prototype).

## API notes for this version

- TileLang's bundled TVM uses the `tvm.tirx` namespace (and `tvm.s_tir`), not
  `tvm.tir`: `tvm.tirx.PrimFunc`, `tvm.tirx.stmt_functor.post_order_visit`,
  `tvm.tirx.transform.*`.
- The CUDA pipeline is `tilelang/cuda/pipeline.py::CUDAPassPipelineBody`. It
  calls each pass directly (`mod = Pass()(mod)`), not through one
  `tvm.transform.Sequential`, so a `PassInstrument` on the enclosing
  `PassContext` sees every pass. `tilelang.engine.lower.lower` appends its own
  instruments to the current context and keeps ours.
- Three unnamed `pass_fn` passes run first (`PreLowerSemanticCheck`:
  NestedLoopChecker, ParallelLocalIndexChecker, FragmentLoopChecker), and one
  last (`HoistBroadcastValues`, in `_prepare_device_codegen_mod`).

## Phase 2 (MoE) frontend notes, same versions

- Helpers that build loops must be `@T.macro`; the eager builder only
  source-transforms the decorated function (`'ForFrame' object is not iterable`
  otherwise). Macros rename global buffers to their parameter names in the IR.
- No `from __future__ import annotations` in kernel modules (the `T.Tensor`
  hints are evaluated at trace time).
- Scalar lets inside `T.Parallel` are `tvm.tirx.stmt.Bind` (fields `var`,
  `value`; a statement, no body), not `LetStmt`.
- Manual pipeline stages: `T.Pipelined(n, order=[...], stage=[...])` without
  `num_stages`; one entry per scheduled statement (copies, compute loops, gemm).
- `T.gemm(..., clear_accum=True)` appears as `args[9]` of `tl.tileop.gemm`.

## Phase 3 notes, same versions

- `T.ceildiv(n[0], 64)` as a `T.serial` extent survives #19 as `(n[0] + 63) // 64`.
- A guarded statement in stage 0 of a manual `T.Pipelined` is multi-buffered
  as if unconditional (InjectSoftwarePipeline); keep conditional producers
  outside pipelined loops. Manual `order/stage` lists count guarded statements.
- Buffers must be allocated (`T.alloc_*`) inside the `@T.macro` body that owns
  the loops, or they lose their names (`buffer_N`) and the scope rewrite misses
  them; pass them to inner macros as parameters.
- `T.cast(value, dtype)` (value first). `T.if_then_else` with no else in a
  `T.Parallel` store is a plain `IfThenElse` statement, kept through #19.
- `T.Pipelined` rings restart per loop: a ring that should span several
  N-tiles needs one flattened loop with guarded per-N-tile statements.

## Phase 4 §0 notes (ISA v0.1.2), same versions

- D4 per-level `incr`: operands must be affine in every enclosing loop
  variable over the sample grid (all iterations of static loops, {0, 1} of
  dynamic ones); `T.min`/`T.max` clamps on a loop variable are rejected
  (`operand is not affine`). The idx table has a `-1` pad row at `E` so the
  count prefetch of the last expert needs no clamp.
- A fragment reused across a dynamic loop by different queues (the shared
  `idx_all`/`mask` of two counts) creates a pop in front of the `LOOP`; the
  collapsed `VCOUNT` counts the `mask` fragment as its buffer, so split both.
- `T.copy(a_frag, b_frag)` between two VEC fragments lowers to `VFILL 0; VADD`
  (two VEC instructions), good enough for a one-row copy that moves a release.
- `Machine.reset()` keeps `stats` and `t`; a program run after another on the
  same machine reports cumulative steps.

## Phase 4 notes (MLA, KDA), same versions

- `from __future__ import annotations` breaks `@T.prim_func` signature parsing
  (`T.Tensor([seq_len, d], dtype)` annotations are evaluated by name): keep it
  out of kernel modules.
- A 3-D buffer must be indexed with every axis in `T.copy`/`T.gemm` region
  operands (`Q_shared[(c + 1) % 2, :, :]`, not `Q_shared[(c + 1) % 2]`).
- Column slices of a fragment wider than one tile (`acc_o[:, c*128:(c+1)*128]`)
  are legal gemm destinations and survive #19 as `tl.region` calls with a
  column offset; D4 maps the slice to the slot.
- `T.exp2` inside a `T.Parallel` store keeps its operands unsimplified through
  #19 (`Kh[i, j] * T.exp2(G[i, j])` arrives as `Mul(load, Call(exp2, load))`);
  `-x` arrives as `x * -1` (f32) and `0 - x` for some int forms.
- A bf16 `alloc_fragment` written by `T.copy` from an f32 fragment is a
  `Cast` on the store; `Rl[i, j] = R[i, j] - Rh[i, j]` arrives with the cast on
  the bf16 operand only.
- Gathers into an f32 fragment from a bf16 global carry a `Cast` around the
  `if_then_else`; the frontend accepts `T.cast(0, dtype)` for the else branch.
- `T.alloc_shared` with scope `tcta.state` via the scope map works like the
  others (the shared→scope rewrite is name-based).
- The simulator's `_op_VNEG` goes through `_binop`, which requires `imm` or
  `vec`; D4 emits `VMUL imm=-1.0` for negation (finding for the main session).

## Phase 5 notes (ISA v0.1.3, cycle model), same versions

- v0.1.3 forms used by D4 (`tile_tir_to_program.py`): `T.copy(acc_frag,
  inp_or_wgt_shared)` → `VCVT dst=INP/WGT` (`vcvt_operand`); `T.copy(acc,
  state[r0:r0+64, :])` → `VCVT dst=STATE, row0=r0` (`vcvt_state`, `r0` must be
  a static multiple of 64); `T.copy(acc_f32, acc_bf16)` with a later read of
  the source → `VCVT to_bf16=True` into a fresh slot (`vcvt_copy`); VEC→VEC
  and ACC→ACC copies → `VCOPY`; `-x` → `VNEG`. An `MMA` (accumulate=False)
  followed by `VGATE mul` with a 2-D mask fuses to `MMA.MASKED mode=mul`;
  the transposed `MMA.T` has no masked form, so KDA's `VGATE mul` masks on
  the `MMA.T` products stay (finding).
- `VCVT dst=ACC` is an f32 move unless `to_bf16`; a kernel that relied on the
  v0.1.2 "cast on copy" into ACC must now say so (KDA's hi/lo split does).
- `kda_kernel_roundtrip.py` keeps the v0.1.2 authoring (DRAM scratch round
  trips, STATE mirror reload) as the before-case of the latency sweep; it
  imports shapes and the host side from `kda_kernel.py`.
- `mla_kernel.build_mla_kernel(inp_slots=n)`: for n ≠ 2 the chunk loop is
  Python-unrolled with static slot indices (`Q_shared[slot, :, :]`), n − 2 Q
  chunks resident (rope first), the rest streamed through a 2-slot ring; the
  default form (n = 2) is byte-identical to phase 4.
- Cycle model (`sim/tcta/cycle.py`): a post-mortem replay of the functional
  run's recorded dispatch order and per-queue streams. Two rules that matter
  when reading its numbers: a PUSH never holds its queue unless the FIFO is
  full (the queue issues the next LOAD while the previous one is in flight);
  bank-port holds are intervals and a hold that starts later than the
  candidate cycle does not block (queues are replayed in ready-time order, not
  global time order, so the later-starting instruction is never delayed).
- `DeviceTiming.request_channels` (default 1 = the task's rule) is a what-if
  knob: a LOAD.2D/STORE.2D striped over k channels, gather rows round-robin
  over k channels.
