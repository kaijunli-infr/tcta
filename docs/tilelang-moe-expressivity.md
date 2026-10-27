# TileLang expressivity for the MoE decode kernel (E2)

Phase 2 of the cut-off prototype (`docs/tilelang-phase2-moe-task.md`). Kernel:
`compiler/prototype/moe_kernel.py`; cut PrimFuncs: `compiler/prototype/dumps/cut_moe.py`
(MXFP4 weights, normative since phase 3), `cut_moe_bf16.py`, `cut_router.py`,
`cut_moe_persistent.py` / `cut_moe_combine.py` (phase 3); programs:
`dumps/program_moe_e0.txt`, `program_moe_bf16_e0.txt`, `program_router_t0.txt`,
`program_moe_persistent.txt`, `program_moe_combine_t0.txt`. Sections 1–8 are
the phase-2 record; §9 lists what phase 3 (ISA v0.1.1) changed.
Versions as pinned in `compiler/prototype/PINNED.md` (tilelang 0.1.15).

The question for each ISA feature: is there a TileLang construct for it, does it
survive passes 1–19, and does the cut PrimFunc still carry what D4 needs.

## 1. Token gather → `LOAD.IDX`

**What TileLang has.** No tile-level gather. `tilelang/language/` has only
`tma_gather4`/`tma_scatter4` (sm_100a TMA, exactly 4 rows, dtype-matched
global→shared), nothing that takes an index buffer in a `T.copy` source region.
A `T.copy` of one row per `T.Parallel` iteration (task form b) is rejected by
the semantic check: *"Only elementwise operations are allowed inside a parallel
loop"*. So the gather is authored elementwise:

```python
for i, j in T.Parallel(64, 128):
    x_s[i, j] = T.if_then_else(idx_s[i] >= 0, x[idx_s[i], kk * 128 + j], T.cast(0, dtype))
```

**After the cut** (`cut_moe.py`): verbatim, as a `parallel(64) × parallel(128)`
nest around one `BufferStore` with the `if_then_else`; nothing before #19
touches it (LayoutInference, which would have assigned per-thread fragments, is
after the cut). D4 recognises the shape — destination in `tcta.inp`, a global
source indexed by a `tcta.vec` int32 load in the row position, zero on a
negative index — and emits one `LOAD.IDX dst=INP[0], idx=VEC[8], nrows=64,
row_bytes=1024, cols_valid=128, col0=kk*128`. The `-1 → zero row` rule of ISA
§6.1 is exactly the `if_then_else`. The index list itself is `T.copy(idx[e,
t*64:(t+1)*64], idx_s)` → `LOAD.2D i32` into a VEC row tile (ISA §5.3 allows
index tiles to be loaded, not only produced by `VTOPK`); because LOAD both
writes and reads the index tile, the VEC→LOAD edge is not needed for the gather.

Cost: one `LOAD.IDX` per K-tile of the token row (4 per 64-token tile; the DMA
may cache the row across them, §6.1). Verdict: **expressible, not as a tile op;
the elementwise form is a stable, recognisable pattern.**

## 2. Dynamic trip count → `VCOUNT` + `LOOP r`

**`T.serial` with a dynamic extent** from a buffer (`for t in T.serial(n_tiles[bx])`
or from a fragment scalar) is accepted by the frontend and by every pass up to
#19; the cut PrimFunc keeps `for t in range(n[0])`. Static `T.Pipelined` loops
nested inside it are pipelined normally (ring, prologue/epilogue, commit/wait).

**`T.Pipelined` with a dynamic extent** (`for t in T.Pipelined(n_tiles[bx],
num_stages=2)`) is also accepted — PipelinePlanning and InjectSoftwarePipeline
do not refuse it — but the result (`dumps/probe_dynamic_pipelined.py`) is:

```
if 0 < n_tiles[bx]: <stage-0 copies, T.where(0 < n_tiles[bx])>; ptx_commit_group
if 1 < n_tiles[bx]: ...
for t in T.serial(n_tiles[bx] - 2): ...
for t_1 in T.unroll(n_tiles[bx] + 2 - n_tiles[bx]): T.where(2 <= t_1 + n_tiles[bx] and t_1 < 2) ...
```

Predicated prologue/epilogue and a `serial(n − 2)` body. The ISA's only
conditional is `LOOP r` with `r == 0` (§7); there is no predicate and no
register arithmetic, so this form has no mapping. **Record for E3: PipelinePlanning
does act on a dynamic-extent loop, by predication, which is unusable here.**
The kernel therefore uses `T.serial` for the token-tile loop and `T.Pipelined`
only for the static weight streams inside it.

**Where the count comes from.** `VCOUNT r, idx, bound` counts in-range entries
of an index row; `LOOP r` iterates `r` times; there is no ALU (§7). The loop is
over 64-row *tiles*, so `r = ceil(count/64)`, which cannot be derived on the
device from the token list. The kernel therefore takes a `heads[e, t]` row —
the first token of each tile, `-1` for an empty tile (a strided view of
`idx_lists` the host/router prepares alongside it) — and computes

```python
for i in T.Parallel(t_max): mask[i] = T.if_then_else(heads_s[i] >= 0, 1, 0)
T.reduce_sum(mask, n, dim=0)
for t in T.serial(n[0]): ...
```

which is `VCOUNT`'s definition written in TileLang. D4 matches the triple and
emits `LOAD.2D i32 heads → VEC[9]; VCOUNT R0, VEC[9], bound=INT32_MAX; LOOP R0`
(the frontend condition is `>= 0`, so the bound is the int32 maximum; `mask`
and `n` get no slot). A zero-token expert gives `R0 = 0`: the loop is skipped,
no token inside it is issued, FINISH finds empty FIFOs
(`test_zero_token_expert_runs_zero_iterations`). `heads` all `-1` is the only
host input the zero case needs.

Verdict: **expressible; the tile count must be host-provided in v0.1** (ISA
candidate: `VCOUNT.TILES r, idx, bound, 64` or a `LOOP r, step=64`).

## 3. Router top-k → `VTOPK`

No TileLang counterpart (no sort/top-k tile op). Authored as the task allows:

```python
row = T.alloc_fragment([1, E], "float32"); idx = T.alloc_fragment([1, 8], "int32")
T.copy(logits[bx:bx + 1, 0:E], row)
T.call_extern("handle", "tcta.vtopk", T.address_of(idx[0, 0]), T.address_of(row[0, 0]), k)
T.copy(idx[0:1, 0:k], topk[bx:bx + 1, 0:k])
```

Survives the cut unchanged (`cut_router.py`, 18 lines); D4 maps it 1:1 to
`LOAD.2D f32 → VEC row; VTOPK idx=VEC[9], src=VEC[8], k=6; STORE.2D` (8
instructions). All 256 tokens reproduce `topk_idx` (`test_router_vtopk_matches_topk_idx`).
The index row is stored as **f32**: `STORE.2D` has no `i32` qualifier (ISA §4),
the kernel's `topk` is int32. Verdict: **extern, as permitted.**

## 4. MXFP4 weights → `LOAD.2D … f4.b32`

What TileLang does with the uint8 payload:

| Form | Result |
|---|---|
| `T.copy(Wq[uint8 128×64] → W_s[fp16 128×128])` | **Accepted** by the frontend and all passes — as a cast of 64 columns, with the shape mismatch unreported. Not a dequant. |
| `T.gemm(x_s, Wq_s[uint8], acc)` | **Refused** at trace time (`T.gemm N shape check failed: N_B = 64, N_C = 128`; dtype never reached). |
| payload copy + scales copy + explicit dequant `T.Parallel` | **Accepted**; the loop survives the cut as a `parallel(128)×parallel(128)` nest with three `Bind` scalars and one `BufferStore` into the fp16 weight tile. |

So the authoring form is the explicit dequant (Appendix A written out: low
nibble first, E2M1 magnitude table as nested `if_then_else`, `× 2^(e−127)`);
pure TileLang arithmetic, no extern, and it computes the right thing on a GPU
too. D4 does not interpret the arithmetic: a parallel store into an fp16
`tcta.wgt` tile that reads one uint8 `tcta.wgt` buffer with `N/2` columns
(payload) and one with `N/32` (scales) is the pattern; the two copies become
one `LOAD.2D f4.b32` with `scales_base/scales_stride` from the scales copy's
region and the destination from the dequant loop's store; the staging buffers
get no slot. The simulator's own dequant (`dtypes.dequant_row`) is the reference
the test compares against, so the pattern's meaning is checked end to end
(`test_every_expert_matches_reference[f4.b32]`, max rowmax error 3.7e-3).

**The finding: automatic pipeline planning gives an ISA-wrong schedule.** With
`T.Pipelined(n, num_stages=2)`, PipelinePlanning puts the two copies in stage 0
and the dequant *with the gemm* in stage 1; InjectSoftwarePipeline then
multi-buffers the uint8 staging tiles (`Wq_s[2,128,64]`) and leaves `W_s` a
single fp16 tile. On a GPU that is correct: the dequant materialises `W_s`
before the next fill overwrites the staging slot. Folded into the DMA, the fp16
tile *is* the ring slot, and the schedule `fill(nn+2) → slot nn%2; gemm(nn)
reads slot nn%2` clobbers the operand before use. The first f4 run produced
garbage with a clean token trace (the refill's LOAD→MMA token simply made the
gemm wait for the wrong data). The fix is in the frontend: manual stages,

```python
T.Pipelined(n, order=[0, 1, 2, 3], stage=[0, 0, 0, 1])   # copy, copy, dequant | gemm
```

which makes `W_s` the `[2,128,128]` ring and the staging tiles single buffers;
the mapping is then one `LOAD.2D f4.b32` per weight tile into the ring, and
the gemm reads the slot filled one iteration earlier. **Dequant-on-load means
the dequant is producer-stage work; TileLang's planner cannot know that.**

Observed in passing (not verified, no GPU): in the manual-stage body TileLang
emits `ptx_wait_group(1)` before the dequant that reads the group just
committed; on a GPU that wait looks one short. For the ISA it is irrelevant (the
dequant is a no-op; the gemm's operand is covered by the `fill_wait` of the
previous iteration), and the token checker confirms the ISA-side ordering.

Also recorded: the fixture `moe_grouped` has bf16 expert weights (`W1/W3/W2`
as bf16-valued f32); the MXFP4 image is produced by the test
(`moe_data.quantize_f4_b32`, power-of-two block scales per Appendix A) and the
reference uses the dequantized weights, since the quantisation error itself
(~0.2 relative) is far outside `tol_rowmax`. Both weight formats are tested;
the bf16 form also reproduces the fixture's `y` after the NumPy scatter-add.

## 5. SwiGLU and the combine

- `h3 *= h1` across two accumulators has no `VMUL` form (`VMUL acc, vec` is a
  row broadcast). `VGATE acc, vec, mul` is elementwise over a full tile and the
  simulator accepts an ACC tile as its `vec` operand; the ISA text says `vec ∈
  VEC`, so this stretches the documented operand. `silu(h1)·h3` is computed as
  `h3 *= h1; h1 = sigmoid(h1); h3 *= h1` to avoid a copy of `h1` (there is no
  ACC→ACC copy). In TileLang these are three `T.Parallel` loops (`*`,
  `T.sigmoid`); D4 maps them to `VGATE`, `VSIGMOID`, `VGATE`.
- `h3` is converted in place (`T.copy(H[2+n], Hc[n])` → `VCVT inplace`) and is
  the A operand of the W2 gemms (MMA A-from-ACC, §6.3); `Hc` aliases `H` with a
  slot offset of 2.
- Combine: `out *= w_s[i]` (`VSCALE_ACC` with the per-row combine weight loaded
  by `LOAD.2D f32` as a VEC column) and `STORE.2D f32` of the weighted tile to
  `Y[e, t*64.., n*128..]`; the test scatter-adds in NumPy. `STORE.IDX` is the
  one simulator gap this phase exposes, as the task predicted (not added).

## 6. Slot plan as mapped

`INP[0]` token K-tile (re-gathered per K-tile, 4 `LOAD.IDX`); `WGT[0..1]` one
weight ring for both the `[W1|W3]` stream (16 tiles) and the `W2` stream (8
tiles; the same `W_s` buffer, TileLang re-declares it per pipelined loop);
`ACC[0..3]` h1/h3 × 2 N-tiles, `ACC[4]` output tile; `VEC[0]` combine weights,
`VEC[8]` index row, `VEC[9]` heads row. Six ACC, two INP, four WGT: fits the
v0.1 device descriptor with one ACC and two WGT slots to spare.

## 7. TileLang constructs with no tile-level form (summary)

| Needed | TileLang 0.1.15 | Used instead |
|---|---|---|
| tile-level row gather | none (`tma_gather4` is sm_100a, 4 rows) | elementwise `if_then_else` gather, collapsed by D4 |
| top-k on a row | none | `T.call_extern("tcta.vtopk")` |
| dequant-on-load | none (`T.copy` uint8→fp16 is a cast) | explicit dequant loop in the producer stage, folded by D4 |
| pipelined loop over a runtime count | `T.Pipelined(dyn)` → predicated prologue/epilogue | `T.serial(n[0])` + static `T.Pipelined` inside |
| `ceil(count / 64)` | expressible (`T.ceildiv`) | phase 2: host `heads` row; phase 3: `T.ceildiv(n[0], 64)` → `LOOP r, step=64` (ISA v0.1.1) |
| elementwise ACC×ACC | `T.Parallel` multiply (fine) | `VGATE mul` with an ACC operand |
| int32 store | `T.copy` to an int32 global (fine) | phase 2: `STORE.2D f32`; phase 3: `STORE.2D i32` |
| row scatter | guarded `if dst[i] >= 0: Y[dst[i], j] = out[i, j]` in `T.Parallel` (fine, survives #19) | `STORE.IDX` (ISA v0.1.1) |

## 8. Notes on the frontend mechanics (for whoever extends the kernel)

- The eager builder source-transforms only the decorated function; helpers that
  contain `for … in T.Parallel/T.serial/T.Pipelined` must be `@T.macro`
  (otherwise `'ForFrame' object is not iterable`). Macros rename global buffers
  to their parameter names in the IR (`Wq` for both `W13q` and `W2q`), so D4
  resolves DRAM bases by PrimFunc parameter, not buffer name.
- `from __future__ import annotations` breaks the `T.Tensor` hints
  (`NameError` at trace time): the hints are evaluated by TileLang.
- Scalar lets in a `T.Parallel` body appear as `tvm.tirx.stmt.Bind` (`var`,
  `value`), not `LetStmt`; `cut.free_vars` and D4's pattern matcher handle them.

## 9. Phase 3 additions (ISA v0.1.1)

- **Dynamic trip count without host help.** `for t in T.serial(T.ceildiv(n[0], 64))`
  with `n = reduce_sum(idx_row >= 0)` survives #19 as `(n[0] + 63) // 64`
  and D4 maps it to `VCOUNT R; LOOP R, step=64`. The `heads` row is gone.
  `LOAD.IDX` zero-fills the `-1` rows of the last partial tile, `STORE.IDX`
  skips them, so no predication is needed in the body.
- **Scatter.** `for i, j in T.Parallel(64, 128): if dst_v[i] >= 0: Yp[dst_v[i], c + j] = out[i, j]`
  is the authoring form of `STORE.IDX` (guarded elementwise store from an ACC
  tile through an int32 VEC index row). The index row must be VEC-written:
  the simulator has no LOAD→STORE edge, so `T.copy(dst_s, dst_v)` (a VEC→VEC
  row copy, mapped to `VSEL dst, a, a, mask`) relays the loaded row.
- **Persistent kernel.** `for ee in T.serial(E // 2): expert(2*ee, n_a, n_b); expert(2*ee + 1, n_b, n_a)`
  around the dynamic token-tile loop is accepted through #19 (nested dynamic
  loop, three `reduce_sum`s: one before the loop, two prefetches per pair).
  D4 unrolls `ee` because the simulator applies `incr` to the innermost loop
  only; the TileLang form is the nested loop.
- **Rings for overlap (F3).** Token tile `x_s[kk % 2]` (2 INP slots), weight
  ring `W_s[· % 3]` (`num_stages=3`), output tile `out[no % 2]` (2 ACC slots).
  The W2 phase is one `T.Pipelined` over `n_o · n_k2` steps with guarded
  `fill` / scale / scatter statements so the ring spans the whole phase; the
  manual `order/stage` lists count the guarded statements. The gather stays
  outside the pipelined loop: inside it as a guarded stage-0 statement,
  TileLang multi-buffers it as if it ran every iteration (`x_s` became
  `[2, 2, 64, 128]`) and the GEMM reads a slot the gather never wrote.
- **Combine kernel.** Per 64-token tile and N-tile: `T.fill(acc, 0)`,
  `T.copy(y_shared tile, part)`, `acc += part`, then `k` times a guarded
  gather of partial rows `Yp[idx_s[i], c + j]` into `part` (ACC) and
  `acc += part` (`VGATE add`), `T.copy(acc, y)`. Six ACC-side adds and six
  `LOAD.IDX` per N-tile; no MMA. `LOAD.IDX` rounds the gathered f32 partials to
  bf16 (simulator rule), which is where the combine's 1.8e-3 error comes from.
- **MXFP4 is the fixture's format.** `moe_data.expert_weights` concatenates
  the fixture's `W1_payload | W3_payload` (and scales) along N; the bf16
  variant loads `W*_deq`, so both programs compute on identical values and
  both are checked against `y_parts`.

## 10. Phase 4 §0 additions (ISA v0.1.2)

- **Nested persistent program.** With per-level `incr` (`"region.base@1"`,
  `"region.scales_base@1"`, ISA §5.7) D4 no longer unrolls the expert loop:
  `for ee in T.serial(E // 2)` becomes `LOOP n=16 { count → R1; LOOP r=0, step=64;
  count → R0; LOOP r=1, step=64 }`, 527 instructions for 32 experts (8,025
  before). D4 fits every loop-variant operand as an affine function of the
  loop variables over a sample grid and emits one `incr` key per level; a
  static loop nested in a dynamic one, or enclosing one, is a hardware `LOOP
  n`; a static loop with no dynamic loop around or inside it is still unrolled
  (the pipelined K/N loops of the tile, 14 of them).
- **Scatter index row straight from the load.** `STORE.IDX` now reads the
  `dst_s` row that `LOAD.2D` wrote (LOAD→STORE and STORE→LOAD edges, ISA
  §8.1); the `T.copy(dst_s, dst_v)` relay (`VSEL`) is gone, one VEC
  instruction and two tokens per tile fewer. Authoring form:
  `if dst_s[i] >= 0: Yp[dst_s[i], c + j] = out[i, j]` on the loaded row.
- **f32 gathers.** `LOAD.IDX` into ACC keeps f32, so the combine's partials
  are exact and `y` is 1.519e-3 from the fixture (1.8e-3 / 2.15e-3 before).
- **One buffer per count register.** The two prefetched counts of an outer
  iteration must not share their token-list/mask fragments. A shared buffer
  makes the second count's load depend on the first `VCOUNT`'s release across
  the inner `LOOP r`; because a dynamic loop may run zero times D4 pops that
  release in front of the `LOOP`, and the LOAD queue then waits there for the
  VEC queue (2–3 idle LOAD steps per boundary, 1138 vs 1092 steps). With
  `idx_a/idx_b`, `mask_a/mask_b` each release reaches a load two sections
  later. General rule: a buffer reused on both sides of a dynamic loop by
  different queues puts a pop at the loop entry.
- **Early release of per-tile VEC inputs.** `T.copy(w_s, w_c)` right after the
  weight-column load (VEC-owned copy, mapped to `VFILL 0; VADD`) releases
  `w_s` to the LOAD queue at the start of the tile instead of after the last
  `VSCALE_ACC`; the four scales read `w_c`.
- **Prefetch or not.** `build_moe_persistent_kernel(prefetch=False)` is the
  single-register form `for e in T.serial(E): count(e); for t in …` → `LOOP
  n=32 { count; LOOP r=0 }`, 274 instructions (the ~270 the phase-4 task
  expected). Its `LOOP r` waits on `reg_pending` for its own `VCOUNT`: 3 idle
  LOAD steps per boundary, 1153 steps vs 1092 (5.6 %). The prefetch form is the
  default.
