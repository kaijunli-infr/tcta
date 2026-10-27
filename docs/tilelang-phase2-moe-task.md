# Task: phase 2 of the TileLang cut — the MoE decode kernel

Handoff instruction for a fresh session. Prerequisite: the FA prototype
(`docs/tilelang-cutoff-verdict.md`, PASS). Same repo, same environment, same
§7 rules as `tilelang-lowering-cutoff-task.md`. Read that file and its
addendum first; this one only states what is different.

## 0. Why this kernel

The FA prototype showed that TileLang's lowering is reusable through
`InjectSoftwarePipeline` for a kernel whose loop has a static trip count and
whose loads are dense 2-D copies. The product kernel is MoE decode, which is
neither: the per-expert loop count comes from the router at run time, the
token rows are gathered by an index list, and expert weights are MXFP4 with
block scales. This phase answers whether TileLang can author that kernel at
all, and where the cut lands when it can.

The three ISA features under test (ISA v0.1 §5.3, §6.1, §7):
`LOAD.IDX` with a VEC index tile and `-1` skips, `VTOPK`/`VCOUNT` feeding a
register-count `LOOP`, and the `f4.b32` dequant-on-load qualifier.

## 1. Deliverables

| # | Deliverable | Definition of done |
|---|---|---|
| E1 | `compiler/prototype/moe_kernel.py` | The MoE layer of `fixtures/moe_grouped` (B=256, d=512, d_ff=256, 32 experts, top-6, shared expert) authored in TileLang, one tile program per expert, in whichever of the three forms of §2 works |
| E2 | `docs/tilelang-moe-expressivity.md` | For each of the three ISA features: how it was expressed in TileLang (or that it could not be), which construct, and what the frontend PrimFunc looks like for it |
| E3 | `cut.py` extended, cut PrimFunc saved as `dumps/cut_moe.py` | Same four rewrites as FA; record whether `PipelinePlanning` did anything with a dynamic-extent loop |
| E4 | `tile_tir_to_program.py` extended | Maps `T.gather`-or-equivalent → `LOAD.IDX`, the dynamic loop → `LOOP r=…`, the weight copies → `LOAD.2D` with `f4.b32`, the SwiGLU elementwise → VEC/SFU |
| E5 | `tests/test_tilelang_moe.py` | For every expert: router indices from `VTOPK` equal `topk_idx`; per-expert output equals the fixture within `tol_rowmax`; token checking on; zero-token experts run zero iterations without a fault |
| E6 | Verdict appended to `docs/tilelang-cutoff-verdict.md` as "Phase 2" | PASS / PASS-WITH-CAVEATS / FAIL per §4 below, plus the list of TileLang constructs that had no tile-level form |

E1 and E2 first; they decide whether E3–E5 are worth doing.

## 2. How to express the three features (try in this order)

**Gather.** (a) TileLang's own gather/scatter if 0.1.15 has a tile-level one
(`T.copy` with an index buffer in the source region, or a `T.gather`-style
op; check `tilelang/language/`). (b) A `T.copy` whose source row index is
`idx[i]` inside a `T.Parallel(block_M)` loop: this is element-level in the
frontend and the question is whether the cut PrimFunc still shows it as a
recognisable row-gather pattern that D4 can collapse into one `LOAD.IDX`.
(c) If neither survives the cut, author the gather as an opaque extern
(`T.call_extern("tcta.load_idx", ...)`) and record that TileLang has no
tile-level gather, which is itself the finding.

**Dynamic loop count.** The expert's token count is a scalar in a buffer
(`counts[e]`). Try `for t in T.serial(counts[e])` (dynamic extent) and
`T.Pipelined(counts[e], num_stages=2)`. Record whether `PipelinePlanning`
accepts a dynamic extent (expected: it may refuse or emit a prologue/epilogue
guarded by `if`). The ISA form is `VCOUNT r, idx, bound` then `LOOP r`; D4
emits that from the loop extent expression. Zero count must produce zero
iterations with no token issued inside the loop (ISA §7).

**Router top-k.** `VTOPK` has no TileLang counterpart. Author it as
`T.call_extern("tcta.vtopk", ...)` on a `[1, E]` row tile and map it 1:1;
do not implement sort in TileLang. This is acceptable because the router is
one instruction in the ISA by design.

**MXFP4 weights.** Load with `T.copy` from a `uint8` global buffer of the
`f4.b32` payload; the scale grid is a second global buffer. The cut PrimFunc
will show two copies per weight tile; D4 maps the pair to one `LOAD.2D` with
`dtype="f4.b32"` and `scales_base/scales_stride` on the `Region`. If TileLang
refuses a `uint8` payload with a `float16` destination, keep both as `uint8`
and let the ISA qualifier do the conversion (that is what the DMA does).

**SwiGLU.** `silu(x W1) * (x W3)` → two gemms into two ACC tiles, `VSIGMOID`
+ `VMUL` on one, `VMUL` across, then the W2 gemm. The `VGATE`/`VSEL`
instructions are not needed. Expert output combine (`y[rows] += w * y_e`) is
a `STORE.IDX` with a per-row weight; v0.1 has `STORE.IDX` in the ISA but not
in the simulator. **Do not add it.** Instead store each expert's weighted
output tile to a per-expert scratch region in DRAM with `STORE.2D`, and let
the test do the scatter-add in NumPy. Record `STORE.IDX` as the one
simulator gap this phase exposes.

## 3. Slot plan and shapes

Tiles: `T_M=64` token rows, `T_K=128`, `T_N=128`. Per expert `e` with
`count[e]` tokens: `ceil(count/64)` token tiles, the last one ragged (`-1`
rows zero-filled by `LOAD.IDX`; the `rows_valid` of the final tile comes
from `count % 64`). W1/W3 are `[512, 256]` → 4 K-tiles × 2 N-tiles each;
W2 is `[256, 512]` → 2 K-tiles × 4 N-tiles. Weights stream once per expert;
token tiles stream inside. Slot allocation is yours; keep INP for token
tiles, WGT for weights with a 2-deep ring, ACC for the three accumulators.

## 4. Decision rule

- **PASS**: all three features have a TileLang authoring form (externs
  allowed for `VTOPK` only), the cut is at #19 or #18, E5 is green.
- **PASS-WITH-CAVEATS**: the gather or the dynamic loop needs an extern or
  a cut earlier than #18, but E5 is green. State which.
- **FAIL**: the kernel cannot be authored in TileLang without changing
  TileLang, or E5 cannot pass because the cut PrimFunc loses the information
  D4 needs (e.g. the index buffer or the loop extent is folded away).

A FAIL here does not reverse the FA verdict; it bounds what the TileLang
front end covers and tells us which kernels are written in TensorIR
directly against the ISA intrinsics.

## 5. Not in scope

Multi-chip dispatch (`SEND`/`RECV`), the V4.1-Flash slice shapes (d=5120,
384 experts), KDA, MLA, DSpark, changes to the ISA or simulator (`STORE.IDX`
included), autotuning, GPU work.
