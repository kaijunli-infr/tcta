# Task: phase 4 — MLA decode and KDA through the cut; Gate 1 closure

Prerequisites: phases 1–3 (all PASS). Same environment, same §7 rules.
ISA/simulator are now **v0.1.2** (sync per §0). After this phase every kernel
family in the plan has run through TileLang → #19 cut → ISA → simulator,
which is the Gate 1 exit condition for the compiler and ISA.

## 0. Sync (half a day)

Unpack the attached `isa/`, `sim/`, `tests/test_sim.py`. Changes, all
responding to phase-3 findings:

| Finding | Change |
|---|---|
| `incr` innermost-only → expert loop unrolled | **Per-level increments**: `incr={"region.base": s0, "region.base@1": s1, "dst_mod": 2}`; `@L` = loop level from innermost (0). Also `region.scales_base`. |
| No LOAD↔STORE edge | `(LOAD, STORE)` and `(STORE, LOAD)` added to `EDGES` and ISA §8.1 |
| `LOAD.IDX` rounds to bf16 into ACC | Rounds only for INP/WGT; ACC/VEC keep f32 |
| `PUSH`/`POP` cost an issue slot | **Free sync** (ISA §8.5): a queue drains leading syncs in the same step as its next work instruction; `stats["sync"]` counts them. Step totals are now an issue-slot proxy. |
| FIFO level 6 > depth 4 | Left as is; depth is a device parameter (config C: 16). Keep reporting the per-edge max. |

Then:
1. `pytest tests -q` must be green before anything else.
2. Re-emit the persistent MoE program as **one `LOOP n=E` with `LOOP r, step=64` inside** using per-level `incr` (no unrolling). Expect ~270 instructions instead of 8,025. Re-run the overlap report; numbers and F3 must be unchanged.
3. Re-run the overlap report's step counts under the free-sync rule and replace the "not a performance figure" caveat with the new counts (they are now an issue-slot proxy; still not cycles).
4. Replace the `VSEL` copy idiom for the scatter index row with a direct `LOAD → STORE.IDX` token (the new edge).

## 1. Deliverables

| # | Deliverable | Definition of done |
|---|---|---|
| G1 | `compiler/prototype/mla_kernel.py` | MLA decode, absorbed form, from `fixtures/mla_decode_absorbed`: 16 heads × 4 rows stacked into one 64-row Q tile, K = `[L, 576]` streamed as 9 K-tiles of 64 → `S [64, Bk]` in `ACC`, online softmax, `O [64, 512]` in the **wide ACC** (`wide=1`, 4 consecutive slots), V = the first 512 columns of the same K rows (two `T.copy`s from the same DRAM rows, or one copy and a sliced gemm operand). Cut at #19, mapped, matches `O` within `tol_rowmax` |
| G2 | `compiler/prototype/kda_kernel.py` | KDA chunkwise from `fixtures/kda_chunk64`: per 64-token chunk, intra-chunk `MMA.MASKED` + the Appendix B Neumann triangular-solve macro, `VGATE decay` on the resident `STATE` tile, inter-chunk update `S ← α·S + Kᵀ·U`, output `O = Q·S`. `STATE` persists across the chunk loop (`LOOP n=T/C`) and must match every `S_chunks[c]` checkpoint and the final `o` |
| G3 | `docs/tilelang-mla-kda-expressivity.md` | Same format as the MoE expressivity record: how the wide accumulator, the masked GEMM, the resident state and the triangular solve were expressed in TileLang, and what survived #19 |
| G4 | `tests/test_tilelang_mla_kda.py` | G1 and G2 green with token checking |
| G5 | Phase 4 section in the verdict + `docs/gate1-report.md` | Gate 1 report: for each of the five kernel families (GEMM, FA/GQA, MLA, KDA, MoE+combine) the instruction count, token-graph summary, numerics, and the list of ISA features it exercises; the consolidated v0.2 candidate list; and the open items the cycle model must answer (ring depths, FIFO depths, gather efficiency, VEC/SFU sizing) |

## 2. Things to know before starting

**Wide accumulator.** `MMA ... wide=1` on `ACC(i)` treats slots `i..i+3` as
one `[64, 512]` tile; `VSCALE_ACC` and `STORE.2D` accept `wide`. TileLang will
see a single `acc_o[64, 512]` fragment; D4 maps it to the wide form when the
accumulator's N exceeds `T_N`. `VCVT inplace` on `S` is unchanged. The
simulator's `_read_tile(..., wide=True)` concatenates the four slots.

**Masked GEMM.** `MMA.MASKED acc, a, b, mask` adds a VEC mask tile (0/−inf)
before writeback. In TileLang author the mask as a `[64, 64]` fragment built
with `T.Parallel` and `if_then_else` on `i > j`, and the GEMM as `T.gemm`
followed by an elementwise add of the mask; D4 fuses the pair when the add
immediately follows the gemm and the mask buffer is loop-invariant.

**Resident STATE.** `STATE(0)` is a `[128, 128]` FP32 tile written by MMA and
VEC and read by MMA across loop iterations. Its tokens are the same
reader/writer rule as ACC; the first write before the loop (`VFILL 0`) is a
pre-loop push, consistent with the phase-3 level-invariant placement. KDA's
`d = 128` means one STATE tile; do not generalize to multi-tile state here.

**Triangular solve.** Appendix B gives the macro: `T = −L` (masked), then 6
steps of `MMA.MASKED` + `VADD`. Author it in TileLang as a static
`T.serial(6)` over two fragments; the only new mapping is that the loop body
is two GEMMs and two VEC ops with no loads, so it carries no LOAD tokens.
Numerics: it is exact for strictly lower-triangular `L` of size ≤ 64, so the
checkpoint comparison should sit well inside tolerance; if it does not, the
gate placement in `kda_recurrent` is the suspect, not the macro.

**KDA gate placement.** `ref/kernels.py::kda_recurrent` is the frozen
reference. If the Kimi Linear commit you pinned places α differently, change
that function, regenerate, and record the change in G3; never adapt the
kernel to a reference you have not regenerated.

## 3. Decision rule

- **PASS**: G1 and G2 authored in TileLang through #19; G4 green; G5 written.
- **PASS-WITH-CAVEATS**: either kernel needs an extern beyond `VTOPK`-class
  (one instruction with no TileLang form) or a mapping idiom not in the
  current pattern set, but G4 is green. Name it.
- **FAIL**: the wide accumulator or the resident state cannot be expressed
  so that #19 preserves it, or the checkpoint comparison fails for a reason
  traced to the ISA rather than the reference. Either is a Gate 1 blocker
  and goes back to the main session with the cut PrimFunc attached.

## 4. Not in scope

Multi-chip, V4.1-Flash shapes (576-wide KV entries via `LOAD.IDX col0`
windows are phase 5 with the cycle model), FP8/MX attention, DSpark,
autotuning, the cycle model itself. No ISA or simulator changes; findings go
to the main session as before.
