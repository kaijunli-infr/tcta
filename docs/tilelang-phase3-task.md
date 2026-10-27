# Task: phase 3 — persistent expert loop, combine, and the ISA v0.1.1 sync

Prerequisites: phases 1 and 2 (`docs/tilelang-cutoff-verdict.md`, both PASS).
Same environment, same §7 rules, with one change: **the ISA and simulator
have been updated by the main session** (see §0). Everything else holds.

## 0. Sync first (half a day)

The main session changed `isa/`, `sim/`, `ref/`, `tests/` in response to the
phase-2 findings. Replace those four directories with the attached copies,
then:

1. Re-run `pytest tests -q`. Expected: the FA and MoE TileLang tests still
   pass (the simulator additions are backward compatible: `Machine.reset()`,
   `STORE.IDX`, `i32` store qualifier, `LOOP r, step`, `VGATE` with an ACC
   operand, `asm.Program.vgate(acc, vec, gate_op)` fixed, `storeidx()` added).
2. Regenerate `fixtures/` (`python ref/kernels.py fixtures`). `moe_grouped`
   now has **f4.b32 expert weights as the normative values** (`W*_payload`,
   `W*_scales`, plus `W*_deq`), `y_parts[token, slot, :]` and `dst_rows`
   (= `token·k + slot`) for the combine; `flash_attn_gqa` defaults to
   `causal=False`. Update `moe_data.py` and `test_tilelang_moe.py` to read the
   fixture's f4 images instead of building their own, and to check the f4
   path against `y_parts`, which retires phase-2 caveat 5.
3. Replace the host-prepared `heads[e, t]` row with `LOOP r=count, step=64`
   (`VCOUNT` on `idx_lists[e]` gives `count`). Retires caveat 1. The last
   iteration's `-1` rows are already zero-filled by `LOAD.IDX`.
4. Replace the `STORE.2D` + NumPy scatter-add combine with `STORE.IDX` to a
   `[B·k, d]` buffer at `dst_rows`. Retires the `STORE.IDX` simulator gap.
5. Record in the verdict's Phase 2 section which caveats are now closed.

## 1. Deliverables

| # | Deliverable | Definition of done |
|---|---|---|
| F1 | `compiler/prototype/moe_combine_kernel.py` | The reduce kernel: for each 64-token tile, `LOAD.IDX` the `k` rows per token from the `[B·k, d]` buffer, weighted sum in VEC with `topk_w`, add the shared expert, `STORE.2D` the result. Authored in TileLang, cut at #19, mapped, tested against `fixtures/moe_grouped::y` |
| F2 | `compiler/prototype/moe_persistent.py` | One program for **all** experts: a static outer `LOOP n=E` over experts with the dynamic `LOOP r, step=64` inside; weights stream per expert; the token-tile gather for expert `e+1` must overlap the GEMMs of expert `e`. Authored in TileLang if the nested dynamic loop survives the cut; otherwise emitted directly as a `Program` by extending `tile_tir_to_program` with an expert-loop wrapper, and say so |
| F3 | `docs/overlap-report.md` | Evidence of overlap from the simulator's step trace: for each expert boundary, the step at which the LOAD queue issued the first `LOAD.IDX` of expert `e+1` versus the step at which MMA finished expert `e`. Add a `trace` list to `Machine` (step, queue, op) if needed — that is a simulator change the main session approves in advance, under 30 lines, no semantic effect |
| F4 | `tests/test_tilelang_moe.py` extended | F1 and F2 green with token checking; `y` matches the fixture within `tol_rowmax` end to end (persistent program → combine program) |
| F5 | Phase 3 section in the verdict | PASS/FAIL per §3; and one paragraph answering: does a decoupled machine need software pipelining in the instruction stream at all, given F3? |

## 2. Slot plan for F2 (the thing phase 2 did not have)

INP: a **2-slot ring for token tiles** (`dst_mod 2` on the gather). The
release token for slot `s` is pushed after the last MMA that reads it (the W3
GEMM of that tile), and the gather into slot `s` pops it. With that, the LOAD
queue runs ahead into the next tile, and across the expert boundary into the
next expert's first tile, without any prologue/epilogue code: that is ISA §15,
bullet 1, and F3 is the test of it. Weights keep the 2-deep WGT ring from
phase 2.

The combine weight column and `VCOUNT`'s index row are per-expert VEC tiles;
give them 2 slots too or the LOAD→VEC token will serialize the expert loop.

## 3. Decision rule

- **PASS**: F1 and F2 authored in TileLang through the #19 cut; F3 shows
  the first gather of expert `e+1` issued before the last MMA of expert `e`
  for every boundary with a non-empty next expert.
- **PASS-WITH-CAVEATS**: F2 needs the expert-loop wrapper outside TileLang
  (nested dynamic loop does not survive the cut), but F3 holds.
- **FAIL**: F3 shows no overlap even with the 2-slot ring. That would mean
  the token placement rule is wrong, and the verdict must say where the
  serialization comes from.

## 4. Not in scope

Multi-chip (`SEND`/`RECV`), V4.1-Flash shapes, KDA/MLA, the cycle model.
Any simulator change beyond the trace list in F3 goes back to the main
session as a finding.
