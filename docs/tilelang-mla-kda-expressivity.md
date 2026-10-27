# TileLang expressivity for MLA decode and KDA (G3)

Phase 4 of the cut-off prototype (`docs/tilelang-phase4-task.md`), ISA/simulator
v0.1.2. Kernels: `compiler/prototype/mla_kernel.py` (G1),
`compiler/prototype/kda_kernel.py` (G2), plus `compiler/prototype/gemm_kernel.py`
(the GEMM family, for the Gate 1 report). Cut PrimFuncs:
`compiler/prototype/dumps/cut_mla.py`, `cut_kda.py`, `cut_gemm.py`; programs:
`dumps/program_mla.txt` (123 instructions), `program_kda.txt` (521),
`program_gemm.txt` (13). Tests: `tests/test_tilelang_mla_kda.py` (13 tests).
Versions as pinned in `compiler/prototype/PINNED.md` (tilelang 0.1.15).

Same question as the MoE record, per ISA feature: is there a TileLang construct
for it, does it survive passes 1–19, and does the cut PrimFunc still carry what
D4 (`tile_tir_to_program.py`) needs. The summary table is §12.

## 1. Wide accumulator → `ACC.W` (MLA)

**What TileLang has.** Nothing special is needed: a fragment can be any shape,
and a `T.gemm` destination can be a column slice of it.

```python
acc_o = T.alloc_fragment([64, 512], "float32")
T.fill(acc_o, 0)
...
for i, j in T.Parallel(64, 512):
    acc_o[i, j] *= scores_scale[i]                # the online-softmax rescale
for c in T.serial(4):
    T.gemm(acc_s_cast, K_shared[c, :, :], acc_o[:, c * 128:(c + 1) * 128])
...
for i, j in T.Parallel(64, 512):
    acc_o[i, j] /= logsum[i]
T.copy(acc_o, O)
```

**After the cut** (`cut_mla.py`): the fragment is one `(64, 512)` buffer in
`tcta.acc`; the gemm destinations arrive as `T.region(acc_o[0, c * 128], 3, 64,
128)` with the column offset in the region's base index (and repeated as the
gemm's `offset` trailing argument); the fill and the two elementwise loops run
over the whole `64 × 512` fragment; the final copy is one region of `64 × 512`.
Nothing before #19 splits the fragment (LayoutInference, which would, is after
the cut).

**D4 mapping.** An ACC fragment wider than `T_N` gets `cols // T_N` consecutive
slots (`acc_o → ACC[1..4]`). `T.fill` → four `VFILL`; a gemm into a column
slice → `MMA acc=ACC[1 + c]` (the slice folds to a constant slot once the chunk
loop is unrolled); the two elementwise scalings → one `VSCALE_ACC acc=ACC[1],
wide=True` each (the `1/logsum` form is the FA pattern `VRCP` + `VSCALE_ACC
wide`); the final copy → one `STORE.2D src=ACC[1], wide=True` of the f32
`[64, 512]` tile. The reads/writes of the wide ops cover all four slots, so the
tokens are the same reader/writer rule as a plain tile: the rescale waits for
the four PV gemms of the previous KV tile, the four gemms wait for the rescale.

**Verdict: expressible without a new construct; survives #19 as one
fragment; mapped with the `wide` qualifiers the task described.** `MMA
wide=1` itself is not used: the array's B tile is `T_N = 128` wide, so a
`[64, 512]` output is four MMAs into the four slots, which is what the sliced
gemm says.

## 2. The 576-deep reduction and V as K's columns (MLA)

K rows are `[c_kv | k_rope]` = 576 columns = 4.5 tiles. The kernel runs the
reduction as one 64-deep chunk (columns 512..575, partial INP/WGT tiles) and
four 128-deep chunks:

```python
T.copy(Q[:, 512:576], Q_shared[0, :, 0:64])
T.copy(K[rows, 512:576], K_shared[3, :, 0:64])
T.gemm(Q_shared[0, :, 0:64], K_shared[3, :, 0:64], acc_s, transpose_B=True)
for c in T.serial(4):
    T.copy(Q[:, c*128:(c+1)*128], Q_shared[(c + 1) % 2, :, :])
    T.copy(K[rows, c*128:(c+1)*128], K_shared[c, :, :])
    T.gemm(Q_shared[(c + 1) % 2, :, :], K_shared[c, :, :], acc_s, transpose_B=True)
```

The four 128-column K chunks are exactly the V tile of the same rows, so they
stay in the 4-slot WGT ring and the PV gemms read them (`T.gemm(acc_s_cast,
K_shared[c, :, :], acc_o[:, c*128:(c+1)*128])`): V is never re-read, one
`T.copy` of the K rows serves both products (the task's "one copy and a sliced
gemm operand" option). The rope chunk borrows WGT slot 3 before its V occupant
arrives (the ring index is `n_full - 1` by construction, D4 resolves it
statically). Partial-tile copies survive as regions with `rows_valid = 64,
cols_valid = 64` and map to `LOAD.2D` with those validities and an `MMA.T`
over a 64-deep contraction.

**What does not fit: Q.** The stacked Q tile is 576 columns = 5 INP tiles; the
KV260 descriptor has 2 INP slots, so Q is re-streamed from DRAM at every KV
tile through a 2-slot ring: 10 `LOAD.2D` per KV tile (5 Q, 5 K) instead of 5,
and the program reads 2.36 MB for a 1.18 MB latent cache. This is a device
parameter, not an ISA gap: with ≥ 5 INP slots the five Q loads move in front
of the loop (the FA form). Recorded for the Gate 1 report as an INP-depth
question for the cycle model, together with the MoE finding that the LOAD
queue is the one that saturates.

## 3. Online softmax over the wide tile (MLA)

Identical to `fa_kernel.py` (`reduce_max`, `exp2` with the folded
`log2(e)/sqrt(576)` scale, `reduce_sum`, `logsum` update, bf16 cast of P in
place). The one difference is that the rescale and the final normalisation
touch 512 columns, and both are a single `VSCALE_ACC wide` instead of four
instructions. Per KV tile the program issues 10 LOAD, 9 MMA and 18 VEC
instructions; over the run the VEC queue is busy on 296 of 451 steps (LOAD
160, MMA 144): at equal issue costs MLA decode is VEC-issue-bound, the
opposite of the LOAD-bound MoE tile. Numerics: `O` rowmax error 2.15e-3
against `tol_rowmax` 7.8e-3.

## 4. KDA: from the recurrence to the chunk form

`ref/kernels.py::kda_recurrent` is the frozen reference and was not changed
(the pinned gate placement, α applied to the state before the delta-rule
update and the same α inside `k_t^T diag(a_t) S_{t-1}`, is what the chunk
algebra below assumes; the exact-arithmetic check confirms it):

    S_t = diag(a_t) S_{t-1} - b_t k_t (k_t^T diag(a_t) S_{t-1}) + b_t k_t v_t^T
    o_t = S_t^T q_t

With G_i = Σ_{s≤i} log2 a_s inside the chunk (cumulative, inclusive; per
channel), K̂ = K ⊙ 2^G, K̄ = K ⊙ 2^−G, Q̂ = Q ⊙ 2^G and B = diag(β):

    N = (K̂ K̄ᵀ) ⊙ M_strict · B            strictly lower 64 × 64
    U = (I + N)⁻¹ B (V − K̂ S)              S = state entering the chunk
    O = Q̂ S + ((Q̂ K̄ᵀ) ⊙ M_incl) U
    S ← diag(2^G_last) S + (Kᵀ ⊙ 2^(G_last − G_j)) U

In float64 this reproduces the reference's `S_final` and `o` to 1e-6
(`scratchpad kda_proto.py`, exact-algebra check), so the fixture, the gate
placement and the derivation agree; no regeneration was needed. The kernel
takes the gate in the log2 domain (`g = log2 α`, bf16, the interface of the
Kimi Linear kernels) and two extra DRAM layouts prepared by the host: `gt =
gᵀ` and `kt = kᵀ` (`[d, T]`), because the state update contracts over tokens
and the ISA has no transposed load for the A operand (only `transpose_B`).

## 5. The masked GEMM: `MMA.MASKED` is not usable here

The task's mapping (gemm followed by an elementwise add of a 0/−inf mask,
fused to `MMA.MASKED`) suits a softmax, where −inf becomes zero after the
exponential. In the delta-rule chunk the masked products feed further GEMMs
(the Neumann series, `W U`) and additions; a −inf entry poisons them. The
masks are therefore multiplicative 0/1 tiles:

```python
T.gemm(Kh_c, kbar_w, Tm[0, :, :], transpose_B=True)
for i, j in T.Parallel(64, 64):
    Tm[0, i, j] = Tm[0, i, j] * ms_v[i, j]          # M_strict, an f32 [64, 64] VEC tile
```

`ms_v`, `mi_v` (M_incl) and `eye_v` (I) are loaded once before the chunk loop
(`T.copy(mstrict, ms_v)` → `LOAD.2D f32` into 2-D VEC tiles) and applied with
`VGATE acc, vec, mul` / `add`; the simulator's `VGATE` already accepts a 2-D
VEC source (ISA §5.3 names it for ACC sources; the VEC form is what the
instruction's name says). D4 extends its `VGATE` pattern to `tcta.vec`
sources. Cost: one VEC instruction per masked product, two per chunk, and no
`MMA.MASKED` in the program (`test_kda_program_shape` asserts it).
**Finding for the ISA:** a zero-mask (multiplicative) variant of `MMA.MASKED`,
or simply documenting `VGATE mul` with a VEC mask as the triangular idiom,
would make the task's "masked GEMM" a one-instruction form again.

## 6. The triangular solve (Appendix B)

Authored as the repeated-squaring product of the Neumann series: with
T₀ = −N (strictly lower, so N⁶⁴ = 0) and P₀ = I + T₀,

    T_{s+1} = T_s²,   P ← P (I + T_{s+1}),   s = 0..4

gives P = Σ_{j<64} (−N)^j = (I + N)⁻¹ exactly: six factors, the task's "6
steps", as an initial `P = I − N` plus a static `for s in T.serial(5)`.

```python
for s in T.serial(N_SOLVE):
    T.copy(Tm[s % 2, :, :], ab_s)                    # bf16 A operand into INP, T_s stays f32
    T.clear(Tm[(s + 1) % 2, :, :])
    T.gemm(ab_s, t_w, Tm[(s + 1) % 2, :, :])         # T_{s+1} = T_s T_s
    T.copy(Tm[(s + 1) % 2, :, :], scr_t); T.copy(scr_t, t_w)   # T_{s+1} as the next B operand (§7)
    T.copy(P, pb_s)
    T.gemm(pb_s, t_w, P)                             # P += P T_{s+1}
```

The loop survives #19 as a static serial loop of extent 5 (`test_kda_cut_...`)
and D4 unrolls it (a static loop inside a hardware loop with no dynamic loop
below it). The body is two GEMMs, three VEC converts and one DRAM round trip;
the task's "no loads" holds only for a machine that can feed a B operand
from ACC (§7). `P` accumulates in place (`T.gemm(..., P)` with the default
`clear_accum=False` → `MMA accumulate=True`), the `I + T` factor costs
nothing because `P·I = P` is the accumulator itself. Numerics: T_s shrinks
quadratically (the stored T₅ is ≈ 1e-8), `P` is good to bf16 rounding of its
inputs; `U = P R` matched the NumPy model with the same roundings to
3.6e-3 and the exact solve well inside tolerance (§9).

## 7. Computed B operands → DRAM round trips

The array's B operand is read from WGT or STATE, and only `LOAD` writes
those scratchpads (ISA §1 table). Every matrix computed on the device and
then used on the B side — K̄ᵀ (`kbar_w`), the five T_{s+1} (`t_w`), R and U
as hi/lo pairs (`r_w`, `u_w`) — goes `STORE.2D bf16` to a DRAM scratch region
and `LOAD.2D` back:

```python
T.copy(Kb, scr_kbar)          # STORE.2D (ACC -> DRAM, bf16)
T.copy(scr_kbar, kbar_w)      # LOAD.2D (DRAM -> WGT)
```

Both are ordinary `T.copy`s on `global` scratch buffers that the host never
reads, and they survive #19 verbatim. D4 treats each DRAM buffer as a
token-tracked object, so the load waits for the store (STORE→LOAD, the
v0.1.2 edge) and the next store into the same region waits for the load
(LOAD→STORE): 49 + 42 tokens in the program, 13 round trips per chunk (1 K̄,
5 T, 2 R, 2 U, plus the 2 checkpoint halves and `o`, which are real outputs).
At equal issue costs the round trips are 26 of the chunk's 140 work
instructions and 1.3 MB of the program's 1.7 MB DRAM writes for 256 tokens.
**Finding for the ISA (v0.2 candidate):** an ACC→WGT move (`VCVT dst=WGT`, or
an MMA that reads B from ACC) removes all 13 per chunk and the STORE→LOAD
latency from the solve's critical path. The A side has no such problem:
`VCVT dst=INP` (D4's non-inplace acc→inp copy) feeds `P` and `T_s` to the
array while keeping their f32 values.

## 8. Resident state: two ACC tiles plus a STATE mirror

The `[128, 128]` f32 state is carried across the chunk loop in two `[64, 128]`
ACC tiles (`S_acc[2, 64, 128]`, `ACC[0..1]`), decayed with `VSCALE_ACC` (the
per-channel 2^G_last) and updated with three accumulating MMAs per half; it is
never reloaded. The STATE scratchpad holds the f32 copy used as the **B
operand** of `K̂ S` and `Q̂ S`: `T.copy(S_all[c, :, :], S_st)` → `LOAD.2D f32
[128, 128]` into `STATE[0]` (64 KB per chunk) from the DRAM mirror written at
the end of the previous chunk, which is also the per-chunk checkpoint the
fixture asks for (`S_all[c + 1]`).

Why not the task's picture (STATE written by MMA and VEC, read by MMA)? The
array writes at most `T_M = 64` rows per MMA and `VSCALE_ACC`/`VGATE` operate
on 64-row tiles, so a 128-row state is updated in two halves, and the ISA has
no row-offset form of a destination (`STATE(0)[64:128]`). Splitting the state
into two ACC tiles is the form that needs no new operand; the price is the
DRAM mirror and the STATE reload per chunk (one LOAD.2D, 64 KB). **Finding:**
either a row-offset destination for MMA/VEC on STATE, or 128-row state
operations, would let the resident state live in STATE alone. The
`T.alloc_shared` + `tcta.state` scope survives #19 like the other scopes
(`test_kda_cut_keeps_state_scope_and_chunk_loop`).

## 9. Numerics: bf16 at every array input, and the hi/lo splits

The simulator rounds every A and B operand to bf16 (f32 accumulate). With
that applied to every GEMM of §4 the NumPy model gives `o` 8.17e-3 rowmax
error — above the 7.8e-3 tolerance — and `S` ≤ 6.7e-3. The sensitive
operands are U (feeds O and S), R (feeds U) and K̄ᵀ-side Kᵀ (feeds S). Each is
fed as a bf16 pair `x_hi = bf16(x)`, `x_lo = x − x_hi`:

```python
T.copy(R, Rh)                                 # bf16 copy, R kept (VCVT non-inplace)
for i, j in T.Parallel(64, 128):
    Rl[i, j] = R[i, j] - Rh[i, j]             # VFILL 0; VGATE add R; VMUL -1 Rh; VGATE add; VMUL -1
T.gemm(pb_s, r_w[0, :, :], U); T.gemm(pb_s, r_w[1, :, :], U)
```

Result on the simulator: `o` 5.99e-3, `S` checkpoints 1.46 / 1.36 / 1.15 /
1.27e-3 — the NumPy model's prediction to three digits. Cost: 27 MMAs per
chunk instead of 20, and the splitting arithmetic on the VEC queue. Splitting
Q̂ as well would bring `o` to 4.9e-3 at 2 more MMAs; not done. **Finding:**
the delta-rule kernel is the first of the five families whose plain bf16
schedule misses the tolerance; a bf16×2 (or f32) operand mode on the array, or
the host-side decision that the Kimi Linear kernels are fp16/f32 at these
points, belongs in the Gate 1 open items. D4 recognises the split idiom
(`dst = a - b` over ACC tiles, non-inplace bf16 copy when the source is read
afterwards); the in-place `VCVT` stays the mapping whenever the f32 source is
dead.

## 10. The log-domain gate

- G = cumulative sum of `log2 α` inside the chunk: `T.gemm(tri_s, g_w, G)`
  with the lower-triangular ones matrix resident in INP and the 64 × 128 gate
  rows loaded per chunk (`LOAD.2D` into WGT). No `cumsum` tile op exists in
  TileLang and none is needed.
- K̂ = K ⊙ 2^G: `Kh[i, j] = Kh[i, j] * T.exp2(G[i, j])` → `VGATE acc, G, decay`
  (the ISA's `acc *= exp2(gate)` with an ACC gate). Q̂ the same; K̄ with the
  negated gate (`G[i, j] = -G[i, j]` → `VMUL imm=-1`, see §11).
- 2^(G_last − G_j) for the state update, per channel row: `Gx = Gtᵀ-chunk ·
  triu` (the same cumsum on the transposed gate), `Gx = −Gx`, `reduce_max`
  → `m = −G_last` (G is non-increasing along tokens), `Gx −= m` (`VSUB`),
  then `Kt *= exp2(Gx)` (`VGATE decay`) and `m = exp2(−m)` for the state
  decay (`VMUL −1`, `VEXP2` on the row vector, `VSCALE_ACC`).

Everything above survives #19 as elementwise `T.Parallel` stores and
`tl.tileop` calls; `T.exp2` and the negation arrive unsimplified (`Mul(x,
Call(exp2, g))`, `x * -1`), which is what D4 matches.

## 11. Mapping idioms added to D4 for this phase

| TileLang form | After #19 | Program |
|---|---|---|
| f32 `[64, 512]` fragment, sliced gemm destinations | one buffer, `T.region(acc_o[0, c*128], …)` | 4 consecutive ACC slots, `MMA acc=ACC[1+c]`, `VSCALE_ACC wide`, `STORE.2D wide`, 4 × `VFILL` |
| `T.copy(acc_f32, acc_bf16)` with the f32 source read later | `tl.tileop.copy` | `VCVT` non-inplace into the bf16 buffer's own slot (in place when the source is dead, as before) |
| `T.copy(acc, inp_shared)` | copy acc → inp | `VCVT dst=INP` (A operand without destroying f32) |
| `T.copy(acc_f32, acc_f32)` | copy | `VFILL 0; VGATE add` |
| `acc *= exp2(acc2)` | `Mul(load, exp2(load))` | `VGATE decay` |
| `acc (*|+)= vec2d[i, j]` | elementwise with a `tcta.vec` 2-D source | `VGATE mul/add`, VEC source |
| `x = -x` (ACC or VEC row) | `x * -1` | `VMUL imm=-1` (the simulator's `VNEG` handler requires an `imm`/`vec` operand the assembler never emits: finding) |
| `acc[i, j] -= vec[i]` | `Sub(load, load)` | `VSUB` |
| `dst = a - b` over ACC tiles | `Sub`, cast on the bf16 operand | copy a; `VMUL -1` b; `VGATE add`; `VMUL -1` b |
| `v = exp2(v)` (row vector) | `Call(exp2)` | `VEXP2` |
| gather into an f32 fragment | `Cast(if_then_else(...))` | `LOAD.IDX` into ACC (f32 kept, v0.1.2) |
| `T.copy(global_scratch, wgt)` after `T.copy(acc, global_scratch)` | two copies | `STORE.2D`; `LOAD.2D` with STORE→LOAD / LOAD→STORE tokens (DRAM buffers are token objects) |
| `tcta.state` scope | `T.sblock_alloc_buffer(..., scope="tcta.state")` | `LOAD.2D` into `STATE[0]`, read as `MMA b=STATE[0]` |
| more live buffers than slots | — | live-range slot sharing (`report.shared_slots`); programs that fit are unchanged |

None of these is an extern. The gemm with a STATE B operand and the sliced
wide destinations needed no change to `cut.py`.

## 12. Slot plan and device configuration (KDA)

The KDA chunk has 21 ACC-scoped buffers (state halves, the three decayed
operands, V, two solve tiles, P, R/U and their splits, O, W, the transposed
gate and K tiles and their splits), 6 WGT operands (two of them pairs), 3 INP
tiles, 5 two-dimensional VEC tiles and 2 index rows. The KV260 descriptor
(INP 2, WGT 4, ACC 6, VEC 16, STATE 1, FIFO 4) cannot hold it even with slot
sharing (`test_kda_needs_config_c`); the kernel is mapped and run with the ISA
§11 config-C descriptor (`CONFIG_C`: INP 8, WGT 16, ACC 16, VEC 64, STATE 2,
FIFO 16). After live-range sharing the program uses INP 0..2, WGT 0..7,
ACC 0..7 (14 buffers share a slot with another), VEC 0..4 + 8..9, STATE 0; the
per-edge FIFO maximum is 8 (MMA→LOAD). Sharing is textual: D4 colours buffers
over their statement ranges, with a buffer read before it is written inside a
loop body live over the whole loop; the token pass then serialises the shared
slot's readers and writers like any other WAR/WAW pair. Programs that fit
their descriptor keep the one-slot-per-buffer plan, so the FA, MoE and MLA
programs are unchanged (the suite's instruction and token counts hold).

## 13. Summary: what survived #19

| ISA feature | TileLang form | Through #19 | D4 | Verdict |
|---|---|---|---|---|
| wide accumulator `ACC.W` | `[64, 512]` fragment, sliced gemm dsts | one fragment | 4 slots + `wide` ops | expressible, no new construct |
| 576-deep reduction, V = K columns | 1 partial + 4 full chunks, one K copy | verbatim | `LOAD.2D` partial, `MMA.T`, resident WGT ring | expressible; Q re-streamed (INP depth) |
| `MMA.MASKED` | not used; `T.gemm` + `× 0/1 mask` | verbatim | `VGATE mul` with a VEC mask | expressible; the −inf form is the wrong primitive for the delta rule |
| triangular solve | `P = I − N`; static `T.serial(5)` of two gemms | static loop kept | unrolled; 10 MMA + round trips | expressible; needs §7 |
| resident `STATE` | `tcta.state` B mirror + two ACC tiles | scope kept | `LOAD.2D` STATE, `VSCALE_ACC`, 3 MMA per half | expressible; row-offset / 128-row ops would remove the mirror |
| `VGATE decay` | `x *= T.exp2(g)` | unsimplified `Mul/Call` | `VGATE decay` | expressible |
| computed B operands | `T.copy` to/from a global scratch | verbatim | `STORE.2D`/`LOAD.2D` + the v0.1.2 edges | expressible; costly (ISA finding) |
| bf16 numerics | hi/lo pairs via `T.copy` + elementwise sub | verbatim | `VCVT` non-inplace, `acc_sub` | needed to meet tolerance (finding) |
| partial-tile GEMM | sliced `T.copy`, `[50,100]`/`[100,96]` tiles | verbatim | `LOAD.2D` validities, one `MMA` | expressible |
