"""MoE decode layer of fixtures/moe_grouped in TileLang (E1, phase 3 F1/F2).

Shapes: B=256 tokens, d=512, d_ff=256, 32 experts, top-6. Tiles T_M=64, T_K=T_N=128.
Three kernels:

- `build_moe_expert_kernel`     one tile program per expert (grid = experts).
- `build_moe_persistent_kernel` one program for all experts: a static loop over experts
                                with the dynamic token-tile loop inside (F2).
- `build_moe_combine_kernel`    the reduce: per 64-token tile, gather the k expert parts
                                of each token from the [B*k, d] scatter buffer, add the
                                shared expert, store y (F1).

How the ISA features are authored (details in docs/tilelang-moe-expressivity.md):

- Token gather (LOAD.IDX): TileLang has no tile-level gather; the elementwise form
      x_s[i, j] = if_then_else(idx_s[i] >= 0, x[idx_s[i], kk*128 + j], 0)
  survives the cut verbatim and D4 collapses the loop nest into one LOAD.IDX.
- Dynamic trip count (VCOUNT + LOOP r, step): the count of non-negative entries of the
  expert's token list is `n = reduce_sum(idx_all >= 0)` (VCOUNT's definition) and the
  token-tile loop is `for t in T.serial(T.ceildiv(n[0], 64))`, which the cut keeps as
  `(n[0] + 63) // 64` and D4 maps to `LOOP r, step=64` (ISA v0.1.1 §7). No host-side
  ceil, no `heads` row (phase-2 caveat 1 closed).
- Scatter (STORE.IDX): the weighted output tile goes to rows `dst[e, tile rows]`
  (= token*k + slot) of the [B*k, d] buffer:
      if dst_s[i] >= 0: Yp[dst_s[i], no*128 + j] = out[i, j]
  inside T.Parallel; D4 collapses it into one STORE.IDX (-1 rows skipped).
- Router top-k (VTOPK): `build_router_kernel`, `T.call_extern("tcta.vtopk")`.
- MXFP4 weights (f4.b32): payload + scales copies and an explicit dequant loop in the
  producer stage of a manually staged T.Pipelined (TileLang's automatic planning puts the
  dequant with the gemm, which is wrong once dequant is folded into the load).

Persistent program (F2): TileLang's static `for ee in T.serial(E // 2)` around two
dynamic token-tile loops survives the cut; the per-expert operands (weight bases, list
rows) vary with ee and D4 emits them as per-level increments (`region.base@1`, ISA
v0.1.2 §5.7) inside one hardware LOOP n=E/2 with the two LOOP r, step=64 inside (v0.1.1
applied increments to the innermost loop only, which forced an unrolled program of
8,025 instructions). Two experts per iteration because the count prefetch alternates
between two count registers. The count of expert e+1 is computed at the top of expert
e's section into the other count register
(VEC is an in-order queue: a VCOUNT issued after expert e's VEC work would execute only
after expert e's last MMA, and the front end could not enter expert e+1's loop before
that). The gather alternates between two INP slots (K-tile parity), so the LOAD queue
gathers the next K-tile, the next token tile and the next expert's first tile while the
MMA queue works (ISA §15 bullet 1); the per-tile combine-weight and destination rows are
loaded after the weight fills of the tile so their WAR tokens never hold the LOAD queue.

SwiGLU: h3 *= h1; h1 = sigmoid(h1); h3 *= h1 gives silu(h1) * h3 with no copy of h1;
the ACC x ACC multiply is `VGATE acc, src, mul` with an ACC source (ISA v0.1.1 §5.3).
"""

# no `from __future__ import annotations`: TileLang evaluates the T.Tensor hints at trace time
import tilelang.language as T

# fixtures/moe_grouped
B, D, DFF, E, TOPK = 256, 512, 256, 32, 6
BLOCK_M, BLOCK_K, BLOCK_N = 64, 128, 128
T_MAX = 2  # token tiles per expert the list layout provides room for (fixture max count 59 -> 1 used)

# scope tables for cut.lower_to_tile_tir (task §3 applied by name, as for FA)
SCOPE_MAP = {
    "x_s": "tcta.inp",
    "W_s": "tcta.wgt", "Wq_s": "tcta.wgt", "Ws_s": "tcta.wgt",
    "H": "tcta.acc", "Hc": "tcta.acc", "out": "tcta.acc",
    "idx_s": "tcta.vec", "dst_s": "tcta.vec", "idx_all": "tcta.vec", "idx_a": "tcta.vec", "idx_b": "tcta.vec", "mask": "tcta.vec", "mask_a": "tcta.vec", "mask_b": "tcta.vec",
    "n": "tcta.vec", "n_a": "tcta.vec", "n_b": "tcta.vec", "w_s": "tcta.vec", "w_c": "tcta.vec",
}
COMBINE_SCOPE_MAP = {"acc": "tcta.acc", "part": "tcta.acc", "idx_s": "tcta.vec"}
ROUTER_SCOPE_MAP = {"row": "tcta.vec", "idx": "tcta.vec"}


def _expert_macros(d, d_ff, experts, t_max, block_M, block_K, block_N, num_stages, weight_format, dtype):
    """The macros shared by the per-expert and the persistent kernels."""
    assert weight_format in ("bf16", "f4.b32")
    assert d % block_K == 0 and (2 * d_ff) % block_N == 0 and d_ff % block_K == 0 and d % block_N == 0
    n_k = d // block_K                 # 4 K-tiles of a token row
    n_h = 2 * d_ff // block_N          # 4 N-tiles of [W1 | W3]
    n_k2 = d_ff // block_K             # 2 K-tiles of h
    n_o = d // block_N                 # 4 N-tiles of the output
    rows = t_max * block_M
    accum = "float32"
    f4 = weight_format == "f4.b32"

    @T.macro
    def dequant(Wq_s, Ws_s, W_s):
        # E2M1 code -> value, low nibble first; UE8M0 scale 2^(e-127). Appendix A of the ISA.
        for i, j in T.Parallel(block_K, block_N):
            byte = T.Cast("int32", Wq_s[i, j // 2])
            nib = (byte >> ((j % 2) * 4)) & 15
            m = nib & 7
            mag = T.if_then_else(
                m < 2, T.Cast(accum, m) * 0.5,
                T.if_then_else(m < 4, 1.0 + T.Cast(accum, m - 2) * 0.5,
                               T.if_then_else(m < 6, 2.0 + T.Cast(accum, m - 4), 4.0 + T.Cast(accum, m - 6) * 2.0)))
            scale = T.exp2(T.Cast(accum, Ws_s[i // 32, j // 32]) - 127.0)
            W_s[i, j] = T.Cast(dtype, T.if_then_else((nib >> 3) == 1, -mag, mag) * scale)

    @T.macro
    def load(Wt, Wq_s, Ws_s, W_s, e, k0, n0):
        """One [block_K, block_N] weight tile into W_s: a bf16 copy, or payload + scales + dequant."""
        if not f4:
            T.copy(Wt[e, k0:k0 + block_K, n0:n0 + block_N], W_s)
        else:
            Wq, Ws = Wt
            T.copy(Wq[e, k0:k0 + block_K, n0 // 2:(n0 + block_N) // 2], Wq_s)
            T.copy(Ws[e, k0 // 32:(k0 + block_K) // 32, n0 // 32:(n0 + block_N) // 32], Ws_s)
            dequant(Wq_s, Ws_s, W_s)

    def weight_stream(n, head, tail):
        """The pipelined weight loop. bf16: TileLang plans the stages (copy -> stage 0, gemm -> stage 1).
        f4.b32: the dequant must sit in the producer stage with the copies (see module docstring), so the
        stages are given by hand, one entry per statement of the loop body (the planner counts the
        statements inside a guard): the guarded accumulator fill if `head` (with the gemm), payload copy,
        scales copy, dequant, gemm, and the `tail` statements of the guarded tail (with the gemm)."""
        if not f4:
            return T.Pipelined(n, num_stages=num_stages)
        s = num_stages - 1
        stage = ([s] if head else []) + [0, 0, 0, s] + [s] * tail
        return T.Pipelined(n, order=list(range(len(stage))), stage=stage)

    @T.macro
    def count(idx, e, idx_all, mask, n):
        """n = number of tokens routed to expert e (VCOUNT on the expert's token list)."""
        T.copy(idx[e, 0:rows], idx_all)
        for i in T.Parallel(rows):
            mask[i] = T.if_then_else(idx_all[i] >= 0, 1, 0)
        T.reduce_sum(mask, n, dim=0)

    @T.macro
    def token_tile(x, W13t, W2t, idx, dst, w, Yp, e, t, x_s, W_s, Wq_s, Ws_s, H, Hc, out, idx_s, dst_s, w_s, w_c):
        """One 64-token tile of expert e: gather, [W1|W3] gemms, SwiGLU, W2 gemms, scale, scatter."""
        T.copy(idx[e, t * block_M:(t + 1) * block_M], idx_s)
        T.fill(H, 0)
        # [W1 | W3]: per K-tile, the gather (LOAD.IDX, -1 rows read as zero) into the INP slot of the
        # K-tile's parity, then the pipelined stream over the N-tiles. The gather stays outside the
        # pipelined loop: inside it as a guarded stage-0 statement TileLang multi-buffers it as if it ran
        # every iteration and the gemm reads a slot the guarded gather never wrote (recorded in E2).
        for kk in T.serial(n_k):
            for i, j in T.Parallel(block_M, block_K):
                x_s[kk % 2, i, j] = T.if_then_else(idx_s[i] >= 0, x[idx_s[i], kk * block_K + j], T.cast(0, dtype))
            for nn in weight_stream(n_h, head=False, tail=0):
                load(W13t, Wq_s, Ws_s, W_s, e, kk * block_K, nn * block_N)
                T.gemm(x_s[kk % 2, :, :], W_s, H[nn, :, :])
        # SwiGLU: silu(h1) * h3 = sigmoid(h1) * (h1 * h3), in place
        for nn in T.serial(n_k2):
            for i, j in T.Parallel(block_M, block_N):
                H[n_k2 + nn, i, j] = H[n_k2 + nn, i, j] * H[nn, i, j]
            for i, j in T.Parallel(block_M, block_N):
                H[nn, i, j] = T.sigmoid(H[nn, i, j])
            for i, j in T.Parallel(block_M, block_N):
                H[n_k2 + nn, i, j] = H[n_k2 + nn, i, j] * H[nn, i, j]
            T.copy(H[n_k2 + nn, :, :], Hc[nn, :, :])
        # The per-row combine weight and the scatter rows are loaded after the [W1|W3] fills so that
        # their WAR tokens (previous tile's VEC/STORE reads -> LOAD) are long satisfied when the LOAD
        # queue gets here. STORE.IDX reads the loaded row directly (LOAD->STORE edge, ISA v0.1.2 §8.1).
        # The weight column is copied once into a VEC-owned vector (w_c) that the four VSCALE_ACCs read:
        # the only VEC->LOAD release of the tile is then the copy's, pushed early. Without it the release
        # comes after the last VSCALE_ACC (after the tile's last MMA), and the surplus token that the
        # expert prefetch (VCOUNT) leaves on the same FIFO forces a pop at the next LOOP entry that
        # waits for it: head-of-line blocking at every expert boundary (phase 4 §0 finding).
        T.copy(w[e, t * block_M:(t + 1) * block_M], w_s)
        T.copy(w_s, w_c)
        T.copy(dst[e, t * block_M:(t + 1) * block_M], dst_s)
        # W2: one pipelined stream over all (N-tile, K-tile) weight tiles; two output accumulators
        # (N-tile parity) so that a gemm never waits for the STORE.IDX of the previous N-tile
        # (MMA -> VEC -> STORE -> MMA round trip); weighted and scattered to rows token*k + slot of Yp
        for m in weight_stream(n_o * n_k2, head=True, tail=2):
            no = m // n_k2
            kk2 = m % n_k2
            if kk2 == 0:
                T.fill(out[no % 2, :, :], 0)
            load(W2t, Wq_s, Ws_s, W_s, e, kk2 * block_K, no * block_N)
            T.gemm(Hc[kk2, :, :], W_s, out[no % 2, :, :])
            if kk2 == n_k2 - 1:
                for i, j in T.Parallel(block_M, block_N):
                    out[no % 2, i, j] = out[no % 2, i, j] * w_c[i]
                for i, j in T.Parallel(block_M, block_N):
                    if dst_s[i] >= 0:
                        Yp[dst_s[i], no * block_N + j] = out[no % 2, i, j]

    return dict(f4=f4, rows=rows, accum=accum, n_h=n_h, n_k2=n_k2, count=count, token_tile=token_tile)


def build_moe_expert_kernel(
    batch: int = B,
    d: int = D,
    d_ff: int = DFF,
    experts: int = E,
    k: int = TOPK,
    t_max: int = T_MAX,
    block_M: int = BLOCK_M,
    block_K: int = BLOCK_K,
    block_N: int = BLOCK_N,
    num_stages: int = 3,
    threads: int = 128,
    weight_format: str = "f4.b32",
    dtype: str = "float16",
):
    """One expert's tile program (grid = experts).

    Buffers: x[B, d]; weights W13q[E, d, d_ff] + W13s[E, d/32, 2*d_ff/32] (uint8 payload of
    [W1 | W3], 2 nibbles per byte, and UE8M0 scales) and W2q[E, d_ff, d/2] + W2s[E, d_ff/32, d/32],
    or W13[E, d, 2*d_ff] / W2[E, d_ff, d] in bf16; idx[E, t_max*64] token ids (-1 pad);
    dst[E, t_max*64] scatter rows (token*k + slot, -1 pad); w[E, t_max*64] combine weights;
    Yp[B*k, d] f32 scatter buffer (row token*k + slot).
    """
    M = _expert_macros(d, d_ff, experts, t_max, block_M, block_K, block_N, num_stages, weight_format, dtype)
    f4, rows, accum, count, token_tile = M["f4"], M["rows"], M["accum"], M["count"], M["token_tile"]
    n_h, n_k2 = M["n_h"], M["n_k2"]

    @T.macro
    def body(x, W13t, W2t, idx, dst, w, Yp, bx):
        # scratch (allocated here so the buffers carry these names into the IR: the scope table and
        # D4's slot plan key on them)
        x_s = T.alloc_shared([2, block_M, block_K], dtype)
        W_s = T.alloc_shared([block_K, block_N], dtype)
        Wq_s = T.alloc_shared([block_K, block_N // 2], "uint8") if f4 else None
        Ws_s = T.alloc_shared([block_K // 32, block_N // 32], "uint8") if f4 else None
        H = T.alloc_fragment([n_h, block_M, block_N], accum)
        Hc = T.alloc_fragment([n_k2, block_M, block_N], dtype)
        out = T.alloc_fragment([2, block_M, block_N], accum)
        idx_s = T.alloc_fragment([block_M], "int32")
        dst_s = T.alloc_fragment([block_M], "int32")
        w_s = T.alloc_fragment([block_M], accum)
        w_c = T.alloc_fragment([block_M], accum)
        idx_all = T.alloc_fragment([rows], "int32")
        mask = T.alloc_fragment([rows], "int32")
        n = T.alloc_fragment([1], "int32")
        count(idx, bx, idx_all, mask, n)
        for t in T.serial(T.ceildiv(n[0], block_M)):
            token_tile(x, W13t, W2t, idx, dst, w, Yp, bx, t, x_s, W_s, Wq_s, Ws_s, H, Hc, out, idx_s, dst_s, w_s, w_c)

    if not f4:
        @T.prim_func
        def main(
            x: T.Tensor([batch, d], dtype),
            W13: T.Tensor([experts, d, 2 * d_ff], dtype),
            W2: T.Tensor([experts, d_ff, d], dtype),
            idx: T.Tensor([experts, rows], "int32"),
            dst: T.Tensor([experts, rows], "int32"),
            w: T.Tensor([experts, rows], accum),
            Yp: T.Tensor([batch * k, d], accum),
        ):
            with T.Kernel(experts, threads=threads) as (bx,):
                body(x, W13, W2, idx, dst, w, Yp, bx)
        return main

    @T.prim_func
    def main_f4(
        x: T.Tensor([batch, d], dtype),
        W13q: T.Tensor([experts, d, d_ff], "uint8"),
        W13s: T.Tensor([experts, d // 32, 2 * d_ff // 32], "uint8"),
        W2q: T.Tensor([experts, d_ff, d // 2], "uint8"),
        W2s: T.Tensor([experts, d_ff // 32, d // 32], "uint8"),
        idx: T.Tensor([experts, rows], "int32"),
        dst: T.Tensor([experts, rows], "int32"),
        w: T.Tensor([experts, rows], accum),
        Yp: T.Tensor([batch * k, d], accum),
    ):
        with T.Kernel(experts, threads=threads) as (bx,):
            body(x, (W13q, W13s), (W2q, W2s), idx, dst, w, Yp, bx)
    return main_f4


def build_moe_persistent_kernel(
    batch: int = B,
    d: int = D,
    d_ff: int = DFF,
    experts: int = E,
    k: int = TOPK,
    t_max: int = T_MAX,
    block_M: int = BLOCK_M,
    block_K: int = BLOCK_K,
    block_N: int = BLOCK_N,
    num_stages: int = 3,
    threads: int = 128,
    weight_format: str = "f4.b32",
    dtype: str = "float16",
    prefetch: bool = True,
):
    """All experts in one program (grid = 1): `for e in T.serial(experts)` around the token-tile loop.

    With `prefetch` (default) experts are handled in pairs so that the token count of expert e+1
    is produced into the other count buffer (register) while expert e runs (the idx table has a
    -1 pad row at E); the program is `LOOP n=E/2 { count; LOOP r=0; count; LOOP r=1 }`. Without
    it the count of expert e is taken right before its tile loop, `LOOP n=E { count; LOOP r }`
    (half the body, one count register, the LOOP waits for the VCOUNT). Same buffers as
    `build_moe_expert_kernel`.
    """
    assert experts % 2 == 0 or not prefetch
    M = _expert_macros(d, d_ff, experts, t_max, block_M, block_K, block_N, num_stages, weight_format, dtype)
    f4, rows, accum, count, token_tile = M["f4"], M["rows"], M["accum"], M["count"], M["token_tile"]
    n_h, n_k2 = M["n_h"], M["n_k2"]

    @T.macro
    def expert(x, W13t, W2t, idx, dst, w, Yp, e, n_cur, n_next, idx_all, mask,
               x_s, W_s, Wq_s, Ws_s, H, Hc, out, idx_s, dst_s, w_s, w_c):
        count(idx, e + 1, idx_all, mask, n_next)   # prefetch the next expert's count (idx has a -1 pad row at E)
        for t in T.serial(T.ceildiv(n_cur[0], block_M)):
            token_tile(x, W13t, W2t, idx, dst, w, Yp, e, t, x_s, W_s, Wq_s, Ws_s, H, Hc, out, idx_s, dst_s, w_s, w_c)

    @T.macro
    def body(x, W13t, W2t, idx, dst, w, Yp):
        x_s = T.alloc_shared([2, block_M, block_K], dtype)
        W_s = T.alloc_shared([block_K, block_N], dtype)
        Wq_s = T.alloc_shared([block_K, block_N // 2], "uint8") if f4 else None
        Ws_s = T.alloc_shared([block_K // 32, block_N // 32], "uint8") if f4 else None
        H = T.alloc_fragment([n_h, block_M, block_N], accum)
        Hc = T.alloc_fragment([n_k2, block_M, block_N], dtype)
        out = T.alloc_fragment([2, block_M, block_N], accum)
        idx_s = T.alloc_fragment([block_M], "int32")
        dst_s = T.alloc_fragment([block_M], "int32")
        w_s = T.alloc_fragment([block_M], accum)
        w_c = T.alloc_fragment([block_M], accum)
        # One token-list buffer per count register. With a shared buffer the second count's load
        # needs the first VCOUNT's release, pushed before `LOOP r=0`; a zero-trip loop must stay
        # transparent, so D4 pops that release in front of the LOOP and the LOAD queue then waits for
        # the VCOUNT (a LOAD -> VEC -> LOAD round trip, 2-3 idle LOAD steps at every expert boundary).
        # With two buffers each release goes to the load two sections later and nothing waits.
        idx_a = T.alloc_fragment([rows], "int32")
        idx_b = T.alloc_fragment([rows], "int32")
        mask_a = T.alloc_fragment([rows], "int32")
        mask_b = T.alloc_fragment([rows], "int32")
        n_a = T.alloc_fragment([1], "int32")
        n_b = T.alloc_fragment([1], "int32")
        count(idx, 0, idx_b, mask_b, n_a)
        for ee in T.serial(experts // 2):
            expert(x, W13t, W2t, idx, dst, w, Yp, 2 * ee, n_a, n_b, idx_a, mask_a,
                   x_s, W_s, Wq_s, Ws_s, H, Hc, out, idx_s, dst_s, w_s, w_c)
            expert(x, W13t, W2t, idx, dst, w, Yp, 2 * ee + 1, n_b, n_a, idx_b, mask_b,
                   x_s, W_s, Wq_s, Ws_s, H, Hc, out, idx_s, dst_s, w_s, w_c)

    @T.macro
    def body_single(x, W13t, W2t, idx, dst, w, Yp):
        x_s = T.alloc_shared([2, block_M, block_K], dtype)
        W_s = T.alloc_shared([block_K, block_N], dtype)
        Wq_s = T.alloc_shared([block_K, block_N // 2], "uint8") if f4 else None
        Ws_s = T.alloc_shared([block_K // 32, block_N // 32], "uint8") if f4 else None
        H = T.alloc_fragment([n_h, block_M, block_N], accum)
        Hc = T.alloc_fragment([n_k2, block_M, block_N], dtype)
        out = T.alloc_fragment([2, block_M, block_N], accum)
        idx_s = T.alloc_fragment([block_M], "int32")
        dst_s = T.alloc_fragment([block_M], "int32")
        w_s = T.alloc_fragment([block_M], accum)
        w_c = T.alloc_fragment([block_M], accum)
        idx_all = T.alloc_fragment([rows], "int32")
        mask = T.alloc_fragment([rows], "int32")
        n = T.alloc_fragment([1], "int32")
        for e in T.serial(experts):
            count(idx, e, idx_all, mask, n)
            for t in T.serial(T.ceildiv(n[0], block_M)):
                token_tile(x, W13t, W2t, idx, dst, w, Yp, e, t, x_s, W_s, Wq_s, Ws_s, H, Hc, out, idx_s, dst_s, w_s, w_c)

    if not prefetch:
        body = body_single

    if not f4:
        @T.prim_func
        def main(
            x: T.Tensor([batch, d], dtype),
            W13: T.Tensor([experts, d, 2 * d_ff], dtype),
            W2: T.Tensor([experts, d_ff, d], dtype),
            idx: T.Tensor([experts + 1, rows], "int32"),
            dst: T.Tensor([experts, rows], "int32"),
            w: T.Tensor([experts, rows], accum),
            Yp: T.Tensor([batch * k, d], accum),
        ):
            with T.Kernel(1, threads=threads) as (bx,):
                body(x, W13, W2, idx, dst, w, Yp)
        return main

    @T.prim_func
    def main_f4(
        x: T.Tensor([batch, d], dtype),
        W13q: T.Tensor([experts, d, d_ff], "uint8"),
        W13s: T.Tensor([experts, d // 32, 2 * d_ff // 32], "uint8"),
        W2q: T.Tensor([experts, d_ff, d // 2], "uint8"),
        W2s: T.Tensor([experts, d_ff // 32, d // 32], "uint8"),
        idx: T.Tensor([experts + 1, rows], "int32"),
        dst: T.Tensor([experts, rows], "int32"),
        w: T.Tensor([experts, rows], accum),
        Yp: T.Tensor([batch * k, d], accum),
    ):
        with T.Kernel(1, threads=threads) as (bx,):
            body(x, (W13q, W13s), (W2q, W2s), idx, dst, w, Yp)
    return main_f4


def build_moe_combine_kernel(batch: int = B, d: int = D, k: int = TOPK, block_M: int = BLOCK_M,
                             block_N: int = BLOCK_N, threads: int = 128):
    """The combine reduce (F1), one program per 64-token tile (grid = B/64).

    y[tile rows] = y_shared[tile rows] + sum_s Yp[rows_idx[s, tile rows], :] where rows_idx[s, tok]
    = tok*k + s (the host's view of the scatter layout; -1 would skip). The expert parts are
    already weighted by topk_w (VSCALE_ACC in the expert kernel, as the fixture's y_parts), so
    the combine is a sum: each part is gathered into an ACC tile (LOAD.IDX) and added
    (VGATE add with an ACC source); the shared expert comes through the same tile by LOAD.2D,
    and the accumulator is VEC-owned (T.fill) so that its reuse needs only STORE->VEC tokens.
    """
    n_o = d // block_N

    @T.prim_func
    def combine(
        Yp: T.Tensor([batch * k, d], "float32"),
        y_shared: T.Tensor([batch, d], "float32"),
        rows_idx: T.Tensor([k, batch], "int32"),
        y: T.Tensor([batch, d], "float32"),
    ):
        with T.Kernel(batch // block_M, threads=threads) as (bx,):
            acc = T.alloc_fragment([block_M, block_N], "float32")
            part = T.alloc_fragment([block_M, block_N], "float32")
            idx_s = T.alloc_fragment([block_M], "int32")
            for no in T.serial(n_o):
                T.fill(acc, 0)
                T.copy(y_shared[bx * block_M:(bx + 1) * block_M, no * block_N:(no + 1) * block_N], part)
                for i, j in T.Parallel(block_M, block_N):
                    acc[i, j] = acc[i, j] + part[i, j]
                for s in T.serial(k):
                    T.copy(rows_idx[s, bx * block_M:(bx + 1) * block_M], idx_s)
                    for i, j in T.Parallel(block_M, block_N):
                        part[i, j] = T.if_then_else(idx_s[i] >= 0, Yp[idx_s[i], no * block_N + j], T.cast(0, "float32"))
                    for i, j in T.Parallel(block_M, block_N):
                        acc[i, j] = acc[i, j] + part[i, j]
                T.copy(acc, y[bx * block_M:(bx + 1) * block_M, no * block_N:(no + 1) * block_N])

    return combine


def build_router_kernel(batch: int = B, experts: int = E, k: int = TOPK, threads: int = 128):
    """Top-k expert indices for one token per program: VTOPK as an opaque extern on a [1, E] row;
    the int32 index row is stored with the `i32` qualifier (ISA v0.1.1 §4)."""
    k_pad = 8

    @T.prim_func
    def router(logits: T.Tensor([batch, experts], "float32"), topk: T.Tensor([batch, k_pad], "int32")):
        with T.Kernel(batch, threads=threads) as (bx,):
            row = T.alloc_fragment([1, experts], "float32")
            idx = T.alloc_fragment([1, k_pad], "int32")
            T.copy(logits[bx:bx + 1, 0:experts], row)
            T.call_extern("handle", "tcta.vtopk", T.address_of(idx[0, 0]), T.address_of(row[0, 0]), k)
            T.copy(idx[0:1, 0:k], topk[bx:bx + 1, 0:k])

    return router
