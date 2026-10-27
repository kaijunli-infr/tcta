"""FlashAttention-2 forward kernel in TileLang, reduced for the lowering cut-off task.

Derived from tilelang-cutoff/cutoff_attn/kernel.py::build_cutoff_attention
(backend="cuda") with the band mask, cutoff/causal, skip_tiles, TilesVisited,
the CPU backend branches and the one_trip workaround removed. The KV loop is a
plain T.Pipelined over every KV tile.

Other changes from the source kernel, needed because the mask loop is gone:
- acc_s is cleared with T.clear before the QK^T gemm (the mask loop used to
  initialise it to 0 / -inf).
- acc_o, logsum, scores_max are initialised with T.fill rather than T.Parallel
  loops, so every init is a tile op.
- O_shared is removed: acc_o is copied straight to Output. The task's scope
  table (INP/WGT/ACC/VEC/global) has no slot for an output staging buffer.

Layout BHSD. Self-attention, no causal mask.
"""

from typing import Optional

import tilelang.language as T

LOG2E = 1.44269504

# Task-wide shapes (match fixtures/flash_attn_gqa, one head at a time).
BATCH, HEADS, SEQ_LEN, DIM = 1, 8, 256, 64
BLOCK_M, BLOCK_N = 64, 64


def build_fa_kernel(
    batch: int = BATCH,
    heads: int = HEADS,
    seq_len: int = SEQ_LEN,
    dim: int = DIM,
    block_M: int = BLOCK_M,
    block_N: int = BLOCK_N,
    num_stages: int = 2,
    threads: int = 128,
    sm_scale: Optional[float] = None,
    dtype: str = "float16",
    scopes: Optional[dict] = None,
):
    """Return the TileLang PrimFunc.

    ``scopes`` optionally overrides the storage scope per buffer name
    (e.g. {"K_shared": "tcta.wgt"}). Default: TileLang's own scopes
    (shared.dyn for alloc_shared, local.fragment for alloc_fragment).
    """
    assert seq_len % block_M == 0 and seq_len % block_N == 0
    if sm_scale is None:
        sm_scale = (1.0 / dim) ** 0.5
    scale = sm_scale * LOG2E
    shape = [batch, heads, seq_len, dim]
    accum_dtype = "float32"
    n_q_tiles = seq_len // block_M
    n_kv_tiles = seq_len // block_N
    scopes = scopes or {}

    def smem(name, shp, dt):
        return T.alloc_shared(shp, dt, scope=scopes.get(name, "shared.dyn"))

    def frag(name, shp, dt):
        return T.alloc_fragment(shp, dt, scope=scopes.get(name, "local.fragment"))

    @T.prim_func
    def main(
        Q: T.Tensor(shape, dtype),
        K: T.Tensor(shape, dtype),
        V: T.Tensor(shape, dtype),
        Output: T.Tensor(shape, dtype),
    ):
        with T.Kernel(n_q_tiles, heads, batch, threads=threads) as (bx, by, bz):
            Q_shared = smem("Q_shared", [block_M, dim], dtype)
            K_shared = smem("K_shared", [block_N, dim], dtype)
            V_shared = smem("V_shared", [block_N, dim], dtype)
            acc_s = frag("acc_s", [block_M, block_N], accum_dtype)
            acc_s_cast = frag("acc_s_cast", [block_M, block_N], dtype)
            acc_o = frag("acc_o", [block_M, dim], accum_dtype)
            scores_max = frag("scores_max", [block_M], accum_dtype)
            scores_max_prev = frag("scores_max_prev", [block_M], accum_dtype)
            scores_scale = frag("scores_scale", [block_M], accum_dtype)
            scores_sum = frag("scores_sum", [block_M], accum_dtype)
            logsum = frag("logsum", [block_M], accum_dtype)

            T.copy(Q[bz, by, bx * block_M : (bx + 1) * block_M, :], Q_shared)
            T.fill(acc_o, 0)
            T.fill(logsum, 0)
            T.fill(scores_max, -T.infinity(accum_dtype))

            for k in T.Pipelined(0, n_kv_tiles, num_stages=num_stages):
                T.copy(K[bz, by, k * block_N : (k + 1) * block_N, :], K_shared)
                T.clear(acc_s)
                T.gemm(Q_shared, K_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)

                # online softmax
                T.copy(scores_max, scores_max_prev)
                T.fill(scores_max, -T.infinity(accum_dtype))
                T.reduce_max(acc_s, scores_max, dim=1, clear=False)
                for i in T.Parallel(block_M):
                    scores_max[i] = T.max(scores_max[i], scores_max_prev[i])
                for i in T.Parallel(block_M):
                    scores_scale[i] = T.exp2(scores_max_prev[i] * scale - scores_max[i] * scale)
                for i, j in T.Parallel(block_M, block_N):
                    acc_s[i, j] = T.exp2(acc_s[i, j] * scale - scores_max[i] * scale)
                T.reduce_sum(acc_s, scores_sum, dim=1)
                for i in T.Parallel(block_M):
                    logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
                T.copy(acc_s, acc_s_cast)
                for i, j in T.Parallel(block_M, dim):
                    acc_o[i, j] *= scores_scale[i]

                T.copy(V[bz, by, k * block_N : (k + 1) * block_N, :], V_shared)
                T.gemm(acc_s_cast, V_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)

            for i, j in T.Parallel(block_M, dim):
                acc_o[i, j] /= logsum[i]
            T.copy(acc_o, Output[bz, by, bx * block_M : (bx + 1) * block_M, :])

    return main
