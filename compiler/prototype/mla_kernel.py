"""MLA decode, weight-absorbed form, in TileLang (phase 4, G1).

Fixture `fixtures/mla_decode_absorbed`: 16 heads x 4 decode rows share one latent cache
(`K = [c_kv | k_rope]`, [L, 576]); V is the first 512 columns of the same rows. The 64 query
rows (head-major, row = h*4 + b) are one 64-row tile against a 64-key KV tile, exactly the
FlashAttention inner loop of `fa_kernel.py` with two differences:

- **576-deep QK^T.** The reduction runs as one 64-deep chunk (the rope columns, a partial
  INP/WGT tile) and four 128-deep chunks. The four 128-column K chunks are the V tiles of the
  same KV tile, so they stay in the 4-slot WGT ring for the PV gemms and V is never re-read;
  the rope chunk uses slot 3 before its V occupant arrives. Q does not fit the 2-slot INP
  scratchpad (576 columns = 4.5 tiles), so the Q chunks are re-streamed from DRAM at every KV
  tile through a 2-slot ring (10 LOAD.2D per KV tile: 5 Q, 5 K; the device-parameter finding
  is in the expressivity record).
- **Wide accumulator.** `acc_o[64, 512]` is one fragment; D4 maps it to four consecutive ACC
  slots (`ACC.W`, ISA §6.5): the PV gemms write the four 128-column slices, the online-softmax
  rescale and the final 1/l scale are one `VSCALE_ACC wide`, the output one `STORE.2D wide`.

Numerics follow the fixture's reference (`ref/kernels.py::fa_reference`): exp2 with the
log2(e)/sqrt(576) scale folded in, P rounded to bf16 before the PV gemm, O stored as f32.
"""


import os
import sys

import numpy as np
import tilelang.language as T

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "ref"))
import common  # noqa: E402

LOG2E = 1.44269504

N_H, BQ, SEQ_LEN, D_K, D_V = 16, 4, 1024, 576, 512
ROWS = N_H * BQ          # 64 stacked query rows
BLOCK_N = 64             # keys per KV tile
T_N = 128                # tile width: the four full K chunks / V tiles

SCOPE_MAP = {
    "Q_shared": "tcta.inp", "K_shared": "tcta.wgt",
    "acc_s": "tcta.acc", "acc_s_cast": "tcta.acc", "acc_o": "tcta.acc",
    "scores_max": "tcta.vec", "scores_max_prev": "tcta.vec", "scores_scale": "tcta.vec",
    "scores_sum": "tcta.vec", "logsum": "tcta.vec",
}

MB = 1 << 20
DRAM = {"Q": 0, "K": 1 * MB, "O": 4 * MB}
DRAM_BYTES = 5 * MB


def build_mla_kernel(
    n_h: int = N_H,
    bq: int = BQ,
    seq_len: int = SEQ_LEN,
    d_k: int = D_K,
    d_v: int = D_V,
    block_N: int = BLOCK_N,
    threads: int = 128,
    dtype: str = "float16",
    inp_slots: int = 2,
    kv_unroll: int = 1,
):
    """`inp_slots` (phase 5 sweep): INP slots available for Q. 2 (the v1 descriptor): every Q chunk is
    re-streamed through a 2-slot ring at each KV tile. 3 or 4: the rope chunk (and chunk 0) stay resident,
    the rest stream through a 2-slot ring. >= 5: all 4.5 Q tiles resident, loaded once before the KV loop.

    `kv_unroll` (phase 6, ring-depth sweep): the K ring holds the five chunks of `kv_unroll` KV tiles
    (5 * kv_unroll WGT slots), so the LOAD queue can stream the next tiles' K while this tile is consumed;
    the KV loop is unrolled by that factor so every slot index is static."""
    rows = n_h * bq
    assert rows == 64 and d_v % T_N == 0 and 0 < d_k - d_v <= T_N and seq_len % block_N == 0
    n_kv = seq_len // block_N
    n_full = d_v // T_N                     # 128-deep chunks (and V tiles)
    rope = d_k - d_v                        # the trailing 64-deep chunk
    scale = (1.0 / d_k) ** 0.5 * LOG2E
    accum = "float32"
    if inp_slots != 2 or kv_unroll != 1:
        return _build_mla_resident(rows, n_kv, n_full, rope, scale, d_k, d_v, block_N, seq_len, threads, dtype, accum, inp_slots, kv_unroll)

    @T.prim_func
    def main(
        Q: T.Tensor([rows, d_k], dtype),
        K: T.Tensor([seq_len, d_k], dtype),
        O: T.Tensor([rows, d_v], accum),
    ):
        with T.Kernel(1, threads=threads) as (bx,):
            Q_shared = T.alloc_shared([2, rows, T_N], dtype)
            K_shared = T.alloc_shared([n_full, block_N, T_N], dtype)
            acc_s = T.alloc_fragment([rows, block_N], accum)
            acc_s_cast = T.alloc_fragment([rows, block_N], dtype)
            acc_o = T.alloc_fragment([rows, d_v], accum)
            scores_max = T.alloc_fragment([rows], accum)
            scores_max_prev = T.alloc_fragment([rows], accum)
            scores_scale = T.alloc_fragment([rows], accum)
            scores_sum = T.alloc_fragment([rows], accum)
            logsum = T.alloc_fragment([rows], accum)

            T.fill(acc_o, 0)
            T.fill(logsum, 0)
            T.fill(scores_max, -T.infinity(accum))

            for kv in T.serial(n_kv):
                T.clear(acc_s)
                # S = Q K^T over 576 channels: the rope chunk (64 deep, partial tiles; WGT slot 3 is
                # free until the last full chunk arrives), then the four 128-deep chunks, whose K
                # tiles are the V tiles of this KV tile.
                T.copy(Q[:, d_v:d_k], Q_shared[0, :, 0:rope])
                T.copy(K[kv * block_N:(kv + 1) * block_N, d_v:d_k], K_shared[n_full - 1, :, 0:rope])
                T.gemm(Q_shared[0, :, 0:rope], K_shared[n_full - 1, :, 0:rope], acc_s, transpose_B=True)
                for c in T.serial(n_full):
                    T.copy(Q[:, c * T_N:(c + 1) * T_N], Q_shared[(c + 1) % 2, :, :])
                    T.copy(K[kv * block_N:(kv + 1) * block_N, c * T_N:(c + 1) * T_N], K_shared[c, :, :])
                    T.gemm(Q_shared[(c + 1) % 2, :, :], K_shared[c, :, :], acc_s, transpose_B=True)

                # online softmax (the FA kernel's form)
                T.copy(scores_max, scores_max_prev)
                T.fill(scores_max, -T.infinity(accum))
                T.reduce_max(acc_s, scores_max, dim=1, clear=False)
                for i in T.Parallel(rows):
                    scores_max[i] = T.max(scores_max[i], scores_max_prev[i])
                for i in T.Parallel(rows):
                    scores_scale[i] = T.exp2(scores_max_prev[i] * scale - scores_max[i] * scale)
                for i, j in T.Parallel(rows, block_N):
                    acc_s[i, j] = T.exp2(acc_s[i, j] * scale - scores_max[i] * scale)
                T.reduce_sum(acc_s, scores_sum, dim=1)
                for i in T.Parallel(rows):
                    logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
                T.copy(acc_s, acc_s_cast)
                for i, j in T.Parallel(rows, d_v):
                    acc_o[i, j] *= scores_scale[i]           # VSCALE_ACC wide

                # O += P V, V = the four resident K chunks (columns 0..511 of the KV rows)
                for c in T.serial(n_full):
                    T.gemm(acc_s_cast, K_shared[c, :, :], acc_o[:, c * T_N:(c + 1) * T_N])

            for i, j in T.Parallel(rows, d_v):
                acc_o[i, j] /= logsum[i]
            T.copy(acc_o, O)                                  # STORE.2D wide, f32

    return main


def _build_mla_resident(rows, n_kv, n_full, rope, scale, d_k, d_v, block_N, seq_len, threads, dtype, accum, inp_slots, kv_unroll=1):
    """The same kernel with `n_res` Q chunks resident in INP (the rope chunk first, then chunk 0, ...) and
    the remaining chunks streamed through a 2-slot ring (one slot if only one is left over). The chunk loop
    is written in Python so every Q slot index is static. The K ring has one slot per chunk (the rope chunk
    included) of `kv_unroll` consecutive KV tiles."""
    n_chunks = n_full + 1                                   # rope + 4 full
    u = kv_unroll
    assert n_kv % u == 0
    # u == 1: the phase-4/5 form, 4 K slots with the rope chunk in slot 3 before its V occupant arrives;
    # u > 1: one slot per chunk, rope included, for each of the u tiles in the ring
    per_tile = n_full if u == 1 else n_chunks
    rope_slot = n_full - 1 if u == 1 else n_full
    n_res = min(n_chunks, max(0, inp_slots - 2)) if inp_slots < n_chunks else n_chunks
    ring = min(2, inp_slots - n_res)
    assert ring >= 1 or n_res == n_chunks
    # chunk order in the loop: rope (index n_full) first, then 0..n_full-1
    order = [n_full] + list(range(n_full))
    slot_of = {}
    for pos, ch in enumerate(order):
        if pos < n_res:
            slot_of[ch] = pos
        else:
            slot_of[ch] = n_res + (pos - n_res) % max(ring, 1)
    streamed = [ch for pos, ch in enumerate(order) if pos >= n_res]

    @T.prim_func
    def main(
        Q: T.Tensor([rows, d_k], dtype),
        K: T.Tensor([seq_len, d_k], dtype),
        O: T.Tensor([rows, d_v], accum),
    ):
        with T.Kernel(1, threads=threads) as (bx,):
            Q_shared = T.alloc_shared([inp_slots, rows, T_N], dtype)
            K_shared = T.alloc_shared([per_tile * u, block_N, T_N], dtype)
            acc_s = T.alloc_fragment([rows, block_N], accum)
            acc_s_cast = T.alloc_fragment([rows, block_N], dtype)
            acc_o = T.alloc_fragment([rows, d_v], accum)
            scores_max = T.alloc_fragment([rows], accum)
            scores_max_prev = T.alloc_fragment([rows], accum)
            scores_scale = T.alloc_fragment([rows], accum)
            scores_sum = T.alloc_fragment([rows], accum)
            logsum = T.alloc_fragment([rows], accum)

            T.fill(acc_o, 0)
            T.fill(logsum, 0)
            T.fill(scores_max, -T.infinity(accum))
            for ch in order[:n_res]:                           # resident Q chunks, loaded once
                if ch == n_full:
                    T.copy(Q[:, d_v:d_k], Q_shared[slot_of[ch], :, 0:rope])
                else:
                    T.copy(Q[:, ch * T_N:(ch + 1) * T_N], Q_shared[slot_of[ch], :, :])

            for kv2 in T.serial(n_kv // u):
                for j in range(u):                              # Python-unrolled: static K slots j*5 .. j*5+4
                    kv = kv2 * u + j
                    base = j * per_tile
                    T.clear(acc_s)
                    for ch in order:
                        if ch == n_full:
                            if ch in streamed:
                                T.copy(Q[:, d_v:d_k], Q_shared[slot_of[ch], :, 0:rope])
                            T.copy(K[kv * block_N:(kv + 1) * block_N, d_v:d_k], K_shared[base + rope_slot, :, 0:rope])
                            T.gemm(Q_shared[slot_of[ch], :, 0:rope], K_shared[base + rope_slot, :, 0:rope], acc_s, transpose_B=True)
                        else:
                            if ch in streamed:
                                T.copy(Q[:, ch * T_N:(ch + 1) * T_N], Q_shared[slot_of[ch], :, :])
                            T.copy(K[kv * block_N:(kv + 1) * block_N, ch * T_N:(ch + 1) * T_N], K_shared[base + ch, :, :])
                            T.gemm(Q_shared[slot_of[ch], :, :], K_shared[base + ch, :, :], acc_s, transpose_B=True)

                    T.copy(scores_max, scores_max_prev)
                    T.fill(scores_max, -T.infinity(accum))
                    T.reduce_max(acc_s, scores_max, dim=1, clear=False)
                    for i in T.Parallel(rows):
                        scores_max[i] = T.max(scores_max[i], scores_max_prev[i])
                    for i in T.Parallel(rows):
                        scores_scale[i] = T.exp2(scores_max_prev[i] * scale - scores_max[i] * scale)
                    for i, jj in T.Parallel(rows, block_N):
                        acc_s[i, jj] = T.exp2(acc_s[i, jj] * scale - scores_max[i] * scale)
                    T.reduce_sum(acc_s, scores_sum, dim=1)
                    for i in T.Parallel(rows):
                        logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
                    T.copy(acc_s, acc_s_cast)
                    for i, jj in T.Parallel(rows, d_v):
                        acc_o[i, jj] *= scores_scale[i]
                    for c in range(n_full):
                        T.gemm(acc_s_cast, K_shared[base + c, :, :], acc_o[:, c * T_N:(c + 1) * T_N])

            for i, j in T.Parallel(rows, d_v):
                acc_o[i, j] /= logsum[i]
            T.copy(acc_o, O)

    return main


# ---------------------------------------------------------------- host side

def stacked_q(fx: dict) -> np.ndarray:
    """q_abs [n_h, bq, d_k] -> [n_h*bq, d_k], row = h*bq + b (the order of O)."""
    return fx["q_abs"].reshape(-1, fx["q_abs"].shape[-1])


def load_dram(m, fx: dict) -> None:
    m.dram_write(DRAM["Q"], common.bf16_bytes(stacked_q(fx)).ravel())
    m.dram_write(DRAM["K"], common.bf16_bytes(fx["K"]).ravel())


def read_o(m, fx: dict) -> np.ndarray:
    n_h, bq, d_v = fx["O"].shape
    n = n_h * bq * d_v
    return m.dram[DRAM["O"]: DRAM["O"] + 4 * n].view(np.float32).reshape(n_h, bq, d_v).copy()
