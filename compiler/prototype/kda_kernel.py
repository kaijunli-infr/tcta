"""KDA (Kimi Delta Attention) chunkwise forward in TileLang (phase 4, G2; re-emitted on ISA v0.1.3 in phase 5 §0).

Fixture `fixtures/kda_chunk64`: T = 256 tokens, d = 128, chunk C = 64, per-channel decay
alpha[t, :] in (0.9, 1], beta[t] in (0.3, 1]. The frozen reference is the per-token recurrence
`ref/kernels.py::kda_recurrent`:

    S_t = diag(a_t) S_{t-1} - b_t k_t (k_t^T diag(a_t) S_{t-1}) + b_t k_t v_t^T,   o_t = S_t^T q_t

Chunk form (derived from it, exact in float64; `docs/tilelang-mla-kda-expressivity.md` §2). With
G_i = sum_{s<=i} log2 a_s inside the chunk (cumulative, inclusive), K^ = K * 2^G, K~ = K * 2^-G,
Q^ = Q * 2^G (elementwise) and B = diag(beta):

    N = (K^ K~^T) * M_strict * B          (strictly lower, 64x64)
    U = (I + N)^-1 B (V - K^ S)           (I + N)^-1 = sum_j (-N)^j, Appendix B repeated squaring
    O = Q^ S + ((Q^ K~^T) * M_incl) U
    S <- diag(2^G_last) S + (K^T * 2^(G_last - G_j)) U

How the ISA shapes the authoring (v0.1.3; the v0.1.2 form and its findings are in the expressivity
record and in `docs/gate1-report.md`):

- Every computed matrix used as a B operand (K~, the T_s of the solve, R, U) is moved from ACC into
  a WGT tile with `VCVT dst=WGT` (bf16), the device-computed-operand path of v0.1.3. The DRAM scratch
  regions and their STORE->LOAD round trips (13 per chunk in v0.1.2) are gone. A operands that must
  keep their f32 value (P, T_s, the decayed K^T) are converted into INP tiles (`VCVT dst=INP`).
- The gate is taken in the log2 domain (`g = log2 alpha`, bf16); the in-chunk cumulative sum is a
  GEMM with the lower-triangular ones matrix, and K^T is a second DRAM layout of k (`kt`) because
  the state update contracts over tokens and the ISA has no transposed load.
- The [128, 128] state is updated in two [64, 128] ACC halves (`S_acc`: an MMA writes at most T_M
  rows, `VSCALE_ACC` scales 64 rows) and each half is moved into the 128-row STATE tile with
  `VCVT dst=STATE, row0 = 0 | 64` (f32), where the next chunk's `K^ S` and `Q^ S` read it as the B
  operand. No DRAM mirror and no 64 KB reload; the per-chunk `STORE.2D` of the halves is the
  fixture's checkpoint output (`S_all[c + 1]`), not a round trip.
- Masks: the two masked products (N and the W of O) are Gram products `X Y^T` (`MMA.T`), and
  `MMA.MASKED` exists only in the untransposed form, so the 0/1 masks stay `VGATE mul` after the
  GEMM (one VEC instruction each). D4 fuses `gemm; acc *= mask` into `MMA.MASKED mul` where the
  gemm is untransposed.
- bf16 operands at every array input put the plain schedule at 8.2e-3 rowmax error on `o`
  (tolerance 7.8e-3). U, R (B side) and the decayed K^T (A side) are fed as hi + lo bf16 pairs
  (`x_hi = bf16(x)` into its own ACC tile, `x -= x_hi` in place, both moved into operand tiles):
  o 6.0e-3, S 1.5e-3.

Device configuration: the kernel needs 5 INP, 8 WGT, about 12 ACC (after slot sharing) and 5
two-dimensional VEC tiles, more than the v1 KV260 descriptor (2 / 4 / 6 / 8); it is mapped and
run with the ISA §11 config-C descriptor (`CONFIG_C`).
"""

import os
import sys

import numpy as np
import tilelang.language as T

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "sim"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "ref"))
import common  # noqa: E402
from tcta.isa import SP, DEVICE_CONFIG_C, DeviceConfig  # noqa: E402

SEQ, D, CHUNK = 256, 128, 64
N_CHUNKS = SEQ // CHUNK
N_SOLVE = 5                 # (I+N)^-1 = prod_{s<=5} (I + N^(2^s)) covers 64 terms for a 64x64 strictly lower N

# ISA §11, 22FDX config C (model): 8 / 16 / 8 + 2 wide / 64 / 2 slots, FIFO depth 16
CONFIG_C = DEVICE_CONFIG_C          # ISA §11 config C (v0.1.4: INP 5, WGT 32, ACC 8 + 2 wide, VEC 64, STATE 2, FIFO 8)

SCOPE_MAP = {
    "tri_s": "tcta.inp", "pb_s": "tcta.inp", "ab_s": "tcta.inp", "Kth_c": "tcta.inp", "Ktl_c": "tcta.inp",
    "g_w": "tcta.wgt", "triu_w": "tcta.wgt", "kbar_w": "tcta.wgt", "t_w": "tcta.wgt", "r_w": "tcta.wgt", "u_w": "tcta.wgt",
    "S_st": "tcta.state",
    "S_acc": "tcta.acc", "G": "tcta.acc", "Kh": "tcta.acc", "Kh_c": "tcta.acc", "Qh": "tcta.acc", "Qh_c": "tcta.acc",
    "Kb": "tcta.acc", "Vc": "tcta.acc", "Tm": "tcta.acc", "P": "tcta.acc", "R": "tcta.acc", "Rh": "tcta.acc",
    "U": "tcta.acc", "Uh": "tcta.acc", "O": "tcta.acc", "W": "tcta.acc", "W_c": "tcta.acc",
    "Gt": "tcta.acc", "Gt_c": "tcta.acc", "Gx": "tcta.acc", "Kt": "tcta.acc", "Kth": "tcta.acc",
    "idx_s": "tcta.vec", "hidx_s": "tcta.vec", "ms_v": "tcta.vec", "mi_v": "tcta.vec", "eye_v": "tcta.vec",
    "beta_v": "tcta.vec", "m_v": "tcta.vec",
}

MB = 1 << 20
DRAM = {"q": 0, "k": 64 * 1024, "v": 128 * 1024, "g": 192 * 1024, "gt": 256 * 1024, "kt": 320 * 1024,
        "beta": 384 * 1024, "rows": 388 * 1024, "hrows": 392 * 1024,
        "tri": 400 * 1024, "triu": 408 * 1024, "mstrict": 416 * 1024, "mincl": 432 * 1024, "eye": 448 * 1024,
        "S_all": 2 * MB, "o": 3 * MB}
DRAM_BYTES = 4 * MB


def build_kda_kernel(seq: int = SEQ, d: int = D, chunk: int = CHUNK, threads: int = 128, dtype: str = "float16"):
    assert d == 128 and chunk == 64 and seq % chunk == 0
    n_chunks = seq // chunk
    accum = "float32"
    half = d // 2                # the two 64-row halves of the state

    @T.prim_func
    def main(
        q: T.Tensor([seq, d], dtype),
        k: T.Tensor([seq, d], dtype),
        v: T.Tensor([seq, d], dtype),
        g: T.Tensor([seq, d], dtype),             # log2 alpha
        gt: T.Tensor([d, seq], dtype),            # g transposed
        kt: T.Tensor([d, seq], dtype),            # k transposed
        beta: T.Tensor([n_chunks, chunk], accum),
        rows: T.Tensor([n_chunks, chunk], "int32"),   # arange: identity gather rows (token index)
        hrows: T.Tensor([2, half], "int32"),          # arange: identity gather rows (channel index)
        tri: T.Tensor([chunk, chunk], dtype),     # lower-triangular ones, diagonal included
        triu: T.Tensor([chunk, chunk], dtype),    # its transpose
        mstrict: T.Tensor([chunk, chunk], accum), # 0/1 strictly lower
        mincl: T.Tensor([chunk, chunk], accum),   # 0/1 lower, diagonal included
        eye: T.Tensor([chunk, chunk], accum),
        S_all: T.Tensor([n_chunks + 1, d, d], accum),   # S_all[c+1] = state after chunk c (checkpoints)
        o: T.Tensor([seq, d], accum),
    ):
        with T.Kernel(1, threads=threads) as (bx,):
            tri_s = T.alloc_shared([chunk, chunk], dtype)
            pb_s = T.alloc_shared([chunk, chunk], dtype)          # A-operand converts of P
            ab_s = T.alloc_shared([chunk, chunk], dtype)          # A-operand converts of T_s
            Kth_c = T.alloc_shared([half, chunk], dtype)          # A-operand converts of the decayed K^T (hi, lo)
            Ktl_c = T.alloc_shared([half, chunk], dtype)
            g_w = T.alloc_shared([chunk, d], dtype)
            triu_w = T.alloc_shared([chunk, chunk], dtype)
            kbar_w = T.alloc_shared([chunk, d], dtype)            # computed B operands: VCVT dst=WGT
            t_w = T.alloc_shared([chunk, chunk], dtype)
            r_w = T.alloc_shared([2, chunk, d], dtype)
            u_w = T.alloc_shared([2, chunk, d], dtype)
            S_st = T.alloc_shared([d, d], accum)                  # the resident state (B operand)

            S_acc = T.alloc_fragment([2, half, d], accum)         # the state update, two 64-row halves
            G = T.alloc_fragment([chunk, d], accum)
            Kh = T.alloc_fragment([chunk, d], accum)
            Kh_c = T.alloc_fragment([chunk, d], dtype)
            Qh = T.alloc_fragment([chunk, d], accum)
            Qh_c = T.alloc_fragment([chunk, d], dtype)
            Kb = T.alloc_fragment([chunk, d], accum)
            Vc = T.alloc_fragment([chunk, d], accum)
            Tm = T.alloc_fragment([2, chunk, chunk], accum)
            P = T.alloc_fragment([chunk, chunk], accum)
            R = T.alloc_fragment([chunk, d], accum)
            Rh = T.alloc_fragment([chunk, d], dtype)
            U = T.alloc_fragment([chunk, d], accum)
            Uh = T.alloc_fragment([chunk, d], dtype)
            O = T.alloc_fragment([chunk, d], accum)
            W = T.alloc_fragment([chunk, chunk], accum)
            W_c = T.alloc_fragment([chunk, chunk], dtype)
            Gt = T.alloc_fragment([half, chunk], accum)
            Gt_c = T.alloc_fragment([half, chunk], dtype)
            Gx = T.alloc_fragment([half, chunk], accum)
            Kt = T.alloc_fragment([half, chunk], accum)
            Kth = T.alloc_fragment([half, chunk], dtype)
            idx_s = T.alloc_fragment([chunk], "int32")
            hidx_s = T.alloc_fragment([half], "int32")
            ms_v = T.alloc_fragment([chunk, chunk], accum)
            mi_v = T.alloc_fragment([chunk, chunk], accum)
            eye_v = T.alloc_fragment([chunk, chunk], accum)
            beta_v = T.alloc_fragment([chunk], accum)
            m_v = T.alloc_fragment([half], accum)

            T.fill(S_acc, 0)
            T.fill(S_st, 0)                                           # S_0 = 0, in STATE
            T.copy(tri, tri_s)
            T.copy(triu, triu_w)
            T.copy(mstrict, ms_v)
            T.copy(mincl, mi_v)
            T.copy(eye, eye_v)

            for c in T.serial(n_chunks):
                # ---- decayed operands: G = cumsum(g) via the triangular ones GEMM; K^, Q^, K~
                T.copy(rows[c, :], idx_s)
                T.copy(g[c * chunk:(c + 1) * chunk, :], g_w)
                T.clear(G)
                T.gemm(tri_s, g_w, G)
                for i, j in T.Parallel(chunk, d):
                    Kh[i, j] = T.if_then_else(idx_s[i] >= 0, k[idx_s[i], j], T.cast(0, dtype))
                for i, j in T.Parallel(chunk, d):
                    Qh[i, j] = T.if_then_else(idx_s[i] >= 0, q[idx_s[i], j], T.cast(0, dtype))
                for i, j in T.Parallel(chunk, d):
                    Kb[i, j] = T.if_then_else(idx_s[i] >= 0, k[idx_s[i], j], T.cast(0, dtype))
                for i, j in T.Parallel(chunk, d):
                    Vc[i, j] = T.if_then_else(idx_s[i] >= 0, v[idx_s[i], j], T.cast(0, dtype))
                for i, j in T.Parallel(chunk, d):
                    Kh[i, j] = Kh[i, j] * T.exp2(G[i, j])                 # VGATE decay
                for i, j in T.Parallel(chunk, d):
                    Qh[i, j] = Qh[i, j] * T.exp2(G[i, j])
                for i, j in T.Parallel(chunk, d):
                    G[i, j] = -G[i, j]
                for i, j in T.Parallel(chunk, d):
                    Kb[i, j] = Kb[i, j] * T.exp2(G[i, j])                 # K~ = K * 2^-G
                T.copy(Kb, kbar_w)                                        # K~ as a B operand: VCVT dst=WGT
                T.copy(Kh, Kh_c)                                          # in-place bf16 (A operand)
                T.copy(Qh, Qh_c)

                # ---- N = (K^ K~^T) * M_strict * beta ; T_0 = -N ; P_0 = I + T_0
                T.copy(beta[c, :], beta_v)
                T.clear(Tm[0, :, :])
                T.gemm(Kh_c, kbar_w, Tm[0, :, :], transpose_B=True)
                for i, j in T.Parallel(chunk, chunk):
                    Tm[0, i, j] = Tm[0, i, j] * ms_v[i, j]                # VGATE mul, 0/1 mask (MMA.T has no masked form)
                for i, j in T.Parallel(chunk, chunk):
                    Tm[0, i, j] = Tm[0, i, j] * beta_v[i]
                for i, j in T.Parallel(chunk, chunk):
                    Tm[0, i, j] = -Tm[0, i, j]
                T.copy(Tm[0, :, :], P)                                    # VCOPY
                for i, j in T.Parallel(chunk, chunk):
                    P[i, j] = P[i, j] + eye_v[i, j]
                T.copy(Tm[0, :, :], t_w)                                  # T_0 as a B operand

                # ---- Appendix B: T_{s+1} = T_s^2, P <- P (I + T_{s+1}), five steps
                for s in T.serial(N_SOLVE):
                    T.copy(Tm[s % 2, :, :], ab_s)                         # bf16 A copy, T_s stays f32
                    T.clear(Tm[(s + 1) % 2, :, :])
                    T.gemm(ab_s, t_w, Tm[(s + 1) % 2, :, :])
                    T.copy(Tm[(s + 1) % 2, :, :], t_w)
                    T.copy(P, pb_s)
                    T.gemm(pb_s, t_w, P)                                  # P += P T_{s+1}

                # ---- R = beta (V - K^ S) ; U = P R  (R and U as hi + lo bf16 pairs)
                T.clear(R)
                T.gemm(Kh_c, S_st, R)
                for i, j in T.Parallel(chunk, d):
                    R[i, j] = -R[i, j]
                for i, j in T.Parallel(chunk, d):
                    R[i, j] = R[i, j] + Vc[i, j]
                for i, j in T.Parallel(chunk, d):
                    R[i, j] = R[i, j] * beta_v[i]
                T.copy(R, Rh)                                             # R_hi = bf16(R), R kept
                T.copy(Rh, r_w[0, :, :])
                for i, j in T.Parallel(chunk, d):
                    R[i, j] = R[i, j] - Rh[i, j]                          # R_lo, in place
                T.copy(R, r_w[1, :, :])
                T.copy(P, pb_s)
                T.clear(U)
                T.gemm(pb_s, r_w[0, :, :], U)
                T.gemm(pb_s, r_w[1, :, :], U)
                T.copy(U, Uh)
                T.copy(Uh, u_w[0, :, :])
                for i, j in T.Parallel(chunk, d):
                    U[i, j] = U[i, j] - Uh[i, j]
                T.copy(U, u_w[1, :, :])

                # ---- O = Q^ S + ((Q^ K~^T) * M_incl) U
                T.clear(O)
                T.gemm(Qh_c, S_st, O)
                T.clear(W)
                T.gemm(Qh_c, kbar_w, W, transpose_B=True)
                for i, j in T.Parallel(chunk, chunk):
                    W[i, j] = W[i, j] * mi_v[i, j]
                T.copy(W, W_c)
                T.gemm(W_c, u_w[0, :, :], O)
                T.gemm(W_c, u_w[1, :, :], O)
                T.copy(O, o[c * chunk:(c + 1) * chunk, :])

                # ---- state update per 64-channel half: S_h <- 2^G_last S_h + (K^T * 2^(G_last - G_j)) U
                for h in T.serial(2):
                    T.copy(hrows[h, :], hidx_s)
                    for i, j in T.Parallel(half, chunk):
                        Gt[i, j] = T.if_then_else(hidx_s[i] >= 0, gt[hidx_s[i], c * chunk + j], T.cast(0, dtype))
                    T.copy(Gt, Gt_c)
                    T.clear(Gx)
                    T.gemm(Gt_c, triu_w, Gx)                              # G^T (cumsum along tokens)
                    for i, j in T.Parallel(half, chunk):
                        Gx[i, j] = -Gx[i, j]
                    T.reduce_max(Gx, m_v, dim=1, clear=True)              # m = -G_last (G is non-increasing)
                    for i, j in T.Parallel(half, chunk):
                        Gx[i, j] = Gx[i, j] - m_v[i]                      # G_last - G_j
                    for i, j in T.Parallel(half, chunk):
                        Kt[i, j] = T.if_then_else(hidx_s[i] >= 0, kt[hidx_s[i], c * chunk + j], T.cast(0, dtype))
                    for i, j in T.Parallel(half, chunk):
                        Kt[i, j] = Kt[i, j] * T.exp2(Gx[i, j])
                    T.copy(Kt, Kth)                                       # hi = bf16(Kt), Kt kept
                    T.copy(Kth, Kth_c)                                    # A operand
                    for i, j in T.Parallel(half, chunk):
                        Kt[i, j] = Kt[i, j] - Kth[i, j]                   # lo, in place
                    T.copy(Kt, Ktl_c)
                    for i in T.Parallel(half):
                        m_v[i] = -m_v[i]
                    for i in T.Parallel(half):
                        m_v[i] = T.exp2(m_v[i])                           # 2^G_last
                    for i, j in T.Parallel(half, d):
                        S_acc[h, i, j] = S_acc[h, i, j] * m_v[i]          # decay
                    T.gemm(Kth_c, u_w[0, :, :], S_acc[h, :, :])
                    T.gemm(Kth_c, u_w[1, :, :], S_acc[h, :, :])
                    T.gemm(Ktl_c, u_w[0, :, :], S_acc[h, :, :])
                    T.copy(S_acc[h, :, :], S_st[h * half:(h + 1) * half, :])          # VCVT dst=STATE, row0
                    T.copy(S_acc[h, :, :], S_all[c + 1, h * half:(h + 1) * half, :])  # checkpoint

    return main


# ---------------------------------------------------------------- host side

def host_inputs(fx: dict) -> dict:
    """The DRAM images: bf16 q/k/v, log2-domain gate and its transpose, k transposed, f32 beta,
    identity gather rows, the triangular constants and the 0/1 masks."""
    C = CHUNK
    g = common.bf16(np.log2(fx["alpha"].astype(np.float64)).astype(np.float32))
    return {
        "q": common.bf16_bytes(fx["q"]), "k": common.bf16_bytes(fx["k"]), "v": common.bf16_bytes(fx["v"]),
        "g": common.bf16_bytes(g), "gt": common.bf16_bytes(np.ascontiguousarray(g.T)),
        "kt": common.bf16_bytes(np.ascontiguousarray(fx["k"].T)),
        "beta": fx["beta"].astype(np.float32).reshape(-1, C),
        "rows": np.arange(fx["q"].shape[0], dtype=np.int32).reshape(-1, C), "hrows": np.arange(D, dtype=np.int32).reshape(2, -1),
        "tri": common.bf16_bytes(np.tril(np.ones((C, C), np.float32))),
        "triu": common.bf16_bytes(np.triu(np.ones((C, C), np.float32))),
        "mstrict": np.tril(np.ones((C, C), np.float32), -1), "mincl": np.tril(np.ones((C, C), np.float32)),
        "eye": np.eye(C, dtype=np.float32),
    }


def load_dram(m, fx: dict) -> None:
    for name, arr in host_inputs(fx).items():
        m.dram_write(DRAM[name], np.ascontiguousarray(arr).ravel())
    m.dram_write(DRAM["S_all"], np.zeros((N_CHUNKS + 1) * D * D, np.float32))   # S_all[0] = 0 (host-side convention)


def read_outputs(m, fx: dict):
    """-> (o [T, d], S_all[1:] [n_chunks, d, d])."""
    T_, d = fx["o"].shape
    o = m.dram[DRAM["o"]: DRAM["o"] + 4 * T_ * d].view(np.float32).reshape(T_, d).copy()
    n = (N_CHUNKS + 1) * d * d
    S_all = m.dram[DRAM["S_all"]: DRAM["S_all"] + 4 * n].view(np.float32).reshape(N_CHUNKS + 1, d, d).copy()
    return o, S_all[1:]
