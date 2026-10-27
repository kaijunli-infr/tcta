"""KDA chunkwise forward, the ISA v0.1.2 authoring (phase 4, G2), kept for the cycle model's before/after
comparison of the DRAM round trips (phase 5 H3, `dram_latency` sweep). `kda_kernel.py` is the current form.

Differences to `kda_kernel.py`: every device-computed B operand (K~, the T_s of the solve, R, U) takes a
STORE.2D -> DRAM scratch -> LOAD.2D round trip into WGT (13 per chunk), the [128, 128] state is two ACC
halves mirrored to DRAM and reloaded into STATE (64 KB `LOAD.2D` per chunk), and the hi/lo splits keep
separate lo buffers. Same algebra, same numerics (o 5.99e-3, S <= 1.46e-3), same 27 MMAs per chunk. Maps and
runs on the v0.1.3 simulator (D4 emits the kept-source bf16 casts with `to_bf16`).
"""

import os
import sys

import tilelang.language as T

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kda_kernel import CHUNK, CONFIG_C, D, MB, N_CHUNKS, N_SOLVE, SEQ, DRAM_BYTES, host_inputs, read_outputs  # noqa: E402,F401

SCOPE_MAP = {
    "tri_s": "tcta.inp", "pb_s": "tcta.inp", "ab_s": "tcta.inp",
    "g_w": "tcta.wgt", "triu_w": "tcta.wgt", "kbar_w": "tcta.wgt", "t_w": "tcta.wgt", "r_w": "tcta.wgt", "u_w": "tcta.wgt",
    "S_st": "tcta.state",
    "S_acc": "tcta.acc", "G": "tcta.acc", "Kh": "tcta.acc", "Kh_c": "tcta.acc", "Qh": "tcta.acc", "Qh_c": "tcta.acc",
    "Kb": "tcta.acc", "Vc": "tcta.acc", "Tm": "tcta.acc", "P": "tcta.acc", "R": "tcta.acc", "Rh": "tcta.acc", "Rl": "tcta.acc",
    "U": "tcta.acc", "Uh": "tcta.acc", "Ul": "tcta.acc", "O": "tcta.acc", "W": "tcta.acc", "W_c": "tcta.acc",
    "Gt": "tcta.acc", "Gt_c": "tcta.acc", "Gx": "tcta.acc", "Kt": "tcta.acc", "Kth": "tcta.acc", "Ktl": "tcta.acc",
    "Kth_c": "tcta.acc", "Ktl_c": "tcta.acc",
    "idx_s": "tcta.vec", "hidx_s": "tcta.vec", "ms_v": "tcta.vec", "mi_v": "tcta.vec", "eye_v": "tcta.vec",
    "beta_v": "tcta.vec", "m_v": "tcta.vec",
}

DRAM = {"q": 0, "k": 64 * 1024, "v": 128 * 1024, "g": 192 * 1024, "gt": 256 * 1024, "kt": 320 * 1024,
        "beta": 384 * 1024, "rows": 388 * 1024, "hrows": 392 * 1024,
        "tri": 400 * 1024, "triu": 408 * 1024, "mstrict": 416 * 1024, "mincl": 432 * 1024, "eye": 448 * 1024,
        "scr_kbar": 1 * MB, "scr_t": 1 * MB + 64 * 1024, "scr_r": 1 * MB + 128 * 1024, "scr_u": 1 * MB + 192 * 1024,
        "S_all": 2 * MB, "o": 3 * MB}


def build_kda_kernel(seq: int = SEQ, d: int = D, chunk: int = CHUNK, threads: int = 128, dtype: str = "float16"):
    assert d == 128 and chunk == 64 and seq % chunk == 0
    n_chunks = seq // chunk
    accum = "float32"
    half = d // 2

    @T.prim_func
    def main(
        q: T.Tensor([seq, d], dtype),
        k: T.Tensor([seq, d], dtype),
        v: T.Tensor([seq, d], dtype),
        g: T.Tensor([seq, d], dtype),
        gt: T.Tensor([d, seq], dtype),
        kt: T.Tensor([d, seq], dtype),
        beta: T.Tensor([n_chunks, chunk], accum),
        rows: T.Tensor([n_chunks, chunk], "int32"),
        hrows: T.Tensor([2, half], "int32"),
        tri: T.Tensor([chunk, chunk], dtype),
        triu: T.Tensor([chunk, chunk], dtype),
        mstrict: T.Tensor([chunk, chunk], accum),
        mincl: T.Tensor([chunk, chunk], accum),
        eye: T.Tensor([chunk, chunk], accum),
        scr_kbar: T.Tensor([chunk, d], dtype),
        scr_t: T.Tensor([chunk, chunk], dtype),
        scr_r: T.Tensor([2, chunk, d], dtype),
        scr_u: T.Tensor([2, chunk, d], dtype),
        S_all: T.Tensor([n_chunks + 1, d, d], accum),
        o: T.Tensor([seq, d], accum),
    ):
        with T.Kernel(1, threads=threads) as (bx,):
            tri_s = T.alloc_shared([chunk, chunk], dtype)
            pb_s = T.alloc_shared([chunk, chunk], dtype)
            ab_s = T.alloc_shared([chunk, chunk], dtype)
            g_w = T.alloc_shared([chunk, d], dtype)
            triu_w = T.alloc_shared([chunk, chunk], dtype)
            kbar_w = T.alloc_shared([chunk, d], dtype)
            t_w = T.alloc_shared([chunk, chunk], dtype)
            r_w = T.alloc_shared([2, chunk, d], dtype)
            u_w = T.alloc_shared([2, chunk, d], dtype)
            S_st = T.alloc_shared([d, d], accum)

            S_acc = T.alloc_fragment([2, half, d], accum)
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
            Rl = T.alloc_fragment([chunk, d], accum)
            U = T.alloc_fragment([chunk, d], accum)
            Uh = T.alloc_fragment([chunk, d], dtype)
            Ul = T.alloc_fragment([chunk, d], accum)
            O = T.alloc_fragment([chunk, d], accum)
            W = T.alloc_fragment([chunk, chunk], accum)
            W_c = T.alloc_fragment([chunk, chunk], dtype)
            Gt = T.alloc_fragment([half, chunk], accum)
            Gt_c = T.alloc_fragment([half, chunk], dtype)
            Gx = T.alloc_fragment([half, chunk], accum)
            Kt = T.alloc_fragment([half, chunk], accum)
            Kth = T.alloc_fragment([half, chunk], dtype)
            Ktl = T.alloc_fragment([half, chunk], accum)
            Kth_c = T.alloc_fragment([half, chunk], dtype)
            Ktl_c = T.alloc_fragment([half, chunk], dtype)
            idx_s = T.alloc_fragment([chunk], "int32")
            hidx_s = T.alloc_fragment([half], "int32")
            ms_v = T.alloc_fragment([chunk, chunk], accum)
            mi_v = T.alloc_fragment([chunk, chunk], accum)
            eye_v = T.alloc_fragment([chunk, chunk], accum)
            beta_v = T.alloc_fragment([chunk], accum)
            m_v = T.alloc_fragment([half], accum)

            T.fill(S_acc, 0)
            T.copy(tri, tri_s)
            T.copy(mstrict, ms_v)
            T.copy(mincl, mi_v)
            T.copy(eye, eye_v)

            for c in T.serial(n_chunks):
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
                    Kh[i, j] = Kh[i, j] * T.exp2(G[i, j])
                for i, j in T.Parallel(chunk, d):
                    Qh[i, j] = Qh[i, j] * T.exp2(G[i, j])
                for i, j in T.Parallel(chunk, d):
                    G[i, j] = -G[i, j]
                for i, j in T.Parallel(chunk, d):
                    Kb[i, j] = Kb[i, j] * T.exp2(G[i, j])
                T.copy(Kb, scr_kbar)                                      # round trip 1
                T.copy(scr_kbar, kbar_w)
                T.copy(Kh, Kh_c)
                T.copy(Qh, Qh_c)

                T.copy(beta[c, :], beta_v)
                T.clear(Tm[0, :, :])
                T.gemm(Kh_c, kbar_w, Tm[0, :, :], transpose_B=True)
                for i, j in T.Parallel(chunk, chunk):
                    Tm[0, i, j] = Tm[0, i, j] * ms_v[i, j]
                for i, j in T.Parallel(chunk, chunk):
                    Tm[0, i, j] = Tm[0, i, j] * beta_v[i]
                for i, j in T.Parallel(chunk, chunk):
                    Tm[0, i, j] = -Tm[0, i, j]
                T.copy(Tm[0, :, :], P)
                for i, j in T.Parallel(chunk, chunk):
                    P[i, j] = P[i, j] + eye_v[i, j]
                T.copy(Tm[0, :, :], scr_t)                                # round trip 2
                T.copy(scr_t, t_w)

                for s in T.serial(N_SOLVE):
                    T.copy(Tm[s % 2, :, :], ab_s)
                    T.clear(Tm[(s + 1) % 2, :, :])
                    T.gemm(ab_s, t_w, Tm[(s + 1) % 2, :, :])
                    T.copy(Tm[(s + 1) % 2, :, :], scr_t)                  # round trips 3..7
                    T.copy(scr_t, t_w)
                    T.copy(P, pb_s)
                    T.gemm(pb_s, t_w, P)

                T.copy(S_all[c, :, :], S_st)                              # state reload from the mirror
                T.clear(R)
                T.gemm(Kh_c, S_st, R)
                for i, j in T.Parallel(chunk, d):
                    R[i, j] = -R[i, j]
                for i, j in T.Parallel(chunk, d):
                    R[i, j] = R[i, j] + Vc[i, j]
                for i, j in T.Parallel(chunk, d):
                    R[i, j] = R[i, j] * beta_v[i]
                T.copy(R, Rh)
                for i, j in T.Parallel(chunk, d):
                    Rl[i, j] = R[i, j] - Rh[i, j]
                T.copy(Rh, scr_r[0, :, :])                                # round trips 8, 9
                T.copy(Rl, scr_r[1, :, :])
                T.copy(scr_r[0, :, :], r_w[0, :, :])
                T.copy(scr_r[1, :, :], r_w[1, :, :])
                T.copy(P, pb_s)
                T.clear(U)
                T.gemm(pb_s, r_w[0, :, :], U)
                T.gemm(pb_s, r_w[1, :, :], U)
                T.copy(U, Uh)
                for i, j in T.Parallel(chunk, d):
                    Ul[i, j] = U[i, j] - Uh[i, j]
                T.copy(Uh, scr_u[0, :, :])                                # round trips 10, 11
                T.copy(Ul, scr_u[1, :, :])
                T.copy(scr_u[0, :, :], u_w[0, :, :])
                T.copy(scr_u[1, :, :], u_w[1, :, :])

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

                for h in T.serial(2):
                    T.copy(hrows[h, :], hidx_s)
                    T.copy(triu, triu_w)
                    for i, j in T.Parallel(half, chunk):
                        Gt[i, j] = T.if_then_else(hidx_s[i] >= 0, gt[hidx_s[i], c * chunk + j], T.cast(0, dtype))
                    T.copy(Gt, Gt_c)
                    T.clear(Gx)
                    T.gemm(Gt_c, triu_w, Gx)
                    for i, j in T.Parallel(half, chunk):
                        Gx[i, j] = -Gx[i, j]
                    T.reduce_max(Gx, m_v, dim=1, clear=True)
                    for i, j in T.Parallel(half, chunk):
                        Gx[i, j] = Gx[i, j] - m_v[i]
                    for i, j in T.Parallel(half, chunk):
                        Kt[i, j] = T.if_then_else(hidx_s[i] >= 0, kt[hidx_s[i], c * chunk + j], T.cast(0, dtype))
                    for i, j in T.Parallel(half, chunk):
                        Kt[i, j] = Kt[i, j] * T.exp2(Gx[i, j])
                    T.copy(Kt, Kth)
                    for i, j in T.Parallel(half, chunk):
                        Ktl[i, j] = Kt[i, j] - Kth[i, j]
                    T.copy(Kth, Kth_c)
                    T.copy(Ktl, Ktl_c)
                    for i in T.Parallel(half):
                        m_v[i] = -m_v[i]
                    for i in T.Parallel(half):
                        m_v[i] = T.exp2(m_v[i])
                    for i, j in T.Parallel(half, d):
                        S_acc[h, i, j] = S_acc[h, i, j] * m_v[i]
                    T.gemm(Kth_c, u_w[0, :, :], S_acc[h, :, :])
                    T.gemm(Kth_c, u_w[1, :, :], S_acc[h, :, :])
                    T.gemm(Ktl_c, u_w[0, :, :], S_acc[h, :, :])
                    T.copy(S_acc[h, :, :], S_all[c + 1, h * half:(h + 1) * half, :])   # mirror + checkpoint

    return main


def load_dram(m, fx: dict) -> None:
    import numpy as np
    for name, arr in host_inputs(fx).items():
        m.dram_write(DRAM[name], np.ascontiguousarray(arr).ravel())
    m.dram_write(DRAM["S_all"], np.zeros((N_CHUNKS + 1) * D * D, np.float32))   # S_all[0] = 0: the first reload
