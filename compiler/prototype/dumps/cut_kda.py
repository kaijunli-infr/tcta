# G1/G2 output: lower_to_tile_tir(kda_kernel.build_kda_kernel()) at tl.InjectSoftwarePipeline
# from tvm.script import tirx as T

@T.prim_func
def main(q_handle: T.handle, k_handle: T.handle, v_handle: T.handle, g_handle: T.handle, gt_handle: T.handle, kt_handle: T.handle, beta_handle: T.handle, rows_handle: T.handle, hrows_handle: T.handle, tri_handle: T.handle, triu_handle: T.handle, mstrict_handle: T.handle, mincl_handle: T.handle, eye_handle: T.handle, scr_kbar_handle: T.handle, scr_t_handle: T.handle, scr_r_handle: T.handle, scr_u_handle: T.handle, S_all_handle: T.handle, o_handle: T.handle, bx: T.int32):
    T.func_attr({"target": T.target({"arch": "sm_80", "keys": ["cuda", "gpu"], "kind": "cuda", "max_num_threads": 1024, "tag": "", "thread_warp_size": 32}), "tcta.arch": "sm_80", "tcta.cut": "tl.InjectSoftwarePipeline"})
    q = T.match_buffer(q_handle, (256, 128), "float16", strides=(128, 1))
    k = T.match_buffer(k_handle, (256, 128), "float16", strides=(128, 1))
    v = T.match_buffer(v_handle, (256, 128), "float16", strides=(128, 1))
    g = T.match_buffer(g_handle, (256, 128), "float16", strides=(128, 1))
    gt = T.match_buffer(gt_handle, (128, 256), "float16", strides=(256, 1))
    kt = T.match_buffer(kt_handle, (128, 256), "float16", strides=(256, 1))
    beta = T.match_buffer(beta_handle, (4, 64), strides=(64, 1))
    rows = T.match_buffer(rows_handle, (4, 64), "int32", strides=(64, 1))
    hrows = T.match_buffer(hrows_handle, (2, 64), "int32", strides=(64, 1))
    tri = T.match_buffer(tri_handle, (64, 64), "float16", strides=(64, 1))
    triu = T.match_buffer(triu_handle, (64, 64), "float16", strides=(64, 1))
    mstrict = T.match_buffer(mstrict_handle, (64, 64), strides=(64, 1))
    mincl = T.match_buffer(mincl_handle, (64, 64), strides=(64, 1))
    eye = T.match_buffer(eye_handle, (64, 64), strides=(64, 1))
    scr_kbar = T.match_buffer(scr_kbar_handle, (64, 128), "float16", strides=(128, 1))
    scr_t = T.match_buffer(scr_t_handle, (64, 64), "float16", strides=(64, 1))
    scr_r = T.match_buffer(scr_r_handle, (2, 64, 128), "float16", strides=(8192, 128, 1))
    scr_u = T.match_buffer(scr_u_handle, (2, 64, 128), "float16", strides=(8192, 128, 1))
    S_all = T.match_buffer(S_all_handle, (5, 128, 128), strides=(16384, 128, 1))
    o = T.match_buffer(o_handle, (256, 128), strides=(128, 1))
    with T.sblock("root"):
        T.reads()
        T.writes()
        with T.sblock("tilelang_root"):
            T.reads()
            T.writes()
            tri_s = T.sblock_alloc_buffer((64, 64), "float16", scope="tcta.inp")
            pb_s = T.sblock_alloc_buffer((64, 64), "float16", scope="tcta.inp")
            ab_s = T.sblock_alloc_buffer((64, 64), "float16", scope="tcta.inp")
            g_w = T.sblock_alloc_buffer((64, 128), "float16", scope="tcta.wgt")
            triu_w = T.sblock_alloc_buffer((64, 64), "float16", scope="tcta.wgt")
            kbar_w = T.sblock_alloc_buffer((64, 128), "float16", scope="tcta.wgt")
            t_w = T.sblock_alloc_buffer((64, 64), "float16", scope="tcta.wgt")
            r_w = T.sblock_alloc_buffer((2, 64, 128), "float16", scope="tcta.wgt")
            u_w = T.sblock_alloc_buffer((2, 64, 128), "float16", scope="tcta.wgt")
            S_st = T.sblock_alloc_buffer((128, 128), scope="tcta.state")
            S_acc = T.sblock_alloc_buffer((2, 64, 128), scope="tcta.acc")
            G = T.sblock_alloc_buffer((64, 128), scope="tcta.acc")
            Kh = T.sblock_alloc_buffer((64, 128), scope="tcta.acc")
            Kh_c = T.sblock_alloc_buffer((64, 128), "float16", scope="tcta.acc")
            Qh = T.sblock_alloc_buffer((64, 128), scope="tcta.acc")
            Qh_c = T.sblock_alloc_buffer((64, 128), "float16", scope="tcta.acc")
            Kb = T.sblock_alloc_buffer((64, 128), scope="tcta.acc")
            Vc = T.sblock_alloc_buffer((64, 128), scope="tcta.acc")
            Tm = T.sblock_alloc_buffer((2, 64, 64), scope="tcta.acc")
            P = T.sblock_alloc_buffer((64, 64), scope="tcta.acc")
            R_1 = T.sblock_alloc_buffer((64, 128), scope="tcta.acc")
            Rh = T.sblock_alloc_buffer((64, 128), "float16", scope="tcta.acc")
            Rl = T.sblock_alloc_buffer((64, 128), scope="tcta.acc")
            U = T.sblock_alloc_buffer((64, 128), scope="tcta.acc")
            Uh = T.sblock_alloc_buffer((64, 128), "float16", scope="tcta.acc")
            Ul = T.sblock_alloc_buffer((64, 128), scope="tcta.acc")
            O = T.sblock_alloc_buffer((64, 128), scope="tcta.acc")
            W = T.sblock_alloc_buffer((64, 64), scope="tcta.acc")
            W_c = T.sblock_alloc_buffer((64, 64), "float16", scope="tcta.acc")
            Gt = T.sblock_alloc_buffer((64, 64), scope="tcta.acc")
            Gt_c = T.sblock_alloc_buffer((64, 64), "float16", scope="tcta.acc")
            Gx = T.sblock_alloc_buffer((64, 64), scope="tcta.acc")
            Kt = T.sblock_alloc_buffer((64, 64), scope="tcta.acc")
            Kth = T.sblock_alloc_buffer((64, 64), "float16", scope="tcta.acc")
            Ktl = T.sblock_alloc_buffer((64, 64), scope="tcta.acc")
            Kth_c = T.sblock_alloc_buffer((64, 64), "float16", scope="tcta.acc")
            Ktl_c = T.sblock_alloc_buffer((64, 64), "float16", scope="tcta.acc")
            idx_s = T.sblock_alloc_buffer((64,), "int32", scope="tcta.vec")
            hidx_s = T.sblock_alloc_buffer((64,), "int32", scope="tcta.vec")
            ms_v = T.sblock_alloc_buffer((64, 64), scope="tcta.vec")
            mi_v = T.sblock_alloc_buffer((64, 64), scope="tcta.vec")
            eye_v = T.sblock_alloc_buffer((64, 64), scope="tcta.vec")
            beta_v = T.sblock_alloc_buffer((64,), scope="tcta.vec")
            m_v = T.sblock_alloc_buffer((64,), scope="tcta.vec")
            T.fill(T.region(S_acc[0, 0, 0], 2, 2, 64, 128), 0)
            T.copy(T.region(tri[0, 0], 1, 64, 64), T.region(tri_s[0, 0], 2, 64, 64))
            T.copy(T.region(mstrict[0, 0], 1, 64, 64), T.region(ms_v[0, 0], 2, 64, 64))
            T.copy(T.region(mincl[0, 0], 1, 64, 64), T.region(mi_v[0, 0], 2, 64, 64))
            T.copy(T.region(eye[0, 0], 1, 64, 64), T.region(eye_v[0, 0], 2, 64, 64))
            for c in range(4):
                T.copy(T.region(rows[c, 0], 1, 1, 64), T.region(idx_s[0], 2, 64))
                T.copy(T.region(g[c * 64, 0], 1, 64, 128), T.region(g_w[0, 0], 2, 64, 128))
                T.fill(T.region(G[0, 0], 2, 64, 128), 0)
                T.gemm(T.region(tri_s[0, 0], 1, 64, 64), T.region(g_w[0, 0], 1, 64, 128), T.region(G[0, 0], 3, 64, 128), T.bool(False), T.bool(False), 64, 128, 64, 0, T.bool(False), 0, 0, 0)
                for i in T.parallel(64):
                    for j in T.parallel(128):
                        Kh[i, j] = T.Cast("float32", T.if_then_else(0 <= idx_s[i], k[idx_s[i], j], T.float16(0.0)))
                for i in T.parallel(64):
                    for j in T.parallel(128):
                        Qh[i, j] = T.Cast("float32", T.if_then_else(0 <= idx_s[i], q[idx_s[i], j], T.float16(0.0)))
                for i in T.parallel(64):
                    for j in T.parallel(128):
                        Kb[i, j] = T.Cast("float32", T.if_then_else(0 <= idx_s[i], k[idx_s[i], j], T.float16(0.0)))
                for i in T.parallel(64):
                    for j in T.parallel(128):
                        Vc[i, j] = T.Cast("float32", T.if_then_else(0 <= idx_s[i], v[idx_s[i], j], T.float16(0.0)))
                for i in T.parallel(64):
                    for j in T.parallel(128):
                        Kh[i, j] = Kh[i, j] * T.exp2(G[i, j])
                for i in T.parallel(64):
                    for j in T.parallel(128):
                        Qh[i, j] = Qh[i, j] * T.exp2(G[i, j])
                for i in T.parallel(64):
                    for j in T.parallel(128):
                        G[i, j] = G[i, j] * T.float32(-1.0)
                for i in T.parallel(64):
                    for j in T.parallel(128):
                        Kb[i, j] = Kb[i, j] * T.exp2(G[i, j])
                T.copy(T.region(Kb[0, 0], 1, 64, 128), T.region(scr_kbar[0, 0], 2, 64, 128))
                T.copy(T.region(scr_kbar[0, 0], 1, 64, 128), T.region(kbar_w[0, 0], 2, 64, 128))
                T.copy(T.region(Kh[0, 0], 1, 64, 128), T.region(Kh_c[0, 0], 2, 64, 128))
                T.copy(T.region(Qh[0, 0], 1, 64, 128), T.region(Qh_c[0, 0], 2, 64, 128))
                T.copy(T.region(beta[c, 0], 1, 1, 64), T.region(beta_v[0], 2, 64))
                T.fill(T.region(Tm[0, 0, 0], 2, 1, 64, 64), 0)
                T.gemm(T.region(Kh_c[0, 0], 1, 64, 128), T.region(kbar_w[0, 0], 1, 64, 128), T.region(Tm[0, 0, 0], 3, 1, 64, 64), T.bool(False), T.bool(True), 64, 64, 128, 0, T.bool(False), 0, 0, 0)
                for i in T.parallel(64):
                    for j in T.parallel(64):
                        Tm[0, i, j] = Tm[0, i, j] * ms_v[i, j]
                for i in T.parallel(64):
                    for j in T.parallel(64):
                        Tm[0, i, j] = Tm[0, i, j] * beta_v[i]
                for i in T.parallel(64):
                    for j in T.parallel(64):
                        Tm[0, i, j] = Tm[0, i, j] * T.float32(-1.0)
                T.copy(T.region(Tm[0, 0, 0], 1, 1, 64, 64), T.region(P[0, 0], 2, 64, 64))
                for i in T.parallel(64):
                    for j in T.parallel(64):
                        P[i, j] = P[i, j] + eye_v[i, j]
                T.copy(T.region(Tm[0, 0, 0], 1, 1, 64, 64), T.region(scr_t[0, 0], 2, 64, 64))
                T.copy(T.region(scr_t[0, 0], 1, 64, 64), T.region(t_w[0, 0], 2, 64, 64))
                for s in range(5):
                    T.copy(T.region(Tm[s % 2, 0, 0], 1, 1, 64, 64), T.region(ab_s[0, 0], 2, 64, 64))
                    T.fill(T.region(Tm[(s + 1) % 2, 0, 0], 2, 1, 64, 64), 0)
                    T.gemm(T.region(ab_s[0, 0], 1, 64, 64), T.region(t_w[0, 0], 1, 64, 64), T.region(Tm[(s + 1) % 2, 0, 0], 3, 1, 64, 64), T.bool(False), T.bool(False), 64, 64, 64, 0, T.bool(False), 0, 0, 0)
                    T.copy(T.region(Tm[(s + 1) % 2, 0, 0], 1, 1, 64, 64), T.region(scr_t[0, 0], 2, 64, 64))
                    T.copy(T.region(scr_t[0, 0], 1, 64, 64), T.region(t_w[0, 0], 2, 64, 64))
                    T.copy(T.region(P[0, 0], 1, 64, 64), T.region(pb_s[0, 0], 2, 64, 64))
                    T.gemm(T.region(pb_s[0, 0], 1, 64, 64), T.region(t_w[0, 0], 1, 64, 64), T.region(P[0, 0], 3, 64, 64), T.bool(False), T.bool(False), 64, 64, 64, 0, T.bool(False), 0, 0, 0)
                T.copy(T.region(S_all[c, 0, 0], 1, 1, 128, 128), T.region(S_st[0, 0], 2, 128, 128))
                T.fill(T.region(R_1[0, 0], 2, 64, 128), 0)
                T.gemm(T.region(Kh_c[0, 0], 1, 64, 128), T.region(S_st[0, 0], 1, 128, 128), T.region(R_1[0, 0], 3, 64, 128), T.bool(False), T.bool(False), 64, 128, 128, 0, T.bool(False), 0, 0, 0)
                for i in T.parallel(64):
                    for j in T.parallel(128):
                        R_1[i, j] = R_1[i, j] * T.float32(-1.0)
                for i in T.parallel(64):
                    for j in T.parallel(128):
                        R_1[i, j] = R_1[i, j] + Vc[i, j]
                for i in T.parallel(64):
                    for j in T.parallel(128):
                        R_1[i, j] = R_1[i, j] * beta_v[i]
                T.copy(T.region(R_1[0, 0], 1, 64, 128), T.region(Rh[0, 0], 2, 64, 128))
                for i in T.parallel(64):
                    for j in T.parallel(128):
                        Rl[i, j] = R_1[i, j] - T.Cast("float32", Rh[i, j])
                T.copy(T.region(Rh[0, 0], 1, 64, 128), T.region(scr_r[0, 0, 0], 2, 1, 64, 128))
                T.copy(T.region(Rl[0, 0], 1, 64, 128), T.region(scr_r[1, 0, 0], 2, 1, 64, 128))
                T.copy(T.region(scr_r[0, 0, 0], 1, 1, 64, 128), T.region(r_w[0, 0, 0], 2, 1, 64, 128))
                T.copy(T.region(scr_r[1, 0, 0], 1, 1, 64, 128), T.region(r_w[1, 0, 0], 2, 1, 64, 128))
                T.copy(T.region(P[0, 0], 1, 64, 64), T.region(pb_s[0, 0], 2, 64, 64))
                T.fill(T.region(U[0, 0], 2, 64, 128), 0)
                T.gemm(T.region(pb_s[0, 0], 1, 64, 64), T.region(r_w[0, 0, 0], 1, 1, 64, 128), T.region(U[0, 0], 3, 64, 128), T.bool(False), T.bool(False), 64, 128, 64, 0, T.bool(False), 0, 0, 0)
                T.gemm(T.region(pb_s[0, 0], 1, 64, 64), T.region(r_w[1, 0, 0], 1, 1, 64, 128), T.region(U[0, 0], 3, 64, 128), T.bool(False), T.bool(False), 64, 128, 64, 0, T.bool(False), 0, 0, 0)
                T.copy(T.region(U[0, 0], 1, 64, 128), T.region(Uh[0, 0], 2, 64, 128))
                for i in T.parallel(64):
                    for j in T.parallel(128):
                        Ul[i, j] = U[i, j] - T.Cast("float32", Uh[i, j])
                T.copy(T.region(Uh[0, 0], 1, 64, 128), T.region(scr_u[0, 0, 0], 2, 1, 64, 128))
                T.copy(T.region(Ul[0, 0], 1, 64, 128), T.region(scr_u[1, 0, 0], 2, 1, 64, 128))
                T.copy(T.region(scr_u[0, 0, 0], 1, 1, 64, 128), T.region(u_w[0, 0, 0], 2, 1, 64, 128))
                T.copy(T.region(scr_u[1, 0, 0], 1, 1, 64, 128), T.region(u_w[1, 0, 0], 2, 1, 64, 128))
                T.fill(T.region(O[0, 0], 2, 64, 128), 0)
                T.gemm(T.region(Qh_c[0, 0], 1, 64, 128), T.region(S_st[0, 0], 1, 128, 128), T.region(O[0, 0], 3, 64, 128), T.bool(False), T.bool(False), 64, 128, 128, 0, T.bool(False), 0, 0, 0)
                T.fill(T.region(W[0, 0], 2, 64, 64), 0)
                T.gemm(T.region(Qh_c[0, 0], 1, 64, 128), T.region(kbar_w[0, 0], 1, 64, 128), T.region(W[0, 0], 3, 64, 64), T.bool(False), T.bool(True), 64, 64, 128, 0, T.bool(False), 0, 0, 0)
                for i in T.parallel(64):
                    for j in T.parallel(64):
                        W[i, j] = W[i, j] * mi_v[i, j]
                T.copy(T.region(W[0, 0], 1, 64, 64), T.region(W_c[0, 0], 2, 64, 64))
                T.gemm(T.region(W_c[0, 0], 1, 64, 64), T.region(u_w[0, 0, 0], 1, 1, 64, 128), T.region(O[0, 0], 3, 64, 128), T.bool(False), T.bool(False), 64, 128, 64, 0, T.bool(False), 0, 0, 0)
                T.gemm(T.region(W_c[0, 0], 1, 64, 64), T.region(u_w[1, 0, 0], 1, 1, 64, 128), T.region(O[0, 0], 3, 64, 128), T.bool(False), T.bool(False), 64, 128, 64, 0, T.bool(False), 0, 0, 0)
                T.copy(T.region(O[0, 0], 1, 64, 128), T.region(o[c * 64, 0], 2, 64, 128))
                for h in range(2):
                    T.copy(T.region(hrows[h, 0], 1, 1, 64), T.region(hidx_s[0], 2, 64))
                    T.copy(T.region(triu[0, 0], 1, 64, 64), T.region(triu_w[0, 0], 2, 64, 64))
                    for i in T.parallel(64):
                        for j in T.parallel(64):
                            Gt[i, j] = T.Cast("float32", T.if_then_else(0 <= hidx_s[i], gt[hidx_s[i], c * 64 + j], T.float16(0.0)))
                    T.copy(T.region(Gt[0, 0], 1, 64, 64), T.region(Gt_c[0, 0], 2, 64, 64))
                    T.fill(T.region(Gx[0, 0], 2, 64, 64), 0)
                    T.gemm(T.region(Gt_c[0, 0], 1, 64, 64), T.region(triu_w[0, 0], 1, 64, 64), T.region(Gx[0, 0], 3, 64, 64), T.bool(False), T.bool(False), 64, 64, 64, 0, T.bool(False), 0, 0, 0)
                    for i in T.parallel(64):
                        for j in T.parallel(64):
                            Gx[i, j] = Gx[i, j] * T.float32(-1.0)
                    T.reduce(T.region(Gx[0, 0], 1, 64, 64), T.region(m_v[0], 2, 64), "max", 1, T.bool(True))
                    for i in T.parallel(64):
                        for j in T.parallel(64):
                            Gx[i, j] = Gx[i, j] - m_v[i]
                    for i in T.parallel(64):
                        for j in T.parallel(64):
                            Kt[i, j] = T.Cast("float32", T.if_then_else(0 <= hidx_s[i], kt[hidx_s[i], c * 64 + j], T.float16(0.0)))
                    for i in T.parallel(64):
                        for j in T.parallel(64):
                            Kt[i, j] = Kt[i, j] * T.exp2(Gx[i, j])
                    T.copy(T.region(Kt[0, 0], 1, 64, 64), T.region(Kth[0, 0], 2, 64, 64))
                    for i in T.parallel(64):
                        for j in T.parallel(64):
                            Ktl[i, j] = Kt[i, j] - T.Cast("float32", Kth[i, j])
                    T.copy(T.region(Kth[0, 0], 1, 64, 64), T.region(Kth_c[0, 0], 2, 64, 64))
                    T.copy(T.region(Ktl[0, 0], 1, 64, 64), T.region(Ktl_c[0, 0], 2, 64, 64))
                    for i in T.parallel(64):
                        m_v[i] = m_v[i] * T.float32(-1.0)
                    for i in T.parallel(64):
                        m_v[i] = T.exp2(m_v[i])
                    for i in T.parallel(64):
                        for j in T.parallel(128):
                            S_acc[h, i, j] = S_acc[h, i, j] * m_v[i]
                    T.gemm(T.region(Kth_c[0, 0], 1, 64, 64), T.region(u_w[0, 0, 0], 1, 1, 64, 128), T.region(S_acc[h, 0, 0], 3, 1, 64, 128), T.bool(False), T.bool(False), 64, 128, 64, 0, T.bool(False), 0, 0, 0)
                    T.gemm(T.region(Kth_c[0, 0], 1, 64, 64), T.region(u_w[1, 0, 0], 1, 1, 64, 128), T.region(S_acc[h, 0, 0], 3, 1, 64, 128), T.bool(False), T.bool(False), 64, 128, 64, 0, T.bool(False), 0, 0, 0)
                    T.gemm(T.region(Ktl_c[0, 0], 1, 64, 64), T.region(u_w[0, 0, 0], 1, 1, 64, 128), T.region(S_acc[h, 0, 0], 3, 1, 64, 128), T.bool(False), T.bool(False), 64, 128, 64, 0, T.bool(False), 0, 0, 0)
                    T.copy(T.region(S_acc[h, 0, 0], 1, 1, 64, 128), T.region(S_all[c + 1, h * 64, 0], 2, 1, 64, 128))