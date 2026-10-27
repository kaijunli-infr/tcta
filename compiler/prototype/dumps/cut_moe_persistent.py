# E3/F2 output: lower_to_tile_tir(moe_kernel.cut_moe_persistent) at tl.InjectSoftwarePipeline
# from tvm.script import tirx as T

@T.prim_func
def main_f4(x_handle: T.handle, W13q_handle: T.handle, W13s_handle: T.handle, W2q_handle: T.handle, W2s_handle: T.handle, idx_handle: T.handle, dst_handle: T.handle, w_handle: T.handle, Yp_handle: T.handle, bx: T.int32):
    T.func_attr({"target": T.target({"arch": "sm_80", "keys": ["cuda", "gpu"], "kind": "cuda", "max_num_threads": 1024, "tag": "", "thread_warp_size": 32}), "tcta.arch": "sm_80", "tcta.cut": "tl.InjectSoftwarePipeline"})
    x = T.match_buffer(x_handle, (256, 512), "float16", strides=(512, 1))
    Wq = T.match_buffer(W13q_handle, (32, 512, 256), "uint8", strides=(131072, 256, 1))
    Ws = T.match_buffer(W13s_handle, (32, 16, 16), "uint8", strides=(256, 16, 1))
    Wq_1 = T.match_buffer(W2q_handle, (32, 256, 256), "uint8", strides=(65536, 256, 1))
    Ws_1 = T.match_buffer(W2s_handle, (32, 8, 16), "uint8", strides=(128, 16, 1))
    idx = T.match_buffer(idx_handle, (33, 128), "int32", strides=(128, 1))
    dst = T.match_buffer(dst_handle, (32, 128), "int32", strides=(128, 1))
    w = T.match_buffer(w_handle, (32, 128), strides=(128, 1))
    Yp = T.match_buffer(Yp_handle, (1536, 512), strides=(512, 1))
    with T.sblock("root"):
        T.reads()
        T.writes()
        with T.sblock("tilelang_root"):
            T.reads()
            T.writes()
            x_s = T.sblock_alloc_buffer((2, 64, 128), "float16", scope="tcta.inp")
            W_s = T.sblock_alloc_buffer((3, 128, 128), "float16", scope="tcta.wgt")
            Wq_s = T.sblock_alloc_buffer((128, 64), "uint8", scope="tcta.wgt")
            Ws_s = T.sblock_alloc_buffer((4, 4), "uint8", scope="tcta.wgt")
            H = T.sblock_alloc_buffer((4, 64, 128), scope="tcta.acc")
            Hc = T.sblock_alloc_buffer((2, 64, 128), "float16", scope="tcta.acc")
            out = T.sblock_alloc_buffer((2, 64, 128), scope="tcta.acc")
            idx_s = T.sblock_alloc_buffer((64,), "int32", scope="tcta.vec")
            dst_s = T.sblock_alloc_buffer((64,), "int32", scope="tcta.vec")
            w_s = T.sblock_alloc_buffer((64,), scope="tcta.vec")
            w_c = T.sblock_alloc_buffer((64,), scope="tcta.vec")
            idx_a = T.sblock_alloc_buffer((128,), "int32", scope="tcta.vec")
            idx_b = T.sblock_alloc_buffer((128,), "int32", scope="tcta.vec")
            mask_a = T.sblock_alloc_buffer((128,), "int32", scope="tcta.vec")
            mask_b = T.sblock_alloc_buffer((128,), "int32", scope="tcta.vec")
            n_a = T.sblock_alloc_buffer((1,), "int32", scope="tcta.vec")
            n_b = T.sblock_alloc_buffer((1,), "int32", scope="tcta.vec")
            T.copy(T.region(idx[0, 0], 1, 1, 128), T.region(idx_b[0], 2, 128))
            for i in T.parallel(128):
                mask_b[i] = T.if_then_else(0 <= idx_b[i], 1, 0)
            T.reduce(T.region(mask_b[0], 1, 128), T.region(n_a[0], 2, 1), "sum", 0, T.bool(True))
            for ee in range(16):
                T.copy(T.region(idx[ee * 2 + 1, 0], 1, 1, 128), T.region(idx_a[0], 2, 128))
                for i in T.parallel(128):
                    mask_a[i] = T.if_then_else(0 <= idx_a[i], 1, 0)
                T.reduce(T.region(mask_a[0], 1, 128), T.region(n_b[0], 2, 1), "sum", 0, T.bool(True))
                for t in range((n_a[0] + 63) // 64):
                    T.copy(T.region(idx[ee * 2, t * 64], 1, 1, 64), T.region(idx_s[0], 2, 64))
                    T.fill(T.region(H[0, 0, 0], 2, 4, 64, 128), 0)
                    for kk in range(4):
                        for i in T.parallel(64):
                            for j in T.parallel(128):
                                x_s[kk % 2, i, j] = T.if_then_else(0 <= idx_s[i], x[idx_s[i], kk * 128 + j], T.float16(0.0))
                        W_s_1 = T.Buffer((3, 128, 128), "float16", data=W_s.data, scope="tcta.wgt")
                        with T.sblock(""):
                            T.reads(Wq[ee * 2, kk * 128:kk * 128 + 128, 0:64], Wq_s[0:128, 0:64], Ws[ee * 2, kk * 4:kk * 4 + 4, 0:4], Ws_s[0:4, 0:4], Wq[ee * 2, kk * 128:kk * 128 + 128, T.Mul(0, 64):T.Mul(0, 64) + 64], Ws[ee * 2, kk * 4:kk * 4 + 4, T.Mul(0, 4):T.Mul(0, 4) + 4])
                            T.writes(Wq_s[0:128, 0:64], Ws_s[0:4, 0:4], W_s_1[0:3, 0:128, 0:128], Wq_s[0:128, 0:64], Ws_s[0:4, 0:4])
                            with T.sblock(""):
                                T.reads(Wq[ee * 2, kk * 128, T.Mul(0, 64)], Wq_s[0, 0], Wq[ee * 2, kk * 128:kk * 128 + 128, T.Mul(0, 64):T.Mul(0, 64) + 64])
                                T.writes(Wq_s[0:128, 0:64])
                                T.copy(T.region(Wq[ee * 2, kk * 128, T.Mul(0, 64)], 1, 1, 128, 64), T.region(Wq_s[0, 0], 2, 128, 64))
                            with T.sblock(""):
                                T.reads(Ws[ee * 2, kk * 4, T.Mul(0, 4)], Ws_s[0, 0], Ws[ee * 2, kk * 4:kk * 4 + 4, T.Mul(0, 4):T.Mul(0, 4) + 4])
                                T.writes(Ws_s[0:4, 0:4])
                                T.copy(T.region(Ws[ee * 2, kk * 4, T.Mul(0, 4)], 1, 1, 4, 4), T.region(Ws_s[0, 0], 2, 4, 4))
                            T.call_extern("handle", "tcta.fill_issued")
                            T.call_extern("handle", "tcta.fill_wait", 0)
                            with T.sblock(""):
                                T.reads(Wq_s[0:128, 0:64], Ws_s[0:4, 0:4])
                                T.writes(W_s_1[0:3, 0:128, 0:128])
                                for i in T.parallel(128):
                                    for j in T.parallel(128):
                                        byte: T.int32 = T.Cast("int32", Wq_s[i, j // 2])
                                        mag: T.float32 = T.if_then_else(T.bitwise_and(T.bitwise_and(T.shift_right(byte, j % 2 * 4), 15), 7) < 2, T.Cast("float32", T.bitwise_and(T.bitwise_and(T.shift_right(byte, j % 2 * 4), 15), 7)) * T.float32(0.5), T.if_then_else(T.bitwise_and(T.bitwise_and(T.shift_right(byte, j % 2 * 4), 15), 7) < 4, T.float32(1.0) + T.Cast("float32", T.bitwise_and(T.bitwise_and(T.shift_right(byte, j % 2 * 4), 15), 7) - 2) * T.float32(0.5), T.if_then_else(T.bitwise_and(T.bitwise_and(T.shift_right(byte, j % 2 * 4), 15), 7) < 6, T.float32(2.0) + T.Cast("float32", T.bitwise_and(T.bitwise_and(T.shift_right(byte, j % 2 * 4), 15), 7) - 4), T.float32(4.0) + T.Cast("float32", T.bitwise_and(T.bitwise_and(T.shift_right(byte, j % 2 * 4), 15), 7) - 6) * T.float32(2.0))))
                                        scale: T.float32 = T.exp2(T.Cast("float32", Ws_s[i // 32, j // 32]) - T.float32(127.0))
                                        W_s_1[T.FloorMod(0, 3), i, j] = T.Cast("float16", T.if_then_else(T.shift_right(T.bitwise_and(T.shift_right(byte, j % 2 * 4), 15), 3) == 1, mag * T.float32(-1.0), mag) * scale)
                        with T.sblock(""):
                            T.reads(Wq[ee * 2, kk * 128:kk * 128 + 128, 64:128], Wq_s[0:128, 0:64], Ws[ee * 2, kk * 4:kk * 4 + 4, 4:8], Ws_s[0:4, 0:4], Wq[ee * 2, kk * 128:kk * 128 + 128, T.Mul(1, 64):T.Mul(1, 64) + 64], Ws[ee * 2, kk * 4:kk * 4 + 4, T.Mul(1, 4):T.Mul(1, 4) + 4])
                            T.writes(Wq_s[0:128, 0:64], Ws_s[0:4, 0:4], W_s_1[0:3, 0:128, 0:128], Wq_s[0:128, 0:64], Ws_s[0:4, 0:4])
                            with T.sblock(""):
                                T.reads(Wq[ee * 2, kk * 128, T.Mul(1, 64)], Wq_s[0, 0], Wq[ee * 2, kk * 128:kk * 128 + 128, T.Mul(1, 64):T.Mul(1, 64) + 64])
                                T.writes(Wq_s[0:128, 0:64])
                                T.copy(T.region(Wq[ee * 2, kk * 128, T.Mul(1, 64)], 1, 1, 128, 64), T.region(Wq_s[0, 0], 2, 128, 64))
                            with T.sblock(""):
                                T.reads(Ws[ee * 2, kk * 4, T.Mul(1, 4)], Ws_s[0, 0], Ws[ee * 2, kk * 4:kk * 4 + 4, T.Mul(1, 4):T.Mul(1, 4) + 4])
                                T.writes(Ws_s[0:4, 0:4])
                                T.copy(T.region(Ws[ee * 2, kk * 4, T.Mul(1, 4)], 1, 1, 4, 4), T.region(Ws_s[0, 0], 2, 4, 4))
                            T.call_extern("handle", "tcta.fill_issued")
                            T.call_extern("handle", "tcta.fill_wait", 0)
                            with T.sblock(""):
                                T.reads(Wq_s[0:128, 0:64], Ws_s[0:4, 0:4])
                                T.writes(W_s_1[0:3, 0:128, 0:128])
                                for i in T.parallel(128):
                                    for j in T.parallel(128):
                                        byte: T.int32 = T.Cast("int32", Wq_s[i, j // 2])
                                        mag: T.float32 = T.if_then_else(T.bitwise_and(T.bitwise_and(T.shift_right(byte, j % 2 * 4), 15), 7) < 2, T.Cast("float32", T.bitwise_and(T.bitwise_and(T.shift_right(byte, j % 2 * 4), 15), 7)) * T.float32(0.5), T.if_then_else(T.bitwise_and(T.bitwise_and(T.shift_right(byte, j % 2 * 4), 15), 7) < 4, T.float32(1.0) + T.Cast("float32", T.bitwise_and(T.bitwise_and(T.shift_right(byte, j % 2 * 4), 15), 7) - 2) * T.float32(0.5), T.if_then_else(T.bitwise_and(T.bitwise_and(T.shift_right(byte, j % 2 * 4), 15), 7) < 6, T.float32(2.0) + T.Cast("float32", T.bitwise_and(T.bitwise_and(T.shift_right(byte, j % 2 * 4), 15), 7) - 4), T.float32(4.0) + T.Cast("float32", T.bitwise_and(T.bitwise_and(T.shift_right(byte, j % 2 * 4), 15), 7) - 6) * T.float32(2.0))))
                                        scale: T.float32 = T.exp2(T.Cast("float32", Ws_s[i // 32, j // 32]) - T.float32(127.0))
                                        W_s_1[T.FloorMod(1, 3), i, j] = T.Cast("float16", T.if_then_else(T.shift_right(T.bitwise_and(T.shift_right(byte, j % 2 * 4), 15), 3) == 1, mag * T.float32(-1.0), mag) * scale)
                        with T.sblock(""):
                            nn = T.int32()
                            T.reads(Wq[ee * 2, kk * 128:kk * 128 + 128, 128:256], Wq_s[0:128, 0:64], Ws[ee * 2, kk * 4:kk * 4 + 4, 8:16], Ws_s[0:4, 0:4], x_s[kk % 2, 0:64, 0:128], W_s_1[0:3, 0:128, 0:128], H[0:2, 0:64, 0:128], Wq[ee * 2, kk * 128:kk * 128 + 128, (nn + 2) * 64:(nn + 2) * 64 + 64], Ws[ee * 2, kk * 4:kk * 4 + 4, (nn + 2) * 4:(nn + 2) * 4 + 4], x_s[kk % 2, 0:64, 0:128], W_s_1[nn % 3, 0:128, 0:128], H[nn, 0:64, 0:128])
                            T.writes(Wq_s[0:128, 0:64], Ws_s[0:4, 0:4], W_s_1[0:3, 0:128, 0:128], H[0:2, 0:64, 0:128], Wq_s[0:128, 0:64], Ws_s[0:4, 0:4], H[nn, 0:64, 0:128])
                            for nn in T.serial(2, annotations={"tl_pipelined_num_stages": 3}):
                                with T.sblock(""):
                                    T.reads(Wq[ee * 2, kk * 128, (nn + 2) * 64], Wq_s[0, 0], Wq[ee * 2, kk * 128:kk * 128 + 128, (nn + 2) * 64:(nn + 2) * 64 + 64])
                                    T.writes(Wq_s[0:128, 0:64])
                                    T.copy(T.region(Wq[ee * 2, kk * 128, (nn + 2) * 64], 1, 1, 128, 64), T.region(Wq_s[0, 0], 2, 128, 64))
                                with T.sblock(""):
                                    T.reads(Ws[ee * 2, kk * 4, (nn + 2) * 4], Ws_s[0, 0], Ws[ee * 2, kk * 4:kk * 4 + 4, (nn + 2) * 4:(nn + 2) * 4 + 4])
                                    T.writes(Ws_s[0:4, 0:4])
                                    T.copy(T.region(Ws[ee * 2, kk * 4, (nn + 2) * 4], 1, 1, 4, 4), T.region(Ws_s[0, 0], 2, 4, 4))
                                T.call_extern("handle", "tcta.fill_issued")
                                T.call_extern("handle", "tcta.fill_wait", 2)
                                with T.sblock(""):
                                    T.reads(Wq_s[0:128, 0:64], Ws_s[0:4, 0:4])
                                    T.writes(W_s_1[0:3, 0:128, 0:128])
                                    for i in T.parallel(128):
                                        for j in T.parallel(128):
                                            byte: T.int32 = T.Cast("int32", Wq_s[i, j // 2])
                                            mag: T.float32 = T.if_then_else(T.bitwise_and(T.bitwise_and(T.shift_right(byte, j % 2 * 4), 15), 7) < 2, T.Cast("float32", T.bitwise_and(T.bitwise_and(T.shift_right(byte, j % 2 * 4), 15), 7)) * T.float32(0.5), T.if_then_else(T.bitwise_and(T.bitwise_and(T.shift_right(byte, j % 2 * 4), 15), 7) < 4, T.float32(1.0) + T.Cast("float32", T.bitwise_and(T.bitwise_and(T.shift_right(byte, j % 2 * 4), 15), 7) - 2) * T.float32(0.5), T.if_then_else(T.bitwise_and(T.bitwise_and(T.shift_right(byte, j % 2 * 4), 15), 7) < 6, T.float32(2.0) + T.Cast("float32", T.bitwise_and(T.bitwise_and(T.shift_right(byte, j % 2 * 4), 15), 7) - 4), T.float32(4.0) + T.Cast("float32", T.bitwise_and(T.bitwise_and(T.shift_right(byte, j % 2 * 4), 15), 7) - 6) * T.float32(2.0))))
                                            scale: T.float32 = T.exp2(T.Cast("float32", Ws_s[i // 32, j // 32]) - T.float32(127.0))
                                            W_s_1[(nn + 2) % 3, i, j] = T.Cast("float16", T.if_then_else(T.shift_right(T.bitwise_and(T.shift_right(byte, j % 2 * 4), 15), 3) == 1, mag * T.float32(-1.0), mag) * scale)
                                with T.sblock(""):
                                    T.reads(x_s[kk % 2, 0, 0], W_s_1[0:3, 0, 0], H[nn, 0, 0], x_s[kk % 2, 0:64, 0:128], W_s_1[0:3, 0:128, 0:128], H[nn, 0:64, 0:128])
                                    T.writes(H[nn, 0:64, 0:128])
                                    T.gemm(T.region(x_s[kk % 2, 0, 0], 1, 1, 64, 128), T.region(W_s_1[nn % 3, 0, 0], 1, 1, 128, 128), T.region(H[nn, 0, 0], 3, 1, 64, 128), T.bool(False), T.bool(False), 64, 128, 128, 0, T.bool(False), 0, 0, 0)
                        with T.sblock(""):
                            T.reads(x_s[kk % 2, 0, 0], W_s_1[T.FloorMod(2, 3), 0, 0], H[2, 0, 0], x_s[kk % 2, 0:64, 0:128], W_s_1[T.FloorMod(2, 3), 0:128, 0:128], H[2, 0:64, 0:128])
                            T.writes(H[2, 0:64, 0:128])
                            T.gemm(T.region(x_s[kk % 2, 0, 0], 1, 1, 64, 128), T.region(W_s_1[T.FloorMod(2, 3), 0, 0], 1, 1, 128, 128), T.region(H[2, 0, 0], 3, 1, 64, 128), T.bool(False), T.bool(False), 64, 128, 128, 0, T.bool(False), 0, 0, 0)
                        with T.sblock(""):
                            T.reads(x_s[kk % 2, 0, 0], W_s_1[T.FloorMod(3, 3), 0, 0], H[3, 0, 0], x_s[kk % 2, 0:64, 0:128], W_s_1[T.FloorMod(3, 3), 0:128, 0:128], H[3, 0:64, 0:128])
                            T.writes(H[3, 0:64, 0:128])
                            T.gemm(T.region(x_s[kk % 2, 0, 0], 1, 1, 64, 128), T.region(W_s_1[T.FloorMod(3, 3), 0, 0], 1, 1, 128, 128), T.region(H[3, 0, 0], 3, 1, 64, 128), T.bool(False), T.bool(False), 64, 128, 128, 0, T.bool(False), 0, 0, 0)
                    for nn in range(2):
                        for i in T.parallel(64):
                            for j in T.parallel(128):
                                H[nn + 2, i, j] = H[nn + 2, i, j] * H[nn, i, j]
                        for i in T.parallel(64):
                            for j in T.parallel(128):
                                H[nn, i, j] = T.sigmoid(H[nn, i, j])
                        for i in T.parallel(64):
                            for j in T.parallel(128):
                                H[nn + 2, i, j] = H[nn + 2, i, j] * H[nn, i, j]
                        T.copy(T.region(H[nn + 2, 0, 0], 1, 1, 64, 128), T.region(Hc[nn, 0, 0], 2, 1, 64, 128))
                    T.copy(T.region(w[ee * 2, t * 64], 1, 1, 64), T.region(w_s[0], 2, 64))
                    T.copy(T.region(w_s[0], 1, 64), T.region(w_c[0], 2, 64))
                    T.copy(T.region(dst[ee * 2, t * 64], 1, 1, 64), T.region(dst_s[0], 2, 64))
                    W_s_1 = T.Buffer((3, 128, 128), "float16", data=W_s.data, scope="tcta.wgt")
                    with T.sblock(""):
                        T.reads(Wq_1[ee * 2, 0:128, 0:64], Wq_s[0:128, 0:64], Ws_1[ee * 2, 0:4, 0:4], Ws_s[0:4, 0:4], Wq_1[ee * 2, T.FloorMod(0, 2) * 128:T.FloorMod(0, 2) * 128 + 128, T.FloorDiv(0, 2) * 64:T.FloorDiv(0, 2) * 64 + 64], Ws_1[ee * 2, T.FloorMod(0, 2) * 4:T.FloorMod(0, 2) * 4 + 4, T.FloorDiv(0, 2) * 4:T.FloorDiv(0, 2) * 4 + 4])
                        T.writes(Wq_s[0:128, 0:64], Ws_s[0:4, 0:4], W_s_1[0:3, 0:128, 0:128], Wq_s[0:128, 0:64], Ws_s[0:4, 0:4])
                        with T.sblock(""):
                            T.reads(Wq_1[ee * 2, T.FloorMod(0, 2) * 128, T.FloorDiv(0, 2) * 64], Wq_s[0, 0], Wq_1[ee * 2, T.FloorMod(0, 2) * 128:T.FloorMod(0, 2) * 128 + 128, T.FloorDiv(0, 2) * 64:T.FloorDiv(0, 2) * 64 + 64])
                            T.writes(Wq_s[0:128, 0:64])
                            T.copy(T.region(Wq_1[ee * 2, T.FloorMod(0, 2) * 128, T.FloorDiv(0, 2) * 64], 1, 1, 128, 64), T.region(Wq_s[0, 0], 2, 128, 64))
                        with T.sblock(""):
                            T.reads(Ws_1[ee * 2, T.FloorMod(0, 2) * 4, T.FloorDiv(0, 2) * 4], Ws_s[0, 0], Ws_1[ee * 2, T.FloorMod(0, 2) * 4:T.FloorMod(0, 2) * 4 + 4, T.FloorDiv(0, 2) * 4:T.FloorDiv(0, 2) * 4 + 4])
                            T.writes(Ws_s[0:4, 0:4])
                            T.copy(T.region(Ws_1[ee * 2, T.FloorMod(0, 2) * 4, T.FloorDiv(0, 2) * 4], 1, 1, 4, 4), T.region(Ws_s[0, 0], 2, 4, 4))
                        T.call_extern("handle", "tcta.fill_issued")
                        T.call_extern("handle", "tcta.fill_wait", 0)
                        with T.sblock(""):
                            T.reads(Wq_s[0:128, 0:64], Ws_s[0:4, 0:4])
                            T.writes(W_s_1[0:3, 0:128, 0:128])
                            for i in T.parallel(128):
                                for j in T.parallel(128):
                                    byte: T.int32 = T.Cast("int32", Wq_s[i, j // 2])
                                    mag: T.float32 = T.if_then_else(T.bitwise_and(T.bitwise_and(T.shift_right(byte, j % 2 * 4), 15), 7) < 2, T.Cast("float32", T.bitwise_and(T.bitwise_and(T.shift_right(byte, j % 2 * 4), 15), 7)) * T.float32(0.5), T.if_then_else(T.bitwise_and(T.bitwise_and(T.shift_right(byte, j % 2 * 4), 15), 7) < 4, T.float32(1.0) + T.Cast("float32", T.bitwise_and(T.bitwise_and(T.shift_right(byte, j % 2 * 4), 15), 7) - 2) * T.float32(0.5), T.if_then_else(T.bitwise_and(T.bitwise_and(T.shift_right(byte, j % 2 * 4), 15), 7) < 6, T.float32(2.0) + T.Cast("float32", T.bitwise_and(T.bitwise_and(T.shift_right(byte, j % 2 * 4), 15), 7) - 4), T.float32(4.0) + T.Cast("float32", T.bitwise_and(T.bitwise_and(T.shift_right(byte, j % 2 * 4), 15), 7) - 6) * T.float32(2.0))))
                                    scale: T.float32 = T.exp2(T.Cast("float32", Ws_s[i // 32, j // 32]) - T.float32(127.0))
                                    W_s_1[T.FloorMod(0, 3), i, j] = T.Cast("float16", T.if_then_else(T.shift_right(T.bitwise_and(T.shift_right(byte, j % 2 * 4), 15), 3) == 1, mag * T.float32(-1.0), mag) * scale)
                    with T.sblock(""):
                        T.reads(Wq_1[ee * 2, 128:256, 0:64], Wq_s[0:128, 0:64], Ws_1[ee * 2, 4:8, 0:4], Ws_s[0:4, 0:4], Wq_1[ee * 2, T.FloorMod(1, 2) * 128:T.FloorMod(1, 2) * 128 + 128, T.FloorDiv(1, 2) * 64:T.FloorDiv(1, 2) * 64 + 64], Ws_1[ee * 2, T.FloorMod(1, 2) * 4:T.FloorMod(1, 2) * 4 + 4, T.FloorDiv(1, 2) * 4:T.FloorDiv(1, 2) * 4 + 4])
                        T.writes(Wq_s[0:128, 0:64], Ws_s[0:4, 0:4], W_s_1[0:3, 0:128, 0:128], Wq_s[0:128, 0:64], Ws_s[0:4, 0:4])
                        with T.sblock(""):
                            T.reads(Wq_1[ee * 2, T.FloorMod(1, 2) * 128, T.FloorDiv(1, 2) * 64], Wq_s[0, 0], Wq_1[ee * 2, T.FloorMod(1, 2) * 128:T.FloorMod(1, 2) * 128 + 128, T.FloorDiv(1, 2) * 64:T.FloorDiv(1, 2) * 64 + 64])
                            T.writes(Wq_s[0:128, 0:64])
                            T.copy(T.region(Wq_1[ee * 2, T.FloorMod(1, 2) * 128, T.FloorDiv(1, 2) * 64], 1, 1, 128, 64), T.region(Wq_s[0, 0], 2, 128, 64))
                        with T.sblock(""):
                            T.reads(Ws_1[ee * 2, T.FloorMod(1, 2) * 4, T.FloorDiv(1, 2) * 4], Ws_s[0, 0], Ws_1[ee * 2, T.FloorMod(1, 2) * 4:T.FloorMod(1, 2) * 4 + 4, T.FloorDiv(1, 2) * 4:T.FloorDiv(1, 2) * 4 + 4])
                            T.writes(Ws_s[0:4, 0:4])
                            T.copy(T.region(Ws_1[ee * 2, T.FloorMod(1, 2) * 4, T.FloorDiv(1, 2) * 4], 1, 1, 4, 4), T.region(Ws_s[0, 0], 2, 4, 4))
                        T.call_extern("handle", "tcta.fill_issued")
                        T.call_extern("handle", "tcta.fill_wait", 0)
                        with T.sblock(""):
                            T.reads(Wq_s[0:128, 0:64], Ws_s[0:4, 0:4])
                            T.writes(W_s_1[0:3, 0:128, 0:128])
                            for i in T.parallel(128):
                                for j in T.parallel(128):
                                    byte: T.int32 = T.Cast("int32", Wq_s[i, j // 2])
                                    mag: T.float32 = T.if_then_else(T.bitwise_and(T.bitwise_and(T.shift_right(byte, j % 2 * 4), 15), 7) < 2, T.Cast("float32", T.bitwise_and(T.bitwise_and(T.shift_right(byte, j % 2 * 4), 15), 7)) * T.float32(0.5), T.if_then_else(T.bitwise_and(T.bitwise_and(T.shift_right(byte, j % 2 * 4), 15), 7) < 4, T.float32(1.0) + T.Cast("float32", T.bitwise_and(T.bitwise_and(T.shift_right(byte, j % 2 * 4), 15), 7) - 2) * T.float32(0.5), T.if_then_else(T.bitwise_and(T.bitwise_and(T.shift_right(byte, j % 2 * 4), 15), 7) < 6, T.float32(2.0) + T.Cast("float32", T.bitwise_and(T.bitwise_and(T.shift_right(byte, j % 2 * 4), 15), 7) - 4), T.float32(4.0) + T.Cast("float32", T.bitwise_and(T.bitwise_and(T.shift_right(byte, j % 2 * 4), 15), 7) - 6) * T.float32(2.0))))
                                    scale: T.float32 = T.exp2(T.Cast("float32", Ws_s[i // 32, j // 32]) - T.float32(127.0))
                                    W_s_1[T.FloorMod(1, 3), i, j] = T.Cast("float16", T.if_then_else(T.shift_right(T.bitwise_and(T.shift_right(byte, j % 2 * 4), 15), 3) == 1, mag * T.float32(-1.0), mag) * scale)
                    with T.sblock(""):
                        m = T.int32()
                        T.reads(out[0:2, 0:64, 0:128], Wq_1[ee * 2, 0:256, 64:256], Wq_s[0:128, 0:64], Ws_1[ee * 2, 0:8, 4:16], Ws_s[0:4, 0:4], Hc[0:2, 0:64, 0:128], W_s_1[0:3, 0:128, 0:128], w_c[0:64], dst_s[0:64], Wq_1[ee * 2, (m + 2) % 2 * 128:(m + 2) % 2 * 128 + 128, (m + 2) // 2 * 64:(m + 2) // 2 * 64 + 64], Ws_1[ee * 2, (m + 2) % 2 * 4:(m + 2) % 2 * 4 + 4, (m + 2) // 2 * 4:(m + 2) // 2 * 4 + 4], Hc[m % 2, 0:64, 0:128], W_s_1[m % 3, 0:128, 0:128], out[m % 4 // 2, 0:64, 0:128])
                        T.writes(out[0:2, 0:64, 0:128], Wq_s[0:128, 0:64], Ws_s[0:4, 0:4], W_s_1[0:3, 0:128, 0:128], Yp[0:1536, 0:384], out[m % 4 // 2, 0:64, 0:128], Wq_s[0:128, 0:64], Ws_s[0:4, 0:4], out[m % 4 // 2, 0:64, 0:128])
                        for m in T.serial(6, annotations={"tl_pipelined_num_stages": 3}):
                            with T.sblock(""):
                                T.reads(out[m % 4 // 2, 0, 0])
                                T.writes(out[m % 4 // 2, 0:64, 0:128])
                                if m % 2 == 0:
                                    T.fill(T.region(out[m % 4 // 2, 0, 0], 2, 1, 64, 128), 0)
                            with T.sblock(""):
                                T.reads(Wq_1[ee * 2, (m + 2) % 2 * 128, (m + 2) // 2 * 64], Wq_s[0, 0], Wq_1[ee * 2, (m + 2) % 2 * 128:(m + 2) % 2 * 128 + 128, (m + 2) // 2 * 64:(m + 2) // 2 * 64 + 64])
                                T.writes(Wq_s[0:128, 0:64])
                                T.copy(T.region(Wq_1[ee * 2, (m + 2) % 2 * 128, (m + 2) // 2 * 64], 1, 1, 128, 64), T.region(Wq_s[0, 0], 2, 128, 64))
                            with T.sblock(""):
                                T.reads(Ws_1[ee * 2, (m + 2) % 2 * 4, (m + 2) // 2 * 4], Ws_s[0, 0], Ws_1[ee * 2, (m + 2) % 2 * 4:(m + 2) % 2 * 4 + 4, (m + 2) // 2 * 4:(m + 2) // 2 * 4 + 4])
                                T.writes(Ws_s[0:4, 0:4])
                                T.copy(T.region(Ws_1[ee * 2, (m + 2) % 2 * 4, (m + 2) // 2 * 4], 1, 1, 4, 4), T.region(Ws_s[0, 0], 2, 4, 4))
                            T.call_extern("handle", "tcta.fill_issued")
                            T.call_extern("handle", "tcta.fill_wait", 2)
                            with T.sblock(""):
                                T.reads(Wq_s[0:128, 0:64], Ws_s[0:4, 0:4])
                                T.writes(W_s_1[0:3, 0:128, 0:128])
                                for i in T.parallel(128):
                                    for j in T.parallel(128):
                                        byte: T.int32 = T.Cast("int32", Wq_s[i, j // 2])
                                        mag: T.float32 = T.if_then_else(T.bitwise_and(T.bitwise_and(T.shift_right(byte, j % 2 * 4), 15), 7) < 2, T.Cast("float32", T.bitwise_and(T.bitwise_and(T.shift_right(byte, j % 2 * 4), 15), 7)) * T.float32(0.5), T.if_then_else(T.bitwise_and(T.bitwise_and(T.shift_right(byte, j % 2 * 4), 15), 7) < 4, T.float32(1.0) + T.Cast("float32", T.bitwise_and(T.bitwise_and(T.shift_right(byte, j % 2 * 4), 15), 7) - 2) * T.float32(0.5), T.if_then_else(T.bitwise_and(T.bitwise_and(T.shift_right(byte, j % 2 * 4), 15), 7) < 6, T.float32(2.0) + T.Cast("float32", T.bitwise_and(T.bitwise_and(T.shift_right(byte, j % 2 * 4), 15), 7) - 4), T.float32(4.0) + T.Cast("float32", T.bitwise_and(T.bitwise_and(T.shift_right(byte, j % 2 * 4), 15), 7) - 6) * T.float32(2.0))))
                                        scale: T.float32 = T.exp2(T.Cast("float32", Ws_s[i // 32, j // 32]) - T.float32(127.0))
                                        W_s_1[(m + 2) % 3, i, j] = T.Cast("float16", T.if_then_else(T.shift_right(T.bitwise_and(T.shift_right(byte, j % 2 * 4), 15), 3) == 1, mag * T.float32(-1.0), mag) * scale)
                            with T.sblock(""):
                                T.reads(Hc[m % 2, 0, 0], W_s_1[0:3, 0, 0], out[m % 4 // 2, 0, 0], Hc[m % 2, 0:64, 0:128], W_s_1[0:3, 0:128, 0:128], out[m % 4 // 2, 0:64, 0:128])
                                T.writes(out[m % 4 // 2, 0:64, 0:128])
                                T.gemm(T.region(Hc[m % 2, 0, 0], 1, 1, 64, 128), T.region(W_s_1[m % 3, 0, 0], 1, 1, 128, 128), T.region(out[m % 4 // 2, 0, 0], 3, 1, 64, 128), T.bool(False), T.bool(False), 64, 128, 128, 0, T.bool(False), 0, 0, 0)
                            with T.sblock(""):
                                T.reads(out[m % 4 // 2, 0:64, 0:128], w_c[0:64])
                                T.writes(out[m % 4 // 2, 0:64, 0:128])
                                if m % 2 == 1:
                                    for i in T.parallel(64):
                                        for j in T.parallel(128):
                                            out[m % 4 // 2, i, j] = out[m % 4 // 2, i, j] * w_c[i]
                            with T.sblock(""):
                                T.reads(dst_s[0:64], out[m % 4 // 2, 0:64, 0:128])
                                T.writes(Yp[0:1536, m // 2 * 128:m // 2 * 128 + 128])
                                if m % 2 == 1:
                                    for i in T.parallel(64):
                                        for j in T.parallel(128):
                                            if 0 <= dst_s[i]:
                                                Yp[dst_s[i], m // 2 * 128 + j] = out[m % 4 // 2, i, j]
                    with T.sblock(""):
                        T.reads(out[1, 0:64, 0:128], Hc[0, 0:64, 0:128], W_s_1[0:3, 0:128, 0:128], w_c[0:64], dst_s[0:64], Hc[T.FloorMod(6, 2), 0:64, 0:128], W_s_1[T.FloorMod(6, 3), 0:128, 0:128], out[T.FloorMod(6, 4) // 2, 0:64, 0:128])
                        T.writes(out[1, 0:64, 0:128], Yp[0:1536, 384:512], out[T.FloorMod(6, 4) // 2, 0:64, 0:128], out[T.FloorMod(6, 4) // 2, 0:64, 0:128])
                        with T.sblock(""):
                            T.reads(out[T.FloorMod(6, 4) // 2, 0, 0])
                            T.writes(out[T.FloorMod(6, 4) // 2, 0:64, 0:128])
                            if T.FloorMod(6, 2) == 0:
                                T.fill(T.region(out[T.FloorMod(6, 4) // 2, 0, 0], 2, 1, 64, 128), 0)
                        with T.sblock(""):
                            T.reads(Hc[T.FloorMod(6, 2), 0, 0], W_s_1[0:3, 0, 0], out[T.FloorMod(6, 4) // 2, 0, 0], Hc[T.FloorMod(6, 2), 0:64, 0:128], W_s_1[0:3, 0:128, 0:128], out[T.FloorMod(6, 4) // 2, 0:64, 0:128])
                            T.writes(out[T.FloorMod(6, 4) // 2, 0:64, 0:128])
                            T.gemm(T.region(Hc[T.FloorMod(6, 2), 0, 0], 1, 1, 64, 128), T.region(W_s_1[T.FloorMod(6, 3), 0, 0], 1, 1, 128, 128), T.region(out[T.FloorMod(6, 4) // 2, 0, 0], 3, 1, 64, 128), T.bool(False), T.bool(False), 64, 128, 128, 0, T.bool(False), 0, 0, 0)
                        with T.sblock(""):
                            T.reads(out[T.FloorMod(6, 4) // 2, 0:64, 0:128], w_c[0:64])
                            T.writes(out[T.FloorMod(6, 4) // 2, 0:64, 0:128])
                            if T.FloorMod(6, 2) == 1:
                                for i in T.parallel(64):
                                    for j in T.parallel(128):
                                        out[T.FloorMod(6, 4) // 2, i, j] = out[T.FloorMod(6, 4) // 2, i, j] * w_c[i]
                        with T.sblock(""):
                            T.reads(dst_s[0:64], out[T.FloorMod(6, 4) // 2, 0:64, 0:128])
                            T.writes(Yp[0:1536, T.FloorDiv(6, 2) * 128:T.FloorDiv(6, 2) * 128 + 128])
                            if T.FloorMod(6, 2) == 1:
                                for i in T.parallel(64):
                                    for j in T.parallel(128):
                                        if 0 <= dst_s[i]:
                                            Yp[dst_s[i], T.FloorDiv(6, 2) * 128 + j] = out[T.FloorMod(6, 4) // 2, i, j]
                    with T.sblock(""):
                        T.reads(out[1, 0:64, 0:128], Hc[1, 0:64, 0:128], W_s_1[0:3, 0:128, 0:128], w_c[0:64], dst_s[0:64], Hc[T.FloorMod(7, 2), 0:64, 0:128], W_s_1[T.FloorMod(7, 3), 0:128, 0:128], out[T.FloorMod(7, 4) // 2, 0:64, 0:128])
                        T.writes(out[1, 0:64, 0:128], Yp[0:1536, 384:512], out[T.FloorMod(7, 4) // 2, 0:64, 0:128], out[T.FloorMod(7, 4) // 2, 0:64, 0:128])
                        with T.sblock(""):
                            T.reads(out[T.FloorMod(7, 4) // 2, 0, 0])
                            T.writes(out[T.FloorMod(7, 4) // 2, 0:64, 0:128])
                            if T.FloorMod(7, 2) == 0:
                                T.fill(T.region(out[T.FloorMod(7, 4) // 2, 0, 0], 2, 1, 64, 128), 0)
                        with T.sblock(""):
                            T.reads(Hc[T.FloorMod(7, 2), 0, 0], W_s_1[0:3, 0, 0], out[T.FloorMod(7, 4) // 2, 0, 0], Hc[T.FloorMod(7, 2), 0:64, 0:128], W_s_1[0:3, 0:128, 0:128], out[T.FloorMod(7, 4) // 2, 0:64, 0:128])
                            T.writes(out[T.FloorMod(7, 4) // 2, 0:64, 0:128])
                            T.gemm(T.region(Hc[T.FloorMod(7, 2), 0, 0], 1, 1, 64, 128), T.region(W_s_1[T.FloorMod(7, 3), 0, 0], 1, 1, 128, 128), T.region(out[T.FloorMod(7, 4) // 2, 0, 0], 3, 1, 64, 128), T.bool(False), T.bool(False), 64, 128, 128, 0, T.bool(False), 0, 0, 0)
                        with T.sblock(""):
                            T.reads(out[T.FloorMod(7, 4) // 2, 0:64, 0:128], w_c[0:64])
                            T.writes(out[T.FloorMod(7, 4) // 2, 0:64, 0:128])
                            if T.FloorMod(7, 2) == 1:
                                for i in T.parallel(64):
                                    for j in T.parallel(128):
                                        out[T.FloorMod(7, 4) // 2, i, j] = out[T.FloorMod(7, 4) // 2, i, j] * w_c[i]
                        with T.sblock(""):
                            T.reads(dst_s[0:64], out[T.FloorMod(7, 4) // 2, 0:64, 0:128])
                            T.writes(Yp[0:1536, T.FloorDiv(7, 2) * 128:T.FloorDiv(7, 2) * 128 + 128])
                            if T.FloorMod(7, 2) == 1:
                                for i in T.parallel(64):
                                    for j in T.parallel(128):
                                        if 0 <= dst_s[i]:
                                            Yp[dst_s[i], T.FloorDiv(7, 2) * 128 + j] = out[T.FloorMod(7, 4) // 2, i, j]
                T.copy(T.region(idx[ee * 2 + 2, 0], 1, 1, 128), T.region(idx_b[0], 2, 128))
                for i in T.parallel(128):
                    mask_b[i] = T.if_then_else(0 <= idx_b[i], 1, 0)
                T.reduce(T.region(mask_b[0], 1, 128), T.region(n_a[0], 2, 1), "sum", 0, T.bool(True))
                for t in range((n_b[0] + 63) // 64):
                    T.copy(T.region(idx[ee * 2 + 1, t * 64], 1, 1, 64), T.region(idx_s[0], 2, 64))
                    T.fill(T.region(H[0, 0, 0], 2, 4, 64, 128), 0)
                    for kk in range(4):
                        for i in T.parallel(64):
                            for j in T.parallel(128):
                                x_s[kk % 2, i, j] = T.if_then_else(0 <= idx_s[i], x[idx_s[i], kk * 128 + j], T.float16(0.0))
                        W_s_1 = T.Buffer((3, 128, 128), "float16", data=W_s.data, scope="tcta.wgt")
                        with T.sblock(""):
                            T.reads(Wq[ee * 2 + 1, kk * 128:kk * 128 + 128, 0:64], Wq_s[0:128, 0:64], Ws[ee * 2 + 1, kk * 4:kk * 4 + 4, 0:4], Ws_s[0:4, 0:4], Wq[ee * 2 + 1, kk * 128:kk * 128 + 128, T.Mul(0, 64):T.Mul(0, 64) + 64], Ws[ee * 2 + 1, kk * 4:kk * 4 + 4, T.Mul(0, 4):T.Mul(0, 4) + 4])
                            T.writes(Wq_s[0:128, 0:64], Ws_s[0:4, 0:4], W_s_1[0:3, 0:128, 0:128], Wq_s[0:128, 0:64], Ws_s[0:4, 0:4])
                            with T.sblock(""):
                                T.reads(Wq[ee * 2 + 1, kk * 128, T.Mul(0, 64)], Wq_s[0, 0], Wq[ee * 2 + 1, kk * 128:kk * 128 + 128, T.Mul(0, 64):T.Mul(0, 64) + 64])
                                T.writes(Wq_s[0:128, 0:64])
                                T.copy(T.region(Wq[ee * 2 + 1, kk * 128, T.Mul(0, 64)], 1, 1, 128, 64), T.region(Wq_s[0, 0], 2, 128, 64))
                            with T.sblock(""):
                                T.reads(Ws[ee * 2 + 1, kk * 4, T.Mul(0, 4)], Ws_s[0, 0], Ws[ee * 2 + 1, kk * 4:kk * 4 + 4, T.Mul(0, 4):T.Mul(0, 4) + 4])
                                T.writes(Ws_s[0:4, 0:4])
                                T.copy(T.region(Ws[ee * 2 + 1, kk * 4, T.Mul(0, 4)], 1, 1, 4, 4), T.region(Ws_s[0, 0], 2, 4, 4))
                            T.call_extern("handle", "tcta.fill_issued")
                            T.call_extern("handle", "tcta.fill_wait", 0)
                            with T.sblock(""):
                                T.reads(Wq_s[0:128, 0:64], Ws_s[0:4, 0:4])
                                T.writes(W_s_1[0:3, 0:128, 0:128])
                                for i in T.parallel(128):
                                    for j in T.parallel(128):
                                        byte: T.int32 = T.Cast("int32", Wq_s[i, j // 2])
                                        mag: T.float32 = T.if_then_else(T.bitwise_and(T.bitwise_and(T.shift_right(byte, j % 2 * 4), 15), 7) < 2, T.Cast("float32", T.bitwise_and(T.bitwise_and(T.shift_right(byte, j % 2 * 4), 15), 7)) * T.float32(0.5), T.if_then_else(T.bitwise_and(T.bitwise_and(T.shift_right(byte, j % 2 * 4), 15), 7) < 4, T.float32(1.0) + T.Cast("float32", T.bitwise_and(T.bitwise_and(T.shift_right(byte, j % 2 * 4), 15), 7) - 2) * T.float32(0.5), T.if_then_else(T.bitwise_and(T.bitwise_and(T.shift_right(byte, j % 2 * 4), 15), 7) < 6, T.float32(2.0) + T.Cast("float32", T.bitwise_and(T.bitwise_and(T.shift_right(byte, j % 2 * 4), 15), 7) - 4), T.float32(4.0) + T.Cast("float32", T.bitwise_and(T.bitwise_and(T.shift_right(byte, j % 2 * 4), 15), 7) - 6) * T.float32(2.0))))
                                        scale: T.float32 = T.exp2(T.Cast("float32", Ws_s[i // 32, j // 32]) - T.float32(127.0))
                                        W_s_1[T.FloorMod(0, 3), i, j] = T.Cast("float16", T.if_then_else(T.shift_right(T.bitwise_and(T.shift_right(byte, j % 2 * 4), 15), 3) == 1, mag * T.float32(-1.0), mag) * scale)
                        with T.sblock(""):
                            T.reads(Wq[ee * 2 + 1, kk * 128:kk * 128 + 128, 64:128], Wq_s[0:128, 0:64], Ws[ee * 2 + 1, kk * 4:kk * 4 + 4, 4:8], Ws_s[0:4, 0:4], Wq[ee * 2 + 1, kk * 128:kk * 128 + 128, T.Mul(1, 64):T.Mul(1, 64) + 64], Ws[ee * 2 + 1, kk * 4:kk * 4 + 4, T.Mul(1, 4):T.Mul(1, 4) + 4])
                            T.writes(Wq_s[0:128, 0:64], Ws_s[0:4, 0:4], W_s_1[0:3, 0:128, 0:128], Wq_s[0:128, 0:64], Ws_s[0:4, 0:4])
                            with T.sblock(""):
                                T.reads(Wq[ee * 2 + 1, kk * 128, T.Mul(1, 64)], Wq_s[0, 0], Wq[ee * 2 + 1, kk * 128:kk * 128 + 128, T.Mul(1, 64):T.Mul(1, 64) + 64])
                                T.writes(Wq_s[0:128, 0:64])
                                T.copy(T.region(Wq[ee * 2 + 1, kk * 128, T.Mul(1, 64)], 1, 1, 128, 64), T.region(Wq_s[0, 0], 2, 128, 64))
                            with T.sblock(""):
                                T.reads(Ws[ee * 2 + 1, kk * 4, T.Mul(1, 4)], Ws_s[0, 0], Ws[ee * 2 + 1, kk * 4:kk * 4 + 4, T.Mul(1, 4):T.Mul(1, 4) + 4])
                                T.writes(Ws_s[0:4, 0:4])
                                T.copy(T.region(Ws[ee * 2 + 1, kk * 4, T.Mul(1, 4)], 1, 1, 4, 4), T.region(Ws_s[0, 0], 2, 4, 4))
                            T.call_extern("handle", "tcta.fill_issued")
                            T.call_extern("handle", "tcta.fill_wait", 0)
                            with T.sblock(""):
                                T.reads(Wq_s[0:128, 0:64], Ws_s[0:4, 0:4])
                                T.writes(W_s_1[0:3, 0:128, 0:128])
                                for i in T.parallel(128):
                                    for j in T.parallel(128):
                                        byte: T.int32 = T.Cast("int32", Wq_s[i, j // 2])
                                        mag: T.float32 = T.if_then_else(T.bitwise_and(T.bitwise_and(T.shift_right(byte, j % 2 * 4), 15), 7) < 2, T.Cast("float32", T.bitwise_and(T.bitwise_and(T.shift_right(byte, j % 2 * 4), 15), 7)) * T.float32(0.5), T.if_then_else(T.bitwise_and(T.bitwise_and(T.shift_right(byte, j % 2 * 4), 15), 7) < 4, T.float32(1.0) + T.Cast("float32", T.bitwise_and(T.bitwise_and(T.shift_right(byte, j % 2 * 4), 15), 7) - 2) * T.float32(0.5), T.if_then_else(T.bitwise_and(T.bitwise_and(T.shift_right(byte, j % 2 * 4), 15), 7) < 6, T.float32(2.0) + T.Cast("float32", T.bitwise_and(T.bitwise_and(T.shift_right(byte, j % 2 * 4), 15), 7) - 4), T.float32(4.0) + T.Cast("float32", T.bitwise_and(T.bitwise_and(T.shift_right(byte, j % 2 * 4), 15), 7) - 6) * T.float32(2.0))))
                                        scale: T.float32 = T.exp2(T.Cast("float32", Ws_s[i // 32, j // 32]) - T.float32(127.0))
                                        W_s_1[T.FloorMod(1, 3), i, j] = T.Cast("float16", T.if_then_else(T.shift_right(T.bitwise_and(T.shift_right(byte, j % 2 * 4), 15), 3) == 1, mag * T.float32(-1.0), mag) * scale)
                        with T.sblock(""):
                            nn = T.int32()
                            T.reads(Wq[ee * 2 + 1, kk * 128:kk * 128 + 128, 128:256], Wq_s[0:128, 0:64], Ws[ee * 2 + 1, kk * 4:kk * 4 + 4, 8:16], Ws_s[0:4, 0:4], x_s[kk % 2, 0:64, 0:128], W_s_1[0:3, 0:128, 0:128], H[0:2, 0:64, 0:128], Wq[ee * 2 + 1, kk * 128:kk * 128 + 128, (nn + 2) * 64:(nn + 2) * 64 + 64], Ws[ee * 2 + 1, kk * 4:kk * 4 + 4, (nn + 2) * 4:(nn + 2) * 4 + 4], x_s[kk % 2, 0:64, 0:128], W_s_1[nn % 3, 0:128, 0:128], H[nn, 0:64, 0:128])
                            T.writes(Wq_s[0:128, 0:64], Ws_s[0:4, 0:4], W_s_1[0:3, 0:128, 0:128], H[0:2, 0:64, 0:128], Wq_s[0:128, 0:64], Ws_s[0:4, 0:4], H[nn, 0:64, 0:128])
                            for nn in T.serial(2, annotations={"tl_pipelined_num_stages": 3}):
                                with T.sblock(""):
                                    T.reads(Wq[ee * 2 + 1, kk * 128, (nn + 2) * 64], Wq_s[0, 0], Wq[ee * 2 + 1, kk * 128:kk * 128 + 128, (nn + 2) * 64:(nn + 2) * 64 + 64])
                                    T.writes(Wq_s[0:128, 0:64])
                                    T.copy(T.region(Wq[ee * 2 + 1, kk * 128, (nn + 2) * 64], 1, 1, 128, 64), T.region(Wq_s[0, 0], 2, 128, 64))
                                with T.sblock(""):
                                    T.reads(Ws[ee * 2 + 1, kk * 4, (nn + 2) * 4], Ws_s[0, 0], Ws[ee * 2 + 1, kk * 4:kk * 4 + 4, (nn + 2) * 4:(nn + 2) * 4 + 4])
                                    T.writes(Ws_s[0:4, 0:4])
                                    T.copy(T.region(Ws[ee * 2 + 1, kk * 4, (nn + 2) * 4], 1, 1, 4, 4), T.region(Ws_s[0, 0], 2, 4, 4))
                                T.call_extern("handle", "tcta.fill_issued")
                                T.call_extern("handle", "tcta.fill_wait", 2)
                                with T.sblock(""):
                                    T.reads(Wq_s[0:128, 0:64], Ws_s[0:4, 0:4])
                                    T.writes(W_s_1[0:3, 0:128, 0:128])
                                    for i in T.parallel(128):
                                        for j in T.parallel(128):
                                            byte: T.int32 = T.Cast("int32", Wq_s[i, j // 2])
                                            mag: T.float32 = T.if_then_else(T.bitwise_and(T.bitwise_and(T.shift_right(byte, j % 2 * 4), 15), 7) < 2, T.Cast("float32", T.bitwise_and(T.bitwise_and(T.shift_right(byte, j % 2 * 4), 15), 7)) * T.float32(0.5), T.if_then_else(T.bitwise_and(T.bitwise_and(T.shift_right(byte, j % 2 * 4), 15), 7) < 4, T.float32(1.0) + T.Cast("float32", T.bitwise_and(T.bitwise_and(T.shift_right(byte, j % 2 * 4), 15), 7) - 2) * T.float32(0.5), T.if_then_else(T.bitwise_and(T.bitwise_and(T.shift_right(byte, j % 2 * 4), 15), 7) < 6, T.float32(2.0) + T.Cast("float32", T.bitwise_and(T.bitwise_and(T.shift_right(byte, j % 2 * 4), 15), 7) - 4), T.float32(4.0) + T.Cast("float32", T.bitwise_and(T.bitwise_and(T.shift_right(byte, j % 2 * 4), 15), 7) - 6) * T.float32(2.0))))
                                            scale: T.float32 = T.exp2(T.Cast("float32", Ws_s[i // 32, j // 32]) - T.float32(127.0))
                                            W_s_1[(nn + 2) % 3, i, j] = T.Cast("float16", T.if_then_else(T.shift_right(T.bitwise_and(T.shift_right(byte, j % 2 * 4), 15), 3) == 1, mag * T.float32(-1.0), mag) * scale)
                                with T.sblock(""):
                                    T.reads(x_s[kk % 2, 0, 0], W_s_1[0:3, 0, 0], H[nn, 0, 0], x_s[kk % 2, 0:64, 0:128], W_s_1[0:3, 0:128, 0:128], H[nn, 0:64, 0:128])
                                    T.writes(H[nn, 0:64, 0:128])
                                    T.gemm(T.region(x_s[kk % 2, 0, 0], 1, 1, 64, 128), T.region(W_s_1[nn % 3, 0, 0], 1, 1, 128, 128), T.region(H[nn, 0, 0], 3, 1, 64, 128), T.bool(False), T.bool(False), 64, 128, 128, 0, T.bool(False), 0, 0, 0)
                        with T.sblock(""):
                            T.reads(x_s[kk % 2, 0, 0], W_s_1[T.FloorMod(2, 3), 0, 0], H[2, 0, 0], x_s[kk % 2, 0:64, 0:128], W_s_1[T.FloorMod(2, 3), 0:128, 0:128], H[2, 0:64, 0:128])
                            T.writes(H[2, 0:64, 0:128])
                            T.gemm(T.region(x_s[kk % 2, 0, 0], 1, 1, 64, 128), T.region(W_s_1[T.FloorMod(2, 3), 0, 0], 1, 1, 128, 128), T.region(H[2, 0, 0], 3, 1, 64, 128), T.bool(False), T.bool(False), 64, 128, 128, 0, T.bool(False), 0, 0, 0)
                        with T.sblock(""):
                            T.reads(x_s[kk % 2, 0, 0], W_s_1[T.FloorMod(3, 3), 0, 0], H[3, 0, 0], x_s[kk % 2, 0:64, 0:128], W_s_1[T.FloorMod(3, 3), 0:128, 0:128], H[3, 0:64, 0:128])
                            T.writes(H[3, 0:64, 0:128])
                            T.gemm(T.region(x_s[kk % 2, 0, 0], 1, 1, 64, 128), T.region(W_s_1[T.FloorMod(3, 3), 0, 0], 1, 1, 128, 128), T.region(H[3, 0, 0], 3, 1, 64, 128), T.bool(False), T.bool(False), 64, 128, 128, 0, T.bool(False), 0, 0, 0)
                    for nn in range(2):
                        for i in T.parallel(64):
                            for j in T.parallel(128):
                                H[nn + 2, i, j] = H[nn + 2, i, j] * H[nn, i, j]
                        for i in T.parallel(64):
                            for j in T.parallel(128):
                                H[nn, i, j] = T.sigmoid(H[nn, i, j])
                        for i in T.parallel(64):
                            for j in T.parallel(128):
                                H[nn + 2, i, j] = H[nn + 2, i, j] * H[nn, i, j]
                        T.copy(T.region(H[nn + 2, 0, 0], 1, 1, 64, 128), T.region(Hc[nn, 0, 0], 2, 1, 64, 128))
                    T.copy(T.region(w[ee * 2 + 1, t * 64], 1, 1, 64), T.region(w_s[0], 2, 64))
                    T.copy(T.region(w_s[0], 1, 64), T.region(w_c[0], 2, 64))
                    T.copy(T.region(dst[ee * 2 + 1, t * 64], 1, 1, 64), T.region(dst_s[0], 2, 64))
                    with T.sblock(""):
                        T.reads(Wq_1[ee * 2 + 1, 0:128, 0:64], Wq_s[0:128, 0:64], Ws_1[ee * 2 + 1, 0:4, 0:4], Ws_s[0:4, 0:4], Wq_1[ee * 2 + 1, T.FloorMod(0, 2) * 128:T.FloorMod(0, 2) * 128 + 128, T.FloorDiv(0, 2) * 64:T.FloorDiv(0, 2) * 64 + 64], Ws_1[ee * 2 + 1, T.FloorMod(0, 2) * 4:T.FloorMod(0, 2) * 4 + 4, T.FloorDiv(0, 2) * 4:T.FloorDiv(0, 2) * 4 + 4])
                        T.writes(Wq_s[0:128, 0:64], Ws_s[0:4, 0:4], W_s[0:3, 0:128, 0:128], Wq_s[0:128, 0:64], Ws_s[0:4, 0:4])
                        with T.sblock(""):
                            T.reads(Wq_1[ee * 2 + 1, T.FloorMod(0, 2) * 128, T.FloorDiv(0, 2) * 64], Wq_s[0, 0], Wq_1[ee * 2 + 1, T.FloorMod(0, 2) * 128:T.FloorMod(0, 2) * 128 + 128, T.FloorDiv(0, 2) * 64:T.FloorDiv(0, 2) * 64 + 64])
                            T.writes(Wq_s[0:128, 0:64])
                            T.copy(T.region(Wq_1[ee * 2 + 1, T.FloorMod(0, 2) * 128, T.FloorDiv(0, 2) * 64], 1, 1, 128, 64), T.region(Wq_s[0, 0], 2, 128, 64))
                        with T.sblock(""):
                            T.reads(Ws_1[ee * 2 + 1, T.FloorMod(0, 2) * 4, T.FloorDiv(0, 2) * 4], Ws_s[0, 0], Ws_1[ee * 2 + 1, T.FloorMod(0, 2) * 4:T.FloorMod(0, 2) * 4 + 4, T.FloorDiv(0, 2) * 4:T.FloorDiv(0, 2) * 4 + 4])
                            T.writes(Ws_s[0:4, 0:4])
                            T.copy(T.region(Ws_1[ee * 2 + 1, T.FloorMod(0, 2) * 4, T.FloorDiv(0, 2) * 4], 1, 1, 4, 4), T.region(Ws_s[0, 0], 2, 4, 4))
                        T.call_extern("handle", "tcta.fill_issued")
                        T.call_extern("handle", "tcta.fill_wait", 0)
                        with T.sblock(""):
                            T.reads(Wq_s[0:128, 0:64], Ws_s[0:4, 0:4])
                            T.writes(W_s[0:3, 0:128, 0:128])
                            for i in T.parallel(128):
                                for j in T.parallel(128):
                                    byte: T.int32 = T.Cast("int32", Wq_s[i, j // 2])
                                    mag: T.float32 = T.if_then_else(T.bitwise_and(T.bitwise_and(T.shift_right(byte, j % 2 * 4), 15), 7) < 2, T.Cast("float32", T.bitwise_and(T.bitwise_and(T.shift_right(byte, j % 2 * 4), 15), 7)) * T.float32(0.5), T.if_then_else(T.bitwise_and(T.bitwise_and(T.shift_right(byte, j % 2 * 4), 15), 7) < 4, T.float32(1.0) + T.Cast("float32", T.bitwise_and(T.bitwise_and(T.shift_right(byte, j % 2 * 4), 15), 7) - 2) * T.float32(0.5), T.if_then_else(T.bitwise_and(T.bitwise_and(T.shift_right(byte, j % 2 * 4), 15), 7) < 6, T.float32(2.0) + T.Cast("float32", T.bitwise_and(T.bitwise_and(T.shift_right(byte, j % 2 * 4), 15), 7) - 4), T.float32(4.0) + T.Cast("float32", T.bitwise_and(T.bitwise_and(T.shift_right(byte, j % 2 * 4), 15), 7) - 6) * T.float32(2.0))))
                                    scale: T.float32 = T.exp2(T.Cast("float32", Ws_s[i // 32, j // 32]) - T.float32(127.0))
                                    W_s[T.FloorMod(0, 3), i, j] = T.Cast("float16", T.if_then_else(T.shift_right(T.bitwise_and(T.shift_right(byte, j % 2 * 4), 15), 3) == 1, mag * T.float32(-1.0), mag) * scale)
                    with T.sblock(""):
                        T.reads(Wq_1[ee * 2 + 1, 128:256, 0:64], Wq_s[0:128, 0:64], Ws_1[ee * 2 + 1, 4:8, 0:4], Ws_s[0:4, 0:4], Wq_1[ee * 2 + 1, T.FloorMod(1, 2) * 128:T.FloorMod(1, 2) * 128 + 128, T.FloorDiv(1, 2) * 64:T.FloorDiv(1, 2) * 64 + 64], Ws_1[ee * 2 + 1, T.FloorMod(1, 2) * 4:T.FloorMod(1, 2) * 4 + 4, T.FloorDiv(1, 2) * 4:T.FloorDiv(1, 2) * 4 + 4])
                        T.writes(Wq_s[0:128, 0:64], Ws_s[0:4, 0:4], W_s[0:3, 0:128, 0:128], Wq_s[0:128, 0:64], Ws_s[0:4, 0:4])
                        with T.sblock(""):
                            T.reads(Wq_1[ee * 2 + 1, T.FloorMod(1, 2) * 128, T.FloorDiv(1, 2) * 64], Wq_s[0, 0], Wq_1[ee * 2 + 1, T.FloorMod(1, 2) * 128:T.FloorMod(1, 2) * 128 + 128, T.FloorDiv(1, 2) * 64:T.FloorDiv(1, 2) * 64 + 64])
                            T.writes(Wq_s[0:128, 0:64])
                            T.copy(T.region(Wq_1[ee * 2 + 1, T.FloorMod(1, 2) * 128, T.FloorDiv(1, 2) * 64], 1, 1, 128, 64), T.region(Wq_s[0, 0], 2, 128, 64))
                        with T.sblock(""):
                            T.reads(Ws_1[ee * 2 + 1, T.FloorMod(1, 2) * 4, T.FloorDiv(1, 2) * 4], Ws_s[0, 0], Ws_1[ee * 2 + 1, T.FloorMod(1, 2) * 4:T.FloorMod(1, 2) * 4 + 4, T.FloorDiv(1, 2) * 4:T.FloorDiv(1, 2) * 4 + 4])
                            T.writes(Ws_s[0:4, 0:4])
                            T.copy(T.region(Ws_1[ee * 2 + 1, T.FloorMod(1, 2) * 4, T.FloorDiv(1, 2) * 4], 1, 1, 4, 4), T.region(Ws_s[0, 0], 2, 4, 4))
                        T.call_extern("handle", "tcta.fill_issued")
                        T.call_extern("handle", "tcta.fill_wait", 0)
                        with T.sblock(""):
                            T.reads(Wq_s[0:128, 0:64], Ws_s[0:4, 0:4])
                            T.writes(W_s[0:3, 0:128, 0:128])
                            for i in T.parallel(128):
                                for j in T.parallel(128):
                                    byte: T.int32 = T.Cast("int32", Wq_s[i, j // 2])
                                    mag: T.float32 = T.if_then_else(T.bitwise_and(T.bitwise_and(T.shift_right(byte, j % 2 * 4), 15), 7) < 2, T.Cast("float32", T.bitwise_and(T.bitwise_and(T.shift_right(byte, j % 2 * 4), 15), 7)) * T.float32(0.5), T.if_then_else(T.bitwise_and(T.bitwise_and(T.shift_right(byte, j % 2 * 4), 15), 7) < 4, T.float32(1.0) + T.Cast("float32", T.bitwise_and(T.bitwise_and(T.shift_right(byte, j % 2 * 4), 15), 7) - 2) * T.float32(0.5), T.if_then_else(T.bitwise_and(T.bitwise_and(T.shift_right(byte, j % 2 * 4), 15), 7) < 6, T.float32(2.0) + T.Cast("float32", T.bitwise_and(T.bitwise_and(T.shift_right(byte, j % 2 * 4), 15), 7) - 4), T.float32(4.0) + T.Cast("float32", T.bitwise_and(T.bitwise_and(T.shift_right(byte, j % 2 * 4), 15), 7) - 6) * T.float32(2.0))))
                                    scale: T.float32 = T.exp2(T.Cast("float32", Ws_s[i // 32, j // 32]) - T.float32(127.0))
                                    W_s[T.FloorMod(1, 3), i, j] = T.Cast("float16", T.if_then_else(T.shift_right(T.bitwise_and(T.shift_right(byte, j % 2 * 4), 15), 3) == 1, mag * T.float32(-1.0), mag) * scale)
                    with T.sblock(""):
                        m = T.int32()
                        T.reads(out[0:2, 0:64, 0:128], Wq_1[ee * 2 + 1, 0:256, 64:256], Wq_s[0:128, 0:64], Ws_1[ee * 2 + 1, 0:8, 4:16], Ws_s[0:4, 0:4], Hc[0:2, 0:64, 0:128], W_s[0:3, 0:128, 0:128], w_c[0:64], dst_s[0:64], Wq_1[ee * 2 + 1, (m + 2) % 2 * 128:(m + 2) % 2 * 128 + 128, (m + 2) // 2 * 64:(m + 2) // 2 * 64 + 64], Ws_1[ee * 2 + 1, (m + 2) % 2 * 4:(m + 2) % 2 * 4 + 4, (m + 2) // 2 * 4:(m + 2) // 2 * 4 + 4], Hc[m % 2, 0:64, 0:128], W_s[m % 3, 0:128, 0:128], out[m % 4 // 2, 0:64, 0:128])
                        T.writes(out[0:2, 0:64, 0:128], Wq_s[0:128, 0:64], Ws_s[0:4, 0:4], W_s[0:3, 0:128, 0:128], Yp[0:1536, 0:384], out[m % 4 // 2, 0:64, 0:128], Wq_s[0:128, 0:64], Ws_s[0:4, 0:4], out[m % 4 // 2, 0:64, 0:128])
                        for m in T.serial(6, annotations={"tl_pipelined_num_stages": 3}):
                            with T.sblock(""):
                                T.reads(out[m % 4 // 2, 0, 0])
                                T.writes(out[m % 4 // 2, 0:64, 0:128])
                                if m % 2 == 0:
                                    T.fill(T.region(out[m % 4 // 2, 0, 0], 2, 1, 64, 128), 0)
                            with T.sblock(""):
                                T.reads(Wq_1[ee * 2 + 1, (m + 2) % 2 * 128, (m + 2) // 2 * 64], Wq_s[0, 0], Wq_1[ee * 2 + 1, (m + 2) % 2 * 128:(m + 2) % 2 * 128 + 128, (m + 2) // 2 * 64:(m + 2) // 2 * 64 + 64])
                                T.writes(Wq_s[0:128, 0:64])
                                T.copy(T.region(Wq_1[ee * 2 + 1, (m + 2) % 2 * 128, (m + 2) // 2 * 64], 1, 1, 128, 64), T.region(Wq_s[0, 0], 2, 128, 64))
                            with T.sblock(""):
                                T.reads(Ws_1[ee * 2 + 1, (m + 2) % 2 * 4, (m + 2) // 2 * 4], Ws_s[0, 0], Ws_1[ee * 2 + 1, (m + 2) % 2 * 4:(m + 2) % 2 * 4 + 4, (m + 2) // 2 * 4:(m + 2) // 2 * 4 + 4])
                                T.writes(Ws_s[0:4, 0:4])
                                T.copy(T.region(Ws_1[ee * 2 + 1, (m + 2) % 2 * 4, (m + 2) // 2 * 4], 1, 1, 4, 4), T.region(Ws_s[0, 0], 2, 4, 4))
                            T.call_extern("handle", "tcta.fill_issued")
                            T.call_extern("handle", "tcta.fill_wait", 2)
                            with T.sblock(""):
                                T.reads(Wq_s[0:128, 0:64], Ws_s[0:4, 0:4])
                                T.writes(W_s[0:3, 0:128, 0:128])
                                for i in T.parallel(128):
                                    for j in T.parallel(128):
                                        byte: T.int32 = T.Cast("int32", Wq_s[i, j // 2])
                                        mag: T.float32 = T.if_then_else(T.bitwise_and(T.bitwise_and(T.shift_right(byte, j % 2 * 4), 15), 7) < 2, T.Cast("float32", T.bitwise_and(T.bitwise_and(T.shift_right(byte, j % 2 * 4), 15), 7)) * T.float32(0.5), T.if_then_else(T.bitwise_and(T.bitwise_and(T.shift_right(byte, j % 2 * 4), 15), 7) < 4, T.float32(1.0) + T.Cast("float32", T.bitwise_and(T.bitwise_and(T.shift_right(byte, j % 2 * 4), 15), 7) - 2) * T.float32(0.5), T.if_then_else(T.bitwise_and(T.bitwise_and(T.shift_right(byte, j % 2 * 4), 15), 7) < 6, T.float32(2.0) + T.Cast("float32", T.bitwise_and(T.bitwise_and(T.shift_right(byte, j % 2 * 4), 15), 7) - 4), T.float32(4.0) + T.Cast("float32", T.bitwise_and(T.bitwise_and(T.shift_right(byte, j % 2 * 4), 15), 7) - 6) * T.float32(2.0))))
                                        scale: T.float32 = T.exp2(T.Cast("float32", Ws_s[i // 32, j // 32]) - T.float32(127.0))
                                        W_s[(m + 2) % 3, i, j] = T.Cast("float16", T.if_then_else(T.shift_right(T.bitwise_and(T.shift_right(byte, j % 2 * 4), 15), 3) == 1, mag * T.float32(-1.0), mag) * scale)
                            with T.sblock(""):
                                T.reads(Hc[m % 2, 0, 0], W_s[0:3, 0, 0], out[m % 4 // 2, 0, 0], Hc[m % 2, 0:64, 0:128], W_s[0:3, 0:128, 0:128], out[m % 4 // 2, 0:64, 0:128])
                                T.writes(out[m % 4 // 2, 0:64, 0:128])
                                T.gemm(T.region(Hc[m % 2, 0, 0], 1, 1, 64, 128), T.region(W_s[m % 3, 0, 0], 1, 1, 128, 128), T.region(out[m % 4 // 2, 0, 0], 3, 1, 64, 128), T.bool(False), T.bool(False), 64, 128, 128, 0, T.bool(False), 0, 0, 0)
                            with T.sblock(""):
                                T.reads(out[m % 4 // 2, 0:64, 0:128], w_c[0:64])
                                T.writes(out[m % 4 // 2, 0:64, 0:128])
                                if m % 2 == 1:
                                    for i in T.parallel(64):
                                        for j in T.parallel(128):
                                            out[m % 4 // 2, i, j] = out[m % 4 // 2, i, j] * w_c[i]
                            with T.sblock(""):
                                T.reads(dst_s[0:64], out[m % 4 // 2, 0:64, 0:128])
                                T.writes(Yp[0:1536, m // 2 * 128:m // 2 * 128 + 128])
                                if m % 2 == 1:
                                    for i in T.parallel(64):
                                        for j in T.parallel(128):
                                            if 0 <= dst_s[i]:
                                                Yp[dst_s[i], m // 2 * 128 + j] = out[m % 4 // 2, i, j]
                    with T.sblock(""):
                        T.reads(out[1, 0:64, 0:128], Hc[0, 0:64, 0:128], W_s[0:3, 0:128, 0:128], w_c[0:64], dst_s[0:64], Hc[T.FloorMod(6, 2), 0:64, 0:128], W_s[T.FloorMod(6, 3), 0:128, 0:128], out[T.FloorMod(6, 4) // 2, 0:64, 0:128])
                        T.writes(out[1, 0:64, 0:128], Yp[0:1536, 384:512], out[T.FloorMod(6, 4) // 2, 0:64, 0:128], out[T.FloorMod(6, 4) // 2, 0:64, 0:128])
                        with T.sblock(""):
                            T.reads(out[T.FloorMod(6, 4) // 2, 0, 0])
                            T.writes(out[T.FloorMod(6, 4) // 2, 0:64, 0:128])
                            if T.FloorMod(6, 2) == 0:
                                T.fill(T.region(out[T.FloorMod(6, 4) // 2, 0, 0], 2, 1, 64, 128), 0)
                        with T.sblock(""):
                            T.reads(Hc[T.FloorMod(6, 2), 0, 0], W_s[0:3, 0, 0], out[T.FloorMod(6, 4) // 2, 0, 0], Hc[T.FloorMod(6, 2), 0:64, 0:128], W_s[0:3, 0:128, 0:128], out[T.FloorMod(6, 4) // 2, 0:64, 0:128])
                            T.writes(out[T.FloorMod(6, 4) // 2, 0:64, 0:128])
                            T.gemm(T.region(Hc[T.FloorMod(6, 2), 0, 0], 1, 1, 64, 128), T.region(W_s[T.FloorMod(6, 3), 0, 0], 1, 1, 128, 128), T.region(out[T.FloorMod(6, 4) // 2, 0, 0], 3, 1, 64, 128), T.bool(False), T.bool(False), 64, 128, 128, 0, T.bool(False), 0, 0, 0)
                        with T.sblock(""):
                            T.reads(out[T.FloorMod(6, 4) // 2, 0:64, 0:128], w_c[0:64])
                            T.writes(out[T.FloorMod(6, 4) // 2, 0:64, 0:128])
                            if T.FloorMod(6, 2) == 1:
                                for i in T.parallel(64):
                                    for j in T.parallel(128):
                                        out[T.FloorMod(6, 4) // 2, i, j] = out[T.FloorMod(6, 4) // 2, i, j] * w_c[i]
                        with T.sblock(""):
                            T.reads(dst_s[0:64], out[T.FloorMod(6, 4) // 2, 0:64, 0:128])
                            T.writes(Yp[0:1536, T.FloorDiv(6, 2) * 128:T.FloorDiv(6, 2) * 128 + 128])
                            if T.FloorMod(6, 2) == 1:
                                for i in T.parallel(64):
                                    for j in T.parallel(128):
                                        if 0 <= dst_s[i]:
                                            Yp[dst_s[i], T.FloorDiv(6, 2) * 128 + j] = out[T.FloorMod(6, 4) // 2, i, j]
                    with T.sblock(""):
                        T.reads(out[1, 0:64, 0:128], Hc[1, 0:64, 0:128], W_s[0:3, 0:128, 0:128], w_c[0:64], dst_s[0:64], Hc[T.FloorMod(7, 2), 0:64, 0:128], W_s[T.FloorMod(7, 3), 0:128, 0:128], out[T.FloorMod(7, 4) // 2, 0:64, 0:128])
                        T.writes(out[1, 0:64, 0:128], Yp[0:1536, 384:512], out[T.FloorMod(7, 4) // 2, 0:64, 0:128], out[T.FloorMod(7, 4) // 2, 0:64, 0:128])
                        with T.sblock(""):
                            T.reads(out[T.FloorMod(7, 4) // 2, 0, 0])
                            T.writes(out[T.FloorMod(7, 4) // 2, 0:64, 0:128])
                            if T.FloorMod(7, 2) == 0:
                                T.fill(T.region(out[T.FloorMod(7, 4) // 2, 0, 0], 2, 1, 64, 128), 0)
                        with T.sblock(""):
                            T.reads(Hc[T.FloorMod(7, 2), 0, 0], W_s[0:3, 0, 0], out[T.FloorMod(7, 4) // 2, 0, 0], Hc[T.FloorMod(7, 2), 0:64, 0:128], W_s[0:3, 0:128, 0:128], out[T.FloorMod(7, 4) // 2, 0:64, 0:128])
                            T.writes(out[T.FloorMod(7, 4) // 2, 0:64, 0:128])
                            T.gemm(T.region(Hc[T.FloorMod(7, 2), 0, 0], 1, 1, 64, 128), T.region(W_s[T.FloorMod(7, 3), 0, 0], 1, 1, 128, 128), T.region(out[T.FloorMod(7, 4) // 2, 0, 0], 3, 1, 64, 128), T.bool(False), T.bool(False), 64, 128, 128, 0, T.bool(False), 0, 0, 0)
                        with T.sblock(""):
                            T.reads(out[T.FloorMod(7, 4) // 2, 0:64, 0:128], w_c[0:64])
                            T.writes(out[T.FloorMod(7, 4) // 2, 0:64, 0:128])
                            if T.FloorMod(7, 2) == 1:
                                for i in T.parallel(64):
                                    for j in T.parallel(128):
                                        out[T.FloorMod(7, 4) // 2, i, j] = out[T.FloorMod(7, 4) // 2, i, j] * w_c[i]
                        with T.sblock(""):
                            T.reads(dst_s[0:64], out[T.FloorMod(7, 4) // 2, 0:64, 0:128])
                            T.writes(Yp[0:1536, T.FloorDiv(7, 2) * 128:T.FloorDiv(7, 2) * 128 + 128])
                            if T.FloorMod(7, 2) == 1:
                                for i in T.parallel(64):
                                    for j in T.parallel(128):
                                        if 0 <= dst_s[i]:
                                            Yp[dst_s[i], T.FloorDiv(7, 2) * 128 + j] = out[T.FloorMod(7, 4) // 2, i, j]