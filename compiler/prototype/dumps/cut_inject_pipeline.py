# D3 output: lower_to_tile_tir(build_fa_kernel(), cut='inject_pipeline')
# from tvm.script import tirx as T

@T.prim_func
def main(Q_handle: T.handle, K_handle: T.handle, V_handle: T.handle, Output_handle: T.handle, bx: T.int32, by: T.int32):
    T.func_attr({"target": T.target({"arch": "sm_80", "keys": ["cuda", "gpu"], "kind": "cuda", "max_num_threads": 1024, "tag": "", "thread_warp_size": 32}), "tcta.arch": "sm_80", "tcta.cut": "tl.InjectSoftwarePipeline"})
    Q = T.match_buffer(Q_handle, (1, 8, 256, 64), "float16", strides=(131072, 16384, 64, 1))
    K = T.match_buffer(K_handle, (1, 8, 256, 64), "float16", strides=(131072, 16384, 64, 1))
    V = T.match_buffer(V_handle, (1, 8, 256, 64), "float16", strides=(131072, 16384, 64, 1))
    Output = T.match_buffer(Output_handle, (1, 8, 256, 64), "float16", strides=(131072, 16384, 64, 1))
    with T.sblock("root"):
        T.reads()
        T.writes()
        with T.sblock("tilelang_root"):
            T.reads()
            T.writes()
            Q_shared = T.sblock_alloc_buffer((64, 64), "float16", scope="tcta.inp")
            K_shared = T.sblock_alloc_buffer((2, 64, 64), "float16", scope="tcta.wgt")
            V_shared = T.sblock_alloc_buffer((2, 64, 64), "float16", scope="tcta.wgt")
            acc_s = T.sblock_alloc_buffer((64, 64), scope="tcta.acc")
            acc_s_cast = T.sblock_alloc_buffer((64, 64), "float16", scope="tcta.acc")
            acc_o = T.sblock_alloc_buffer((64, 64), scope="tcta.acc")
            scores_max = T.sblock_alloc_buffer((64,), scope="tcta.vec")
            scores_max_prev = T.sblock_alloc_buffer((64,), scope="tcta.vec")
            scores_scale = T.sblock_alloc_buffer((64,), scope="tcta.vec")
            scores_sum = T.sblock_alloc_buffer((64,), scope="tcta.vec")
            logsum = T.sblock_alloc_buffer((64,), scope="tcta.vec")
            T.copy(T.region(Q[0, by, bx * 64, 0], 1, 1, 1, 64, 64), T.region(Q_shared[0, 0], 2, 64, 64))
            T.fill(T.region(acc_o[0, 0], 2, 64, 64), 0)
            T.fill(T.region(logsum[0], 2, 64), 0)
            T.fill(T.region(scores_max[0], 2, 64), T.infinity("float32") * T.float32(-1.0))
            with T.sblock(""):
                T.reads(K[0, by, 0:64, 0:64], K_shared[0:2, 0, 0], V[0, by, 0:64, 0:64], V_shared[0:2, 0, 0], K[0, by, T.Mul(0, 64):T.Mul(0, 64) + 64, 0:64], V[0, by, T.Mul(0, 64):T.Mul(0, 64) + 64, 0:64])
                T.writes(K_shared[0:2, 0:64, 0:64], V_shared[0:2, 0:64, 0:64], K_shared[T.FloorMod(0, 2), 0:64, 0:64], V_shared[T.FloorMod(0, 2), 0:64, 0:64])
                with T.sblock(""):
                    T.reads(K[0, by, T.Mul(0, 64), 0], K_shared[0:2, 0, 0], K[0, by, T.Mul(0, 64):T.Mul(0, 64) + 64, 0:64])
                    T.writes(K_shared[0:2, 0:64, 0:64])
                    T.copy(T.region(K[0, by, T.Mul(0, 64), 0], 1, 1, 1, 64, 64), T.region(K_shared[T.FloorMod(0, 2), 0, 0], 2, 1, 64, 64))
                T.call_extern("handle", "tcta.fill_issued")
                with T.sblock(""):
                    T.reads(V[0, by, T.Mul(0, 64), 0], V_shared[0:2, 0, 0], V[0, by, T.Mul(0, 64):T.Mul(0, 64) + 64, 0:64])
                    T.writes(V_shared[0:2, 0:64, 0:64])
                    T.copy(T.region(V[0, by, T.Mul(0, 64), 0], 1, 1, 1, 64, 64), T.region(V_shared[T.FloorMod(0, 2), 0, 0], 2, 1, 64, 64))
                T.call_extern("handle", "tcta.fill_issued")
            with T.sblock(""):
                T.reads(K[0, by, 64:128, 0:64], K_shared[0:2, 0, 0], V[0, by, 64:128, 0:64], V_shared[0:2, 0, 0], K[0, by, T.Mul(1, 64):T.Mul(1, 64) + 64, 0:64], V[0, by, T.Mul(1, 64):T.Mul(1, 64) + 64, 0:64])
                T.writes(K_shared[0:2, 0:64, 0:64], V_shared[0:2, 0:64, 0:64], K_shared[T.FloorMod(1, 2), 0:64, 0:64], V_shared[T.FloorMod(1, 2), 0:64, 0:64])
                with T.sblock(""):
                    T.reads(K[0, by, T.Mul(1, 64), 0], K_shared[0:2, 0, 0], K[0, by, T.Mul(1, 64):T.Mul(1, 64) + 64, 0:64])
                    T.writes(K_shared[0:2, 0:64, 0:64])
                    T.copy(T.region(K[0, by, T.Mul(1, 64), 0], 1, 1, 1, 64, 64), T.region(K_shared[T.FloorMod(1, 2), 0, 0], 2, 1, 64, 64))
                T.call_extern("handle", "tcta.fill_issued")
                with T.sblock(""):
                    T.reads(V[0, by, T.Mul(1, 64), 0], V_shared[0:2, 0, 0], V[0, by, T.Mul(1, 64):T.Mul(1, 64) + 64, 0:64])
                    T.writes(V_shared[0:2, 0:64, 0:64])
                    T.copy(T.region(V[0, by, T.Mul(1, 64), 0], 1, 1, 1, 64, 64), T.region(V_shared[T.FloorMod(1, 2), 0, 0], 2, 1, 64, 64))
                T.call_extern("handle", "tcta.fill_issued")
            with T.sblock(""):
                k = T.int32()
                T.reads(acc_s[0:64, 0:64], Q_shared[0:64, 0:64], K_shared[0:2, 0:64, 0:64], K[0, by, 128:256, 0:64], scores_max[0:64], scores_max_prev[0:64], scores_sum[0:64], logsum[0:64], scores_scale[0:64], acc_s_cast[0:64, 0:64], acc_o[0:64, 0:64], V_shared[0:2, 0:64, 0:64], V[0, by, 128:256, 0:64], Q_shared[0:64, 0:64], K_shared[k % 2, 0:64, 0:64], acc_s[0:64, 0:64], K[0, by, (k + 2) * 64:(k + 2) * 64 + 64, 0:64], scores_max[0:64], acc_s[0:64, 0:64], scores_max[0:64], acc_s[0:64, 0:64], acc_s[0:64, 0:64], acc_s_cast[0:64, 0:64], V_shared[k % 2, 0:64, 0:64], acc_o[0:64, 0:64], V[0, by, (k + 2) * 64:(k + 2) * 64 + 64, 0:64])
                T.writes(acc_s[0:64, 0:64], K_shared[0:2, 0:64, 0:64], scores_max_prev[0:64], scores_max[0:64], scores_scale[0:64], scores_sum[0:64], logsum[0:64], acc_s_cast[0:64, 0:64], acc_o[0:64, 0:64], V_shared[0:2, 0:64, 0:64], acc_s[0:64, 0:64], acc_s[0:64, 0:64], K_shared[(k + 2) % 2, 0:64, 0:64], scores_max_prev[0:64], scores_max[0:64], scores_max[0:64], scores_sum[0:64], acc_s_cast[0:64, 0:64], acc_o[0:64, 0:64], V_shared[(k + 2) % 2, 0:64, 0:64])
                for k in T.serial(2, annotations={"tl_pipelined_num_stages": 2}):
                    with T.sblock(""):
                        T.reads(acc_s[0, 0])
                        T.writes(acc_s[0:64, 0:64])
                        T.fill(T.region(acc_s[0, 0], 2, 64, 64), 0)
                    T.call_extern("handle", "tcta.fill_wait", 3)
                    with T.sblock(""):
                        T.reads(Q_shared[0, 0], K_shared[0:2, 0, 0], acc_s[0, 0], Q_shared[0:64, 0:64], K_shared[0:2, 0:64, 0:64], acc_s[0:64, 0:64])
                        T.writes(acc_s[0:64, 0:64])
                        T.gemm(T.region(Q_shared[0, 0], 1, 64, 64), T.region(K_shared[k % 2, 0, 0], 1, 1, 64, 64), T.region(acc_s[0, 0], 3, 64, 64), T.bool(False), T.bool(True), 64, 64, 64, 1, T.bool(False), 0, 0, 0)
                    with T.sblock(""):
                        T.reads(K[0, by, (k + 2) * 64, 0], K_shared[0:2, 0, 0], K[0, by, (k + 2) * 64:(k + 2) * 64 + 64, 0:64])
                        T.writes(K_shared[0:2, 0:64, 0:64])
                        T.copy(T.region(K[0, by, (k + 2) * 64, 0], 1, 1, 1, 64, 64), T.region(K_shared[(k + 2) % 2, 0, 0], 2, 1, 64, 64))
                    T.call_extern("handle", "tcta.fill_issued")
                    with T.sblock(""):
                        T.reads(scores_max[0], scores_max_prev[0], scores_max[0:64])
                        T.writes(scores_max_prev[0:64])
                        T.copy(T.region(scores_max[0], 1, 64), T.region(scores_max_prev[0], 2, 64))
                    with T.sblock(""):
                        T.reads(scores_max[0])
                        T.writes(scores_max[0:64])
                        T.fill(T.region(scores_max[0], 2, 64), T.infinity("float32") * T.float32(-1.0))
                    with T.sblock(""):
                        T.reads(acc_s[0, 0], scores_max[0], acc_s[0:64, 0:64], scores_max[0:64])
                        T.writes(scores_max[0:64])
                        T.reduce(T.region(acc_s[0, 0], 1, 64, 64), T.region(scores_max[0], 2, 64), "max", 1, T.bool(False))
                    with T.sblock(""):
                        T.reads(scores_max[0:64], scores_max_prev[0:64])
                        T.writes(scores_max[0:64])
                        for i in T.parallel(64):
                            scores_max[i] = T.max(scores_max[i], scores_max_prev[i])
                    with T.sblock(""):
                        T.reads(scores_max_prev[0:64], scores_max[0:64])
                        T.writes(scores_scale[0:64])
                        for i in T.parallel(64):
                            scores_scale[i] = T.exp2(scores_max_prev[i] * T.float32(0.18033688) - scores_max[i] * T.float32(0.18033688))
                    with T.sblock(""):
                        T.reads(acc_s[0:64, 0:64], scores_max[0:64])
                        T.writes(acc_s[0:64, 0:64])
                        for i in T.parallel(64):
                            for j in T.parallel(64):
                                acc_s[i, j] = T.exp2(acc_s[i, j] * T.float32(0.18033688) - scores_max[i] * T.float32(0.18033688))
                    with T.sblock(""):
                        T.reads(acc_s[0, 0], scores_sum[0], acc_s[0:64, 0:64])
                        T.writes(scores_sum[0:64])
                        T.reduce(T.region(acc_s[0, 0], 1, 64, 64), T.region(scores_sum[0], 2, 64), "sum", 1, T.bool(True))
                    with T.sblock(""):
                        T.reads(logsum[0:64], scores_scale[0:64], scores_sum[0:64])
                        T.writes(logsum[0:64])
                        for i in T.parallel(64):
                            logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
                    with T.sblock(""):
                        T.reads(acc_s[0, 0], acc_s_cast[0, 0], acc_s[0:64, 0:64])
                        T.writes(acc_s_cast[0:64, 0:64])
                        T.copy(T.region(acc_s[0, 0], 1, 64, 64), T.region(acc_s_cast[0, 0], 2, 64, 64))
                    with T.sblock(""):
                        T.reads(acc_o[0:64, 0:64], scores_scale[0:64])
                        T.writes(acc_o[0:64, 0:64])
                        for i in T.parallel(64):
                            for j in T.parallel(64):
                                acc_o[i, j] = acc_o[i, j] * scores_scale[i]
                    T.call_extern("handle", "tcta.fill_wait", 3)
                    with T.sblock(""):
                        T.reads(acc_s_cast[0, 0], V_shared[0:2, 0, 0], acc_o[0, 0], acc_s_cast[0:64, 0:64], V_shared[0:2, 0:64, 0:64], acc_o[0:64, 0:64])
                        T.writes(acc_o[0:64, 0:64])
                        T.gemm(T.region(acc_s_cast[0, 0], 1, 64, 64), T.region(V_shared[k % 2, 0, 0], 1, 1, 64, 64), T.region(acc_o[0, 0], 3, 64, 64), T.bool(False), T.bool(False), 64, 64, 64, 1, T.bool(False), 0, 0, 0)
                    with T.sblock(""):
                        T.reads(V[0, by, (k + 2) * 64, 0], V_shared[0:2, 0, 0], V[0, by, (k + 2) * 64:(k + 2) * 64 + 64, 0:64])
                        T.writes(V_shared[0:2, 0:64, 0:64])
                        T.copy(T.region(V[0, by, (k + 2) * 64, 0], 1, 1, 1, 64, 64), T.region(V_shared[(k + 2) % 2, 0, 0], 2, 1, 64, 64))
                    T.call_extern("handle", "tcta.fill_issued")
            with T.sblock(""):
                T.reads(acc_s[0:64, 0:64], Q_shared[0:64, 0:64], K_shared[0:2, 0:64, 0:64], scores_max[0:64], scores_max_prev[0:64], scores_sum[0:64], logsum[0:64], scores_scale[0:64], acc_s_cast[0:64, 0:64], acc_o[0:64, 0:64], V_shared[0:2, 0:64, 0:64], Q_shared[0:64, 0:64], K_shared[T.FloorMod(2, 2), 0:64, 0:64], acc_s[0:64, 0:64], scores_max[0:64], acc_s[0:64, 0:64], scores_max[0:64], acc_s[0:64, 0:64], acc_s[0:64, 0:64], acc_s_cast[0:64, 0:64], V_shared[T.FloorMod(2, 2), 0:64, 0:64], acc_o[0:64, 0:64])
                T.writes(acc_s[0:64, 0:64], scores_max_prev[0:64], scores_max[0:64], scores_scale[0:64], scores_sum[0:64], logsum[0:64], acc_s_cast[0:64, 0:64], acc_o[0:64, 0:64], acc_s[0:64, 0:64], acc_s[0:64, 0:64], scores_max_prev[0:64], scores_max[0:64], scores_max[0:64], scores_sum[0:64], acc_s_cast[0:64, 0:64], acc_o[0:64, 0:64])
                with T.sblock(""):
                    T.reads(acc_s[0, 0])
                    T.writes(acc_s[0:64, 0:64])
                    T.fill(T.region(acc_s[0, 0], 2, 64, 64), 0)
                T.call_extern("handle", "tcta.fill_wait", 3)
                with T.sblock(""):
                    T.reads(Q_shared[0, 0], K_shared[0:2, 0, 0], acc_s[0, 0], Q_shared[0:64, 0:64], K_shared[0:2, 0:64, 0:64], acc_s[0:64, 0:64])
                    T.writes(acc_s[0:64, 0:64])
                    T.gemm(T.region(Q_shared[0, 0], 1, 64, 64), T.region(K_shared[T.FloorMod(2, 2), 0, 0], 1, 1, 64, 64), T.region(acc_s[0, 0], 3, 64, 64), T.bool(False), T.bool(True), 64, 64, 64, 1, T.bool(False), 0, 0, 0)
                with T.sblock(""):
                    T.reads(scores_max[0], scores_max_prev[0], scores_max[0:64])
                    T.writes(scores_max_prev[0:64])
                    T.copy(T.region(scores_max[0], 1, 64), T.region(scores_max_prev[0], 2, 64))
                with T.sblock(""):
                    T.reads(scores_max[0])
                    T.writes(scores_max[0:64])
                    T.fill(T.region(scores_max[0], 2, 64), T.infinity("float32") * T.float32(-1.0))
                with T.sblock(""):
                    T.reads(acc_s[0, 0], scores_max[0], acc_s[0:64, 0:64], scores_max[0:64])
                    T.writes(scores_max[0:64])
                    T.reduce(T.region(acc_s[0, 0], 1, 64, 64), T.region(scores_max[0], 2, 64), "max", 1, T.bool(False))
                with T.sblock(""):
                    T.reads(scores_max[0:64], scores_max_prev[0:64])
                    T.writes(scores_max[0:64])
                    for i in T.parallel(64):
                        scores_max[i] = T.max(scores_max[i], scores_max_prev[i])
                with T.sblock(""):
                    T.reads(scores_max_prev[0:64], scores_max[0:64])
                    T.writes(scores_scale[0:64])
                    for i in T.parallel(64):
                        scores_scale[i] = T.exp2(scores_max_prev[i] * T.float32(0.18033688) - scores_max[i] * T.float32(0.18033688))
                with T.sblock(""):
                    T.reads(acc_s[0:64, 0:64], scores_max[0:64])
                    T.writes(acc_s[0:64, 0:64])
                    for i in T.parallel(64):
                        for j in T.parallel(64):
                            acc_s[i, j] = T.exp2(acc_s[i, j] * T.float32(0.18033688) - scores_max[i] * T.float32(0.18033688))
                with T.sblock(""):
                    T.reads(acc_s[0, 0], scores_sum[0], acc_s[0:64, 0:64])
                    T.writes(scores_sum[0:64])
                    T.reduce(T.region(acc_s[0, 0], 1, 64, 64), T.region(scores_sum[0], 2, 64), "sum", 1, T.bool(True))
                with T.sblock(""):
                    T.reads(logsum[0:64], scores_scale[0:64], scores_sum[0:64])
                    T.writes(logsum[0:64])
                    for i in T.parallel(64):
                        logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
                with T.sblock(""):
                    T.reads(acc_s[0, 0], acc_s_cast[0, 0], acc_s[0:64, 0:64])
                    T.writes(acc_s_cast[0:64, 0:64])
                    T.copy(T.region(acc_s[0, 0], 1, 64, 64), T.region(acc_s_cast[0, 0], 2, 64, 64))
                with T.sblock(""):
                    T.reads(acc_o[0:64, 0:64], scores_scale[0:64])
                    T.writes(acc_o[0:64, 0:64])
                    for i in T.parallel(64):
                        for j in T.parallel(64):
                            acc_o[i, j] = acc_o[i, j] * scores_scale[i]
                T.call_extern("handle", "tcta.fill_wait", 2)
                with T.sblock(""):
                    T.reads(acc_s_cast[0, 0], V_shared[0:2, 0, 0], acc_o[0, 0], acc_s_cast[0:64, 0:64], V_shared[0:2, 0:64, 0:64], acc_o[0:64, 0:64])
                    T.writes(acc_o[0:64, 0:64])
                    T.gemm(T.region(acc_s_cast[0, 0], 1, 64, 64), T.region(V_shared[T.FloorMod(2, 2), 0, 0], 1, 1, 64, 64), T.region(acc_o[0, 0], 3, 64, 64), T.bool(False), T.bool(False), 64, 64, 64, 1, T.bool(False), 0, 0, 0)
            with T.sblock(""):
                T.reads(acc_s[0:64, 0:64], Q_shared[0:64, 0:64], K_shared[0:2, 0:64, 0:64], scores_max[0:64], scores_max_prev[0:64], scores_sum[0:64], logsum[0:64], scores_scale[0:64], acc_s_cast[0:64, 0:64], acc_o[0:64, 0:64], V_shared[0:2, 0:64, 0:64], Q_shared[0:64, 0:64], K_shared[T.FloorMod(3, 2), 0:64, 0:64], acc_s[0:64, 0:64], scores_max[0:64], acc_s[0:64, 0:64], scores_max[0:64], acc_s[0:64, 0:64], acc_s[0:64, 0:64], acc_s_cast[0:64, 0:64], V_shared[T.FloorMod(3, 2), 0:64, 0:64], acc_o[0:64, 0:64])
                T.writes(acc_s[0:64, 0:64], scores_max_prev[0:64], scores_max[0:64], scores_scale[0:64], scores_sum[0:64], logsum[0:64], acc_s_cast[0:64, 0:64], acc_o[0:64, 0:64], acc_s[0:64, 0:64], acc_s[0:64, 0:64], scores_max_prev[0:64], scores_max[0:64], scores_max[0:64], scores_sum[0:64], acc_s_cast[0:64, 0:64], acc_o[0:64, 0:64])
                with T.sblock(""):
                    T.reads(acc_s[0, 0])
                    T.writes(acc_s[0:64, 0:64])
                    T.fill(T.region(acc_s[0, 0], 2, 64, 64), 0)
                T.call_extern("handle", "tcta.fill_wait", 1)
                with T.sblock(""):
                    T.reads(Q_shared[0, 0], K_shared[0:2, 0, 0], acc_s[0, 0], Q_shared[0:64, 0:64], K_shared[0:2, 0:64, 0:64], acc_s[0:64, 0:64])
                    T.writes(acc_s[0:64, 0:64])
                    T.gemm(T.region(Q_shared[0, 0], 1, 64, 64), T.region(K_shared[T.FloorMod(3, 2), 0, 0], 1, 1, 64, 64), T.region(acc_s[0, 0], 3, 64, 64), T.bool(False), T.bool(True), 64, 64, 64, 1, T.bool(False), 0, 0, 0)
                with T.sblock(""):
                    T.reads(scores_max[0], scores_max_prev[0], scores_max[0:64])
                    T.writes(scores_max_prev[0:64])
                    T.copy(T.region(scores_max[0], 1, 64), T.region(scores_max_prev[0], 2, 64))
                with T.sblock(""):
                    T.reads(scores_max[0])
                    T.writes(scores_max[0:64])
                    T.fill(T.region(scores_max[0], 2, 64), T.infinity("float32") * T.float32(-1.0))
                with T.sblock(""):
                    T.reads(acc_s[0, 0], scores_max[0], acc_s[0:64, 0:64], scores_max[0:64])
                    T.writes(scores_max[0:64])
                    T.reduce(T.region(acc_s[0, 0], 1, 64, 64), T.region(scores_max[0], 2, 64), "max", 1, T.bool(False))
                with T.sblock(""):
                    T.reads(scores_max[0:64], scores_max_prev[0:64])
                    T.writes(scores_max[0:64])
                    for i in T.parallel(64):
                        scores_max[i] = T.max(scores_max[i], scores_max_prev[i])
                with T.sblock(""):
                    T.reads(scores_max_prev[0:64], scores_max[0:64])
                    T.writes(scores_scale[0:64])
                    for i in T.parallel(64):
                        scores_scale[i] = T.exp2(scores_max_prev[i] * T.float32(0.18033688) - scores_max[i] * T.float32(0.18033688))
                with T.sblock(""):
                    T.reads(acc_s[0:64, 0:64], scores_max[0:64])
                    T.writes(acc_s[0:64, 0:64])
                    for i in T.parallel(64):
                        for j in T.parallel(64):
                            acc_s[i, j] = T.exp2(acc_s[i, j] * T.float32(0.18033688) - scores_max[i] * T.float32(0.18033688))
                with T.sblock(""):
                    T.reads(acc_s[0, 0], scores_sum[0], acc_s[0:64, 0:64])
                    T.writes(scores_sum[0:64])
                    T.reduce(T.region(acc_s[0, 0], 1, 64, 64), T.region(scores_sum[0], 2, 64), "sum", 1, T.bool(True))
                with T.sblock(""):
                    T.reads(logsum[0:64], scores_scale[0:64], scores_sum[0:64])
                    T.writes(logsum[0:64])
                    for i in T.parallel(64):
                        logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
                with T.sblock(""):
                    T.reads(acc_s[0, 0], acc_s_cast[0, 0], acc_s[0:64, 0:64])
                    T.writes(acc_s_cast[0:64, 0:64])
                    T.copy(T.region(acc_s[0, 0], 1, 64, 64), T.region(acc_s_cast[0, 0], 2, 64, 64))
                with T.sblock(""):
                    T.reads(acc_o[0:64, 0:64], scores_scale[0:64])
                    T.writes(acc_o[0:64, 0:64])
                    for i in T.parallel(64):
                        for j in T.parallel(64):
                            acc_o[i, j] = acc_o[i, j] * scores_scale[i]
                T.call_extern("handle", "tcta.fill_wait", 0)
                with T.sblock(""):
                    T.reads(acc_s_cast[0, 0], V_shared[0:2, 0, 0], acc_o[0, 0], acc_s_cast[0:64, 0:64], V_shared[0:2, 0:64, 0:64], acc_o[0:64, 0:64])
                    T.writes(acc_o[0:64, 0:64])
                    T.gemm(T.region(acc_s_cast[0, 0], 1, 64, 64), T.region(V_shared[T.FloorMod(3, 2), 0, 0], 1, 1, 64, 64), T.region(acc_o[0, 0], 3, 64, 64), T.bool(False), T.bool(False), 64, 64, 64, 1, T.bool(False), 0, 0, 0)
            for i in T.parallel(64):
                for j in T.parallel(64):
                    acc_o[i, j] = acc_o[i, j] / logsum[i]
            T.copy(T.region(acc_o[0, 0], 1, 64, 64), T.region(Output[0, by, bx * 64, 0], 2, 1, 1, 64, 64))