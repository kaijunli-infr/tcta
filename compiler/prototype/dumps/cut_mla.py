# G1/G2 output: lower_to_tile_tir(mla_kernel.build_mla_kernel()) at tl.InjectSoftwarePipeline
# from tvm.script import tirx as T

@T.prim_func
def main(Q_handle: T.handle, K_handle: T.handle, O_handle: T.handle, bx: T.int32):
    T.func_attr({"target": T.target({"arch": "sm_80", "keys": ["cuda", "gpu"], "kind": "cuda", "max_num_threads": 1024, "tag": "", "thread_warp_size": 32}), "tcta.arch": "sm_80", "tcta.cut": "tl.InjectSoftwarePipeline"})
    Q = T.match_buffer(Q_handle, (64, 576), "float16", strides=(576, 1))
    K = T.match_buffer(K_handle, (1024, 576), "float16", strides=(576, 1))
    O = T.match_buffer(O_handle, (64, 512), strides=(512, 1))
    with T.sblock("root"):
        T.reads()
        T.writes()
        with T.sblock("tilelang_root"):
            T.reads()
            T.writes()
            Q_shared = T.sblock_alloc_buffer((2, 64, 128), "float16", scope="tcta.inp")
            K_shared = T.sblock_alloc_buffer((4, 64, 128), "float16", scope="tcta.wgt")
            acc_s = T.sblock_alloc_buffer((64, 64), scope="tcta.acc")
            acc_s_cast = T.sblock_alloc_buffer((64, 64), "float16", scope="tcta.acc")
            acc_o = T.sblock_alloc_buffer((64, 512), scope="tcta.acc")
            scores_max = T.sblock_alloc_buffer((64,), scope="tcta.vec")
            scores_max_prev = T.sblock_alloc_buffer((64,), scope="tcta.vec")
            scores_scale = T.sblock_alloc_buffer((64,), scope="tcta.vec")
            scores_sum = T.sblock_alloc_buffer((64,), scope="tcta.vec")
            logsum = T.sblock_alloc_buffer((64,), scope="tcta.vec")
            T.fill(T.region(acc_o[0, 0], 2, 64, 512), 0)
            T.fill(T.region(logsum[0], 2, 64), 0)
            T.fill(T.region(scores_max[0], 2, 64), T.infinity("float32") * T.float32(-1.0))
            for kv in range(16):
                T.fill(T.region(acc_s[0, 0], 2, 64, 64), 0)
                T.copy(T.region(Q[0, 512], 1, 64, 64), T.region(Q_shared[0, 0, 0], 2, 1, 64, 64))
                T.copy(T.region(K[kv * 64, 512], 1, 64, 64), T.region(K_shared[3, 0, 0], 2, 1, 64, 64))
                T.gemm(T.region(Q_shared[0, 0, 0], 1, 1, 64, 64), T.region(K_shared[3, 0, 0], 1, 1, 64, 64), T.region(acc_s[0, 0], 3, 64, 64), T.bool(False), T.bool(True), 64, 64, 64, 0, T.bool(False), 0, 0, 0)
                for c in range(4):
                    T.copy(T.region(Q[0, c * 128], 1, 64, 128), T.region(Q_shared[(c + 1) % 2, 0, 0], 2, 1, 64, 128))
                    T.copy(T.region(K[kv * 64, c * 128], 1, 64, 128), T.region(K_shared[c, 0, 0], 2, 1, 64, 128))
                    T.gemm(T.region(Q_shared[(c + 1) % 2, 0, 0], 1, 1, 64, 128), T.region(K_shared[c, 0, 0], 1, 1, 64, 128), T.region(acc_s[0, 0], 3, 64, 64), T.bool(False), T.bool(True), 64, 64, 128, 0, T.bool(False), 0, 0, 0)
                T.copy(T.region(scores_max[0], 1, 64), T.region(scores_max_prev[0], 2, 64))
                T.fill(T.region(scores_max[0], 2, 64), T.infinity("float32") * T.float32(-1.0))
                T.reduce(T.region(acc_s[0, 0], 1, 64, 64), T.region(scores_max[0], 2, 64), "max", 1, T.bool(False))
                for i in T.parallel(64):
                    scores_max[i] = T.max(scores_max[i], scores_max_prev[i])
                for i in T.parallel(64):
                    scores_scale[i] = T.exp2(scores_max_prev[i] * T.float32(0.06011229333333333) - scores_max[i] * T.float32(0.06011229333333333))
                for i in T.parallel(64):
                    for j in T.parallel(64):
                        acc_s[i, j] = T.exp2(acc_s[i, j] * T.float32(0.06011229333333333) - scores_max[i] * T.float32(0.06011229333333333))
                T.reduce(T.region(acc_s[0, 0], 1, 64, 64), T.region(scores_sum[0], 2, 64), "sum", 1, T.bool(True))
                for i in T.parallel(64):
                    logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
                T.copy(T.region(acc_s[0, 0], 1, 64, 64), T.region(acc_s_cast[0, 0], 2, 64, 64))
                for i in T.parallel(64):
                    for j in T.parallel(512):
                        acc_o[i, j] = acc_o[i, j] * scores_scale[i]
                for c in range(4):
                    T.gemm(T.region(acc_s_cast[0, 0], 1, 64, 64), T.region(K_shared[c, 0, 0], 1, 1, 64, 128), T.region(acc_o[0, c * 128], 3, 64, 128), T.bool(False), T.bool(False), 64, 128, 64, 0, T.bool(False), 0, 0, c * 128)
            for i in T.parallel(64):
                for j in T.parallel(512):
                    acc_o[i, j] = acc_o[i, j] / logsum[i]
            T.copy(T.region(acc_o[0, 0], 1, 64, 512), T.region(O[0, 0], 2, 64, 512))