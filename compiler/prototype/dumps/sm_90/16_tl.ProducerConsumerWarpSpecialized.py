# after pass 16: tl.ProducerConsumerWarpSpecialized  (changed: True)
# from tvm.script import ir as I
# from tvm.script import tirx as T

@I.ir_module
class Module:
    @T.prim_func
    def main(Q_handle: T.handle, K_handle: T.handle, V_handle: T.handle, Output_handle: T.handle):
        T.func_attr({"target": T.target({"arch": "sm_90", "host": {"keys": ["cpu"], "kind": "c", "tag": ""}, "keys": ["cuda", "gpu"], "kind": "cuda", "max_num_threads": 1024, "tag": "", "thread_warp_size": 32}), "tl_tiled_ws_applied": 1})
        Q = T.match_buffer(Q_handle, (1, 8, 256, 64), "float16", strides=(131072, 16384, 64, 1))
        K = T.match_buffer(K_handle, (1, 8, 256, 64), "float16", strides=(131072, 16384, 64, 1))
        V = T.match_buffer(V_handle, (1, 8, 256, 64), "float16", strides=(131072, 16384, 64, 1))
        Output = T.match_buffer(Output_handle, (1, 8, 256, 64), "float16", strides=(131072, 16384, 64, 1))
        # with T.sblock("root"):
        bx = T.launch_thread("blockIdx.x", 4)
        by = T.launch_thread("blockIdx.y", 8)
        bz = T.launch_thread("blockIdx.z", 1)
        tx = T.launch_thread("threadIdx.x", 256)
        ty = T.launch_thread("threadIdx.y", 1)
        tz = T.launch_thread("threadIdx.z", 1)
        with T.sblock("tilelang_root"):
            T.reads()
            T.writes()
            mbarrier = T.handle("uint64", "shared.barrier")
            T.sblock_attr({"barrier_init": {mbarrier: [1, 1, 1, 1, 128, 128, 128, 128, 1]}, "tl.launch_threads": [128, 1, 1]})
            Q_shared = T.sblock_alloc_buffer((64, 64), "float16", scope="shared.dyn")
            K_shared = T.sblock_alloc_buffer((2, 64, 64), "float16", scope="shared.dyn")
            V_shared = T.sblock_alloc_buffer((2, 64, 64), "float16", scope="shared.dyn")
            acc_s = T.sblock_alloc_buffer((64, 64), scope="local.fragment")
            acc_s_cast = T.sblock_alloc_buffer((64, 64), "float16", scope="local.fragment")
            acc_o = T.sblock_alloc_buffer((64, 64), scope="local.fragment")
            scores_max = T.sblock_alloc_buffer((64,), scope="local.fragment")
            scores_max_prev = T.sblock_alloc_buffer((64,), scope="local.fragment")
            scores_scale = T.sblock_alloc_buffer((64,), scope="local.fragment")
            scores_sum = T.sblock_alloc_buffer((64,), scope="local.fragment")
            logsum = T.sblock_alloc_buffer((64,), scope="local.fragment")
            mbarrier_1 = T.sblock_alloc_buffer((9,), "uint64", data=mbarrier, scope="shared.barrier")
            T.tma_copy(T.region(Q[0, by, bx * 64, 0], 1, 1, 1, 64, 64), T.region(Q_shared[0, 0], 2, 64, 64), barrier=mbarrier_1[8], is_tma_copy=1, emit_arrive=1)
            T.attr([128, 128], "kWarpSpecializationScope", 0)
            if tx < 128:
                for k in range(4):
                    T.mbarrier_wait_parity(mbarrier_1[4 + k % 2], T.bitwise_xor(k // 2 % 2, 1))
                    T.tma_copy(T.region(K[0, by, k * 64, 0], 1, 1, 1, 64, 64), T.region(K_shared[k % 2, 0, 0], 2, 1, 64, 64), barrier=mbarrier_1[k % 2], is_tma_copy=1, emit_arrive=1, tl.pipeline_mbar_phase_expr=k // 2 % 2)
                    T.mbarrier_wait_parity(mbarrier_1[6 + k % 2], T.bitwise_xor(k // 2 % 2, 1))
                    T.tma_copy(T.region(V[0, by, k * 64, 0], 1, 1, 1, 64, 64), T.region(V_shared[k % 2, 0, 0], 2, 1, 64, 64), barrier=mbarrier_1[2 + k % 2], is_tma_copy=1, emit_arrive=1, tl.pipeline_mbar_phase_expr=k // 2 % 2)
            else:
                T.fill(T.region(acc_o[0, 0], 2, 64, 64), 0)
                T.fill(T.region(logsum[0], 2, 64), 0)
                T.fill(T.region(scores_max[0], 2, 64), T.infinity("float32") * T.float32(-1.0))
                for k in range(4):
                    T.fill(T.region(acc_s[0, 0], 2, 64, 64), 0)
                    if k == 0:
                        T.mbarrier_wait_parity(mbarrier_1[8], 0)
                    T.mbarrier_wait_parity(mbarrier_1[k % 2], k // 2 % 2)
                    T.gemm(T.region(Q_shared[0, 0], 1, 64, 64), T.region(K_shared[k % 2, 0, 0], 1, 1, 64, 64), T.region(acc_s[0, 0], 3, 64, 64), T.bool(False), T.bool(True), 64, 64, 64, 1, T.bool(False), 0, 0, 0, tl.pipeline_mbar_phase_expr=k // 2 % 2)
                    T.ptx_arrive_barrier(mbarrier_1[4 + k % 2])
                    T.copy(T.region(scores_max[0], 1, 64), T.region(scores_max_prev[0], 2, 64), tl.pipeline_mbar_phase_expr=k // 2 % 2)
                    T.fill(T.region(scores_max[0], 2, 64), T.infinity("float32") * T.float32(-1.0))
                    T.reduce(T.region(acc_s[0, 0], 1, 64, 64), T.region(scores_max[0], 2, 64), "max", 1, T.bool(False))
                    for i in T.parallel(64):
                        scores_max[i] = T.max(scores_max[i], scores_max_prev[i])
                    for i in T.parallel(64):
                        scores_scale[i] = T.exp2(scores_max_prev[i] * T.float32(0.18033688) - scores_max[i] * T.float32(0.18033688))
                    for i in T.parallel(64):
                        for j in T.parallel(64):
                            acc_s[i, j] = T.exp2(acc_s[i, j] * T.float32(0.18033688) - scores_max[i] * T.float32(0.18033688))
                    T.reduce(T.region(acc_s[0, 0], 1, 64, 64), T.region(scores_sum[0], 2, 64), "sum", 1, T.bool(True))
                    for i in T.parallel(64):
                        logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
                    T.copy(T.region(acc_s[0, 0], 1, 64, 64), T.region(acc_s_cast[0, 0], 2, 64, 64), tl.pipeline_mbar_phase_expr=k // 2 % 2)
                    for i in T.parallel(64):
                        for j in T.parallel(64):
                            acc_o[i, j] = acc_o[i, j] * scores_scale[i]
                    T.mbarrier_wait_parity(mbarrier_1[2 + k % 2], k // 2 % 2)
                    T.gemm(T.region(acc_s_cast[0, 0], 1, 64, 64), T.region(V_shared[k % 2, 0, 0], 1, 1, 64, 64), T.region(acc_o[0, 0], 3, 64, 64), T.bool(False), T.bool(False), 64, 64, 64, 1, T.bool(False), 0, 0, 0, tl.pipeline_mbar_phase_expr=k // 2 % 2)
                    T.ptx_arrive_barrier(mbarrier_1[6 + k % 2])
                for i in T.parallel(64):
                    for j in T.parallel(64):
                        acc_o[i, j] = acc_o[i, j] / logsum[i]
                T.copy(T.region(acc_o[0, 0], 1, 64, 64), T.region(Output[0, by, bx * 64, 0], 2, 1, 1, 64, 64))