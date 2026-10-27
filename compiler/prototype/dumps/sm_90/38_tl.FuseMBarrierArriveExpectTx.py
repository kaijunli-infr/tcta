# after pass 38: tl.FuseMBarrierArriveExpectTx  (changed: True)
# from tvm.script import ir as I
# from tvm.script import tirx as T

@I.ir_module
class Module:
    @T.prim_func
    def main(Q_handle: T.handle, K_handle: T.handle, V_handle: T.handle, Output_handle: T.handle):
        T.func_attr({"target": T.target({"arch": "sm_90", "host": {"keys": ["cpu"], "kind": "c", "tag": ""}, "keys": ["cuda", "gpu"], "kind": "cuda", "max_num_threads": 1024, "tag": "", "thread_warp_size": 32}), "tl.has_tma": T.bool(True), "tl.smem_alignment_map": {"K_shared": 1024, "Q_shared": 1024, "V_shared": 1024}, "tl_tiled_ws_applied": 1})
        Q = T.match_buffer(Q_handle, (1, 8, 256, 64), "float16", strides=(131072, 16384, 64, 1))
        K = T.match_buffer(K_handle, (1, 8, 256, 64), "float16", strides=(131072, 16384, 64, 1))
        V = T.match_buffer(V_handle, (1, 8, 256, 64), "float16", strides=(131072, 16384, 64, 1))
        Output = T.match_buffer(Output_handle, (1, 8, 256, 64), "float16", strides=(131072, 16384, 64, 1))
        with T.sblock("root"):
            T.reads()
            T.writes()
            Q_shared = T.Buffer((1, 8, 512), "float16", scope="shared.dyn")
            K_shared = T.Buffer((2, 1, 8, 512), "float16", scope="shared.dyn")
            scores_max_prev = T.Buffer((2,), scope="local")
            scores_max = T.Buffer((2,), scope="local")
            scores_sum = T.Buffer((2,), scope="local")
            acc_s = T.Buffer((32,), scope="local")
            acc_s_cast = T.Buffer((32,), "float16", scope="local")
            scores_scale = T.Buffer((2,), scope="local")
            V_shared = T.Buffer((2, 1, 8, 512), "float16", scope="shared.dyn")
            logsum = T.Buffer((2,), scope="local")
            acc_o = T.Buffer((32,), scope="local")
            T.sblock_attr({"layout_map": {Q_shared: metadata["tl.Layout"][0], K_shared: metadata["tl.Layout"][1], scores_max_prev: metadata["tl.Fragment"][0], scores_max: metadata["tl.Fragment"][1], scores_sum: metadata["tl.Fragment"][2], acc_s: metadata["tl.Fragment"][3], acc_s_cast: metadata["tl.Fragment"][4], scores_scale: metadata["tl.Fragment"][5], V_shared: metadata["tl.Layout"][2], logsum: metadata["tl.Fragment"][6], acc_o: metadata["tl.Fragment"][7]}})
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
                T.sblock_attr({"barrier_init": {mbarrier: [1, 1, 1, 1, 128, 128, 128, 128, 1]}, "layout_map": {Q_shared: metadata["tl.Layout"][0], K_shared: metadata["tl.Layout"][1], scores_max_prev: metadata["tl.Fragment"][0], scores_max: metadata["tl.Fragment"][1], scores_sum: metadata["tl.Fragment"][2], acc_s: metadata["tl.Fragment"][3], acc_s_cast: metadata["tl.Fragment"][4], scores_scale: metadata["tl.Fragment"][5], V_shared: metadata["tl.Layout"][2], logsum: metadata["tl.Fragment"][6], acc_o: metadata["tl.Fragment"][7]}, "tl.launch_threads": [128, 1, 1]})
                mbarrier_1 = T.sblock_alloc_buffer((9,), "uint64", data=mbarrier, scope="shared.barrier")
                Q_shared = T.sblock_alloc_buffer((1, 8, 512), "float16", data=Q_shared.data, scope="shared.dyn")
                K_shared = T.sblock_alloc_buffer((2, 1, 8, 512), "float16", data=K_shared.data, scope="shared.dyn")
                V_shared = T.sblock_alloc_buffer((2, 1, 8, 512), "float16", data=V_shared.data, scope="shared.dyn")
                acc_o = T.sblock_alloc_buffer((32,), data=acc_o.data, scope="local")
                scores_max = T.sblock_alloc_buffer((2,), data=scores_max.data, scope="local")
                logsum = T.sblock_alloc_buffer((2,), data=logsum.data, scope="local")
                if T.tl_shuffle_elect(0):
                    T.ptx_init_barrier_thread_count(mbarrier_1[0], 1)
                    T.ptx_init_barrier_thread_count(mbarrier_1[1], 1)
                    T.ptx_init_barrier_thread_count(mbarrier_1[2], 1)
                    T.ptx_init_barrier_thread_count(mbarrier_1[3], 1)
                    T.ptx_init_barrier_thread_count(mbarrier_1[4], 128)
                    T.ptx_init_barrier_thread_count(mbarrier_1[5], 128)
                    T.ptx_init_barrier_thread_count(mbarrier_1[6], 128)
                    T.ptx_init_barrier_thread_count(mbarrier_1[7], 128)
                    T.ptx_init_barrier_thread_count(mbarrier_1[8], 1)
                T.ptx_fence_barrier_init()
                T.tvm_storage_sync("shared")
                if T.tl_shuffle_elect(256):
                    T.ptx_arrive_barrier_expect_tx(mbarrier_1[8], T.int64(8192))
                    T.tma_load(T.create_tma_descriptor(6, 4, Q.data, 64, 256, 8, 1, T.int64(2), T.int64(128), T.int64(32768), T.int64(262144), 64, 64, 1, 1, 1, 1, 1, 1, 0, 3, 2, 0), mbarrier_1[8], T.tvm_access_ptr(T.type_annotation("float16"), Q_shared.data, 0, 4096, 2), 0, bx * 64, by, 0, 0)
                T.attr([128, 128], "kWarpSpecializationScope", 0)
                if tx < 128:
                    for k in range(4):
                        T.mbarrier_wait_parity(mbarrier_1[k % 2 + 4], T.bitwise_xor(k // 2, 1))
                        if T.tl_shuffle_elect(128):
                            T.ptx_arrive_barrier_expect_tx(mbarrier_1[k % 2], T.int64(8192))
                            T.tma_load(T.create_tma_descriptor(6, 4, K.data, 64, 256, 8, 1, T.int64(2), T.int64(128), T.int64(32768), T.int64(262144), 64, 64, 1, 1, 1, 1, 1, 1, 0, 3, 2, 0), mbarrier_1[k % 2], T.tvm_access_ptr(T.type_annotation("float16"), K_shared.data, k % 2 * 4096, 4096, 2), 0, k * 64, by, 0, 0)
                        T.mbarrier_wait_parity(mbarrier_1[k % 2 + 6], T.bitwise_xor(k // 2, 1))
                        if T.tl_shuffle_elect(128):
                            T.ptx_arrive_barrier_expect_tx(mbarrier_1[k % 2 + 2], T.int64(8192))
                            T.tma_load(T.create_tma_descriptor(6, 4, V.data, 64, 256, 8, 1, T.int64(2), T.int64(128), T.int64(32768), T.int64(262144), 64, 64, 1, 1, 1, 1, 1, 1, 0, 3, 2, 0), mbarrier_1[k % 2 + 2], T.tvm_access_ptr(T.type_annotation("float16"), V_shared.data, k % 2 * 4096, 4096, 2), 0, k * 64, by, 0, 0)
                else:
                    for i in T.unroll(8):
                        for vec in T.vectorized(4):
                            acc_o[i * 4 + vec] = T.float32(0.0)
                    for i in T.vectorized(2):
                        logsum[i] = T.float32(0.0)
                    for i in T.vectorized(2):
                        scores_max[i] = T.infinity("float32") * T.float32(-1.0)
                    for k in range(4):
                        with T.sblock(""):
                            T.reads(mbarrier_1[k % 2:k % 2 + (9 - k % 2)], Q_shared[0, 0:8, 0:512], K_shared[0:2, 0, 0:8, 0:512], scores_max[0:2], logsum[0:2], V_shared[0:2, 0, 0:8, 0:512], acc_o[0:32])
                            T.writes(scores_max[0:2], logsum[0:2], acc_o[0:32])
                            acc_s = T.sblock_alloc_buffer((32,), data=acc_s.data, scope="local")
                            acc_s_cast = T.sblock_alloc_buffer((32,), "float16", data=acc_s_cast.data, scope="local")
                            scores_max_prev = T.sblock_alloc_buffer((2,), data=scores_max_prev.data, scope="local")
                            scores_scale = T.sblock_alloc_buffer((2,), data=scores_scale.data, scope="local")
                            scores_sum = T.sblock_alloc_buffer((2,), data=scores_sum.data, scope="local")
                            for i in T.unroll(8):
                                for vec in T.vectorized(4):
                                    acc_s[i * 4 + vec] = T.float32(0.0)
                            if k == 0:
                                T.mbarrier_wait_parity(mbarrier_1[8], 0)
                            T.mbarrier_wait_parity(mbarrier_1[k % 2], k // 2)
                            with T.sblock("_gemm_ssr"):
                                T.reads()
                                T.writes()
                                T.sblock_attr({"lexical_alloc_scope": 1})
                                desc_a = T.sblock_alloc_buffer((1,), "uint64", scope="local.descriptor.wgmma")
                                desc_b = T.sblock_alloc_buffer((1,), "uint64", scope="local.descriptor.wgmma")
                                T.initialize_wgmma_descriptor(desc_a[0], T.tvm_access_ptr(T.type_annotation("float16"), Q_shared.data, 0, 4096, 1), 1, 1, 64)
                                T.initialize_wgmma_descriptor(desc_b[0], T.tvm_access_ptr(T.type_annotation("float16"), K_shared.data, 0, 8192, 1), 1, 1, 64)
                                T.increase_descriptor_offset(desc_b[0], k % 2 * 8192)
                                T.warpgroup_fence_operand("float32", acc_s.data, 0, 32)
                                T.warpgroup_arrive()
                                for warp_j in T.unroll(1):
                                    for i in T.unroll(1):
                                        for ki in T.unroll(4):
                                            T.ptx_wgmma_ss("m64n64k16", T.bool(True), T.bool(True), "fp16", "fp16", "fp32", desc_a.data, T.shift_right(ki * 32, 4), desc_b.data, T.shift_right(ki * 32, 4), acc_s.data, 0, 1, 1, 1)
                                T.warpgroup_commit_batch()
                                T.warpgroup_wait(0)
                                T.warpgroup_fence_operand("float32", acc_s.data, 0, 32)
                            T.ptx_arrive_barrier(mbarrier_1[k % 2 + 4])
                            for i in T.vectorized(2):
                                scores_max_prev[i] = scores_max[i]
                            for i in T.vectorized(2):
                                scores_max[i] = T.infinity("float32") * T.float32(-1.0)
                            scores_max_clear = T.alloc_buffer((2,), scope="local")
                            for i in T.unroll(2):
                                scores_max_clear[i] = T.float32("-inf")
                                for rv in T.unroll(16):
                                    scores_max_clear[i] = T.max(scores_max_clear[i], acc_s[rv % 8 * 4 + i * 2 + rv // 8])
                                scores_max_clear[i] = T.call_extern("float32", "tl::AllReduce<tl::MaxOp, 4, 1, 128, tl::NamedBarrier<128>>::run", scores_max_clear[i])
                                scores_max[i] = T.max(scores_max[i], scores_max_clear[i])
                            for i in T.vectorized(2):
                                scores_max[i] = T.max(scores_max[i], scores_max_prev[i])
                            for i in T.vectorized(2):
                                scores_scale[i] = T.exp2(scores_max_prev[i] * T.float32(0.18033688) - scores_max[i] * T.float32(0.18033688))
                            for i in T.unroll(16):
                                for vec in T.vectorized(2):
                                    acc_s[i * 2 + vec] = T.exp2(acc_s[i * 2 + vec] * T.float32(0.18033688) - scores_max[i % 2] * T.float32(0.18033688))
                            for i in T.unroll(2):
                                scores_sum[i] = T.float32(0.0)
                                for rv in T.unroll(16):
                                    scores_sum[i] = scores_sum[i] + acc_s[rv % 8 * 4 + i * 2 + rv // 8]
                                scores_sum[i] = T.call_extern("float32", "tl::AllReduce<tl::SumOp, 4, 1, 128, tl::NamedBarrier<128>>::run", scores_sum[i])
                            for i in T.vectorized(2):
                                logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
                            for i in T.unroll(8):
                                for vec in T.vectorized(4):
                                    acc_s_cast[i * 4 + vec] = T.Cast("float16", acc_s[i * 4 + vec])
                            for i in T.unroll(16):
                                for vec in T.vectorized(2):
                                    acc_o[i * 2 + vec] = acc_o[i * 2 + vec] * scores_scale[i % 2]
                            T.mbarrier_wait_parity(mbarrier_1[k % 2 + 2], k // 2)
                            with T.sblock("_gemm_rsr"):
                                T.reads()
                                T.writes()
                                T.sblock_attr({"lexical_alloc_scope": 1})
                                desc_b = T.sblock_alloc_buffer((1,), "uint64", scope="local.descriptor.wgmma")
                                T.initialize_wgmma_descriptor(desc_b[0], T.tvm_access_ptr(T.type_annotation("float16"), V_shared.data, 0, 8192, 1), 1, 0, 64)
                                T.increase_descriptor_offset(desc_b[0], k % 2 * 8192)
                                T.warpgroup_fence_operand("float16", acc_s_cast.data, 0, 16)
                                T.warpgroup_fence_operand("float32", acc_o.data, 0, 32)
                                T.warpgroup_arrive()
                                for warp_j in T.unroll(1):
                                    for i in T.unroll(1):
                                        for ki in T.unroll(4):
                                            T.ptx_wgmma_rs("m64n64k16", T.bool(False), "fp16", "fp16", "fp32", acc_s_cast.data, ki * 8, desc_b.data, T.shift_right(ki * 2048, 4), acc_o.data, 0, 1, 1, 1)
                                T.warpgroup_commit_batch()
                                T.warpgroup_wait(0)
                                T.warpgroup_fence_operand("float32", acc_o.data, 0, 32)
                                T.warpgroup_fence_operand("float16", acc_s_cast.data, 0, 16)
                            T.ptx_arrive_barrier(mbarrier_1[k % 2 + 6])
                    for i in T.unroll(16):
                        for vec in T.vectorized(2):
                            acc_o[i * 2 + vec] = acc_o[i * 2 + vec] / logsum[i % 2]
                    for i in T.unroll(16):
                        with T.sblock("decoupled_cast"):
                            T.reads()
                            T.writes()
                            T.sblock_attr({"lexical_alloc_scope": 1})
                            Output_local_cast = T.alloc_buffer((2,), "float16", scope="local")
                            for vec in T.vectorized(2):
                                Output_local_cast[vec] = T.Cast("float16", acc_o[i * 2 + vec])
                            for vec_copy in T.vectorized(2):
                                Output[0, by, bx * 64 + tx // 32 * 16 + i % 2 * 8 + tx % 32 // 4 - 64, i // 2 * 8 + tx % 4 * 2 + vec_copy] = Output_local_cast[vec_copy]

# Metadata omitted. Use show_meta=True in script() method to show it.