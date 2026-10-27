# after pass 55: tirx.AnnotateEntryFunc  (changed: True)
# from tvm.script import ir as I
# from tvm.script import tirx as T

@I.ir_module
class Module:
    @T.prim_func
    def main(Q_handle: T.handle, K_handle: T.handle, V_handle: T.handle, Output_handle: T.handle):
        T.func_attr({"target": T.target({"arch": "sm_90", "host": {"keys": ["cpu"], "kind": "c", "tag": ""}, "keys": ["cuda", "gpu"], "kind": "cuda", "max_num_threads": 1024, "tag": "", "thread_warp_size": 32}), "tirx.is_entry_func": True, "tl.has_tma": T.bool(True), "tl.smem_alignment_map": {"K_shared": 1024, "Q_shared": 1024, "V_shared": 1024}, "tl_tiled_ws_applied": 1})
        Q = T.match_buffer(Q_handle, (1, 8, 256, 64), "float16", strides=(131072, 16384, 64, 1))
        K = T.match_buffer(K_handle, (1, 8, 256, 64), "float16", strides=(131072, 16384, 64, 1))
        V = T.match_buffer(V_handle, (1, 8, 256, 64), "float16", strides=(131072, 16384, 64, 1))
        Output = T.match_buffer(Output_handle, (1, 8, 256, 64), "float16", strides=(131072, 16384, 64, 1))
        bx = T.launch_thread("blockIdx.x", 4)
        mbarrier = T.alloc_buffer((9,), "uint64", scope="shared.barrier")
        Q_shared = T.alloc_buffer((4096,), "float16", scope="shared.dyn")
        K_shared = T.alloc_buffer((8192,), "float16", scope="shared.dyn")
        V_shared = T.alloc_buffer((8192,), "float16", scope="shared.dyn")
        acc_o = T.alloc_buffer((32,), scope="local")
        logsum = T.alloc_buffer((2,), scope="local")
        scores_max = T.alloc_buffer((2,), scope="local")
        acc_s = T.alloc_buffer((32,), scope="local")
        scores_max_prev = T.alloc_buffer((2,), scope="local")
        scores_max_clear = T.alloc_buffer((2,), scope="local")
        scores_scale = T.alloc_buffer((2,), scope="local")
        scores_sum = T.alloc_buffer((2,), scope="local")
        acc_s_cast = T.alloc_buffer((32,), "float16", scope="local")
        by = T.launch_thread("blockIdx.y", 8)
        bz = T.launch_thread("blockIdx.z", 1)
        tx = T.launch_thread("threadIdx.x", 256)
        ty = T.launch_thread("threadIdx.y", 1)
        tz = T.launch_thread("threadIdx.z", 1)
        mbarrier_1 = T.decl_buffer((9,), "uint64", data=mbarrier.data, scope="shared.barrier")
        acc_o_1 = T.decl_buffer((32,), data=acc_o.data, scope="local")
        scores_max_1 = T.decl_buffer((2,), data=scores_max.data, scope="local")
        logsum_1 = T.decl_buffer((2,), data=logsum.data, scope="local")
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
            T.ptx_arrive_barrier_expect_tx(mbarrier_1[8], 8192)
            T.tma_load(T.create_tma_descriptor(6, 4, Q.data, 64, 256, 8, 1, 2, 128, 32768, 262144, 64, 64, 1, 1, 1, 1, 1, 1, 0, 3, 2, 0), mbarrier_1[8], T.tvm_access_ptr(T.type_annotation("float16"), Q_shared.data, 0, 4096, 2), 0, bx * 64, by, 0, 0)
        T.attr([128, 128], "kWarpSpecializationScope", 0)
        if tx < 128:
            for k in range(4):
                T.mbarrier_wait_parity(mbarrier_1[k % 2 + 4], T.bitwise_xor(k // 2, 1))
                if T.tl_shuffle_elect(128):
                    T.ptx_arrive_barrier_expect_tx(mbarrier_1[k % 2], 8192)
                    T.tma_load(T.create_tma_descriptor(6, 4, K.data, 64, 256, 8, 1, 2, 128, 32768, 262144, 64, 64, 1, 1, 1, 1, 1, 1, 0, 3, 2, 0), mbarrier_1[k % 2], T.tvm_access_ptr(T.type_annotation("float16"), K_shared.data, k % 2 * 4096, 4096, 2), 0, k * 64, by, 0, 0)
                T.mbarrier_wait_parity(mbarrier_1[k % 2 + 6], T.bitwise_xor(k // 2, 1))
                if T.tl_shuffle_elect(128):
                    T.ptx_arrive_barrier_expect_tx(mbarrier_1[k % 2 + 2], 8192)
                    T.tma_load(T.create_tma_descriptor(6, 4, V.data, 64, 256, 8, 1, 2, 128, 32768, 262144, 64, 64, 1, 1, 1, 1, 1, 1, 0, 3, 2, 0), mbarrier_1[k % 2 + 2], T.tvm_access_ptr(T.type_annotation("float16"), V_shared.data, k % 2 * 4096, 4096, 2), 0, k * 64, by, 0, 0)
        else:
            for i in T.unroll(8):
                acc_o_1[i * 4:i * 4 + 4] = T.Broadcast(T.float32(0.0), 4)
            logsum_1[0:2] = T.Broadcast(T.float32(0.0), 2)
            scores_max_1[0:2] = T.Broadcast(T.infinity("float32") * T.float32(-1.0), 2)
            for k in range(4):
                acc_s_1 = T.decl_buffer((32,), data=acc_s.data, scope="local")
                acc_s_cast_1 = T.decl_buffer((32,), "float16", data=acc_s_cast.data, scope="local")
                scores_max_prev_1 = T.decl_buffer((2,), data=scores_max_prev.data, scope="local")
                scores_scale_1 = T.decl_buffer((2,), data=scores_scale.data, scope="local")
                scores_sum_1 = T.decl_buffer((2,), data=scores_sum.data, scope="local")
                for i in T.unroll(8):
                    acc_s_1[i * 4:i * 4 + 4] = T.Broadcast(T.float32(0.0), 4)
                if k == 0:
                    T.mbarrier_wait_parity(mbarrier_1[8], 0)
                T.mbarrier_wait_parity(mbarrier_1[k % 2], k // 2)
                with T.attr(0, "lexical_alloc_scope", 1):
                    desc_a = T.alloc_buffer((1,), "uint64", scope="local.descriptor.wgmma")
                    desc_b = T.alloc_buffer((1,), "uint64", scope="local.descriptor.wgmma")
                    desc_a_1 = T.decl_buffer((1,), "uint64", data=desc_a.data, scope="local.descriptor.wgmma")
                    desc_b_1 = T.decl_buffer((1,), "uint64", data=desc_b.data, scope="local.descriptor.wgmma")
                    T.initialize_wgmma_descriptor(desc_a_1[0], T.tvm_access_ptr(T.type_annotation("float16"), Q_shared.data, 0, 4096, 1), 1, 1, 64)
                    T.initialize_wgmma_descriptor(desc_b_1[0], T.tvm_access_ptr(T.type_annotation("float16"), K_shared.data, 0, 8192, 1), 1, 1, 64)
                    T.increase_descriptor_offset(desc_b_1[0], k % 2 * 8192)
                    T.warpgroup_fence_operand("float32", acc_s.data, 0, 32)
                    T.warpgroup_arrive()
                    for ki in T.unroll(4):
                        T.ptx_wgmma_ss("m64n64k16", T.bool(True), T.bool(True), "fp16", "fp16", "fp32", desc_a.data, T.shift_right(ki * 32, 4), desc_b.data, T.shift_right(ki * 32, 4), acc_s.data, 0, 1, 1, 1)
                    T.warpgroup_commit_batch()
                    T.warpgroup_wait(0)
                    T.warpgroup_fence_operand("float32", acc_s.data, 0, 32)
                T.ptx_arrive_barrier(mbarrier_1[k % 2 + 4])
                scores_max_prev_1[0:2] = scores_max_1[0:2]
                scores_max_1[0:2] = T.Broadcast(T.infinity("float32") * T.float32(-1.0), 2)
                for i in T.unroll(2):
                    scores_max_clear[i] = T.float32("-inf")
                    for rv in T.unroll(16):
                        scores_max_clear[i] = T.max(scores_max_clear[i], acc_s_1[rv % 8 * 4 + i * 2 + rv // 8])
                    scores_max_clear[i] = T.call_extern("float32", "tl::AllReduce<tl::MaxOp, 4, 1, 128, tl::NamedBarrier<128>>::run", scores_max_clear[i])
                    scores_max_1[i] = T.max(scores_max_1[i], scores_max_clear[i])
                scores_max_1[0:2] = T.max(scores_max_1[0:2], scores_max_prev_1[0:2])
                scores_scale_1[0:2] = T.exp2(scores_max_prev_1[0:2] * T.Broadcast(T.float32(0.18033688), 2) - scores_max_1[0:2] * T.Broadcast(T.float32(0.18033688), 2))
                for i in T.unroll(16):
                    acc_s_1[i * 2:i * 2 + 2] = T.exp2(acc_s_1[i * 2:i * 2 + 2] * T.Broadcast(T.float32(0.18033688), 2) - T.Broadcast(scores_max_1[i % 2] * T.float32(0.18033688), 2))
                for i in T.unroll(2):
                    scores_sum_1[i] = T.float32(0.0)
                    for rv in T.unroll(16):
                        scores_sum_1[i] = scores_sum_1[i] + acc_s_1[rv % 8 * 4 + i * 2 + rv // 8]
                    scores_sum_1[i] = T.call_extern("float32", "tl::AllReduce<tl::SumOp, 4, 1, 128, tl::NamedBarrier<128>>::run", scores_sum_1[i])
                logsum_1[0:2] = logsum_1[0:2] * scores_scale_1[0:2] + scores_sum_1[0:2]
                for i in T.unroll(8):
                    acc_s_cast_1[i * 4:i * 4 + 4] = T.Cast("float16x4", acc_s_1[i * 4:i * 4 + 4])
                for i in T.unroll(16):
                    acc_o_1[i * 2:i * 2 + 2] = acc_o_1[i * 2:i * 2 + 2] * T.Broadcast(scores_scale_1[i % 2], 2)
                T.mbarrier_wait_parity(mbarrier_1[k % 2 + 2], k // 2)
                with T.attr(0, "lexical_alloc_scope", 1):
                    desc_b = T.alloc_buffer((1,), "uint64", scope="local.descriptor.wgmma")
                    desc_b_1 = T.decl_buffer((1,), "uint64", data=desc_b.data, scope="local.descriptor.wgmma")
                    T.initialize_wgmma_descriptor(desc_b_1[0], T.tvm_access_ptr(T.type_annotation("float16"), V_shared.data, 0, 8192, 1), 1, 0, 64)
                    T.increase_descriptor_offset(desc_b_1[0], k % 2 * 8192)
                    T.warpgroup_fence_operand("float16", acc_s_cast.data, 0, 16)
                    T.warpgroup_fence_operand("float32", acc_o.data, 0, 32)
                    T.warpgroup_arrive()
                    for ki in T.unroll(4):
                        T.ptx_wgmma_rs("m64n64k16", T.bool(False), "fp16", "fp16", "fp32", acc_s_cast.data, ki * 8, desc_b.data, T.shift_right(ki * 2048, 4), acc_o.data, 0, 1, 1, 1)
                    T.warpgroup_commit_batch()
                    T.warpgroup_wait(0)
                    T.warpgroup_fence_operand("float32", acc_o.data, 0, 32)
                    T.warpgroup_fence_operand("float16", acc_s_cast.data, 0, 16)
                T.ptx_arrive_barrier(mbarrier_1[k % 2 + 6])
            for i in T.unroll(16):
                acc_o_1[i * 2:i * 2 + 2] = acc_o_1[i * 2:i * 2 + 2] / T.Broadcast(logsum_1[i % 2], 2)
            for i in T.unroll(16):
                T.attr(0, "lexical_alloc_scope", 1)
                Output_local_cast = T.alloc_buffer((2,), "float16", scope="local")
                Output_local_cast[0:2] = T.Cast("float16x2", acc_o_1[i * 2:i * 2 + 2])
                Output_1 = T.Buffer((131072,), "float16", data=Output.data)
                Output_1[by * 16384 + bx * 4096 + tx // 32 * 1024 + i % 2 * 512 + tx % 32 // 4 * 64 + i // 2 * 8 + tx % 4 * 2 - 4096:by * 16384 + bx * 4096 + tx // 32 * 1024 + i % 2 * 512 + tx % 32 // 4 * 64 + i // 2 * 8 + tx % 4 * 2 - 4096 + 2] = Output_local_cast[0:2]