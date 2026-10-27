# after pass 68: tl.LowerDeviceKernelLaunch  (changed: True)
# from tvm.script import ir as I
# from tvm.script import tirx as T

@I.ir_module
class Module:
    @T.prim_func
    def main_kernel(K: T.handle("float16", "global"), Output: T.handle("float16", "global"), Q: T.handle("float16", "global"), V: T.handle("float16", "global")):
        T.func_attr({"calling_conv": 2, "dyn_shared_memory_buf": 40960, "target": T.target({"arch": "sm_80", "keys": ["cuda", "gpu"], "kind": "cuda", "max_num_threads": 1024, "tag": "", "thread_warp_size": 32}), "thread_extent": {"blockIdx.x": 4, "blockIdx.y": 8, "blockIdx.z": 1, "threadIdx.x": 128, "threadIdx.y": 1, "threadIdx.z": 1}, "tirx.is_global_func": T.bool(True), "tirx.kernel_launch_params": ["blockIdx.x", "blockIdx.y", "blockIdx.z", "threadIdx.x", "threadIdx.y", "threadIdx.z", "tirx.use_dyn_shared_memory"], "tirx.noalias": True, "tl.non_restrict_params": [], "tl.readonly_param_indices": [0, 2, 3]})
        Output_1 = T.decl_buffer((131072,), "float16", data=Output)
        Q_1 = T.decl_buffer((131072,), "float16", data=Q)
        bx = T.launch_thread("blockIdx.x", 4)
        buf_dyn_shmem = T.alloc_buffer((40960,), "uint8", scope="shared.dyn")
        Q_shared: T.handle("float16", "shared.dyn") = T.handle_add_byte_offset(buf_dyn_shmem.data, 0)
        K_shared: T.handle("float16", "shared.dyn") = T.handle_add_byte_offset(buf_dyn_shmem.data, 8192)
        V_shared: T.handle("float16", "shared.dyn") = T.handle_add_byte_offset(buf_dyn_shmem.data, 24576)
        acc_o = T.alloc_buffer((32,), scope="local")
        logsum = T.alloc_buffer((2,), scope="local")
        scores_max = T.alloc_buffer((2,), scope="local")
        acc_s = T.alloc_buffer((32,), scope="local")
        scores_max_prev = T.alloc_buffer((2,), scope="local")
        scores_scale = T.alloc_buffer((2,), scope="local")
        scores_sum = T.alloc_buffer((2,), scope="local")
        acc_s_cast = T.alloc_buffer((32,), "float16", scope="local")
        scores_max_clear = T.alloc_buffer((2,), scope="local")
        scores_max_clear_1 = T.alloc_buffer((2,), scope="local")
        scores_max_clear_2 = T.alloc_buffer((2,), scope="local")
        by = T.launch_thread("blockIdx.y", 8)
        bz = T.launch_thread("blockIdx.z", 1)
        thread_binding = T.launch_thread("threadIdx.x", 128)
        ty = T.launch_thread("threadIdx.y", 1)
        tz = T.launch_thread("threadIdx.z", 1)
        acc_o_1 = T.decl_buffer((32,), data=acc_o.data, scope="local")
        logsum_1 = T.decl_buffer((2,), data=logsum.data, scope="local")
        Q_shared_1 = T.decl_buffer((4096,), "float16", data=Q_shared, scope="shared.dyn")
        acc_o_2 = T.decl_buffer((32,), data=acc_o.data, scope="local")
        logsum_2 = T.decl_buffer((2,), data=logsum.data, scope="local")
        scores_max_1 = T.decl_buffer((2,), data=scores_max.data, scope="local")
        acc_s_1 = T.decl_buffer((32,), data=acc_s.data, scope="local")
        scores_max_prev_1 = T.decl_buffer((2,), data=scores_max_prev.data, scope="local")
        scores_max_2 = T.decl_buffer((2,), data=scores_max.data, scope="local")
        scores_max_3 = T.decl_buffer((2,), data=scores_max.data, scope="local")
        acc_s_2 = T.decl_buffer((32,), data=acc_s.data, scope="local")
        scores_max_4 = T.decl_buffer((2,), data=scores_max.data, scope="local")
        scores_max_5 = T.decl_buffer((2,), data=scores_max.data, scope="local")
        scores_max_prev_2 = T.decl_buffer((2,), data=scores_max_prev.data, scope="local")
        scores_scale_1 = T.decl_buffer((2,), data=scores_scale.data, scope="local")
        scores_max_prev_3 = T.decl_buffer((2,), data=scores_max_prev.data, scope="local")
        scores_max_6 = T.decl_buffer((2,), data=scores_max.data, scope="local")
        acc_s_3 = T.decl_buffer((32,), data=acc_s.data, scope="local")
        scores_max_7 = T.decl_buffer((2,), data=scores_max.data, scope="local")
        scores_sum_1 = T.decl_buffer((2,), data=scores_sum.data, scope="local")
        acc_s_4 = T.decl_buffer((32,), data=acc_s.data, scope="local")
        logsum_3 = T.decl_buffer((2,), data=logsum.data, scope="local")
        scores_scale_2 = T.decl_buffer((2,), data=scores_scale.data, scope="local")
        scores_sum_2 = T.decl_buffer((2,), data=scores_sum.data, scope="local")
        acc_s_cast_1 = T.decl_buffer((32,), "float16", data=acc_s_cast.data, scope="local")
        acc_s_5 = T.decl_buffer((32,), data=acc_s.data, scope="local")
        acc_o_3 = T.decl_buffer((32,), data=acc_o.data, scope="local")
        scores_scale_3 = T.decl_buffer((2,), data=scores_scale.data, scope="local")
        acc_s_6 = T.decl_buffer((32,), data=acc_s.data, scope="local")
        scores_max_prev_4 = T.decl_buffer((2,), data=scores_max_prev.data, scope="local")
        scores_max_8 = T.decl_buffer((2,), data=scores_max.data, scope="local")
        scores_max_9 = T.decl_buffer((2,), data=scores_max.data, scope="local")
        acc_s_7 = T.decl_buffer((32,), data=acc_s.data, scope="local")
        scores_max_10 = T.decl_buffer((2,), data=scores_max.data, scope="local")
        scores_max_11 = T.decl_buffer((2,), data=scores_max.data, scope="local")
        scores_max_prev_5 = T.decl_buffer((2,), data=scores_max_prev.data, scope="local")
        scores_scale_4 = T.decl_buffer((2,), data=scores_scale.data, scope="local")
        scores_max_prev_6 = T.decl_buffer((2,), data=scores_max_prev.data, scope="local")
        scores_max_12 = T.decl_buffer((2,), data=scores_max.data, scope="local")
        acc_s_8 = T.decl_buffer((32,), data=acc_s.data, scope="local")
        scores_max_13 = T.decl_buffer((2,), data=scores_max.data, scope="local")
        scores_sum_3 = T.decl_buffer((2,), data=scores_sum.data, scope="local")
        acc_s_9 = T.decl_buffer((32,), data=acc_s.data, scope="local")
        logsum_4 = T.decl_buffer((2,), data=logsum.data, scope="local")
        scores_scale_5 = T.decl_buffer((2,), data=scores_scale.data, scope="local")
        scores_sum_4 = T.decl_buffer((2,), data=scores_sum.data, scope="local")
        acc_s_cast_2 = T.decl_buffer((32,), "float16", data=acc_s_cast.data, scope="local")
        acc_s_10 = T.decl_buffer((32,), data=acc_s.data, scope="local")
        acc_o_4 = T.decl_buffer((32,), data=acc_o.data, scope="local")
        scores_scale_6 = T.decl_buffer((2,), data=scores_scale.data, scope="local")
        acc_s_11 = T.decl_buffer((32,), data=acc_s.data, scope="local")
        scores_max_prev_7 = T.decl_buffer((2,), data=scores_max_prev.data, scope="local")
        scores_max_14 = T.decl_buffer((2,), data=scores_max.data, scope="local")
        scores_max_15 = T.decl_buffer((2,), data=scores_max.data, scope="local")
        acc_s_12 = T.decl_buffer((32,), data=acc_s.data, scope="local")
        scores_max_16 = T.decl_buffer((2,), data=scores_max.data, scope="local")
        scores_max_17 = T.decl_buffer((2,), data=scores_max.data, scope="local")
        scores_max_prev_8 = T.decl_buffer((2,), data=scores_max_prev.data, scope="local")
        scores_scale_7 = T.decl_buffer((2,), data=scores_scale.data, scope="local")
        scores_max_prev_9 = T.decl_buffer((2,), data=scores_max_prev.data, scope="local")
        scores_max_18 = T.decl_buffer((2,), data=scores_max.data, scope="local")
        acc_s_13 = T.decl_buffer((32,), data=acc_s.data, scope="local")
        scores_max_19 = T.decl_buffer((2,), data=scores_max.data, scope="local")
        scores_sum_5 = T.decl_buffer((2,), data=scores_sum.data, scope="local")
        acc_s_14 = T.decl_buffer((32,), data=acc_s.data, scope="local")
        logsum_5 = T.decl_buffer((2,), data=logsum.data, scope="local")
        scores_scale_8 = T.decl_buffer((2,), data=scores_scale.data, scope="local")
        scores_sum_6 = T.decl_buffer((2,), data=scores_sum.data, scope="local")
        acc_s_cast_3 = T.decl_buffer((32,), "float16", data=acc_s_cast.data, scope="local")
        acc_s_15 = T.decl_buffer((32,), data=acc_s.data, scope="local")
        acc_o_5 = T.decl_buffer((32,), data=acc_o.data, scope="local")
        scores_scale_9 = T.decl_buffer((2,), data=scores_scale.data, scope="local")
        for i in T.unroll(4):
            Q_shared_1[i * 1024 + thread_binding // 8 * 64 + (thread_binding % 64 // 32 + thread_binding % 8 // 4) % 2 * 32 + (thread_binding % 32 // 16 + thread_binding % 4 // 2) % 2 * 16 + (thread_binding % 16 // 8 + thread_binding % 2) % 2 * 8:i * 1024 + thread_binding // 8 * 64 + (thread_binding % 64 // 32 + thread_binding % 8 // 4) % 2 * 32 + (thread_binding % 32 // 16 + thread_binding % 4 // 2) % 2 * 16 + (thread_binding % 16 // 8 + thread_binding % 2) % 2 * 8 + 8] = Q_1[by * 16384 + bx * 4096 + i * 1024 + thread_binding * 8:by * 16384 + bx * 4096 + i * 1024 + thread_binding * 8 + 8]
        for i in T.unroll(8):
            acc_o_2[i * 4:i * 4 + 4] = T.Broadcast(T.float32(0.0), 4)
        logsum_2[0:2] = T.Broadcast(T.float32(0.0), 2)
        scores_max_1[0:2] = T.Broadcast(T.infinity("float32") * T.float32(-1.0), 2)
        T.tvm_storage_sync("shared.dyn")
        for i in T.unroll(4):
            T.ptx_cp_async(T.tvm_access_ptr(T.type_annotation("float16"), K_shared, i * 1024 + thread_binding // 8 * 64 + (thread_binding % 64 // 32 + thread_binding % 8 // 4) % 2 * 32 + (thread_binding % 32 // 16 + thread_binding % 4 // 2) % 2 * 16 + (thread_binding % 16 // 8 + thread_binding % 2) % 2 * 8, 1, 2), T.tvm_access_ptr(T.type_annotation("float16"), K, by * 16384 + i * 1024 + thread_binding * 8, 1, 1), 8)
        T.ptx_commit_group()
        for i in T.unroll(4):
            T.ptx_cp_async(T.tvm_access_ptr(T.type_annotation("float16"), V_shared, i * 1024 + thread_binding // 8 * 64 + (thread_binding % 64 // 32 + thread_binding % 8 // 4) % 2 * 32 + (thread_binding % 32 // 16 + thread_binding % 4 // 2) % 2 * 16 + (thread_binding % 16 // 8 + thread_binding % 2) % 2 * 8, 1, 2), T.tvm_access_ptr(T.type_annotation("float16"), V, by * 16384 + i * 1024 + thread_binding * 8, 1, 1), 8)
        T.ptx_commit_group()
        for i in T.unroll(4):
            T.ptx_cp_async(T.tvm_access_ptr(T.type_annotation("float16"), K_shared, i * 1024 + thread_binding // 8 * 64 + (thread_binding % 64 // 32 + thread_binding % 8 // 4) % 2 * 32 + (thread_binding % 32 // 16 + thread_binding % 4 // 2) % 2 * 16 + (thread_binding % 16 // 8 + thread_binding % 2) % 2 * 8 + 4096, 1, 2), T.tvm_access_ptr(T.type_annotation("float16"), K, by * 16384 + i * 1024 + thread_binding * 8 + 4096, 1, 1), 8)
        T.ptx_commit_group()
        for i in T.unroll(4):
            T.ptx_cp_async(T.tvm_access_ptr(T.type_annotation("float16"), V_shared, i * 1024 + thread_binding // 8 * 64 + (thread_binding % 64 // 32 + thread_binding % 8 // 4) % 2 * 32 + (thread_binding % 32 // 16 + thread_binding % 4 // 2) % 2 * 16 + (thread_binding % 16 // 8 + thread_binding % 2) % 2 * 8 + 4096, 1, 2), T.tvm_access_ptr(T.type_annotation("float16"), V, by * 16384 + i * 1024 + thread_binding * 8 + 4096, 1, 1), 8)
        T.ptx_commit_group()
        for k in T.serial(2, annotations={"tl_pipelined_num_stages": 2}):
            for i in T.unroll(8):
                acc_s_1[i * 4:i * 4 + 4] = T.Broadcast(T.float32(0.0), 4)
            T.ptx_wait_group(3)
            T.tvm_storage_sync("shared")
            with T.attr(0, "lexical_alloc_scope", 1):
                A_local = T.alloc_buffer((8,), "float16", scope="local")
                B_local = T.alloc_buffer((32,), "float16", scope="local")
                for ki in range(4):
                    T.ptx_ldmatrix(T.bool(False), 4, T.tvm_access_ptr(T.type_annotation("float16"), Q_shared, thread_binding // 32 * 1024 + thread_binding % 16 * 64 + (thread_binding % 8 // 4 + ki // 2) % 2 * 32 + (thread_binding % 4 // 2 + ki % 2) % 2 * 16 + (thread_binding % 32 // 16 + thread_binding % 2) % 2 * 8, 8, 1), T.tvm_access_ptr(T.type_annotation("float16"), A_local.data, 0, 8, 2))
                    for i in T.unroll(4):
                        T.ptx_ldmatrix(T.bool(False), 4, T.tvm_access_ptr(T.type_annotation("float16"), K_shared, k * 4096 + i * 1024 + thread_binding % 32 // 16 * 512 + thread_binding % 8 * 64 + (thread_binding % 8 // 4 + ki // 2) % 2 * 32 + (thread_binding % 4 // 2 + ki % 2) % 2 * 16 + (thread_binding % 16 // 8 + thread_binding % 2) % 2 * 8, 8, 1), T.tvm_access_ptr(T.type_annotation("float16"), B_local.data, i * 8, 8, 2))
                    for j in range(4):
                        T.ptx_mma("float32", "m16n8k16", "row", "col", "fp16", "fp16", "fp32", A_local.data, 0, B_local.data, j * 8, acc_s.data, j * 8, T.bool(False))
                        T.ptx_mma("float32", "m16n8k16", "row", "col", "fp16", "fp16", "fp32", A_local.data, 0, B_local.data, j * 8 + 4, acc_s.data, j * 8 + 4, T.bool(False))
            T.tvm_storage_sync("shared.dyn")
            for i in T.unroll(4):
                T.ptx_cp_async(T.tvm_access_ptr(T.type_annotation("float16"), K_shared, k * 4096 + i * 1024 + thread_binding // 8 * 64 + (thread_binding % 64 // 32 + thread_binding % 8 // 4) % 2 * 32 + (thread_binding % 32 // 16 + thread_binding % 4 // 2) % 2 * 16 + (thread_binding % 16 // 8 + thread_binding % 2) % 2 * 8, 1, 2), T.tvm_access_ptr(T.type_annotation("float16"), K, by * 16384 + k * 4096 + i * 1024 + thread_binding * 8 + 8192, 1, 1), 8)
            T.ptx_commit_group()
            scores_max_prev_1[0:2] = scores_max_2[0:2]
            scores_max_3[0:2] = T.Broadcast(T.infinity("float32") * T.float32(-1.0), 2)
            for i in T.unroll(2):
                scores_max_clear[i] = T.float32("-inf")
                for rv in T.unroll(16):
                    scores_max_clear[i] = T.max(scores_max_clear[i], acc_s_2[rv % 8 * 4 + i * 2 + rv // 8])
                scores_max_clear[i] = T.call_extern("float32", "tl::AllReduce<tl::MaxOp, 4, 1, 0, tl::NamedBarrier<128>>::run", scores_max_clear[i])
                scores_max_4[i] = T.max(scores_max_4[i], scores_max_clear[i])
            scores_max_5[0:2] = T.max(scores_max_5[0:2], scores_max_prev_2[0:2])
            scores_scale_1[0:2] = T.exp2(scores_max_prev_3[0:2] * T.Broadcast(T.float32(0.18033688), 2) - scores_max_6[0:2] * T.Broadcast(T.float32(0.18033688), 2))
            for i in T.unroll(16):
                acc_s_3[i * 2:i * 2 + 2] = T.exp2(acc_s_3[i * 2:i * 2 + 2] * T.Broadcast(T.float32(0.18033688), 2) - T.Broadcast(scores_max_7[i % 2] * T.float32(0.18033688), 2))
            for i in T.unroll(2):
                scores_sum_1[i] = T.float32(0.0)
                for rv in T.unroll(16):
                    scores_sum_1[i] = scores_sum_1[i] + acc_s_4[rv % 8 * 4 + i * 2 + rv // 8]
                scores_sum_1[i] = T.call_extern("float32", "tl::AllReduce<tl::SumOp, 4, 1, 0, tl::NamedBarrier<128>>::run", scores_sum_1[i])
            logsum_3[0:2] = logsum_3[0:2] * scores_scale_2[0:2] + scores_sum_2[0:2]
            for i in T.unroll(8):
                acc_s_cast_1[i * 4:i * 4 + 4] = T.Cast("float16x4", acc_s_5[i * 4:i * 4 + 4])
            for i in T.unroll(16):
                acc_o_3[i * 2:i * 2 + 2] = acc_o_3[i * 2:i * 2 + 2] * T.Broadcast(scores_scale_3[i % 2], 2)
            T.ptx_wait_group(3)
            T.tvm_storage_sync("shared")
            with T.attr(0, "lexical_alloc_scope", 1):
                B_local = T.alloc_buffer((32,), "float16", scope="local")
                for ki in range(4):
                    for i in T.unroll(4):
                        T.ptx_ldmatrix(T.bool(True), 4, T.tvm_access_ptr(T.type_annotation("float16"), V_shared, k * 4096 + ki * 1024 + thread_binding % 16 * 64 + (thread_binding % 8 // 4 + i // 2) % 2 * 32 + (thread_binding % 4 // 2 + i % 2) % 2 * 16 + (thread_binding % 32 // 16 + thread_binding % 2) % 2 * 8, 8, 1), T.tvm_access_ptr(T.type_annotation("float16"), B_local.data, i * 8, 8, 2))
                    for j in range(4):
                        T.ptx_mma("float32", "m16n8k16", "row", "col", "fp16", "fp16", "fp32", acc_s_cast.data, ki * 8, B_local.data, j * 8, acc_o.data, j * 8, T.bool(False))
                        T.ptx_mma("float32", "m16n8k16", "row", "col", "fp16", "fp16", "fp32", acc_s_cast.data, ki * 8, B_local.data, j * 8 + 4, acc_o.data, j * 8 + 4, T.bool(False))
            T.tvm_storage_sync("shared.dyn")
            for i in T.unroll(4):
                T.ptx_cp_async(T.tvm_access_ptr(T.type_annotation("float16"), V_shared, k * 4096 + i * 1024 + thread_binding // 8 * 64 + (thread_binding % 64 // 32 + thread_binding % 8 // 4) % 2 * 32 + (thread_binding % 32 // 16 + thread_binding % 4 // 2) % 2 * 16 + (thread_binding % 16 // 8 + thread_binding % 2) % 2 * 8, 1, 2), T.tvm_access_ptr(T.type_annotation("float16"), V, by * 16384 + k * 4096 + i * 1024 + thread_binding * 8 + 8192, 1, 1), 8)
            T.ptx_commit_group()
        for i in T.unroll(8):
            acc_s_6[i * 4:i * 4 + 4] = T.Broadcast(T.float32(0.0), 4)
        T.ptx_wait_group(3)
        T.tvm_storage_sync("shared")
        with T.attr(0, "lexical_alloc_scope", 1):
            A_local = T.alloc_buffer((8,), "float16", scope="local")
            B_local = T.alloc_buffer((32,), "float16", scope="local")
            for ki in range(4):
                T.ptx_ldmatrix(T.bool(False), 4, T.tvm_access_ptr(T.type_annotation("float16"), Q_shared, thread_binding // 32 * 1024 + thread_binding % 16 * 64 + (thread_binding % 8 // 4 + ki // 2) % 2 * 32 + (thread_binding % 4 // 2 + ki % 2) % 2 * 16 + (thread_binding % 32 // 16 + thread_binding % 2) % 2 * 8, 8, 1), T.tvm_access_ptr(T.type_annotation("float16"), A_local.data, 0, 8, 2))
                for i in T.unroll(4):
                    T.ptx_ldmatrix(T.bool(False), 4, T.tvm_access_ptr(T.type_annotation("float16"), K_shared, i * 1024 + thread_binding % 32 // 16 * 512 + thread_binding % 8 * 64 + (thread_binding % 8 // 4 + ki // 2) % 2 * 32 + (thread_binding % 4 // 2 + ki % 2) % 2 * 16 + (thread_binding % 16 // 8 + thread_binding % 2) % 2 * 8, 8, 1), T.tvm_access_ptr(T.type_annotation("float16"), B_local.data, i * 8, 8, 2))
                for j in range(4):
                    T.ptx_mma("float32", "m16n8k16", "row", "col", "fp16", "fp16", "fp32", A_local.data, 0, B_local.data, j * 8, acc_s.data, j * 8, T.bool(False))
                    T.ptx_mma("float32", "m16n8k16", "row", "col", "fp16", "fp16", "fp32", A_local.data, 0, B_local.data, j * 8 + 4, acc_s.data, j * 8 + 4, T.bool(False))
        scores_max_prev_4[0:2] = scores_max_8[0:2]
        scores_max_9[0:2] = T.Broadcast(T.infinity("float32") * T.float32(-1.0), 2)
        for i in T.unroll(2):
            scores_max_clear_1[i] = T.float32("-inf")
            for rv in T.unroll(16):
                scores_max_clear_1[i] = T.max(scores_max_clear_1[i], acc_s_7[rv % 8 * 4 + i * 2 + rv // 8])
            scores_max_clear_1[i] = T.call_extern("float32", "tl::AllReduce<tl::MaxOp, 4, 1, 0, tl::NamedBarrier<128>>::run", scores_max_clear_1[i])
            scores_max_10[i] = T.max(scores_max_10[i], scores_max_clear_1[i])
        scores_max_11[0:2] = T.max(scores_max_11[0:2], scores_max_prev_5[0:2])
        scores_scale_4[0:2] = T.exp2(scores_max_prev_6[0:2] * T.Broadcast(T.float32(0.18033688), 2) - scores_max_12[0:2] * T.Broadcast(T.float32(0.18033688), 2))
        for i in T.unroll(16):
            acc_s_8[i * 2:i * 2 + 2] = T.exp2(acc_s_8[i * 2:i * 2 + 2] * T.Broadcast(T.float32(0.18033688), 2) - T.Broadcast(scores_max_13[i % 2] * T.float32(0.18033688), 2))
        for i in T.unroll(2):
            scores_sum_3[i] = T.float32(0.0)
            for rv in T.unroll(16):
                scores_sum_3[i] = scores_sum_3[i] + acc_s_9[rv % 8 * 4 + i * 2 + rv // 8]
            scores_sum_3[i] = T.call_extern("float32", "tl::AllReduce<tl::SumOp, 4, 1, 0, tl::NamedBarrier<128>>::run", scores_sum_3[i])
        logsum_4[0:2] = logsum_4[0:2] * scores_scale_5[0:2] + scores_sum_4[0:2]
        for i in T.unroll(8):
            acc_s_cast_2[i * 4:i * 4 + 4] = T.Cast("float16x4", acc_s_10[i * 4:i * 4 + 4])
        for i in T.unroll(16):
            acc_o_4[i * 2:i * 2 + 2] = acc_o_4[i * 2:i * 2 + 2] * T.Broadcast(scores_scale_6[i % 2], 2)
        T.ptx_wait_group(2)
        T.tvm_storage_sync("shared")
        with T.attr(0, "lexical_alloc_scope", 1):
            B_local = T.alloc_buffer((32,), "float16", scope="local")
            for ki in range(4):
                for i in T.unroll(4):
                    T.ptx_ldmatrix(T.bool(True), 4, T.tvm_access_ptr(T.type_annotation("float16"), V_shared, ki * 1024 + thread_binding % 16 * 64 + (thread_binding % 8 // 4 + i // 2) % 2 * 32 + (thread_binding % 4 // 2 + i % 2) % 2 * 16 + (thread_binding % 32 // 16 + thread_binding % 2) % 2 * 8, 8, 1), T.tvm_access_ptr(T.type_annotation("float16"), B_local.data, i * 8, 8, 2))
                for j in range(4):
                    T.ptx_mma("float32", "m16n8k16", "row", "col", "fp16", "fp16", "fp32", acc_s_cast.data, ki * 8, B_local.data, j * 8, acc_o.data, j * 8, T.bool(False))
                    T.ptx_mma("float32", "m16n8k16", "row", "col", "fp16", "fp16", "fp32", acc_s_cast.data, ki * 8, B_local.data, j * 8 + 4, acc_o.data, j * 8 + 4, T.bool(False))
        for i in T.unroll(8):
            acc_s_11[i * 4:i * 4 + 4] = T.Broadcast(T.float32(0.0), 4)
        T.ptx_wait_group(1)
        T.tvm_storage_sync("shared")
        with T.attr(0, "lexical_alloc_scope", 1):
            A_local = T.alloc_buffer((8,), "float16", scope="local")
            B_local = T.alloc_buffer((32,), "float16", scope="local")
            for ki in range(4):
                T.ptx_ldmatrix(T.bool(False), 4, T.tvm_access_ptr(T.type_annotation("float16"), Q_shared, thread_binding // 32 * 1024 + thread_binding % 16 * 64 + (thread_binding % 8 // 4 + ki // 2) % 2 * 32 + (thread_binding % 4 // 2 + ki % 2) % 2 * 16 + (thread_binding % 32 // 16 + thread_binding % 2) % 2 * 8, 8, 1), T.tvm_access_ptr(T.type_annotation("float16"), A_local.data, 0, 8, 2))
                for i in T.unroll(4):
                    T.ptx_ldmatrix(T.bool(False), 4, T.tvm_access_ptr(T.type_annotation("float16"), K_shared, i * 1024 + thread_binding % 32 // 16 * 512 + thread_binding % 8 * 64 + (thread_binding % 8 // 4 + ki // 2) % 2 * 32 + (thread_binding % 4 // 2 + ki % 2) % 2 * 16 + (thread_binding % 16 // 8 + thread_binding % 2) % 2 * 8 + 4096, 8, 1), T.tvm_access_ptr(T.type_annotation("float16"), B_local.data, i * 8, 8, 2))
                for j in range(4):
                    T.ptx_mma("float32", "m16n8k16", "row", "col", "fp16", "fp16", "fp32", A_local.data, 0, B_local.data, j * 8, acc_s.data, j * 8, T.bool(False))
                    T.ptx_mma("float32", "m16n8k16", "row", "col", "fp16", "fp16", "fp32", A_local.data, 0, B_local.data, j * 8 + 4, acc_s.data, j * 8 + 4, T.bool(False))
        scores_max_prev_7[0:2] = scores_max_14[0:2]
        scores_max_15[0:2] = T.Broadcast(T.infinity("float32") * T.float32(-1.0), 2)
        for i in T.unroll(2):
            scores_max_clear_2[i] = T.float32("-inf")
            for rv in T.unroll(16):
                scores_max_clear_2[i] = T.max(scores_max_clear_2[i], acc_s_12[rv % 8 * 4 + i * 2 + rv // 8])
            scores_max_clear_2[i] = T.call_extern("float32", "tl::AllReduce<tl::MaxOp, 4, 1, 0, tl::NamedBarrier<128>>::run", scores_max_clear_2[i])
            scores_max_16[i] = T.max(scores_max_16[i], scores_max_clear_2[i])
        scores_max_17[0:2] = T.max(scores_max_17[0:2], scores_max_prev_8[0:2])
        scores_scale_7[0:2] = T.exp2(scores_max_prev_9[0:2] * T.Broadcast(T.float32(0.18033688), 2) - scores_max_18[0:2] * T.Broadcast(T.float32(0.18033688), 2))
        for i in T.unroll(16):
            acc_s_13[i * 2:i * 2 + 2] = T.exp2(acc_s_13[i * 2:i * 2 + 2] * T.Broadcast(T.float32(0.18033688), 2) - T.Broadcast(scores_max_19[i % 2] * T.float32(0.18033688), 2))
        for i in T.unroll(2):
            scores_sum_5[i] = T.float32(0.0)
            for rv in T.unroll(16):
                scores_sum_5[i] = scores_sum_5[i] + acc_s_14[rv % 8 * 4 + i * 2 + rv // 8]
            scores_sum_5[i] = T.call_extern("float32", "tl::AllReduce<tl::SumOp, 4, 1, 0, tl::NamedBarrier<128>>::run", scores_sum_5[i])
        logsum_5[0:2] = logsum_5[0:2] * scores_scale_8[0:2] + scores_sum_6[0:2]
        for i in T.unroll(8):
            acc_s_cast_3[i * 4:i * 4 + 4] = T.Cast("float16x4", acc_s_15[i * 4:i * 4 + 4])
        for i in T.unroll(16):
            acc_o_5[i * 2:i * 2 + 2] = acc_o_5[i * 2:i * 2 + 2] * T.Broadcast(scores_scale_9[i % 2], 2)
        T.ptx_wait_group(0)
        T.tvm_storage_sync("shared")
        with T.attr(0, "lexical_alloc_scope", 1):
            B_local = T.alloc_buffer((32,), "float16", scope="local")
            for ki in range(4):
                for i in T.unroll(4):
                    T.ptx_ldmatrix(T.bool(True), 4, T.tvm_access_ptr(T.type_annotation("float16"), V_shared, ki * 1024 + thread_binding % 16 * 64 + (thread_binding % 8 // 4 + i // 2) % 2 * 32 + (thread_binding % 4 // 2 + i % 2) % 2 * 16 + (thread_binding % 32 // 16 + thread_binding % 2) % 2 * 8 + 4096, 8, 1), T.tvm_access_ptr(T.type_annotation("float16"), B_local.data, i * 8, 8, 2))
                for j in range(4):
                    T.ptx_mma("float32", "m16n8k16", "row", "col", "fp16", "fp16", "fp32", acc_s_cast.data, ki * 8, B_local.data, j * 8, acc_o.data, j * 8, T.bool(False))
                    T.ptx_mma("float32", "m16n8k16", "row", "col", "fp16", "fp16", "fp32", acc_s_cast.data, ki * 8, B_local.data, j * 8 + 4, acc_o.data, j * 8 + 4, T.bool(False))
        for i in T.unroll(16):
            acc_o_1[i * 2:i * 2 + 2] = acc_o_1[i * 2:i * 2 + 2] / T.Broadcast(logsum_1[i % 2], 2)
        for i in T.unroll(16):
            T.attr(0, "lexical_alloc_scope", 1)
            Output_local_cast = T.alloc_buffer((2,), "float16", scope="local")
            Output_local_cast[0:2] = T.Cast("float16x2", acc_o_1[i * 2:i * 2 + 2])
            Output_1[by * 16384 + bx * 4096 + thread_binding // 32 * 1024 + i % 2 * 512 + thread_binding % 32 // 4 * 64 + i // 2 * 8 + thread_binding % 4 * 2:by * 16384 + bx * 4096 + thread_binding // 32 * 1024 + i % 2 * 512 + thread_binding % 32 // 4 * 64 + i // 2 * 8 + thread_binding % 4 * 2 + 2] = Output_local_cast[0:2]

    @T.prim_func
    def main(self_handle: T.handle, args: T.handle, num_args: T.int32, result: T.handle("void", "global")) -> T.int32:
        T.func_attr({"calling_conv": 1, "global_symbol": "__tvm_ffi_main", "target": T.target({"keys": ["cpu"], "kind": "c", "tag": ""}), "thread_extent": {}, "tirx.is_entry_func": True, "tl.has_tma": T.bool(False), "tl.readonly_param_indices": [0, 1, 2, 3], "tma_descriptor_args": {}})
        assert num_args == 4, ("RuntimeError", ["main: num_args should be 4"])
        assert not T.isnullptr(args), ("RuntimeError", ["main: args pointer is NULL"])
        Q_handle_type_index: T.int32 = T.tvm_struct_get(args, 0, 13, "int32")
        assert Q_handle_type_index == 0 or Q_handle_type_index == 4 or Q_handle_type_index == 7 or 64 <= Q_handle_type_index, ("RuntimeError", ["kernel main input Q expected pointer or tensor handle"])
        K_handle_type_index: T.int32 = T.tvm_struct_get(args, 1, 13, "int32")
        assert K_handle_type_index == 0 or K_handle_type_index == 4 or K_handle_type_index == 7 or 64 <= K_handle_type_index, ("RuntimeError", ["kernel main input K expected pointer or tensor handle"])
        V_handle_type_index: T.int32 = T.tvm_struct_get(args, 2, 13, "int32")
        assert V_handle_type_index == 0 or V_handle_type_index == 4 or V_handle_type_index == 7 or 64 <= V_handle_type_index, ("RuntimeError", ["kernel main input V expected pointer or tensor handle"])
        Output_handle_type_index: T.int32 = T.tvm_struct_get(args, 3, 13, "int32")
        assert Output_handle_type_index == 0 or Output_handle_type_index == 4 or Output_handle_type_index == 7 or 64 <= Output_handle_type_index, ("RuntimeError", ["kernel main input Output expected pointer or tensor handle"])
        Q_handle: T.handle = T.Select(Q_handle_type_index == 70, T.handle_add_byte_offset(T.tvm_struct_get(args, 0, 15, "handle"), 24), T.tvm_struct_get(args, 0, 15, "handle"))
        K_handle: T.handle = T.Select(K_handle_type_index == 70, T.handle_add_byte_offset(T.tvm_struct_get(args, 1, 15, "handle"), 24), T.tvm_struct_get(args, 1, 15, "handle"))
        V_handle: T.handle = T.Select(V_handle_type_index == 70, T.handle_add_byte_offset(T.tvm_struct_get(args, 2, 15, "handle"), 24), T.tvm_struct_get(args, 2, 15, "handle"))
        Output_handle: T.handle = T.Select(Output_handle_type_index == 70, T.handle_add_byte_offset(T.tvm_struct_get(args, 3, 15, "handle"), 24), T.tvm_struct_get(args, 3, 15, "handle"))
        main_Q_is_null: T.bool = T.isnullptr(Q_handle)
        assert not main_Q_is_null, ("RuntimeError", ["main.Q is expected to have non-NULL pointer"])
        main_K_is_null: T.bool = T.isnullptr(K_handle)
        assert not main_K_is_null, ("RuntimeError", ["main.K is expected to have non-NULL pointer"])
        main_V_is_null: T.bool = T.isnullptr(V_handle)
        assert not main_V_is_null, ("RuntimeError", ["main.V is expected to have non-NULL pointer"])
        main_Output_is_null: T.bool = T.isnullptr(Output_handle)
        assert not main_Output_is_null, ("RuntimeError", ["main.Output is expected to have non-NULL pointer"])
        main_Q_shape: T.handle("int64", "global") = T.tvm_struct_get(Q_handle, 0, 2, "handle")
        main_Q_shape_1 = T.decl_buffer((4,), "int64", data=main_Q_shape)
        main_K_shape: T.handle("int64", "global") = T.tvm_struct_get(K_handle, 0, 2, "handle")
        main_K_shape_1 = T.decl_buffer((4,), "int64", data=main_K_shape)
        main_V_shape: T.handle("int64", "global") = T.tvm_struct_get(V_handle, 0, 2, "handle")
        main_V_shape_1 = T.decl_buffer((4,), "int64", data=main_V_shape)
        main_Output_shape: T.handle("int64", "global") = T.tvm_struct_get(Output_handle, 0, 2, "handle")
        main_Output_shape_1 = T.decl_buffer((4,), "int64", data=main_Output_shape)
        assert T.tvm_struct_get(Q_handle, 0, 4, "int32") == 4, ("RuntimeError", ["kernel main input Q ndim mismatch, expected 4"])
        main_Q_strides: T.handle("int64", "global") = T.tvm_struct_get(Q_handle, 0, 3, "handle")
        main_Q_strides_1 = T.decl_buffer((4,), "int64", data=main_Q_strides)
        dev_id: T.int32 = T.tvm_struct_get(Q_handle, 0, 9, "int32")
        Q: T.handle("float16", "global") = T.tvm_struct_get(Q_handle, 0, 1, "handle")
        T.attr(Q, "storage_alignment", 64)
        assert T.tvm_struct_get(K_handle, 0, 4, "int32") == 4, ("RuntimeError", ["kernel main input K ndim mismatch, expected 4"])
        main_K_strides: T.handle("int64", "global") = T.tvm_struct_get(K_handle, 0, 3, "handle")
        main_K_strides_1 = T.decl_buffer((4,), "int64", data=main_K_strides)
        K: T.handle("float16", "global") = T.tvm_struct_get(K_handle, 0, 1, "handle")
        T.attr(K, "storage_alignment", 64)
        assert T.tvm_struct_get(V_handle, 0, 4, "int32") == 4, ("RuntimeError", ["kernel main input V ndim mismatch, expected 4"])
        main_V_strides: T.handle("int64", "global") = T.tvm_struct_get(V_handle, 0, 3, "handle")
        main_V_strides_1 = T.decl_buffer((4,), "int64", data=main_V_strides)
        V: T.handle("float16", "global") = T.tvm_struct_get(V_handle, 0, 1, "handle")
        T.attr(V, "storage_alignment", 64)
        assert T.tvm_struct_get(Output_handle, 0, 4, "int32") == 4, ("RuntimeError", ["kernel main input Output ndim mismatch, expected 4"])
        main_Output_strides: T.handle("int64", "global") = T.tvm_struct_get(Output_handle, 0, 3, "handle")
        main_Output_strides_1 = T.decl_buffer((4,), "int64", data=main_Output_strides)
        Output: T.handle("float16", "global") = T.tvm_struct_get(Output_handle, 0, 1, "handle")
        T.attr(Output, "storage_alignment", 64)
        T.attr("default", "device_id", dev_id)
        T.attr("default", "device_type", 2)
        assert T.tvm_struct_get(Q_handle, 0, 5, "uint8") == T.uint8(2) and T.tvm_struct_get(Q_handle, 0, 6, "uint8") == T.uint8(16) and T.tvm_struct_get(Q_handle, 0, 7, "uint16") == T.uint16(1), ("RuntimeError", ["kernel main input Q dtype mismatch, expected float16"])
        assert T.Cast("int32", main_Q_shape_1[0]) == 1, ("RuntimeError", ["kernel main input Q shape[0] violates packed ABI constraint"])
        assert T.Cast("int32", main_Q_shape_1[1]) == 8, ("RuntimeError", ["kernel main input Q shape[1] violates packed ABI constraint"])
        assert T.Cast("int32", main_Q_shape_1[2]) == 256, ("RuntimeError", ["kernel main input Q shape[2] violates packed ABI constraint"])
        assert T.Cast("int32", main_Q_shape_1[3]) == 64, ("RuntimeError", ["kernel main input Q shape[3] violates packed ABI constraint"])
        assert T.if_then_else(T.isnullptr(main_Q_strides), 1, T.Cast("int32", main_Q_strides_1[3])) == 1, ("RuntimeError", ["kernel main input Q strides[3] violates packed ABI constraint"])
        assert T.if_then_else(T.isnullptr(main_Q_strides), 1, T.Cast("int32", main_Q_strides_1[2])) == 64, ("RuntimeError", ["kernel main input Q strides[2] violates packed ABI constraint"])
        assert T.if_then_else(T.isnullptr(main_Q_strides), 1, T.Cast("int32", main_Q_strides_1[1])) == 16384, ("RuntimeError", ["kernel main input Q strides[1] violates packed ABI constraint"])
        assert T.uint64(0) == T.tvm_struct_get(Q_handle, 0, 8, "uint64"), ("RuntimeError", ["kernel main input Q byte_offset violates packed ABI constraint"])
        assert T.tvm_struct_get(Q_handle, 0, 10, "int32") == 2, ("RuntimeError", ["kernel main input Q device_type mismatch, expected cuda"])
        assert not T.isnullptr(Q), ("RuntimeError", ["kernel main input Q data pointer is NULL"])
        assert T.tvm_struct_get(K_handle, 0, 5, "uint8") == T.uint8(2) and T.tvm_struct_get(K_handle, 0, 6, "uint8") == T.uint8(16) and T.tvm_struct_get(K_handle, 0, 7, "uint16") == T.uint16(1), ("RuntimeError", ["kernel main input K dtype mismatch, expected float16"])
        assert T.Cast("int32", main_K_shape_1[0]) == 1, ("RuntimeError", ["kernel main input K shape[0] violates packed ABI constraint"])
        assert T.Cast("int32", main_K_shape_1[1]) == 8, ("RuntimeError", ["kernel main input K shape[1] violates packed ABI constraint"])
        assert T.Cast("int32", main_K_shape_1[2]) == 256, ("RuntimeError", ["kernel main input K shape[2] violates packed ABI constraint"])
        assert T.Cast("int32", main_K_shape_1[3]) == 64, ("RuntimeError", ["kernel main input K shape[3] violates packed ABI constraint"])
        assert T.if_then_else(T.isnullptr(main_K_strides), 1, T.Cast("int32", main_K_strides_1[3])) == 1, ("RuntimeError", ["kernel main input K strides[3] violates packed ABI constraint"])
        assert T.if_then_else(T.isnullptr(main_K_strides), 1, T.Cast("int32", main_K_strides_1[2])) == 64, ("RuntimeError", ["kernel main input K strides[2] violates packed ABI constraint"])
        assert T.if_then_else(T.isnullptr(main_K_strides), 1, T.Cast("int32", main_K_strides_1[1])) == 16384, ("RuntimeError", ["kernel main input K strides[1] violates packed ABI constraint"])
        assert T.uint64(0) == T.tvm_struct_get(K_handle, 0, 8, "uint64"), ("RuntimeError", ["kernel main input K byte_offset violates packed ABI constraint"])
        assert T.tvm_struct_get(K_handle, 0, 9, "int32") == T.tvm_struct_get(Q_handle, 0, 9, "int32"), ("RuntimeError", ["kernel main input K device_id violates packed ABI constraint"])
        assert T.tvm_struct_get(K_handle, 0, 10, "int32") == 2, ("RuntimeError", ["kernel main input K device_type mismatch, expected cuda"])
        assert not T.isnullptr(K), ("RuntimeError", ["kernel main input K data pointer is NULL"])
        assert T.tvm_struct_get(V_handle, 0, 5, "uint8") == T.uint8(2) and T.tvm_struct_get(V_handle, 0, 6, "uint8") == T.uint8(16) and T.tvm_struct_get(V_handle, 0, 7, "uint16") == T.uint16(1), ("RuntimeError", ["kernel main input V dtype mismatch, expected float16"])
        assert T.Cast("int32", main_V_shape_1[0]) == 1, ("RuntimeError", ["kernel main input V shape[0] violates packed ABI constraint"])
        assert T.Cast("int32", main_V_shape_1[1]) == 8, ("RuntimeError", ["kernel main input V shape[1] violates packed ABI constraint"])
        assert T.Cast("int32", main_V_shape_1[2]) == 256, ("RuntimeError", ["kernel main input V shape[2] violates packed ABI constraint"])
        assert T.Cast("int32", main_V_shape_1[3]) == 64, ("RuntimeError", ["kernel main input V shape[3] violates packed ABI constraint"])
        assert T.if_then_else(T.isnullptr(main_V_strides), 1, T.Cast("int32", main_V_strides_1[3])) == 1, ("RuntimeError", ["kernel main input V strides[3] violates packed ABI constraint"])
        assert T.if_then_else(T.isnullptr(main_V_strides), 1, T.Cast("int32", main_V_strides_1[2])) == 64, ("RuntimeError", ["kernel main input V strides[2] violates packed ABI constraint"])
        assert T.if_then_else(T.isnullptr(main_V_strides), 1, T.Cast("int32", main_V_strides_1[1])) == 16384, ("RuntimeError", ["kernel main input V strides[1] violates packed ABI constraint"])
        assert T.uint64(0) == T.tvm_struct_get(V_handle, 0, 8, "uint64"), ("RuntimeError", ["kernel main input V byte_offset violates packed ABI constraint"])
        assert T.tvm_struct_get(V_handle, 0, 9, "int32") == T.tvm_struct_get(Q_handle, 0, 9, "int32"), ("RuntimeError", ["kernel main input V device_id violates packed ABI constraint"])
        assert T.tvm_struct_get(V_handle, 0, 10, "int32") == 2, ("RuntimeError", ["kernel main input V device_type mismatch, expected cuda"])
        assert not T.isnullptr(V), ("RuntimeError", ["kernel main input V data pointer is NULL"])
        assert T.tvm_struct_get(Output_handle, 0, 5, "uint8") == T.uint8(2) and T.tvm_struct_get(Output_handle, 0, 6, "uint8") == T.uint8(16) and T.tvm_struct_get(Output_handle, 0, 7, "uint16") == T.uint16(1), ("RuntimeError", ["kernel main input Output dtype mismatch, expected float16"])
        assert T.Cast("int32", main_Output_shape_1[0]) == 1, ("RuntimeError", ["kernel main input Output shape[0] violates packed ABI constraint"])
        assert T.Cast("int32", main_Output_shape_1[1]) == 8, ("RuntimeError", ["kernel main input Output shape[1] violates packed ABI constraint"])
        assert T.Cast("int32", main_Output_shape_1[2]) == 256, ("RuntimeError", ["kernel main input Output shape[2] violates packed ABI constraint"])
        assert T.Cast("int32", main_Output_shape_1[3]) == 64, ("RuntimeError", ["kernel main input Output shape[3] violates packed ABI constraint"])
        assert T.if_then_else(T.isnullptr(main_Output_strides), 1, T.Cast("int32", main_Output_strides_1[3])) == 1, ("RuntimeError", ["kernel main input Output strides[3] violates packed ABI constraint"])
        assert T.if_then_else(T.isnullptr(main_Output_strides), 1, T.Cast("int32", main_Output_strides_1[2])) == 64, ("RuntimeError", ["kernel main input Output strides[2] violates packed ABI constraint"])
        assert T.if_then_else(T.isnullptr(main_Output_strides), 1, T.Cast("int32", main_Output_strides_1[1])) == 16384, ("RuntimeError", ["kernel main input Output strides[1] violates packed ABI constraint"])
        assert T.uint64(0) == T.tvm_struct_get(Output_handle, 0, 8, "uint64"), ("RuntimeError", ["kernel main input Output byte_offset violates packed ABI constraint"])
        assert T.tvm_struct_get(Output_handle, 0, 9, "int32") == T.tvm_struct_get(Q_handle, 0, 9, "int32"), ("RuntimeError", ["kernel main input Output device_id violates packed ABI constraint"])
        assert T.tvm_struct_get(Output_handle, 0, 10, "int32") == 2, ("RuntimeError", ["kernel main input Output device_type mismatch, expected cuda"])
        assert not T.isnullptr(Output), ("RuntimeError", ["kernel main input Output data pointer is NULL"])
        Q_1 = T.decl_buffer((1, 8, 256, 64), "float16", data=Q, strides=(131072, 16384, 64, 1))
        K_1 = T.decl_buffer((1, 8, 256, 64), "float16", data=K, strides=(131072, 16384, 64, 1))
        V_1 = T.decl_buffer((1, 8, 256, 64), "float16", data=V, strides=(131072, 16384, 64, 1))
        Output_1 = T.decl_buffer((1, 8, 256, 64), "float16", data=Output, strides=(131072, 16384, 64, 1))
        T.call_packed("__tvm_set_device", 2, dev_id)
        with T.attr(0, "compute_scope", "main_compute_"):
            T.call_packed("main_kernel", K, Output, Q, V, 4, 8, 1, 128, 1, 1, 40960)
        return 0