# after pass 52: s_tir.InferFragment  (changed: False)
# from tvm.script import ir as I
# from tvm.script import tirx as T

@I.ir_module
class Module:
    @T.prim_func
    def main(Q_handle: T.handle, K_handle: T.handle, V_handle: T.handle, Output_handle: T.handle):
        T.func_attr({"target": T.target({"arch": "sm_80", "host": {"keys": ["cpu"], "kind": "c", "tag": ""}, "keys": ["cuda", "gpu"], "kind": "cuda", "max_num_threads": 1024, "tag": "", "thread_warp_size": 32}), "tirx.is_entry_func": True, "tl.has_tma": T.bool(False)})
        Q = T.match_buffer(Q_handle, (1, 8, 256, 64), "float16", strides=(131072, 16384, 64, 1))
        K = T.match_buffer(K_handle, (1, 8, 256, 64), "float16", strides=(131072, 16384, 64, 1))
        V = T.match_buffer(V_handle, (1, 8, 256, 64), "float16", strides=(131072, 16384, 64, 1))
        Output = T.match_buffer(Output_handle, (1, 8, 256, 64), "float16", strides=(131072, 16384, 64, 1))
        bx = T.launch_thread("blockIdx.x", 4)
        Q_shared = T.alloc_buffer((4096,), "float16", scope="shared.dyn")
        acc_o = T.alloc_buffer((32,), scope="local")
        logsum = T.alloc_buffer((2,), scope="local")
        scores_max = T.alloc_buffer((2,), scope="local")
        K_shared = T.alloc_buffer((8192,), "float16", scope="shared.dyn")
        V_shared = T.alloc_buffer((8192,), "float16", scope="shared.dyn")
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
        Q_shared_1 = T.decl_buffer((4096,), "float16", data=Q_shared.data, scope="shared.dyn")
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
            Q_1 = T.Buffer((131072,), "float16", data=Q.data)
            Q_shared_1[i * 1024 + thread_binding // 8 * 64 + (thread_binding % 64 // 32 + thread_binding % 8 // 4) % 2 * 32 + (thread_binding % 32 // 16 + thread_binding % 4 // 2) % 2 * 16 + (thread_binding % 16 // 8 + thread_binding % 2) % 2 * 8:i * 1024 + thread_binding // 8 * 64 + (thread_binding % 64 // 32 + thread_binding % 8 // 4) % 2 * 32 + (thread_binding % 32 // 16 + thread_binding % 4 // 2) % 2 * 16 + (thread_binding % 16 // 8 + thread_binding % 2) % 2 * 8 + 8] = Q_1[by * 16384 + bx * 4096 + i * 1024 + thread_binding * 8:by * 16384 + bx * 4096 + i * 1024 + thread_binding * 8 + 8]
        for i in T.unroll(8):
            acc_o_2[i * 4:i * 4 + 4] = T.Broadcast(T.float32(0.0), 4)
        logsum_2[0:2] = T.Broadcast(T.float32(0.0), 2)
        scores_max_1[0:2] = T.Broadcast(T.infinity("float32") * T.float32(-1.0), 2)
        for i in T.unroll(4):
            T.ptx_cp_async(T.tvm_access_ptr(T.type_annotation("float16"), K_shared.data, i * 1024 + thread_binding // 8 * 64 + (thread_binding % 64 // 32 + thread_binding % 8 // 4) % 2 * 32 + (thread_binding % 32 // 16 + thread_binding % 4 // 2) % 2 * 16 + (thread_binding % 16 // 8 + thread_binding % 2) % 2 * 8, 1, 2), T.tvm_access_ptr(T.type_annotation("float16"), K.data, by * 16384 + i * 1024 + thread_binding * 8, 1, 1), 8)
        T.ptx_commit_group()
        for i in T.unroll(4):
            T.ptx_cp_async(T.tvm_access_ptr(T.type_annotation("float16"), V_shared.data, i * 1024 + thread_binding // 8 * 64 + (thread_binding % 64 // 32 + thread_binding % 8 // 4) % 2 * 32 + (thread_binding % 32 // 16 + thread_binding % 4 // 2) % 2 * 16 + (thread_binding % 16 // 8 + thread_binding % 2) % 2 * 8, 1, 2), T.tvm_access_ptr(T.type_annotation("float16"), V.data, by * 16384 + i * 1024 + thread_binding * 8, 1, 1), 8)
        T.ptx_commit_group()
        for i in T.unroll(4):
            T.ptx_cp_async(T.tvm_access_ptr(T.type_annotation("float16"), K_shared.data, i * 1024 + thread_binding // 8 * 64 + (thread_binding % 64 // 32 + thread_binding % 8 // 4) % 2 * 32 + (thread_binding % 32 // 16 + thread_binding % 4 // 2) % 2 * 16 + (thread_binding % 16 // 8 + thread_binding % 2) % 2 * 8 + 4096, 1, 2), T.tvm_access_ptr(T.type_annotation("float16"), K.data, by * 16384 + i * 1024 + thread_binding * 8 + 4096, 1, 1), 8)
        T.ptx_commit_group()
        for i in T.unroll(4):
            T.ptx_cp_async(T.tvm_access_ptr(T.type_annotation("float16"), V_shared.data, i * 1024 + thread_binding // 8 * 64 + (thread_binding % 64 // 32 + thread_binding % 8 // 4) % 2 * 32 + (thread_binding % 32 // 16 + thread_binding % 4 // 2) % 2 * 16 + (thread_binding % 16 // 8 + thread_binding % 2) % 2 * 8 + 4096, 1, 2), T.tvm_access_ptr(T.type_annotation("float16"), V.data, by * 16384 + i * 1024 + thread_binding * 8 + 4096, 1, 1), 8)
        T.ptx_commit_group()
        for k in T.serial(2, annotations={"tl_pipelined_num_stages": 2}):
            for i in T.unroll(8):
                acc_s_1[i * 4:i * 4 + 4] = T.Broadcast(T.float32(0.0), 4)
            T.ptx_wait_group(3)
            with T.attr(0, "lexical_alloc_scope", 1):
                A_local = T.alloc_buffer((8,), "float16", scope="local")
                B_local = T.alloc_buffer((32,), "float16", scope="local")
                for ki in range(4):
                    T.ptx_ldmatrix(T.bool(False), 4, T.tvm_access_ptr(T.type_annotation("float16"), Q_shared.data, thread_binding // 32 * 1024 + thread_binding % 16 * 64 + (thread_binding % 8 // 4 + ki // 2) % 2 * 32 + (thread_binding % 4 // 2 + ki % 2) % 2 * 16 + (thread_binding % 32 // 16 + thread_binding % 2) % 2 * 8, 8, 1), T.tvm_access_ptr(T.type_annotation("float16"), A_local.data, 0, 8, 2))
                    for i in T.unroll(4):
                        T.ptx_ldmatrix(T.bool(False), 4, T.tvm_access_ptr(T.type_annotation("float16"), K_shared.data, k * 4096 + i * 1024 + thread_binding % 32 // 16 * 512 + thread_binding % 8 * 64 + (thread_binding % 8 // 4 + ki // 2) % 2 * 32 + (thread_binding % 4 // 2 + ki % 2) % 2 * 16 + (thread_binding % 16 // 8 + thread_binding % 2) % 2 * 8, 8, 1), T.tvm_access_ptr(T.type_annotation("float16"), B_local.data, i * 8, 8, 2))
                    for j in range(4):
                        T.ptx_mma("float32", "m16n8k16", "row", "col", "fp16", "fp16", "fp32", A_local.data, 0, B_local.data, j * 8, acc_s.data, j * 8, T.bool(False))
                        T.ptx_mma("float32", "m16n8k16", "row", "col", "fp16", "fp16", "fp32", A_local.data, 0, B_local.data, j * 8 + 4, acc_s.data, j * 8 + 4, T.bool(False))
            for i in T.unroll(4):
                T.ptx_cp_async(T.tvm_access_ptr(T.type_annotation("float16"), K_shared.data, k * 4096 + i * 1024 + thread_binding // 8 * 64 + (thread_binding % 64 // 32 + thread_binding % 8 // 4) % 2 * 32 + (thread_binding % 32 // 16 + thread_binding % 4 // 2) % 2 * 16 + (thread_binding % 16 // 8 + thread_binding % 2) % 2 * 8, 1, 2), T.tvm_access_ptr(T.type_annotation("float16"), K.data, by * 16384 + k * 4096 + i * 1024 + thread_binding * 8 + 8192, 1, 1), 8)
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
            with T.attr(0, "lexical_alloc_scope", 1):
                B_local = T.alloc_buffer((32,), "float16", scope="local")
                for ki in range(4):
                    for i in T.unroll(4):
                        T.ptx_ldmatrix(T.bool(True), 4, T.tvm_access_ptr(T.type_annotation("float16"), V_shared.data, k * 4096 + ki * 1024 + thread_binding % 16 * 64 + (thread_binding % 8 // 4 + i // 2) % 2 * 32 + (thread_binding % 4 // 2 + i % 2) % 2 * 16 + (thread_binding % 32 // 16 + thread_binding % 2) % 2 * 8, 8, 1), T.tvm_access_ptr(T.type_annotation("float16"), B_local.data, i * 8, 8, 2))
                    for j in range(4):
                        T.ptx_mma("float32", "m16n8k16", "row", "col", "fp16", "fp16", "fp32", acc_s_cast.data, ki * 8, B_local.data, j * 8, acc_o.data, j * 8, T.bool(False))
                        T.ptx_mma("float32", "m16n8k16", "row", "col", "fp16", "fp16", "fp32", acc_s_cast.data, ki * 8, B_local.data, j * 8 + 4, acc_o.data, j * 8 + 4, T.bool(False))
            for i in T.unroll(4):
                T.ptx_cp_async(T.tvm_access_ptr(T.type_annotation("float16"), V_shared.data, k * 4096 + i * 1024 + thread_binding // 8 * 64 + (thread_binding % 64 // 32 + thread_binding % 8 // 4) % 2 * 32 + (thread_binding % 32 // 16 + thread_binding % 4 // 2) % 2 * 16 + (thread_binding % 16 // 8 + thread_binding % 2) % 2 * 8, 1, 2), T.tvm_access_ptr(T.type_annotation("float16"), V.data, by * 16384 + k * 4096 + i * 1024 + thread_binding * 8 + 8192, 1, 1), 8)
            T.ptx_commit_group()
        for i in T.unroll(8):
            acc_s_6[i * 4:i * 4 + 4] = T.Broadcast(T.float32(0.0), 4)
        T.ptx_wait_group(3)
        with T.attr(0, "lexical_alloc_scope", 1):
            A_local = T.alloc_buffer((8,), "float16", scope="local")
            B_local = T.alloc_buffer((32,), "float16", scope="local")
            for ki in range(4):
                T.ptx_ldmatrix(T.bool(False), 4, T.tvm_access_ptr(T.type_annotation("float16"), Q_shared.data, thread_binding // 32 * 1024 + thread_binding % 16 * 64 + (thread_binding % 8 // 4 + ki // 2) % 2 * 32 + (thread_binding % 4 // 2 + ki % 2) % 2 * 16 + (thread_binding % 32 // 16 + thread_binding % 2) % 2 * 8, 8, 1), T.tvm_access_ptr(T.type_annotation("float16"), A_local.data, 0, 8, 2))
                for i in T.unroll(4):
                    T.ptx_ldmatrix(T.bool(False), 4, T.tvm_access_ptr(T.type_annotation("float16"), K_shared.data, i * 1024 + thread_binding % 32 // 16 * 512 + thread_binding % 8 * 64 + (thread_binding % 8 // 4 + ki // 2) % 2 * 32 + (thread_binding % 4 // 2 + ki % 2) % 2 * 16 + (thread_binding % 16 // 8 + thread_binding % 2) % 2 * 8, 8, 1), T.tvm_access_ptr(T.type_annotation("float16"), B_local.data, i * 8, 8, 2))
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
        with T.attr(0, "lexical_alloc_scope", 1):
            B_local = T.alloc_buffer((32,), "float16", scope="local")
            for ki in range(4):
                for i in T.unroll(4):
                    T.ptx_ldmatrix(T.bool(True), 4, T.tvm_access_ptr(T.type_annotation("float16"), V_shared.data, ki * 1024 + thread_binding % 16 * 64 + (thread_binding % 8 // 4 + i // 2) % 2 * 32 + (thread_binding % 4 // 2 + i % 2) % 2 * 16 + (thread_binding % 32 // 16 + thread_binding % 2) % 2 * 8, 8, 1), T.tvm_access_ptr(T.type_annotation("float16"), B_local.data, i * 8, 8, 2))
                for j in range(4):
                    T.ptx_mma("float32", "m16n8k16", "row", "col", "fp16", "fp16", "fp32", acc_s_cast.data, ki * 8, B_local.data, j * 8, acc_o.data, j * 8, T.bool(False))
                    T.ptx_mma("float32", "m16n8k16", "row", "col", "fp16", "fp16", "fp32", acc_s_cast.data, ki * 8, B_local.data, j * 8 + 4, acc_o.data, j * 8 + 4, T.bool(False))
        for i in T.unroll(8):
            acc_s_11[i * 4:i * 4 + 4] = T.Broadcast(T.float32(0.0), 4)
        T.ptx_wait_group(1)
        with T.attr(0, "lexical_alloc_scope", 1):
            A_local = T.alloc_buffer((8,), "float16", scope="local")
            B_local = T.alloc_buffer((32,), "float16", scope="local")
            for ki in range(4):
                T.ptx_ldmatrix(T.bool(False), 4, T.tvm_access_ptr(T.type_annotation("float16"), Q_shared.data, thread_binding // 32 * 1024 + thread_binding % 16 * 64 + (thread_binding % 8 // 4 + ki // 2) % 2 * 32 + (thread_binding % 4 // 2 + ki % 2) % 2 * 16 + (thread_binding % 32 // 16 + thread_binding % 2) % 2 * 8, 8, 1), T.tvm_access_ptr(T.type_annotation("float16"), A_local.data, 0, 8, 2))
                for i in T.unroll(4):
                    T.ptx_ldmatrix(T.bool(False), 4, T.tvm_access_ptr(T.type_annotation("float16"), K_shared.data, i * 1024 + thread_binding % 32 // 16 * 512 + thread_binding % 8 * 64 + (thread_binding % 8 // 4 + ki // 2) % 2 * 32 + (thread_binding % 4 // 2 + ki % 2) % 2 * 16 + (thread_binding % 16 // 8 + thread_binding % 2) % 2 * 8 + 4096, 8, 1), T.tvm_access_ptr(T.type_annotation("float16"), B_local.data, i * 8, 8, 2))
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
        with T.attr(0, "lexical_alloc_scope", 1):
            B_local = T.alloc_buffer((32,), "float16", scope="local")
            for ki in range(4):
                for i in T.unroll(4):
                    T.ptx_ldmatrix(T.bool(True), 4, T.tvm_access_ptr(T.type_annotation("float16"), V_shared.data, ki * 1024 + thread_binding % 16 * 64 + (thread_binding % 8 // 4 + i // 2) % 2 * 32 + (thread_binding % 4 // 2 + i % 2) % 2 * 16 + (thread_binding % 32 // 16 + thread_binding % 2) % 2 * 8 + 4096, 8, 1), T.tvm_access_ptr(T.type_annotation("float16"), B_local.data, i * 8, 8, 2))
                for j in range(4):
                    T.ptx_mma("float32", "m16n8k16", "row", "col", "fp16", "fp16", "fp32", acc_s_cast.data, ki * 8, B_local.data, j * 8, acc_o.data, j * 8, T.bool(False))
                    T.ptx_mma("float32", "m16n8k16", "row", "col", "fp16", "fp16", "fp32", acc_s_cast.data, ki * 8, B_local.data, j * 8 + 4, acc_o.data, j * 8 + 4, T.bool(False))
        for i in T.unroll(16):
            acc_o_1[i * 2:i * 2 + 2] = acc_o_1[i * 2:i * 2 + 2] / T.Broadcast(logsum_1[i % 2], 2)
        for i in T.unroll(16):
            T.attr(0, "lexical_alloc_scope", 1)
            Output_local_cast = T.alloc_buffer((2,), "float16", scope="local")
            Output_local_cast[0:2] = T.Cast("float16x2", acc_o_1[i * 2:i * 2 + 2])
            Output_1 = T.Buffer((131072,), "float16", data=Output.data)
            Output_1[by * 16384 + bx * 4096 + thread_binding // 32 * 1024 + i % 2 * 512 + thread_binding % 32 // 4 * 64 + i // 2 * 8 + thread_binding % 4 * 2:by * 16384 + bx * 4096 + thread_binding // 32 * 1024 + i % 2 * 512 + thread_binding % 32 // 4 * 64 + i // 2 * 8 + thread_binding % 4 * 2 + 2] = Output_local_cast[0:2]