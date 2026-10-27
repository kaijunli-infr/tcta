# after pass 41: tirx.Simplify  (changed: False)
# from tvm.script import ir as I
# from tvm.script import tirx as T

@I.ir_module
class Module:
    @T.prim_func
    def main(Q_handle: T.handle, K_handle: T.handle, V_handle: T.handle, Output_handle: T.handle):
        T.func_attr({"target": T.target({"arch": "sm_80", "host": {"keys": ["cpu"], "kind": "c", "tag": ""}, "keys": ["cuda", "gpu"], "kind": "cuda", "max_num_threads": 1024, "tag": "", "thread_warp_size": 32}), "tl.has_tma": T.bool(False)})
        Q = T.match_buffer(Q_handle, (1, 8, 256, 64), "float16", strides=(131072, 16384, 64, 1))
        K = T.match_buffer(K_handle, (1, 8, 256, 64), "float16", strides=(131072, 16384, 64, 1))
        V = T.match_buffer(V_handle, (1, 8, 256, 64), "float16", strides=(131072, 16384, 64, 1))
        Output = T.match_buffer(Output_handle, (1, 8, 256, 64), "float16", strides=(131072, 16384, 64, 1))
        bx = T.launch_thread("blockIdx.x", 4)
        by = T.launch_thread("blockIdx.y", 8)
        bz = T.launch_thread("blockIdx.z", 1)
        thread_binding = T.launch_thread("threadIdx.x", 128)
        ty = T.launch_thread("threadIdx.y", 1)
        tz = T.launch_thread("threadIdx.z", 1)
        Q_shared = T.alloc_buffer((4096,), "float16", scope="shared.dyn")
        Q_shared_1 = T.decl_buffer((4096,), "float16", data=Q_shared.data, scope="shared.dyn")
        K_shared = T.alloc_buffer((8192,), "float16", scope="shared.dyn")
        K_shared_1 = T.decl_buffer((8192,), "float16", data=K_shared.data, scope="shared.dyn")
        V_shared = T.alloc_buffer((8192,), "float16", scope="shared.dyn")
        V_shared_1 = T.decl_buffer((8192,), "float16", data=V_shared.data, scope="shared.dyn")
        acc_s = T.alloc_buffer((32,), scope="local")
        acc_s_1 = T.decl_buffer((32,), data=acc_s.data, scope="local")
        acc_s_cast = T.alloc_buffer((32,), "float16", scope="local")
        acc_s_cast_1 = T.decl_buffer((32,), "float16", data=acc_s_cast.data, scope="local")
        acc_o = T.alloc_buffer((32,), scope="local")
        acc_o_1 = T.decl_buffer((32,), data=acc_o.data, scope="local")
        scores_max = T.alloc_buffer((2,), scope="local")
        scores_max_1 = T.decl_buffer((2,), data=scores_max.data, scope="local")
        scores_max_prev = T.alloc_buffer((2,), scope="local")
        scores_max_prev_1 = T.decl_buffer((2,), data=scores_max_prev.data, scope="local")
        scores_scale = T.alloc_buffer((2,), scope="local")
        scores_scale_1 = T.decl_buffer((2,), data=scores_scale.data, scope="local")
        scores_sum = T.alloc_buffer((2,), scope="local")
        scores_sum_1 = T.decl_buffer((2,), data=scores_sum.data, scope="local")
        logsum = T.alloc_buffer((2,), scope="local")
        logsum_1 = T.decl_buffer((2,), data=logsum.data, scope="local")
        Q_shared_2 = T.alloc_buffer((4096,), "float16", scope="shared.dyn")
        Q_shared_3 = T.decl_buffer((4096,), "float16", data=Q_shared.data, scope="shared.dyn")
        acc_o_2 = T.alloc_buffer((32,), scope="local")
        acc_o_3 = T.decl_buffer((32,), data=acc_o.data, scope="local")
        logsum_2 = T.alloc_buffer((2,), scope="local")
        logsum_3 = T.decl_buffer((2,), data=logsum.data, scope="local")
        scores_max_2 = T.alloc_buffer((2,), scope="local")
        scores_max_3 = T.decl_buffer((2,), data=scores_max.data, scope="local")
        acc_s_2 = T.alloc_buffer((32,), scope="local")
        acc_s_3 = T.decl_buffer((32,), data=acc_s.data, scope="local")
        scores_max_prev_2 = T.alloc_buffer((2,), scope="local")
        scores_max_prev_3 = T.decl_buffer((2,), data=scores_max_prev.data, scope="local")
        scores_max_4 = T.alloc_buffer((2,), scope="local")
        scores_max_5 = T.decl_buffer((2,), data=scores_max.data, scope="local")
        scores_max_6 = T.alloc_buffer((2,), scope="local")
        scores_max_7 = T.decl_buffer((2,), data=scores_max.data, scope="local")
        acc_s_4 = T.alloc_buffer((32,), scope="local")
        acc_s_5 = T.decl_buffer((32,), data=acc_s.data, scope="local")
        scores_max_8 = T.alloc_buffer((2,), scope="local")
        scores_max_9 = T.decl_buffer((2,), data=scores_max.data, scope="local")
        scores_max_10 = T.alloc_buffer((2,), scope="local")
        scores_max_11 = T.decl_buffer((2,), data=scores_max.data, scope="local")
        scores_max_prev_4 = T.alloc_buffer((2,), scope="local")
        scores_max_prev_5 = T.decl_buffer((2,), data=scores_max_prev.data, scope="local")
        scores_scale_2 = T.alloc_buffer((2,), scope="local")
        scores_scale_3 = T.decl_buffer((2,), data=scores_scale.data, scope="local")
        scores_max_prev_6 = T.alloc_buffer((2,), scope="local")
        scores_max_prev_7 = T.decl_buffer((2,), data=scores_max_prev.data, scope="local")
        scores_max_12 = T.alloc_buffer((2,), scope="local")
        scores_max_13 = T.decl_buffer((2,), data=scores_max.data, scope="local")
        acc_s_6 = T.alloc_buffer((32,), scope="local")
        acc_s_7 = T.decl_buffer((32,), data=acc_s.data, scope="local")
        scores_max_14 = T.alloc_buffer((2,), scope="local")
        scores_max_15 = T.decl_buffer((2,), data=scores_max.data, scope="local")
        scores_sum_2 = T.alloc_buffer((2,), scope="local")
        scores_sum_3 = T.decl_buffer((2,), data=scores_sum.data, scope="local")
        acc_s_8 = T.alloc_buffer((32,), scope="local")
        acc_s_9 = T.decl_buffer((32,), data=acc_s.data, scope="local")
        logsum_4 = T.alloc_buffer((2,), scope="local")
        logsum_5 = T.decl_buffer((2,), data=logsum.data, scope="local")
        scores_scale_4 = T.alloc_buffer((2,), scope="local")
        scores_scale_5 = T.decl_buffer((2,), data=scores_scale.data, scope="local")
        scores_sum_4 = T.alloc_buffer((2,), scope="local")
        scores_sum_5 = T.decl_buffer((2,), data=scores_sum.data, scope="local")
        acc_s_cast_2 = T.alloc_buffer((32,), "float16", scope="local")
        acc_s_cast_3 = T.decl_buffer((32,), "float16", data=acc_s_cast.data, scope="local")
        acc_s_10 = T.alloc_buffer((32,), scope="local")
        acc_s_11 = T.decl_buffer((32,), data=acc_s.data, scope="local")
        acc_o_4 = T.alloc_buffer((32,), scope="local")
        acc_o_5 = T.decl_buffer((32,), data=acc_o.data, scope="local")
        scores_scale_6 = T.alloc_buffer((2,), scope="local")
        scores_scale_7 = T.decl_buffer((2,), data=scores_scale.data, scope="local")
        acc_s_12 = T.alloc_buffer((32,), scope="local")
        acc_s_13 = T.decl_buffer((32,), data=acc_s.data, scope="local")
        scores_max_prev_8 = T.alloc_buffer((2,), scope="local")
        scores_max_prev_9 = T.decl_buffer((2,), data=scores_max_prev.data, scope="local")
        scores_max_16 = T.alloc_buffer((2,), scope="local")
        scores_max_17 = T.decl_buffer((2,), data=scores_max.data, scope="local")
        scores_max_18 = T.alloc_buffer((2,), scope="local")
        scores_max_19 = T.decl_buffer((2,), data=scores_max.data, scope="local")
        acc_s_14 = T.alloc_buffer((32,), scope="local")
        acc_s_15 = T.decl_buffer((32,), data=acc_s.data, scope="local")
        scores_max_20 = T.alloc_buffer((2,), scope="local")
        scores_max_21 = T.decl_buffer((2,), data=scores_max.data, scope="local")
        scores_max_22 = T.alloc_buffer((2,), scope="local")
        scores_max_23 = T.decl_buffer((2,), data=scores_max.data, scope="local")
        scores_max_prev_10 = T.alloc_buffer((2,), scope="local")
        scores_max_prev_11 = T.decl_buffer((2,), data=scores_max_prev.data, scope="local")
        scores_scale_8 = T.alloc_buffer((2,), scope="local")
        scores_scale_9 = T.decl_buffer((2,), data=scores_scale.data, scope="local")
        scores_max_prev_12 = T.alloc_buffer((2,), scope="local")
        scores_max_prev_13 = T.decl_buffer((2,), data=scores_max_prev.data, scope="local")
        scores_max_24 = T.alloc_buffer((2,), scope="local")
        scores_max_25 = T.decl_buffer((2,), data=scores_max.data, scope="local")
        acc_s_16 = T.alloc_buffer((32,), scope="local")
        acc_s_17 = T.decl_buffer((32,), data=acc_s.data, scope="local")
        scores_max_26 = T.alloc_buffer((2,), scope="local")
        scores_max_27 = T.decl_buffer((2,), data=scores_max.data, scope="local")
        scores_sum_6 = T.alloc_buffer((2,), scope="local")
        scores_sum_7 = T.decl_buffer((2,), data=scores_sum.data, scope="local")
        acc_s_18 = T.alloc_buffer((32,), scope="local")
        acc_s_19 = T.decl_buffer((32,), data=acc_s.data, scope="local")
        logsum_6 = T.alloc_buffer((2,), scope="local")
        logsum_7 = T.decl_buffer((2,), data=logsum.data, scope="local")
        scores_scale_10 = T.alloc_buffer((2,), scope="local")
        scores_scale_11 = T.decl_buffer((2,), data=scores_scale.data, scope="local")
        scores_sum_8 = T.alloc_buffer((2,), scope="local")
        scores_sum_9 = T.decl_buffer((2,), data=scores_sum.data, scope="local")
        acc_s_cast_4 = T.alloc_buffer((32,), "float16", scope="local")
        acc_s_cast_5 = T.decl_buffer((32,), "float16", data=acc_s_cast.data, scope="local")
        acc_s_20 = T.alloc_buffer((32,), scope="local")
        acc_s_21 = T.decl_buffer((32,), data=acc_s.data, scope="local")
        acc_o_6 = T.alloc_buffer((32,), scope="local")
        acc_o_7 = T.decl_buffer((32,), data=acc_o.data, scope="local")
        scores_scale_12 = T.alloc_buffer((2,), scope="local")
        scores_scale_13 = T.decl_buffer((2,), data=scores_scale.data, scope="local")
        acc_s_22 = T.alloc_buffer((32,), scope="local")
        acc_s_23 = T.decl_buffer((32,), data=acc_s.data, scope="local")
        scores_max_prev_14 = T.alloc_buffer((2,), scope="local")
        scores_max_prev_15 = T.decl_buffer((2,), data=scores_max_prev.data, scope="local")
        scores_max_28 = T.alloc_buffer((2,), scope="local")
        scores_max_29 = T.decl_buffer((2,), data=scores_max.data, scope="local")
        scores_max_30 = T.alloc_buffer((2,), scope="local")
        scores_max_31 = T.decl_buffer((2,), data=scores_max.data, scope="local")
        acc_s_24 = T.alloc_buffer((32,), scope="local")
        acc_s_25 = T.decl_buffer((32,), data=acc_s.data, scope="local")
        scores_max_32 = T.alloc_buffer((2,), scope="local")
        scores_max_33 = T.decl_buffer((2,), data=scores_max.data, scope="local")
        scores_max_34 = T.alloc_buffer((2,), scope="local")
        scores_max_35 = T.decl_buffer((2,), data=scores_max.data, scope="local")
        scores_max_prev_16 = T.alloc_buffer((2,), scope="local")
        scores_max_prev_17 = T.decl_buffer((2,), data=scores_max_prev.data, scope="local")
        scores_scale_14 = T.alloc_buffer((2,), scope="local")
        scores_scale_15 = T.decl_buffer((2,), data=scores_scale.data, scope="local")
        scores_max_prev_18 = T.alloc_buffer((2,), scope="local")
        scores_max_prev_19 = T.decl_buffer((2,), data=scores_max_prev.data, scope="local")
        scores_max_36 = T.alloc_buffer((2,), scope="local")
        scores_max_37 = T.decl_buffer((2,), data=scores_max.data, scope="local")
        acc_s_26 = T.alloc_buffer((32,), scope="local")
        acc_s_27 = T.decl_buffer((32,), data=acc_s.data, scope="local")
        scores_max_38 = T.alloc_buffer((2,), scope="local")
        scores_max_39 = T.decl_buffer((2,), data=scores_max.data, scope="local")
        scores_sum_10 = T.alloc_buffer((2,), scope="local")
        scores_sum_11 = T.decl_buffer((2,), data=scores_sum.data, scope="local")
        acc_s_28 = T.alloc_buffer((32,), scope="local")
        acc_s_29 = T.decl_buffer((32,), data=acc_s.data, scope="local")
        logsum_8 = T.alloc_buffer((2,), scope="local")
        logsum_9 = T.decl_buffer((2,), data=logsum.data, scope="local")
        scores_scale_16 = T.alloc_buffer((2,), scope="local")
        scores_scale_17 = T.decl_buffer((2,), data=scores_scale.data, scope="local")
        scores_sum_12 = T.alloc_buffer((2,), scope="local")
        scores_sum_13 = T.decl_buffer((2,), data=scores_sum.data, scope="local")
        acc_s_cast_6 = T.alloc_buffer((32,), "float16", scope="local")
        acc_s_cast_7 = T.decl_buffer((32,), "float16", data=acc_s_cast.data, scope="local")
        acc_s_30 = T.alloc_buffer((32,), scope="local")
        acc_s_31 = T.decl_buffer((32,), data=acc_s.data, scope="local")
        acc_o_8 = T.alloc_buffer((32,), scope="local")
        acc_o_9 = T.decl_buffer((32,), data=acc_o.data, scope="local")
        scores_scale_18 = T.alloc_buffer((2,), scope="local")
        scores_scale_19 = T.decl_buffer((2,), data=scores_scale.data, scope="local")
        for i in T.unroll(4):
            for vec in T.vectorized(8):
                Q_1 = T.Buffer((131072,), "float16", data=Q.data)
                Q_shared_3[i * 1024 + thread_binding // 8 * 64 + (thread_binding % 64 // 32 + thread_binding % 8 // 4) % 2 * 32 + (thread_binding % 32 // 16 + thread_binding % 4 // 2) % 2 * 16 + (thread_binding % 16 // 8 + thread_binding % 2) % 2 * 8 + vec] = Q_1[by * 16384 + bx * 4096 + i * 1024 + thread_binding * 8 + vec]
        for i in T.unroll(8):
            for vec in T.vectorized(4):
                acc_o_3[i * 4 + vec] = T.float32(0.0)
        for i in T.vectorized(2):
            logsum_3[i] = T.float32(0.0)
        for i in T.vectorized(2):
            scores_max_3[i] = T.infinity("float32") * T.float32(-1.0)
        for i in T.unroll(4):
            for vec in T.vectorized(8):
                T.ptx_cp_async(T.tvm_access_ptr(T.type_annotation("float16"), K_shared.data, i * 1024 + thread_binding // 8 * 64 + (thread_binding % 64 // 32 + thread_binding % 8 // 4) % 2 * 32 + (thread_binding % 32 // 16 + thread_binding % 4 // 2) % 2 * 16 + (thread_binding % 16 // 8 + thread_binding % 2) % 2 * 8 + vec, 1, 2), T.tvm_access_ptr(T.type_annotation("float16"), K.data, by * 16384 + i * 1024 + thread_binding * 8 + vec, 1, 1), 1)
        T.ptx_commit_group()
        for i in T.unroll(4):
            for vec in T.vectorized(8):
                T.ptx_cp_async(T.tvm_access_ptr(T.type_annotation("float16"), V_shared.data, i * 1024 + thread_binding // 8 * 64 + (thread_binding % 64 // 32 + thread_binding % 8 // 4) % 2 * 32 + (thread_binding % 32 // 16 + thread_binding % 4 // 2) % 2 * 16 + (thread_binding % 16 // 8 + thread_binding % 2) % 2 * 8 + vec, 1, 2), T.tvm_access_ptr(T.type_annotation("float16"), V.data, by * 16384 + i * 1024 + thread_binding * 8 + vec, 1, 1), 1)
        T.ptx_commit_group()
        for i in T.unroll(4):
            for vec in T.vectorized(8):
                T.ptx_cp_async(T.tvm_access_ptr(T.type_annotation("float16"), K_shared.data, i * 1024 + thread_binding // 8 * 64 + (thread_binding % 64 // 32 + thread_binding % 8 // 4) % 2 * 32 + (thread_binding % 32 // 16 + thread_binding % 4 // 2) % 2 * 16 + (thread_binding % 16 // 8 + thread_binding % 2) % 2 * 8 + vec + 4096, 1, 2), T.tvm_access_ptr(T.type_annotation("float16"), K.data, by * 16384 + i * 1024 + thread_binding * 8 + vec + 4096, 1, 1), 1)
        T.ptx_commit_group()
        for i in T.unroll(4):
            for vec in T.vectorized(8):
                T.ptx_cp_async(T.tvm_access_ptr(T.type_annotation("float16"), V_shared.data, i * 1024 + thread_binding // 8 * 64 + (thread_binding % 64 // 32 + thread_binding % 8 // 4) % 2 * 32 + (thread_binding % 32 // 16 + thread_binding % 4 // 2) % 2 * 16 + (thread_binding % 16 // 8 + thread_binding % 2) % 2 * 8 + vec + 4096, 1, 2), T.tvm_access_ptr(T.type_annotation("float16"), V.data, by * 16384 + i * 1024 + thread_binding * 8 + vec + 4096, 1, 1), 1)
        T.ptx_commit_group()
        for k in T.serial(2, annotations={"tl_pipelined_num_stages": 2}):
            for i in T.unroll(8):
                for vec in T.vectorized(4):
                    acc_s_3[i * 4 + vec] = T.float32(0.0)
            T.ptx_wait_group(3)
            with T.attr(0, "lexical_alloc_scope", 1):
                for ki in range(4):
                    A_local = T.alloc_buffer((8,), "float16", scope="local")
                    A_local_1 = T.decl_buffer((8,), "float16", data=A_local.data, scope="local")
                    B_local = T.alloc_buffer((32,), "float16", scope="local")
                    B_local_1 = T.decl_buffer((32,), "float16", data=B_local.data, scope="local")
                    T.ptx_ldmatrix(T.bool(False), 4, T.tvm_access_ptr(T.type_annotation("float16"), Q_shared.data, thread_binding // 32 * 1024 + thread_binding % 16 * 64 + ((ki * 16 + thread_binding % 32 // 16 * 8) // 32 + thread_binding % 8 // 4) % 2 * 32 + (thread_binding % 4 // 2 + ki % 2) % 2 * 16 + (thread_binding % 32 // 16 + thread_binding % 2) % 2 * 8, 8, 1), T.tvm_access_ptr(T.type_annotation("float16"), A_local.data, 0, 8, 2))
                    for i in T.unroll(4):
                        T.ptx_ldmatrix(T.bool(False), 4, T.tvm_access_ptr(T.type_annotation("float16"), K_shared.data, k * 4096 + i * 1024 + thread_binding % 32 // 16 * 512 + thread_binding % 8 * 64 + ((ki * 16 + thread_binding % 16 // 8 * 8) // 32 + thread_binding % 8 // 4) % 2 * 32 + (thread_binding % 4 // 2 + ki % 2) % 2 * 16 + (thread_binding % 16 // 8 + thread_binding % 2) % 2 * 8, 8, 1), T.tvm_access_ptr(T.type_annotation("float16"), B_local.data, i * 8, 8, 2))
                    for j in range(4):
                        T.ptx_mma("float32", "m16n8k16", "row", "col", "fp16", "fp16", "fp32", A_local.data, 0, B_local.data, j * 8, acc_s.data, j * 8, T.bool(False))
                        T.ptx_mma("float32", "m16n8k16", "row", "col", "fp16", "fp16", "fp32", A_local.data, 0, B_local.data, j * 8 + 4, acc_s.data, j * 8 + 4, T.bool(False))
            for i in T.unroll(4):
                for vec in T.vectorized(8):
                    T.ptx_cp_async(T.tvm_access_ptr(T.type_annotation("float16"), K_shared.data, k * 4096 + i * 1024 + thread_binding // 8 * 64 + (thread_binding % 64 // 32 + thread_binding % 8 // 4) % 2 * 32 + (thread_binding % 32 // 16 + thread_binding % 4 // 2) % 2 * 16 + (thread_binding % 16 // 8 + thread_binding % 2) % 2 * 8 + vec, 1, 2), T.tvm_access_ptr(T.type_annotation("float16"), K.data, by * 16384 + k * 4096 + i * 1024 + thread_binding * 8 + vec + 8192, 1, 1), 1)
            T.ptx_commit_group()
            for i in T.vectorized(2):
                scores_max_prev_3[i] = scores_max_5[i]
            for i in T.vectorized(2):
                scores_max_7[i] = T.infinity("float32") * T.float32(-1.0)
            scores_max_clear = T.alloc_buffer((2,), scope="local")
            for i in T.unroll(2):
                scores_max_clear[i] = T.float32("-inf")
                for rv in T.unroll(16):
                    scores_max_clear[i] = T.max(scores_max_clear[i], acc_s_5[rv % 8 * 4 + i * 2 + rv // 8])
                scores_max_clear[i] = T.call_extern("float32", "tl::AllReduce<tl::MaxOp, 4, 1, 0, tl::NamedBarrier<128>>::run", scores_max_clear[i])
                scores_max_9[i] = T.max(scores_max_9[i], scores_max_clear[i])
            for i in T.vectorized(2):
                scores_max_11[i] = T.max(scores_max_11[i], scores_max_prev_5[i])
            for i in T.vectorized(2):
                scores_scale_3[i] = T.exp2(scores_max_prev_7[i] * T.float32(0.18033688) - scores_max_13[i] * T.float32(0.18033688))
            for i in T.unroll(16):
                for vec in T.vectorized(2):
                    acc_s_7[i * 2 + vec] = T.exp2(acc_s_7[i * 2 + vec] * T.float32(0.18033688) - scores_max_15[i % 2] * T.float32(0.18033688))
            for i in T.unroll(2):
                scores_sum_3[i] = T.float32(0.0)
                for rv in T.unroll(16):
                    scores_sum_3[i] = scores_sum_3[i] + acc_s_9[rv % 8 * 4 + i * 2 + rv // 8]
                scores_sum_3[i] = T.call_extern("float32", "tl::AllReduce<tl::SumOp, 4, 1, 0, tl::NamedBarrier<128>>::run", scores_sum_3[i])
            for i in T.vectorized(2):
                logsum_5[i] = logsum_5[i] * scores_scale_5[i] + scores_sum_5[i]
            for i in T.unroll(8):
                for vec in T.vectorized(4):
                    acc_s_cast_3[i * 4 + vec] = T.Cast("float16", acc_s_11[i * 4 + vec])
            for i in T.unroll(16):
                for vec in T.vectorized(2):
                    acc_o_5[i * 2 + vec] = acc_o_5[i * 2 + vec] * scores_scale_7[i % 2]
            T.ptx_wait_group(3)
            with T.attr(0, "lexical_alloc_scope", 1):
                for ki in range(4):
                    B_local = T.alloc_buffer((32,), "float16", scope="local")
                    B_local_1 = T.decl_buffer((32,), "float16", data=B_local.data, scope="local")
                    for i in T.unroll(4):
                        T.ptx_ldmatrix(T.bool(True), 4, T.tvm_access_ptr(T.type_annotation("float16"), V_shared.data, k * 4096 + ki * 1024 + thread_binding % 16 * 64 + ((i * 16 + thread_binding % 32 // 16 * 8) // 32 + thread_binding % 8 // 4) % 2 * 32 + (thread_binding % 4 // 2 + i % 2) % 2 * 16 + (thread_binding % 32 // 16 + thread_binding % 2) % 2 * 8, 8, 1), T.tvm_access_ptr(T.type_annotation("float16"), B_local.data, i * 8, 8, 2))
                    for j in range(4):
                        T.ptx_mma("float32", "m16n8k16", "row", "col", "fp16", "fp16", "fp32", acc_s_cast.data, ki * 8, B_local.data, j * 8, acc_o.data, j * 8, T.bool(False))
                        T.ptx_mma("float32", "m16n8k16", "row", "col", "fp16", "fp16", "fp32", acc_s_cast.data, ki * 8, B_local.data, j * 8 + 4, acc_o.data, j * 8 + 4, T.bool(False))
            for i in T.unroll(4):
                for vec in T.vectorized(8):
                    T.ptx_cp_async(T.tvm_access_ptr(T.type_annotation("float16"), V_shared.data, k * 4096 + i * 1024 + thread_binding // 8 * 64 + (thread_binding % 64 // 32 + thread_binding % 8 // 4) % 2 * 32 + (thread_binding % 32 // 16 + thread_binding % 4 // 2) % 2 * 16 + (thread_binding % 16 // 8 + thread_binding % 2) % 2 * 8 + vec, 1, 2), T.tvm_access_ptr(T.type_annotation("float16"), V.data, by * 16384 + k * 4096 + i * 1024 + thread_binding * 8 + vec + 8192, 1, 1), 1)
            T.ptx_commit_group()
        for i in T.unroll(8):
            for vec in T.vectorized(4):
                acc_s_13[i * 4 + vec] = T.float32(0.0)
        T.ptx_wait_group(3)
        with T.attr(0, "lexical_alloc_scope", 1):
            for ki in range(4):
                A_local = T.alloc_buffer((8,), "float16", scope="local")
                A_local_1 = T.decl_buffer((8,), "float16", data=A_local.data, scope="local")
                B_local = T.alloc_buffer((32,), "float16", scope="local")
                B_local_1 = T.decl_buffer((32,), "float16", data=B_local.data, scope="local")
                T.ptx_ldmatrix(T.bool(False), 4, T.tvm_access_ptr(T.type_annotation("float16"), Q_shared.data, thread_binding // 32 * 1024 + thread_binding % 16 * 64 + ((ki * 16 + thread_binding % 32 // 16 * 8) // 32 + thread_binding % 8 // 4) % 2 * 32 + (thread_binding % 4 // 2 + ki % 2) % 2 * 16 + (thread_binding % 32 // 16 + thread_binding % 2) % 2 * 8, 8, 1), T.tvm_access_ptr(T.type_annotation("float16"), A_local.data, 0, 8, 2))
                for i in T.unroll(4):
                    T.ptx_ldmatrix(T.bool(False), 4, T.tvm_access_ptr(T.type_annotation("float16"), K_shared.data, i * 1024 + thread_binding % 32 // 16 * 512 + thread_binding % 8 * 64 + ((ki * 16 + thread_binding % 16 // 8 * 8) // 32 + thread_binding % 8 // 4) % 2 * 32 + (thread_binding % 4 // 2 + ki % 2) % 2 * 16 + (thread_binding % 16 // 8 + thread_binding % 2) % 2 * 8, 8, 1), T.tvm_access_ptr(T.type_annotation("float16"), B_local.data, i * 8, 8, 2))
                for j in range(4):
                    T.ptx_mma("float32", "m16n8k16", "row", "col", "fp16", "fp16", "fp32", A_local.data, 0, B_local.data, j * 8, acc_s.data, j * 8, T.bool(False))
                    T.ptx_mma("float32", "m16n8k16", "row", "col", "fp16", "fp16", "fp32", A_local.data, 0, B_local.data, j * 8 + 4, acc_s.data, j * 8 + 4, T.bool(False))
        for i in T.vectorized(2):
            scores_max_prev_9[i] = scores_max_17[i]
        for i in T.vectorized(2):
            scores_max_19[i] = T.infinity("float32") * T.float32(-1.0)
        scores_max_clear = T.alloc_buffer((2,), scope="local")
        for i in T.unroll(2):
            scores_max_clear[i] = T.float32("-inf")
            for rv in T.unroll(16):
                scores_max_clear[i] = T.max(scores_max_clear[i], acc_s_15[rv % 8 * 4 + i * 2 + rv // 8])
            scores_max_clear[i] = T.call_extern("float32", "tl::AllReduce<tl::MaxOp, 4, 1, 0, tl::NamedBarrier<128>>::run", scores_max_clear[i])
            scores_max_21[i] = T.max(scores_max_21[i], scores_max_clear[i])
        for i in T.vectorized(2):
            scores_max_23[i] = T.max(scores_max_23[i], scores_max_prev_11[i])
        for i in T.vectorized(2):
            scores_scale_9[i] = T.exp2(scores_max_prev_13[i] * T.float32(0.18033688) - scores_max_25[i] * T.float32(0.18033688))
        for i in T.unroll(16):
            for vec in T.vectorized(2):
                acc_s_17[i * 2 + vec] = T.exp2(acc_s_17[i * 2 + vec] * T.float32(0.18033688) - scores_max_27[i % 2] * T.float32(0.18033688))
        for i in T.unroll(2):
            scores_sum_7[i] = T.float32(0.0)
            for rv in T.unroll(16):
                scores_sum_7[i] = scores_sum_7[i] + acc_s_19[rv % 8 * 4 + i * 2 + rv // 8]
            scores_sum_7[i] = T.call_extern("float32", "tl::AllReduce<tl::SumOp, 4, 1, 0, tl::NamedBarrier<128>>::run", scores_sum_7[i])
        for i in T.vectorized(2):
            logsum_7[i] = logsum_7[i] * scores_scale_11[i] + scores_sum_9[i]
        for i in T.unroll(8):
            for vec in T.vectorized(4):
                acc_s_cast_5[i * 4 + vec] = T.Cast("float16", acc_s_21[i * 4 + vec])
        for i in T.unroll(16):
            for vec in T.vectorized(2):
                acc_o_7[i * 2 + vec] = acc_o_7[i * 2 + vec] * scores_scale_13[i % 2]
        T.ptx_wait_group(2)
        with T.attr(0, "lexical_alloc_scope", 1):
            for ki in range(4):
                B_local = T.alloc_buffer((32,), "float16", scope="local")
                B_local_1 = T.decl_buffer((32,), "float16", data=B_local.data, scope="local")
                for i in T.unroll(4):
                    T.ptx_ldmatrix(T.bool(True), 4, T.tvm_access_ptr(T.type_annotation("float16"), V_shared.data, ki * 1024 + thread_binding % 16 * 64 + ((i * 16 + thread_binding % 32 // 16 * 8) // 32 + thread_binding % 8 // 4) % 2 * 32 + (thread_binding % 4 // 2 + i % 2) % 2 * 16 + (thread_binding % 32 // 16 + thread_binding % 2) % 2 * 8, 8, 1), T.tvm_access_ptr(T.type_annotation("float16"), B_local.data, i * 8, 8, 2))
                for j in range(4):
                    T.ptx_mma("float32", "m16n8k16", "row", "col", "fp16", "fp16", "fp32", acc_s_cast.data, ki * 8, B_local.data, j * 8, acc_o.data, j * 8, T.bool(False))
                    T.ptx_mma("float32", "m16n8k16", "row", "col", "fp16", "fp16", "fp32", acc_s_cast.data, ki * 8, B_local.data, j * 8 + 4, acc_o.data, j * 8 + 4, T.bool(False))
        for i in T.unroll(8):
            for vec in T.vectorized(4):
                acc_s_23[i * 4 + vec] = T.float32(0.0)
        T.ptx_wait_group(1)
        with T.attr(0, "lexical_alloc_scope", 1):
            for ki in range(4):
                A_local = T.alloc_buffer((8,), "float16", scope="local")
                A_local_1 = T.decl_buffer((8,), "float16", data=A_local.data, scope="local")
                B_local = T.alloc_buffer((32,), "float16", scope="local")
                B_local_1 = T.decl_buffer((32,), "float16", data=B_local.data, scope="local")
                T.ptx_ldmatrix(T.bool(False), 4, T.tvm_access_ptr(T.type_annotation("float16"), Q_shared.data, thread_binding // 32 * 1024 + thread_binding % 16 * 64 + ((ki * 16 + thread_binding % 32 // 16 * 8) // 32 + thread_binding % 8 // 4) % 2 * 32 + (thread_binding % 4 // 2 + ki % 2) % 2 * 16 + (thread_binding % 32 // 16 + thread_binding % 2) % 2 * 8, 8, 1), T.tvm_access_ptr(T.type_annotation("float16"), A_local.data, 0, 8, 2))
                for i in T.unroll(4):
                    T.ptx_ldmatrix(T.bool(False), 4, T.tvm_access_ptr(T.type_annotation("float16"), K_shared.data, i * 1024 + thread_binding % 32 // 16 * 512 + thread_binding % 8 * 64 + ((ki * 16 + thread_binding % 16 // 8 * 8) // 32 + thread_binding % 8 // 4) % 2 * 32 + (thread_binding % 4 // 2 + ki % 2) % 2 * 16 + (thread_binding % 16 // 8 + thread_binding % 2) % 2 * 8 + 4096, 8, 1), T.tvm_access_ptr(T.type_annotation("float16"), B_local.data, i * 8, 8, 2))
                for j in range(4):
                    T.ptx_mma("float32", "m16n8k16", "row", "col", "fp16", "fp16", "fp32", A_local.data, 0, B_local.data, j * 8, acc_s.data, j * 8, T.bool(False))
                    T.ptx_mma("float32", "m16n8k16", "row", "col", "fp16", "fp16", "fp32", A_local.data, 0, B_local.data, j * 8 + 4, acc_s.data, j * 8 + 4, T.bool(False))
        for i in T.vectorized(2):
            scores_max_prev_15[i] = scores_max_29[i]
        for i in T.vectorized(2):
            scores_max_31[i] = T.infinity("float32") * T.float32(-1.0)
        scores_max_clear_1 = T.alloc_buffer((2,), scope="local")
        for i in T.unroll(2):
            scores_max_clear_1[i] = T.float32("-inf")
            for rv in T.unroll(16):
                scores_max_clear_1[i] = T.max(scores_max_clear_1[i], acc_s_25[rv % 8 * 4 + i * 2 + rv // 8])
            scores_max_clear_1[i] = T.call_extern("float32", "tl::AllReduce<tl::MaxOp, 4, 1, 0, tl::NamedBarrier<128>>::run", scores_max_clear_1[i])
            scores_max_33[i] = T.max(scores_max_33[i], scores_max_clear_1[i])
        for i in T.vectorized(2):
            scores_max_35[i] = T.max(scores_max_35[i], scores_max_prev_17[i])
        for i in T.vectorized(2):
            scores_scale_15[i] = T.exp2(scores_max_prev_19[i] * T.float32(0.18033688) - scores_max_37[i] * T.float32(0.18033688))
        for i in T.unroll(16):
            for vec in T.vectorized(2):
                acc_s_27[i * 2 + vec] = T.exp2(acc_s_27[i * 2 + vec] * T.float32(0.18033688) - scores_max_39[i % 2] * T.float32(0.18033688))
        for i in T.unroll(2):
            scores_sum_11[i] = T.float32(0.0)
            for rv in T.unroll(16):
                scores_sum_11[i] = scores_sum_11[i] + acc_s_29[rv % 8 * 4 + i * 2 + rv // 8]
            scores_sum_11[i] = T.call_extern("float32", "tl::AllReduce<tl::SumOp, 4, 1, 0, tl::NamedBarrier<128>>::run", scores_sum_11[i])
        for i in T.vectorized(2):
            logsum_9[i] = logsum_9[i] * scores_scale_17[i] + scores_sum_13[i]
        for i in T.unroll(8):
            for vec in T.vectorized(4):
                acc_s_cast_7[i * 4 + vec] = T.Cast("float16", acc_s_31[i * 4 + vec])
        for i in T.unroll(16):
            for vec in T.vectorized(2):
                acc_o_9[i * 2 + vec] = acc_o_9[i * 2 + vec] * scores_scale_19[i % 2]
        T.ptx_wait_group(0)
        with T.attr(0, "lexical_alloc_scope", 1):
            for ki in range(4):
                B_local = T.alloc_buffer((32,), "float16", scope="local")
                B_local_1 = T.decl_buffer((32,), "float16", data=B_local.data, scope="local")
                for i in T.unroll(4):
                    T.ptx_ldmatrix(T.bool(True), 4, T.tvm_access_ptr(T.type_annotation("float16"), V_shared.data, ki * 1024 + thread_binding % 16 * 64 + ((i * 16 + thread_binding % 32 // 16 * 8) // 32 + thread_binding % 8 // 4) % 2 * 32 + (thread_binding % 4 // 2 + i % 2) % 2 * 16 + (thread_binding % 32 // 16 + thread_binding % 2) % 2 * 8 + 4096, 8, 1), T.tvm_access_ptr(T.type_annotation("float16"), B_local.data, i * 8, 8, 2))
                for j in range(4):
                    T.ptx_mma("float32", "m16n8k16", "row", "col", "fp16", "fp16", "fp32", acc_s_cast.data, ki * 8, B_local.data, j * 8, acc_o.data, j * 8, T.bool(False))
                    T.ptx_mma("float32", "m16n8k16", "row", "col", "fp16", "fp16", "fp32", acc_s_cast.data, ki * 8, B_local.data, j * 8 + 4, acc_o.data, j * 8 + 4, T.bool(False))
        for i in T.unroll(16):
            for vec in T.vectorized(2):
                acc_o_1[i * 2 + vec] = acc_o_1[i * 2 + vec] / logsum_1[i % 2]
        for i in T.unroll(16):
            T.attr(0, "lexical_alloc_scope", 1)
            Output_local_cast = T.alloc_buffer((2,), "float16", scope="local")
            for vec in T.vectorized(2):
                Output_local_cast[vec] = T.Cast("float16", acc_o_1[i * 2 + vec])
            for vec_copy in T.vectorized(2):
                Output_1 = T.Buffer((131072,), "float16", data=Output.data)
                Output_1[by * 16384 + bx * 4096 + thread_binding // 32 * 1024 + i % 2 * 512 + thread_binding % 32 // 4 * 64 + i // 2 * 8 + thread_binding % 4 * 2 + vec_copy] = Output_local_cast[vec_copy]