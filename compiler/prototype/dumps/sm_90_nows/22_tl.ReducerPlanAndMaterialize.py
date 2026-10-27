# after pass 22: tl.ReducerPlanAndMaterialize  (changed: False)
# from tvm.script import ir as I
# from tvm.script import tirx as T

@I.ir_module
class Module:
    @T.prim_func
    def main(Q_handle: T.handle, K_handle: T.handle, V_handle: T.handle, Output_handle: T.handle):
        T.func_attr({"target": T.target({"arch": "sm_90", "host": {"keys": ["cpu"], "kind": "c", "tag": ""}, "keys": ["cuda", "gpu"], "kind": "cuda", "max_num_threads": 1024, "tag": "", "thread_warp_size": 32})})
        Q = T.match_buffer(Q_handle, (1, 8, 256, 64), "float16", strides=(131072, 16384, 64, 1))
        K = T.match_buffer(K_handle, (1, 8, 256, 64), "float16", strides=(131072, 16384, 64, 1))
        V = T.match_buffer(V_handle, (1, 8, 256, 64), "float16", strides=(131072, 16384, 64, 1))
        Output = T.match_buffer(Output_handle, (1, 8, 256, 64), "float16", strides=(131072, 16384, 64, 1))
        with T.sblock("root"):
            by = T.int32()
            bx = T.int32()
            k = T.int32()
            T.reads(Q[0, by, bx * 64, 0], K[0, by, T.min(0, k * 64 + 128):T.min(0, k * 64 + 128) + (T.max(255, k * 64 + 191) + 1 - T.min(0, k * 64 + 128)), 0:64], V[0, by, T.min(0, k * 64 + 128):T.min(0, k * 64 + 128) + (T.max(255, k * 64 + 191) + 1 - T.min(0, k * 64 + 128)), 0:64], Output[0, by, bx * 64, 0])
            T.writes()
            Q_shared = T.Buffer((64, 64), "float16", scope="shared.dyn")
            K_shared = T.Buffer((2, 64, 64), "float16", scope="shared.dyn")
            acc_s = T.Buffer((64, 64), scope="local.fragment")
            acc_s_cast = T.Buffer((64, 64), "float16", scope="local.fragment")
            V_shared = T.Buffer((2, 64, 64), "float16", scope="shared.dyn")
            acc_o = T.Buffer((64, 64), scope="local.fragment")
            scores_max = T.Buffer((64,), scope="local.fragment")
            scores_max_prev = T.Buffer((64,), scope="local.fragment")
            scores_sum = T.Buffer((64,), scope="local.fragment")
            scores_scale = T.Buffer((64,), scope="local.fragment")
            logsum = T.Buffer((64,), scope="local.fragment")
            T.sblock_attr({"layout_map": {Q_shared: metadata["tl.Layout"][0], K_shared: metadata["tl.Layout"][1], acc_s: metadata["tl.Fragment"][0], acc_s_cast: metadata["tl.Fragment"][1], V_shared: metadata["tl.Layout"][2], acc_o: metadata["tl.Fragment"][2], scores_max: metadata["tl.Fragment"][3], scores_max_prev: metadata["tl.Fragment"][4], scores_sum: metadata["tl.Fragment"][5], scores_scale: metadata["tl.Fragment"][6], logsum: metadata["tl.Fragment"][7]}})
            T.launch_thread(bx, 4)
            T.launch_thread(by, 8)
            bz = T.launch_thread("blockIdx.z", 1)
            tx = T.launch_thread("threadIdx.x", 128)
            ty = T.launch_thread("threadIdx.y", 1)
            tz = T.launch_thread("threadIdx.z", 1)
            with T.sblock("tilelang_root"):
                T.reads(Q[0, by, bx * 64, 0], K[0, by, T.min(0, k * 64 + 128):T.min(0, k * 64 + 128) + (T.max(255, k * 64 + 191) + 1 - T.min(0, k * 64 + 128)), 0:64], K_shared[0:2, 0:64, 0:64], V[0, by, T.min(0, k * 64 + 128):T.min(0, k * 64 + 128) + (T.max(255, k * 64 + 191) + 1 - T.min(0, k * 64 + 128)), 0:64], V_shared[0:2, 0:64, 0:64], Output[0, by, bx * 64, 0])
                T.writes(K_shared[0:2, 0:64, 0:64], V_shared[0:2, 0:64, 0:64])
                T.sblock_attr({"layout_map": {Q_shared: metadata["tl.Layout"][0], K_shared: metadata["tl.Layout"][1], acc_s: metadata["tl.Fragment"][0], acc_s_cast: metadata["tl.Fragment"][1], V_shared: metadata["tl.Layout"][2], acc_o: metadata["tl.Fragment"][2], scores_max: metadata["tl.Fragment"][3], scores_max_prev: metadata["tl.Fragment"][4], scores_sum: metadata["tl.Fragment"][5], scores_scale: metadata["tl.Fragment"][6], logsum: metadata["tl.Fragment"][7]}, "tl.launch_threads": [128, 1, 1]})
                Q_shared = T.sblock_alloc_buffer((64, 64), "float16", data=Q_shared.data, scope="shared.dyn")
                K_shared = T.sblock_alloc_buffer((2, 64, 64), "float16", data=K_shared.data, scope="shared.dyn")
                V_shared = T.sblock_alloc_buffer((2, 64, 64), "float16", data=V_shared.data, scope="shared.dyn")
                acc_s = T.sblock_alloc_buffer((64, 64), data=acc_s.data, scope="local.fragment")
                acc_s_cast = T.sblock_alloc_buffer((64, 64), "float16", data=acc_s_cast.data, scope="local.fragment")
                acc_o = T.sblock_alloc_buffer((64, 64), data=acc_o.data, scope="local.fragment")
                scores_max = T.sblock_alloc_buffer((64,), data=scores_max.data, scope="local.fragment")
                scores_max_prev = T.sblock_alloc_buffer((64,), data=scores_max_prev.data, scope="local.fragment")
                scores_scale = T.sblock_alloc_buffer((64,), data=scores_scale.data, scope="local.fragment")
                scores_sum = T.sblock_alloc_buffer((64,), data=scores_sum.data, scope="local.fragment")
                logsum = T.sblock_alloc_buffer((64,), data=logsum.data, scope="local.fragment")
                T.copy(T.region(Q[0, by, bx * 64, 0], 1, 1, 1, 64, 64), T.region(Q_shared[0, 0], 2, 64, 64))
                T.fill(T.region(acc_o[0, 0], 2, 64, 64), 0)
                T.fill(T.region(logsum[0], 2, 64), 0)
                T.fill(T.region(scores_max[0], 2, 64), T.infinity("float32") * T.float32(-1.0))
                with T.sblock(""):
                    T.reads(K[0, by, 0:64, 0:64], K_shared[0:2, 0, 0], V[0, by, 0:64, 0:64], V_shared[0:2, 0, 0], K[0, by, 0:64, 0:64], V[0, by, 0:64, 0:64])
                    T.writes(K_shared[0:2, 0:64, 0:64], V_shared[0:2, 0:64, 0:64], K_shared[0, 0:64, 0:64], V_shared[0, 0:64, 0:64])
                    T.sblock_attr({"layout_map": {Q_shared: metadata["tl.Layout"][0], K_shared: metadata["tl.Layout"][1], acc_s: metadata["tl.Fragment"][0], acc_s_cast: metadata["tl.Fragment"][1], V_shared: metadata["tl.Layout"][2], acc_o: metadata["tl.Fragment"][2], scores_max: metadata["tl.Fragment"][3], scores_max_prev: metadata["tl.Fragment"][4], scores_sum: metadata["tl.Fragment"][5], scores_scale: metadata["tl.Fragment"][6], logsum: metadata["tl.Fragment"][7]}})
                    with T.sblock(""):
                        T.reads(K[0, by, 0, 0], K_shared[0:2, 0, 0], K[0, by, 0:64, 0:64])
                        T.writes(K_shared[0:2, 0:64, 0:64])
                        T.sblock_attr({"layout_map": {Q_shared: metadata["tl.Layout"][0], K_shared: metadata["tl.Layout"][1], acc_s: metadata["tl.Fragment"][0], acc_s_cast: metadata["tl.Fragment"][1], V_shared: metadata["tl.Layout"][2], acc_o: metadata["tl.Fragment"][2], scores_max: metadata["tl.Fragment"][3], scores_max_prev: metadata["tl.Fragment"][4], scores_sum: metadata["tl.Fragment"][5], scores_scale: metadata["tl.Fragment"][6], logsum: metadata["tl.Fragment"][7]}})
                        T.copy(T.region(K[0, by, 0, 0], 1, 1, 1, 64, 64), T.region(K_shared[0, 0, 0], 2, 1, 64, 64), no_implicit_async_commit_wait=1)
                    T.ptx_commit_group()
                    with T.sblock(""):
                        T.reads(V[0, by, 0, 0], V_shared[0:2, 0, 0], V[0, by, 0:64, 0:64])
                        T.writes(V_shared[0:2, 0:64, 0:64])
                        T.sblock_attr({"layout_map": {Q_shared: metadata["tl.Layout"][0], K_shared: metadata["tl.Layout"][1], acc_s: metadata["tl.Fragment"][0], acc_s_cast: metadata["tl.Fragment"][1], V_shared: metadata["tl.Layout"][2], acc_o: metadata["tl.Fragment"][2], scores_max: metadata["tl.Fragment"][3], scores_max_prev: metadata["tl.Fragment"][4], scores_sum: metadata["tl.Fragment"][5], scores_scale: metadata["tl.Fragment"][6], logsum: metadata["tl.Fragment"][7]}})
                        T.copy(T.region(V[0, by, 0, 0], 1, 1, 1, 64, 64), T.region(V_shared[0, 0, 0], 2, 1, 64, 64), no_implicit_async_commit_wait=1)
                    T.ptx_commit_group()
                with T.sblock(""):
                    T.reads(K[0, by, 64:128, 0:64], K_shared[0:2, 0, 0], V[0, by, 64:128, 0:64], V_shared[0:2, 0, 0], K[0, by, 64:128, 0:64], V[0, by, 64:128, 0:64])
                    T.writes(K_shared[0:2, 0:64, 0:64], V_shared[0:2, 0:64, 0:64], K_shared[1, 0:64, 0:64], V_shared[1, 0:64, 0:64])
                    T.sblock_attr({"layout_map": {Q_shared: metadata["tl.Layout"][0], K_shared: metadata["tl.Layout"][1], acc_s: metadata["tl.Fragment"][0], acc_s_cast: metadata["tl.Fragment"][1], V_shared: metadata["tl.Layout"][2], acc_o: metadata["tl.Fragment"][2], scores_max: metadata["tl.Fragment"][3], scores_max_prev: metadata["tl.Fragment"][4], scores_sum: metadata["tl.Fragment"][5], scores_scale: metadata["tl.Fragment"][6], logsum: metadata["tl.Fragment"][7]}})
                    with T.sblock(""):
                        T.reads(K[0, by, 64, 0], K_shared[0:2, 0, 0], K[0, by, 64:128, 0:64])
                        T.writes(K_shared[0:2, 0:64, 0:64])
                        T.sblock_attr({"layout_map": {Q_shared: metadata["tl.Layout"][0], K_shared: metadata["tl.Layout"][1], acc_s: metadata["tl.Fragment"][0], acc_s_cast: metadata["tl.Fragment"][1], V_shared: metadata["tl.Layout"][2], acc_o: metadata["tl.Fragment"][2], scores_max: metadata["tl.Fragment"][3], scores_max_prev: metadata["tl.Fragment"][4], scores_sum: metadata["tl.Fragment"][5], scores_scale: metadata["tl.Fragment"][6], logsum: metadata["tl.Fragment"][7]}})
                        T.copy(T.region(K[0, by, 64, 0], 1, 1, 1, 64, 64), T.region(K_shared[1, 0, 0], 2, 1, 64, 64), no_implicit_async_commit_wait=1)
                    T.ptx_commit_group()
                    with T.sblock(""):
                        T.reads(V[0, by, 64, 0], V_shared[0:2, 0, 0], V[0, by, 64:128, 0:64])
                        T.writes(V_shared[0:2, 0:64, 0:64])
                        T.sblock_attr({"layout_map": {Q_shared: metadata["tl.Layout"][0], K_shared: metadata["tl.Layout"][1], acc_s: metadata["tl.Fragment"][0], acc_s_cast: metadata["tl.Fragment"][1], V_shared: metadata["tl.Layout"][2], acc_o: metadata["tl.Fragment"][2], scores_max: metadata["tl.Fragment"][3], scores_max_prev: metadata["tl.Fragment"][4], scores_sum: metadata["tl.Fragment"][5], scores_scale: metadata["tl.Fragment"][6], logsum: metadata["tl.Fragment"][7]}})
                        T.copy(T.region(V[0, by, 64, 0], 1, 1, 1, 64, 64), T.region(V_shared[1, 0, 0], 2, 1, 64, 64), no_implicit_async_commit_wait=1)
                    T.ptx_commit_group()
                with T.sblock(""):
                    T.reads(acc_s[0:64, 0:64], Q_shared[0:64, 0:64], K_shared[0:2, 0:64, 0:64], K[0, by, 128:256, 0:64], scores_max[0:64], scores_max_prev[0:64], scores_sum[0:64], logsum[0:64], scores_scale[0:64], acc_s_cast[0:64, 0:64], acc_o[0:64, 0:64], V_shared[0:2, 0:64, 0:64], V[0, by, 128:256, 0:64], Q_shared[0:64, 0:64], K_shared[k % 2, 0:64, 0:64], acc_s[0:64, 0:64], K[0, by, k * 64 + 128:k * 64 + 128 + 64, 0:64], scores_max[0:64], acc_s[0:64, 0:64], scores_max[0:64], acc_s[0:64, 0:64], acc_s[0:64, 0:64], acc_s_cast[0:64, 0:64], V_shared[k % 2, 0:64, 0:64], acc_o[0:64, 0:64], V[0, by, k * 64 + 128:k * 64 + 128 + 64, 0:64])
                    T.writes(acc_s[0:64, 0:64], K_shared[0:2, 0:64, 0:64], scores_max_prev[0:64], scores_max[0:64], scores_scale[0:64], scores_sum[0:64], logsum[0:64], acc_s_cast[0:64, 0:64], acc_o[0:64, 0:64], V_shared[0:2, 0:64, 0:64], acc_s[0:64, 0:64], acc_s[0:64, 0:64], K_shared[k % 2, 0:64, 0:64], scores_max_prev[0:64], scores_max[0:64], scores_max[0:64], scores_sum[0:64], acc_s_cast[0:64, 0:64], acc_o[0:64, 0:64], V_shared[k % 2, 0:64, 0:64])
                    T.sblock_attr({"layout_map": {Q_shared: metadata["tl.Layout"][0], K_shared: metadata["tl.Layout"][1], acc_s: metadata["tl.Fragment"][0], acc_s_cast: metadata["tl.Fragment"][1], V_shared: metadata["tl.Layout"][2], acc_o: metadata["tl.Fragment"][2], scores_max: metadata["tl.Fragment"][3], scores_max_prev: metadata["tl.Fragment"][4], scores_sum: metadata["tl.Fragment"][5], scores_scale: metadata["tl.Fragment"][6], logsum: metadata["tl.Fragment"][7]}})
                    for k in T.serial(2, annotations={"tl_pipelined_num_stages": 2}):
                        with T.sblock(""):
                            T.reads(acc_s[0, 0])
                            T.writes(acc_s[0:64, 0:64])
                            T.sblock_attr({"layout_map": {Q_shared: metadata["tl.Layout"][0], K_shared: metadata["tl.Layout"][1], acc_s: metadata["tl.Fragment"][0], acc_s_cast: metadata["tl.Fragment"][1], V_shared: metadata["tl.Layout"][2], acc_o: metadata["tl.Fragment"][2], scores_max: metadata["tl.Fragment"][3], scores_max_prev: metadata["tl.Fragment"][4], scores_sum: metadata["tl.Fragment"][5], scores_scale: metadata["tl.Fragment"][6], logsum: metadata["tl.Fragment"][7]}})
                            T.fill(T.region(acc_s[0, 0], 2, 64, 64), 0)
                        T.ptx_wait_group(3)
                        with T.sblock(""):
                            T.reads(Q_shared[0, 0], K_shared[0:2, 0, 0], acc_s[0, 0], Q_shared[0:64, 0:64], K_shared[0:2, 0:64, 0:64], acc_s[0:64, 0:64])
                            T.writes(acc_s[0:64, 0:64])
                            T.sblock_attr({"layout_map": {Q_shared: metadata["tl.Layout"][0], K_shared: metadata["tl.Layout"][1], acc_s: metadata["tl.Fragment"][0], acc_s_cast: metadata["tl.Fragment"][1], V_shared: metadata["tl.Layout"][2], acc_o: metadata["tl.Fragment"][2], scores_max: metadata["tl.Fragment"][3], scores_max_prev: metadata["tl.Fragment"][4], scores_sum: metadata["tl.Fragment"][5], scores_scale: metadata["tl.Fragment"][6], logsum: metadata["tl.Fragment"][7]}})
                            T.gemm(T.region(Q_shared[0, 0], 1, 64, 64), T.region(K_shared[k, 0, 0], 1, 1, 64, 64), T.region(acc_s[0, 0], 3, 64, 64), T.bool(False), T.bool(True), 64, 64, 64, 1, T.bool(False), 0, 0, 0)
                        with T.sblock(""):
                            T.reads(K[0, by, k * 64 + 128, 0], K_shared[0:2, 0, 0], K[0, by, k * 64 + 128:k * 64 + 128 + 64, 0:64])
                            T.writes(K_shared[0:2, 0:64, 0:64])
                            T.sblock_attr({"layout_map": {Q_shared: metadata["tl.Layout"][0], K_shared: metadata["tl.Layout"][1], acc_s: metadata["tl.Fragment"][0], acc_s_cast: metadata["tl.Fragment"][1], V_shared: metadata["tl.Layout"][2], acc_o: metadata["tl.Fragment"][2], scores_max: metadata["tl.Fragment"][3], scores_max_prev: metadata["tl.Fragment"][4], scores_sum: metadata["tl.Fragment"][5], scores_scale: metadata["tl.Fragment"][6], logsum: metadata["tl.Fragment"][7]}})
                            T.copy(T.region(K[0, by, k * 64 + 128, 0], 1, 1, 1, 64, 64), T.region(K_shared[k, 0, 0], 2, 1, 64, 64), no_implicit_async_commit_wait=1)
                        T.ptx_commit_group()
                        with T.sblock(""):
                            T.reads(scores_max[0], scores_max_prev[0], scores_max[0:64])
                            T.writes(scores_max_prev[0:64])
                            T.sblock_attr({"layout_map": {Q_shared: metadata["tl.Layout"][0], K_shared: metadata["tl.Layout"][1], acc_s: metadata["tl.Fragment"][0], acc_s_cast: metadata["tl.Fragment"][1], V_shared: metadata["tl.Layout"][2], acc_o: metadata["tl.Fragment"][2], scores_max: metadata["tl.Fragment"][3], scores_max_prev: metadata["tl.Fragment"][4], scores_sum: metadata["tl.Fragment"][5], scores_scale: metadata["tl.Fragment"][6], logsum: metadata["tl.Fragment"][7]}})
                            T.copy(T.region(scores_max[0], 1, 64), T.region(scores_max_prev[0], 2, 64))
                        with T.sblock(""):
                            T.reads(scores_max[0])
                            T.writes(scores_max[0:64])
                            T.sblock_attr({"layout_map": {Q_shared: metadata["tl.Layout"][0], K_shared: metadata["tl.Layout"][1], acc_s: metadata["tl.Fragment"][0], acc_s_cast: metadata["tl.Fragment"][1], V_shared: metadata["tl.Layout"][2], acc_o: metadata["tl.Fragment"][2], scores_max: metadata["tl.Fragment"][3], scores_max_prev: metadata["tl.Fragment"][4], scores_sum: metadata["tl.Fragment"][5], scores_scale: metadata["tl.Fragment"][6], logsum: metadata["tl.Fragment"][7]}})
                            T.fill(T.region(scores_max[0], 2, 64), T.infinity("float32") * T.float32(-1.0))
                        with T.sblock(""):
                            T.reads(acc_s[0, 0], scores_max[0], acc_s[0:64, 0:64], scores_max[0:64])
                            T.writes(scores_max[0:64])
                            T.sblock_attr({"layout_map": {Q_shared: metadata["tl.Layout"][0], K_shared: metadata["tl.Layout"][1], acc_s: metadata["tl.Fragment"][0], acc_s_cast: metadata["tl.Fragment"][1], V_shared: metadata["tl.Layout"][2], acc_o: metadata["tl.Fragment"][2], scores_max: metadata["tl.Fragment"][3], scores_max_prev: metadata["tl.Fragment"][4], scores_sum: metadata["tl.Fragment"][5], scores_scale: metadata["tl.Fragment"][6], logsum: metadata["tl.Fragment"][7]}})
                            T.reduce(T.region(acc_s[0, 0], 1, 64, 64), T.region(scores_max[0], 2, 64), "max", 1, T.bool(False))
                        with T.sblock(""):
                            T.reads(scores_max[0:64], scores_max_prev[0:64])
                            T.writes(scores_max[0:64])
                            T.sblock_attr({"layout_map": {Q_shared: metadata["tl.Layout"][0], K_shared: metadata["tl.Layout"][1], acc_s: metadata["tl.Fragment"][0], acc_s_cast: metadata["tl.Fragment"][1], V_shared: metadata["tl.Layout"][2], acc_o: metadata["tl.Fragment"][2], scores_max: metadata["tl.Fragment"][3], scores_max_prev: metadata["tl.Fragment"][4], scores_sum: metadata["tl.Fragment"][5], scores_scale: metadata["tl.Fragment"][6], logsum: metadata["tl.Fragment"][7]}})
                            for i in T.parallel(64, annotations={"parallel_loop_layout": metadata["tl.Fragment"][3]}):
                                scores_max[i] = T.max(scores_max[i], scores_max_prev[i])
                        with T.sblock(""):
                            T.reads(scores_max_prev[0:64], scores_max[0:64])
                            T.writes(scores_scale[0:64])
                            T.sblock_attr({"layout_map": {Q_shared: metadata["tl.Layout"][0], K_shared: metadata["tl.Layout"][1], acc_s: metadata["tl.Fragment"][0], acc_s_cast: metadata["tl.Fragment"][1], V_shared: metadata["tl.Layout"][2], acc_o: metadata["tl.Fragment"][2], scores_max: metadata["tl.Fragment"][3], scores_max_prev: metadata["tl.Fragment"][4], scores_sum: metadata["tl.Fragment"][5], scores_scale: metadata["tl.Fragment"][6], logsum: metadata["tl.Fragment"][7]}})
                            for i in T.parallel(64, annotations={"parallel_loop_layout": metadata["tl.Fragment"][6]}):
                                scores_scale[i] = T.exp2(scores_max_prev[i] * T.float32(0.18033688) - scores_max[i] * T.float32(0.18033688))
                        with T.sblock(""):
                            T.reads(acc_s[0:64, 0:64], scores_max[0:64])
                            T.writes(acc_s[0:64, 0:64])
                            T.sblock_attr({"layout_map": {Q_shared: metadata["tl.Layout"][0], K_shared: metadata["tl.Layout"][1], acc_s: metadata["tl.Fragment"][0], acc_s_cast: metadata["tl.Fragment"][1], V_shared: metadata["tl.Layout"][2], acc_o: metadata["tl.Fragment"][2], scores_max: metadata["tl.Fragment"][3], scores_max_prev: metadata["tl.Fragment"][4], scores_sum: metadata["tl.Fragment"][5], scores_scale: metadata["tl.Fragment"][6], logsum: metadata["tl.Fragment"][7]}})
                            for i in T.parallel(64, annotations={"parallel_loop_layout": metadata["tl.Fragment"][0]}):
                                for j in T.parallel(64):
                                    acc_s[i, j] = T.exp2(acc_s[i, j] * T.float32(0.18033688) - scores_max[i] * T.float32(0.18033688))
                        with T.sblock(""):
                            T.reads(acc_s[0, 0], scores_sum[0], acc_s[0:64, 0:64])
                            T.writes(scores_sum[0:64])
                            T.sblock_attr({"layout_map": {Q_shared: metadata["tl.Layout"][0], K_shared: metadata["tl.Layout"][1], acc_s: metadata["tl.Fragment"][0], acc_s_cast: metadata["tl.Fragment"][1], V_shared: metadata["tl.Layout"][2], acc_o: metadata["tl.Fragment"][2], scores_max: metadata["tl.Fragment"][3], scores_max_prev: metadata["tl.Fragment"][4], scores_sum: metadata["tl.Fragment"][5], scores_scale: metadata["tl.Fragment"][6], logsum: metadata["tl.Fragment"][7]}})
                            T.reduce(T.region(acc_s[0, 0], 1, 64, 64), T.region(scores_sum[0], 2, 64), "sum", 1, T.bool(True))
                        with T.sblock(""):
                            T.reads(logsum[0:64], scores_scale[0:64], scores_sum[0:64])
                            T.writes(logsum[0:64])
                            T.sblock_attr({"layout_map": {Q_shared: metadata["tl.Layout"][0], K_shared: metadata["tl.Layout"][1], acc_s: metadata["tl.Fragment"][0], acc_s_cast: metadata["tl.Fragment"][1], V_shared: metadata["tl.Layout"][2], acc_o: metadata["tl.Fragment"][2], scores_max: metadata["tl.Fragment"][3], scores_max_prev: metadata["tl.Fragment"][4], scores_sum: metadata["tl.Fragment"][5], scores_scale: metadata["tl.Fragment"][6], logsum: metadata["tl.Fragment"][7]}})
                            for i in T.parallel(64, annotations={"parallel_loop_layout": metadata["tl.Fragment"][7]}):
                                logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
                        with T.sblock(""):
                            T.reads(acc_s[0, 0], acc_s_cast[0, 0], acc_s[0:64, 0:64])
                            T.writes(acc_s_cast[0:64, 0:64])
                            T.sblock_attr({"layout_map": {Q_shared: metadata["tl.Layout"][0], K_shared: metadata["tl.Layout"][1], acc_s: metadata["tl.Fragment"][0], acc_s_cast: metadata["tl.Fragment"][1], V_shared: metadata["tl.Layout"][2], acc_o: metadata["tl.Fragment"][2], scores_max: metadata["tl.Fragment"][3], scores_max_prev: metadata["tl.Fragment"][4], scores_sum: metadata["tl.Fragment"][5], scores_scale: metadata["tl.Fragment"][6], logsum: metadata["tl.Fragment"][7]}})
                            T.copy(T.region(acc_s[0, 0], 1, 64, 64), T.region(acc_s_cast[0, 0], 2, 64, 64))
                        with T.sblock(""):
                            T.reads(acc_o[0:64, 0:64], scores_scale[0:64])
                            T.writes(acc_o[0:64, 0:64])
                            T.sblock_attr({"layout_map": {Q_shared: metadata["tl.Layout"][0], K_shared: metadata["tl.Layout"][1], acc_s: metadata["tl.Fragment"][0], acc_s_cast: metadata["tl.Fragment"][1], V_shared: metadata["tl.Layout"][2], acc_o: metadata["tl.Fragment"][2], scores_max: metadata["tl.Fragment"][3], scores_max_prev: metadata["tl.Fragment"][4], scores_sum: metadata["tl.Fragment"][5], scores_scale: metadata["tl.Fragment"][6], logsum: metadata["tl.Fragment"][7]}})
                            for i in T.parallel(64, annotations={"parallel_loop_layout": metadata["tl.Fragment"][2]}):
                                for j in T.parallel(64):
                                    acc_o[i, j] = acc_o[i, j] * scores_scale[i]
                        T.ptx_wait_group(3)
                        with T.sblock(""):
                            T.reads(acc_s_cast[0, 0], V_shared[0:2, 0, 0], acc_o[0, 0], acc_s_cast[0:64, 0:64], V_shared[0:2, 0:64, 0:64], acc_o[0:64, 0:64])
                            T.writes(acc_o[0:64, 0:64])
                            T.sblock_attr({"layout_map": {Q_shared: metadata["tl.Layout"][0], K_shared: metadata["tl.Layout"][1], acc_s: metadata["tl.Fragment"][0], acc_s_cast: metadata["tl.Fragment"][1], V_shared: metadata["tl.Layout"][2], acc_o: metadata["tl.Fragment"][2], scores_max: metadata["tl.Fragment"][3], scores_max_prev: metadata["tl.Fragment"][4], scores_sum: metadata["tl.Fragment"][5], scores_scale: metadata["tl.Fragment"][6], logsum: metadata["tl.Fragment"][7]}})
                            T.gemm(T.region(acc_s_cast[0, 0], 1, 64, 64), T.region(V_shared[k, 0, 0], 1, 1, 64, 64), T.region(acc_o[0, 0], 3, 64, 64), T.bool(False), T.bool(False), 64, 64, 64, 1, T.bool(False), 0, 0, 0)
                        with T.sblock(""):
                            T.reads(V[0, by, k * 64 + 128, 0], V_shared[0:2, 0, 0], V[0, by, k * 64 + 128:k * 64 + 128 + 64, 0:64])
                            T.writes(V_shared[0:2, 0:64, 0:64])
                            T.sblock_attr({"layout_map": {Q_shared: metadata["tl.Layout"][0], K_shared: metadata["tl.Layout"][1], acc_s: metadata["tl.Fragment"][0], acc_s_cast: metadata["tl.Fragment"][1], V_shared: metadata["tl.Layout"][2], acc_o: metadata["tl.Fragment"][2], scores_max: metadata["tl.Fragment"][3], scores_max_prev: metadata["tl.Fragment"][4], scores_sum: metadata["tl.Fragment"][5], scores_scale: metadata["tl.Fragment"][6], logsum: metadata["tl.Fragment"][7]}})
                            T.copy(T.region(V[0, by, k * 64 + 128, 0], 1, 1, 1, 64, 64), T.region(V_shared[k, 0, 0], 2, 1, 64, 64), no_implicit_async_commit_wait=1)
                        T.ptx_commit_group()
                with T.sblock(""):
                    T.reads(acc_s[0:64, 0:64], Q_shared[0:64, 0:64], K_shared[0:2, 0:64, 0:64], scores_max[0:64], scores_max_prev[0:64], scores_sum[0:64], logsum[0:64], scores_scale[0:64], acc_s_cast[0:64, 0:64], acc_o[0:64, 0:64], V_shared[0:2, 0:64, 0:64], Q_shared[0:64, 0:64], K_shared[0, 0:64, 0:64], acc_s[0:64, 0:64], scores_max[0:64], acc_s[0:64, 0:64], scores_max[0:64], acc_s[0:64, 0:64], acc_s[0:64, 0:64], acc_s_cast[0:64, 0:64], V_shared[0, 0:64, 0:64], acc_o[0:64, 0:64])
                    T.writes(acc_s[0:64, 0:64], scores_max_prev[0:64], scores_max[0:64], scores_scale[0:64], scores_sum[0:64], logsum[0:64], acc_s_cast[0:64, 0:64], acc_o[0:64, 0:64], acc_s[0:64, 0:64], acc_s[0:64, 0:64], scores_max_prev[0:64], scores_max[0:64], scores_max[0:64], scores_sum[0:64], acc_s_cast[0:64, 0:64], acc_o[0:64, 0:64])
                    T.sblock_attr({"layout_map": {Q_shared: metadata["tl.Layout"][0], K_shared: metadata["tl.Layout"][1], acc_s: metadata["tl.Fragment"][0], acc_s_cast: metadata["tl.Fragment"][1], V_shared: metadata["tl.Layout"][2], acc_o: metadata["tl.Fragment"][2], scores_max: metadata["tl.Fragment"][3], scores_max_prev: metadata["tl.Fragment"][4], scores_sum: metadata["tl.Fragment"][5], scores_scale: metadata["tl.Fragment"][6], logsum: metadata["tl.Fragment"][7]}})
                    with T.sblock(""):
                        T.reads(acc_s[0, 0])
                        T.writes(acc_s[0:64, 0:64])
                        T.sblock_attr({"layout_map": {Q_shared: metadata["tl.Layout"][0], K_shared: metadata["tl.Layout"][1], acc_s: metadata["tl.Fragment"][0], acc_s_cast: metadata["tl.Fragment"][1], V_shared: metadata["tl.Layout"][2], acc_o: metadata["tl.Fragment"][2], scores_max: metadata["tl.Fragment"][3], scores_max_prev: metadata["tl.Fragment"][4], scores_sum: metadata["tl.Fragment"][5], scores_scale: metadata["tl.Fragment"][6], logsum: metadata["tl.Fragment"][7]}})
                        T.fill(T.region(acc_s[0, 0], 2, 64, 64), 0)
                    T.ptx_wait_group(3)
                    with T.sblock(""):
                        T.reads(Q_shared[0, 0], K_shared[0:2, 0, 0], acc_s[0, 0], Q_shared[0:64, 0:64], K_shared[0:2, 0:64, 0:64], acc_s[0:64, 0:64])
                        T.writes(acc_s[0:64, 0:64])
                        T.sblock_attr({"layout_map": {Q_shared: metadata["tl.Layout"][0], K_shared: metadata["tl.Layout"][1], acc_s: metadata["tl.Fragment"][0], acc_s_cast: metadata["tl.Fragment"][1], V_shared: metadata["tl.Layout"][2], acc_o: metadata["tl.Fragment"][2], scores_max: metadata["tl.Fragment"][3], scores_max_prev: metadata["tl.Fragment"][4], scores_sum: metadata["tl.Fragment"][5], scores_scale: metadata["tl.Fragment"][6], logsum: metadata["tl.Fragment"][7]}})
                        T.gemm(T.region(Q_shared[0, 0], 1, 64, 64), T.region(K_shared[0, 0, 0], 1, 1, 64, 64), T.region(acc_s[0, 0], 3, 64, 64), T.bool(False), T.bool(True), 64, 64, 64, 1, T.bool(False), 0, 0, 0)
                    with T.sblock(""):
                        T.reads(scores_max[0], scores_max_prev[0], scores_max[0:64])
                        T.writes(scores_max_prev[0:64])
                        T.sblock_attr({"layout_map": {Q_shared: metadata["tl.Layout"][0], K_shared: metadata["tl.Layout"][1], acc_s: metadata["tl.Fragment"][0], acc_s_cast: metadata["tl.Fragment"][1], V_shared: metadata["tl.Layout"][2], acc_o: metadata["tl.Fragment"][2], scores_max: metadata["tl.Fragment"][3], scores_max_prev: metadata["tl.Fragment"][4], scores_sum: metadata["tl.Fragment"][5], scores_scale: metadata["tl.Fragment"][6], logsum: metadata["tl.Fragment"][7]}})
                        T.copy(T.region(scores_max[0], 1, 64), T.region(scores_max_prev[0], 2, 64))
                    with T.sblock(""):
                        T.reads(scores_max[0])
                        T.writes(scores_max[0:64])
                        T.sblock_attr({"layout_map": {Q_shared: metadata["tl.Layout"][0], K_shared: metadata["tl.Layout"][1], acc_s: metadata["tl.Fragment"][0], acc_s_cast: metadata["tl.Fragment"][1], V_shared: metadata["tl.Layout"][2], acc_o: metadata["tl.Fragment"][2], scores_max: metadata["tl.Fragment"][3], scores_max_prev: metadata["tl.Fragment"][4], scores_sum: metadata["tl.Fragment"][5], scores_scale: metadata["tl.Fragment"][6], logsum: metadata["tl.Fragment"][7]}})
                        T.fill(T.region(scores_max[0], 2, 64), T.infinity("float32") * T.float32(-1.0))
                    with T.sblock(""):
                        T.reads(acc_s[0, 0], scores_max[0], acc_s[0:64, 0:64], scores_max[0:64])
                        T.writes(scores_max[0:64])
                        T.sblock_attr({"layout_map": {Q_shared: metadata["tl.Layout"][0], K_shared: metadata["tl.Layout"][1], acc_s: metadata["tl.Fragment"][0], acc_s_cast: metadata["tl.Fragment"][1], V_shared: metadata["tl.Layout"][2], acc_o: metadata["tl.Fragment"][2], scores_max: metadata["tl.Fragment"][3], scores_max_prev: metadata["tl.Fragment"][4], scores_sum: metadata["tl.Fragment"][5], scores_scale: metadata["tl.Fragment"][6], logsum: metadata["tl.Fragment"][7]}})
                        T.reduce(T.region(acc_s[0, 0], 1, 64, 64), T.region(scores_max[0], 2, 64), "max", 1, T.bool(False))
                    with T.sblock(""):
                        T.reads(scores_max[0:64], scores_max_prev[0:64])
                        T.writes(scores_max[0:64])
                        T.sblock_attr({"layout_map": {Q_shared: metadata["tl.Layout"][0], K_shared: metadata["tl.Layout"][1], acc_s: metadata["tl.Fragment"][0], acc_s_cast: metadata["tl.Fragment"][1], V_shared: metadata["tl.Layout"][2], acc_o: metadata["tl.Fragment"][2], scores_max: metadata["tl.Fragment"][3], scores_max_prev: metadata["tl.Fragment"][4], scores_sum: metadata["tl.Fragment"][5], scores_scale: metadata["tl.Fragment"][6], logsum: metadata["tl.Fragment"][7]}})
                        for i in T.parallel(64, annotations={"parallel_loop_layout": metadata["tl.Fragment"][3]}):
                            scores_max[i] = T.max(scores_max[i], scores_max_prev[i])
                    with T.sblock(""):
                        T.reads(scores_max_prev[0:64], scores_max[0:64])
                        T.writes(scores_scale[0:64])
                        T.sblock_attr({"layout_map": {Q_shared: metadata["tl.Layout"][0], K_shared: metadata["tl.Layout"][1], acc_s: metadata["tl.Fragment"][0], acc_s_cast: metadata["tl.Fragment"][1], V_shared: metadata["tl.Layout"][2], acc_o: metadata["tl.Fragment"][2], scores_max: metadata["tl.Fragment"][3], scores_max_prev: metadata["tl.Fragment"][4], scores_sum: metadata["tl.Fragment"][5], scores_scale: metadata["tl.Fragment"][6], logsum: metadata["tl.Fragment"][7]}})
                        for i in T.parallel(64, annotations={"parallel_loop_layout": metadata["tl.Fragment"][6]}):
                            scores_scale[i] = T.exp2(scores_max_prev[i] * T.float32(0.18033688) - scores_max[i] * T.float32(0.18033688))
                    with T.sblock(""):
                        T.reads(acc_s[0:64, 0:64], scores_max[0:64])
                        T.writes(acc_s[0:64, 0:64])
                        T.sblock_attr({"layout_map": {Q_shared: metadata["tl.Layout"][0], K_shared: metadata["tl.Layout"][1], acc_s: metadata["tl.Fragment"][0], acc_s_cast: metadata["tl.Fragment"][1], V_shared: metadata["tl.Layout"][2], acc_o: metadata["tl.Fragment"][2], scores_max: metadata["tl.Fragment"][3], scores_max_prev: metadata["tl.Fragment"][4], scores_sum: metadata["tl.Fragment"][5], scores_scale: metadata["tl.Fragment"][6], logsum: metadata["tl.Fragment"][7]}})
                        for i in T.parallel(64, annotations={"parallel_loop_layout": metadata["tl.Fragment"][0]}):
                            for j in T.parallel(64):
                                acc_s[i, j] = T.exp2(acc_s[i, j] * T.float32(0.18033688) - scores_max[i] * T.float32(0.18033688))
                    with T.sblock(""):
                        T.reads(acc_s[0, 0], scores_sum[0], acc_s[0:64, 0:64])
                        T.writes(scores_sum[0:64])
                        T.sblock_attr({"layout_map": {Q_shared: metadata["tl.Layout"][0], K_shared: metadata["tl.Layout"][1], acc_s: metadata["tl.Fragment"][0], acc_s_cast: metadata["tl.Fragment"][1], V_shared: metadata["tl.Layout"][2], acc_o: metadata["tl.Fragment"][2], scores_max: metadata["tl.Fragment"][3], scores_max_prev: metadata["tl.Fragment"][4], scores_sum: metadata["tl.Fragment"][5], scores_scale: metadata["tl.Fragment"][6], logsum: metadata["tl.Fragment"][7]}})
                        T.reduce(T.region(acc_s[0, 0], 1, 64, 64), T.region(scores_sum[0], 2, 64), "sum", 1, T.bool(True))
                    with T.sblock(""):
                        T.reads(logsum[0:64], scores_scale[0:64], scores_sum[0:64])
                        T.writes(logsum[0:64])
                        T.sblock_attr({"layout_map": {Q_shared: metadata["tl.Layout"][0], K_shared: metadata["tl.Layout"][1], acc_s: metadata["tl.Fragment"][0], acc_s_cast: metadata["tl.Fragment"][1], V_shared: metadata["tl.Layout"][2], acc_o: metadata["tl.Fragment"][2], scores_max: metadata["tl.Fragment"][3], scores_max_prev: metadata["tl.Fragment"][4], scores_sum: metadata["tl.Fragment"][5], scores_scale: metadata["tl.Fragment"][6], logsum: metadata["tl.Fragment"][7]}})
                        for i in T.parallel(64, annotations={"parallel_loop_layout": metadata["tl.Fragment"][7]}):
                            logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
                    with T.sblock(""):
                        T.reads(acc_s[0, 0], acc_s_cast[0, 0], acc_s[0:64, 0:64])
                        T.writes(acc_s_cast[0:64, 0:64])
                        T.sblock_attr({"layout_map": {Q_shared: metadata["tl.Layout"][0], K_shared: metadata["tl.Layout"][1], acc_s: metadata["tl.Fragment"][0], acc_s_cast: metadata["tl.Fragment"][1], V_shared: metadata["tl.Layout"][2], acc_o: metadata["tl.Fragment"][2], scores_max: metadata["tl.Fragment"][3], scores_max_prev: metadata["tl.Fragment"][4], scores_sum: metadata["tl.Fragment"][5], scores_scale: metadata["tl.Fragment"][6], logsum: metadata["tl.Fragment"][7]}})
                        T.copy(T.region(acc_s[0, 0], 1, 64, 64), T.region(acc_s_cast[0, 0], 2, 64, 64))
                    with T.sblock(""):
                        T.reads(acc_o[0:64, 0:64], scores_scale[0:64])
                        T.writes(acc_o[0:64, 0:64])
                        T.sblock_attr({"layout_map": {Q_shared: metadata["tl.Layout"][0], K_shared: metadata["tl.Layout"][1], acc_s: metadata["tl.Fragment"][0], acc_s_cast: metadata["tl.Fragment"][1], V_shared: metadata["tl.Layout"][2], acc_o: metadata["tl.Fragment"][2], scores_max: metadata["tl.Fragment"][3], scores_max_prev: metadata["tl.Fragment"][4], scores_sum: metadata["tl.Fragment"][5], scores_scale: metadata["tl.Fragment"][6], logsum: metadata["tl.Fragment"][7]}})
                        for i in T.parallel(64, annotations={"parallel_loop_layout": metadata["tl.Fragment"][2]}):
                            for j in T.parallel(64):
                                acc_o[i, j] = acc_o[i, j] * scores_scale[i]
                    T.ptx_wait_group(2)
                    with T.sblock(""):
                        T.reads(acc_s_cast[0, 0], V_shared[0:2, 0, 0], acc_o[0, 0], acc_s_cast[0:64, 0:64], V_shared[0:2, 0:64, 0:64], acc_o[0:64, 0:64])
                        T.writes(acc_o[0:64, 0:64])
                        T.sblock_attr({"layout_map": {Q_shared: metadata["tl.Layout"][0], K_shared: metadata["tl.Layout"][1], acc_s: metadata["tl.Fragment"][0], acc_s_cast: metadata["tl.Fragment"][1], V_shared: metadata["tl.Layout"][2], acc_o: metadata["tl.Fragment"][2], scores_max: metadata["tl.Fragment"][3], scores_max_prev: metadata["tl.Fragment"][4], scores_sum: metadata["tl.Fragment"][5], scores_scale: metadata["tl.Fragment"][6], logsum: metadata["tl.Fragment"][7]}})
                        T.gemm(T.region(acc_s_cast[0, 0], 1, 64, 64), T.region(V_shared[0, 0, 0], 1, 1, 64, 64), T.region(acc_o[0, 0], 3, 64, 64), T.bool(False), T.bool(False), 64, 64, 64, 1, T.bool(False), 0, 0, 0)
                with T.sblock(""):
                    T.reads(acc_s[0:64, 0:64], Q_shared[0:64, 0:64], K_shared[0:2, 0:64, 0:64], scores_max[0:64], scores_max_prev[0:64], scores_sum[0:64], logsum[0:64], scores_scale[0:64], acc_s_cast[0:64, 0:64], acc_o[0:64, 0:64], V_shared[0:2, 0:64, 0:64], Q_shared[0:64, 0:64], K_shared[1, 0:64, 0:64], acc_s[0:64, 0:64], scores_max[0:64], acc_s[0:64, 0:64], scores_max[0:64], acc_s[0:64, 0:64], acc_s[0:64, 0:64], acc_s_cast[0:64, 0:64], V_shared[1, 0:64, 0:64], acc_o[0:64, 0:64])
                    T.writes(acc_s[0:64, 0:64], scores_max_prev[0:64], scores_max[0:64], scores_scale[0:64], scores_sum[0:64], logsum[0:64], acc_s_cast[0:64, 0:64], acc_o[0:64, 0:64], acc_s[0:64, 0:64], acc_s[0:64, 0:64], scores_max_prev[0:64], scores_max[0:64], scores_max[0:64], scores_sum[0:64], acc_s_cast[0:64, 0:64], acc_o[0:64, 0:64])
                    T.sblock_attr({"layout_map": {Q_shared: metadata["tl.Layout"][0], K_shared: metadata["tl.Layout"][1], acc_s: metadata["tl.Fragment"][0], acc_s_cast: metadata["tl.Fragment"][1], V_shared: metadata["tl.Layout"][2], acc_o: metadata["tl.Fragment"][2], scores_max: metadata["tl.Fragment"][3], scores_max_prev: metadata["tl.Fragment"][4], scores_sum: metadata["tl.Fragment"][5], scores_scale: metadata["tl.Fragment"][6], logsum: metadata["tl.Fragment"][7]}})
                    with T.sblock(""):
                        T.reads(acc_s[0, 0])
                        T.writes(acc_s[0:64, 0:64])
                        T.sblock_attr({"layout_map": {Q_shared: metadata["tl.Layout"][0], K_shared: metadata["tl.Layout"][1], acc_s: metadata["tl.Fragment"][0], acc_s_cast: metadata["tl.Fragment"][1], V_shared: metadata["tl.Layout"][2], acc_o: metadata["tl.Fragment"][2], scores_max: metadata["tl.Fragment"][3], scores_max_prev: metadata["tl.Fragment"][4], scores_sum: metadata["tl.Fragment"][5], scores_scale: metadata["tl.Fragment"][6], logsum: metadata["tl.Fragment"][7]}})
                        T.fill(T.region(acc_s[0, 0], 2, 64, 64), 0)
                    T.ptx_wait_group(1)
                    with T.sblock(""):
                        T.reads(Q_shared[0, 0], K_shared[0:2, 0, 0], acc_s[0, 0], Q_shared[0:64, 0:64], K_shared[0:2, 0:64, 0:64], acc_s[0:64, 0:64])
                        T.writes(acc_s[0:64, 0:64])
                        T.sblock_attr({"layout_map": {Q_shared: metadata["tl.Layout"][0], K_shared: metadata["tl.Layout"][1], acc_s: metadata["tl.Fragment"][0], acc_s_cast: metadata["tl.Fragment"][1], V_shared: metadata["tl.Layout"][2], acc_o: metadata["tl.Fragment"][2], scores_max: metadata["tl.Fragment"][3], scores_max_prev: metadata["tl.Fragment"][4], scores_sum: metadata["tl.Fragment"][5], scores_scale: metadata["tl.Fragment"][6], logsum: metadata["tl.Fragment"][7]}})
                        T.gemm(T.region(Q_shared[0, 0], 1, 64, 64), T.region(K_shared[1, 0, 0], 1, 1, 64, 64), T.region(acc_s[0, 0], 3, 64, 64), T.bool(False), T.bool(True), 64, 64, 64, 1, T.bool(False), 0, 0, 0)
                    with T.sblock(""):
                        T.reads(scores_max[0], scores_max_prev[0], scores_max[0:64])
                        T.writes(scores_max_prev[0:64])
                        T.sblock_attr({"layout_map": {Q_shared: metadata["tl.Layout"][0], K_shared: metadata["tl.Layout"][1], acc_s: metadata["tl.Fragment"][0], acc_s_cast: metadata["tl.Fragment"][1], V_shared: metadata["tl.Layout"][2], acc_o: metadata["tl.Fragment"][2], scores_max: metadata["tl.Fragment"][3], scores_max_prev: metadata["tl.Fragment"][4], scores_sum: metadata["tl.Fragment"][5], scores_scale: metadata["tl.Fragment"][6], logsum: metadata["tl.Fragment"][7]}})
                        T.copy(T.region(scores_max[0], 1, 64), T.region(scores_max_prev[0], 2, 64))
                    with T.sblock(""):
                        T.reads(scores_max[0])
                        T.writes(scores_max[0:64])
                        T.sblock_attr({"layout_map": {Q_shared: metadata["tl.Layout"][0], K_shared: metadata["tl.Layout"][1], acc_s: metadata["tl.Fragment"][0], acc_s_cast: metadata["tl.Fragment"][1], V_shared: metadata["tl.Layout"][2], acc_o: metadata["tl.Fragment"][2], scores_max: metadata["tl.Fragment"][3], scores_max_prev: metadata["tl.Fragment"][4], scores_sum: metadata["tl.Fragment"][5], scores_scale: metadata["tl.Fragment"][6], logsum: metadata["tl.Fragment"][7]}})
                        T.fill(T.region(scores_max[0], 2, 64), T.infinity("float32") * T.float32(-1.0))
                    with T.sblock(""):
                        T.reads(acc_s[0, 0], scores_max[0], acc_s[0:64, 0:64], scores_max[0:64])
                        T.writes(scores_max[0:64])
                        T.sblock_attr({"layout_map": {Q_shared: metadata["tl.Layout"][0], K_shared: metadata["tl.Layout"][1], acc_s: metadata["tl.Fragment"][0], acc_s_cast: metadata["tl.Fragment"][1], V_shared: metadata["tl.Layout"][2], acc_o: metadata["tl.Fragment"][2], scores_max: metadata["tl.Fragment"][3], scores_max_prev: metadata["tl.Fragment"][4], scores_sum: metadata["tl.Fragment"][5], scores_scale: metadata["tl.Fragment"][6], logsum: metadata["tl.Fragment"][7]}})
                        T.reduce(T.region(acc_s[0, 0], 1, 64, 64), T.region(scores_max[0], 2, 64), "max", 1, T.bool(False))
                    with T.sblock(""):
                        T.reads(scores_max[0:64], scores_max_prev[0:64])
                        T.writes(scores_max[0:64])
                        T.sblock_attr({"layout_map": {Q_shared: metadata["tl.Layout"][0], K_shared: metadata["tl.Layout"][1], acc_s: metadata["tl.Fragment"][0], acc_s_cast: metadata["tl.Fragment"][1], V_shared: metadata["tl.Layout"][2], acc_o: metadata["tl.Fragment"][2], scores_max: metadata["tl.Fragment"][3], scores_max_prev: metadata["tl.Fragment"][4], scores_sum: metadata["tl.Fragment"][5], scores_scale: metadata["tl.Fragment"][6], logsum: metadata["tl.Fragment"][7]}})
                        for i in T.parallel(64, annotations={"parallel_loop_layout": metadata["tl.Fragment"][3]}):
                            scores_max[i] = T.max(scores_max[i], scores_max_prev[i])
                    with T.sblock(""):
                        T.reads(scores_max_prev[0:64], scores_max[0:64])
                        T.writes(scores_scale[0:64])
                        T.sblock_attr({"layout_map": {Q_shared: metadata["tl.Layout"][0], K_shared: metadata["tl.Layout"][1], acc_s: metadata["tl.Fragment"][0], acc_s_cast: metadata["tl.Fragment"][1], V_shared: metadata["tl.Layout"][2], acc_o: metadata["tl.Fragment"][2], scores_max: metadata["tl.Fragment"][3], scores_max_prev: metadata["tl.Fragment"][4], scores_sum: metadata["tl.Fragment"][5], scores_scale: metadata["tl.Fragment"][6], logsum: metadata["tl.Fragment"][7]}})
                        for i in T.parallel(64, annotations={"parallel_loop_layout": metadata["tl.Fragment"][6]}):
                            scores_scale[i] = T.exp2(scores_max_prev[i] * T.float32(0.18033688) - scores_max[i] * T.float32(0.18033688))
                    with T.sblock(""):
                        T.reads(acc_s[0:64, 0:64], scores_max[0:64])
                        T.writes(acc_s[0:64, 0:64])
                        T.sblock_attr({"layout_map": {Q_shared: metadata["tl.Layout"][0], K_shared: metadata["tl.Layout"][1], acc_s: metadata["tl.Fragment"][0], acc_s_cast: metadata["tl.Fragment"][1], V_shared: metadata["tl.Layout"][2], acc_o: metadata["tl.Fragment"][2], scores_max: metadata["tl.Fragment"][3], scores_max_prev: metadata["tl.Fragment"][4], scores_sum: metadata["tl.Fragment"][5], scores_scale: metadata["tl.Fragment"][6], logsum: metadata["tl.Fragment"][7]}})
                        for i in T.parallel(64, annotations={"parallel_loop_layout": metadata["tl.Fragment"][0]}):
                            for j in T.parallel(64):
                                acc_s[i, j] = T.exp2(acc_s[i, j] * T.float32(0.18033688) - scores_max[i] * T.float32(0.18033688))
                    with T.sblock(""):
                        T.reads(acc_s[0, 0], scores_sum[0], acc_s[0:64, 0:64])
                        T.writes(scores_sum[0:64])
                        T.sblock_attr({"layout_map": {Q_shared: metadata["tl.Layout"][0], K_shared: metadata["tl.Layout"][1], acc_s: metadata["tl.Fragment"][0], acc_s_cast: metadata["tl.Fragment"][1], V_shared: metadata["tl.Layout"][2], acc_o: metadata["tl.Fragment"][2], scores_max: metadata["tl.Fragment"][3], scores_max_prev: metadata["tl.Fragment"][4], scores_sum: metadata["tl.Fragment"][5], scores_scale: metadata["tl.Fragment"][6], logsum: metadata["tl.Fragment"][7]}})
                        T.reduce(T.region(acc_s[0, 0], 1, 64, 64), T.region(scores_sum[0], 2, 64), "sum", 1, T.bool(True))
                    with T.sblock(""):
                        T.reads(logsum[0:64], scores_scale[0:64], scores_sum[0:64])
                        T.writes(logsum[0:64])
                        T.sblock_attr({"layout_map": {Q_shared: metadata["tl.Layout"][0], K_shared: metadata["tl.Layout"][1], acc_s: metadata["tl.Fragment"][0], acc_s_cast: metadata["tl.Fragment"][1], V_shared: metadata["tl.Layout"][2], acc_o: metadata["tl.Fragment"][2], scores_max: metadata["tl.Fragment"][3], scores_max_prev: metadata["tl.Fragment"][4], scores_sum: metadata["tl.Fragment"][5], scores_scale: metadata["tl.Fragment"][6], logsum: metadata["tl.Fragment"][7]}})
                        for i in T.parallel(64, annotations={"parallel_loop_layout": metadata["tl.Fragment"][7]}):
                            logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
                    with T.sblock(""):
                        T.reads(acc_s[0, 0], acc_s_cast[0, 0], acc_s[0:64, 0:64])
                        T.writes(acc_s_cast[0:64, 0:64])
                        T.sblock_attr({"layout_map": {Q_shared: metadata["tl.Layout"][0], K_shared: metadata["tl.Layout"][1], acc_s: metadata["tl.Fragment"][0], acc_s_cast: metadata["tl.Fragment"][1], V_shared: metadata["tl.Layout"][2], acc_o: metadata["tl.Fragment"][2], scores_max: metadata["tl.Fragment"][3], scores_max_prev: metadata["tl.Fragment"][4], scores_sum: metadata["tl.Fragment"][5], scores_scale: metadata["tl.Fragment"][6], logsum: metadata["tl.Fragment"][7]}})
                        T.copy(T.region(acc_s[0, 0], 1, 64, 64), T.region(acc_s_cast[0, 0], 2, 64, 64))
                    with T.sblock(""):
                        T.reads(acc_o[0:64, 0:64], scores_scale[0:64])
                        T.writes(acc_o[0:64, 0:64])
                        T.sblock_attr({"layout_map": {Q_shared: metadata["tl.Layout"][0], K_shared: metadata["tl.Layout"][1], acc_s: metadata["tl.Fragment"][0], acc_s_cast: metadata["tl.Fragment"][1], V_shared: metadata["tl.Layout"][2], acc_o: metadata["tl.Fragment"][2], scores_max: metadata["tl.Fragment"][3], scores_max_prev: metadata["tl.Fragment"][4], scores_sum: metadata["tl.Fragment"][5], scores_scale: metadata["tl.Fragment"][6], logsum: metadata["tl.Fragment"][7]}})
                        for i in T.parallel(64, annotations={"parallel_loop_layout": metadata["tl.Fragment"][2]}):
                            for j in T.parallel(64):
                                acc_o[i, j] = acc_o[i, j] * scores_scale[i]
                    T.ptx_wait_group(0)
                    with T.sblock(""):
                        T.reads(acc_s_cast[0, 0], V_shared[0:2, 0, 0], acc_o[0, 0], acc_s_cast[0:64, 0:64], V_shared[0:2, 0:64, 0:64], acc_o[0:64, 0:64])
                        T.writes(acc_o[0:64, 0:64])
                        T.sblock_attr({"layout_map": {Q_shared: metadata["tl.Layout"][0], K_shared: metadata["tl.Layout"][1], acc_s: metadata["tl.Fragment"][0], acc_s_cast: metadata["tl.Fragment"][1], V_shared: metadata["tl.Layout"][2], acc_o: metadata["tl.Fragment"][2], scores_max: metadata["tl.Fragment"][3], scores_max_prev: metadata["tl.Fragment"][4], scores_sum: metadata["tl.Fragment"][5], scores_scale: metadata["tl.Fragment"][6], logsum: metadata["tl.Fragment"][7]}})
                        T.gemm(T.region(acc_s_cast[0, 0], 1, 64, 64), T.region(V_shared[1, 0, 0], 1, 1, 64, 64), T.region(acc_o[0, 0], 3, 64, 64), T.bool(False), T.bool(False), 64, 64, 64, 1, T.bool(False), 0, 0, 0)
                for i in T.parallel(64, annotations={"parallel_loop_layout": metadata["tl.Fragment"][2]}):
                    for j in T.parallel(64):
                        acc_o[i, j] = acc_o[i, j] / logsum[i]
                T.copy(T.region(acc_o[0, 0], 1, 64, 64), T.region(Output[0, by, bx * 64, 0], 2, 1, 1, 64, 64))

# Metadata omitted. Use show_meta=True in script() method to show it.