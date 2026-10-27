# E3/F2 output: lower_to_tile_tir(moe_kernel.cut_moe_combine) at tl.InjectSoftwarePipeline
# from tvm.script import tirx as T

@T.prim_func
def combine(Yp_handle: T.handle, y_shared_handle: T.handle, rows_idx_handle: T.handle, y_handle: T.handle, bx: T.int32):
    T.func_attr({"target": T.target({"arch": "sm_80", "keys": ["cuda", "gpu"], "kind": "cuda", "max_num_threads": 1024, "tag": "", "thread_warp_size": 32}), "tcta.arch": "sm_80", "tcta.cut": "tl.InjectSoftwarePipeline"})
    Yp = T.match_buffer(Yp_handle, (1536, 512), strides=(512, 1))
    y_shared = T.match_buffer(y_shared_handle, (256, 512), strides=(512, 1))
    rows_idx = T.match_buffer(rows_idx_handle, (6, 256), "int32", strides=(256, 1))
    y = T.match_buffer(y_handle, (256, 512), strides=(512, 1))
    with T.sblock("root"):
        T.reads()
        T.writes()
        with T.sblock("tilelang_root"):
            T.reads()
            T.writes()
            acc = T.sblock_alloc_buffer((64, 128), scope="tcta.acc")
            part = T.sblock_alloc_buffer((64, 128), scope="tcta.acc")
            idx_s = T.sblock_alloc_buffer((64,), "int32", scope="tcta.vec")
            for no in range(4):
                T.fill(T.region(acc[0, 0], 2, 64, 128), 0)
                T.copy(T.region(y_shared[bx * 64, no * 128], 1, 64, 128), T.region(part[0, 0], 2, 64, 128))
                for i in T.parallel(64):
                    for j in T.parallel(128):
                        acc[i, j] = acc[i, j] + part[i, j]
                for s in range(6):
                    T.copy(T.region(rows_idx[s, bx * 64], 1, 1, 64), T.region(idx_s[0], 2, 64))
                    for i in T.parallel(64):
                        for j in T.parallel(128):
                            part[i, j] = T.if_then_else(0 <= idx_s[i], Yp[idx_s[i], no * 128 + j], T.float32(0.0))
                    for i in T.parallel(64):
                        for j in T.parallel(128):
                            acc[i, j] = acc[i, j] + part[i, j]
                T.copy(T.region(acc[0, 0], 1, 64, 128), T.region(y[bx * 64, no * 128], 2, 64, 128))