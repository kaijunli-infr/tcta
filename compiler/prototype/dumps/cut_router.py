# E3/F2 output: lower_to_tile_tir(moe_kernel.cut_router) at tl.InjectSoftwarePipeline
# from tvm.script import tirx as T

@T.prim_func
def router(logits_handle: T.handle, topk_handle: T.handle, bx: T.int32):
    T.func_attr({"target": T.target({"arch": "sm_80", "keys": ["cuda", "gpu"], "kind": "cuda", "max_num_threads": 1024, "tag": "", "thread_warp_size": 32}), "tcta.arch": "sm_80", "tcta.cut": "tl.InjectSoftwarePipeline"})
    logits = T.match_buffer(logits_handle, (256, 32), strides=(32, 1))
    topk = T.match_buffer(topk_handle, (256, 8), "int32", strides=(8, 1))
    with T.sblock("root"):
        T.reads()
        T.writes()
        with T.sblock("tilelang_root"):
            T.reads()
            T.writes()
            row = T.sblock_alloc_buffer((1, 32), scope="tcta.vec")
            idx = T.sblock_alloc_buffer((1, 8), "int32", scope="tcta.vec")
            T.copy(T.region(logits[bx, 0], 1, 1, 32), T.region(row[0, 0], 2, 1, 32))
            T.call_extern("handle", "tcta.vtopk", T.address_of(idx[0, 0]), T.address_of(row[0, 0]), 6)
            T.copy(T.region(idx[0, 0], 1, 1, 6), T.region(topk[bx, 0], 2, 1, 6))