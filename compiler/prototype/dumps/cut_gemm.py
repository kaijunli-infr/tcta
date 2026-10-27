# G1/G2 output: lower_to_tile_tir(gemm_kernel.build_gemm_kernel()) at tl.InjectSoftwarePipeline
# from tvm.script import tirx as T

@T.prim_func
def main(A_handle: T.handle, B_handle: T.handle, C_handle: T.handle, bx: T.int32):
    T.func_attr({"target": T.target({"arch": "sm_80", "keys": ["cuda", "gpu"], "kind": "cuda", "max_num_threads": 1024, "tag": "", "thread_warp_size": 32}), "tcta.arch": "sm_80", "tcta.cut": "tl.InjectSoftwarePipeline"})
    A = T.match_buffer(A_handle, (64, 128), "float16", strides=(128, 1))
    B = T.match_buffer(B_handle, (128, 128), "float16", strides=(128, 1))
    C = T.match_buffer(C_handle, (50, 96), strides=(96, 1))
    with T.sblock("root"):
        T.reads()
        T.writes()
        with T.sblock("tilelang_root"):
            T.reads()
            T.writes()
            A_s = T.sblock_alloc_buffer((50, 100), "float16", scope="tcta.inp")
            B_s = T.sblock_alloc_buffer((100, 96), "float16", scope="tcta.wgt")
            C_acc = T.sblock_alloc_buffer((50, 96), scope="tcta.acc")
            T.copy(T.region(A[0, 0], 1, 50, 100), T.region(A_s[0, 0], 2, 50, 100))
            T.copy(T.region(B[0, 0], 1, 100, 96), T.region(B_s[0, 0], 2, 100, 96))
            T.fill(T.region(C_acc[0, 0], 2, 50, 96), 0)
            T.gemm(T.region(A_s[0, 0], 1, 50, 100), T.region(B_s[0, 0], 1, 100, 96), T.region(C_acc[0, 0], 3, 50, 96), T.bool(False), T.bool(False), 50, 96, 100, 0, T.bool(False), 0, 0, 0)
            T.copy(T.region(C_acc[0, 0], 1, 50, 96), T.region(C[0, 0], 2, 50, 96))