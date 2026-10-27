# E2 evidence: T.Pipelined over a dynamic extent n_tiles[bx], after tl.InjectSoftwarePipeline (#19), before any rewrite.
# PipelinePlanning accepts it; the prologue/epilogue are predicated (if 0 < n, T.where), which has no ISA form.
# from tvm.script import tirx as T

@T.prim_func
def dyn(x_handle: T.handle, W_handle: T.handle, n_tiles_handle: T.handle, Y_handle: T.handle):
    T.func_attr({"target": T.target({"arch": "sm_80", "keys": ["cuda", "gpu"], "kind": "cuda", "max_num_threads": 1024, "tag": "", "thread_warp_size": 32})})
    x = T.match_buffer(x_handle, (256, 512), "float16", strides=(512, 1))
    W = T.match_buffer(W_handle, (32, 512, 256), "float16", strides=(131072, 256, 1))
    n_tiles = T.match_buffer(n_tiles_handle, (32,), "int32", strides=(1,))
    Y = T.match_buffer(Y_handle, (32, 128, 256), strides=(32768, 256, 1))
    with T.sblock("root"):
        bx = T.int32()
        t = T.int32()
        T.reads(n_tiles[bx], x[T.min(0, t * 64 + 128):T.min(0, t * 64 + 128) + (T.max(T.max(T.max(127, n_tiles[bx] * 64 - 1), t * 64 + 191), n_tiles[bx] * 64 - 64) + 1 - T.min(0, t * 64 + 128)), 0:128], W[bx, 0:128, 0:128], Y[bx, T.min(0, n_tiles[bx] * 64 - 128):T.min(0, n_tiles[bx] * 64 - 128) + (T.max(T.max(T.max(n_tiles[bx], 2) * 64 + T.min(0, n_tiles[bx] * 64 - 128) - 64, n_tiles[bx] * 64 - 192), n_tiles[bx] * 64 - 64) + 1 - T.min(0, n_tiles[bx] * 64 - 128)), 0])
        t_1 = T.int32()
        T.writes(Y[bx, T.min(T.min(T.min(0, t * 64), n_tiles[bx] * 64 - 128), t_1 * 64 + n_tiles[bx] * 64 - 128):T.min(T.min(T.min(0, t * 64), n_tiles[bx] * 64 - 128), t_1 * 64 + n_tiles[bx] * 64 - 128) + (T.max(T.max(T.max(T.max(n_tiles[bx] * 64 - 129, t * 64 + 63), n_tiles[bx] * 64 - 1), t_1 * 64 + n_tiles[bx] * 64 - 65), n_tiles[bx] * 64 - 129) + 1 - T.min(T.min(T.min(0, t * 64), n_tiles[bx] * 64 - 128), t_1 * 64 + n_tiles[bx] * 64 - 128)), 0:128])
        T.launch_thread(bx, 32)
        tx = T.launch_thread("threadIdx.x", 128)
        ty = T.launch_thread("threadIdx.y", 1)
        tz = T.launch_thread("threadIdx.z", 1)
        with T.sblock("tilelang_root"):
            x_s = T.Buffer((2, 64, 128), "float16", scope="shared.dyn")
            W_s = T.Buffer((2, 128, 128), "float16", scope="shared.dyn")
            T.reads(n_tiles[bx], x[T.min(0, t * 64 + 128):T.min(0, t * 64 + 128) + (T.max(T.max(T.max(127, n_tiles[bx] * 64 - 1), t * 64 + 191), n_tiles[bx] * 64 - 64) + 1 - T.min(0, t * 64 + 128)), 0:128], x_s[0:2, 0:64, 0:128], W[bx, 0:128, 0:128], W_s[0:2, 0:128, 0:128], Y[bx, T.min(0, n_tiles[bx] * 64 - 128):T.min(0, n_tiles[bx] * 64 - 128) + (T.max(n_tiles[bx], 2) * 64 - 63), 0])
            T.writes(x_s[0:2, 0:64, 0:128], W_s[0:2, 0:128, 0:128], Y[bx, T.min(T.min(T.min(0, t * 64), n_tiles[bx] * 64 - 128), t_1 * 64 + n_tiles[bx] * 64 - 128):T.min(T.min(T.min(0, t * 64), n_tiles[bx] * 64 - 128), t_1 * 64 + n_tiles[bx] * 64 - 128) + (T.max(T.max(T.max(n_tiles[bx] * 64 - 129, t * 64 + 63), n_tiles[bx] * 64 - 1), t_1 * 64 + n_tiles[bx] * 64 - 65) + 1 - T.min(T.min(T.min(0, t * 64), n_tiles[bx] * 64 - 128), t_1 * 64 + n_tiles[bx] * 64 - 128)), 0:128])
            T.sblock_attr({"tl.launch_threads": [128, 1, 1]})
            x_s = T.sblock_alloc_buffer((2, 64, 128), "float16", data=x_s.data, scope="shared.dyn")
            W_s = T.sblock_alloc_buffer((2, 128, 128), "float16", data=W_s.data, scope="shared.dyn")
            acc = T.sblock_alloc_buffer((64, 128), scope="local.fragment")
            with T.sblock(""):
                T.reads(n_tiles[bx], x[0:64, 0:128], x_s[0:2, 0, 0], W[bx, 0:128, 0:128], W_s[0:2, 0, 0], x[T.Mul(0, 64):T.Mul(0, 64) + 64, 0:128], W[bx, 0:128, 0:128])
                T.writes(x_s[0:2, 0:64, 0:128], W_s[0:2, 0:128, 0:128], x_s[T.FloorMod(0, 2), 0:64, 0:128], W_s[T.FloorMod(0, 2), 0:128, 0:128])
                if 0 < n_tiles[bx]:
                    with T.sblock(""):
                        T.where(0 < n_tiles[bx])
                        T.reads(x[T.Mul(0, 64), 0], x_s[0:2, 0, 0], x[T.Mul(0, 64):T.Mul(0, 64) + 64, 0:128])
                        T.writes(x_s[0:2, 0:64, 0:128])
                        T.copy(T.region(x[T.Mul(0, 64), 0], 1, 64, 128), T.region(x_s[T.FloorMod(0, 2), 0, 0], 2, 1, 64, 128), no_implicit_async_commit_wait=1)
                    with T.sblock(""):
                        T.where(0 < n_tiles[bx])
                        T.reads(W[bx, 0, 0], W_s[0:2, 0, 0], W[bx, 0:128, 0:128])
                        T.writes(W_s[0:2, 0:128, 0:128])
                        T.copy(T.region(W[bx, 0, 0], 1, 1, 128, 128), T.region(W_s[T.FloorMod(0, 2), 0, 0], 2, 1, 128, 128), no_implicit_async_commit_wait=1)
                    T.ptx_commit_group()
                else:
                    T.evaluate(0)
            with T.sblock(""):
                T.reads(n_tiles[bx], x[64:128, 0:128], x_s[0:2, 0, 0], W[bx, 0:128, 0:128], W_s[0:2, 0, 0], x[T.Mul(1, 64):T.Mul(1, 64) + 64, 0:128], W[bx, 0:128, 0:128])
                T.writes(x_s[0:2, 0:64, 0:128], W_s[0:2, 0:128, 0:128], x_s[T.FloorMod(1, 2), 0:64, 0:128], W_s[T.FloorMod(1, 2), 0:128, 0:128])
                if 1 < n_tiles[bx]:
                    with T.sblock(""):
                        T.where(1 < n_tiles[bx])
                        T.reads(x[T.Mul(1, 64), 0], x_s[0:2, 0, 0], x[T.Mul(1, 64):T.Mul(1, 64) + 64, 0:128])
                        T.writes(x_s[0:2, 0:64, 0:128])
                        T.copy(T.region(x[T.Mul(1, 64), 0], 1, 64, 128), T.region(x_s[T.FloorMod(1, 2), 0, 0], 2, 1, 64, 128), no_implicit_async_commit_wait=1)
                    with T.sblock(""):
                        T.where(1 < n_tiles[bx])
                        T.reads(W[bx, 0, 0], W_s[0:2, 0, 0], W[bx, 0:128, 0:128])
                        T.writes(W_s[0:2, 0:128, 0:128])
                        T.copy(T.region(W[bx, 0, 0], 1, 1, 128, 128), T.region(W_s[T.FloorMod(1, 2), 0, 0], 2, 1, 128, 128), no_implicit_async_commit_wait=1)
                    T.ptx_commit_group()
                else:
                    T.evaluate(0)
            with T.sblock(""):
                T.reads(n_tiles[bx], x_s[0:2, 0:64, 0:128], W_s[0:2, 0:128, 0:128], acc[0:64, 0:128], x[128:128 + (n_tiles[bx] * 64 - 128), 0:128], W[bx, 0:128, 0:128], Y[bx, 0:n_tiles[bx] * 64 - 191, 0], x_s[t % 2, 0:64, 0:128], W_s[t % 2, 0:128, 0:128], x[(t + 2) * 64:(t + 2) * 64 + 64, 0:128], W[bx, 0:128, 0:128], acc[0:64, 0:128])
                T.writes(acc[0:64, 0:128], x_s[0:2, 0:64, 0:128], W_s[0:2, 0:128, 0:128], Y[bx, 0:n_tiles[bx] * 64 - 128, 0:128], acc[0:64, 0:128], x_s[(t + 2) % 2, 0:64, 0:128], W_s[(t + 2) % 2, 0:128, 0:128], Y[bx, t * 64:t * 64 + 64, 0:128])
                for t in T.serial(n_tiles[bx] - 2, annotations={"tl_pipelined_num_stages": 2}):
                    T.ptx_wait_group(1)
                    with T.sblock(""):
                        T.reads(x_s[0:2, 0, 0], W_s[0:2, 0, 0], acc[0, 0], x_s[0:2, 0:64, 0:128], W_s[0:2, 0:128, 0:128])
                        T.writes(acc[0:64, 0:128])
                        T.gemm(T.region(x_s[t % 2, 0, 0], 1, 1, 64, 128), T.region(W_s[t % 2, 0, 0], 1, 1, 128, 128), T.region(acc[0, 0], 2, 64, 128), T.bool(False), T.bool(False), 64, 128, 128, 0, T.bool(True), 0, 0, 0)
                    with T.sblock(""):
                        T.reads(x[(t + 2) * 64, 0], x_s[0:2, 0, 0], x[(t + 2) * 64:(t + 2) * 64 + 64, 0:128])
                        T.writes(x_s[0:2, 0:64, 0:128])
                        T.copy(T.region(x[(t + 2) * 64, 0], 1, 64, 128), T.region(x_s[(t + 2) % 2, 0, 0], 2, 1, 64, 128), no_implicit_async_commit_wait=1)
                    with T.sblock(""):
                        T.reads(W[bx, 0, 0], W_s[0:2, 0, 0], W[bx, 0:128, 0:128])
                        T.writes(W_s[0:2, 0:128, 0:128])
                        T.copy(T.region(W[bx, 0, 0], 1, 1, 128, 128), T.region(W_s[(t + 2) % 2, 0, 0], 2, 1, 128, 128), no_implicit_async_commit_wait=1)
                    T.ptx_commit_group()
                    with T.sblock(""):
                        T.reads(acc[0, 0], Y[bx, t * 64, 0], acc[0:64, 0:128])
                        T.writes(Y[bx, t * 64:t * 64 + 64, 0:128])
                        T.copy(T.region(acc[0, 0], 1, 64, 128), T.region(Y[bx, t * 64, 0], 2, 1, 64, 128))
            with T.sblock(""):
                T.reads(n_tiles[bx], x_s[0:2, 0:64, 0:128], W_s[0:2, 0:128, 0:128], acc[0:64, 0:128], Y[bx, n_tiles[bx] * 64 - 128:n_tiles[bx] * 64 - 128 + 65, 0], x_s[(t_1 + n_tiles[bx] - 2) % 2, 0:64, 0:128], W_s[(t_1 + n_tiles[bx] - 2) % 2, 0:128, 0:128], acc[0:64, 0:128])
                T.writes(acc[0:64, 0:128], Y[bx, n_tiles[bx] * 64 - 128:n_tiles[bx] * 64 - 128 + 128, 0:128], acc[0:64, 0:128], Y[bx, (t_1 + n_tiles[bx] - 2) * 64:(t_1 + n_tiles[bx] - 2) * 64 + 64, 0:128])
                for t_1 in T.unroll(n_tiles[bx] + 2 - n_tiles[bx], annotations={"tl_pipelined_num_stages": 2}):
                    T.ptx_wait_group(1 - t_1)
                    with T.sblock(""):
                        T.where(2 <= t_1 + n_tiles[bx] and t_1 < 2)
                        T.reads(x_s[0:2, 0, 0], W_s[0:2, 0, 0], acc[0, 0], x_s[0:2, 0:64, 0:128], W_s[0:2, 0:128, 0:128])
                        T.writes(acc[0:64, 0:128])
                        T.gemm(T.region(x_s[(t_1 + n_tiles[bx] - 2) % 2, 0, 0], 1, 1, 64, 128), T.region(W_s[(t_1 + n_tiles[bx] - 2) % 2, 0, 0], 1, 1, 128, 128), T.region(acc[0, 0], 2, 64, 128), T.bool(False), T.bool(False), 64, 128, 128, 0, T.bool(True), 0, 0, 0)
                    with T.sblock(""):
                        T.where(2 <= t_1 + n_tiles[bx] and t_1 < 2)
                        T.reads(acc[0, 0], Y[bx, (t_1 + n_tiles[bx] - 2) * 64, 0], acc[0:64, 0:128])
                        T.writes(Y[bx, (t_1 + n_tiles[bx] - 2) * 64:(t_1 + n_tiles[bx] - 2) * 64 + 64, 0:128])
                        T.copy(T.region(acc[0, 0], 1, 64, 128), T.region(Y[bx, (t_1 + n_tiles[bx] - 2) * 64, 0], 2, 1, 64, 128))