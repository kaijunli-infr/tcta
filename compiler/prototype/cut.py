"""D3: stop TileLang's CUDA lowering at the tile level and finish with the task's rewrites.

    lower_to_tile_tir(prim_func, cut="inject_pipeline") -> tvm.tirx.PrimFunc

Steps (addendum §D3):
1. TileLang's own passes, exactly as `tilelang/cuda/pipeline.py::CUDAPassPipelineBodyPrologue`
   orders them, up to and including the cut pass. The prologue is run unmodified
   with a PassInstrument that captures the module after the cut pass and aborts
   the rest, so the pass sequence is TileLang's, not a re-listing of it.
   Cuts: "inject_pipeline" = #19 tl.InjectSoftwarePipeline (primary),
         "pipeline_planning" = #18 tl.PipelinePlanning (fallback).
2. Scope rewrite (after the cut; the frontend rejects tcta scopes before it, see
   docs/tilelang-passes.md): shared.dyn / local.fragment -> tcta.* by buffer name.
3. Event rewrite: ptx_commit_group() -> call_extern("tcta.fill_issued"),
   ptx_wait_group(n) -> call_extern("tcta.fill_wait", n); the
   no_implicit_async_commit_wait annotation on copies is dropped.
4. Shell strip: the six thread_extent AttrStmts (T.launch_thread) are removed,
   bx/by become int32 PrimFunc params, bz/tx/ty/tz are dropped (unused in the
   body), the tl.launch_threads block annotation is removed, and the two wrapper
   blocks ("root", "tilelang_root") lose their reads/writes region lists, which
   TileLang's pipeline pass left referencing the loop var `k` outside its loop.

Target: sm_80. (sm_90 with tl.disable_warp_specialized gives identical IR; pass
arch="sm_90" to use that path, the flag is set automatically.)
"""

from __future__ import annotations

import os
import sys

import tilelang
from tilelang import tvm
from tvm import tirx
from tvm.tirx.stmt_functor import ir_transform, post_order_visit

CUTS = {
    "inject_pipeline": "tl.InjectSoftwarePipeline",
    "pipeline_planning": "tl.PipelinePlanning",
}

SCOPE_MAP = {
    "Q_shared": "tcta.inp",
    "K_shared": "tcta.wgt",
    "V_shared": "tcta.wgt",
    "acc_s": "tcta.acc",
    "acc_s_cast": "tcta.acc",
    "acc_o": "tcta.acc",
    "scores_max": "tcta.vec",
    "scores_max_prev": "tcta.vec",
    "scores_scale": "tcta.vec",
    "scores_sum": "tcta.vec",
    "logsum": "tcta.vec",
}
TCTA_SCOPES = frozenset({"tcta.inp", "tcta.wgt", "tcta.acc", "tcta.vec"})

EVENT_FILL_ISSUED = "tcta.fill_issued"
EVENT_FILL_WAIT = "tcta.fill_wait"

PASS_CONFIGS = {tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True}


# --------------------------------------------------------------------------
# 1. TileLang passes up to the cut
# --------------------------------------------------------------------------

@tvm.instrument.pass_instrument
class _CaptureAfterPass:
    """Capture the module after the named top-level pass and abort the pipeline."""

    def __init__(self, pass_name: str):
        self.cut_name = pass_name
        self.depth = 0
        self.captured = None
        self.seen: list[str] = []

    def run_before_pass(self, mod, info):
        self.depth += 1

    def run_after_pass(self, mod, info):
        self.depth -= 1
        if self.depth > 0:  # nested pass inside another pass
            return
        self.seen.append(info.name)
        if info.name == self.cut_name:
            self.captured = mod
            # Any exception aborts the prologue; TVM re-raises it as RuntimeError.
            raise RuntimeError(f"cut after {info.name}")


def run_tilelang_passes(func: tirx.PrimFunc, cut: str = "inject_pipeline", arch: str = "sm_80") -> tvm.IRModule:
    """Run TileLang's CUDA prologue and return the module right after the cut pass."""
    from tilelang.cuda.pipeline import CUDAPassPipelineBodyPrologue
    from tilelang.engine.semantic_check import PreLowerSemanticCheck

    cut_name = CUTS[cut]
    cfg = dict(PASS_CONFIGS)
    if arch != "sm_80":
        cfg[tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED] = True
    inst = _CaptureAfterPass(cut_name)
    mod = tvm.IRModule({"main": func})
    target = tvm.target.Target({"kind": "cuda", "arch": arch})
    try:
        with tvm.transform.PassContext(config=cfg, instruments=[inst]), target:
            PreLowerSemanticCheck(mod)
            CUDAPassPipelineBodyPrologue(mod, target)
    except Exception:
        if inst.captured is None:
            raise
    if inst.captured is None:
        raise RuntimeError(f"pass {cut_name} never ran; saw {inst.seen}")
    return inst.captured


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def collect_buffers(stmt) -> dict[str, list[tirx.Buffer]]:
    """All distinct Buffer objects referenced in stmt, grouped by name."""
    found: dict[str, list[tirx.Buffer]] = {}

    def add(b):
        lst = found.setdefault(b.name, [])
        if not any(x.same_as(b) for x in lst):
            lst.append(b)

    def visit(n):
        if isinstance(n, (tirx.BufferLoad, tirx.BufferStore)):
            add(n.buffer)
        elif isinstance(n, tirx.SBlock):
            for b in n.alloc_buffers:
                add(b)
            for r in list(n.reads) + list(n.writes):
                add(r.buffer)
            for m in n.match_buffers:
                add(m.buffer)
                add(m.source.buffer)

    post_order_visit(stmt, visit)
    return found


def _call_name(call: tirx.Call) -> str:
    return call.op.name if hasattr(call.op, "name") else str(call.op)


def _rebuild_block(blk: tirx.SBlock, **kw) -> tirx.SBlock:
    args = dict(
        iter_vars=list(blk.iter_vars),
        reads=list(blk.reads),
        writes=list(blk.writes),
        name_hint=blk.name_hint,
        body=blk.body,
        init=blk.init,
        alloc_buffers=list(blk.alloc_buffers),
        match_buffers=list(blk.match_buffers),
        annotations=dict(blk.annotations) if blk.annotations else None,
    )
    args.update(kw)
    return tirx.SBlock(**args)


# --------------------------------------------------------------------------
# 2. scope rewrite
# --------------------------------------------------------------------------

def rewrite_scopes(func: tirx.PrimFunc, scope_map: dict[str, str] = SCOPE_MAP) -> tirx.PrimFunc:
    """Re-declare every buffer named in scope_map with its tcta scope.

    A buffer's scope lives in its data Var's pointer type, so each buffer gets a
    new data Var (shared between Buffer objects that shared the old one, e.g. the
    pipeline pass's ring re-declaration `data=K_shared.data`) and a new Buffer.
    """
    buffers = collect_buffers(func.body)
    new_data: list[tuple[tirx.Var, tirx.Var]] = []
    pairs: list[tuple[tirx.Buffer, tirx.Buffer]] = []

    def data_for(old: tirx.Buffer, scope: str) -> tirx.Var:
        for od, nd in new_data:
            if od.same_as(old.data):
                return nd
        nd = tirx.Var(old.data.name, tvm.ir.PointerType(tvm.ir.PrimType(old.dtype), scope))
        new_data.append((old.data, nd))
        return nd

    for name, scope in scope_map.items():
        for old in buffers.get(name, []):
            nb = tirx.decl_buffer(
                list(old.shape), old.dtype, old.name,
                data=data_for(old, scope),
                strides=list(old.strides) if len(old.strides) else None,
                elem_offset=old.elem_offset,
                scope=scope,
                data_alignment=old.data_alignment,
                offset_factor=old.offset_factor,
            )
            pairs.append((old, nb))

    def nb(b):
        for old, new in pairs:
            if old.same_as(b):
                return new
        return b

    def region(r):
        return tirx.BufferRegion(nb(r.buffer), list(r.region))

    def post(node):
        if isinstance(node, tirx.BufferLoad):
            n = nb(node.buffer)
            return None if n.same_as(node.buffer) else tirx.BufferLoad(n, list(node.indices), node.predicate)
        if isinstance(node, tirx.BufferStore):
            n = nb(node.buffer)
            return None if n.same_as(node.buffer) else tirx.BufferStore(n, node.value, list(node.indices), node.predicate)
        if isinstance(node, tirx.SBlock):
            return _rebuild_block(
                node,
                reads=[region(r) for r in node.reads],
                writes=[region(r) for r in node.writes],
                alloc_buffers=[nb(b) for b in node.alloc_buffers],
                match_buffers=[tirx.MatchBufferRegion(nb(m.buffer), region(m.source)) for m in node.match_buffers],
            )
        return None

    body = ir_transform(func.body, None, post)
    return tirx.PrimFunc(func.params, body, func.ret_type, func.buffer_map, func.attrs)


# --------------------------------------------------------------------------
# 3. event rewrite
# --------------------------------------------------------------------------

DROPPED_COPY_ANNOTATIONS = ("no_implicit_async_commit_wait",)


def rewrite_events(func: tirx.PrimFunc) -> tirx.PrimFunc:
    def post(node):
        if isinstance(node, tirx.Evaluate) and isinstance(node.value, tirx.Call):
            call = node.value
            name = _call_name(call)
            if name.endswith("ptx_commit_group"):
                return tirx.Evaluate(tirx.call_extern(call.dtype, EVENT_FILL_ISSUED))
            if name.endswith("ptx_wait_group"):
                return tirx.Evaluate(tirx.call_extern(call.dtype, EVENT_FILL_WAIT, call.args[0]))
        if isinstance(node, tirx.Call) and node.annotations:
            ann = dict(node.annotations)
            kept = {k: v for k, v in ann.items() if k not in DROPPED_COPY_ANNOTATIONS}
            if len(kept) != len(ann):
                return tirx.Call(node.dtype, node.op, list(node.args), kept or None)
        return None

    body = ir_transform(func.body, None, post)
    return tirx.PrimFunc(func.params, body, func.ret_type, func.buffer_map, func.attrs)


# --------------------------------------------------------------------------
# 4. shell strip
# --------------------------------------------------------------------------

WRAPPER_BLOCKS = ("root", "tilelang_root")
TILE_PARAM_TAGS = ("blockIdx.x", "blockIdx.y")


def strip_launch_shell(func: tirx.PrimFunc) -> tirx.PrimFunc:
    thread_vars: dict[str, tirx.Var] = {}

    def post(node):
        if isinstance(node, tirx.AttrStmt) and node.attr_key == "thread_extent":
            iv = node.node
            thread_vars[iv.thread_tag] = iv.var
            return node.body
        if isinstance(node, tirx.SBlock) and node.name_hint in WRAPPER_BLOCKS:
            ann = dict(node.annotations) if node.annotations else {}
            ann.pop("tl.launch_threads", None)
            return _rebuild_block(node, reads=[], writes=[], annotations=ann or None)
        return None

    body = ir_transform(func.body, None, post)
    tile_params = [thread_vars[t] for t in TILE_PARAM_TAGS if t in thread_vars]
    params = list(func.params) + tile_params

    # Nothing else may be free in the body: thread vars bz/tx/ty/tz were only
    # bound by the shell and must not be read.
    defs = list(params) + [b.data for b in func.buffer_map.values()]
    free = free_vars(body, defs)
    if free:
        raise RuntimeError(f"free vars after shell strip: {sorted(v.name for v in free)}")

    return tirx.PrimFunc(params, body, func.ret_type, func.buffer_map, func.attrs)


_LET_TYPES = tuple(t for t in (getattr(tirx, "LetStmt", None), getattr(tirx, "Let", None), getattr(tirx, "Bind", None)) if t is not None)


def free_vars(stmt, defs) -> list[tirx.Var]:
    """Vars used in stmt that no For/SBlock/Let/AttrStmt/param defines.

    Order-insensitive on purpose: tvm.tirx.analysis.undefined_vars rejects the
    #19 IR because InjectSoftwarePipeline leaves the loop var `k` in the
    reads/writes of the block that *encloses* its loop (a use before the
    definition, in TileLang's own output).
    """
    defined = list(defs)
    used: list[tirx.Var] = []

    def visit(n):
        if isinstance(n, tirx.Var):
            used.append(n)
        elif isinstance(n, tirx.For):
            defined.append(n.loop_var)
        elif isinstance(n, tirx.SBlock):
            defined.extend(iv.var for iv in n.iter_vars)
            defined.extend(b.data for b in n.alloc_buffers)
            defined.extend(m.buffer.data for m in n.match_buffers)
        elif _LET_TYPES and isinstance(n, _LET_TYPES):
            defined.append(n.var)
        elif isinstance(n, tirx.AttrStmt) and isinstance(n.node, tirx.IterVar):
            defined.append(n.node.var)

    post_order_visit(stmt, visit)
    out: list[tirx.Var] = []
    for v in used:
        if not any(v.same_as(d) for d in defined) and not any(v.same_as(o) for o in out):
            out.append(v)
    return out


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------

def lower_to_tile_tir(prim_func: tirx.PrimFunc, cut: str = "inject_pipeline", arch: str = "sm_80",
                      scope_map: dict[str, str] = SCOPE_MAP) -> tirx.PrimFunc:
    mod = run_tilelang_passes(prim_func, cut=cut, arch=arch)
    func = mod["main"]
    func = rewrite_scopes(func, scope_map)
    func = rewrite_events(func)
    func = strip_launch_shell(func)
    return func.with_attr("tcta.cut", CUTS[cut]).with_attr("tcta.arch", arch)


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from fa_kernel import build_fa_kernel

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dumps")
    for cut in CUTS:
        f = lower_to_tile_tir(build_fa_kernel(), cut=cut)
        path = os.path.join(out_dir, f"cut_{cut}.py")
        with open(path, "w") as fh:
            fh.write(f"# D3 output: lower_to_tile_tir(build_fa_kernel(), cut={cut!r})\n")
            fh.write(f.script())
        print(f"{cut}: wrote {path}")

    # phase 2/3 (E3, F1, F2): the MoE kernels in both weight formats, the persistent program,
    # the combine, and the router
    import moe_kernel
    for name, func, smap in (
        ("cut_moe", moe_kernel.build_moe_expert_kernel(), moe_kernel.SCOPE_MAP),
        ("cut_moe_bf16", moe_kernel.build_moe_expert_kernel(weight_format="bf16"), moe_kernel.SCOPE_MAP),
        ("cut_moe_persistent", moe_kernel.build_moe_persistent_kernel(), moe_kernel.SCOPE_MAP),
        ("cut_moe_combine", moe_kernel.build_moe_combine_kernel(), moe_kernel.COMBINE_SCOPE_MAP),
        ("cut_router", moe_kernel.build_router_kernel(), moe_kernel.ROUTER_SCOPE_MAP),
    ):
        f = lower_to_tile_tir(func, scope_map=smap)
        path = os.path.join(out_dir, f"{name}.py")
        with open(path, "w") as fh:
            fh.write(f"# E3/F2 output: lower_to_tile_tir(moe_kernel.{name}) at tl.InjectSoftwarePipeline\n")
            fh.write(f.script())
        print(f"{name}: wrote {path}")

    # phase 4 (G1, G2) and the GEMM family: MLA decode (wide accumulator), KDA chunkwise (resident
    # state, DRAM round trips for computed B operands), partial-tile GEMM
    import gemm_kernel
    import kda_kernel
    import mla_kernel
    for name, func, smap in (
        ("cut_gemm", gemm_kernel.build_gemm_kernel(), gemm_kernel.SCOPE_MAP),
        ("cut_mla", mla_kernel.build_mla_kernel(), mla_kernel.SCOPE_MAP),
        ("cut_kda", kda_kernel.build_kda_kernel(), kda_kernel.SCOPE_MAP),
    ):
        f = lower_to_tile_tir(func, scope_map=smap)
        path = os.path.join(out_dir, f"{name}.py")
        with open(path, "w") as fh:
            fh.write(f"# G1/G2 output: lower_to_tile_tir({name[4:]}_kernel.build_{name[4:]}_kernel()) at tl.InjectSoftwarePipeline\n")
            fh.write(f.script())
        print(f"{name}: wrote {path}")
