"""D3 (and later D5) tests for the TileLang cut-off prototype.

Needs tilelang 0.1.15 (see compiler/prototype/PINNED.md); skipped otherwise.
Run: python -m pytest tests/test_tilelang_cut.py -q
"""
import os
import sys

import pytest

tilelang = pytest.importorskip("tilelang")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "compiler", "prototype"))
sys.path.insert(0, os.path.join(HERE, "..", "sim"))

from tvm import tirx  # noqa: E402
from tvm.tirx.stmt_functor import post_order_visit  # noqa: E402

from cut import EVENT_FILL_ISSUED, EVENT_FILL_WAIT, SCOPE_MAP, collect_buffers, lower_to_tile_tir  # noqa: E402
from fa_kernel import build_fa_kernel  # noqa: E402

FORBIDDEN = ("launch_thread", "threadIdx", "blockIdx", "ptx_", "mbarrier", "tma", "cp_async", "mma", "wgmma")


@pytest.fixture(scope="module")
def cut_func():
    return lower_to_tile_tir(build_fa_kernel(), cut="inject_pipeline")


def _calls(func):
    calls = []

    def visit(n):
        if isinstance(n, tirx.Call):
            calls.append(n)

    post_order_visit(func.body, visit)
    return calls


def _name(call):
    return call.op.name if hasattr(call.op, "name") else str(call.op)


def _extern_name(call):
    return call.args[0].value if _name(call) == "tirx.call_extern" else None


def _region_buffer(region_call):
    return region_call.args[0].buffer


# ---------------------------------------------------------------- D3 asserts

def test_no_thread_axes_or_gpu_intrinsics(cut_func):
    text = cut_func.script()
    for bad in FORBIDDEN:
        assert bad not in text, f"{bad!r} survives the cut"


def test_every_buffer_has_a_tcta_or_global_scope(cut_func):
    bufs = collect_buffers(cut_func.body)
    assert set(SCOPE_MAP) <= set(bufs), "a kernel buffer disappeared"
    for name, lst in bufs.items():
        for b in lst:
            want = SCOPE_MAP.get(name, "global")
            assert b.scope() == want, (name, b.scope(), want)


def test_fill_events_match_wgt_copies(cut_func):
    calls = _calls(cut_func)
    wgt_copies = [c for c in calls if _name(c) == "tl.tileop.copy" and _region_buffer(c.args[1]).scope() == "tcta.wgt"]
    issued = [c for c in calls if _extern_name(c) == EVENT_FILL_ISSUED]
    waits = [c for c in calls if _extern_name(c) == EVENT_FILL_WAIT]
    gemms = [c for c in calls if _name(c) == "tl.tileop.gemm"]
    assert len(wgt_copies) > 0
    assert len(issued) == len(wgt_copies)
    assert len(waits) == len(gemms)  # one wait before every gemm in this schedule
    assert all(isinstance(w.args[1], tirx.IntImm) for w in waits), "fill_wait count must be static"


def test_copy_annotations_dropped(cut_func):
    for c in _calls(cut_func):
        if _name(c) == "tl.tileop.copy" and c.annotations:
            assert "no_implicit_async_commit_wait" not in dict(c.annotations)


def test_tile_coordinates_are_params(cut_func):
    names = [p.name for p in cut_func.params]
    assert names[-2:] == ["bx", "by"]
    assert all(str(p.dtype) == "int32" for p in cut_func.params[-2:])
    assert cut_func.attrs["tcta.cut"] == "tl.InjectSoftwarePipeline"


def test_pipeline_structure_survives(cut_func):
    """The 2-deep K/V ring and the pipelined loop built by TileLang are intact."""
    bufs = collect_buffers(cut_func.body)
    for name in ("K_shared", "V_shared"):
        assert any([int(x) for x in b.shape] == [2, 64, 64] for b in bufs[name]), name
    loops = []
    post_order_visit(cut_func.body, lambda n: loops.append(n) if isinstance(n, tirx.For) and n.annotations and "tl_pipelined_num_stages" in n.annotations else None)
    assert len(loops) == 1 and int(loops[0].extent) == 2


def test_fallback_cut_pipeline_planning():
    f = lower_to_tile_tir(build_fa_kernel(), cut="pipeline_planning")
    text = f.script()
    for bad in FORBIDDEN:
        assert bad not in text, bad
    assert "software_pipeline_stage" in text
    assert EVENT_FILL_ISSUED not in text  # nothing to rewrite before InjectSoftwarePipeline
    assert f.attrs["tcta.cut"] == "tl.PipelinePlanning"


# ---------------------------------------------------------------- D5: end to end

import numpy as np  # noqa: E402

sys.path.insert(0, os.path.join(HERE, "..", "ref"))
import common  # noqa: E402
import kernels  # noqa: E402
from tcta import Machine  # noqa: E402
from tile_tir_to_program import tile_tir_to_program  # noqa: E402

# Per-edge (pushes, pops) of tests/test_sim.py::test_flash_attention_inner_loop with nk=4:
# before loop: push LOAD->MMA (Q), 4x push MMA->LOAD, pop MMA<-LOAD (Q);
# per iteration: 2 pop LOAD<-MMA, 2 push LOAD->MMA, pop MMA<-LOAD, push MMA->VEC, pop VEC<-MMA,
#                2 push VEC->MMA, 2 pop MMA<-VEC, pop MMA<-LOAD, 2 push MMA->LOAD, push MMA->VEC, pop VEC<-MMA;
# after: 4 pop LOAD<-MMA, push VEC->STORE, pop STORE<-VEC.
HAND_EDGE_COUNTS = {"LOAD->MMA": (9, 9), "MMA->LOAD": (12, 12), "MMA->VEC": (8, 8), "VEC->MMA": (8, 8), "VEC->STORE": (1, 1)}
# The addendum's event-driven rules give 8/8 on MMA->LOAD: the hand schedule pre-releases 2*num_stages
# bank tokens and drains them; the software-pipelined TIR's prologue fills go into never-read banks, so
# no pre-release is needed and the 4 epilogue-gemm releases are drained instead.
EXPECTED_MMA_LOAD = (8, 8)
# Phase 3's level-invariant token pass: the steady-state iteration's O rescale depends on the previous
# iteration's PV gemm, so MMA->VEC is primed once before LOOP and drained once after it instead of
# being popped at the end of the pushing iteration (the hand schedule's convention): one extra pair.
EXPECTED_MMA_VEC = (9, 9)

DRAM = {"Q": 0, "K": 1 << 19, "V": 1 << 20, "Output": 1 << 21}
HEADS, SEQ, DIM, BLOCK = 8, 256, 64, 64


@pytest.fixture(scope="module")
def fixture(tmp_path_factory):
    # causal=False: the task kernel has no mask (task §2/§7); the generator defaults to causal=True.
    fx = kernels.gen_flash_attention(str(tmp_path_factory.mktemp("fx")), causal=False)
    return fx.arrays, fx.meta


def _load_dram(m, d):
    G = d["Q"].shape[0] // d["K"].shape[0]
    kv = [h // G for h in range(HEADS)]  # each query head against KV head h // 4
    m.dram_write(DRAM["Q"], common.bf16_bytes(d["Q"]).ravel())
    m.dram_write(DRAM["K"], common.bf16_bytes(d["K"][kv]).ravel())
    m.dram_write(DRAM["V"], common.bf16_bytes(d["V"][kv]).ravel())


def _read_tile(m, by, bx):
    addr = DRAM["Output"] + (by * SEQ * DIM + bx * BLOCK * DIM) * 2
    bits = m.dram[addr: addr + BLOCK * DIM * 2].view(np.uint16).astype(np.uint32) << 16
    return bits.view(np.float32).reshape(BLOCK, DIM)


def test_end_to_end_matches_fixture(cut_func, fixture):
    d, meta = fixture
    out = np.zeros((HEADS, SEQ, DIM), np.float32)
    for by in range(HEADS):
        for bx in range(SEQ // BLOCK):
            m = Machine()
            _load_dram(m, d)
            prog, rep = tile_tir_to_program(cut_func, bx=bx, by=by, dram=DRAM)
            st = m.run(prog.instrs)  # token checking is always on; any fault raises
            assert st["ok"]
            out[by, bx * BLOCK:(bx + 1) * BLOCK] = _read_tile(m, by, bx)
    errs = [common.rowmax_err(out[h], d["O"][h]) for h in range(HEADS)]
    assert max(errs) < meta["tol_rowmax"], errs


def test_token_counts_per_edge(cut_func):
    _, rep = tile_tir_to_program(cut_func, bx=0, by=0, dram=DRAM)
    for edge, want in HAND_EDGE_COUNTS.items():
        if edge == "MMA->LOAD":
            assert rep.edge_counts[edge] == EXPECTED_MMA_LOAD, rep.edge_counts
        elif edge == "MMA->VEC":
            assert rep.edge_counts[edge] == EXPECTED_MMA_VEC, rep.edge_counts
        else:
            assert rep.edge_counts[edge] == want, (edge, rep.edge_counts[edge], want)
    assert set(rep.edge_counts) == set(HAND_EDGE_COUNTS)
    assert all(pushes == pops for pushes, pops in rep.edge_counts.values())


def test_fill_wait_counts_static_and_nonnegative(cut_func):
    _, rep = tile_tir_to_program(cut_func, bx=0, by=0, dram=DRAM)
    assert len(rep.fill_wait_pops) == 8
    assert all(pops >= 0 for *_, pops in rep.fill_wait_pops)
    assert all(pops == 1 for *_, pops in rep.fill_wait_pops)
    assert rep.drained == 4 and rep.loop_extent == 2
