"""E3/E5/F4 tests for the MoE decode kernels of the TileLang cut-off prototype (phases 2 and 3).

Needs tilelang 0.1.15 (compiler/prototype/PINNED.md); skipped otherwise.
Run: python -m pytest tests/test_tilelang_moe.py -q
"""
import os
import sys

import numpy as np
import pytest

tilelang = pytest.importorskip("tilelang")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "compiler", "prototype"))
sys.path.insert(0, os.path.join(HERE, "..", "sim"))
sys.path.insert(0, os.path.join(HERE, "..", "ref"))

from tvm import tirx  # noqa: E402
from tvm.tirx.stmt_functor import post_order_visit  # noqa: E402

import common  # noqa: E402
import kernels  # noqa: E402
import moe_data  # noqa: E402
import moe_kernel  # noqa: E402
import overlap_report  # noqa: E402
from cut import EVENT_FILL_ISSUED, collect_buffers, lower_to_tile_tir  # noqa: E402
from tcta import Machine  # noqa: E402
from tile_tir_to_program import tile_tir_to_program  # noqa: E402

FORBIDDEN = ("launch_thread", "threadIdx", "blockIdx", "ptx_", "mbarrier", "tma", "cp_async", "mma", "wgmma", "T.where")
B, D, DFF, E, K, BM = moe_kernel.B, moe_kernel.D, moe_kernel.DFF, moe_kernel.E, moe_kernel.TOPK, moe_kernel.BLOCK_M
ROWS = moe_kernel.T_MAX * BM
MMA_PER_TILE = 16 + 8               # [W1|W3] gemms + W2 gemms per 64-token tile
IDX_PER_TILE = (4 + 4) * BM         # 4 LOAD.IDX + 4 STORE.IDX rows per tile


@pytest.fixture(scope="module")
def fixture(tmp_path_factory):
    fx = kernels.gen_moe(str(tmp_path_factory.mktemp("fx")))
    return fx.arrays, fx.meta


@pytest.fixture(scope="module")
def cut_f4():
    return lower_to_tile_tir(moe_kernel.build_moe_expert_kernel(), scope_map=moe_kernel.SCOPE_MAP)


@pytest.fixture(scope="module")
def cut_bf16():
    return lower_to_tile_tir(moe_kernel.build_moe_expert_kernel(weight_format="bf16"), scope_map=moe_kernel.SCOPE_MAP)


@pytest.fixture(scope="module")
def cut_persistent():
    return lower_to_tile_tir(moe_kernel.build_moe_persistent_kernel(), scope_map=moe_kernel.SCOPE_MAP)


@pytest.fixture(scope="module")
def cut_combine():
    return lower_to_tile_tir(moe_kernel.build_moe_combine_kernel(), scope_map=moe_kernel.COMBINE_SCOPE_MAP)


@pytest.fixture(scope="module")
def cut_router():
    return lower_to_tile_tir(moe_kernel.build_router_kernel(), scope_map=moe_kernel.ROUTER_SCOPE_MAP)


SCOPE_OF = {"cut_f4": moe_kernel.SCOPE_MAP, "cut_bf16": moe_kernel.SCOPE_MAP, "cut_persistent": moe_kernel.SCOPE_MAP,
            "cut_combine": moe_kernel.COMBINE_SCOPE_MAP, "cut_router": moe_kernel.ROUTER_SCOPE_MAP}


def _calls(func):
    calls = []
    post_order_visit(func.body, lambda n: calls.append(n) if isinstance(n, tirx.Call) else None)
    return calls


def _name(c):
    return c.op.name if hasattr(c.op, "name") else str(c.op)


def _serial_loops(func):
    loops = []
    post_order_visit(func.body, lambda n: loops.append(n) if isinstance(n, tirx.For) and n.kind != tirx.ForKind.PARALLEL else None)
    return loops


def _dynamic(loops):
    return [l for l in loops if not isinstance(l.extent, tirx.IntImm)]


def _count_buffer(extent):
    """(n[0] + 63) // 64 -> n"""
    assert isinstance(extent, tirx.FloorDiv) and int(extent.b) == BM and isinstance(extent.a, tirx.Add), extent
    assert isinstance(extent.a.a, tirx.BufferLoad) and int(extent.a.b) == BM - 1
    return extent.a.a.buffer.name


# ---------------------------------------------------------------- E3: the cut

@pytest.mark.parametrize("which", list(SCOPE_OF))
def test_cut_is_tile_level(which, request):
    f = request.getfixturevalue(which)
    text = f.script()
    for bad in FORBIDDEN:
        assert bad not in text, f"{bad!r} survives the cut"
    scalars = [p for p in f.params if p not in f.buffer_map]
    assert len(scalars) == 1 and str(scalars[0].dtype) == "int32"      # the block index (macros may rename bx)
    assert f.attrs["tcta.cut"] == "tl.InjectSoftwarePipeline"
    for name, lst in collect_buffers(f.body).items():
        for b in lst:
            assert b.scope() == SCOPE_OF[which].get(name, "global"), (name, b.scope())


def test_dynamic_loop_and_rings_survive(cut_bf16):
    """The token-tile loop keeps its dynamic extent ceildiv(n[0], 64); the weight stream is a 3-deep ring with
    fill events (num_stages=3, the phase-3 default; under v0.1.2 free sync the ring depth no longer decides the
    boundary overlap, see F3); the gather alternates between two INP slots."""
    dyn = _dynamic(_serial_loops(cut_bf16))
    assert len(dyn) == 1 and _count_buffer(dyn[0].extent) == "n"
    bufs = collect_buffers(cut_bf16.body)
    assert any([int(x) for x in b.shape] == [3, 128, 128] for b in bufs["W_s"])
    assert [int(x) for x in bufs["x_s"][0].shape] == [2, 64, 128]
    calls = _calls(cut_bf16)
    wgt_copies = [c for c in calls if _name(c) == "tl.tileop.copy" and c.args[1].args[0].buffer.scope() == "tcta.wgt"]
    issued = [c for c in calls if _name(c) == "tirx.call_extern" and c.args[0].value == EVENT_FILL_ISSUED]
    assert len(wgt_copies) == len(issued) == (2 + 1) + (3 + 1)  # prologue + body fills of the [W1|W3] and W2 streams
    assert not dyn[0].annotations or "software_pipeline_stage" not in dyn[0].annotations


def test_persistent_nested_dynamic_loop_survives(cut_persistent):
    """F2: a static expert loop around the dynamic token-tile loop is accepted by every pass up to #19."""
    loops = _serial_loops(cut_persistent)
    outer = [l for l in loops if l.loop_var.name == "ee"]
    assert len(outer) == 1 and int(outer[0].extent) == E // 2
    dyn = _dynamic(loops)
    assert len(dyn) == 2 and {_count_buffer(l.extent) for l in dyn} == {"n_a", "n_b"}
    reduces = [c for c in _calls(cut_persistent) if _name(c) == "tl.tileop.reduce"]
    assert len(reduces) == 3                       # count(0) before the loop, two prefetches per pair


def test_gather_and_scatter_survive_as_elementwise(cut_f4):
    text = cut_f4.script()
    assert "x[idx_s[i], kk * 128 + j]" in text and "if_then_else(0 <= idx_s[i]" in text
    assert "if 0 <= dst_s[i]:" in text and "Yp[dst_s[i], m // 2 * 128 + j] = out[m % 4 // 2, i, j]" in text  # flattened W2 stream, out double-buffered


# ---------------------------------------------------------------- E5/F4: end to end on the simulator

def _machine(fx, idx, w, dst, weight_format):
    m = Machine(dram_bytes=moe_data.DRAM_BYTES)
    W13, W2 = moe_data.load_dram(m, fx, idx, w, dst, weight_format)
    return m, W13, W2


@pytest.mark.parametrize("weight_format", ["f4.b32", "bf16"])
def test_every_expert_matches_y_parts(fixture, cut_f4, cut_bf16, weight_format):
    """Per-expert programs on one machine (Machine.reset between kernels) fill the [B*k, d] scatter buffer;
    it must equal the fixture's y_parts (f4.b32 weights are the normative values)."""
    fx, meta = fixture
    cut = cut_f4 if weight_format == "f4.b32" else cut_bf16
    idx, w, dst = moe_data.pad_lists(fx)
    m, _, _ = _machine(fx, idx, w, dst, weight_format)
    for e in range(E):
        m.reset()                                    # stats persist across reset: compare deltas
        before = dict(m.stats)
        prog, rep = tile_tir_to_program(cut, bx=e, dram=moe_data.DRAM)
        st = m.run(prog.instrs)
        assert st["ok"]
        n_tiles = -(-int(fx["counts"][e]) // BM)
        assert st["mma"] - before["mma"] == n_tiles * MMA_PER_TILE
        assert (st["idx_rows"] + st["idx_skips"]) - (before["idx_rows"] + before["idx_skips"]) == n_tiles * IDX_PER_TILE
        assert rep.loop_extent == "R0" and rep.loops == [(0, "R0", BM)]
    parts = moe_data.read_parts(m, B, K, D)
    assert np.all(np.abs(parts).sum(axis=-1) > 0), "every (token, slot) row is written"
    assert common.rowmax_err(parts.reshape(-1, D), fx["y_parts"].reshape(-1, D)) < meta["tol_rowmax"]


def test_combine_matches_y(fixture, cut_combine):
    """F1 alone: the combine programs over the fixture's y_parts reproduce y (shared expert added on device)."""
    fx, meta = fixture
    m = Machine(dram_bytes=moe_data.DRAM_BYTES)
    m.dram_write(moe_data.DRAM["Yp"], fx["y_parts"].astype(np.float32).ravel())
    y, sts = overlap_report.run_combine(m, fx)
    assert all(s["ok"] for s in sts)
    assert sts[-1]["idx_rows"] == len(sts) * K * BM * (D // 128)   # cumulative: 6 gathers x 64 rows x 4 N-tiles per program
    assert common.rowmax_err(y, fx["y"]) < meta["tol_rowmax"]
    _, rep = tile_tir_to_program(cut_combine, bx=0, dram=moe_data.DRAM)
    assert rep.loop_extent == 4 and rep.collapsed["gather"] == K and rep.collapsed["vgate_add"] == K + 1


def test_persistent_program_end_to_end(fixture, cut_persistent):
    """F2 + F1: one program for all experts, then the combine; y matches the fixture."""
    fx, meta = fixture
    m, st, prog, rep, trace = overlap_report.run_persistent(fx, "f4.b32")
    assert st["ok"] and st["mma"] == int(sum(-(-c // BM) for c in fx["counts"])) * MMA_PER_TILE
    # v0.1.2: one static LOOP n=E/2 (per-level incr) holding the two LOOP r, step=64 token-tile loops
    assert rep.loops == [(0, E // 2, 1), (1, "R0", BM), (2, "R1", BM)] and rep.unrolled_loops == 14
    assert rep.body_end_pops == 0, "nothing is popped at the end of an iteration: the queues run ahead"
    assert all(pushes == pops for pushes, pops in rep.edge_counts.values()), rep.edge_counts
    parts = moe_data.read_parts(m, B, K, D)
    assert common.rowmax_err(parts.reshape(-1, D), fx["y_parts"].reshape(-1, D)) < meta["tol_rowmax"]
    y, sts = overlap_report.run_combine(m, fx)
    assert all(s["ok"] for s in sts)
    assert common.rowmax_err(y, fx["y"]) < meta["tol_rowmax"]


def test_overlap_at_every_expert_boundary(fixture):
    """F3 under v0.1.2 (free sync): the LOAD queue is the critical queue, so the boundary is measured on it.
    Between the last LOAD of expert e and the first LOAD of expert e+1 the queue idles exactly one step (the
    next count's LOAD.2D, which is work of no expert); the first gather of e+1 follows the last MMA of e by
    at most one step, its token dependencies were satisfied before that MMA, and the MMA queue restarts
    within the gather -> fill -> MMA latency."""
    fx, _ = fixture
    m, st, prog, rep, trace = overlap_report.run_persistent(fx, "f4.b32")
    rows = overlap_report.boundaries(prog, trace, fx["counts"], rep)
    assert len(rows) == E - 1                       # every expert of the fixture has tokens
    assert all(r[6] == 1 for r in rows), rows       # LOAD idle steps at the boundary: the count load only
    assert all(r[3] >= -1 for r in rows), rows      # first LOAD.IDX of e+1 at most one step after the last MMA of e
    assert all(r[8] is None or r[8] >= 0 for r in rows), rows   # the gather's tokens were ready by then
    assert all(r[7] <= 3 for r in rows), rows       # MMA gap at the boundary (intra-expert cadence: 1)


def test_persistent_without_prefetch(fixture):
    """The single-register form `LOOP n=E { count; LOOP r }` (the ~270-instruction program the phase-4 task
    expected): same numerics, but the LOOP waits for its own VCOUNT, so the LOAD queue idles 3 steps at every
    boundary instead of 1 (the LOAD -> VEC -> LOAD round trip of the count)."""
    fx, meta = fixture
    m, st, prog, rep, trace = overlap_report.run_persistent(fx, "f4.b32", prefetch=False)
    assert st["ok"] and rep.loops == [(0, E, 1), (1, "R0", BM)] and rep.n_instrs < 300, (rep.loops, rep.n_instrs)
    parts = moe_data.read_parts(m, B, K, D)
    assert common.rowmax_err(parts.reshape(-1, D), fx["y_parts"].reshape(-1, D)) < meta["tol_rowmax"]
    rows = overlap_report.boundaries(prog, trace, fx["counts"], rep)
    assert len(rows) == E - 1 and all(r[6] >= 2 for r in rows), rows


def test_router_vtopk_matches_topk_idx(fixture, cut_router):
    fx, _ = fixture
    logits = fx["logits"].astype(np.float32)
    m = Machine()
    m.dram_write(moe_data.DRAM["logits"], logits.ravel())
    for t in range(logits.shape[0]):
        m.reset()                                    # one program per token on one machine (Machine.reset, v0.1.1)
        prog, rep = tile_tir_to_program(cut_router, bx=t, dram=moe_data.DRAM)
        assert rep.collapsed.get("vtopk") == 1
        assert m.run(prog.instrs)["ok"]
        base = moe_data.DRAM["topk"] + t * 8 * 4
        got = m.dram[base: base + K * 4].view(np.int32)   # stored with the i32 qualifier
        assert np.array_equal(got, fx["topk_idx"][t]), t


def test_zero_token_expert_runs_zero_iterations(fixture, cut_f4):
    fx, _ = fixture
    idx, w, dst = moe_data.pad_lists(fx)
    e = 3
    idx[e] = -1
    dst[e] = -1
    w[e] = 0
    m, _, _ = _machine(fx, idx, w, dst, "f4.b32")
    host_bytes = m.stats["dram_write_bytes"]        # the host's own DRAM writes count too
    prog, rep = tile_tir_to_program(cut_f4, bx=e, dram=moe_data.DRAM)
    st = m.run(prog.instrs)
    assert st["ok"] and st["mma"] == 0 and st["idx_rows"] == 0 and st["dram_write_bytes"] == host_bytes
    assert rep.loop_extent == "R0"
    assert not moe_data.read_parts(m, B, K, D).any()


def test_zero_token_expert_in_persistent_program(fixture):
    """LOOP r = 0 inside the persistent program: the expert is skipped, the other boundaries are unchanged."""
    fx, meta = fixture
    idx, w, dst = moe_data.pad_lists(fx)
    e = 5
    idx[e] = -1
    dst[e] = -1
    w[e] = 0
    counts = fx["counts"].copy()
    counts[e] = 0
    m, st, prog, rep, trace = overlap_report.run_persistent(fx, "f4.b32", idx, w, dst)
    assert st["ok"] and st["mma"] == (E - 1) * MMA_PER_TILE
    parts = moe_data.read_parts(m, B, K, D)
    ref = fx["y_parts"].copy()
    tok, slot = fx["idx_lists"][e], fx["dst_rows"][e] - fx["idx_lists"][e] * K
    ref[tok[tok >= 0], slot[tok >= 0]] = 0
    assert common.rowmax_err(parts.reshape(-1, D), ref.reshape(-1, D)) < meta["tol_rowmax"]
    rows = overlap_report.boundaries(prog, trace, counts, rep)
    assert len(rows) == E - 3 and all(r[6] == 1 and r[7] <= 3 for r in rows), rows


def test_two_token_tiles(fixture, cut_f4):
    """A synthetic expert with > 64 tokens: LOOP r, step=64 runs two iterations (ceil), the DRAM bases advance."""
    fx, meta = fixture
    idx, w, dst = moe_data.pad_lists(fx)
    e = 0
    tokens = np.unique(np.concatenate([fx["idx_lists"][0], fx["idx_lists"][1]]))
    tokens = tokens[tokens >= 0][:ROWS]
    assert BM < tokens.size < 2 * BM
    idx[e] = -1
    idx[e, :tokens.size] = tokens
    dst[e] = -1
    dst[e, :tokens.size] = np.arange(tokens.size)           # any distinct rows of the scatter buffer
    w[e] = 0
    w[e, :tokens.size] = 0.5 + np.arange(tokens.size) / tokens.size
    m, W13, W2 = _machine(fx, idx, w, dst, "f4.b32")
    prog, rep = tile_tir_to_program(cut_f4, bx=e, dram=moe_data.DRAM)
    st = m.run(prog.instrs)
    assert st["ok"] and st["mma"] == 2 * MMA_PER_TILE
    got = moe_data.read_rows(m, "Yp", tokens.size, D)
    ref = moe_data.reference_expert_output(fx, W13[e], W2[e], idx[e], w[e])[:tokens.size]
    assert common.rowmax_err(got, ref) < meta["tol_rowmax"]


def test_token_graph(cut_f4, cut_persistent):
    for cut, n in ((cut_f4, 1), (cut_persistent, 2)):     # the persistent body holds two token tiles
        _, rep = tile_tir_to_program(cut, bx=0, dram=moe_data.DRAM)
        assert all(pushes == pops for pushes, pops in rep.edge_counts.values()), rep.edge_counts
        assert rep.body_end_pops == 0
        assert rep.collapsed["gather"] == 4 * n and rep.collapsed["scatter"] == 4 * n
        assert rep.collapsed["vcount"] == (1 if n == 1 else n + 1)
        assert rep.collapsed["dequant_load"] == rep.collapsed["dequant_fold"] == MMA_PER_TILE * n
        assert all(pops >= 0 for *_, pops in rep.fill_wait_pops)
        assert rep.drained + rep.body_end_pops + rep.hoisted_pops <= rep.primed_pushes + 32   # a handful of tail tokens, nothing systematic
