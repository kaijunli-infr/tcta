"""G4 tests: MLA decode (G1), KDA chunkwise (G2) and the partial-tile GEMM through the TileLang cut.

Needs tilelang 0.1.15 (compiler/prototype/PINNED.md); skipped otherwise.
Run: python -m pytest tests/test_tilelang_mla_kda.py -q
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
import gemm_kernel  # noqa: E402
import kda_kernel  # noqa: E402
import kda_kernel_roundtrip  # noqa: E402
import kernels  # noqa: E402
import mla_kernel  # noqa: E402
from tcta.isa import SP, DeviceConfig  # noqa: E402
from cut import collect_buffers, lower_to_tile_tir  # noqa: E402
from tcta import Machine  # noqa: E402
from tile_tir_to_program import listing, tile_tir_to_program  # noqa: E402

FORBIDDEN = ("launch_thread", "threadIdx", "blockIdx", "ptx_", "mbarrier", "tma", "cp_async", "mma", "wgmma", "T.where")
TOL = 2.0 ** -7                     # tol_rowmax of every fixture


def serial_loops(func):
    out = []

    def visit(n):
        if isinstance(n, tirx.For) and n.kind == tirx.ForKind.SERIAL:
            out.append(n)

    post_order_visit(func.body, visit)
    return out


def work_lines(prog):
    return [ln for ln in listing(prog).splitlines() if " PUSH(" not in ln and " POP(" not in ln]


# ------------------------------------------------------------------ fixtures

@pytest.fixture(scope="module")
def mla_fx(tmp_path_factory):
    return kernels.gen_mla_decode(str(tmp_path_factory.mktemp("mla"))).arrays


@pytest.fixture(scope="module")
def kda_fx(tmp_path_factory):
    return kernels.gen_kda(str(tmp_path_factory.mktemp("kda"))).arrays


@pytest.fixture(scope="module")
def gemm_fx(tmp_path_factory):
    return kernels.gen_gemm(str(tmp_path_factory.mktemp("gemm"))).arrays


@pytest.fixture(scope="module")
def cut_mla():
    return lower_to_tile_tir(mla_kernel.build_mla_kernel(), scope_map=mla_kernel.SCOPE_MAP)


@pytest.fixture(scope="module")
def cut_kda():
    return lower_to_tile_tir(kda_kernel.build_kda_kernel(), scope_map=kda_kernel.SCOPE_MAP)


@pytest.fixture(scope="module")
def cut_gemm():
    return lower_to_tile_tir(gemm_kernel.build_gemm_kernel(), scope_map=gemm_kernel.SCOPE_MAP)


@pytest.fixture(scope="module")
def mla_program(cut_mla):
    return tile_tir_to_program(cut_mla, bx=0, dram=mla_kernel.DRAM)


@pytest.fixture(scope="module")
def kda_program(cut_kda):
    return tile_tir_to_program(cut_kda, bx=0, dram=kda_kernel.DRAM, cfg=kda_kernel.CONFIG_C)


@pytest.fixture(scope="module")
def mla_run(mla_program, mla_fx):
    prog, rep = mla_program
    m = Machine(dram_bytes=mla_kernel.DRAM_BYTES)
    mla_kernel.load_dram(m, mla_fx)
    st = m.run(prog.instrs)
    return m, st


@pytest.fixture(scope="module")
def kda_run(kda_program, kda_fx):
    prog, rep = kda_program
    m = Machine(cfg=kda_kernel.CONFIG_C, dram_bytes=kda_kernel.DRAM_BYTES)
    kda_kernel.load_dram(m, kda_fx)
    st = m.run(prog.instrs)
    return m, st


# ------------------------------------------------------------------ cut structure (#19)

@pytest.mark.parametrize("which", ["mla", "kda", "gemm"])
def test_cut_has_no_gpu_lowering(which, cut_mla, cut_kda, cut_gemm):
    f = {"mla": cut_mla, "kda": cut_kda, "gemm": cut_gemm}[which]
    script = f.script()
    assert not any(tok in script for tok in FORBIDDEN), [t for t in FORBIDDEN if t in script]
    assert f.attrs["tcta.cut"] == "tl.InjectSoftwarePipeline"


def test_mla_cut_keeps_wide_accumulator_and_kv_loop(cut_mla):
    """The [64, 512] accumulator is one fragment after #19 (no split into tiles), its column
    slices are the gemm destinations, and the KV loop is the one serial loop left."""
    bufs = collect_buffers(cut_mla.body)
    acc_o = [b for b in bufs["acc_o"]]
    assert acc_o and all([int(x) for x in b.shape] == [64, 512] for b in acc_o)
    assert all(b.scope() == "tcta.acc" for b in acc_o)
    for name, scope in mla_kernel.SCOPE_MAP.items():
        assert all(b.scope() == scope for b in bufs[name]), name
    loops = serial_loops(cut_mla)
    assert len(loops) == 3 and sorted(int(l.extent) for l in loops) == [4, 4, 16]
    script = cut_mla.script()
    assert "acc_o[0:64, " in script or "acc_o[0:64, c * 128" in script or "T.region(acc_o[0, c * 128]" in script, \
        "sliced wide-accumulator gemm destination not found"


def test_kda_cut_keeps_state_scope_and_chunk_loop(cut_kda):
    bufs = collect_buffers(cut_kda.body)
    assert all(b.scope() == "tcta.state" for b in bufs["S_st"]) and all([int(x) for x in b.shape] == [128, 128] for b in bufs["S_st"])
    assert all([int(x) for x in b.shape] == [2, 64, 128] for b in bufs["S_acc"])
    for name, scope in kda_kernel.SCOPE_MAP.items():
        assert all(b.scope() == scope for b in bufs[name]), name
    extents = sorted(int(l.extent) for l in serial_loops(cut_kda))
    assert extents == [2, 4, 5]                     # state halves, chunks, Neumann steps


# ------------------------------------------------------------------ mapping (D4)

def test_mla_program_shape(mla_program):
    prog, rep = mla_program
    assert rep.n_instrs == 122                       # 123 on v0.1.2; VCOPY replaced VFILL+VADD for scores_max_prev
    assert rep.loops == [(0, 16, 1)]                 # LOOP n=16 over KV tiles; the chunk loops are unrolled
    assert rep.slots["acc_o"] == "ACC[1]" and rep.slots["acc_s"] == "ACC[0]"
    lines = work_lines(prog)
    assert sum("VSCALE_ACC(acc=ACC[1]" in ln and "wide=True" in ln for ln in lines) == 2    # rescale + final 1/l
    assert sum("STORE.2D(src=ACC[1]" in ln and "wide=True" in ln for ln in lines) == 1
    assert sum("MMA(acc=ACC[" in ln for ln in lines) == 4 and all(f"MMA(acc=ACC[{s}]" in "".join(lines) for s in (1, 2, 3, 4))
    assert sum("MMA.T(" in ln for ln in lines) == 5                                           # QK^T chunks (1 rope + 4)
    assert sum("LOAD.2D(" in ln for ln in lines) == 10                                        # 5 Q + 5 K per KV tile
    assert rep.shared_slots == 0


def test_mla_token_graph(mla_program):
    prog, rep = mla_program
    assert set(rep.edge_counts) == {"LOAD->MMA", "MMA->LOAD", "MMA->VEC", "VEC->MMA", "VEC->STORE"}
    assert all(p == q for p, q in rep.edge_counts.values())
    assert rep.edge_counts["LOAD->MMA"] == (160, 160) and rep.edge_counts["MMA->LOAD"] == (150, 150)
    assert rep.body_end_pops == 0 and rep.hoisted_pops == 0


def test_kda_program_shape(kda_program):
    """v0.1.3 form: computed B operands move ACC->WGT with VCVT, the state halves move into the 128-row
    STATE tile with VCVT row0, no DRAM scratch (521 instructions on v0.1.2, 352 here)."""
    prog, rep = kda_program
    assert rep.loops == [(0, 4, 1)]                  # LOOP n=4 over chunks; solve and half loops unrolled
    assert 330 <= rep.n_instrs <= 380
    lines = work_lines(prog)
    text = "\n".join(lines)
    assert "MMA.MASKED" not in text                  # the masked products are MMA.T, which has no masked form: VGATE mul
    assert rep.collapsed["vgate_mul"] == 2
    assert "op=decay" in text                        # VGATE decay
    assert "b=STATE[0]" in text                      # the state as a B operand
    assert rep.slots["S_acc"] == "ACC[0]" and rep.slots["S_st"] == "STATE[0]"
    assert rep.shared_slots > 0                      # the config-C ACC budget is met by live-range sharing
    assert sum("MMA" in ln.split()[1] for ln in lines if len(ln.split()) > 1) == 27   # per chunk, textual
    assert "STORE.2D(src=ACC[0]" in text and "STORE.2D(src=ACC[1]" in text          # checkpoints of both halves
    # the device-computed-operand path and the row-offset state move
    assert sum("VCVT(dst=WGT[" in ln for ln in lines) == 11 and sum("VCVT(dst=INP[" in ln for ln in lines) == 15
    assert rep.collapsed["vcvt_operand"] == 26 and rep.collapsed["vcvt_state"] == 2
    assert "VCVT(dst=STATE[0], src=ACC[0], inplace=False, row0=0)" in text
    assert "VCVT(dst=STATE[0], src=ACC[1], inplace=False, row0=64)" in text
    assert "LOAD.2D(dst=STATE" not in text and "to_bf16=True" in text and "VNEG(" in text
    assert sum(ln.startswith("STORE") for ln in lines) == 3 and sum(ln.startswith("LOAD") for ln in lines) == 18


def test_kda_token_graph_has_no_dram_round_trips(kda_program):
    prog, rep = kda_program
    assert all(p == q for p, q in rep.edge_counts.values())
    assert "LOAD->STORE" not in rep.edge_counts      # no scratch region is stored and loaded back
    assert rep.edge_counts["STORE->LOAD"] == (4, 4)  # WAR only: a gather reuses the slot a checkpoint read
    assert rep.body_end_pops == 0


def test_kda_needs_config_c(cut_kda):
    """The default device descriptor (ACC 6, WGT 4) cannot hold the KDA working set even with slot sharing."""
    with pytest.raises(AssertionError, match="out of"):
        tile_tir_to_program(cut_kda, bx=0, dram=kda_kernel.DRAM)


def test_gemm_program(cut_gemm, gemm_fx):
    prog, rep = tile_tir_to_program(cut_gemm, bx=0, dram=gemm_kernel.DRAM)
    assert rep.n_instrs == 13 and rep.loops == []
    assert rep.edge_counts == {"LOAD->MMA": (2, 2), "MMA->LOAD": (1, 1), "MMA->STORE": (1, 1)}
    text = "\n".join(work_lines(prog))
    assert "rows_valid=50, cols_valid=100" in text and "rows_valid=100, cols_valid=96" in text
    m = Machine(dram_bytes=gemm_kernel.DRAM_BYTES)
    gemm_kernel.load_dram(m, gemm_fx)
    st = m.run(prog.instrs)
    assert st["ok"] and st["mma"] == 1
    assert common.rowmax_err(gemm_kernel.read_c(m, gemm_fx), gemm_fx["C"]) < 1e-5


# ------------------------------------------------------------------ numerics (simulator, token checking on)

def test_mla_matches_fixture(mla_run, mla_fx):
    m, st = mla_run
    assert st["ok"] and st["mma"] == 16 * 9
    err = common.rowmax_err(mla_kernel.read_o(m, mla_fx).reshape(-1, mla_fx["O"].shape[-1]), mla_fx["O"].reshape(-1, mla_fx["O"].shape[-1]))
    assert err < TOL, err


def test_kda_matches_every_checkpoint_and_o(kda_run, kda_fx):
    m, st = kda_run
    assert st["ok"] and st["mma"] == 4 * 27
    assert st["dram_read_bytes"] == 4 * 230400       # inputs only: no scratch reloads, no state reload
    o, S = kda_kernel.read_outputs(m, kda_fx)
    errs = [common.rowmax_err(S[c], kda_fx["S_chunks"][c]) for c in range(S.shape[0])]
    assert all(e < TOL for e in errs), errs
    assert np.allclose(S[-1], kda_fx["S_final"], atol=TOL)
    err_o = common.rowmax_err(o, kda_fx["o"])
    assert err_o < TOL, err_o


# ------------------------------------------------------------------ phase 5 sweep inputs stay valid

def test_kda_roundtrip_form_matches_on_v013(kda_fx):
    """The v0.1.2 authoring (DRAM round trips, state mirror) is kept for the cycle model's before/after; it must
    still map and match on the v0.1.3 simulator."""
    cut = lower_to_tile_tir(kda_kernel_roundtrip.build_kda_kernel(), scope_map=kda_kernel_roundtrip.SCOPE_MAP)
    prog, rep = tile_tir_to_program(cut, bx=0, dram=kda_kernel_roundtrip.DRAM, cfg=kda_kernel.CONFIG_C)
    assert rep.edge_counts["LOAD->STORE"][0] > 0 and rep.edge_counts["STORE->LOAD"][0] > 0
    assert sum(ln.startswith("STORE") for ln in work_lines(prog)) == 14 and "LOAD.2D(dst=STATE" in "\n".join(work_lines(prog))
    m = Machine(cfg=kda_kernel.CONFIG_C, dram_bytes=kda_kernel.DRAM_BYTES)
    kda_kernel_roundtrip.load_dram(m, kda_fx)
    st = m.run(prog.instrs)
    assert st["ok"] and st["mma"] == 4 * 27
    o, S = kda_kernel_roundtrip.read_outputs(m, kda_fx)
    assert common.rowmax_err(o, kda_fx["o"]) < TOL and all(common.rowmax_err(S[c], kda_fx["S_chunks"][c]) < TOL for c in range(4))


@pytest.mark.parametrize("inp_slots,loads_per_tile", [(3, 9), (4, 8), (5, 5), (6, 5)])
def test_mla_resident_q_variants_match(mla_fx, inp_slots, loads_per_tile):
    """INP-slot sweep inputs: with n slots, n-2 Q chunks stay resident (all five from 5 slots); the loop body
    keeps 5 K loads plus the streamed Q loads."""
    cfg = DeviceConfig(slots={SP.INP: inp_slots, SP.WGT: 4, SP.ACC: 6, SP.VEC: 16, SP.STATE: 1})
    cut = lower_to_tile_tir(mla_kernel.build_mla_kernel(inp_slots=inp_slots), scope_map=mla_kernel.SCOPE_MAP)
    prog, rep = tile_tir_to_program(cut, bx=0, dram=mla_kernel.DRAM, cfg=cfg)
    assert rep.loops == [(0, 16, 1)]
    body = prog.instrs[rep.loop_pcs[0][0]: rep.loop_pcs[0][1]]
    assert sum(i.op == "LOAD.2D" for i in body) == loads_per_tile
    m = Machine(cfg=cfg, dram_bytes=mla_kernel.DRAM_BYTES)
    mla_kernel.load_dram(m, mla_fx)
    st = m.run(prog.instrs)
    assert st["ok"] and st["mma"] == 16 * 9
    O = mla_fx["O"].reshape(-1, mla_fx["O"].shape[-1])
    assert common.rowmax_err(mla_kernel.read_o(m, mla_fx).reshape(-1, O.shape[-1]), O) < TOL
