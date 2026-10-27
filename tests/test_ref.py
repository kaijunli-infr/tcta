"""Fixture-driven tests: reference generators -> simulator DMA/format paths.

These tests regenerate small fixtures in a temp dir (fast) and load the large
V4.1 slice from ./fixtures if it exists (python ref/kernels.py fixtures).
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "sim"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ref"))

import numpy as np
import pytest

from tcta import Machine, Program, Region, Q, INP, WGT, ACC, VEC
from tcta import dtypes as dt
import common, kernels

FIX = os.path.join(os.path.dirname(__file__), "..", "fixtures")


def read_f32(m, r, rows, cols):
    return np.stack([m.dram[r.base + i * r.stride[0]: r.base + i * r.stride[0] + cols * 4].view(np.float32)
                     for i in range(rows)])


def test_gemm_fixture_through_sim(tmp_path):
    fx = kernels.gen_gemm(str(tmp_path)); d, meta = fx.arrays, fx.meta
    Mv, Kv, Nv = meta["valid"]
    m = Machine()
    m.dram_write(0, d["A_bytes"]); ra = Region(0, "bf16", d["A"].shape, (d["A"].shape[1] * 2, 2))
    m.dram_write(1 << 16, d["B_bytes"]); rb = Region(1 << 16, "bf16", d["B"].shape, (d["B"].shape[1] * 2, 2))
    ro = Region(1 << 20, "f32", (Mv, Nv), (Nv * 4, 4))
    p = Program()
    p.load2d(INP(0), ra, Mv, Kv); p.push(Q.LOAD, Q.MMA)
    p.load2d(WGT(0), rb, Kv, Nv); p.push(Q.LOAD, Q.MMA)
    p.pop(Q.MMA, Q.LOAD); p.pop(Q.MMA, Q.LOAD)
    p.mma(ACC(0), INP(0), WGT(0)); p.push(Q.MMA, Q.STORE)
    p.pop(Q.STORE, Q.MMA); p.store2d(ACC(0), ro, Mv, Nv); p.finish()
    m.run(p.instrs)
    assert common.rowmax_err(read_f32(m, ro, Mv, Nv), d["C"]) < meta["tol_rowmax"]


def test_moe_fixture_router_and_gather(tmp_path):
    fx = kernels.gen_moe(str(tmp_path), B=128, d=128, E=32); d = fx.arrays
    m = Machine()
    # router: VTOPK on one token's logits must reproduce topk_idx (ties broken identically: stable, descending)
    t = 7
    p = Program()
    p.vfill(VEC(9), array=d["logits"][t][None, :])
    p.vtopk(VEC(8), VEC(9), k=6)
    p.finish()
    m.run(p.instrs)
    assert np.array_equal(m.sp[VEC(8)].data[0, :6].astype(int), d["topk_idx"][t])
    # gather: expert e's token rows via LOAD.IDX with -1 skips
    e = int(np.argmax(d["counts"]))
    rows = d["idx_lists"][e]
    nrows = min(64, rows.size)
    x = d["x"]; stride = dt.pad64(x.shape[1] * 2)
    for i in range(x.shape[0]):
        m.dram_write(i * stride, common.bf16_bytes(x[i]))
    rx = Region(0, "bf16", x.shape, (stride, 2))
    m2 = Machine(); m2.dram[:] = m.dram
    p = Program()
    p.vfill(VEC(8), array=rows[:nrows][None, :].astype(np.float32)); p.push(Q.VEC, Q.LOAD)
    p.pop(Q.LOAD, Q.VEC)
    p.loadidx(INP(0), rx, VEC(8), nrows=nrows, row_bytes=stride, cols_valid=x.shape[1])
    p.finish()
    m2.run(p.instrs)
    got = m2.sp[INP(0)].data[:nrows]
    for i, r in enumerate(rows[:nrows]):
        exp = np.zeros(x.shape[1], np.float32) if r < 0 else x[r]
        assert np.array_equal(got[i], exp)


@pytest.mark.skipif(not os.path.exists(os.path.join(FIX, "v41_flash_decoder_slice")), reason="run ref/kernels.py first")
def test_v41_slice_formats_through_dma():
    d, meta = common.load(os.path.join(FIX, "v41_flash_decoder_slice"))
    m = Machine(dram_bytes=256 << 20)
    # --- KV cache: sparse gather of 64 entries, f4.r16, row_bytes 384 ---
    kv = d["kv_rows"]; L = kv.shape[0]
    m.dram_write(0, kv.reshape(-1))
    rkv = Region(0, "f4.r16", (L, 576), (384, 0))
    sel = d["sel"][3][:64]
    exp = np.stack([dt.dequant_row("f4.r16", kv[i], ncols=576) for i in sel])
    p = Program()
    p.vfill(VEC(8), array=sel[None, :].astype(np.float32)); p.push(Q.VEC, Q.LOAD)
    p.pop(Q.LOAD, Q.VEC)
    # a 576-wide entry lands in 4 full column tiles + one half tile: repeat LOAD.IDX advancing col0
    for t in range(4):
        p.loadidx(WGT(t), rkv, VEC(8), nrows=64, row_bytes=384, cols_valid=128, col0=t * 128)
    p.loadidx(INP(0), rkv, VEC(8), nrows=64, row_bytes=384, cols_valid=64, col0=512)
    p.finish()
    st = m.run(p.instrs)
    assert st["idx_rows"] == 5 * 64
    got = np.concatenate([m.sp[WGT(t)].data[:64, :128] for t in range(4)] + [m.sp[INP(0)].data[:64, :64]], axis=1)
    assert np.array_equal(got, exp)
    # --- expert weight tile: f4.b32 LOAD.2D of a [128,128] block of W1[e] ---
    e = 5
    pay = d["W1_payload"][e]; sc = d["W1_scales"][e]      # [K, N/2], [K/32, N/32]
    K, N2 = pay.shape; N = N2 * 2
    base, sbase = 64 << 20, 128 << 20
    m2 = Machine(dram_bytes=256 << 20)
    m2.dram_write(base, pay.reshape(-1)); m2.dram_write(sbase, sc.reshape(-1))
    r = Region(base, "f4.b32", (K, N), (N2, 0), scales_base=sbase, scales_stride=N // 32)
    p = Program()
    p.load2d(WGT(0), r, 128, 128, row0=256, col0=128); p.push(Q.LOAD, Q.MMA)
    p.pop(Q.MMA, Q.LOAD)
    p.finish()
    m2.run(p.instrs)
    got = m2.sp[WGT(0)].data
    # expected: dequantize the same block with the fixture's own rule
    blk = pay[256:384]; scl = sc[8:12, 4:8]
    exp = np.stack([dt.dequant_row("f4.b32", blk[i, 64:128], ncols=128, block_scales=scl[i // 32]) for i in range(128)])
    assert np.array_equal(got, exp)
    assert meta["E"] == 32 and meta["d"] == 5120
