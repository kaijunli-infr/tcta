"""Conformance tests for the functional simulator (ISA v0.1).

Run: python -m pytest tests/ -q      (from the repo root, with sim/ on PYTHONPATH)
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "sim"))

import numpy as np
import pytest

from tcta import Machine, Program, Region, Fault, Q, INP, WGT, ACC, VEC, DeviceConfig
from tcta import dtypes as dt

rng = np.random.default_rng(0)


def bf16(x):
    return dt.to_bf16(np.asarray(x, np.float32))


def put_bf16(m: Machine, addr: int, arr: np.ndarray) -> Region:
    """Write a float32 matrix to DRAM as BF16 and return a 2-D region."""
    arr = np.asarray(arr, np.float32)
    bits = (bf16(arr).view(np.uint32) >> 16).astype(np.uint16)
    m.dram_write(addr, bits)
    return Region(addr, "bf16", arr.shape, (arr.shape[1] * 2, 2))


def read_f32(m: Machine, r: Region, rows, cols):
    out = np.empty((rows, cols), np.float32)
    for i in range(rows):
        out[i] = m.dram[r.base + i * r.stride[0]: r.base + i * r.stride[0] + cols * 4].view(np.float32)
    return out


# --------------------------------------------------------------------------
# 1. A single tile GEMM, with the minimum token set.
# --------------------------------------------------------------------------
def test_gemm_tile():
    m = Machine()
    A = rng.standard_normal((64, 128)); B = rng.standard_normal((128, 128))
    ra = put_bf16(m, 0, A); rb = put_bf16(m, 1 << 16, B)
    ro = Region(1 << 20, "f32", (64, 128), (128 * 4, 4))
    p = Program()
    p.load2d(INP(0), ra, 64, 128); p.push(Q.LOAD, Q.MMA)
    p.load2d(WGT(0), rb, 128, 128); p.push(Q.LOAD, Q.MMA)
    p.pop(Q.MMA, Q.LOAD); p.pop(Q.MMA, Q.LOAD)
    p.mma(ACC(0), INP(0), WGT(0)); p.push(Q.MMA, Q.STORE)
    p.pop(Q.STORE, Q.MMA); p.store2d(ACC(0), ro, 64, 128)
    p.finish()
    st = m.run(p.instrs)
    assert st["ok"]
    ref = bf16(A) @ bf16(B)
    got = read_f32(m, ro, 64, 128)
    assert np.allclose(got, ref, rtol=1e-5, atol=1e-4)


# --------------------------------------------------------------------------
# 2. Missing POP is a fault, not a stale result.
# --------------------------------------------------------------------------
def test_untokened_read_faults():
    m = Machine()
    ra = put_bf16(m, 0, rng.standard_normal((64, 128)))
    rb = put_bf16(m, 1 << 16, rng.standard_normal((128, 128)))
    p = Program()
    p.load2d(INP(0), ra, 64, 128); p.push(Q.LOAD, Q.MMA)
    p.load2d(WGT(0), rb, 128, 128)           # no PUSH for this one
    p.pop(Q.MMA, Q.LOAD)
    p.mma(ACC(0), INP(0), WGT(0))
    p.finish()
    with pytest.raises(Fault) as e:
        m.run(p.instrs)
    assert e.value.code in ("UNTOKENED_READ", "TOKEN_LEAK")


def test_token_leak_faults():
    m = Machine()
    p = Program()
    p.push(Q.LOAD, Q.MMA)
    p.finish()
    with pytest.raises(Fault) as e:
        m.run(p.instrs)
    assert e.value.code == "TOKEN_LEAK"


def test_deadlock_faults():
    m = Machine(cfg=DeviceConfig(t_deadlock=200))
    p = Program()
    p.pop(Q.MMA, Q.LOAD)   # nothing will ever PUSH
    p.finish()
    with pytest.raises(Fault) as e:
        m.run(p.instrs)
    assert e.value.code == "TOKEN_TIMEOUT"


# --------------------------------------------------------------------------
# 3. FlashAttention inner loop: one Q tile, several K/V tiles, online softmax,
#    P read from ACC in place (D7), WGT ring buffer released by MMA->LOAD tokens.
# --------------------------------------------------------------------------
def fa_reference(Qm, K, V, scale):
    S = (bf16(Qm) @ bf16(K).T) * scale
    P = np.exp2(S - S.max(axis=1, keepdims=True))
    return (bf16(P) @ bf16(V)) / P.sum(axis=1, keepdims=True)


def test_flash_attention_inner_loop():
    d, Bq, Bk, nk = 128, 64, 128, 4
    scale = np.float32(1.0 / np.sqrt(d) * np.log2(np.e))
    Qm = rng.standard_normal((Bq, d)); K = rng.standard_normal((nk * Bk, d)); V = rng.standard_normal((nk * Bk, d))
    m = Machine()
    rq = put_bf16(m, 0, Qm)
    rk = put_bf16(m, 1 << 18, K)
    rv = put_bf16(m, 1 << 20, V)
    ro = Region(1 << 22, "f32", (Bq, d), (d * 4, 4))

    p = Program()
    # Q resident in INP[0]
    p.load2d(INP(0), rq, Bq, d); p.push(Q.LOAD, Q.MMA)
    # running stats: m in VEC[0], l in VEC[1]
    p.vfill(VEC(0), imm=-1e30, rows_valid=Bq, cols_valid=1)
    p.vfill(VEC(1), imm=0.0, rows_valid=Bq, cols_valid=1)
    p.vfill(ACC(1), imm=0.0, rows_valid=Bq, cols_valid=d)       # O accumulator
    # K/V ring: WGT[0..1] = K(j), WGT[2..3] = V(j), 2 stages.
    # MMA pre-releases both stages of both banks so LOAD can pop before every load.
    for _ in range(4):
        p.push(Q.MMA, Q.LOAD)
    p.pop(Q.MMA, Q.LOAD)                                          # Q
    p.loop(n=nk)
    # ---- LOAD queue: wait for a free K bank and a free V bank, then fill them ----
    p.pop(Q.LOAD, Q.MMA); p.pop(Q.LOAD, Q.MMA)
    p.load2d(WGT(0), rk, Bk, d, incr={"region.base": Bk * d * 2, "dst_mod": 2}); p.push(Q.LOAD, Q.MMA)
    p.load2d(WGT(2), rv, Bk, d, incr={"region.base": Bk * d * 2, "dst_mod": 2}); p.push(Q.LOAD, Q.MMA)
    # ---- MMA: S = Q K^T ----
    p.pop(Q.MMA, Q.LOAD)
    p.mma_t(ACC(0), INP(0), WGT(0), incr={"b_mod": 2}); p.push(Q.MMA, Q.VEC)
    # ---- VEC: online softmax ----
    p.pop(Q.VEC, Q.MMA)
    p.vmul(ACC(0), imm=float(scale))
    p.vrowmax(VEC(2), ACC(0))                                     # m_tile
    # m_new = max(m_old, m_tile)
    p.vfill(VEC(6), imm=0.0, rows_valid=Bq, cols_valid=1)
    p.vadd(VEC(6), vec=VEC(2)); p.vmax(VEC(6), vec=VEC(0))        # m_new
    # alpha = exp2(m_old - m_new)
    p.vfill(VEC(7), imm=0.0, rows_valid=Bq, cols_valid=1)
    p.vadd(VEC(7), vec=VEC(0)); p.vsub(VEC(7), vec=VEC(6)); p.vexp2(VEC(7))
    # P = exp2(S - m_new), rowsum into VEC[3]
    p.vexp2_sum(ACC(0), VEC(3), vec=VEC(6))
    # l = l*alpha + rowsum ; m = m_new
    p.vmul(VEC(1), vec=VEC(7)); p.vadd(VEC(1), vec=VEC(3))
    p.vfill(VEC(0), imm=0.0, rows_valid=Bq, cols_valid=1); p.vadd(VEC(0), vec=VEC(6))
    # O *= alpha ; P -> bf16 in place
    p.vscale_acc(ACC(1), VEC(7))
    p.vcvt(ACC(0), ACC(0), inplace=True)
    p.push(Q.VEC, Q.MMA)                                          # P ready
    p.push(Q.VEC, Q.MMA)                                          # O rescaled
    # ---- MMA: O += P V ----
    p.pop(Q.MMA, Q.VEC); p.pop(Q.MMA, Q.VEC); p.pop(Q.MMA, Q.LOAD)
    p.mma(ACC(1), ACC(0), WGT(2), accumulate=True, incr={"b_mod": 2})
    p.push(Q.MMA, Q.LOAD); p.push(Q.MMA, Q.LOAD)                 # release K and V banks
    p.push(Q.MMA, Q.VEC)                                          # O ready for next VEC pass
    p.pop(Q.VEC, Q.MMA)
    p.endloop()
    # drain the 4 pre-released bank tokens that no LOAD will consume
    for _ in range(4):
        p.pop(Q.LOAD, Q.MMA)
    # epilogue: O /= l
    p.vrcp(VEC(1)); p.vscale_acc(ACC(1), VEC(1)); p.push(Q.VEC, Q.STORE)
    p.pop(Q.STORE, Q.VEC); p.store2d(ACC(1), ro, Bq, d)
    p.finish()

    st = m.run(p.instrs)
    assert st["ok"]
    ref = fa_reference(Qm, K, V, scale)
    got = read_f32(m, ro, Bq, d)
    err = np.abs(got - ref).max(axis=1) / np.abs(ref).max(axis=1)
    assert err.max() < 2 ** -7, f"max rel err {err.max():.3e}"


# --------------------------------------------------------------------------
# 4. LOAD.IDX with -1 skips, range fault, and VTOPK -> VEC->LOAD token edge.
# --------------------------------------------------------------------------
def test_load_idx_gather_and_skip():
    m = Machine()
    nrows, d = 64, 128
    table = rng.standard_normal((1000, d))
    rt = put_bf16(m, 0, table)
    rt.stride = (dt.pad64(d * 2), 2)               # pad rows to 64 B multiples
    # rewrite with padded stride
    bits = (bf16(table).view(np.uint32) >> 16).astype(np.uint16)
    for i in range(1000):
        m.dram_write(i * rt.stride[0], bits[i])
    idx = rng.choice(1000, nrows, replace=False).astype(np.float32)
    idx[5] = -1; idx[17] = -1
    ro = Region(1 << 22, "f32", (nrows, d), (d * 4, 4))

    p = Program()
    p.vfill(VEC(8), array=idx[None, :])              # VEC[8] is a row tile
    p.push(Q.VEC, Q.LOAD)
    p.pop(Q.LOAD, Q.VEC)
    p.loadidx(INP(0), rt, VEC(8), nrows=nrows, row_bytes=rt.stride[0], cols_valid=d)
    p.push(Q.LOAD, Q.MMA)
    # identity GEMM to move INP -> ACC for store
    eye = put_bf16(m, 1 << 20, np.eye(d, dtype=np.float32))
    p.load2d(WGT(0), eye, d, d); p.push(Q.LOAD, Q.MMA)
    p.pop(Q.MMA, Q.LOAD); p.pop(Q.MMA, Q.LOAD)
    p.mma(ACC(0), INP(0), WGT(0)); p.push(Q.MMA, Q.STORE)
    p.pop(Q.STORE, Q.MMA); p.store2d(ACC(0), ro, nrows, d)
    p.finish()
    st = m.run(p.instrs)
    assert st["idx_rows"] == nrows - 2 and st["idx_skips"] == 2
    got = read_f32(m, ro, nrows, d)
    for i in range(nrows):
        if idx[i] < 0:
            assert np.all(got[i] == 0)
        else:
            assert np.allclose(got[i], bf16(table[int(idx[i])]), atol=1e-6)


def test_mma_k_zero_faults():
    """Erratum E6: an MMA over an empty K is BAD_SHAPE (the RTL faults on k == 0 and k > FIX_K_MAX)."""
    m = Machine()
    ra = put_bf16(m, 0, rng.standard_normal((64, 128)))
    rb = put_bf16(m, 1 << 16, rng.standard_normal((128, 128)))
    p = Program()
    p.load2d(INP(0), ra, 64, 0); p.push(Q.LOAD, Q.MMA)
    p.load2d(WGT(0), rb, 0, 128); p.push(Q.LOAD, Q.MMA)
    p.pop(Q.MMA, Q.LOAD); p.pop(Q.MMA, Q.LOAD)
    p.mma(ACC(0), INP(0), WGT(0))
    p.finish()
    with pytest.raises(Fault) as e:
        m.run(p.instrs)
    assert e.value.code == "BAD_SHAPE"


def test_load_idx_range_fault():
    m = Machine()
    rt = Region(0, "bf16", (10, 128), (256, 2))
    p = Program()
    p.vfill(VEC(8), array=np.array([[1, 2, 10]], np.float32))     # 10 is out of range
    p.push(Q.VEC, Q.LOAD); p.pop(Q.LOAD, Q.VEC)
    p.loadidx(INP(0), rt, VEC(8), nrows=3, row_bytes=256, cols_valid=128)
    p.finish()
    with pytest.raises(Fault) as e:
        m.run(p.instrs)
    assert e.value.code == "IDX_RANGE"


def test_load_idx_without_token_faults():
    m = Machine()
    rt = Region(0, "bf16", (10, 128), (256, 2))
    p = Program()
    p.vfill(VEC(8), array=np.array([[1, 2, 3]], np.float32))
    p.loadidx(INP(0), rt, VEC(8), nrows=3, row_bytes=256, cols_valid=128)   # no VEC->LOAD token
    p.finish()
    with pytest.raises(Fault) as e:
        m.run(p.instrs)
    assert e.value.code in ("UNTOKENED_READ", "IDX_SHORT")


# --------------------------------------------------------------------------
# 5. Router top-k -> VCOUNT -> register LOOP (ragged expert loop, zero case).
# --------------------------------------------------------------------------
def test_vtopk_vcount_register_loop():
    m = Machine()
    scores = rng.standard_normal((1, 384)).astype(np.float32)
    ro = Region(1 << 22, "f32", (64, 1), (4, 4))
    p = Program()
    p.vfill(VEC(9), array=scores)
    p.vtopk(VEC(8), VEC(9), k=6)
    p.vcount(r=3, idx=VEC(8), bound=384)
    p.loop(r=3)                      # 6 iterations
    p.vfill(VEC(0), imm=1.0, rows_valid=64, cols_valid=1)
    p.endloop()
    p.vfill(VEC(10), array=np.array([[400, 401]], np.float32))
    p.vcount(r=4, idx=VEC(10), bound=384)
    p.loop(r=4)                      # zero iterations: body must be skipped, tokens included
    p.push(Q.VEC, Q.STORE)
    p.endloop()
    p.push(Q.VEC, Q.STORE); p.pop(Q.STORE, Q.VEC); p.store2d(VEC(0), ro, 64, 1)
    p.finish()
    st = m.run(p.instrs)
    assert st["ok"] and m.regs[3] == 6 and m.regs[4] == 0
    top = np.argsort(-scores[0])[:6]
    assert np.array_equal(m.sp[VEC(8)].data[0, :6].astype(int), top)


# --------------------------------------------------------------------------
# 6. f4.r16 dequant round trip (KV entry layout, Appendix A).
# --------------------------------------------------------------------------
def test_f4_r16_roundtrip():
    x = rng.standard_normal(576).astype(np.float32)
    raw = dt.encode_f4_r16_row(x)
    assert raw.size == 384
    y = dt.dequant_row("f4.r16", raw, ncols=576)
    # E2M1 with per-16 scale: expect ~15-25% relative error per element, far smaller on the row norm
    assert np.linalg.norm(y - x) / np.linalg.norm(x) < 0.25


# --------------------------------------------------------------------------
# 7. v0.1.1 additions: STORE.IDX scatter, LOOP r step (ceil), Machine.reset
# --------------------------------------------------------------------------
def test_store_idx_scatter_and_loop_step():
    m = Machine()
    rows = 5; d = 64
    src = rng.standard_normal((rows, d)).astype(np.float32)
    idx = np.array([[7, -1, 3, 9, 0]], np.float32)
    ro = Region(1 << 20, "f32", (16, d), (d * 4, 4))
    p = Program()
    p.vfill(ACC(0), array=src)
    p.vfill(VEC(8), array=idx); p.push(Q.VEC, Q.STORE); p.push(Q.VEC, Q.STORE)
    p.pop(Q.STORE, Q.VEC); p.pop(Q.STORE, Q.VEC)
    p.storeidx(ACC(0), ro, VEC(8), nrows=rows, cols_valid=d)
    # LOOP r with step: R2 = 109 tokens, step 64 -> 2 iterations; R3 = 0 -> none
    p.setreg(2, 109); p.setreg(3, 0)
    p.loop(r=2, step=64); p.vfill(VEC(0), imm=1.0, rows_valid=64, cols_valid=1); p.endloop()
    p.loop(r=3, step=64); p.push(Q.VEC, Q.STORE); p.endloop()
    p.finish()
    st = m.run(p.instrs)
    assert st["idx_rows"] == 4 and st["idx_skips"] == 1
    got = read_f32(m, ro, 16, d)
    for i, r in enumerate(idx[0].astype(int)):
        if r >= 0:
            assert np.array_equal(got[r], src[i])
    # reset lets a second kernel reuse tiles the first one read
    m.reset()
    p2 = Program(); p2.vfill(ACC(0), imm=0.0, rows_valid=1, cols_valid=1); p2.finish()
    assert m.run(p2.instrs)["ok"]


# --------------------------------------------------------------------------
# 8. v0.1.2: per-level incr (outer expert loop advancing inner-loop operands),
#    LOAD->STORE edge, sync ops free in the step model
# --------------------------------------------------------------------------
def test_per_level_incr_nested_loops():
    m = Machine()
    E, T = 3, 2
    # DRAM: E experts x T tiles of [64,128], each tile holds value (e*10 + t)
    stride_t, stride_e = 64 * 128 * 2, T * 64 * 128 * 2
    for e in range(E):
        for t in range(T):
            put_bf16(m, e * stride_e + t * stride_t, np.full((64, 128), e * 10 + t, np.float32))
    r = Region(0, "bf16", (64, 128), (256, 2))
    ro = Region(1 << 22, "f32", (E * T * 64, 1), (4, 4))
    p = Program()
    p.loop(n=E)
    p.loop(n=T)
    p.load2d(INP(0), r, 64, 128, incr={"region.base": stride_t, "region.base@1": stride_e, "dst_mod": 2})
    p.push(Q.LOAD, Q.VEC); p.pop(Q.VEC, Q.LOAD)
    p.endloop(); p.endloop(); p.finish()
    # check the instantiated operands at the front end
    from tcta.machine import _FrontEnd
    fe = _FrontEnd(m, p.instrs)
    seen = []
    while not fe.done:
        fe.dispatch()
        for q, dq in m.queues.items():
            while dq:
                ins = dq.popleft()
                if ins.op == "LOAD.2D":
                    seen.append((ins.args["region"].base, ins.args["dst"].slot))
    assert seen == [(e * stride_e + t * stride_t, t % 2) for e in range(E) for t in range(T)]


def test_load_to_store_edge_and_free_sync():
    m = Machine()
    rt = put_bf16(m, 0, np.arange(64 * 128, dtype=np.float32).reshape(64, 128))
    ro = Region(1 << 22, "bf16", (64, 128), (256, 2))
    p = Program()
    p.load2d(INP(0), rt, 64, 128); p.push(Q.LOAD, Q.STORE)
    p.pop(Q.STORE, Q.LOAD); p.store2d(INP(0), ro, 64, 128)
    p.finish()
    st = m.run(p.instrs)
    assert st["ok"] and st["sync"] == 2 and st["instr"] == 2
    assert np.array_equal(m.dram[ro.base: ro.base + 64 * 256], m.dram[0: 64 * 256])


# --------------------------------------------------------------------------
# 9. v0.1.3: ACC->WGT move (device-computed B operand), MMA.MASKED mul, row-offset
#    MMA into a 128-row STATE tile, VCOPY, VNEG
# --------------------------------------------------------------------------
def test_acc_to_wgt_move_and_state_rows():
    from tcta import STATE
    m = Machine()
    A = rng.standard_normal((64, 128)); B = rng.standard_normal((128, 128))
    ra = put_bf16(m, 0, A); rb = put_bf16(m, 1 << 16, B)
    mask = (np.tril(np.ones((64, 128), np.float32)))
    p = Program()
    p.load2d(INP(0), ra, 64, 128); p.push(Q.LOAD, Q.MMA)
    p.load2d(WGT(0), rb, 128, 128); p.push(Q.LOAD, Q.MMA)
    p.pop(Q.MMA, Q.LOAD); p.pop(Q.MMA, Q.LOAD)
    # C = A B (64x128) -> move to WGT[1] as a B operand, then D = A C^T? keep simple: D = A' * C with A' = A[:, :64]... 
    # Use masked product with multiplicative mask, then move C to WGT and reuse as B for E = A C
    p.vfill(ACC(2), array=mask); p.push(Q.VEC, Q.MMA); p.pop(Q.MMA, Q.VEC)   # [64,128] mask lives in ACC
    p.mma_masked(ACC(0), INP(0), WGT(0), ACC(2), mask_mode="mul"); p.push(Q.MMA, Q.VEC)
    p.pop(Q.VEC, Q.MMA)
    p.vcopy(ACC(1), ACC(0))                                          # exact f32 copy
    p.vneg(ACC(1))                                                   # ACC[1] = -C
    p.vcvt(WGT(1), ACC(0))                                           # bf16 move: WGT[1] = C (64 x 128, K=64)
    p.vcvt(STATE(0), ACC(1), row0=64)                                # f32 move into STATE rows 64..127
    p.push(Q.VEC, Q.MMA); p.push(Q.VEC, Q.MMA)
    p.pop(Q.MMA, Q.VEC); p.pop(Q.MMA, Q.VEC)
    # E = A[:, :64] @ C  (K = 64): emulate A[:, :64] with a fresh partial load
    p.finish()
    st = m.run(p.instrs); assert st["ok"]
    C = bf16(A) @ bf16(B) * mask
    assert np.allclose(m.sp[ACC(0)].data, C, atol=1e-3)
    assert np.allclose(m.sp[WGT(1)].data[:64], bf16(C), atol=1e-6)
    assert np.allclose(m.sp[STATE(0)].data[64:128, :128], -C, atol=1e-3)
    assert m.sp[STATE(0)].valid == (128, 128)


# --------------------------------------------------------------------------
# Phase 7: the accumulation orders are the RTL's (ISA §9, DeviceConfig docstring)
# --------------------------------------------------------------------------
def test_mma_and_rowsum_accumulation_orders_are_sequential():
    from tcta.machine import _mma_sum, _row_sum
    A = bf16(rng.standard_normal((5, 37)) * 3); B = bf16(rng.standard_normal((37, 9)) * 3)
    S = _mma_sum(A, B)
    ref = np.zeros((5, 9), np.float32)
    for i in range(5):
        for j in range(9):
            s = np.float32(0.0)
            for k in range(37):
                s = np.float32(s + np.float32(A[i, k] * B[k, j]))
            ref[i, j] = s
    assert np.array_equal(S, ref)
    X = rng.standard_normal((3, 50)).astype(np.float32) * 1e3
    r = _row_sum(X, 16)
    ref = np.zeros(3, np.float32)
    for i in range(3):
        part = [np.float32(0.0)] * 16
        for c in range(50):
            part[c % 16] = np.float32(part[c % 16] + X[i, c])
        while len(part) > 1:
            part = [np.float32(part[k] + part[k + 1]) for k in range(0, len(part), 2)]
        ref[i] = part[0]
    assert np.array_equal(r, ref)
    # the GEMM tile still matches the BLAS result to float32 accuracy
    m = Machine()
    A = rng.standard_normal((64, 128)); B = rng.standard_normal((128, 128))
    ra = put_bf16(m, 0, A); rb = put_bf16(m, 1 << 16, B)
    ro = Region(1 << 20, "f32", (64, 128), (128 * 4, 4))
    p = Program()
    p.load2d(INP(0), ra, 64, 128); p.push(Q.LOAD, Q.MMA)
    p.load2d(WGT(0), rb, 128, 128); p.push(Q.LOAD, Q.MMA)
    p.pop(Q.MMA, Q.LOAD); p.pop(Q.MMA, Q.LOAD)
    p.mma(ACC(0), INP(0), WGT(0)); p.push(Q.MMA, Q.STORE)
    p.pop(Q.STORE, Q.MMA); p.store2d(ACC(0), ro, 64, 128)
    p.finish()
    assert m.run(p.instrs)["ok"]
    assert np.allclose(read_f32(m, ro, 64, 128), bf16(A) @ bf16(B), rtol=1e-5, atol=1e-4)


def test_fixed48_accumulation_matches_its_scalar_definition():
    """Phase 8 K0.3, erratum E5: the KV260 array's fixed-point window, element for element, against a scalar
    transcription of the format (pre-shifted significands, exact integer sums per B level, one rounding)."""
    from tcta.machine import FIX_B_LEVELS, FIX_SA, FIX_SB, FIX_TOP, _fixed48_to_f32, _mma_sum_fixed48
    A = bf16(np.exp(rng.standard_normal((6, 40)) * 5) * rng.choice([-1.0, 1.0], (6, 40)))
    B = bf16(np.exp(rng.standard_normal((40, 7)) * 5) * rng.choice([-1.0, 1.0], (40, 7)))
    A[1, 3] = 0.0; A[2, :] = 0.0; B[5, 2] = -0.0; A[4, 9] = np.float32(2.0 ** -130)       # zeros, a subnormal
    F = _mma_sum_fixed48(A, B)

    def fields(x):
        b = int(np.float32(x).view(np.uint32)); s = (b >> 31) & 1; e = (b >> 23) & 0xff; f = (b >> 16) & 0x7f
        return s, (1 if e == 0 else e), (f if e == 0 else 0x80 | f)
    def shift(m, d, smax):
        return (m << (smax - d)) if d <= smax else (m >> (d - smax))
    for i in range(6):
        ea_max = max(1, max(fields(A[i, k])[1] for k in range(40)))
        for j in range(7):
            eb_max = max(1, max(fields(B[k, j])[1] for k in range(40)))
            acc = [0] * FIX_B_LEVELS
            for k in range(40):
                sa, ea, ma = fields(A[i, k]); sb, eb, mb = fields(B[k, j])
                a = shift(ma, ea_max - ea, FIX_SA) * (-1 if sa else 1)
                db = eb_max - eb
                for l in range(FIX_B_LEVELS):
                    lo = l * FIX_SB
                    inside = (db >= (lo + 1 if l else 0)) and (db <= lo + FIX_SB or l == FIX_B_LEVELS - 1)
                    if inside:
                        acc[l] += a * shift(mb, db - lo, FIX_SB) * (-1 if sb else 1)
            total = sum(acc[l] << ((FIX_B_LEVELS - 1 - l) * FIX_SB) for l in range(FIX_B_LEVELS))
            W = ea_max + eb_max - 268 - FIX_TOP - (FIX_B_LEVELS - 1) * FIX_SB
            ref = _fixed48_to_f32(np.array([total], np.int64), np.array([W], np.int64), np.zeros(1, bool), np.zeros(1, bool), np.zeros(1, bool))[0]
            assert F[i, j] == ref, (i, j, F[i, j], ref)
            assert abs(acc[0]) < 2 ** 47 and abs(acc[1]) < 2 ** 47


def test_fixed48_accumulation_accuracy_and_specials():
    from tcta.machine import _mma_sum, _mma_sum_fixed48
    A = bf16(rng.standard_normal((64, 128))); B = bf16(rng.standard_normal((128, 128)))
    exact = A.astype(np.float64) @ B.astype(np.float64)
    F = _mma_sum_fixed48(A, B); S = _mma_sum(A, B)
    ulp = np.spacing(np.abs(exact).astype(np.float32)).astype(np.float64)
    assert np.abs(F - exact).max() / ulp.min() >= 0                     # well-defined
    assert (np.abs(F.astype(np.float64) - exact) <= 0.5 * ulp + 1e-30).all()       # correctly rounded on Gaussian data
    assert np.abs(F - exact).max() <= np.abs(S - exact).max()                        # never worse than float32 sequential here
    # softmax-like A (one row maximum, a long tail): every product loses less than 2^-23 of the largest product the
    # row and column could form (machine._mma_sum_fixed48), plus the final rounding
    P = np.exp(rng.standard_normal((64, 128)) * 3); P /= P.max(axis=1, keepdims=True)
    A = bf16(P); exact = A.astype(np.float64) @ B.astype(np.float64)
    bound = np.abs(A).max(axis=1).astype(np.float64)[:, None] * np.abs(B).max(axis=0).astype(np.float64)[None, :]
    assert (np.abs(_mma_sum_fixed48(A, B) - exact) <= 128 * 2.0 ** -23 * bound + 2.0 ** -24 * np.abs(exact) + 1e-30).all()
    # IEEE specials and the extremes of the binary32 range
    A = bf16(np.array([[np.inf, 1.0], [np.inf, -np.inf], [np.nan, 1.0], [0.0, -0.0], [np.inf, 0.0], [-1.0, -0.0], [-np.inf, 2.0],
                       [3e38, 3e38], [1e-40, 1e-40], [2.0 ** -130, 2.0 ** -130]], np.float32))
    B = bf16(np.array([[1.0], [0.0]], np.float32))
    F = _mma_sum_fixed48(A, B).ravel(); S = _mma_sum(A, B).ravel()
    assert np.array_equal(np.isnan(F), np.isnan(S)) and np.array_equal(F[~np.isnan(F)], S[~np.isnan(S)])
    assert F[3] == 0.0 and not np.signbit(F[3])
    # the largest same-sign sum the window must hold (K = 128 maximal products) is exact
    A = bf16(np.full((1, 128), 1.9921875, np.float32)); B = bf16(np.full((128, 1), 1.9921875, np.float32))
    assert _mma_sum_fixed48(A, B)[0, 0] == np.float32(128 * 1.9921875 ** 2)
    # the device configuration selects it
    m = Machine(cfg=DeviceConfig(acc_mode="fixed48"))
    assert m.cfg.acc_mode == "fixed48"
