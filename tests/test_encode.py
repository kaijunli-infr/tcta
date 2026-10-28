"""J1 (phase 7): the binary instruction stream of ISA §12 round-trips every program and changes no result."""
import os
import struct
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "sim"))
sys.path.insert(0, os.path.join(HERE, "..", "compiler", "prototype"))

from tcta import ACC, INP, VEC, WGT, Machine, Program, Q, Region  # noqa: E402
from tcta import dtypes as dt  # noqa: E402
from tcta.encode import (DESC_BYTES, EXT, HEADER_BYTES, MAX_DESC, WORD_BYTES, EncodeError, decode, descriptor_table,  # noqa: E402
                         encode, listing_words, pack_region, unpack_region)
from tcta.isa import Instr  # noqa: E402

rng = np.random.default_rng(0)


def put_bf16(m, addr, arr):
    arr = np.asarray(arr, np.float32)
    bits = (dt.to_bf16(arr).view(np.uint32) >> 16).astype(np.uint16)
    m.dram_write(addr, bits)
    return Region(addr, "bf16", arr.shape, (arr.shape[1] * 2, 2))


GEMM_A = rng.standard_normal((64, 128)); GEMM_B = rng.standard_normal((128, 128))


def gemm_program(m):
    ra = put_bf16(m, 0, GEMM_A); rb = put_bf16(m, 1 << 16, GEMM_B)
    ro = Region(1 << 20, "f32", (64, 128), (128 * 4, 4))
    p = Program()
    p.load2d(INP(0), ra, 64, 128); p.push(Q.LOAD, Q.MMA)
    p.load2d(WGT(0), rb, 128, 128); p.push(Q.LOAD, Q.MMA)
    p.pop(Q.MMA, Q.LOAD); p.pop(Q.MMA, Q.LOAD)
    p.mma(ACC(0), INP(0), WGT(0)); p.push(Q.MMA, Q.STORE)
    p.pop(Q.STORE, Q.MMA); p.store2d(ACC(0), ro, 64, 128)
    p.finish()
    return p


def all_dumped_programs():
    import dump_programs as dp
    from cut import lower_to_tile_tir
    from tile_tir_to_program import tile_tir_to_program
    for name, func, smap, bx, dram, cfg, _ in dp.JOBS:
        f = lower_to_tile_tir(func, scope_map=smap)
        prog, _ = tile_tir_to_program(f, bx=bx, dram=dram, cfg=cfg)
        yield name, prog


def test_every_dumped_program_round_trips():
    seen = 0
    for name, prog in all_dumped_programs():
        blob = encode(prog.instrs)
        back = decode(blob)
        assert back == prog.instrs, name                      # op, queue, args, pc, incr
        descs, _ = descriptor_table(prog.instrs)
        n_words = sum(1 + (len(i.incr) + 1) // 2 for i in prog.instrs)
        assert len(blob) == HEADER_BYTES + len(descs) * DESC_BYTES + n_words * WORD_BYTES, name
        assert len(descs) <= MAX_DESC
        seen += 1
    assert seen >= 13


def test_stream_runs_identically_to_the_program():
    m1 = Machine(); p = gemm_program(m1)
    m2 = Machine(); gemm_program(m2)                           # same DRAM contents
    st1 = m1.run(p.instrs)
    st2 = m2.run(decode(encode(p.instrs)))
    assert st1["ok"] and st2["ok"]
    assert np.array_equal(m1.dram, m2.dram)


def test_immediates_are_binary32_and_the_assembler_rounds_them():
    p = Program()
    p.vfill(ACC(0), imm=-1e30, rows_valid=64, cols_valid=128)
    p.vmul(ACC(0), imm=0.06011229333333333)
    for ins in p.instrs:
        v = ins.args["imm"]
        assert struct.unpack("<f", struct.pack("<f", v))[0] == v
    assert decode(encode(p.instrs)) == p.instrs
    raw = Instr("VMUL", Q.VEC, dict(src=ACC(0), imm=0.06011229333333333), pc=0)
    with pytest.raises(EncodeError):
        encode([raw])


def test_every_remaining_opcode_and_incr_form_round_trips():
    p = Program()
    r = Region(4096, "f8.t", (16, 256), (256, 1), scale_t=0.5)
    r3 = Region(1 << 40, "f4.b32", (8, 128, 128), (1 << 20, 256, 0), scales_base=1 << 41, scales_stride=16)
    p.setreg(0, 7); p.setreg(1, -3)
    p.loop(n=4)
    p.load2d(INP(0), r, 16, 256, incr={"region.base": 4096, "dst_slot@1": 1})
    p.loadidx(ACC(1), r3, VEC(8), 64, 512, 128, col0=128, incr={"col0": 64, "region.scales_base@1": 16, "idx_mod": 2})
    p.loop(r=0, step=64)
    p.mma_t(ACC(0), INP(0), WGT(1), accumulate=True)
    p.mma_masked(ACC(0), INP(0), WGT(0), VEC(1), accumulate=True, mask_mode="mul", row0=64)
    p.mma(ACC(2), ACC(0), WGT(0), accumulate=False, wide=True, row0=0)
    p.endloop()
    p.endloop()
    p.vmask(ACC(0), "causal", q_off=-5, k_off=3)
    p.vmask(ACC(0), "band", q_off=0, k_off=0, width=128)
    p.vmask(ACC(0), "tile", vec=VEC(2))
    p.vexp2(VEC(0), vec=VEC(9)); p.vexp2(VEC(0))
    p.vexp2_sum(VEC(0), VEC(10), vec=VEC(9)); p.vexp2_sum(VEC(0), VEC(10))
    p.vrcp(VEC(10)); p.vlog2(VEC(10)); p.vsigmoid(ACC(0)); p.vneg(ACC(0))
    p.vsub(ACC(0), vec=VEC(9)); p.vadd(ACC(0), imm=1.5); p.vmax(ACC(0), imm=0.0); p.vmul(ACC(0), vec=VEC(8))
    p.vfill(VEC(8), imm=-1e30, rows_valid=1, cols_valid=512)
    p.vscale_acc(ACC(2), VEC(9), wide=True)
    p.vgate(ACC(0), VEC(1), "decay")
    p.vcvt(WGT(1), ACC(0)); p.vcvt(ACC(0), ACC(0), inplace=True); p.vcvt(ACC(1), ACC(0), row0=64, to_bf16=True)
    p.vcopy(VEC(3), VEC(2)); p.vrowmax(VEC(8), ACC(0)); p.vrowsum(VEC(8), ACC(0))
    p.vtopk(VEC(9), VEC(8), 6); p.vtopk(VEC(9), VEC(8), 6, val=VEC(10), base_reg=1)
    p.vcount(1, VEC(9), 2 ** 31 - 1)
    p.vsel(ACC(0), ACC(1), ACC(2), VEC(1))
    p.send(ACC(0), 1, 17); p.recv(ACC(1), 1, 17)
    p.storeidx(ACC(0), r3, VEC(9), 64, 128)
    p.store2d(ACC(2), r, 64, 512, wide=True)
    p.push(Q.VEC, Q.STORE); p.pop(Q.STORE, Q.VEC)
    p.finish()
    back = decode(encode(p.instrs))
    assert back == p.instrs
    ops = {i.op for i in p.instrs}
    from tcta.encode import OPCODES
    missing = set(OPCODES) - ops
    assert missing <= {"LOAD.3D", "RECV.ANY", "SETBASE", "SETMASK"}, missing


def test_descriptor_pack_and_limits():
    r = Region(123, "f4.r16", (1536, 576), (384, 0), 1.0, None, 0)
    assert unpack_region(pack_region(r)) == r
    with pytest.raises(EncodeError):
        pack_region(Region(0, "u4", (1, 1), (1, 1)))
    with pytest.raises(EncodeError):
        pack_region(Region(0, "bf16", (1, 1), (1, 1), scale_t=0.1))     # not a binary32 value
    p = Program()
    for i in range(MAX_DESC + 1):
        p.load2d(INP(0), Region(i * 4096, "bf16", (1, 1), (2, 2)), 1, 1)
    with pytest.raises(EncodeError):
        encode(p.instrs)
    with pytest.raises(EncodeError):
        encode([Instr("LOAD.2D", Q.LOAD, dict(dst=WGT(4096), region=Region(0, "bf16", (1, 1), (2, 2)), rows_valid=1,
                                              cols_valid=1, row0=0, col0=0), pc=0)])
    with pytest.raises(EncodeError):
        encode([Instr("VFILL", Q.VEC, dict(dst=ACC(0), array=np.zeros((2, 2))), pc=0)])


def test_word_layout_matches_section_12():
    m = Machine(); p = gemm_program(m)
    blob = encode(p.instrs)
    words = listing_words(blob)
    w0 = int(words[0], 16)
    assert w0 >> 125 == 0                      # LOAD queue
    assert (w0 >> 119) & 0x3F == 2             # LOAD.2D
    assert (w0 >> 80) & 0xFFFF == 0x0000       # INP[0]
    assert w0 & 0xFFFFFFFF == (64 << 16) | 128  # rows_valid | cols_valid
    w1 = int(words[1], 16)
    assert (w1 >> 119) & 0x3F == 0 and (w1 >> 112) & 0x7F == 1   # PUSH to MMA
    assert all((int(w, 16) >> 119) & 0x3F != EXT for w in words)  # no loop-variant fields in this program
