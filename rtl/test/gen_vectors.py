#!/usr/bin/env python
"""Programs -> Verilator vectors (phase 7, J3/J4).

    python rtl/test/gen_vectors.py              # every program below -> rtl/test/vectors/<name>/
    python rtl/test/gen_vectors.py gemm fa      # a subset

Each vector directory holds what `tb_top.sv` loads and what it checks against:
    stream.hex      §12 instruction words, one 128-bit hex word per line (extension words included)
    desc.hex        the descriptor table, one 384-bit hex entry per line (48 bytes, little-endian byte order reversed
                    into a big-endian hex literal, so $readmemh gives desc[i][383:0] with base at [63:0])
    dram.hex        initial DRAM, sparse $readmemh (@addr markers, one byte per line) covering every touched page
    expect_dram.hex the simulator's DRAM after the run, same pages
    expect.json     status (ok / fault code, queue, pc), t_deadlock, the touched pages, and the output checks
    ref.npy         (fa) the fixture's reference output for the §9 tolerance check
    sim_trace.txt   the simulator's step trace: step queue op pc
"""
from __future__ import annotations

import json
import os
import struct
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
for sub in ("sim", "model", "ref", os.path.join("compiler", "prototype")):
    sys.path.insert(0, os.path.join(ROOT, sub))

from tcta import ACC, INP, VEC, WGT, DeviceConfig, Fault, Machine, Program, Q, Region  # noqa: E402
from tcta import dtypes as dt  # noqa: E402
from tcta.encode import DESC_BYTES, WORD_BYTES, descriptor_table, encode, pack_region  # noqa: E402

PAGE = 4096
rng = np.random.default_rng(7)


def put_bf16(m, addr, arr):
    arr = np.asarray(arr, np.float32)
    bits = (dt.to_bf16(arr).view(np.uint32) >> 16).astype(np.uint16)
    m.dram_write(addr, bits)
    return Region(addr, "bf16", arr.shape, (arr.shape[1] * 2, 2))


# ----------------------------------------------------------------------------- programs
# each builder: (machine) -> (Program, checks, cfg) ; checks: list of dicts the harness evaluates on its DRAM

def prog_dma_copy(m):
    """LOAD.2D f32 into a VEC 2-D tile, STORE.2D f32 back: the DMA path alone."""
    X = rng.standard_normal((64, 64)).astype(np.float32)
    m.dram_write(0, X)
    rin = Region(0, "f32", (64, 64), (256, 4)); rout = Region(1 << 16, "f32", (64, 64), (256, 4))
    p = Program()
    p.load2d(VEC(0), rin, 64, 64); p.push(Q.LOAD, Q.STORE)
    p.pop(Q.STORE, Q.LOAD); p.store2d(VEC(0), rout, 64, 64); p.push(Q.STORE, Q.LOAD)
    p.pop(Q.LOAD, Q.STORE)
    p.finish()
    return p, [dict(kind="bitexact")], None


def prog_mma(m):
    """Full-tile MMAs, f32 stores: plain, accumulate, A from a converted ACC tile (plain and accumulate), so the
    harness measures the array's block period, drain latency and the writeback cost of each form (J5)."""
    A = rng.standard_normal((64, 128)); B = rng.standard_normal((128, 128))
    ra = put_bf16(m, 0, A); rb = put_bf16(m, 1 << 16, B)
    ro = Region(1 << 20, "f32", (64, 128), (128 * 4, 4)); ro2 = Region(1 << 21, "f32", (64, 128), (128 * 4, 4))
    p = Program()
    p.load2d(INP(0), ra, 64, 128); p.push(Q.LOAD, Q.MMA)
    p.load2d(WGT(0), rb, 128, 128); p.push(Q.LOAD, Q.MMA)
    p.pop(Q.MMA, Q.LOAD); p.pop(Q.MMA, Q.LOAD)
    p.mma(ACC(0), INP(0), WGT(0))
    p.mma(ACC(0), INP(0), WGT(0), accumulate=True); p.push(Q.MMA, Q.VEC)
    p.pop(Q.VEC, Q.MMA); p.vcvt(ACC(0), ACC(0), inplace=True); p.push(Q.VEC, Q.MMA); p.push(Q.VEC, Q.STORE)
    p.pop(Q.MMA, Q.VEC); p.mma(ACC(2), ACC(0), WGT(0)); p.mma(ACC(2), ACC(0), WGT(0), accumulate=True); p.push(Q.MMA, Q.STORE)
    p.pop(Q.STORE, Q.VEC); p.store2d(ACC(0), ro, 64, 128)
    p.pop(Q.STORE, Q.MMA); p.store2d(ACC(2), ro2, 64, 128)
    p.finish()
    return p, [dict(kind="bitexact")], None


def prog_vec(m):
    """The FA-style VEC sequence on one tile without SFU ops: bit-exact target for the VEC pipe."""
    A = rng.standard_normal((64, 128)); B = rng.standard_normal((128, 128))
    ra = put_bf16(m, 0, A); rb = put_bf16(m, 1 << 16, B)
    ro = Region(1 << 20, "f32", (64, 128), (128 * 4, 4)); ro2 = Region(1 << 21, "f32", (64, 1), (4, 4))
    p = Program()
    p.load2d(INP(0), ra, 64, 128); p.push(Q.LOAD, Q.MMA)
    p.load2d(WGT(0), rb, 128, 128); p.push(Q.LOAD, Q.MMA)
    p.pop(Q.MMA, Q.LOAD); p.pop(Q.MMA, Q.LOAD)
    p.mma(ACC(0), INP(0), WGT(0)); p.push(Q.MMA, Q.VEC)
    p.pop(Q.VEC, Q.MMA)
    p.vfill(VEC(1), imm=0.0, rows_valid=64, cols_valid=1)
    p.vrowmax(VEC(0), ACC(0)); p.vmax(VEC(0), vec=VEC(1))
    p.vsub(ACC(0), vec=VEC(0)); p.vmul(ACC(0), imm=0.125); p.vadd(ACC(0), imm=1.0)
    p.vrowsum(VEC(2), ACC(0)); p.vscale_acc(ACC(0), VEC(2)); p.vneg(ACC(0))
    p.vcopy(ACC(1), ACC(0)); p.vcvt(ACC(1), ACC(1), inplace=True)
    p.push(Q.VEC, Q.STORE); p.push(Q.VEC, Q.STORE)
    p.pop(Q.STORE, Q.VEC); p.store2d(ACC(0), ro, 64, 128)
    p.pop(Q.STORE, Q.VEC); p.store2d(VEC(2), ro2, 64, 1)
    p.finish()
    return p, [dict(kind="bitexact")], None


def prog_gemm(m):
    from sweeps import fam_gemm
    f = fam_gemm(); f.load(m)
    return f.prog, [dict(kind="bitexact")], None


def prog_fa(m):
    from sweeps import FA_DRAM, fam_fa, fixture
    f = fam_fa(); f.load(m)
    d = fixture("flash_attn_gqa")
    seq, dim, blk = d["O"].shape[1], d["O"].shape[2], 64
    np.save(os.path.join(OUT, "fa", "ref.npy"), np.asarray(d["O"][0, :blk], np.float32))
    return f.prog, [dict(kind="rowmax", addr=FA_DRAM["Output"], rows=blk, cols=dim, dtype="bf16", ref="ref.npy",
                         tol=2.0 ** -7)], None


def prog_loop(m):
    """Front-end timing (J5): dispatch rate, static and nested LOOP overhead, VCOUNT -> LOOP r, loop-variant row0.
    No DRAM check beyond bit-exactness; the harness's FE DISP trace lines are what `calibrate.py` reads."""
    X = rng.standard_normal((16, 16)).astype(np.float32); m.dram_write(0, X)
    idx = np.array([[0, 1, 2, 3, -1, -1, 5, 6]], np.float32); m.dram_write(1 << 12, idx)
    rin = Region(0, "f32", (16, 16), (64, 4)); ridx = Region(1 << 12, "f32", (1, 8), (32, 4))
    rout = Region(1 << 16, "f32", (16, 16), (64, 4)); rout2 = Region(1 << 17, "f32", (1, 16), (64, 4))
    p = Program()
    # idx vector -> VCOUNT R0 (= 6) -> LOOP r
    p.load2d(VEC(1), ridx, 1, 8); p.push(Q.LOAD, Q.VEC)
    p.pop(Q.VEC, Q.LOAD); p.vcount(0, VEC(1), bound=8)
    p.vfill(VEC(2), imm=0.0, rows_valid=1, cols_valid=16)
    p.loop(r=0, step=1); p.vadd(VEC(2), imm=2.0); p.endloop()                         # 6 iterations
    # static nested loops
    p.loop(n=3); p.vadd(VEC(2), imm=1.0); p.loop(n=2); p.vmul(VEC(2), imm=1.5); p.endloop(); p.endloop()
    # loop-variant operands: copy 16 rows in 4 blocks of 4 (row0 += 4 per iteration, one extension word each)
    p.push(Q.STORE, Q.LOAD)
    p.loop(n=4)
    p.pop(Q.LOAD, Q.STORE); p.load2d(VEC(3), rin, 4, 16, row0=0, incr={"row0": 4}); p.push(Q.LOAD, Q.STORE)
    p.pop(Q.STORE, Q.LOAD); p.store2d(VEC(3), rout, 4, 16, row0=0, incr={"row0": 4}); p.push(Q.STORE, Q.LOAD)
    p.endloop()
    p.pop(Q.LOAD, Q.STORE)
    p.push(Q.VEC, Q.STORE); p.pop(Q.STORE, Q.VEC); p.store2d(VEC(2), rout2, 1, 16)
    p.finish()
    return p, [dict(kind="bitexact")], None


def prog_gather(m):
    """The DMA paths the other programs do not touch (phase 8 K0.1, the slice-addressed row buffers): LOAD.IDX of
    f32 rows at an odd column offset (two -1 rows), of bf16 rows and of f4.r16 rows at an odd, non-16-aligned
    column offset (both as MMA B operands), STORE.IDX and an i32 STORE.2D."""
    X = (100 * rng.standard_normal((32, 64))).astype(np.float32); m.dram_write(0, X)
    rx = Region(0, "f32", (32, 64), (256, 4))
    idx = np.full((1, 128), -1, np.float32)
    idx[0, :16] = [3, 0, 31, -1, 7, 7, 12, 1, -1, 30, 2, 5, 9, 11, 13, 29]
    m.dram_write(1 << 12, idx); ridx = Region(1 << 12, "f32", (1, 128), (512, 4))
    B = rng.standard_normal((256, 128)); rb = put_bf16(m, 1 << 16, B)                     # 256-byte rows
    kidx = rng.permutation(256)[:128].astype(np.float32).reshape(1, -1); m.dram_write(1 << 13, kidx)
    ridk = Region(1 << 13, "f32", (1, 128), (512, 4))
    C = rng.standard_normal((64, 256)).astype(np.float32)                                 # f4.r16: 128 B nibbles + 16 B scales -> 192 B rows
    for r in range(64):
        m.dram_write((1 << 18) + r * 192, dt.encode_f4_r16_row(C[r]))
    rc = Region(1 << 18, "f4.r16", (64, 256), (192, 1))
    cidx = (rng.permutation(256)[:128] % 64).astype(np.float32).reshape(1, -1); m.dram_write(1 << 14, cidx)
    ridc = Region(1 << 14, "f32", (1, 128), (512, 4))
    A = rng.standard_normal((64, 128)); ra = put_bf16(m, 1 << 20, A)
    ro = Region(1 << 21, "f32", (32, 32), (128, 4)); ro2 = Region(1 << 22, "i32", (16, 32), (128, 4))
    ro3 = Region(1 << 23, "f32", (64, 128), (512, 4)); ro4 = Region(1 << 24, "f32", (64, 128), (512, 4))
    p = Program()
    p.load2d(VEC(8), ridx, 1, 128)
    p.loadidx(VEC(0), rx, VEC(8), 16, 256, 32, col0=5); p.push(Q.LOAD, Q.STORE)
    p.load2d(VEC(9), ridk, 1, 128)
    p.loadidx(WGT(0), rb, VEC(9), 128, 256, 128); p.push(Q.LOAD, Q.MMA)
    p.load2d(VEC(10), ridc, 1, 128)
    p.loadidx(WGT(1), rc, VEC(10), 128, 192, 128, col0=35); p.push(Q.LOAD, Q.MMA)
    p.load2d(INP(0), ra, 64, 128); p.push(Q.LOAD, Q.MMA)
    p.pop(Q.MMA, Q.LOAD); p.pop(Q.MMA, Q.LOAD); p.pop(Q.MMA, Q.LOAD)
    p.mma(ACC(0), INP(0), WGT(0)); p.push(Q.MMA, Q.STORE)
    p.mma(ACC(1), INP(0), WGT(1)); p.push(Q.MMA, Q.STORE)
    p.pop(Q.STORE, Q.LOAD); p.storeidx(VEC(0), ro, VEC(8), 16, 32); p.store2d(VEC(0), ro2, 16, 32)
    p.pop(Q.STORE, Q.MMA); p.store2d(ACC(0), ro3, 64, 128)
    p.pop(Q.STORE, Q.MMA); p.store2d(ACC(1), ro4, 64, 128)
    p.finish()
    return p, [dict(kind="bitexact")], None


def prog_neg_untokened(m):
    ra = put_bf16(m, 0, rng.standard_normal((64, 128))); rb = put_bf16(m, 1 << 16, rng.standard_normal((128, 128)))
    p = Program()
    p.load2d(INP(0), ra, 64, 128); p.push(Q.LOAD, Q.MMA)
    p.load2d(WGT(0), rb, 128, 128)           # no PUSH for this one
    p.pop(Q.MMA, Q.LOAD)
    p.mma(ACC(0), INP(0), WGT(0))
    p.finish()
    return p, [], None


def prog_neg_leak(m):
    p = Program(); p.push(Q.LOAD, Q.MMA); p.finish()
    return p, [], None


def prog_neg_deadlock(m):
    p = Program(); p.pop(Q.MMA, Q.LOAD); p.finish()
    return p, [], DeviceConfig(t_deadlock=200)


PROGRAMS = dict(dma_copy=prog_dma_copy, mma=prog_mma, vec=prog_vec, gemm=prog_gemm, fa=prog_fa, loop=prog_loop, gather=prog_gather,
                neg_untokened=prog_neg_untokened, neg_leak=prog_neg_leak, neg_deadlock=prog_neg_deadlock)
OUT = os.path.join(HERE, "vectors")


# ----------------------------------------------------------------------------- writers

def touched_pages(*drams):
    pages = set()
    for d in drams:
        nz = np.flatnonzero(d.reshape(-1, PAGE).any(axis=1))
        pages.update(int(x) for x in nz)
    return sorted(pages)


def write_dram_hex(path, dram, pages):
    with open(path, "w") as fh:
        for pg in pages:
            fh.write(f"@{pg * PAGE:x}\n")
            chunk = dram[pg * PAGE:(pg + 1) * PAGE]
            fh.write("\n".join(f"{b:02x}" for b in chunk) + "\n")


def write_vectors(name, builder):
    d = os.path.join(OUT, name); os.makedirs(d, exist_ok=True)
    m0 = Machine()
    prog, checks, cfg = builder(m0)
    dram0 = m0.dram.copy()
    m = Machine(cfg=cfg) if cfg else Machine()
    m.dram[:] = dram0
    m.trace = []
    status, fault = None, None
    try:
        status = m.run(prog.instrs)
    except Fault as e:
        fault = dict(code=e.code, queue=e.queue.value if e.queue else None, pc=e.pc)
    blob = encode(prog.instrs)
    descs, _ = descriptor_table(prog.instrs)
    n_desc = len(descs)
    off = 32 + n_desc * DESC_BYTES
    words = [int.from_bytes(blob[off + i * WORD_BYTES: off + (i + 1) * WORD_BYTES], "little")
             for i in range((len(blob) - off) // WORD_BYTES)]
    with open(os.path.join(d, "stream.hex"), "w") as fh:
        fh.write("\n".join(f"{w:032x}" for w in words) + "\n")
    with open(os.path.join(d, "desc.hex"), "w") as fh:
        fh.write("\n".join(int.from_bytes(pack_region(r), "little").to_bytes(DESC_BYTES, "big").hex() for r in descs) + "\n")
    pages = touched_pages(dram0, m.dram)
    with open(os.path.join(d, "pages.txt"), "w") as fh:
        fh.write("".join(f"{pg * PAGE:x}\n" for pg in pages))
    write_dram_hex(os.path.join(d, "dram.hex"), dram0, pages)
    write_dram_hex(os.path.join(d, "expect_dram.hex"), m.dram, pages)
    with open(os.path.join(d, "sim_trace.txt"), "w") as fh:
        fh.write("\n".join(f"{t} {q.value} {op} {pc}" for t, q, op, pc in m.trace) + "\n")
    meta = dict(name=name, n_instrs=len(prog.instrs), n_words=len(words), n_desc=n_desc,
                t_deadlock=(cfg or DeviceConfig()).t_deadlock, fifo_depth=(cfg or DeviceConfig()).fifo_depth,
                ok=fault is None, fault=fault, steps=status["steps"] if status else None,
                pages=[pg * PAGE for pg in pages], checks=checks)
    with open(os.path.join(d, "expect.json"), "w") as fh:
        json.dump(meta, fh, indent=1)
    print(f"{name:14s} instrs {len(prog.instrs):4d} words {len(words):4d} desc {n_desc:3d} pages {len(pages):3d} "
          f"{'ok steps ' + str(status['steps']) if status else 'fault ' + fault['code']}")


if __name__ == "__main__":
    names = sys.argv[1:] or list(PROGRAMS)
    os.makedirs(os.path.join(OUT, "fa"), exist_ok=True)
    for n in names:
        write_vectors(n, PROGRAMS[n])
