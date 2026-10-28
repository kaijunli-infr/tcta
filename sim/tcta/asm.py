"""Tiny Python assembler for ISA v0.1 instruction streams.

    p = Program()
    p.load2d(WGT(0), region, rows_valid=64, cols_valid=128)
    p.push(Q.LOAD, Q.MMA)
    p.pop(Q.MMA, Q.LOAD)
    p.mma(ACC(0), INP(0), WGT(0))
    ...
    p.finish()
    machine.run(p.instrs)

Instructions are appended in program order; the front end partitions them
by queue at run time (ISA §1).
"""
from __future__ import annotations

import struct

from .isa import QUEUE_OF, SP, Instr, Q, Region, TRef


def _f32(x: float) -> float:
    """Immediates are IEEE binary32 (ISA §12); the simulator has always applied them as float32, so rounding at
    emission changes no result and makes the Program the canonical form the binary encoding round-trips."""
    return struct.unpack("<f", struct.pack("<f", x))[0]


def INP(i): return TRef(SP.INP, i)
def WGT(i): return TRef(SP.WGT, i)
def ACC(i): return TRef(SP.ACC, i)
def VEC(i): return TRef(SP.VEC, i)
def STATE(i): return TRef(SP.STATE, i)


class Program:
    def __init__(self):
        self.instrs: list[Instr] = []

    def _emit(self, op: str, queue: Q | None = None, incr: dict | None = None, **args) -> Instr:
        q = queue or QUEUE_OF[op]
        args = {k: (_f32(v) if isinstance(v, float) else v) for k, v in args.items()}
        ins = Instr(op, q, args, pc=len(self.instrs), incr=incr or {})
        self.instrs.append(ins)
        return ins

    # SYNC -- issued in the producing / consuming queue's stream
    def push(self, frm: Q, to: Q): return self._emit("PUSH", queue=frm, to=to)
    def pop(self, to: Q, frm: Q): return self._emit("POP", queue=to, **{"from": frm})

    # LOAD / STORE
    def load2d(self, dst, region, rows_valid, cols_valid, row0=0, col0=0, incr=None):
        return self._emit("LOAD.2D", incr=incr, dst=dst, region=region, rows_valid=rows_valid,
                          cols_valid=cols_valid, row0=row0, col0=col0)

    def loadidx(self, dst, region, idx, nrows, row_bytes, cols_valid, col0=0, incr=None):
        return self._emit("LOAD.IDX", incr=incr, dst=dst, region=region, idx=idx, nrows=nrows,
                          row_bytes=row_bytes, cols_valid=cols_valid, col0=col0)

    def storeidx(self, src, region, idx, nrows, cols_valid, incr=None):
        return self._emit("STORE.IDX", incr=incr, src=src, region=region, idx=idx, nrows=nrows, cols_valid=cols_valid)

    def store2d(self, src, region, rows_valid, cols_valid, row0=0, col0=0, wide=False, incr=None):
        return self._emit("STORE.2D", incr=incr, src=src, region=region, rows_valid=rows_valid,
                          cols_valid=cols_valid, row0=row0, col0=col0, wide=wide)

    # MMA
    def mma(self, acc, a, b, accumulate=False, wide=False, row0=0, incr=None):
        return self._emit("MMA", incr=incr, acc=acc, a=a, b=b, accumulate=accumulate, wide=wide, row0=row0)

    def mma_t(self, acc, a, b, accumulate=False, incr=None):
        return self._emit("MMA.T", incr=incr, acc=acc, a=a, b=b, accumulate=accumulate)

    def mma_masked(self, acc, a, b, mask, accumulate=False, mask_mode="add", row0=0, incr=None):
        return self._emit("MMA.MASKED", incr=incr, acc=acc, a=a, b=b, mask=mask, accumulate=accumulate,
                          mask_mode=mask_mode, row0=row0)

    # VEC
    def vrowmax(self, dst, src): return self._emit("VROWMAX", dst=dst, src=src)
    def vrowsum(self, dst, src): return self._emit("VROWSUM", dst=dst, src=src)
    def vsub(self, src, vec=None, imm=None): return self._emit("VSUB", src=src, **_v(vec, imm))
    def vmul(self, src, vec=None, imm=None): return self._emit("VMUL", src=src, **_v(vec, imm))
    def vadd(self, src, vec=None, imm=None): return self._emit("VADD", src=src, **_v(vec, imm))
    def vneg(self, src): return self._emit("VNEG", src=src)
    def vmax(self, src, vec=None, imm=None): return self._emit("VMAX", src=src, **_v(vec, imm))
    def vscale_acc(self, acc, vec, wide=False): return self._emit("VSCALE_ACC", acc=acc, vec=vec, wide=wide)
    def vgate(self, acc, vec, gate_op):
        """gate_op in {mul, add, decay}; `vec` may be a 2-D VEC tile or an ACC tile."""
        ins = self._emit("VGATE", acc=acc, vec=vec); ins.args["op"] = gate_op; return ins
    def vcvt(self, dst, src, inplace=False, row0=0, to_bf16=None):
        """ACC->INP/WGT (bf16 move, the MMA B-operand path), ACC->ACC inplace (P-ready), or f32 move into STATE/ACC
        at row offset `row0`. `to_bf16` forces rounding for a STATE/ACC destination."""
        a = dict(dst=dst, src=src, inplace=inplace, row0=row0)
        if to_bf16 is not None: a["to_bf16"] = to_bf16
        return self._emit("VCVT", **a)
    def vcopy(self, dst, src): return self._emit("VCOPY", dst=dst, src=src)
    def vmask(self, acc, kind, q_off=0, k_off=0, width=None, vec=None):
        """kind in {causal, band (width), tile (vec)}; absent optionals are omitted (canonical form, §12)."""
        a = dict(acc=acc, kind=kind, q_off=q_off, k_off=k_off)
        if width is not None: a["width"] = width
        if vec is not None: a["vec"] = vec
        return self._emit("VMASK", **a)
    def vexp2(self, t, vec=None): return self._emit("VEXP2", t=t, **({"vec": vec} if vec is not None else {}))
    def vexp2_sum(self, t, dst, vec=None):
        return self._emit("VEXP2.SUM", t=t, dst=dst, **({"vec": vec} if vec is not None else {}))
    def vrcp(self, t): return self._emit("VRCP", t=t)
    def vlog2(self, t): return self._emit("VLOG2", t=t)
    def vsigmoid(self, t): return self._emit("VSIGMOID", t=t)
    def vfill(self, dst, array=None, imm=None, rows_valid=None, cols_valid=None):
        if array is not None:
            return self._emit("VFILL", dst=dst, array=array)
        return self._emit("VFILL", dst=dst, imm=imm, rows_valid=rows_valid, cols_valid=cols_valid)
    def vtopk(self, idx, src, k, val=None, base_reg=None):
        a = dict(idx=idx, src=src, k=k, axis=1)
        if val is not None: a["val"] = val
        if base_reg is not None: a["base_reg"] = base_reg
        return self._emit("VTOPK", **a)
    def vcount(self, r, idx, bound): return self._emit("VCOUNT", r=r, idx=idx, bound=bound)
    def vsel(self, dst, a, b, mask): return self._emit("VSEL", dst=dst, a=a, b=b, mask=mask)

    # COMM
    def send(self, src, chip, tag): return self._emit("SEND", src=src, chip=chip, tag=tag)
    def recv(self, dst, chip, tag): return self._emit("RECV", dst=dst, chip=chip, tag=tag)

    # control
    def loop(self, n=None, r=None, step=1):
        """Static count `n`, or register `r` with `step`: runs ceil(R[r]/step) iterations (0 if R[r] <= 0)."""
        return self._emit("LOOP", **({"n": n} if r is None else {"r": r, "step": step}))
    def endloop(self): return self._emit("ENDLOOP")
    def setreg(self, r, imm): return self._emit("SETSCALE", r=r, imm=imm)
    def finish(self): return self._emit("FINISH")


def _v(vec, imm):
    if vec is not None:
        return {"vec": vec}
    if imm is not None:
        return {"imm": imm}
    raise ValueError("need vec or imm")
