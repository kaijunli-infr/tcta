"""Functional simulator for ISA v0.1.

Executes a linear instruction stream on five decoupled queues with token
FIFOs, and enforces the token discipline of ISA §8 (UNTOKENED_READ /
UNTOKENED_WRITE / VALID_MISMATCH / ACC_NOT_CONVERTED / IDX_* / TOKEN_*).

It is a functional model: no timing. The scheduler is a fair round-robin
over queues; each queue executes its head instruction if it is not blocked
on a POP. Lack of progress is a deadlock fault.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from . import dtypes as dt
from .isa import EDGES, QUEUE_OF, SP, DeviceConfig, Fault, Instr, Q, Region, TRef


# --------------------------------------------------------------------------
# Tile bookkeeping
# --------------------------------------------------------------------------

@dataclass
class TileState:
    data: np.ndarray
    valid: tuple = (0, 0)          # (rows_valid, cols_valid)
    writer: Optional[Q] = None     # queue of last write
    write_t: int = -1              # logical time of last write
    readers: dict = field(default_factory=dict)  # queue -> last read time since write
    pready: bool = False           # ACC: BF16 P written in place (ISA §6.3)
    is_row: bool = False           # VEC row tile [1, T_VR]


class Scratchpads:
    def __init__(self, cfg: DeviceConfig):
        self.cfg = cfg
        shapes = {
            SP.INP: (cfg.T_M, cfg.T_K), SP.WGT: (cfg.T_K, cfg.T_N),
            SP.ACC: (cfg.T_M, cfg.T_N), SP.STATE: (cfg.T_S, cfg.T_S),
        }
        self.tiles: dict[tuple, TileState] = {}
        for sp, n in cfg.slots.items():
            for i in range(n):
                if sp == SP.VEC:
                    row = i >= cfg.vec_row_slot_start
                    shp = (1, cfg.T_VR) if row else (cfg.T_M, cfg.T_V)
                    self.tiles[(sp, i)] = TileState(np.zeros(shp, np.float32), is_row=row)
                else:
                    self.tiles[(sp, i)] = TileState(np.zeros(shapes[sp], np.float32))

    def __getitem__(self, t: TRef) -> TileState:
        key = (t.sp, t.slot)
        if key not in self.tiles:
            raise Fault("BAD_SHAPE", aux={"tref": repr(t)})
        return self.tiles[key]


# --------------------------------------------------------------------------
# Machine
# --------------------------------------------------------------------------

class Machine:
    def __init__(self, cfg: DeviceConfig | None = None, dram_bytes: int = 64 << 20,
                 check_tokens: bool = True):
        self.cfg = cfg or DeviceConfig()
        self.sp = Scratchpads(self.cfg)
        self.dram = np.zeros(dram_bytes, dtype=np.uint8)
        self.regs = [0] * self.cfg.n_regs
        self.reg_pending = [0] * self.cfg.n_regs
        self.fifos: dict[tuple, deque] = {e: deque() for e in EDGES}
        # last POP time per edge, used by the tile-level token check
        self.last_pop_t: dict[tuple, int] = {e: -1 for e in EDGES}
        self.queues: dict[Q, deque] = {q: deque() for q in Q if q != Q.FE}
        self.t = 0
        self.check = check_tokens
        self.finished = False
        self.status: dict = {}
        self.mailbox: dict = {}   # (chip, tag) -> (data, valid) for emulated COMM
        # optional step trace, (step, queue, op, pc) per executed instruction (syncs included); set to [] to enable
        self.trace: Optional[list] = None
        self.stats = {"instr": 0, "sync": 0, "dram_read_bytes": 0, "dram_write_bytes": 0,
                      "idx_rows": 0, "idx_skips": 0, "mma": 0}

    def reset(self) -> None:
        """Per-kernel reset: scratchpad contents are undefined between kernels, token state is cleared.
        DRAM, registers and stats persist."""
        for ts in self.sp.tiles.values():
            ts.writer, ts.write_t, ts.readers, ts.valid, ts.pready = None, -1, {}, (0, 0), False
        for f in self.fifos.values():
            f.clear()
        self.last_pop_t = {e: -1 for e in self.fifos}
        self.queues = {q: deque() for q in Q if q != Q.FE}
        self.finished = False

    def _trace(self, q, ins) -> None:
        if self.trace is not None:
            self.trace.append((self.t, q, ins.op, ins.pc))

    # ---------------- DRAM helpers ----------------
    def dram_write(self, addr: int, buf: np.ndarray) -> None:
        b = np.ascontiguousarray(buf).view(np.uint8).reshape(-1)
        self.dram[addr: addr + b.size] = b
        self.stats["dram_write_bytes"] += b.size

    def dram_read(self, addr: int, n: int) -> np.ndarray:
        if addr < 0 or addr + n > self.dram.size:
            raise Fault("PAGE_FAULT", aux={"addr": addr, "n": n})
        self.stats["dram_read_bytes"] += n
        return self.dram[addr: addr + n]

    # ---------------- token checks ----------------
    def _note_write(self, t: TRef, q: Q, valid: tuple, pc: int) -> TileState:
        ts = self.sp[t]
        if self.check:
            for rq, rt in ts.readers.items():
                if rq == q:
                    continue
                edge = (rq, q)
                if edge not in EDGES or self.last_pop_t[edge] < rt:
                    raise Fault("UNTOKENED_WRITE", q, pc, tile=repr(t), reader=rq.value)
        ts.writer, ts.write_t, ts.readers, ts.valid, ts.pready = q, self.t, {}, valid, False
        return ts

    def _note_read(self, t: TRef, q: Q, pc: int) -> TileState:
        ts = self.sp[t]
        if self.check and ts.writer is None:
            raise Fault("UNTOKENED_READ", q, pc, tile=repr(t), writer="never written")
        if self.check and ts.writer is not None and ts.writer != q:
            edge = (ts.writer, q)
            if edge not in EDGES or self.last_pop_t[edge] < ts.write_t:
                raise Fault("UNTOKENED_READ", q, pc, tile=repr(t), writer=ts.writer.value)
        ts.readers[q] = self.t
        return ts

    # ---------------- program execution ----------------
    def run(self, program: list[Instr], max_steps: int = 10_000_000) -> dict:
        fe = _FrontEnd(self, program)
        idle = 0
        steps = 0
        while not self.finished:
            progressed = fe.dispatch()
            for q, dq in self.queues.items():
                # SYNC instructions cost no issue slot: drain every leading PUSH/POP that can proceed,
                # then issue at most one work instruction (ISA §8.5).
                while dq and dq[0].op in ("PUSH", "POP"):
                    if not self._exec(dq[0], q):
                        break
                    self._trace(q, dq.popleft()); progressed = True; self.stats["sync"] += 1
                if dq and dq[0].op not in ("PUSH", "POP"):
                    if self._exec(dq[0], q):
                        self._trace(q, dq.popleft()); progressed = True; self.stats["instr"] += 1
                        while dq and dq[0].op == "PUSH" and self._exec(dq[0], q):
                            self._trace(q, dq.popleft()); self.stats["sync"] += 1
            self.t += 1
            steps += 1
            if fe.done and all(not dq for dq in self.queues.values()):
                self._finish(fe)
                break
            idle = 0 if progressed else idle + 1
            if idle > self.cfg.t_deadlock or steps > max_steps:
                blocked = {q.value: repr(dq[0]) for q, dq in self.queues.items() if dq}
                raise Fault("TOKEN_TIMEOUT", aux={"blocked": blocked, "fe_waiting": fe.waiting})
        return self.status

    def _finish(self, fe) -> None:
        leaks = {f"{a.value}->{b.value}": len(d) for (a, b), d in self.fifos.items() if d}
        if leaks:
            raise Fault("TOKEN_LEAK", aux={"fifos": leaks})
        self.finished = True
        self.status = {"ok": True, "steps": self.t, **self.stats}

    def _exec(self, ins: Instr, q: Q) -> bool:
        """Execute one instruction on queue q. Returns False if blocked."""
        op, a, pc = ins.op, ins.args, ins.pc
        if op == "POP":
            edge = (a["from"], q)
            if edge not in EDGES:
                raise Fault("BAD_EDGE", q, pc, edge=f"{a['from'].value}->{q.value}")
            f = self.fifos[edge]
            if not f:
                return False
            f.popleft()
            self.last_pop_t[edge] = self.t
            return True
        if op == "PUSH":
            edge = (q, a["to"])
            if edge not in EDGES:
                raise Fault("BAD_EDGE", q, pc, edge=f"{q.value}->{a['to'].value}")
            f = self.fifos[edge]
            if len(f) >= self.cfg.fifo_depth:
                return False
            f.append(self.t)
            return True
        handler = getattr(self, "_op_" + op.replace(".", "_"), None)
        if handler is None:
            raise Fault("BAD_OPCODE", q, pc, op=op)
        handler(ins, q)
        return True

    # ---------------- LOAD ----------------
    def _row_source(self, r: Region, i: int, j0: int, ncols: int):
        """Return (raw bytes, block_scales) for row i, cols [j0, j0+ncols)."""
        q = r.dtype
        nb = dt.row_bytes_for(q, ncols)
        if q in ("f4.b32", "f8.b32"):
            # payload: row i, element column j0 -> byte offset depends on packing
            per_el = 0.5 if q == "f4.b32" else 1
            addr = r.base + i * r.stride[0] + int(j0 * per_el)
            raw = self.dram_read(addr, nb)
            sb = r.scales_base + (i // 32) * r.scales_stride + j0 // 32
            scales = self.dram_read(sb, ncols // 32)
            return raw, scales
        if q == "f4.r16":
            addr = r.base + i * r.stride[0]  # whole row; j0 must be 0
            assert j0 == 0, "f4.r16 rows are loaded whole"
            return self.dram_read(addr, nb), None
        el = {"bf16": 2, "f32": 4, "i32": 4, "f8.t": 1}[q]
        addr = r.base + i * r.stride[0] + j0 * el
        return self.dram_read(addr, nb), None

    def _op_LOAD_2D(self, ins: Instr, q: Q) -> None:
        a = ins.args
        dst: TRef = a["dst"]; r: Region = a["region"]
        rows, cols = a["rows_valid"], a["cols_valid"]
        ts = self._note_write(dst, q, (rows, cols), ins.pc)
        if rows > ts.data.shape[0] or cols > ts.data.shape[1]:
            raise Fault("BAD_SHAPE", q, ins.pc, dst=repr(dst), rows=rows, cols=cols)
        row0, col0 = a.get("row0", 0), a.get("col0", 0)
        for i in range(rows):
            raw, scales = self._row_source(r, row0 + i, col0, cols)
            v = dt.dequant_row(r.dtype, raw, ncols=cols, scale_t=r.scale_t, block_scales=scales)
            if dst.sp in (SP.INP, SP.WGT):
                v = dt.to_bf16(v)
            ts.data[i, :cols] = v

    _op_LOAD_3D = _op_LOAD_2D  # 3-D strides are folded into row0/col0 by the assembler in v0.1

    def _op_LOAD_IDX(self, ins: Instr, q: Q) -> None:
        a = ins.args
        dst: TRef = a["dst"]; r: Region = a["region"]; idx: TRef = a["idx"]
        nrows, row_bytes = a["nrows"], a["row_bytes"]
        if dst.sp not in (SP.INP, SP.WGT, SP.ACC, SP.VEC):
            raise Fault("BAD_SHAPE", q, ins.pc, dst=repr(dst))
        if row_bytes % dt.ROW_ALIGN:
            raise Fault("BAD_SHAPE", q, ins.pc, row_bytes=row_bytes, reason="not 64B multiple")
        its = self._note_read(idx, q, ins.pc)
        if not its.is_row:
            raise Fault("BAD_SHAPE", q, ins.pc, idx=repr(idx), reason="index must be a row tile")
        if its.valid[1] < nrows:
            raise Fault("IDX_SHORT", q, ins.pc, have=its.valid[1], need=nrows)
        ncols = a["cols_valid"]; col0 = a.get("col0", 0)
        ts = self._note_write(dst, q, (nrows, ncols), ins.pc)
        if ncols > ts.data.shape[1]:
            raise Fault("BAD_SHAPE", q, ins.pc, dst=repr(dst), cols=ncols)
        row_elems = r.shape[1]
        indices = its.data[0, :nrows].astype(np.int64)
        for i, ri in enumerate(indices):
            if ri == -1:
                ts.data[i, :ncols] = 0.0
                self.stats["idx_skips"] += 1
                continue
            if ri < 0 or ri >= r.shape[0]:
                raise Fault("IDX_RANGE", q, ins.pc, i=i, index=int(ri), bound=r.shape[0])
            raw = self.dram_read(r.base + int(ri) * r.stride[0], row_bytes)
            v = dt.dequant_row(r.dtype, raw, ncols=row_elems, scale_t=r.scale_t)   # whole row, then slice
            vv = v[col0: col0 + ncols]
            ts.data[i, :ncols] = dt.to_bf16(vv) if dst.sp in (SP.INP, SP.WGT) else vv
            self.stats["idx_rows"] += 1

    # ---------------- STORE ----------------
    def _op_STORE_2D(self, ins: Instr, q: Q) -> None:
        a = ins.args
        src: TRef = a["src"]; r: Region = a["region"]
        rows, cols = a["rows_valid"], a["cols_valid"]
        ts = self._note_read(src, q, ins.pc)
        data = self._read_tile(src, ts, a.get("wide", False))
        if rows > data.shape[0] or cols > data.shape[1]:
            raise Fault("BAD_SHAPE", q, ins.pc, src=repr(src), rows=rows, cols=cols)
        row0, col0 = a.get("row0", 0), a.get("col0", 0)
        for i in range(rows):
            v = data[i, :cols]
            if r.dtype == "f32":
                buf = v.astype(np.float32)
            elif r.dtype == "i32":
                buf = np.rint(v).astype(np.int32)
            elif r.dtype == "bf16":
                buf = (dt.to_bf16(v).view(np.uint32) >> 16).astype(np.uint16)
            elif r.dtype == "f4.r16":
                buf = dt.encode_f4_r16_row(v)
            else:
                raise Fault("BAD_QUALIFIER", q, ins.pc, q=r.dtype)
            el = {"f32": 4, "i32": 4, "bf16": 2}.get(r.dtype, 0)
            self.dram_write(r.base + (row0 + i) * r.stride[0] + col0 * el, buf)

    def _op_STORE_IDX(self, ins: Instr, q: Q) -> None:
        """Row scatter (ISA §5.2): row i of src goes to DRAM row idx[i]; -1 skips; duplicates undefined."""
        a = ins.args
        src: TRef = a["src"]; r: Region = a["region"]; idx: TRef = a["idx"]
        nrows, ncols = a["nrows"], a["cols_valid"]
        its = self._note_read(idx, q, ins.pc)
        if not its.is_row:
            raise Fault("BAD_SHAPE", q, ins.pc, idx=repr(idx), reason="index must be a row tile")
        if its.valid[1] < nrows:
            raise Fault("IDX_SHORT", q, ins.pc, have=its.valid[1], need=nrows)
        ts = self._note_read(src, q, ins.pc)
        data = ts.data
        indices = its.data[0, :nrows].astype(np.int64)
        el = {"f32": 4, "i32": 4, "bf16": 2}
        if r.dtype not in el:
            raise Fault("BAD_QUALIFIER", q, ins.pc, q=r.dtype)
        for i, ri in enumerate(indices):
            if ri == -1:
                self.stats["idx_skips"] += 1
                continue
            if ri < 0 or ri >= r.shape[0]:
                raise Fault("IDX_RANGE", q, ins.pc, i=i, index=int(ri), bound=r.shape[0])
            v = data[i, :ncols]
            if r.dtype == "f32":
                buf = v.astype(np.float32)
            elif r.dtype == "i32":
                buf = np.rint(v).astype(np.int32)
            else:
                buf = (dt.to_bf16(v).view(np.uint32) >> 16).astype(np.uint16)
            self.dram_write(r.base + int(ri) * r.stride[0], buf)
            self.stats["idx_rows"] += 1

    def _read_tile(self, t: TRef, ts: TileState, wide: bool) -> np.ndarray:
        if not wide:
            return ts.data
        parts = [self.sp[TRef(t.sp, t.slot + k)].data for k in range(4)]
        return np.concatenate(parts, axis=1)

    # ---------------- MMA ----------------
    def _op_MMA(self, ins: Instr, q: Q, transpose_b: bool = False) -> None:
        a = ins.args
        acc: TRef = a["acc"]; A: TRef = a["a"]; B: TRef = a["b"]
        accumulate, wide = a.get("accumulate", False), a.get("wide", False)
        ats = self._note_read(A, q, ins.pc)
        if A.sp == SP.ACC and not ats.pready:
            raise Fault("ACC_NOT_CONVERTED", q, ins.pc, tile=repr(A))
        bts = self._note_read(B, q, ins.pc)
        M, K = ats.valid
        bK, N = bts.valid
        if transpose_b:
            bK, N = N, bK
        if K != bK:
            raise Fault("VALID_MISMATCH", q, ins.pc, a_valid=ats.valid, b_valid=bts.valid, transpose=transpose_b)
        Am = dt.to_bf16(ats.data[:M, :K])
        Bm = bts.data[:bK, :N] if not transpose_b else bts.data[:N, :K].T
        Bm = dt.to_bf16(Bm)
        prod = (Am.astype(np.float32) @ Bm.astype(np.float32)).astype(np.float32)
        if a.get("mask") is not None:
            mts = self._note_read(a["mask"], q, ins.pc)
            if a.get("mask_mode", "add") == "mul":
                prod = prod * mts.data[:M, :N]
            else:
                prod = prod + mts.data[:M, :N]
        r0 = a.get("row0", 0)
        if accumulate:
            cts = self._note_read(acc, q, ins.pc)
            if r0 == 0 and cts.valid != (M, N) and cts.valid != (0, 0):
                raise Fault("VALID_MISMATCH", q, ins.pc, acc_valid=cts.valid, want=(M, N))
            old = self._read_tile(acc, cts, wide)[r0: r0 + M, :N].copy()
        else:
            old = 0.0
        if r0:
            prev_valid = self.sp[acc].valid
            cts = self._note_write(acc, q, (max(prev_valid[0], r0 + M), N), ins.pc)
            cts.data[r0: r0 + M, :N] = old + prod
            self.stats["mma"] += 1
            return
        cts = self._note_write(acc, q, (M, N), ins.pc)
        out = old + prod
        if wide:
            for k in range(4):
                part = out[:, k * self.cfg.T_N:(k + 1) * self.cfg.T_N]
                sub = self.sp[TRef(acc.sp, acc.slot + k)]
                sub.data[:M, :part.shape[1]] = part
                sub.writer, sub.write_t, sub.readers, sub.valid = q, self.t, {}, (M, part.shape[1])
        else:
            cts.data[:M, :N] = out
        self.stats["mma"] += 1

    def _op_MMA_T(self, ins: Instr, q: Q) -> None:
        self._op_MMA(ins, q, transpose_b=True)

    def _op_MMA_MASKED(self, ins: Instr, q: Q) -> None:
        self._op_MMA(ins, q)

    # ---------------- VEC ----------------
    def _rowvec(self, t: TRef, q: Q, pc: int, M: int) -> np.ndarray:
        ts = self._note_read(t, q, pc)
        return ts.data[:M, 0]

    def _op_VROWMAX(self, ins: Instr, q: Q) -> None:
        a = ins.args; src = self._note_read(a["src"], q, ins.pc)
        M, N = src.valid
        d = self._note_write(a["dst"], q, (M, 1), ins.pc)
        d.data[:M, 0] = src.data[:M, :N].max(axis=1)

    def _op_VROWSUM(self, ins: Instr, q: Q) -> None:
        a = ins.args; src = self._note_read(a["src"], q, ins.pc)
        M, N = src.valid
        d = self._note_write(a["dst"], q, (M, 1), ins.pc)
        d.data[:M, 0] = src.data[:M, :N].sum(axis=1)

    def _binop(self, ins: Instr, q: Q, fn) -> None:
        a = ins.args
        ts = self._note_read(a["src"], q, ins.pc)
        M, N = ts.valid
        if "vec" in a:
            v = self._rowvec(a["vec"], q, ins.pc, M)[:, None]
        else:
            v = np.float32(a["imm"])
        out = fn(ts.data[:M, :N], v)
        ts = self._note_write(a["src"], q, (M, N), ins.pc)
        ts.data[:M, :N] = out

    def _op_VSUB(self, ins, q): self._binop(ins, q, lambda x, v: x - v)
    def _op_VMUL(self, ins, q): self._binop(ins, q, lambda x, v: x * v)
    def _op_VADD(self, ins, q): self._binop(ins, q, lambda x, v: x + v)
    def _op_VNEG(self, ins, q):
        a = ins.args; ts = self._note_read(a["src"], q, ins.pc); M, N = ts.valid
        ts = self._note_write(a["src"], q, (M, N), ins.pc); ts.data[:M, :N] = -ts.data[:M, :N]
    def _op_VMAX(self, ins, q): self._binop(ins, q, lambda x, v: np.maximum(x, v))

    def _op_VSCALE_ACC(self, ins: Instr, q: Q) -> None:
        a = ins.args; acc: TRef = a["acc"]; wide = a.get("wide", False)
        ts = self._note_read(acc, q, ins.pc)
        M = ts.valid[0]
        v = self._rowvec(a["vec"], q, ins.pc, M)[:, None]
        slots = range(4) if wide else range(1)
        for k in slots:
            sub = self._note_write(TRef(acc.sp, acc.slot + k), q, self.sp[TRef(acc.sp, acc.slot + k)].valid, ins.pc)
            Mk, Nk = sub.valid
            sub.data[:Mk, :Nk] *= v[:Mk]

    def _op_VGATE(self, ins: Instr, q: Q) -> None:
        a = ins.args; ts = self._note_read(a["acc"], q, ins.pc); M, N = ts.valid
        g = self._note_read(a["vec"], q, ins.pc).data[:M, :N]   # vec may be VEC (2-D) or ACC (ISA §5.3)
        ts = self._note_write(a["acc"], q, (M, N), ins.pc)
        if a["op"] == "mul":
            ts.data[:M, :N] *= g
        elif a["op"] == "add":
            ts.data[:M, :N] += g
        elif a["op"] == "decay":
            ts.data[:M, :N] *= np.exp2(g)
        else:
            raise Fault("BAD_QUALIFIER", q, ins.pc, op=a["op"])

    def _op_VCVT(self, ins: Instr, q: Q) -> None:
        a = ins.args; src: TRef = a["src"]; dst: TRef = a["dst"]
        ts = self._note_read(src, q, ins.pc); M, N = ts.valid
        v = dt.to_bf16(ts.data[:M, :N])
        if a.get("inplace") or dst == src:
            ts = self._note_write(src, q, (M, N), ins.pc)
            ts.data[:M, :N] = v
            ts.pready = True
        else:
            r0 = a.get("row0", 0)
            if dst.sp in (SP.STATE, SP.ACC) and not a.get("to_bf16", False):
                v = ts.data[:M, :N]                       # f32 move into a state/acc tile
            d = self._note_write(dst, q, (r0 + M, N), ins.pc) if r0 else self._note_write(dst, q, (M, N), ins.pc)
            if r0 + M > d.data.shape[0] or N > d.data.shape[1]:
                raise Fault("BAD_SHAPE", q, ins.pc, dst=repr(dst), rows=r0 + M, cols=N)
            d.data[r0: r0 + M, :N] = v

    def _op_VCOPY(self, ins: Instr, q: Q) -> None:
        """Exact copy between tiles of the same scratchpad kind (VEC->VEC, ACC->ACC)."""
        a = ins.args; src: TRef = a["src"]; dst: TRef = a["dst"]
        ts = self._note_read(src, q, ins.pc); M, N = ts.valid
        d = self._note_write(dst, q, (M, N), ins.pc)
        d.data[:M, :N] = ts.data[:M, :N]

    def _op_VMASK(self, ins: Instr, q: Q) -> None:
        a = ins.args; ts = self._note_read(a["acc"], q, ins.pc); M, N = ts.valid
        kind = a["kind"]
        qoff, koff = a.get("q_off", 0), a.get("k_off", 0)  # absolute positions of row 0 / col 0
        qi = np.arange(M)[:, None] + qoff
        kj = np.arange(N)[None, :] + koff
        if kind == "causal":
            m = kj > qi
        elif kind == "band":
            w = a["width"]
            m = (kj > qi) | (kj <= qi - w)
        elif kind == "tile":
            m = self._note_read(a["vec"], q, ins.pc).data[:M, :N] != 0
        else:
            raise Fault("BAD_QUALIFIER", q, ins.pc, kind=kind)
        ts = self._note_write(a["acc"], q, (M, N), ins.pc)
        ts.data[:M, :N][m] = -np.inf

    def _op_VEXP2(self, ins: Instr, q: Q) -> None:
        a = ins.args; ts = self._note_read(a["t"], q, ins.pc); M, N = ts.valid
        x = ts.data[:M, :N]
        if "vec" in a:
            x = x - self._rowvec(a["vec"], q, ins.pc, M)[:, None]
        with np.errstate(over="ignore"):
            y = np.exp2(x.astype(np.float64)).astype(np.float32)
        y[x <= -127] = 0.0
        ts = self._note_write(a["t"], q, (M, N), ins.pc)
        ts.data[:M, :N] = y

    def _op_VEXP2_SUM(self, ins: Instr, q: Q) -> None:
        self._op_VEXP2(ins, q)
        a = ins.args; ts = self.sp[a["t"]]; M, N = ts.valid
        d = self._note_write(a["dst"], q, (M, 1), ins.pc)
        d.data[:M, 0] = ts.data[:M, :N].sum(axis=1)

    def _op_VRCP(self, ins, q):
        a = ins.args; ts = self._note_read(a["t"], q, ins.pc); M, N = ts.valid
        ts = self._note_write(a["t"], q, (M, N), ins.pc); ts.data[:M, :N] = 1.0 / ts.data[:M, :N]

    def _op_VLOG2(self, ins, q):
        a = ins.args; ts = self._note_read(a["t"], q, ins.pc); M, N = ts.valid
        ts = self._note_write(a["t"], q, (M, N), ins.pc); ts.data[:M, :N] = np.log2(ts.data[:M, :N])

    def _op_VSIGMOID(self, ins, q):
        a = ins.args; ts = self._note_read(a["t"], q, ins.pc); M, N = ts.valid
        ts = self._note_write(a["t"], q, (M, N), ins.pc)
        with np.errstate(over="ignore"):
            ts.data[:M, :N] = 1.0 / (1.0 + np.exp(-ts.data[:M, :N]))

    def _op_VFILL(self, ins: Instr, q: Q) -> None:
        """Fill a tile with an immediate or with a host-provided array (test fixtures)."""
        a = ins.args; dst: TRef = a["dst"]
        ts = self.sp[dst]
        if "array" in a:
            arr = np.asarray(a["array"], dtype=np.float32)
            rows, cols = arr.shape
            ts = self._note_write(dst, q, (rows, cols), ins.pc); ts.data[:rows, :cols] = arr
        else:
            rows, cols = a["rows_valid"], a["cols_valid"]
            ts = self._note_write(dst, q, (rows, cols), ins.pc); ts.data[:rows, :cols] = a["imm"]

    def _op_VTOPK(self, ins: Instr, q: Q) -> None:
        a = ins.args; src = self._note_read(a["src"], q, ins.pc)
        M, N = src.valid; k = a["k"]
        if a.get("axis", 1) != 1 or M != 1:
            raise Fault("BAD_SHAPE", q, ins.pc, reason="v0.1 VTOPK: one row, axis=1")
        row = src.data[0, :N]
        order = np.argsort(-row, kind="stable")[:k]
        base = self.regs[a["base_reg"]] if "base_reg" in a else 0
        it = self._note_write(a["idx"], q, (1, k), ins.pc)
        it.data[0, :k] = (order + base).astype(np.float32)
        if "val" in a:
            vt = self._note_write(a["val"], q, (1, k), ins.pc); vt.data[0, :k] = row[order]

    def _op_VCOUNT(self, ins: Instr, q: Q) -> None:
        a = ins.args; it = self._note_read(a["idx"], q, ins.pc)
        n = it.valid[1]; bound = a["bound"]
        v = it.data[0, :n]
        self.regs[a["r"]] = int(((v >= 0) & (v < bound)).sum())
        self.reg_pending[a["r"]] -= 1

    def _op_VSEL(self, ins: Instr, q: Q) -> None:
        a = ins.args
        A = self._note_read(a["a"], q, ins.pc); B = self._note_read(a["b"], q, ins.pc)
        mk = self._note_read(a["mask"], q, ins.pc); M, N = A.valid
        d = self._note_write(a["dst"], q, (M, N), ins.pc)
        d.data[:M, :N] = np.where(mk.data[:M, :N] != 0, A.data[:M, :N], B.data[:M, :N])

    # ---------------- COMM (single-machine emulation) ----------------
    def _op_SEND(self, ins: Instr, q: Q) -> None:
        a = ins.args; ts = self._note_read(a["src"], q, ins.pc)
        self.mailbox[(a["chip"], a["tag"])] = (ts.data.copy(), ts.valid)

    def _op_RECV(self, ins: Instr, q: Q) -> None:
        a = ins.args; key = (a["chip"], a["tag"])
        if key not in self.mailbox:
            raise Fault("TOKEN_TIMEOUT", q, ins.pc, reason="RECV with nothing sent (emulated link)")
        data, valid = self.mailbox.pop(key)
        d = self._note_write(a["dst"], q, valid, ins.pc)
        d.data[...] = data[: d.data.shape[0], : d.data.shape[1]]


# --------------------------------------------------------------------------
# Front end: LOOP expansion, registers, dispatch into queues
# --------------------------------------------------------------------------

class _FrontEnd:
    def __init__(self, m: Machine, program: list[Instr]):
        self.m = m
        self.prog = program
        self.pc = 0
        self.loops: list[dict] = []   # stack of {start_pc, count, i, incr_scale}
        self.done = False
        self.waiting: Optional[str] = None
        self.window = 64              # max dispatched-but-unexecuted per queue

    def dispatch(self) -> bool:
        progressed = False
        for _ in range(16):
            if self.done:
                return progressed
            if any(len(dq) >= self.window for dq in self.m.queues.values()):
                return progressed
            ins = self.prog[self.pc]
            if ins.queue == Q.FE:
                if not self._control(ins):
                    return progressed
                progressed = True
                continue
            self.m.queues[ins.queue].append(self._instantiate(ins))
            if ins.op == "VCOUNT":
                self.m.reg_pending[ins.args["r"]] += 1
            self.pc += 1
            progressed = True
        return progressed

    def _iter_index(self, ins: Instr) -> dict:
        """Current iteration index for every enclosing loop that this pc is inside."""
        return {id(l): l["i"] for l in self.loops}

    def _instantiate(self, ins: Instr) -> Instr:
        """Apply loop-variant increments. Key syntax: "field" (innermost loop) or "field@L" where L is
        the loop level counted from the innermost (0) outward. Multiple levels may act on one field;
        contributions add. `_mod` keys rotate a slot by (i mod n) and may also carry @L."""
        if not ins.incr or not self.loops:
            return Instr(ins.op, ins.queue, dict(ins.args), ins.pc)
        args = dict(ins.args)
        depth = len(self.loops)
        for key, step in ins.incr.items():
            field, _, lvl = key.partition("@")
            lvl = int(lvl) if lvl else 0
            if lvl >= depth:
                continue                       # level not active (e.g. instruction outside that loop)
            i = self.loops[depth - 1 - lvl]["i"]
            if field == "region.base":
                r = args["region"]
                args["region"] = Region(r.base + i * step, r.dtype, r.shape, r.stride, r.scale_t, r.scales_base, r.scales_stride)
            elif field == "region.scales_base":
                r = args["region"]
                args["region"] = Region(r.base, r.dtype, r.shape, r.stride, r.scale_t, r.scales_base + i * step, r.scales_stride)
            elif field.endswith("_slot"):
                t: TRef = args[field[:-5]]
                args[field[:-5]] = TRef(t.sp, t.slot + i * step)
            elif field.endswith("_mod"):
                t: TRef = args[field[:-4]]
                args[field[:-4]] = TRef(t.sp, t.slot + (i % step))
            else:
                args[field] = args[field] + i * step
        return Instr(ins.op, ins.queue, args, ins.pc)

    def _control(self, ins: Instr) -> bool:
        a = ins.args
        if ins.op == "LOOP":
            if "r" in a:
                if self.m.reg_pending[a["r"]] > 0:
                    self.waiting = f"LOOP waits on R{a['r']}"
                    return False
                step = a.get("step", 1)
                count = -(-self.m.regs[a["r"]] // step) if self.m.regs[a["r"]] > 0 else 0
            else:
                count = a["n"]
            self.waiting = None
            if count == 0:
                self._skip_loop()
                return True
            self.loops.append({"start": self.pc + 1, "count": count, "i": 0})
            self.pc += 1
            return True
        if ins.op == "ENDLOOP":
            l = self.loops[-1]
            l["i"] += 1
            if l["i"] < l["count"]:
                self.pc = l["start"]
            else:
                self.loops.pop(); self.pc += 1
            return True
        if ins.op == "FINISH":
            self.done = True
            return True
        if ins.op in ("SETSCALE", "SETBASE", "SETMASK"):
            self.m.regs[a["r"]] = a["imm"]
            self.pc += 1
            return True
        raise Fault("BAD_OPCODE", Q.FE, ins.pc, op=ins.op)

    def _skip_loop(self) -> None:
        depth = 0
        while True:
            self.pc += 1
            op = self.prog[self.pc].op
            if op == "LOOP":
                depth += 1
            elif op == "ENDLOOP":
                if depth == 0:
                    self.pc += 1
                    return
                depth -= 1
