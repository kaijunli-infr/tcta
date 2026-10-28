"""Cycle model (phase 5, H1; phase 6: cores): a timing layer on top of the functional simulator.

    cm = CycleMachine(timing=CONFIG_C, cfg=DeviceConfig(...), dram_bytes=...)
    st = cm.run(prog.instrs)          # the functional run, token checks and faults unchanged
    t = cm.timing_report              # Timing: cycles, per-queue stalls, FIFO levels, channel utilization, Gantt trace

    mt = run_multicore(CONFIG_C_CORE, [dict(instrs=p0, load=l0, cfg=c0), dict(instrs=p1, ...)])
    mt.cycles, mt.eta_bw, mt.cores[i]  # c independent programs against the shared DMA channels (phase 6, I2)

The functional simulator runs first, exactly as `Machine.run` does (same scheduler, same token checks, same
numerics); while it runs, the front end's dispatch order and every executed instruction (with the operand
shapes it saw) are recorded. The timing pass then replays those per-queue streams through a discrete-event
scheduler with the device model of `timing.py`:

- each queue is in order; an instruction starts when it has been dispatched, its queue's previous
  instruction has issued, every POP before it has its token (the producer's PUSH fired at the producer's
  completion), a unit (array, DMA channel, the VEC/SFU pipe) is free, and its scratchpad bank ports are free;
- PUSH is free and takes effect at the producing instruction's completion; a PUSH into a full FIFO blocks
  its queue until the consumer pops (reported per edge);
- the front end dispatches one instruction per cycle; `LOOP r` waits for the VCOUNT that writes R[r];
- cores (phase 6): each core is one program with its own queues, scratchpads, FIFOs, arrays and VEC pipe;
  cores share only the DMA channels, served in arrival order (one request per channel per cycle, a striped
  request taking k channel slots); the wait for a channel is the core's `stall_arbiter`.

Rules in full: `docs/phase5-cycle-model-task.md` §4, `docs/phase6-mlp-task.md` §3 and
`docs/cycle-model-report.md` §1. The model never changes numerics: `CycleMachine.run` returns the same status
and leaves the same DRAM/scratchpad contents as `Machine.run`; `tests/test_cycle.py` checks that no
instruction starts before a token it popped was pushed, which is the one FAIL condition of the tasks.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from . import dtypes as dt
from .isa import EDGES, SP, Instr, Q, TRef
from .machine import Machine, _FrontEnd
from .timing import KV260, DeviceTiming

SFU_OPS = {"VEXP2", "VEXP2.SUM", "VRCP", "VLOG2", "VSIGMOID"}
WORK = ("PUSH", "POP")   # the free sync instructions, excluded from "work"


# --------------------------------------------------------------------------
# records
# --------------------------------------------------------------------------

@dataclass
class Rec:
    """One executed instruction in the timing pass."""
    pc: int
    queue: Q
    op: str
    dispatch: int
    start: int
    end: int
    busy: int                    # unit occupancy (array / channel transfer / VEC pipe cycles)
    stall_token: int = 0
    stall_unit: int = 0
    stall_bank: int = 0
    unit: int = 0                # array / channel index
    bytes: int = 0
    requests: int = 0
    macs: int = 0
    info: dict = field(default_factory=dict)


def _inflight_stats(intervals, cycles):
    """(mean, p90) of a quantity carried by [start, end) intervals (bytes in flight), over `cycles` cycles."""
    if cycles <= 0:
        return 0.0, 0
    ev = []
    for s, e, w in intervals:
        if e > s:
            ev.append((s, w)); ev.append((e, -w))
    ev.sort()
    lvl, last, acc, dur = 0, 0, 0.0, {}
    for t, d in ev:
        if t > last:
            acc += lvl * (t - last); dur[lvl] = dur.get(lvl, 0) + (t - last); last = t
        lvl += d
    if cycles > last:
        dur[0] = dur.get(0, 0) + (cycles - last)
    p90, cum = 0, 0
    for lv in sorted(dur):
        cum += dur[lv]
        if cum >= 0.9 * cycles:
            p90 = lv
            break
    return acc / cycles, p90


@dataclass
class Timing:
    cycles: int
    us: float
    timing: DeviceTiming
    per_queue: dict             # queue -> {n, busy, stall_token, stall_bank, stall_fifo, wait_unit, first, last}; stall_token
                                #   counts cycles the head waited for a token with its unit free, wait_unit the cycles it
                                #   waited for its own queue's unit (back-to-back serialization, not a stall)
    edges: dict                 # "X->Y" -> {pushes, pops, max_level, push_blocked_cycles, depth}
    channels: dict              # "read0" / "write0" -> busy cycles (this core's share)
    banks: dict                 # "ACC/bank2" -> conflict stall cycles charged on it
    recs: list                  # Rec per work instruction, in start order
    dram_read_bytes: int = 0
    dram_write_bytes: int = 0
    macs: int = 0
    fe_cycles: int = 0          # cycle of the last dispatch
    fe_loop_wait: int = 0       # cycles the front end waited on VCOUNT -> LOOP r
    sync_events: list = field(default_factory=list)   # (cycle, queue, "PUSH"|"POP", pc)
    stall_arbiter: int = 0      # cycles this core's LOAD.2D/STORE.2D requests waited for a channel (phase 6)
    stall_arbiter_gather: int = 0   # the same for gather requests (includes a gather's queueing behind its own requests)
    core: int = 0
    gather_bound: dict = field(default_factory=dict)  # requests delayed by the outstanding window / by a busy channel

    @property
    def mma_cycles(self) -> int:
        return self.per_queue.get("MMA", {}).get("busy", 0)

    @property
    def eta_mfu(self) -> float:
        """Useful MACs over peak MACs for the whole run."""
        return self.macs / (self.timing.peak_macs_per_cycle * self.cycles) if self.cycles else 0.0

    @property
    def eta_bw(self) -> float:
        """DRAM bytes moved (read + write) over the read-channel peak for the whole run."""
        return (self.dram_read_bytes + self.dram_write_bytes) / (self.timing.read_bytes_per_cycle * self.cycles) if self.cycles else 0.0

    @property
    def channel_util(self) -> float:
        """Read-channel busy cycles over (cycles x read channels)."""
        b = sum(v for k, v in self.channels.items() if k.startswith("read"))
        return b / (self.timing.read_channels * self.cycles) if self.cycles else 0.0

    def inflight_bytes(self) -> tuple:
        """(mean, p90) DRAM read bytes in flight (requested, not yet landed) over the run."""
        return _inflight_stats([(r.start, r.end, r.bytes) for r in self.recs if r.queue == Q.LOAD], self.cycles)

    def trace(self) -> list:
        """Gantt entries (start, queue, op, pc, end), start order."""
        return [(r.start, r.queue, r.op, r.pc, r.end) for r in self.recs]

    def step_trace(self, include_sync: bool = True) -> list:
        """The functional simulator's `Machine.trace` shape, (cycle, queue, op, pc), so `overlap_report.py`'s
        analyses run on cycles instead of issue steps. PUSH entries carry the cycle the token was accepted
        into the FIFO, POP entries the cycle the consumer took it."""
        ev = [(r.start, r.queue, r.op, r.pc) for r in self.recs]
        if include_sync:
            ev += self.sync_events
        ev.sort(key=lambda x: (x[0], x[1].value, x[3]))
        return ev

    def summary(self) -> str:
        q = ", ".join(f"{k}: busy {v['busy']} tok {v['stall_token']} bank {v['stall_bank']} fifo {v['stall_fifo']}"
                      for k, v in self.per_queue.items())
        return (f"{self.cycles} cycles ({self.us:.1f} us @ {self.timing.f_mhz:.0f} MHz) eta_mfu {self.eta_mfu:.3f} "
                f"eta_bw {self.eta_bw:.3f} | {q}")


@dataclass
class MultiTiming:
    """The timing of c programs run on c cores against the shared channels (phase 6, I2)."""
    cycles: int
    us: float
    timing: DeviceTiming
    cores: list                 # Timing per core
    channels: dict              # shared channel busy cycles
    dram_read_bytes: int = 0
    dram_write_bytes: int = 0
    macs: int = 0

    @property
    def n_cores(self) -> int:
        return len(self.cores)

    @property
    def eta_mfu(self) -> float:
        """Useful MACs over the chip's peak (every core's arrays) for the whole run."""
        return self.macs / (self.n_cores * self.timing.peak_macs_per_cycle * self.cycles) if self.cycles else 0.0

    @property
    def eta_bw(self) -> float:
        return (self.dram_read_bytes + self.dram_write_bytes) / (self.timing.read_bytes_per_cycle * self.cycles) if self.cycles else 0.0

    @property
    def channel_util(self) -> float:
        b = sum(v for k, v in self.channels.items() if k.startswith("read"))
        return b / (self.timing.read_channels * self.cycles) if self.cycles else 0.0

    @property
    def arbiter_wait(self) -> list:
        return [c.stall_arbiter for c in self.cores]

    def inflight_bytes(self) -> tuple:
        iv = [(r.start, r.end, r.bytes) for c in self.cores for r in c.recs if r.queue == Q.LOAD]
        return _inflight_stats(iv, self.cycles)

    def eta_mfu_chip(self, chip_arrays: int) -> float:
        """Useful MACs over a chip with `chip_arrays` arrays in total (so a 2-core run on a 4-array chip is charged
        for the idle arrays)."""
        peak = chip_arrays * self.timing.mac_rows * self.timing.mac_cols
        return self.macs / (peak * self.cycles) if self.cycles else 0.0

    def summary(self) -> str:
        return (f"{self.n_cores} cores: {self.cycles} cycles ({self.us:.1f} us) eta_mfu {self.eta_mfu:.3f} eta_bw {self.eta_bw:.3f} "
                f"channel util {self.channel_util:.3f} per-core cycles {[c.cycles for c in self.cores]} arbiter {self.arbiter_wait}")


# --------------------------------------------------------------------------
# recording front end
# --------------------------------------------------------------------------

class _RecFrontEnd(_FrontEnd):
    """The functional front end, recording the dispatch order: ("ins", Instr) per queue instruction (the very
    object the queue executes), (op, r, n) per control instruction that took effect."""

    def __init__(self, m, program):
        super().__init__(m, program)
        self.rec: list = []

    def _instantiate(self, ins):
        new = super()._instantiate(ins)
        self.rec.append(("ins", new, ins.incr))          # the template's loop-variant fields: its extension words
        return new

    def _control(self, ins):
        depth = len(self.loops)
        ok = super()._control(ins)
        if ok:
            count = self.loops[-1]["count"] if (ins.op == "LOOP" and len(self.loops) > depth) else 0
            self.rec.append((ins.op, ins.args.get("r"), ins.args.get("n"), count))
        return ok


# --------------------------------------------------------------------------
# the machine
# --------------------------------------------------------------------------

class CycleMachine(Machine):
    def __init__(self, timing: DeviceTiming | None = None, defer_timing: bool = False, **kw):
        super().__init__(**kw)
        self.timing = timing or KV260
        self.defer_timing = defer_timing          # True: record only; the timing pass runs in run_multicore
        self._execd: list = []
        self._fe_rec: list = []
        self.timing_report: Optional[Timing] = None

    # ---- functional run with recording (the body of Machine.run, with the recording front end) ----
    def run(self, program: list, max_steps: int = 10_000_000) -> dict:
        fe = _RecFrontEnd(self, program)
        idle = steps = 0
        self._execd, self._fe_rec = [], fe.rec
        while not self.finished:
            progressed = fe.dispatch()
            for q, dq in self.queues.items():
                while dq and dq[0].op in WORK:
                    if not self._exec(dq[0], q):
                        break
                    self._trace(q, dq.popleft()); progressed = True; self.stats["sync"] += 1
                if dq and dq[0].op not in WORK:
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
                from .isa import Fault
                blocked = {q.value: repr(dq[0]) for q, dq in self.queues.items() if dq}
                raise Fault("TOKEN_TIMEOUT", aux={"blocked": blocked, "fe_waiting": fe.waiting})
        if not self.defer_timing:
            self.timing_report = self._time()
            self.status["cycles"] = self.timing_report.cycles
        return self.status

    def _exec(self, ins: Instr, q: Q) -> bool:
        info = None if ins.op in WORK else self._shape_info(ins, q)
        ok = super()._exec(ins, q)
        if ok:
            self._execd.append((q, ins, info))
        return ok

    # ---- operand shapes as the functional simulator saw them, taken before execution ----
    def _shape_info(self, ins: Instr, q: Q) -> dict:
        a, op = ins.args, ins.op
        inf: dict = {}
        if op in ("MMA", "MMA.T", "MMA.MASKED"):
            ats, bts = self.sp[a["a"]], self.sp[a["b"]]
            M, K = ats.valid
            bK, N = bts.valid
            if op == "MMA.T":
                bK, N = N, bK
            if a.get("wide"):
                N = N * 4
            inf.update(M=M, K=K, N=N, acc=bool(a.get("accumulate")))
        elif op in ("LOAD.2D", "LOAD.3D"):
            r = a["region"]
            rows, cols = a["rows_valid"], a["cols_valid"]
            nb = dt.row_bytes_for(r.dtype, cols)
            if r.dtype in ("f4.b32", "f8.b32"):
                nb += cols // 32
            inf.update(rows=rows, cols=cols, bytes=rows * nb, dtype=r.dtype, requests=1)
        elif op == "LOAD.IDX":
            its = self.sp[a["idx"]]
            n = a["nrows"]
            idx = its.data[0, :n]
            live = int((idx != -1).sum())
            inf.update(rows=live, nrows=n, cols=a["cols_valid"], row_bytes=a["row_bytes"], bytes=live * a["row_bytes"],
                       dtype=a["region"].dtype, gather=True)
        elif op == "STORE.2D":
            r = a["region"]
            el = {"f32": 4, "i32": 4, "bf16": 2}.get(r.dtype, 1)
            rows, cols = a["rows_valid"], a["cols_valid"]
            inf.update(rows=rows, cols=cols, bytes=rows * cols * el, dtype=r.dtype, requests=1)
        elif op == "STORE.IDX":
            its = self.sp[a["idx"]]
            r = a["region"]
            n = a["nrows"]
            live = int((its.data[0, :n] != -1).sum())
            el = {"f32": 4, "i32": 4, "bf16": 2}.get(r.dtype, 1)
            rb = dt.pad64(a["cols_valid"] * el)
            inf.update(rows=live, nrows=n, cols=a["cols_valid"], row_bytes=rb, bytes=live * rb, dtype=r.dtype, gather=True)
        elif op in ("SEND", "RECV"):
            t = self.sp[a.get("src", a.get("dst"))]
            inf.update(bytes=t.valid[0] * t.valid[1] * 4)
        else:
            # VEC: the shape of the operand the op streams over
            key = next((k for k in ("src", "acc", "t", "a", "idx", "dst") if k in a and isinstance(a[k], TRef)), None)
            if op == "VFILL":
                if "array" in a:
                    M, N = np.asarray(a["array"]).shape
                else:
                    M, N = a["rows_valid"], a["cols_valid"]
            elif key is not None:
                M, N = self.sp[a[key]].valid
                if M == 0 and N == 0:                        # a destination not yet written (VROWMAX dst etc.)
                    M, N = self.sp[a[key]].data.shape
            else:
                M, N = 1, 1
            if a.get("wide"):
                N = N * 4
            inf.update(M=M, N=N)
        return inf

    # ---- which scratchpad tiles an instruction reads / writes (for the bank ports) ----
    @staticmethod
    def _access(ins: Instr) -> tuple:
        a, op = ins.args, ins.op
        reads, writes = [], []

        def wide_slots(t, wide):
            return [TRef(t.sp, t.slot + k) for k in range(4 if wide else 1)]

        if op in ("LOAD.2D", "LOAD.3D"):
            writes.append(a["dst"])
        elif op == "LOAD.IDX":
            writes.append(a["dst"]); reads.append(a["idx"])
        elif op == "STORE.2D":
            reads += wide_slots(a["src"], a.get("wide"))
        elif op == "STORE.IDX":
            reads += [a["src"], a["idx"]]
        elif op in ("MMA", "MMA.T", "MMA.MASKED"):
            reads += [a["a"], a["b"]]
            if a.get("mask") is not None:
                reads.append(a["mask"])
            if a.get("accumulate"):
                reads += wide_slots(a["acc"], a.get("wide"))
            writes += wide_slots(a["acc"], a.get("wide"))
        elif op in ("SEND",):
            reads.append(a["src"])
        elif op in ("RECV",):
            writes.append(a["dst"])
        else:
            for k in ("src", "a", "b", "mask", "vec", "idx", "t", "acc"):
                if isinstance(a.get(k), TRef):
                    reads.append(a[k])
            for k in ("dst", "idx", "val"):
                if op != "LOAD.IDX" and isinstance(a.get(k), TRef) and (k != "idx" or op in ("VTOPK",)):
                    writes.append(a[k])
            # in-place ops write their operand
            inplace_key = {"VSUB": "src", "VMUL": "src", "VADD": "src", "VMAX": "src", "VNEG": "src", "VGATE": "acc",
                           "VMASK": "acc", "VEXP2": "t", "VEXP2.SUM": "t", "VRCP": "t", "VLOG2": "t", "VSIGMOID": "t",
                           "VCVT": "src" if (a.get("inplace") or a.get("dst") == a.get("src")) else None}.get(op)
            if op == "VSCALE_ACC":
                writes += wide_slots(a["acc"], a.get("wide")); reads += wide_slots(a["acc"], a.get("wide"))
            elif inplace_key and isinstance(a.get(inplace_key), TRef):
                writes.append(a[inplace_key])
            if op == "VCVT" and not (a.get("inplace") or a.get("dst") == a.get("src")):
                writes.append(a["dst"])
        return reads, writes

    # ---- cost models ----
    def _vec_cycles(self, op: str, inf: dict, args: dict) -> int:
        T = self.timing
        M, N = inf.get("M", 1), inf.get("N", 1)
        if T.rtl_rules:
            return _rtl_vec_cycles(T, op, M, N, args)
        lanes = T.sfu_lanes if op in SFU_OPS else T.vec_lanes
        if op == "VEXP2.SUM":
            passes = 1 if T.sfu_fused_sub_sum else 3
        elif op == "VEXP2" and "vec" in args:
            passes = 1 if T.sfu_fused_sub_sum else 2
        elif op == "VTOPK":
            passes = 2
        else:
            passes = 1
        if M == 1 or N == 1:
            elems = max(M, N)
        else:
            elems = M * N
        return max(1, -(-elems // lanes)) * passes

    # ---- the timing pass (one core) ----
    def _time(self) -> Timing:
        return _replay(self.timing, [self])[0]


# --------------------------------------------------------------------------
# the v0 RTL rules (DeviceTiming.rtl_rules; constants measured, docs/calibration-report.md)
# --------------------------------------------------------------------------

def _slices(n: int, lanes: int) -> int:
    return -(-n // lanes) if n > 0 else 0


def _rtl_dma_cycles(T: DeviceTiming, q: Q, inf: dict) -> int:
    """Row-serial DMA engine: per row the memory latency, the row's beats, one scratchpad slice per 16 columns and
    the row-step overhead (LOAD); slice reads, beats and overhead (STORE); setup / metadata per instruction. A
    gather adds the index-slice read per 16 rows (2 cycles) and one cycle per skipped (-1) row."""
    rows, cols = inf.get("rows", 0), inf.get("cols", 0)
    bpc = T.bytes_per_cycle_per_channel
    ncg = _slices(cols, T.vec_lanes)
    if inf.get("gather"):
        row_bytes = inf.get("row_bytes", 0)
    else:
        row_bytes = inf.get("bytes", 0) // rows if rows else 0
    beats = -(-row_bytes // bpc)
    if q == Q.LOAD:
        per_row = T.dram_latency_cycles + beats + ncg + T.dma_load_row_overhead
        total = T.dma_load_setup_cycles + rows * per_row
    else:
        per_row = T.dma_store_slice_cycles * ncg + beats + T.dma_store_row_overhead
        total = T.dma_store_setup_cycles + rows * per_row
    if inf.get("gather"):
        n = inf.get("nrows", rows)
        total += 2 * _slices(n, T.vec_lanes) + (n - rows)
    return total


def _rtl_vec_cycles(T: DeviceTiming, op: str, M: int, N: int, args: dict) -> int:
    """One [row, 16-column] slice per operand read per cycle; a row-vector operand one read per row; SFU ops hold
    the slice for sfu_slice_cycles; VEXP2.SUM one more per slice plus its row-vector read and row-sum write;
    VTOPK k passes over the row; per-instruction overhead."""
    L = T.vec_lanes
    ncg = _slices(N, L)
    has_vec = isinstance(args.get("vec"), TRef)
    if op in ("VEXP2", "VRCP", "VLOG2", "VSIGMOID"):
        per_row = ncg * T.sfu_slice_cycles                       # the row-vector read overlaps the SFU steps
    elif op == "VEXP2.SUM":
        per_row = ncg * (T.sfu_slice_cycles + T.sfu_sum_slice_extra) + 2
    elif op == "VTOPK":
        per_row = int(args.get("k", 1)) * (ncg + 1)
    else:
        reads = {"VFILL": 1, "VSEL": 3, "VGATE": 2}.get(op, 1)   # VFILL: one write per slice
        if op == "VMASK" and isinstance(args.get("vec"), TRef):
            reads, has_vec = 2, False                            # tile mask: mask slice + acc slice
        if op == "VSCALE_ACC":
            has_vec = True
        per_row = reads * ncg + (1 if has_vec else 0)
    return T.vec_op_overhead_cycles + M * per_row


def _rtl_mma_cycles(T: DeviceTiming, M: int, K: int, N: int, accumulate: bool, masked: bool) -> tuple:
    """-> (cycles issue to done, array occupancy). Blocks (row-block major) start every P = ceil16(max(K, 16))
    cycles once the result FIFO holds at most mma_rf_rows - 16 committed rows; a block's valid result rows appear
    P + mma_result_latency + i cycles after its start and are written back one at a time (a row's writeback
    starts when the previous one's started mma_wb_row_cycles (+ extras) ago); padding rows of the last row block
    cost nothing; metadata write and done: 2 cycles."""
    R, C = T.mac_rows, T.mac_cols
    P = max(16, -(-K // 16) * 16)
    nbi, nbj = -(-M // R), -(-N // C)
    nb = nbi * nbj
    wb = T.mma_wb_row_cycles + (T.mma_wb_acc_extra if accumulate else 0) + (T.mma_wb_mask_extra if masked else 0)
    t, started, last_start, committed, wb_free = 0, 0, None, 0, 0
    pending: list = []                                             # arrival cycles of valid rows, in order
    pi = 0
    while started < nb or pi < len(pending):
        if started < nb and (last_start is None or t - last_start >= P) and committed <= T.mma_rf_rows - R:
            i0 = (started // nbj) * R
            valid = min(R, M - i0)
            pending.extend(t + P + T.mma_result_latency + i for i in range(valid))
            committed += valid
            started += 1
            last_start = t
        if pi < len(pending) and pending[pi] <= t and wb_free <= t:
            wb_free = t + wb
            committed -= 1
            pi += 1
        if started >= nb and pi >= len(pending):
            break
        t += 1
    return t + 2, nb * P


# --------------------------------------------------------------------------
# the replay: c cores against shared channels
# --------------------------------------------------------------------------

class _Core:
    """Per-core replay state: the recorded streams and everything the event loop mutates."""

    def __init__(self, cid: int, m: CycleMachine, T: DeviceTiming):
        self.cid, self.m = cid, m
        self.info = {id(ins): inf for q, ins, inf in m._execd}
        self.streams: dict = {q: [] for q in Q if q != Q.FE}
        self.fe_items = list(m._fe_rec)
        for item in self.fe_items:
            if item[0] == "ins":
                self.streams[item[1].queue].append(item[1])
        self.queues = [q for q in self.streams if self.streams[q]]
        self.disp: dict = {}
        self.end_of: dict = {}
        self.fe_i = self.fe_time = self.fe_loop_wait = 0
        self.last_vcount: dict = {}
        self.idx = {q: 0 for q in self.queues}
        self.disp_cnt = {q: 0 for q in self.queues}        # dispatched / taken (IQ occupancy under rtl_rules)
        self.taken_cnt = {q: 0 for q in self.queues}
        self.ready_after = {q: 0 for q in self.queues}
        self.last_end = {q: 0 for q in self.queues}
        self.token_ready = {q: 0 for q in self.queues}
        self.stall_fifo = {q: 0 for q in self.queues}
        self.push_times: dict = {e: [] for e in EDGES}
        self.pop_times: dict = {e: [] for e in EDGES}
        self.blocked: dict = {e: 0 for e in EDGES}
        self.arrays = [0] * T.n_arrays
        self.vec_free = 0
        self.holds: dict = {}                              # (sp, bank, dir) -> [(start, end, share_key)]
        self.bank_stall: dict = {}
        self.recs: list = []
        self.sync_events: list = []
        self.per_q = {q.value: dict(n=0, busy=0, stall_token=0, stall_bank=0, stall_fifo=0, wait_unit=0, first=None, last=0)
                      for q in self.queues}
        self.totals = dict(rb=0, wb=0, macs=0)
        self.ch_busy = {f"read{i}": 0 for i in range(T.read_channels)} | {f"write{i}": 0 for i in range(T.write_channels)}
        self.stall_arbiter = 0
        self.stall_arbiter_gather = 0
        self.gather_bound = dict(window=0, channel=0, requests=0)


def _replay(T: DeviceTiming, machines: list) -> list:
    """Replay the recorded runs of `machines` (one per core) through the device model; -> [Timing per core].
    Shared across cores: the DMA channels (arrival-order arbiter). Everything else is per core."""
    cores = [_Core(i, m, T) for i, m in enumerate(machines)]
    rch = [0] * T.read_channels                       # busy-until per read channel (shared)
    wch = [0] * T.write_channels
    order = [Q.LOAD, Q.MMA, Q.VEC, Q.STORE, Q.COMM]

    def bank_of(t: TRef):
        return (t.sp, t.slot % T.banks.get(t.sp, 1))

    culprits: list = []

    def port_wait(c: _Core, items, t):
        """Earliest cycle >= t at which every needed port has a free slot, given the recorded holds (intervals
        [start, end) of earlier-processed instructions). Only holds that contain the candidate cycle count: a
        hold that starts later does not block (the later-starting instruction was processed first because its
        queue was ahead; the model does not go back and delay it, an approximation stated in the report).
        Holders sharing the requester's non-None share key (operand broadcast) do not count. Records the banks
        that imposed the wait in `culprits`."""
        need = {}
        for x, d, key in items:
            need[(bank_of(x), d)] = key
        tw = t
        advanced: dict = {}
        changed = True
        while changed:
            changed = False
            for (b, d), key in need.items():
                cap = T.ports_per_bank[d]
                iv = c.holds.get((b, d), [])
                tb = tw
                while True:
                    active = sorted(e for (s, e, k) in iv if s <= tb < e and (key is None or k != key))
                    if len(active) < cap:
                        break
                    tb = active[len(active) - cap]            # when enough holders have released
                if tb > tw:
                    tw = tb
                    advanced[b] = tb
                    changed = True
        culprits[:] = [b for b, v in advanced.items() if v == tw]
        return tw

    def hold(c: _Core, items, start, end, prune_before):
        """Record the port holds [start, end); drop holds that ended before `prune_before` (no instruction
        processed later can start earlier than that: ready times are processed in nondecreasing order)."""
        for x, d, key in items:
            k = (bank_of(x), d)
            lst = c.holds.setdefault(k, [])
            lst[:] = [iv for iv in lst if iv[1] > prune_before]
            lst.append((start, end, key))

    def dma(c: _Core, chs, t_issue, inf, requests, req_bytes, gather):
        """Issue DMA requests at t_issue on the earliest-free channel(s); -> (channels, xfer start, end, busy,
        arbiter wait). Requests are served in arrival order across cores because the event loop processes
        issue times in nondecreasing order."""
        bpc = T.bytes_per_cycle_per_channel
        if inf.get("dtype") in ("f4.b32", "f8.b32", "f8.t", "f4.r16"):
            bpc = min(bpc, T.dequant_bytes_per_cycle)
        lat = T.dram_latency_cycles
        k = max(1, min(T.request_channels, len(chs)))
        us = sorted(range(len(chs)), key=lambda i: chs[i])[:k]
        if not gather:
            # striped over the k least-busy channels (k = 1: one request, one channel); busy is channel-cycles,
            # so a striped request of the same bytes costs the same busy on k channels
            busy_until = max(chs[i] for i in us)
            xfer = -(-req_bytes // (bpc * k))
            s = max(busy_until, t_issue + lat)
            e = s + xfer
            for i in us:
                chs[i] = e
            return us, s, e, xfer * k, max(0, busy_until - (t_issue + lat))
        # gather: one request per row / per 64-byte burst, one issue per cycle, max_outstanding in flight.
        # k = 1: every request of the gather queues on one channel; k > 1: round-robin over the k least-busy
        # channels (address-interleaved rows)
        until = {i: chs[i] for i in us}
        xfer = max(1, -(-req_bytes // bpc))
        completions: list = []
        first_s, e_last, busy, wait = None, t_issue, 0, 0
        for j in range(requests):
            ch = us[j % k]
            t_req = t_issue + j
            if j >= T.max_outstanding_rows and completions[j - T.max_outstanding_rows] > t_req:
                t_req = completions[j - T.max_outstanding_rows]
                c.gather_bound["window"] += 1
            s = max(until[ch], t_req + lat)
            if until[ch] > t_req + lat:
                c.gather_bound["channel"] += 1
                wait += until[ch] - (t_req + lat)
            e = s + xfer
            until[ch] = e
            busy += xfer
            completions.append(e)
            e_last = max(e_last, e)
            if first_s is None or s < first_s:
                first_s = s
        c.gather_bound["requests"] += requests
        for i in us:
            chs[i] = until[i]
        return us, first_s if first_s is not None else t_issue, e_last, busy, wait

    # ---- readiness of each stream head ----
    def fe_ready(c: _Core):
        if c.fe_i >= len(c.fe_items):
            return None
        item = c.fe_items[c.fe_i]
        if item[0] == "LOOP" and item[1] is not None:
            vc = c.last_vcount.get(item[1])
            if vc is not None:
                if id(vc) not in c.end_of:
                    return None
                count = item[3] if len(item) > 3 else 0
                return max(c.fe_time, c.end_of[id(vc)] + T.vcount_to_loop_latency + T.loop_r_count_cycles * count)
        if item[0] == "ins" and T.iq_depth:
            q = item[1].queue
            if c.disp_cnt[q] - c.taken_cnt[q] >= T.iq_depth:
                return None                                   # the IQ is full: wait for the queue to take one
        return c.fe_time

    def head_ready(c: _Core, q):
        i = c.idx[q]
        if i >= len(c.streams[q]):
            return None
        ins = c.streams[q][i]
        if id(ins) not in c.disp:
            return None
        if ins.op == "PUSH":
            e = (q, ins.args["to"])
            k = len(c.push_times[e])
            d = T.fifo_depth_of(e)
            room = 0
            if k >= d:
                if len(c.pop_times[e]) <= k - d:
                    return None
                room = c.pop_times[e][k - d]
            return max(c.disp[id(ins)], c.last_end[q], room)
        if ins.op == "POP":
            e = (ins.args["from"], q)
            k = len(c.pop_times[e])
            if len(c.push_times[e]) <= k:
                return None
            return max(c.push_times[e][k] + T.push_visible_cycles, c.disp[id(ins)], c.ready_after[q])
        t = max(c.disp[id(ins)], c.ready_after[q], c.token_ready[q])
        if q == Q.MMA:
            t = max(t, min(c.arrays))
        elif q == Q.VEC:
            t = max(t, c.vec_free)
        return t

    # ---- event loop: always the earliest ready head; ties rotate across cores (a round-robin arbiter for
    #      requests issued in the same cycle, so identical programs on c cores do not always lose to core 0),
    #      then FE before the queues in `order` ----
    rr, nc = 0, len(cores)
    while True:
        cands = []
        for c in cores:
            r = fe_ready(c)
            if r is not None:
                cands.append((r, c.cid, -1, None))
            for n, q in enumerate(order):
                if q in c.idx:
                    r = head_ready(c, q)
                    if r is not None:
                        cands.append((r, c.cid, n, q))
        if not cands:
            left = {f"core{c.cid}/{q.value}": len(c.streams[q]) - c.idx[q] for c in cores for q in c.queues if c.idx[q] < len(c.streams[q])}
            if left or any(c.fe_i < len(c.fe_items) for c in cores):
                heads = {}
                for c in cores:
                    for k, v in _why_blocked_impl(c.streams, c.idx, c.disp, c.push_times, c.pop_times, c.fe_i, c.fe_items).items():
                        heads[f"core{c.cid}/{k}" if len(cores) > 1 else k] = v
                raise CycleDeadlock(heads, left)
            break
        t, _, cid, _, q = min((r, (cid - rr) % nc, cid, n, q) for r, cid, n, q in cands)
        rr += 1
        c = cores[cid]
        if q is None:
            # front end dispatch
            item = c.fe_items[c.fe_i]
            c.fe_i += 1
            if item[0] == "ins":
                ins = item[1]
                if T.rtl_rules:
                    # fetch, decode, dispatch per word (PUSH / POP words included); extension words and their entries
                    incr = item[2] if len(item) > 2 else {}
                    n_ent = len(incr)
                    cost = T.fe_sync_word_cycles if ins.op in WORK else T.fe_word_cycles
                    if n_ent:
                        cost += -(-n_ent // 2) * T.fe_ext_word_cycles + n_ent * T.fe_ext_entry_cycles
                        cost += sum(1 for k in incr if k.endswith("_mod")) * T.fe_mod_entry_cycles
                    c.fe_time += cost
                    c.disp[id(ins)] = c.fe_time
                else:
                    c.disp[id(ins)] = c.fe_time
                    if ins.op not in WORK:
                        c.fe_time += 1
                c.disp_cnt[ins.queue] += 1
                if ins.op == "VCOUNT":
                    c.last_vcount[ins.args["r"]] = ins
            elif item[0] in ("LOOP", "ENDLOOP"):
                c.fe_loop_wait += t - c.fe_time
                c.fe_time = t + T.loop_overhead_cycles
            else:
                c.fe_time = t + 1
            continue
        ins = c.streams[q][c.idx[q]]
        c.idx[q] += 1
        c.taken_cnt[q] += 1
        a = ins.args
        if ins.op == "PUSH":
            # fires at the producer's completion; free (task §1): the queue goes on issuing while the producer
            # is in flight (a scoreboard fires the token). Only a full FIFO holds the queue, until the token is
            # accepted.
            e = (q, a["to"])
            room_wait = t - max(c.disp[id(ins)], c.last_end[q])
            c.blocked[e] += room_wait
            c.stall_fifo[q] += room_wait
            c.push_times[e].append(t)
            if room_wait:
                c.ready_after[q] = max(c.ready_after[q], t)
            c.sync_events.append((t, q, "PUSH", ins.pc))
            continue
        if ins.op == "POP":
            e = (a["from"], q)
            c.pop_times[e].append(t)
            c.token_ready[q] = max(c.token_ready[q], t)
            c.sync_events.append((t, q, "POP", ins.pc))
            continue
        # ---- a work instruction ----
        # t0: dispatched and the queue's previous instruction issued (and its PUSHes accepted);
        # t_u: the queue's own unit is free (the previous MMA's array occupancy, the VEC pipe);
        # t1: every popped token has arrived; then the scratchpad ports.
        inf = c.info.get(id(ins), {}) or {}
        t0 = max(c.disp[id(ins)], c.ready_after[q])
        reads, writes = c.m._access(ins)
        items = [(x, 0, None) for x in reads] + [(x, 1, None) for x in writes]
        unit = 0
        s_bank = 0
        s_arb = 0
        if q == Q.MMA:
            unit = min(range(len(c.arrays)), key=lambda i: c.arrays[i])
            t_u = max(t0, c.arrays[unit])
        elif q == Q.VEC:
            t_u = max(t0, c.vec_free)
        else:
            t_u = t0
        t1 = max(t_u, c.token_ready[q])
        s_tok, s_unit = t1 - t_u, t_u - t0
        if q == Q.MMA:
            if T.operand_broadcast:
                # the arrays of one core read one A tile at once: A reads share the port among MMAs
                items = [(x, d, ("A", x) if (d == 0 and x == a["a"]) else k) for x, d, k in items]
            t3 = port_wait(c, items, t1); s_bank = t3 - t1
            M, K, N = inf["M"], inf["K"], inf["N"]
            if T.rtl_rules:
                # the block schedule of the v0 array; the queue holds the next MMA until this one is written back
                tot, occ = _rtl_mma_cycles(T, M, K, N, bool(a.get("accumulate")), a.get("mask") is not None)
                start, end, busy = t3, t3 + tot, occ
                c.arrays[unit] = end
                hold(c, items, start, end, t1)
            else:
                occ = T.mma_occupancy(M, K, N)
                start, end, busy = t3, t3 + occ + T.mma_latency, occ
                c.arrays[unit] = t3 + occ
                # ports are held while the array streams (occupancy), not through the pipeline latency: a
                # back-to-back accumulate into the same ACC forwards (task §4); consumers wait via tokens
                hold(c, items, start, start + occ, t1)
            c.totals["macs"] += M * K * N
            rec = Rec(ins.pc, q, ins.op, c.disp[id(ins)], start, end, busy, s_tok, s_unit, s_bank, unit, macs=M * K * N, info=inf)
        elif q == Q.VEC:
            t3 = port_wait(c, items, t1); s_bank = t3 - t1
            cyc = c.m._vec_cycles(ins.op, inf, a)
            start, end, busy = t3, t3 + cyc, cyc
            c.vec_free = end
            hold(c, items, start, end, t1)
            rec = Rec(ins.pc, q, ins.op, c.disp[id(ins)], start, end, busy, s_tok, s_unit, s_bank, 0, info=inf)
        elif q in (Q.LOAD, Q.STORE):
            chs = rch if q == Q.LOAD else wch
            gather = bool(inf.get("gather"))
            if gather:
                rows, rb = inf.get("rows", 0), inf.get("row_bytes", 64)
                if T.row_gather_cost == "burst64":
                    per_row = -(-rb // 64)
                    requests, req_bytes = rows * per_row, 64
                else:
                    requests, req_bytes = rows, rb
            else:
                requests, req_bytes = 1, inf.get("bytes", 0)
            # the channel is the unit: the instruction issues at t1 (the DMA engine queues requests behind the
            # channel's current transfer); the data tile's ports are held for the transfer; a gather reads its
            # index vector while it issues requests (one per cycle)
            idx_items = [(a["idx"], 0, None)] if gather and isinstance(a.get("idx"), TRef) else []
            data_items = [it for it in items if it not in idx_items]
            if requests == 0:
                start = end = t1; busy = 0; unit = 0; units = []
            elif T.rtl_rules:
                # the row-serial engine holds one channel for the whole instruction; a busy channel queues it
                unit = min(range(len(chs)), key=lambda i: chs[i]); units = [unit]
                s_x = max(chs[unit], t1); s_arb = s_x - t1
                busy = _rtl_dma_cycles(T, q, inf)
                end = s_x + busy
                chs[unit] = end
                start = t1
                t3 = port_wait(c, data_items, s_x); s_bank = t3 - s_x
                if s_bank:
                    end += s_bank
                    chs[unit] += s_bank
                hold(c, data_items, s_x + s_bank, end, t1)
                if idx_items:
                    hold(c, idx_items, t1, end, t1)
            else:
                units, s_x, end, busy, s_arb = dma(c, chs, t1, inf, requests, req_bytes, gather)
                unit = units[0]
                start = t1
                t3 = port_wait(c, data_items, s_x); s_bank = t3 - s_x
                if s_bank:
                    end += s_bank
                    for i in units:
                        chs[i] += s_bank
                hold(c, data_items, s_x + s_bank, end, t1)
                if idx_items:
                    hold(c, idx_items, t1, t1 + requests, t1)
            for i in units:
                c.ch_busy[("read" if q == Q.LOAD else "write") + str(i)] += busy // len(units)
            if q == Q.LOAD:
                c.totals["rb"] += inf.get("bytes", 0)
            else:
                c.totals["wb"] += inf.get("bytes", 0)
            if gather:
                c.stall_arbiter_gather += s_arb
            else:
                c.stall_arbiter += s_arb
            rec = Rec(ins.pc, q, ins.op, c.disp[id(ins)], start, end, busy, s_tok, 0, s_bank, unit,
                      bytes=inf.get("bytes", 0), requests=requests, info=inf)
        else:  # COMM
            cyc = -(-inf.get("bytes", 0) // T.link_bytes_per_cycle) + T.link_latency_cycles
            start, end, busy = t1, t1 + cyc, cyc
            rec = Rec(ins.pc, q, ins.op, c.disp[id(ins)], start, end, busy, s_tok, 0, 0, 0, bytes=inf.get("bytes", 0), info=inf)
        c.end_of[id(ins)] = end
        c.ready_after[q] = max(c.ready_after[q], (end + 1) if T.rtl_rules else (start + 1))
        c.last_end[q] = end
        c.token_ready[q] = 0
        c.recs.append(rec)
        pq = c.per_q[q.value]
        pq["n"] += 1; pq["busy"] += busy; pq["stall_token"] += s_tok; pq["wait_unit"] += s_unit; pq["stall_bank"] += s_bank
        pq["first"] = start if pq["first"] is None else min(pq["first"], start)
        pq["last"] = max(pq["last"], end)
        if s_bank:
            for b in culprits:
                c.bank_stall[f"{b[0].value}/bank{b[1]}"] = c.bank_stall.get(f"{b[0].value}/bank{b[1]}", 0) + s_bank

    # ---- totals, per core ----
    out = []
    for c in cores:
        cycles = max([r.end for r in c.recs] + [c.fe_time])
        for q in c.queues:
            c.per_q[q.value]["stall_fifo"] = c.stall_fifo[q]
        edges = {}
        for e in EDGES:
            if not c.push_times[e] and not c.pop_times[e]:
                continue
            ev = sorted([(t, 1) for t in c.push_times[e]] + [(t, -1) for t in c.pop_times[e]], key=lambda x: (x[0], x[1]))
            lvl = mx = 0
            for _, d in ev:
                lvl += d
                mx = max(mx, lvl)
            edges[f"{e[0].value}->{e[1].value}"] = dict(pushes=len(c.push_times[e]), pops=len(c.pop_times[e]), max_level=mx,
                                                       push_blocked_cycles=c.blocked[e], depth=T.fifo_depth_of(e))
        c.recs.sort(key=lambda r: (r.start, r.queue.value, r.pc))
        out.append(Timing(cycles=cycles, us=T.us(cycles), timing=T, per_queue=c.per_q, edges=edges, channels=c.ch_busy,
                          banks=c.bank_stall, recs=c.recs, dram_read_bytes=c.totals["rb"], dram_write_bytes=c.totals["wb"],
                          macs=c.totals["macs"], fe_cycles=c.fe_time, fe_loop_wait=c.fe_loop_wait, sync_events=c.sync_events,
                          stall_arbiter=c.stall_arbiter, stall_arbiter_gather=c.stall_arbiter_gather, core=c.cid,
                          gather_bound=dict(c.gather_bound)))
    return out


class CycleDeadlock(RuntimeError):
    """The timing pass found no head that can proceed: a token FIFO at the modelled depth blocks a producer whose
    consumer waits behind it (the functional run, at its own depth, did not deadlock). `heads` names every
    queue's head instruction and what it waits for."""

    def __init__(self, heads: dict, left: dict):
        self.heads, self.left = heads, left
        super().__init__(f"cycle model deadlock: {heads}; instructions left {left}")


def _why_blocked_impl(streams, idx, disp, push_times, pop_times, fe_i, fe_items):
    out = {}
    for q, lst in streams.items():
        i = idx.get(q, len(lst))
        if i >= len(lst):
            continue
        ins = lst[i]
        if id(ins) not in disp:
            out[q.value] = f"{ins.op}@pc{ins.pc}: not dispatched (front end at item {fe_i}/{len(fe_items)})"
        elif ins.op == "PUSH":
            e = (q, ins.args["to"])
            out[q.value] = f"PUSH {e[0].value}->{e[1].value} #{len(push_times[e])}: FIFO full, waits for pop #{len(pop_times[e])}"
        elif ins.op == "POP":
            e = (ins.args["from"], q)
            out[q.value] = f"POP {e[0].value}->{e[1].value} #{len(pop_times[e])}: waits for push #{len(push_times[e])}"
        else:
            out[q.value] = f"{ins.op}@pc{ins.pc}: ready"
    return out


CycleMachine._why_blocked = staticmethod(_why_blocked_impl)


def run_timed(prog_instrs, timing: DeviceTiming, cfg=None, dram_bytes: int = 64 << 20, load=None) -> tuple:
    """Convenience: a CycleMachine run. `load(m)` fills DRAM before the run. -> (status, Timing, machine)."""
    m = CycleMachine(timing=timing, cfg=cfg, dram_bytes=dram_bytes)
    if load is not None:
        load(m)
    st = m.run(prog_instrs)
    return st, m.timing_report, m


def run_multicore(timing: DeviceTiming, cores: list) -> tuple:
    """Phase 6 (I2): run the programs of `cores` (dicts with `instrs`, optional `cfg`, `dram_bytes`, `load`) each
    on its own core (own Machine, queues, FIFOs, scratchpads, arrays, VEC pipe) and replay them together against
    the shared DMA channels. The functional runs are independent and identical to single-core runs. ->
    (MultiTiming, [CycleMachine per core]); each machine's `timing_report` is its core's Timing."""
    machines = []
    for spec in cores:
        m = CycleMachine(timing=timing, defer_timing=True, cfg=spec.get("cfg"), dram_bytes=spec.get("dram_bytes", 64 << 20))
        if spec.get("load") is not None:
            spec["load"](m)
        m.run(spec["instrs"])
        machines.append(m)
    timings = _replay(timing, machines)
    shared = {k: 0 for k in timings[0].channels}
    for t, m in zip(timings, machines):
        m.timing_report = t
        m.status["cycles"] = t.cycles
        for k, v in t.channels.items():
            shared[k] += v
    cycles = max(t.cycles for t in timings)
    mt = MultiTiming(cycles=cycles, us=timing.us(cycles), timing=timing, cores=timings, channels=shared,
                     dram_read_bytes=sum(t.dram_read_bytes for t in timings), dram_write_bytes=sum(t.dram_write_bytes for t in timings),
                     macs=sum(t.macs for t in timings))
    return mt, machines
