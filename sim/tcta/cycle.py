"""Cycle model (phase 5, H1): a timing layer on top of the functional simulator.

    cm = CycleMachine(timing=CONFIG_C, cfg=DeviceConfig(...), dram_bytes=...)
    st = cm.run(prog.instrs)          # the functional run, token checks and faults unchanged
    t = cm.timing_report              # Timing: cycles, per-queue stalls, FIFO levels, channel utilization, Gantt trace

The functional simulator runs first, exactly as `Machine.run` does (same scheduler, same token checks, same
numerics); while it runs, the front end's dispatch order and every executed instruction (with the operand
shapes it saw) are recorded. The timing pass then replays those per-queue streams through a discrete-event
scheduler with the device model of `timing.py`:

- each queue is in order; an instruction starts when it has been dispatched, its queue's previous
  instruction has issued, every POP before it has its token (the producer's PUSH fired at the producer's
  completion), a unit (array, DMA channel, the VEC/SFU pipe) is free, and its scratchpad bank ports are free;
- PUSH is free and takes effect at the producing instruction's completion; a PUSH into a full FIFO blocks
  its queue until the consumer pops (reported per edge);
- the front end dispatches one instruction per cycle; `LOOP r` waits for the VCOUNT that writes R[r].

Rules in full: `docs/phase5-cycle-model-task.md` §4 and `docs/cycle-model-report.md` §1. The model never
changes numerics: `CycleMachine.run` returns the same status and leaves the same DRAM/scratchpad contents as
`Machine.run`; `tests/test_cycle.py` (c) checks that no instruction starts before a token it popped was
pushed, which is the one FAIL condition of the task (§5).
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


@dataclass
class Timing:
    cycles: int
    us: float
    timing: DeviceTiming
    per_queue: dict             # queue -> {n, busy, stall_token, stall_bank, stall_fifo, wait_unit, first, last}; stall_token
                                #   counts cycles the head waited for a token with its unit free, wait_unit the cycles it
                                #   waited for its own queue's unit (back-to-back serialization, not a stall)
    edges: dict                 # "X->Y" -> {pushes, pops, max_level, push_blocked_cycles, depth}
    channels: dict              # "read0" / "write0" -> busy cycles
    banks: dict                 # "ACC[2]" (bank) -> conflict stall cycles charged on it
    recs: list                  # Rec per work instruction, in start order
    dram_read_bytes: int = 0
    dram_write_bytes: int = 0
    macs: int = 0
    fe_cycles: int = 0          # cycle of the last dispatch
    fe_loop_wait: int = 0       # cycles the front end waited on VCOUNT -> LOOP r
    sync_events: list = field(default_factory=list)   # (cycle, queue, "PUSH"|"POP", pc)

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
        self.rec.append(("ins", new))
        return new

    def _control(self, ins):
        ok = super()._control(ins)
        if ok:
            self.rec.append((ins.op, ins.args.get("r"), ins.args.get("n")))
        return ok


# --------------------------------------------------------------------------
# the machine
# --------------------------------------------------------------------------

class CycleMachine(Machine):
    def __init__(self, timing: DeviceTiming | None = None, **kw):
        super().__init__(**kw)
        self.timing = timing or KV260
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
            inf.update(M=M, K=K, N=N)
        elif op in ("LOAD.2D", "LOAD.3D"):
            r = a["region"]
            rows, cols = a["rows_valid"], a["cols_valid"]
            nb = dt.row_bytes_for(r.dtype, cols)
            if r.dtype in ("f4.b32", "f8.b32"):
                nb += cols // 32
            inf.update(rows=rows, bytes=rows * nb, dtype=r.dtype, requests=1)
        elif op == "LOAD.IDX":
            its = self.sp[a["idx"]]
            n = a["nrows"]
            idx = its.data[0, :n]
            live = int((idx != -1).sum())
            inf.update(rows=live, row_bytes=a["row_bytes"], bytes=live * a["row_bytes"], dtype=a["region"].dtype, gather=True)
        elif op == "STORE.2D":
            r = a["region"]
            el = {"f32": 4, "i32": 4, "bf16": 2}.get(r.dtype, 1)
            rows, cols = a["rows_valid"], a["cols_valid"]
            inf.update(rows=rows, bytes=rows * cols * el, dtype=r.dtype, requests=1)
        elif op == "STORE.IDX":
            its = self.sp[a["idx"]]
            r = a["region"]
            n = a["nrows"]
            live = int((its.data[0, :n] != -1).sum())
            el = {"f32": 4, "i32": 4, "bf16": 2}.get(r.dtype, 1)
            rb = dt.pad64(a["cols_valid"] * el)
            inf.update(rows=live, row_bytes=rb, bytes=live * rb, dtype=r.dtype, gather=True)
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

    # ---- the timing pass ----
    def _time(self) -> Timing:
        T = self.timing
        info = {id(ins): inf for q, ins, inf in self._execd}
        streams: dict = {q: [] for q in Q if q != Q.FE}
        fe_items = list(self._fe_rec)
        for item in fe_items:
            if item[0] == "ins":
                streams[item[1].queue].append(item[1])
        queues = [q for q in streams if streams[q]]

        # ---- state ----
        disp: dict = {}                                   # id(ins) -> dispatch cycle
        end_of: dict = {}                                 # id(ins) -> end cycle (work instructions)
        fe_i, fe_time, fe_loop_wait = 0, 0, 0
        last_vcount: dict = {}                            # register -> Instr of the VCOUNT writing it (dispatch order)
        idx = {q: 0 for q in queues}
        ready_after = {q: 0 for q in queues}              # next issue slot of the queue (in-order, PUSH blocking)
        last_end = {q: 0 for q in queues}                 # completion of the queue's previous work instruction
        token_ready = {q: 0 for q in queues}              # tokens popped since the previous work instruction
        stall_fifo = {q: 0 for q in queues}
        push_times: dict = {e: [] for e in EDGES}
        pop_times: dict = {e: [] for e in EDGES}
        blocked: dict = {e: 0 for e in EDGES}
        arrays = [0] * T.n_arrays                         # free-at cycle per array
        rch = [0] * T.read_channels                       # busy-until per read channel
        wch = [0] * T.write_channels
        ch_busy = {f"read{i}": 0 for i in range(T.read_channels)} | {f"write{i}": 0 for i in range(T.write_channels)}
        vec_free = 0
        holds: dict = {}                                  # (sp, bank, dir) -> [release cycles]
        bank_stall: dict = {}
        recs: list = []
        sync_events: list = []
        per_q = {q.value: dict(n=0, busy=0, stall_token=0, stall_bank=0, stall_fifo=0, wait_unit=0, first=None, last=0) for q in queues}
        totals = dict(rb=0, wb=0, macs=0)

        def bank_of(t: TRef):
            return (t.sp, t.slot % T.banks.get(t.sp, 1))

        culprits: list = []

        def port_wait(reads, writes, t):
            """Earliest cycle >= t at which every needed port has a free slot, given the recorded holds (intervals
            [start, end) of earlier-processed instructions). Only holds that contain the candidate cycle count: a
            hold that starts later does not block (the later-starting instruction was processed first because
            its queue was ahead; the model does not go back and delay it, an approximation stated in the
            report). Records the banks that imposed the wait in `culprits`."""
            need = {(bank_of(x), 0) for x in reads} | {(bank_of(x), 1) for x in writes}
            tw = t
            advanced: dict = {}                                   # bank -> the cycle it pushed the wait to
            changed = True
            while changed:
                changed = False
                for (b, d) in need:
                    cap = T.ports_per_bank[d]
                    iv = holds.get((b, d), [])
                    tb = tw
                    while True:
                        active = sorted(e for (s, e) in iv if s <= tb < e)
                        if len(active) < cap:
                            break
                        tb = active[len(active) - cap]            # when enough holders have released
                    if tb > tw:
                        tw = tb
                        advanced[b] = tb
                        changed = True
            culprits[:] = [b for b, v in advanced.items() if v == tw]
            return tw

        def hold(reads, writes, start, end, prune_before):
            """Record the port holds [start, end); drop holds that ended before `prune_before` (no instruction
            processed later can start earlier than that: ready times are processed in nondecreasing order)."""
            for x, d in [(x, 0) for x in reads] + [(x, 1) for x in writes]:
                k = (bank_of(x), d)
                lst = holds.setdefault(k, [])
                lst[:] = [iv for iv in lst if iv[1] > prune_before]
                lst.append((start, end))

        def dma(chs, kind, t_issue, inf, requests, req_bytes, gather):
            """Issue DMA requests at t_issue on the earliest-free channel of `chs`; -> (unit, xfer start, end, busy)."""
            bpc = T.bytes_per_cycle_per_channel
            if inf.get("dtype") in ("f4.b32", "f8.b32", "f8.t", "f4.r16"):
                bpc = min(bpc, T.dequant_bytes_per_cycle)
            u = min(range(len(chs)), key=lambda i: chs[i])
            busy_until = chs[u]
            lat = T.dram_latency_cycles
            if not gather:
                # striped over the k least-busy channels (k = 1 by default: one request, one channel); busy is
                # channel-cycles, so a striped request of the same bytes costs the same busy on k channels
                k = max(1, min(T.request_channels, len(chs)))
                us = sorted(range(len(chs)), key=lambda i: chs[i])[:k]
                busy_until = max(chs[i] for i in us)
                xfer = -(-req_bytes // (bpc * k))
                s = max(busy_until, t_issue + lat)
                e = s + xfer
                for i in us:
                    chs[i] = e
                return us, s, e, xfer * k
            # gather: one request per row / per 64-byte burst, one issue per cycle, max_outstanding in flight.
            # request_channels = 1: every request of the gather queues on one channel (the task's rule);
            # k > 1: the requests go round-robin over the k least-busy channels (address-interleaved rows)
            k = max(1, min(T.request_channels, len(chs)))
            us = sorted(range(len(chs)), key=lambda i: chs[i])[:k]
            until = {i: chs[i] for i in us}
            xfer = max(1, -(-req_bytes // bpc))
            completions: list = []
            first_s, e_last, busy = None, t_issue, 0
            for j in range(requests):
                c = us[j % k]
                t_req = t_issue + j
                if j >= T.max_outstanding_rows:
                    t_req = max(t_req, completions[j - T.max_outstanding_rows])
                s = max(until[c], t_req + lat)
                e = s + xfer
                until[c] = e
                busy += xfer
                completions.append(e)
                e_last = max(e_last, e)
                if first_s is None or s < first_s:
                    first_s = s
            for i in us:
                chs[i] = until[i]
            return us, first_s if first_s is not None else t_issue, e_last, busy

        # ---- readiness of each stream head ----
        def fe_ready():
            if fe_i >= len(fe_items):
                return None
            item = fe_items[fe_i]
            if item[0] == "LOOP" and item[1] is not None:
                vc = last_vcount.get(item[1])
                if vc is not None:
                    if id(vc) not in end_of:
                        return None
                    return max(fe_time, end_of[id(vc)] + T.vcount_to_loop_latency)
            return fe_time

        def head_ready(q):
            i = idx[q]
            if i >= len(streams[q]):
                return None
            ins = streams[q][i]
            if id(ins) not in disp:
                return None
            if ins.op == "PUSH":
                e = (q, ins.args["to"])
                k = len(push_times[e])
                d = T.fifo_depth_of(e)
                room = 0
                if k >= d:
                    if len(pop_times[e]) <= k - d:
                        return None
                    room = pop_times[e][k - d]
                return max(disp[id(ins)], last_end[q], room)
            if ins.op == "POP":
                e = (ins.args["from"], q)
                k = len(pop_times[e])
                if len(push_times[e]) <= k:
                    return None
                return max(push_times[e][k], disp[id(ins)], ready_after[q])
            t = max(disp[id(ins)], ready_after[q], token_ready[q])
            if q == Q.MMA:
                t = max(t, min(arrays))
            elif q == Q.VEC:
                t = max(t, vec_free)
            elif q == Q.LOAD:
                pass                                           # channel queuing is modelled inside dma()
            return t

        # ---- event loop: always the earliest ready head (ties: FE, then queue order) ----
        order = [Q.LOAD, Q.MMA, Q.VEC, Q.STORE, Q.COMM]
        while True:
            cands = []
            r = fe_ready()
            if r is not None:
                cands.append((r, -1, None))
            for n, q in enumerate(order):
                if q in idx:
                    r = head_ready(q)
                    if r is not None:
                        cands.append((r, n, q))
            if not cands:
                left = {q.value: len(streams[q]) - idx[q] for q in queues if idx[q] < len(streams[q])}
                if fe_i < len(fe_items) or left:
                    raise CycleDeadlock(self._why_blocked(streams, idx, disp, push_times, pop_times, fe_i, fe_items), left)
                break
            t, _, q = min(cands)
            if q is None:
                # front end dispatch
                item = fe_items[fe_i]
                fe_i += 1
                if item[0] == "ins":
                    ins = item[1]
                    disp[id(ins)] = fe_time
                    if ins.op not in WORK:
                        fe_time += 1
                        if ins.op == "VCOUNT":
                            last_vcount[ins.args["r"]] = ins
                elif item[0] in ("LOOP", "ENDLOOP"):
                    fe_loop_wait += t - fe_time
                    fe_time = t + T.loop_overhead_cycles
                else:
                    fe_time = t + 1
                continue
            ins = streams[q][idx[q]]
            idx[q] += 1
            a = ins.args
            if ins.op == "PUSH":
                # fires at the producer's completion; free (task §1): the queue goes on issuing while the
                # producer is in flight (a scoreboard fires the token). Only a full FIFO holds the queue, until
                # the token is accepted.
                e = (q, a["to"])
                room_wait = t - max(disp[id(ins)], last_end[q])
                blocked[e] += room_wait
                stall_fifo[q] += room_wait
                push_times[e].append(t)
                if room_wait:
                    ready_after[q] = max(ready_after[q], t)
                sync_events.append((t, q, "PUSH", ins.pc))
                continue
            if ins.op == "POP":
                e = (a["from"], q)
                pop_times[e].append(t)
                token_ready[q] = max(token_ready[q], t)
                sync_events.append((t, q, "POP", ins.pc))
                continue
            # ---- a work instruction ----
            # t0: dispatched and the queue's previous instruction issued (and its PUSHes accepted);
            # t_u: the queue's own unit is free (the previous MMA's array occupancy, the VEC pipe);
            # t1: every popped token has arrived; then the scratchpad ports.
            inf = info.get(id(ins), {}) or {}
            t0 = max(disp[id(ins)], ready_after[q])
            reads, writes = self._access(ins)
            unit = 0
            s_bank = 0
            if q == Q.MMA:
                unit = min(range(len(arrays)), key=lambda i: arrays[i])
                t_u = max(t0, arrays[unit])
            elif q == Q.VEC:
                t_u = max(t0, vec_free)
            else:
                t_u = t0
            t1 = max(t_u, token_ready[q])
            s_tok, s_unit = t1 - t_u, t_u - t0
            if q == Q.MMA:
                t3 = port_wait(reads, writes, t1); s_bank = t3 - t1
                M, K, N = inf["M"], inf["K"], inf["N"]
                occ = T.mma_occupancy(M, K, N)
                start, end, busy = t3, t3 + occ + T.mma_latency, occ
                arrays[unit] = t3 + occ
                totals["macs"] += M * K * N
                # ports are held while the array streams (occupancy), not through the pipeline latency: a
                # back-to-back accumulate into the same ACC forwards (task §4); consumers wait via tokens
                hold(reads, writes, start, start + occ, t1)
                rec = Rec(ins.pc, q, ins.op, disp[id(ins)], start, end, busy, s_tok, s_unit, s_bank, unit, macs=M * K * N, info=inf)
            elif q == Q.VEC:
                t3 = port_wait(reads, writes, t1); s_bank = t3 - t1
                c = self._vec_cycles(ins.op, inf, a)
                start, end, busy = t3, t3 + c, c
                vec_free = end
                hold(reads, writes, start, end, t1)
                rec = Rec(ins.pc, q, ins.op, disp[id(ins)], start, end, busy, s_tok, s_unit, s_bank, 0, info=inf)
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
                # the channel is the unit: the instruction issues at t1 (the DMA engine queues requests behind
                # the channel's current transfer); the ports are held from the first transfer to the end
                # a gather reads its index vector while it issues requests (one per cycle), the data tile's
                # ports are held for the transfer
                idx_reads = [a["idx"]] if gather and isinstance(a.get("idx"), TRef) else []
                data_reads = [x for x in reads if x not in idx_reads]
                if requests == 0:
                    start = end = t1; busy = 0; unit = 0; units = []
                else:
                    units, s_x, end, busy = dma(chs, q, t1, inf, requests, req_bytes, gather)
                    unit = units[0]
                    start = t1
                    t3 = port_wait(data_reads, writes, s_x); s_bank = t3 - s_x
                    if s_bank:
                        end += s_bank
                        for i in units:
                            chs[i] += s_bank
                    hold(data_reads, writes, s_x + s_bank, end, t1)
                    if idx_reads:
                        hold(idx_reads, [], t1, t1 + requests, t1)
                for i in units:
                    ch_busy[("read" if q == Q.LOAD else "write") + str(i)] += busy // len(units)
                if q == Q.LOAD:
                    totals["rb"] += inf.get("bytes", 0)
                else:
                    totals["wb"] += inf.get("bytes", 0)
                rec = Rec(ins.pc, q, ins.op, disp[id(ins)], start, end, busy, s_tok, 0, s_bank, unit,
                          bytes=inf.get("bytes", 0), requests=requests, info=inf)
            else:  # COMM
                c = -(-inf.get("bytes", 0) // T.link_bytes_per_cycle) + T.link_latency_cycles
                start, end, busy = t1, t1 + c, c
                rec = Rec(ins.pc, q, ins.op, disp[id(ins)], start, end, busy, s_tok, 0, 0, 0, bytes=inf.get("bytes", 0), info=inf)
            end_of[id(ins)] = end
            ready_after[q] = max(ready_after[q], start + 1)
            last_end[q] = end
            token_ready[q] = 0
            recs.append(rec)
            pq = per_q[q.value]
            pq["n"] += 1; pq["busy"] += busy; pq["stall_token"] += s_tok; pq["wait_unit"] += s_unit; pq["stall_bank"] += s_bank
            pq["first"] = start if pq["first"] is None else min(pq["first"], start)
            pq["last"] = max(pq["last"], end)
            if s_bank:
                for b in culprits:
                    bank_stall[f"{b[0].value}/bank{b[1]}"] = bank_stall.get(f"{b[0].value}/bank{b[1]}", 0) + s_bank

        # ---- totals ----
        cycles = max([r.end for r in recs] + [fe_time])
        for q in queues:
            per_q[q.value]["stall_fifo"] = stall_fifo[q]
        edges = {}
        for e in EDGES:
            if not push_times[e] and not pop_times[e]:
                continue
            ev = sorted([(t, 1) for t in push_times[e]] + [(t, -1) for t in pop_times[e]], key=lambda x: (x[0], x[1]))
            lvl = mx = 0
            for _, d in ev:
                lvl += d
                mx = max(mx, lvl)
            edges[f"{e[0].value}->{e[1].value}"] = dict(pushes=len(push_times[e]), pops=len(pop_times[e]), max_level=mx,
                                                       push_blocked_cycles=blocked[e], depth=T.fifo_depth_of(e))
        recs.sort(key=lambda r: (r.start, r.queue.value, r.pc))
        return Timing(cycles=cycles, us=T.us(cycles), timing=T, per_queue=per_q, edges=edges, channels=ch_busy,
                      banks=bank_stall, recs=recs, dram_read_bytes=totals["rb"], dram_write_bytes=totals["wb"],
                      macs=totals["macs"], fe_cycles=fe_time, fe_loop_wait=fe_loop_wait, sync_events=sync_events)


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
