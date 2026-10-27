"""D4/E4/F2: map a cut PrimFunc (D3/E3 output) onto a tcta.Program.

    program, report = tile_tir_to_program(cut_func, bx=0, by=0, dram={"Q": 0, "K": ..., "V": ..., "Output": ...})
    program, report = tile_tir_to_program(cut_moe, bx=e, dram=moe_data.DRAM)
    program, report = tile_tir_to_program(cut_persistent, bx=0, dram=moe_data.DRAM)

A pattern pass for fixed schedules: the FA kernel (fa_kernel.py) and the MoE expert,
persistent, combine and router kernels (moe_kernel.py), after cut.lower_to_tile_tir.
Three phases:

Phase A  statement -> (queue, instruction). Walks the body in program order,
         flattening sblocks, into a tree of straight-line ops and hardware loops.
         A static `for` becomes a LOOP n when it is at the top level or encloses a
         dynamic loop (FA's pipelined KV loop, the combine's N-tile loop, the
         persistent kernel's expert loop); a static loop nested in a loop with no
         dynamic loop inside is unrolled (the pipelined weight streams, whose ring
         indices D4 resolves statically). A dynamic `for` whose extent is the VCOUNT
         idiom (mask = idx >= 0; n = reduce_sum(mask); for t in range(ceildiv(n, 64)))
         becomes VCOUNT r + LOOP r, step. Loop-variant operands (DRAM base, scale-grid
         base, ring slot, column) are fitted as affine functions of every enclosing
         hardware loop's variable and emitted as per-level `incr` ("region.base" for
         the innermost loop, "region.base@1" for the next one out; ISA v0.1.2 §5.7).
         MoE collapses: the 64x128 gather loop nest -> LOAD.IDX; the guarded scatter
         nest -> STORE.IDX; payload copy + scales copy + dequant loop -> one LOAD.2D
         f4.b32 into the fp16 ring slot; ACC x ACC multiply/add -> VGATE; T.sigmoid ->
         VSIGMOID; VEC row copy -> VSEL (the v0.1 copy idiom); call_extern tcta.vtopk
         -> VTOPK.
Phase B  tokens, one scheme for every edge over the whole program. The program is
         expanded with every loop body twice (first iteration, steady-state
         iteration) at every nesting level; a per-tile reader/writer analysis over
         that expansion gives each consumer site the push it needs (RAW/WAR/WAW,
         fill_wait(n) -> the (c-n)-th fill's push). Pushes sit after the producing
         instruction: explicit ones from the tcta.fill_issued events and from the
         bank-release convention (after every MMA that reads a WGT slot), plus
         whatever a dependency needs. Tokens are a FIFO count, so pops are
         cumulative per edge: a site pops until the needed push's token has been
         consumed (ISA §8.2 rule 3). Loops are made level-invariant: a body's pops are
         computed for the steady state with a fixed FIFO level L at every iteration
         boundary (the largest L that leaves nothing to pop at body end), and the
         entry level is brought to L before LOOP by popping the surplus or by priming
         pushes; a first-iteration dependency on a push before the loop adds a pop
         (and a prime) before LOOP. A nested loop is a block of its parent's body
         with the same contract: it enters at its own level, leaves at that level,
         and its first-iteration needs are hoisted in front of it; a dependency
         that crosses a block (a site after it needing a push before it) is popped
         before the block, so a zero-trip loop is transparent. The FIFO level is then
         the same at loop entry, at every iteration boundary and at loop exit, so
         LOOP r with any count, including 0, is consistent; primes are drained
         before FINISH. Nothing is popped at the end of an iteration unless the body
         pushes tokens no site consumes, which is what lets the LOAD queue run into
         the next iteration and the next expert while the MMA queue is still
         working (ISA §15). The static FIFO level per edge is reported.
Phase C  emission: items in order, LOOP n|r,step / ENDLOOP (nested), primes and
         hoisted pops before each LOOP, balance pops at body end, drains, FINISH.

Scratchpad slots are allocated in order of first appearance per scope; buffers
that share a data Var (TileLang's ring re-declarations) share slots; an acc->acc
copy aliases its destination to its source (in-place VCVT), with a slot offset
for sliced copies (Hc[n] = H[2+n]). VEC: int32 or [1, n] buffers are row tiles
(slots >= vec_row_slot_start), f32 [n] buffers are column vectors. Count masks,
count scalars and f4 staging buffers get no slot.

TileLang's float16 operands are mapped to the ISA's bf16 format (same byte size
and strides; the simulator rounds to bf16 on load). int32 outputs use the `i32`
store qualifier (ISA v0.1.1 §4).

Phase 4 (mla_kernel.py, kda_kernel.py, gemm_kernel.py) added:
- wide accumulators: a 2-D ACC fragment wider than T_N takes cols // T_N consecutive
  slots; gemms into its column slices address the slot, fill / VSCALE_ACC / STORE.2D
  take the `wide` form (ISA §6.5);
- `tcta.state` scope (STATE scratchpad, loaded by LOAD.2D, read as a B operand);
- DRAM buffers as token-tracked objects, so a tile stored to a scratch region and
  loaded back (a computed B operand) gets STORE->LOAD / LOAD->STORE tokens;
- an acc->acc bf16 copy is in place only when the source is not read afterwards
  (hi/lo splits keep the f32 source: non-inplace VCVT into its own slot); acc->inp
  copies are VCVT into the INP tile; f32 acc->acc copies are VFILL 0 + VGATE add;
- elementwise patterns: `acc *= exp2(acc2)` -> VGATE decay, `acc (*|+)= vec2d` ->
  VGATE with a 2-D VEC source, `x = -x` -> VMUL -1 (the simulator's VNEG needs an
  operand it has none of), `acc -= vec[i]` -> VSUB, `dst = a - b` over ACC tiles ->
  copy, negate b, add, negate back; 1-D `v = exp2(v)` -> VEXP2; gathers with a Cast;
- slot sharing: when a scratchpad is over budget, buffers are coloured over their
  textual live ranges (a value read before it is written inside a loop body is live
  over the whole loop); `report.shared_slots` counts the slots given to more than
  one buffer. Programs that fit keep the one-slot-per-buffer plan unchanged.

Phase 5 §0 (ISA v0.1.3, the Gate 1 freeze candidate):
- acc->wgt / acc->inp copies are `VCVT dst=WGT|INP` (the device-computed-operand path:
  no DRAM round trip); acc->state copies are `VCVT dst=STATE, row0` f32 moves of a
  64-row half into the 128-row state tile; a kept-source bf16 acc->acc cast carries
  `to_bf16` (a plain VCVT into ACC is an f32 move now);
- `VCOPY` replaces the VSEL / VFILL+VADD / VFILL+VGATE copy idioms; `VNEG` replaces
  `VMUL -1`;
- gemm (clear) followed by `acc *= mask2d` fuses into `MMA.MASKED mask_mode=mul`
  when the gemm is untransposed; MMA.T has no masked form, so the Gram products of
  the delta rule keep `VGATE mul`;
- the bank-release push (MMA->LOAD after an MMA that reads WGT) is only emitted for
  WGT slots the LOAD queue wrote; slots filled by VCVT are released by the WAR token
  the pass derives (MMA->VEC).
"""

from __future__ import annotations

import itertools
import os
import sys
from dataclasses import dataclass, field
from enum import Enum

import tilelang  # noqa: F401  (puts the bundled tvm on sys.path)
from tilelang import tvm
from tvm import tirx
from tvm.tirx.stmt_functor import post_order_visit, substitute

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "sim"))
from tcta import Program, Q, Region  # noqa: E402
from tcta.isa import EDGES, SP, DeviceConfig, TRef  # noqa: E402
from tcta.dtypes import pad64  # noqa: E402

from cut import EVENT_FILL_ISSUED, EVENT_FILL_WAIT  # noqa: E402

NEG_INF_IMM = -1e30
INT32_MAX = (1 << 31) - 1
ISA_DTYPE = {"float16": "bf16", "bfloat16": "bf16", "float32": "f32", "int32": "i32"}
STORE_DTYPE = {"float16": "bf16", "bfloat16": "bf16", "float32": "f32", "int32": "i32"}
ELEM_BYTES = {"float16": 2, "bfloat16": 2, "float32": 4, "int32": 4, "uint8": 1}
SP_OF_SCOPE = {"tcta.inp": SP.INP, "tcta.wgt": SP.WGT, "tcta.acc": SP.ACC, "tcta.vec": SP.VEC, "tcta.state": SP.STATE}


class _Pseudo(str, Enum):
    """Not a scratchpad: a DRAM buffer as a token-tracked object, so that a tile stored to a scratch
    region and loaded back (a computed B operand, the KDA state mirror) gets STORE->LOAD / LOAD->STORE
    tokens like any other reader/writer pair (ISA v0.1.2 §8.1)."""
    DRAM = "DRAM"
_CFG = DeviceConfig()                 # default device (ISA §11, v1 KV260); tile_tir_to_program(cfg=) overrides
VEC_ROW_SLOT_START = _CFG.vec_row_slot_start
FIFO_DEPTH = _CFG.fifo_depth
EXTERN_VTOPK = "tcta.vtopk"


# --------------------------------------------------------------------------
# Phase A records
# --------------------------------------------------------------------------

@dataclass
class Op:
    name: str                       # Program method, or "fill_issued" / "fill_wait"
    queue: Q | None
    args: dict = field(default_factory=dict)
    incr: dict = field(default_factory=dict)
    reads: list = field(default_factory=list)    # TRefs
    writes: list = field(default_factory=list)   # TRefs
    meta: dict = field(default_factory=dict)     # pattern bookkeeping

    def is_event(self):
        return self.queue is None


@dataclass
class Loop:
    extent: int | None              # None: dynamic, count in register `reg`
    body: list                      # Ops and nested Loops
    reg: int | None = None
    step: int = 1                   # LOOP r, step: ceil(R[r] / step) iterations
    id: int = 0                     # pre-order index among the program's loops


@dataclass
class MappingReport:
    slots: dict
    loop_extent: int | str | list   # int, "R<n>" for a dynamic loop, or the list for several loops
    fill_wait_pops: list            # (segment, wait n, c, p, fill pops) per fill_wait site
    drained: int                    # pops added before FINISH
    edge_counts: dict               # "X->Y" -> (pushes, pops); a static body counted extent times, a dynamic one once
    n_instrs: int
    body_end_pops: int = 0          # pops at the end of a loop body (pushes no body site consumes)
    hoisted_pops: int = 0           # pops moved before a LOOP (dependency on an earlier write, or surplus at entry)
    primed_pushes: int = 0          # pushes added before a LOOP (dependency on the previous iteration)
    unrolled_loops: int = 0         # static loops unrolled (nested, with no dynamic loop inside)
    shared_slots: int = 0           # slots given to more than one buffer (textual live ranges, over-budget scratchpads only)
    max_fifo_level: dict = field(default_factory=dict)   # "X->Y" -> static maximum FIFO occupancy
    loops: list = field(default_factory=list)            # (id, "R<n>"|extent, step), pre-order
    loop_pcs: dict = field(default_factory=dict)         # loop id -> (pc of LOOP, pc of ENDLOOP)
    loop_tree: list = field(default_factory=list)        # [(id, "R<n>"|extent, step, children), ...]
    collapsed: dict = field(default_factory=dict)        # pattern -> count (gather, scatter, dequant, vcount, ...)


def _call_name(call):
    return call.op.name if hasattr(call.op, "name") else str(call.op)


def _f(e):
    """Evaluate a constant float expression (FloatImm, IntImm, products, tl.infinity)."""
    if isinstance(e, (tirx.FloatImm, tirx.IntImm)):
        return float(e.value)
    if isinstance(e, tirx.Call) and _call_name(e).endswith("infinity"):
        return float("inf")
    if isinstance(e, tirx.Mul):
        return _f(e.a) * _f(e.b)
    if isinstance(e, tirx.Cast):
        return _f(e.value)
    raise ValueError(f"not a constant: {e}")


def _loads(expr):
    out = []
    post_order_visit(expr, lambda n: out.append(n) if isinstance(n, tirx.BufferLoad) else None)
    return out


def _is_zero(e):
    if isinstance(e, tirx.Cast):
        e = e.value
    return isinstance(e, (tirx.FloatImm, tirx.IntImm)) and float(e.value) == 0.0


def _has_dynamic_for(stmt):
    found = []
    post_order_visit(stmt, lambda n: found.append(n) if isinstance(n, tirx.For) and n.kind != tirx.ForKind.PARALLEL
                     and not isinstance(n.extent, tirx.IntImm) else None)
    return bool(found)


# --------------------------------------------------------------------------
# Phase A: statements -> ops
# --------------------------------------------------------------------------

class _Mapper:
    def __init__(self, func: tirx.PrimFunc, bx: int, by: int | None, dram: dict, cfg: DeviceConfig | None = None):
        self.func = func
        self.dram = dram
        self.cfg = cfg or _CFG
        self.ana = tvm.arith.Analyzer()
        self.vmap = {}
        scalars = [p for p in func.params if p not in func.buffer_map]
        if len(scalars) == 1 and scalars[0].name != "bx":
            scalars[0] = None   # a macro renamed the block index (e.g. `e`): the single scalar param is bx
            self.vmap[[p for p in func.params if p not in func.buffer_map][0]] = tirx.IntImm("int32", bx)
        for p in func.params:
            if p.name == "bx":
                self.vmap[p] = tirx.IntImm("int32", bx)
            elif p.name == "by":
                assert by is not None, "cut PrimFunc has a `by` param"
                self.vmap[p] = tirx.IntImm("int32", by)
        assert self.vmap, "cut PrimFunc must have a block-index param"
        self.loop_stack: list[tuple] = []    # (loop var, extent or None for dynamic), outermost first
        self.loop_ids: list[int] = []
        self.n_loops = 0
        self.slots: dict[str, TRef] = {}
        self.ring: dict[str, int] = {}
        self.wide: dict[str, int] = {}       # 2-D ACC buffer wider than T_N -> consecutive slots (ACC.W, ISA §6.5)
        self.vec_temp: TRef | None = None
        self.dequant_dst: set[str] = set()   # fp16 WGT ring written by a dequant loop (the LOAD.2D f4.b32 destination)
        self.payload: set[str] = set()       # uint8 staging buffers: no slot, folded into the load
        self.scales: set[str] = set()
        self.masks: dict[str, str] = {}      # VCOUNT idiom: mask buffer -> index row buffer
        self.count_regs: dict[str, int] = {} # VCOUNT idiom: count buffer -> register
        self.pending_f4: Op | None = None
        self.collapsed: dict[str, int] = {}
        self.unrolled = 0
        self._prescan()
        self._scan_refs()
        self._allocate_slots()

    # ---- ordered buffer references ----
    def _scan_refs(self):
        """Program-order list of (buffer name, is_write) over the whole body, and the position of every
        tl.tileop.copy. A tile op's region operands are classified by argument position; a BufferStore
        is a write after its value's reads. Static loops are not unrolled here: one textual pass."""
        refs: list[tuple] = []              # (buffer name, is_write, statement position)
        copy_pos: list[int] = []
        spans: list[tuple] = []             # (first, last) statement positions of every serial loop body
        pos = [0]

        def rbuf(rc):
            return rc.args[0].buffer.name

        class _Refs(list):
            def append(self, item):
                super().append((item[0], item[1], pos[0]))

        refs = _Refs()

        def loads(e):
            def f(n):
                if isinstance(n, tirx.BufferLoad):
                    refs.append((n.buffer.name, False))
            post_order_visit(e, f)

        def call(n):
            pos[0] += 1
            name = _call_name(n)
            if name == "tl.tileop.copy":
                copy_pos.append(pos[0])
                refs.append((rbuf(n.args[0]), False)); refs.append((rbuf(n.args[1]), True))
            elif name == "tl.tileop.gemm":
                refs.append((rbuf(n.args[0]), False)); refs.append((rbuf(n.args[1]), False))
                clear = len(n.args) > 9 and isinstance(n.args[9], tirx.IntImm) and bool(int(n.args[9]))
                if not clear:
                    refs.append((rbuf(n.args[2]), False))
                refs.append((rbuf(n.args[2]), True))
            elif name == "tl.tileop.fill":
                refs.append((rbuf(n.args[0]), True))
            elif name == "tl.tileop.reduce":
                refs.append((rbuf(n.args[0]), False)); refs.append((rbuf(n.args[1]), True))
            elif name == "tirx.call_extern" and n.args[0].value == EXTERN_VTOPK:
                refs.append((rbuf(n.args[2]), False)); refs.append((rbuf(n.args[1]), True))
            else:
                loads(n)

        def walk(st):
            if isinstance(st, tirx.SBlockRealize):
                walk(st.block.body)
            elif isinstance(st, tirx.SBlock):
                walk(st.body)
            elif isinstance(st, tirx.SeqStmt):
                for x in st.seq:
                    walk(x)
            elif isinstance(st, tirx.Evaluate):
                call(st.value) if isinstance(st.value, tirx.Call) else loads(st.value)
            elif isinstance(st, tirx.For):
                p0 = pos[0] + 1
                walk(st.body)
                if st.kind != tirx.ForKind.PARALLEL:
                    spans.append((p0, pos[0]))
            elif isinstance(st, tirx.IfThenElse):
                loads(st.condition)
                walk(st.then_case)
                if st.else_case is not None:
                    walk(st.else_case)
            elif isinstance(st, tirx.BufferStore):
                pos[0] += 1
                for ix in st.indices:
                    loads(ix)
                loads(st.value)
                refs.append((st.buffer.name, True))
            elif isinstance(st, tirx.Bind):
                loads(st.value)
            elif isinstance(st, (tirx.AttrStmt, tirx.AssertStmt, tirx.DeclBuffer)):
                walk(st.body)
            else:
                raise NotImplementedError(f"_scan_refs: {type(st).__name__}")

        walk(self.func.body)
        self.refs, self.copy_pos, self.spans, self.n_pos = list(refs), copy_pos, spans, pos[0] + 1

    def _read_after(self, name, pos) -> bool:
        return any(n == name and not w and p > pos for n, w, p in self.refs)

    def _live_range(self, names):
        """[first, last] reference positions of a slot group (in-place casts count for their source).
        A group whose first reference inside a loop body is a read carries a value across iterations:
        it is live over the whole loop; read before any write at all, over the whole program."""
        names = set(names) | {d for d, (s, _) in getattr(self, "alias", {}).items() if s in names}
        mine = [(p, w) for n, w, p in self.refs if n in names]
        if not mine:
            return (0, -1)
        if not mine[0][1]:
            return (0, self.n_pos)
        # positions are per statement: a value read by the statement that writes another buffer is live
        # at that statement, so the two never share a slot (dst = a - b fills dst before reading a)
        lo, hi = mine[0][0], mine[-1][0]
        for a, b in self.spans:
            inside = [(p, w) for p, w in mine if a <= p <= b]
            if inside and not inside[0][1]:
                lo, hi = min(lo, a), max(hi, b)
        return (lo, hi)

    # ---- prescan: patterns whose buffers get no slot ----
    def _prescan(self):
        pars, reduces = [], []

        def visit(n):
            if isinstance(n, tirx.For) and n.kind == tirx.ForKind.PARALLEL:
                pars.append(n)
            if isinstance(n, tirx.Call) and _call_name(n) == "tl.tileop.reduce":
                reduces.append(n)

        post_order_visit(self.func.body, visit)
        for loop in pars:
            ivs, store, _, _ = self._parallel_parts(loop)
            if store is None:
                continue
            loads = _loads(store.value)
            dst = store.buffer
            if dst.scope() == "tcta.wgt" and dst.dtype == "float16":
                u8 = [l.buffer for l in loads if l.buffer.dtype == "uint8" and l.buffer.scope() == "tcta.wgt"]
                if u8:
                    ncol = int(dst.shape[-1])
                    for b in u8:
                        (self.payload if int(b.shape[-1]) == ncol // 2 else self.scales).add(b.name)
                    self.dequant_dst.add(dst.name)
            if (len(ivs) == 1 and dst.scope() == "tcta.vec" and dst.dtype == "int32"
                    and isinstance(store.value, tirx.Call) and _call_name(store.value).endswith("if_then_else")
                    and len(loads) == 1 and loads[0].buffer.dtype == "int32" and loads[0].buffer.scope() == "tcta.vec"):
                self.masks[dst.name] = loads[0].buffer.name
        # reduces appear in program order (post-order visit of a SeqStmt keeps it): registers by first use
        for r in reduces:
            src = r.args[0].args[0].buffer
            if src.name in self.masks and r.args[2].value == "sum":
                n = r.args[1].args[0].buffer.name
                if n not in self.count_regs:
                    self.count_regs[n] = len(self.count_regs)

    # ---- slots ----
    def _allocate_slots(self):
        allocs, copies = [], []

        def visit(n):
            if isinstance(n, tirx.SBlock):
                allocs.extend(n.alloc_buffers)
            if isinstance(n, tirx.Call) and _call_name(n) == "tl.tileop.copy":
                copies.append(n)

        post_order_visit(self.func.body, visit)
        # buffers sharing a data Var (ring re-declarations) share slots
        groups: list[list[tirx.Buffer]] = []
        for b in allocs:
            for g in groups:
                if g[0].data.same_as(b.data):
                    g.append(b)
                    break
            else:
                groups.append([b])
        # acc -> acc copies into a bf16 fragment are in-place converts (VCVT inplace, the P-ready form)
        # when the source is not read again: alias dst to src (+ slice offset). A source that is read
        # later (x_hi = bf16(x); x_lo = x - x_hi) keeps its f32 value: the cast gets its own slot.
        alias = {}
        for k, c in enumerate(copies):          # same post order as _scan_refs
            s, d = c.args[0].args[0], c.args[1].args[0]
            if (s.buffer.scope() == "tcta.acc" and d.buffer.scope() == "tcta.acc" and d.buffer.dtype != "float32"
                    and not self._read_after(s.buffer.name, self.copy_pos[k])):
                off = 0
                if len(s.indices) == 3:
                    off = int(self.ana.simplify(s.indices[0] - d.indices[0]))
                alias[d.buffer.name] = (s.buffer.name, off)
        self.alias = alias
        no_slot = set(self.scales) | set(self.payload) | set(self.masks) | set(self.count_regs)
        plan = []                                  # (group names, sp, key, n slots)
        for g in groups:
            names = [b.name for b in g]
            if any(n in alias for n in names) or any(n in no_slot for n in names):
                continue
            b = g[0]
            sp = SP_OF_SCOPE[b.scope()]
            n = max(int(x.shape[0]) if len(x.shape) == 3 else 1 for x in g)
            if sp == SP.ACC and len(b.shape) == 2 and int(b.shape[1]) > self.cfg.T_N:
                # wide accumulator: one fragment over cols//T_N consecutive slots
                assert int(b.shape[1]) % self.cfg.T_N == 0, b.name
                n = int(b.shape[1]) // self.cfg.T_N
                for name in names:
                    self.wide[name] = n
            key = "vec_row" if sp == SP.VEC and self._is_row(b) else sp
            plan.append((names, sp, key, n))
        limit = {key: (self.cfg.slots[SP.VEC] if key == "vec_row" else self.cfg.slots[key]) for key in
                 list(SP) + ["vec_row"]}
        start = {key: (self.cfg.vec_row_slot_start if key == "vec_row" else 0) for key in limit}
        counters = dict(start)
        for names, sp, key, n in plan:
            counters[key] += n
        self.shared_slots = 0
        for key in limit:
            mine = [(names, sp, n) for names, sp, k, n in plan if k == key]
            if counters[key] <= limit[key]:
                # one slot (group) per buffer, in allocation order
                slot = start[key]
                for names, sp, n in mine:
                    for name in names:
                        self.slots[name], self.ring[name] = TRef(sp, slot), n
                    slot += n
                continue
            # over budget: interval colouring over the textual live ranges (first-fit over consecutive slots)
            free_from = {i: 0 for i in range(start[key], limit[key])}
            for names, sp, n in sorted(mine, key=lambda e: self._live_range(e[0])):
                lo, hi = self._live_range(names)
                for s0 in range(start[key], limit[key] - n + 1):
                    if all(free_from[s0 + k] <= lo for k in range(n)):
                        break
                else:
                    demand = "; ".join(f"{e[0][0]}x{e[2]}@{self._live_range(e[0])}" for e in mine)
                    raise AssertionError(f"out of {key if isinstance(key, str) else key.value} slots at {names[0]} (after sharing): {demand}")
                for k in range(n):
                    if free_from[s0 + k] > 0:
                        self.shared_slots += 1
                    free_from[s0 + k] = hi + 1
                for name in names:
                    self.slots[name], self.ring[name] = TRef(sp, s0), n
        counters[SP.VEC] = max([self.slots[n].slot + self.ring[n] for n in self.slots if self.slots[n].sp == SP.VEC
                                and self.slots[n].slot < self.cfg.vec_row_slot_start] + [0])
        for d, (s, off) in alias.items():
            self.slots[d] = TRef(self.slots[s].sp, self.slots[s].slot + off)
            self.ring[d] = self.ring[s]
        self.vec_temp = TRef(SP.VEC, counters[SP.VEC])
        assert counters[SP.VEC] < self.cfg.vec_row_slot_start, "out of 2-D VEC slots (temp)"
        self.buffers = {b.name: b for b in allocs}
        for b in self.func.buffer_map.values():
            self.buffers[b.name] = b

    def dram_ref(self, buf) -> TRef:
        """Token-tracked stand-in for a DRAM buffer (see _Pseudo)."""
        names = sorted(p.name for p in self.func.buffer_map)
        return TRef(_Pseudo.DRAM, names.index(next(p.name for p, b in self.func.buffer_map.items() if b.same_as(buf))))

    def gname(self, buf) -> str:
        """Kernel-argument name of a global buffer (the PrimFunc param minus `_handle`): TileLang's
        macros rename buffers to the macro's parameter names, so `buf.name` is not unique."""
        for p, b in self.func.buffer_map.items():
            if b.same_as(buf):
                return p.name[:-7] if p.name.endswith("_handle") else p.name
        raise KeyError(buf.name)

    @staticmethod
    def _is_row(buf):
        shp = [int(x) for x in buf.shape]
        return (len(shp) == 1 and buf.dtype == "int32") or (len(shp) == 2 and shp[0] == 1)

    # ---- expression evaluation over the enclosing hardware loops ----
    def eval_int(self, expr, assign=None):
        vmap = dict(self.vmap)
        for v, k in (assign or {}).items():
            vmap[v] = tirx.IntImm("int32", k)
        e = self.ana.simplify(substitute(expr, vmap))
        if not isinstance(e, tirx.IntImm):
            raise ValueError(f"index did not fold to a constant: {expr} -> {e}")
        return int(e.value)

    def _grid(self):
        """Sample points over the enclosing loops: every iteration of a static loop, two of a dynamic one."""
        if not self.loop_stack:
            return [()]
        axes = [range(ext) if ext is not None else (0, 1) for _, ext in self.loop_stack]
        return list(itertools.product(*axes))

    def _assign(self, key):
        return {v: k for (v, _), k in zip(self.loop_stack, key)}

    def _affine(self, f):
        """f: {loop var: iteration} -> int. -> (value at iteration 0 of every enclosing loop,
        {level: per-iteration step}) with level 0 the innermost loop; asserts affinity."""
        vals = {key: f(self._assign(key)) for key in self._grid()}
        depth = len(self.loop_stack)
        base = vals[(0,) * depth]
        steps = {}
        for pos in range(depth):
            unit = tuple(1 if i == pos else 0 for i in range(depth))
            s = vals[unit] - base if unit in vals else 0
            if s:
                steps[depth - 1 - pos] = s
        for key, v in vals.items():
            want = base + sum(key[pos] * steps.get(depth - 1 - pos, 0) for pos in range(depth))
            assert v == want, f"operand is not affine in the loop variables: {vals}"
        return base, steps

    def _affine_expr(self, expr):
        return self._affine(lambda a: self.eval_int(expr, a))

    @staticmethod
    def _keys(fld, steps, scale=1):
        """Per-level incr entries: `fld` for the innermost loop, `fld@L` for level L (ISA v0.1.2 §5.7)."""
        return {(fld if lvl == 0 else f"{fld}@{lvl}"): s * scale for lvl, s in steps.items()}

    # ---- operands ----
    def tref(self, bl):
        """Scratch tile for a region's start indices -> (TRef, ring {level: n} or {})."""
        name = bl.buffer.name
        base = self.slots[name]
        if name in self.wide:
            try:
                col = {self.eval_int(bl.indices[1], self._assign(key)) for key in self._grid()}
            except ValueError:
                return base, {}                          # elementwise access over the whole fragment
            assert len(col) == 1, f"column slice of the wide accumulator {name} varies with a loop: {col}"
            c = col.pop()
            assert c % self.cfg.T_N == 0, f"wide accumulator slice of {name} not slot-aligned: {c}"
            return TRef(base.sp, base.slot + c // self.cfg.T_N), {}
        if len(bl.indices) == 3:
            vals = {key: self.eval_int(bl.indices[0], self._assign(key)) for key in self._grid()}
            if len(set(vals.values())) == 1:
                return TRef(base.sp, base.slot + next(iter(vals.values()))), {}
            n = self.ring[name]
            depth = len(self.loop_stack)
            varying = [pos for pos in range(depth)
                       if any(vals[k] != vals[k[:pos] + (0,) + k[pos + 1:]] for k in vals)]
            assert len(varying) == 1, f"ring index of {name} varies with several loops: {vals}"
            pos = varying[0]
            assert all(v == key[pos] % n for key, v in vals.items()), f"ring index of {name} is not i % {n}: {vals}"
            return base, {depth - 1 - pos: n}
        return base, {}

    def dram_offsets(self, bl: tirx.BufferLoad):
        """-> (byte address at iteration 0, {level: byte step}, strides, elem bytes)."""
        buf = bl.buffer
        eb = ELEM_BYTES[buf.dtype]
        strides = [int(s) for s in buf.strides]
        dbase = self.dram[self.gname(buf)]

        def off(a):
            return dbase + sum(self.eval_int(idx, a) * st for idx, st in zip(bl.indices, strides)) * eb

        base, steps = self._affine(off)
        return base, steps, strides, eb

    def dram_region(self, bl: tirx.BufferLoad, extents, dtype_map=ISA_DTYPE):
        base, steps, strides, eb = self.dram_offsets(bl)
        rows, cols = int(extents[-2]), int(extents[-1])
        region = Region(base, dtype_map[bl.buffer.dtype], (rows, cols), (strides[-2] * eb, eb))
        return region, rows, cols, steps

    @staticmethod
    def _region_parts(rc):
        assert _call_name(rc) == "tl.region", rc
        return rc.args[0], [int(x) for x in rc.args[2:]]

    def _valid(self, buf):
        shp = [int(x) for x in buf.shape]
        if len(shp) == 1:
            return (1, shp[0]) if self._is_row(buf) else (shp[0], 1)
        if buf.name in self.wide:
            return shp[0], self.cfg.T_N                  # per slot; the wide ops take all slots
        return shp[-2], shp[-1]

    def wide_slots(self, buf, t: TRef) -> list[TRef]:
        """The slots an instruction on the whole of `buf` touches (one unless `buf` is wide)."""
        return [TRef(t.sp, t.slot + k) for k in range(self.wide.get(buf.name, 1))]

    # ---- statement walk ----
    def linearize(self):
        items = []
        self._walk(self.func.body, items)
        self.collapsed["unrolled_static_loops"] = self.unrolled
        return items

    def _emit(self, ops, out):
        for o in ops:
            o.meta.setdefault("loop", self.loop_ids[-1] if self.loop_ids else None)
        out.extend(ops)

    def _walk(self, stmt, out):
        if isinstance(stmt, tirx.SBlockRealize):
            self._walk(stmt.block.body, out)
        elif isinstance(stmt, tirx.SBlock):
            self._walk(stmt.body, out)
        elif isinstance(stmt, tirx.SeqStmt):
            for s in stmt.seq:
                self._walk(s, out)
        elif isinstance(stmt, tirx.Evaluate):
            self._emit(self._call(stmt.value), out)
        elif isinstance(stmt, tirx.For):
            if stmt.kind == tirx.ForKind.PARALLEL:
                self._emit(self._parallel(stmt), out)
            else:
                self._serial(stmt, out)
        elif isinstance(stmt, tirx.IfThenElse):
            # guards on an unrolled loop variable (gather on the first N-tile, tail on the last K-tile)
            cond = self.ana.simplify(substitute(stmt.condition, self.vmap))
            assert isinstance(cond, tirx.IntImm), f"non-constant guard after unrolling: {stmt.condition}"
            branch = stmt.then_case if int(cond.value) else stmt.else_case
            if branch is not None:
                self._walk(branch, out)
        else:
            raise NotImplementedError(f"statement {type(stmt).__name__}")

    def _unroll(self, stmt, out):
        self.unrolled += 1
        for i in range(int(stmt.extent.value)):
            self._walk(substitute(stmt.body, {stmt.loop_var: tirx.IntImm("int32", i)}), out)

    def _serial(self, stmt, out):
        assert int(stmt.min) == 0
        extent = self.ana.simplify(substitute(stmt.extent, self.vmap))
        if isinstance(extent, tirx.IntImm):
            if self.loop_stack and not _has_dynamic_for(stmt.body):
                # a static loop nested in a loop with nothing dynamic inside (the pipelined weight
                # streams, the SwiGLU N-tiles): unroll, its ring indices are resolved statically
                self._unroll(stmt, out)
                return
            # top-level static loop (FA, combine) or a static loop around a dynamic one (the
            # persistent kernel's expert loop): LOOP n with per-level incr on the body's operands
            self._hw_loop(stmt, int(extent.value), None, 1, out)
            return
        # dynamic extent: the VCOUNT idiom's count, optionally ceildiv'd by the tile height
        assert all(ext is not None for _, ext in self.loop_stack), "dynamic loop inside a dynamic loop"
        name, step = self._dynamic_extent(extent)
        self._hw_loop(stmt, None, self.count_regs[name], step, out)

    def _hw_loop(self, stmt, extent, reg, step, out):
        lid = self.n_loops
        self.n_loops += 1
        self.loop_stack.append((stmt.loop_var, extent))
        self.loop_ids.append(lid)
        body = []
        self._walk(stmt.body, body)
        self.loop_stack.pop()
        self.loop_ids.pop()
        out.append(Loop(extent, body, reg=reg, step=step, id=lid))

    def _dynamic_extent(self, e):
        """n[0] -> (n, 1); (n[0] + s-1) // s -> (n, s)."""
        if isinstance(e, tirx.BufferLoad) and e.buffer.name in self.count_regs:
            return e.buffer.name, 1
        if isinstance(e, tirx.FloorDiv) and isinstance(e.b, tirx.IntImm) and isinstance(e.a, tirx.Add):
            n, c = e.a.a, e.a.b
            if isinstance(n, tirx.BufferLoad) and n.buffer.name in self.count_regs and isinstance(c, tirx.IntImm) \
                    and int(c.value) == int(e.b.value) - 1:
                return n.buffer.name, int(e.b.value)
        raise AssertionError(f"dynamic loop extent {e} is not a VCOUNT count or its ceildiv")

    # ---- tile op calls ----
    def _call(self, call):
        name = _call_name(call)
        if name == "tirx.call_extern":
            ext = call.args[0].value
            if ext == EVENT_FILL_ISSUED:
                return [Op("fill_issued", None)]
            if ext == EVENT_FILL_WAIT:
                return [Op("fill_wait", None, {"n": int(call.args[1])})]
            if ext == EXTERN_VTOPK:
                idx = self.slots[call.args[1].args[0].buffer.name]
                src = self.slots[call.args[2].args[0].buffer.name]
                k = int(call.args[3])
                self.collapse("vtopk")
                return [Op("vtopk", Q.VEC, dict(idx=idx, src=src, k=k), reads=[src], writes=[idx])]
            raise NotImplementedError(ext)
        if name == "tl.tileop.copy":
            return self._copy(call)
        if name == "tl.tileop.fill":
            bl, ext = self._region_parts(call.args[0])
            v = _f(call.args[1])
            imm = NEG_INF_IMM if v == float("-inf") else v
            rows, cols = self._valid(bl.buffer)
            t, _ = self.tref(bl)
            n = ext[0] if len(ext) == 3 else self.wide.get(bl.buffer.name, 1)
            return [Op("vfill", Q.VEC, dict(dst=TRef(t.sp, t.slot + i), imm=imm, rows_valid=rows, cols_valid=cols),
                       writes=[TRef(t.sp, t.slot + i)], meta={"fill_of": TRef(t.sp, t.slot + i), "value": v})
                    for i in range(n)]
        if name == "tl.tileop.gemm":
            a, _ = self.tref(self._region_parts(call.args[0])[0])
            b, bmod = self.tref(self._region_parts(call.args[1])[0])
            c, _ = self.tref(self._region_parts(call.args[2])[0])
            trans_a, trans_b = bool(int(call.args[3])), bool(int(call.args[4]))
            clear = bool(int(call.args[9])) if len(call.args) > 9 and isinstance(call.args[9], tirx.IntImm) else False
            assert not trans_a
            incr = self._keys("b_mod", bmod)
            reads = [a, b] + ([] if clear else [c])
            return [Op("mma_t" if trans_b else "mma", Q.MMA, dict(acc=c, a=a, b=b, accumulate=not clear), incr,
                       reads=reads, writes=[c], meta={"gemm": True})]
        if name == "tl.tileop.reduce":
            sbl = self._region_parts(call.args[0])[0]
            kind = call.args[2].value
            if kind == "sum" and sbl.buffer.name in self.masks:
                idx = self.slots[self.masks[sbl.buffer.name]]
                reg = self.count_regs[self._region_parts(call.args[1])[0].buffer.name]
                self.collapse("vcount")
                return [Op("vcount", Q.VEC, dict(r=reg, idx=idx, bound=INT32_MAX), reads=[idx], meta={"vcount": True})]
            src, _ = self.tref(sbl)
            dst, _ = self.tref(self._region_parts(call.args[1])[0])
            clear = bool(int(call.args[4]))
            if kind == "max":
                return [Op("vrowmax", Q.VEC, dict(dst=dst, src=src), reads=[src], writes=[dst], meta={"rowmax": True, "clear": clear})]
            if kind == "sum":
                return [Op("vrowsum", Q.VEC, dict(dst=dst, src=src), reads=[src], writes=[dst], meta={"rowsum": True})]
            raise NotImplementedError(kind)
        raise NotImplementedError(name)

    def collapse(self, key):
        self.collapsed[key] = self.collapsed.get(key, 0) + 1

    def _copy(self, call):
        sbl, sext = self._region_parts(call.args[0])
        dbl, dext = self._region_parts(call.args[1])
        ss, ds = sbl.buffer.scope(), dbl.buffer.scope()
        if ss == "global":
            if dbl.buffer.name in self.payload:
                return self._f4_payload(sbl, sext, dbl)
            if dbl.buffer.name in self.scales:
                return self._f4_scales(sbl)
            t, mod = self.tref(dbl)
            region, rows, cols, steps = self.dram_region(sbl, sext)
            if t.sp == SP.VEC and len(dbl.buffer.shape) == 1 and not self._is_row(dbl.buffer):
                # per-row vector: a contiguous DRAM slice becomes one column of a 2-D VEC tile
                assert rows == 1
                region = Region(region.base, region.dtype, (cols, 1), (region.stride[1], region.stride[1]))
                rows, cols = cols, 1
            incr = self._keys("region.base", steps)
            incr.update(self._keys("dst_mod", mod))
            return [Op("load2d", Q.LOAD, dict(dst=t, region=region, rows_valid=rows, cols_valid=cols), incr,
                       reads=[self.dram_ref(sbl.buffer)], writes=[t])]
        if ds == "global":
            t, _ = self.tref(sbl)
            region, rows, cols, steps = self.dram_region(dbl, dext, STORE_DTYPE)
            incr = self._keys("region.base", steps)
            slots = self.wide_slots(sbl.buffer, t)
            args = dict(src=t, region=region, rows_valid=rows, cols_valid=cols)
            if len(slots) > 1:
                args["wide"] = True                      # STORE.2D wide: the four slots as one [T_M, 4*T_N] tile
            return [Op("store2d", Q.STORE, args, incr, reads=slots, writes=[self.dram_ref(dbl.buffer)])]
        if ss == "tcta.vec" and ds == "tcta.vec":
            # VCOPY (v0.1.3); before it a row tile went through VSEL dst, a, a, mask and a 2-D tile through
            # VFILL 0 + VADD
            s, _ = self.tref(sbl)
            d, _ = self.tref(dbl)
            self.collapse("vcopy")
            return [Op("vcopy", Q.VEC, dict(dst=d, src=s), reads=[s], writes=[d])]
        if ss == "tcta.acc" and ds in ("tcta.acc", "tcta.inp", "tcta.wgt", "tcta.state"):
            s, _ = self.tref(sbl)
            d, _ = self.tref(dbl)
            if dbl.buffer.name in self.alias:
                assert s == d, "aliased acc->acc copy must land on the source slot"
                return [Op("vcvt", Q.VEC, dict(dst=d, src=s, inplace=True), reads=[s], writes=[s])]
            if ds in ("tcta.inp", "tcta.wgt"):
                # the device-computed-operand path (v0.1.3): bf16 move into an A (INP) or B (WGT) operand tile
                self.collapse("vcvt_operand")
                return [Op("vcvt", Q.VEC, dict(dst=d, src=s), reads=[s], writes=[d])]
            if ds == "tcta.state":
                # f32 move of a T_M-row tile into a taller STATE tile at a row offset (v0.1.3 `row0`)
                row0, steps = self._affine_expr(dbl.indices[0])
                assert not steps, f"STATE row offset of {dbl.buffer.name} varies with a hardware loop"
                assert row0 % self.cfg.T_M == 0, f"STATE row offset {row0} is not a multiple of T_M"
                self.collapse("vcvt_state")
                return [Op("vcvt", Q.VEC, dict(dst=d, src=s, row0=row0), reads=[s], writes=[d])]
            if dbl.buffer.dtype != "float32":
                # bf16 copy into another ACC tile, source kept (the hi part of a hi/lo split); `to_bf16`
                # because a plain VCVT into ACC is an f32 move since v0.1.3
                self.collapse("vcvt_copy")
                return [Op("vcvt", Q.VEC, dict(dst=d, src=s, to_bf16=True), reads=[s], writes=[d])]
            self.collapse("vcopy")
            return [Op("vcopy", Q.VEC, dict(dst=d, src=s), reads=[s], writes=[d])]
        raise NotImplementedError(f"copy {ss} -> {ds}")

    def _f4_payload(self, sbl, sext, dbl):
        """uint8 E2M1 payload tile -> LOAD.2D f4.b32. The destination (the fp16 ring slot) is set by the
        dequant loop that consumes the staging buffers; the scales come from the scales copy. Both
        bases carry per-level incr (expert stride on the outer loop)."""
        base, steps, strides, eb = self.dram_offsets(sbl)
        rows, cols = int(sext[-2]), int(sext[-1]) * 2
        region = Region(base, "f4.b32", (rows, cols), (strides[-2] * eb, 0))
        op = Op("load2d", Q.LOAD, dict(dst=None, region=region, rows_valid=rows, cols_valid=cols),
                self._keys("region.base", steps), meta={"f4": True})
        self.collapse("dequant_load")
        if self.pending_f4 is not None and self.pending_f4.queue is None:   # scales copy came first
            sr = self.pending_f4.args["region"]
            op.args["region"] = Region(base, "f4.b32", (rows, cols), (strides[-2] * eb, 0),
                                       scales_base=sr.scales_base, scales_stride=sr.scales_stride)
            op.incr.update(self.pending_f4.incr)
        self.pending_f4 = op
        return [op]

    def _f4_scales(self, sbl):
        base, steps, strides, eb = self.dram_offsets(sbl)
        sstride = strides[-2] * eb
        sincr = self._keys("region.scales_base", steps)
        if self.pending_f4 is not None and self.pending_f4.args["region"].scales_base is None:
            r = self.pending_f4.args["region"]
            self.pending_f4.args["region"] = Region(r.base, r.dtype, r.shape, r.stride, scales_base=base, scales_stride=sstride)
            self.pending_f4.incr.update(sincr)
            return []
        # payload copy comes later: park the scales in a placeholder
        self.pending_f4 = Op("load2d", None, dict(region=Region(0, "f4.b32", (0, 0), (0, 0), scales_base=base, scales_stride=sstride)),
                             sincr)
        return []

    # ---- T.parallel patterns ----
    @staticmethod
    def _parallel_parts(loop):
        """-> (loop vars, final BufferStore, bind substitution dict, guard predicate or None);
        store None if not a plain nest."""
        ivs = [loop.loop_var]
        body = loop.body
        while isinstance(body, tirx.For) and body.kind == tirx.ForKind.PARALLEL:
            ivs.append(body.loop_var)
            body = body.body
        binds = {}
        if isinstance(body, tirx.SeqStmt):
            for s in body.seq[:-1]:
                if not isinstance(s, tirx.Bind):
                    return ivs, None, binds, None
                binds[s.var] = substitute(s.value, binds)
            body = body.seq[-1]
        pred = None
        if isinstance(body, tirx.IfThenElse) and body.else_case is None:
            pred, body = body.condition, body.then_case
        if not isinstance(body, tirx.BufferStore):
            return ivs, None, binds, None
        if binds:
            body = tirx.BufferStore(body.buffer, substitute(body.value, binds), list(body.indices), body.predicate)
        return ivs, body, binds, pred

    @staticmethod
    def _nonneg_index(cond):
        """`0 <= idx[i]` / `idx[i] >= 0` -> the index BufferLoad, else None."""
        if isinstance(cond, tirx.LE) and _is_zero(cond.a):
            return cond.b
        if isinstance(cond, tirx.GE) and _is_zero(cond.b):
            return cond.a
        return None

    def _parallel(self, loop):
        ivs, body, _, pred = self._parallel_parts(loop)
        assert body is not None, loop
        dst = body.buffer
        val = body.value

        def ld(e, vars_):
            """BufferLoad whose trailing indices are exactly vars_ (leading ones constants)."""
            if not (isinstance(e, tirx.BufferLoad) and len(e.indices) >= len(vars_)):
                return False
            lead, tail = e.indices[:len(e.indices) - len(vars_)], e.indices[len(e.indices) - len(vars_):]
            return (all(isinstance(i, tirx.Var) and i.same_as(v) for i, v in zip(tail, vars_))
                    and all(isinstance(self.ana.simplify(substitute(i, self.vmap)), tirx.IntImm) for i in lead))

        def t_of(bl):
            return self.tref(bl)[0]

        def same_tile(a, b):
            return a.buffer.same_as(b.buffer) and t_of(a) == t_of(b)

        def exp2_of_scaled_diff(e):
            """exp2(a*s - b*s) -> (a, b, s) or None."""
            if not (isinstance(e, tirx.Call) and _call_name(e).endswith("exp2")):
                return None
            x = e.args[0]
            if isinstance(x, tirx.Sub) and isinstance(x.a, tirx.Mul) and isinstance(x.b, tirx.Mul):
                sa, sb = _f(x.a.b), _f(x.b.b)
                assert sa == sb, "both terms must carry the same scale"
                return x.a.a, x.b.a, sa
            return None

        # frontend-only count mask of the VCOUNT idiom: consumed by the reduce
        if dst.name in self.masks:
            return []
        # dequant loop: completes the pending LOAD.2D f4.b32 (issued at the payload copy) with its
        # destination, the fp16 ring slot this loop writes; the loop itself is the DMA's work
        if dst.name in self.dequant_dst:
            assert any(l.buffer.name in self.payload for l in _loads(val)), "dequant loop without a payload read"
            op = self.pending_f4
            assert op is not None and op.queue is Q.LOAD and op.args["dst"] is None, "dequant loop without a pending payload load"
            assert op.args["region"].scales_base is not None, "payload load without scales"
            t = t_of(body)
            op.args["dst"] = t
            op.writes = [t]
            self.pending_f4 = None
            self.collapse("dequant_fold")
            return []
        # guarded row scatter: if idx[i] >= 0: Y[idx[i], col0 + j] = acc[i, j]  -> STORE.IDX
        if dst.scope() == "global":
            assert len(ivs) == 2 and pred is not None, f"unguarded global store in T.parallel: {body}"
            i, j = ivs
            idx_ld = self._nonneg_index(pred)
            assert idx_ld is not None and ld(idx_ld, [i]) and len(body.indices) == 2, f"scatter guard {pred}"
            assert isinstance(body.indices[0], tirx.BufferLoad) and body.indices[0].buffer.same_as(idx_ld.buffer)
            assert ld(val, [i, j]) and val.buffer.scope() in ("tcta.acc", "tcta.vec"), f"scatter source {val}"
            eb = ELEM_BYTES[dst.dtype]
            strides = [int(s) for s in dst.strides]
            col0, csteps = self._affine_expr(substitute(body.indices[1], {j: tirx.IntImm("int32", 0)}))
            rows, cols = self._valid(val.buffer)
            region = Region(self.dram[self.gname(dst)] + col0 * eb, STORE_DTYPE[dst.dtype],
                            (int(dst.shape[0]), cols), (strides[0] * eb, eb))
            src, idx_t = t_of(val), t_of(idx_ld)
            incr = self._keys("region.base", csteps, eb)
            self.collapse("scatter")
            return [Op("storeidx", Q.STORE, dict(src=src, region=region, idx=idx_t, nrows=rows, cols_valid=cols), incr,
                       reads=[src, idx_t], writes=[self.dram_ref(dst)])]

        def strip_cast(e):
            return e.value if isinstance(e, tirx.Cast) else e

        def negated(e):
            """-x as TileLang emits it (x * -1, or 0 - x) -> x, else None."""
            if isinstance(e, tirx.Mul) and isinstance(e.b, (tirx.FloatImm, tirx.IntImm)) and float(e.b.value) == -1.0:
                return e.a
            if isinstance(e, tirx.Sub) and _is_zero(e.a):
                return e.b
            return None

        if len(ivs) == 1:
            i = ivs[0]
            d = t_of(body)
            rows, cols = self._valid(dst)
            neg = negated(val)
            if neg is not None and ld(neg, [i]) and neg.buffer.same_as(dst):
                return [Op("vneg", Q.VEC, dict(src=d), reads=[d], writes=[d])]
            if (isinstance(val, tirx.Call) and _call_name(val).endswith("exp2") and ld(val.args[0], [i])
                    and val.args[0].buffer.same_as(dst)):
                return [Op("vexp2", Q.VEC, dict(t=d), reads=[d], writes=[d])]
            if isinstance(val, tirx.Max) and ld(val.a, [i]) and val.a.buffer.same_as(dst) and ld(val.b, [i]):
                w = t_of(val.b)
                return [Op("vmax", Q.VEC, dict(src=d, vec=w), reads=[d, w], writes=[d])]
            m = exp2_of_scaled_diff(val)
            if m and ld(m[0], [i]) and ld(m[1], [i]):
                a, b, s = t_of(m[0]), t_of(m[1]), m[2]
                return [Op("vfill", Q.VEC, dict(dst=d, imm=0.0, rows_valid=rows, cols_valid=cols), writes=[d]),
                        Op("vadd", Q.VEC, dict(src=d, vec=a), reads=[d, a], writes=[d]),
                        Op("vsub", Q.VEC, dict(src=d, vec=b), reads=[d, b], writes=[d]),
                        Op("vmul", Q.VEC, dict(src=d, imm=s), reads=[d], writes=[d]),
                        Op("vexp2", Q.VEC, dict(t=d), reads=[d], writes=[d])]
            if (isinstance(val, tirx.Add) and isinstance(val.a, tirx.Mul) and ld(val.a.a, [i])
                    and val.a.a.buffer.same_as(dst) and ld(val.a.b, [i]) and ld(val.b, [i])):
                a, b = t_of(val.a.b), t_of(val.b)
                return [Op("vmul", Q.VEC, dict(src=d, vec=a), reads=[d, a], writes=[d]),
                        Op("vadd", Q.VEC, dict(src=d, vec=b), reads=[d, b], writes=[d])]
        elif len(ivs) == 2:
            i, j = ivs
            acc = t_of(body)
            # row gather: x_s[i, j] = idx[i] >= 0 ? X[idx[i], col0 + j] : 0  -> LOAD.IDX (into INP or ACC;
            # a bf16 source gathered into an f32 fragment carries a Cast, which LOAD.IDX performs)
            gat = strip_cast(val)
            if (dst.scope() in ("tcta.inp", "tcta.acc") and isinstance(gat, tirx.Call) and _call_name(gat).endswith("if_then_else")
                    and _is_zero(gat.args[2]) and isinstance(gat.args[1], tirx.BufferLoad)
                    and gat.args[1].buffer.scope() == "global"):
                cond, src = gat.args[0], gat.args[1]
                idx_ld = self._nonneg_index(cond)
                assert idx_ld is not None and ld(idx_ld, [i]) and len(src.indices) == 2, f"gather condition {cond}"
                assert isinstance(src.indices[0], tirx.BufferLoad) and src.indices[0].buffer.same_as(idx_ld.buffer)
                col0, csteps = self._affine_expr(substitute(src.indices[1], {j: tirx.IntImm("int32", 0)}))
                X = src.buffer
                eb = ELEM_BYTES[X.dtype]
                strides = [int(s) for s in X.strides]
                region = Region(self.dram[self.gname(X)], ISA_DTYPE[X.dtype], tuple(int(s) for s in X.shape), (strides[0] * eb, eb))
                rows, cols = self._valid(dst)
                idx_t = t_of(idx_ld)
                incr = self._keys("col0", csteps)
                self.collapse("gather")
                return [Op("loadidx", Q.LOAD, dict(dst=acc, region=region, idx=idx_t, nrows=rows,
                                                   row_bytes=pad64(int(X.shape[1]) * eb), cols_valid=cols, col0=col0),
                           incr, reads=[idx_t, self.dram_ref(X)], writes=[acc])]
            m = exp2_of_scaled_diff(val)
            if m and ld(m[0], [i, j]) and m[0].buffer.same_as(dst) and ld(m[1], [i]):
                v, s = t_of(m[1]), m[2]
                return [Op("exp2_acc", Q.VEC, dict(acc=acc, vec=v, scale=s), reads=[acc, v], writes=[acc])]
            slots = self.wide_slots(dst, acc)
            wide = {"wide": True} if len(slots) > 1 else {}
            rows, cols = self._valid(dst)
            neg = negated(val)
            if neg is not None and ld(neg, [i, j]) and same_tile(neg, body):
                return [Op("vneg", Q.VEC, dict(src=acc), reads=[acc], writes=[acc])]
            if isinstance(val, tirx.Sub):
                a, b = strip_cast(val.a), strip_cast(val.b)
                if ld(a, [i, j]) and same_tile(a, body) and ld(b, [i]):
                    v = t_of(b)                                   # acc[i, j] -= vec[i]: VSUB row broadcast
                    return [Op("vsub", Q.VEC, dict(src=acc, vec=v), reads=[acc, v], writes=[acc])]
                if ld(a, [i, j]) and ld(b, [i, j]) and b.buffer.scope() == "tcta.acc":
                    # dst = a - b over ACC tiles: (copy a into dst,) negate b, add, negate b back
                    x = t_of(b)
                    ops = []
                    if not same_tile(a, body):
                        ops += [Op("vcopy", Q.VEC, dict(dst=acc, src=t_of(a)), reads=[t_of(a)], writes=[acc])]
                    ops += [Op("vneg", Q.VEC, dict(src=x), reads=[x], writes=[x]),
                            Op("vgate", Q.VEC, dict(acc=acc, vec=x, gate_op="add"), reads=[acc, x], writes=[acc]),
                            Op("vneg", Q.VEC, dict(src=x), reads=[x], writes=[x])]
                    self.collapse("acc_sub")
                    return ops
            if (isinstance(val, tirx.Mul) and ld(val.a, [i, j]) and same_tile(val.a, body) and isinstance(val.b, tirx.Call)
                    and _call_name(val.b).endswith("exp2") and ld(val.b.args[0], [i, j]) and val.b.args[0].buffer.scope() == "tcta.acc"):
                g = t_of(val.b.args[0])                           # acc *= exp2(gate): VGATE decay (KDA)
                self.collapse("vgate_decay")
                return [Op("vgate", Q.VEC, dict(acc=acc, vec=g, gate_op="decay"), reads=[acc, g], writes=[acc])]
            if isinstance(val, (tirx.Mul, tirx.Add)) and ld(val.a, [i, j]) and same_tile(val.a, body):
                gate_op = "mul" if isinstance(val, tirx.Mul) else "add"
                if gate_op == "mul" and ld(val.b, [i]):
                    v = t_of(val.b)
                    return [Op("vscale_acc", Q.VEC, dict(acc=acc, vec=v, **wide), reads=slots + [v], writes=slots)]
                if ld(val.b, [i, j]) and val.b.buffer.scope() in ("tcta.acc", "tcta.vec"):
                    # elementwise ACC x ACC (SwiGLU) or ACC x 2-D VEC tile (masks, identity): VGATE acc, src, op
                    g = t_of(val.b)
                    self.collapse(f"vgate_{gate_op}")
                    return [Op("vgate", Q.VEC, dict(acc=acc, vec=g, gate_op=gate_op), reads=[acc, g], writes=[acc])]
            if isinstance(val, tirx.Div) and ld(val.a, [i, j]) and val.a.buffer.same_as(dst) and ld(val.b, [i]):
                l = t_of(val.b)
                return [Op("vrcp", Q.VEC, dict(t=l), reads=[l], writes=[l]),
                        Op("vscale_acc", Q.VEC, dict(acc=acc, vec=l, **wide), reads=slots + [l], writes=slots)]
            if (isinstance(val, tirx.Call) and _call_name(val).endswith("sigmoid") and ld(val.args[0], [i, j])
                    and same_tile(val.args[0], body)):
                self.collapse("vsigmoid")
                return [Op("vsigmoid", Q.VEC, dict(t=acc), reads=[acc], writes=[acc])]
        raise NotImplementedError(f"unrecognised T.parallel body: {body}")

    # ---- Phase A post-filter: fills before gemm/rowmax, exp2+rowsum fusion (recursive over loops) ----
    def postfilter(self, ops):
        out = []
        i = 0
        while i < len(ops):
            op = ops[i]
            if isinstance(op, Loop):
                op.body = self.postfilter(op.body)
                out.append(op)
                i += 1
                continue
            nxt = next((o for o in ops[i + 1:] if isinstance(o, Op) and not o.is_event()), None)
            if (op.name == "mma" and not op.args["accumulate"] and nxt is not None and nxt.name == "vgate"
                    and nxt.args["gate_op"] == "mul" and nxt.args["acc"] == op.args["acc"]
                    and nxt.args["vec"].sp in (SP.VEC, SP.ACC) and nxt.args["vec"].slot < self.cfg.vec_row_slot_start):
                # gemm then `acc *= mask` (a 2-D 0/1 tile in VEC or ACC) -> MMA.MASKED mul (v0.1.3). Only the
                # untransposed form exists: a Gram product (MMA.T) keeps the VGATE.
                mask = nxt.args["vec"]
                fused = Op("mma_masked", Q.MMA, dict(acc=op.args["acc"], a=op.args["a"], b=op.args["b"], mask=mask,
                                                     accumulate=False, mask_mode="mul"), dict(op.incr),
                           reads=op.reads + [mask], writes=list(op.writes), meta=dict(op.meta))
                ops.remove(nxt)
                self.collapse("mma_masked_mul")
                out.append(fused)
                i += 1
                continue
            if op.name == "vfill" and "fill_of" in op.meta:
                t = op.meta["fill_of"]
                # the next op touching the tile; a gemm into it takes accumulate=False instead
                touch = next((o for o in ops[i + 1:] if isinstance(o, Op) and (t in o.reads or t in o.writes)), None)
                if touch is not None and touch.meta.get("gemm") and touch.args["acc"] == t and op.meta["value"] == 0.0:
                    touch.args["accumulate"] = False
                    touch.reads = [r for r in touch.reads if r != t]
                    i += 1
                    continue
                if (nxt is not None and nxt.meta.get("rowmax") and nxt.args["dst"] == t and not nxt.meta["clear"]
                        and op.meta["value"] == float("-inf")):
                    i += 1
                    continue
            if op.name == "exp2_acc":
                acc, v, s = op.args["acc"], op.args["vec"], op.args["scale"]
                tmp = self.vec_temp
                seq = [Op("vmul", Q.VEC, dict(src=acc, imm=s), reads=[acc], writes=[acc]),
                       Op("vfill", Q.VEC, dict(dst=tmp, imm=0.0, rows_valid=64, cols_valid=1), writes=[tmp]),
                       Op("vadd", Q.VEC, dict(src=tmp, vec=v), reads=[tmp, v], writes=[tmp]),
                       Op("vmul", Q.VEC, dict(src=tmp, imm=s), reads=[tmp], writes=[tmp])]
                if nxt is not None and nxt.meta.get("rowsum") and nxt.args["src"] == acc:
                    seq.append(Op("vexp2_sum", Q.VEC, dict(t=acc, dst=nxt.args["dst"], vec=tmp),
                                  reads=[acc, tmp], writes=[acc, nxt.args["dst"]]))
                    ops.remove(nxt)
                else:
                    seq.append(Op("vexp2", Q.VEC, dict(t=acc, vec=tmp), reads=[acc, tmp], writes=[acc]))
                for o in seq:
                    o.meta["loop"] = op.meta.get("loop")
                out.extend(seq)
                i += 1
                continue
            out.append(op)
            i += 1
        return out


# --------------------------------------------------------------------------
# Phase B: tokens
# --------------------------------------------------------------------------
#
# Positions in the two-pass expansion are tuples (j0, pass0, j1, pass1, ..., jk): item indices at
# even entries, the pass (1 = first iteration, 2 = steady state) of each enclosing loop at odd
# entries. A loop's own position is its item tuple (prefix of its body's positions); primes pushed
# before an inner LOOP sit at pos + (PRIME,), which sorts between the LOOP and its body.

PRIME = 0
LEVEL_CAP = 64         # search bound for a loop's boundary FIFO level


def _expand(items, ctx, out):
    for j, it in enumerate(items):
        pos = ctx + (j,)
        if isinstance(it, Loop):
            for ps in (1, 2):
                _expand(it.body, pos + (ps,), out)
        else:
            out.append((pos, it))


def _mirrors(pos):
    """Every position denoting the same op in the other pass of each enclosing loop."""
    slots = list(range(1, len(pos), 2))
    res = []
    for combo in itertools.product((1, 2), repeat=len(slots)):
        p = list(pos)
        for i, v in zip(slots, combo):
            p[i] = v
        res.append(tuple(p))
    return res


def _canon(pos):
    """The steady-state (pass 2 everywhere) name of a position: the key tokens are emitted under."""
    return tuple(2 if (i % 2 == 1 and i < len(pos) - 1) else v for i, v in enumerate(pos))


@dataclass
class _Req:
    edge: tuple          # (producer queue, consumer queue)
    push: tuple          # position of the push it needs
    site: tuple          # position of the consumer


@dataclass
class _Summary:
    """One loop's level-invariant plan for one edge (context-free)."""
    L: int                       # FIFO level at every iteration boundary
    N: int                       # pushes per iteration (body ops, primes of inner loops, inner leftovers)
    k_sites: dict                # op pos (steady-state names) -> pops before it
    hoist: dict                  # inner loop pos -> pops before its LOOP
    prime: dict                  # inner loop pos -> pushes before its LOOP
    end: int                     # pops at the end of the body
    leftovers: list              # positions of the tokens in the FIFO after the loop
    exports: list                # (suffix, c_s): first-iteration needs on pushes outside the loop


class _Tokens:
    """Phase B over the item tree. Token locations after placement:
    before[op pos], after[op pos], before_loop[loop pos], body_end[loop pos], drain."""

    def __init__(self, items):
        self.items = items
        self.seq: list[tuple] = []
        _expand(items, (), self.seq)
        top = [j for j, it in enumerate(items) if isinstance(it, Loop)]
        self.first_loop, self.last_loop = (top[0], top[-1]) if top else (None, None)
        self.before: dict[tuple, list] = {}
        self.after: dict[tuple, list] = {}
        self.before_loop: dict[tuple, list] = {}
        self.body_end: dict[tuple, list] = {}
        self.drain: list = []
        # analysis
        self.pushes: dict[tuple, set] = {}        # edge -> {position}
        self.reqs: list[_Req] = []
        self.writer: dict[TRef, tuple] = {}       # tile -> (queue, position)
        self.access: dict[TRef, dict] = {}        # tile -> {queue: position}
        self.fills: list[tuple] = []              # positions of fill_issued events
        self.fills_waited = 0
        self.fill_wait_pops: list = []
        self.last_load: tuple | None = None       # (position, writes) of the previous LOAD op
        # emission bookkeeping
        self.hoisted = self.body_end_count = self.drained = self.primed = 0
        self.max_level: dict[str, int] = {}

    def label(self, pos):
        if len(pos) > 1:
            return "body"
        if self.first_loop is None or pos[0] < self.first_loop:
            return "pre"
        return "post" if pos[0] > self.last_loop else "mid"

    # ---- analysis ----
    def run(self):
        for pos, op in self.seq:
            self._op(op, pos)
        self._place()
        return self.drained

    def _push(self, edge, pos):
        assert edge in EDGES, f"no token edge {edge[0].value}->{edge[1].value}"
        self.pushes.setdefault(edge, set()).add(pos)

    def _require(self, edge, push_pos, site):
        self._push(edge, push_pos)
        self.reqs.append(_Req(edge, push_pos, site))

    def _op(self, op: Op, pos: tuple):
        if op.name == "fill_issued":
            # the fill completes at the event: retarget the preceding load's writes to it
            self.fills.append(pos)
            self._push((Q.LOAD, Q.MMA), pos)
            if self.last_load is not None:
                for t in self.last_load[1]:
                    self.writer[t] = (Q.LOAD, pos)
                    self.access.setdefault(t, {})[Q.LOAD] = pos
            return
        if op.name == "fill_wait":
            n = op.args["n"]
            c = len(self.fills)
            need = c - n
            fill_pops = max(0, need - self.fills_waited)
            self.fill_wait_pops.append((self.label(pos), n, c, self.fills_waited, fill_pops))
            if need > 0:
                self._require((Q.LOAD, Q.MMA), self.fills[need - 1], pos)
                self.fills_waited = max(self.fills_waited, need)
            return
        q = op.queue
        for t in op.reads:
            if t in self.writer and self.writer[t][0] != q:
                wq, wpos = self.writer[t]
                self._require((wq, q), wpos, pos)
        for t in op.writes:
            for xq, xpos in self.access.get(t, {}).items():
                if xq != q:
                    self._require((xq, q), xpos, pos)
        if q == Q.MMA and any(t.sp == SP.WGT and self.writer.get(t, (None,))[0] == Q.LOAD for t in op.reads):
            self._push((Q.MMA, Q.LOAD), pos)     # bank-release convention (WGT slots the LOAD queue fills)
        for t in op.reads:
            self.access.setdefault(t, {})[q] = pos
        for t in op.writes:
            self.access.setdefault(t, {})[q] = pos
            self.writer[t] = (q, pos)
        self.last_load = (pos, list(op.writes)) if q == Q.LOAD else None

    # ---- placement ----
    def _place(self):
        by_edge: dict[tuple, list] = {}
        for r in self.reqs:
            by_edge.setdefault(r.edge, []).append(r)
        for edge in sorted(set(self.pushes) | set(by_edge), key=lambda e: (e[0].value, e[1].value)):
            self._place_edge(edge, by_edge.get(edge, []))

    def _place_edge(self, edge, reqs):
        x, y = edge
        ename = f"{x.value}->{y.value}"
        pushes: set = set()
        for p in self.pushes.get(edge, ()):
            pushes.update(_mirrors(p))
        by_site: dict[tuple, list] = {}
        for r in reqs:
            by_site.setdefault(r.site, []).append(r.push)
        if os.environ.get("D4_DEBUG") == ename:
            print(f"[D4 {ename}] pushes={sorted(pushes)}")
            for s_, ps in sorted(by_site.items()):
                print(f"[D4 {ename}] site {s_} needs {sorted(ps)}")

        def in_ctx(p, ctx):
            return p[:len(ctx)] == ctx

        def site_needs(items, ctx, j, summaries):
            """Needs of item j of a body (an op's own, or a block's exported first-iteration needs)
            -> [(push pos, c_s)] with c_s the pops inside the block before the need's site (0 for an op)."""
            pj = ctx + (j,)
            if isinstance(items[j], Loop):
                out = []
                for suffix, c_s in summaries[j].exports:
                    for p in by_site.get(pj + (1,) + suffix, []):
                        if not in_ctx(p, pj):
                            out.append((p, c_s))
                return out
            return [(p, 0) for p in by_site.get(pj, [])]

        def run_body(items, ctx1, ctx2, summaries, L, prev_list):
            """One steady-state iteration of a loop body at boundary level L (prev_list: the previous
            iteration's pushes, from the last trial). -> plan dict or None if infeasible. A need on a
            push outside the loop is met in every iteration once it is met in the first, so it is
            left to the first-iteration scan (exports) and ignored here."""
            allpush: list = []
            popped = 0
            k_sites, hoist, prime = {}, {}, {}

            def depth(p):
                """Tokens from the FIFO front up to p's token (0 if already consumed), None if p is
                outside this loop (a first-iteration need the parent must satisfy)."""
                if in_ctx(p, ctx2):
                    return L + allpush.index(p) + 1 if p in allpush else 0
                if in_ctx(p, ctx1):
                    q = ctx2 + p[len(ctx1):]
                    if q in prev_list:
                        jb = len(prev_list) - prev_list.index(q)      # distance from the back, 1 = last push
                        return max(0, L - jb + 1)
                    return 0
                return None

            def crossing_needs(j):
                """Pushes before block j (this or the previous iteration) needed by sites after it."""
                out = []
                for j2 in range(j + 1, len(items)):
                    for p, _ in site_needs(items, ctx2, j2, summaries):
                        if in_ctx(p, ctx1) or (in_ctx(p, ctx2) and p < ctx2 + (j,)):
                            out.append(p)
                return out

            for j, it in enumerate(items):
                pj = ctx2 + (j,)
                if isinstance(it, Loop):
                    S = summaries[j]
                    level_before = L + len(allpush) - popped
                    h = max(0, level_before - S.L)
                    pr = max(0, S.L - level_before)
                    for p, c_s in site_needs(items, ctx2, j, summaries):
                        d = depth(p)
                        if d is None:
                            continue
                        extra = max(0, d - popped - c_s - h)
                        h += extra
                        pr += extra
                    if it.extent is None:
                        # a zero-trip LOOP r must be transparent: a site after it that needs a push before it
                        # gets its pop in front of the loop (a static loop always pops the entry tokens itself)
                        for p in crossing_needs(j):
                            extra = max(0, depth(p) - popped - h)
                            h += extra
                            pr += extra
                    hoist[pj], prime[pj] = h, pr
                    popped += h
                    allpush.extend([pj + (PRIME,)] * pr)
                    popped += S.L
                    allpush.extend(S.leftovers)
                    continue
                k = 0
                for p in by_site.get(pj, []):
                    d = depth(p)
                    if d is None:
                        continue
                    k = max(k, d - popped)
                popped += k
                k_sites[pj] = k
                if popped > L + len(allpush):
                    return None
                if pj in pushes:
                    allpush.append(pj)
            end = len(allpush) - popped
            if end < 0:
                return None
            return dict(allpush=allpush, k_sites=k_sites, hoist=hoist, prime=prime, end=end)

        def first_iteration_exports(items, ctx1, ctx2, summaries, plan):
            """(suffix, c_s) for every first-iteration site whose need lies outside the loop; c_s counts the
            real pops of the first iteration before and at the site (inner loops: their hoist only, so a
            zero-trip inner loop is covered)."""
            exports = []
            cum = 0
            for j, it in enumerate(items):
                pj = ctx2 + (j,)
                if isinstance(it, Loop):
                    for suffix, c_s in summaries[j].exports:
                        if any(not in_ctx(p, ctx1) and not in_ctx(p, ctx2) for p in by_site.get(ctx1 + (j, 1) + suffix, [])):
                            exports.append(((j, 1) + suffix, cum + plan["hoist"][pj] + c_s))
                    cum += plan["hoist"][pj]
                    continue
                cum += plan["k_sites"][pj]
                if any(not in_ctx(p, ctx1) and not in_ctx(p, ctx2) for p in by_site.get(ctx1 + (j,), [])):
                    exports.append(((j,), cum))
            return exports

        summary_of: dict[tuple, _Summary] = {}

        def solve_loop(pos, loop) -> _Summary:
            items = loop.body
            ctx1, ctx2 = pos + (1,), pos + (2,)
            summaries = {j: solve_loop(ctx2 + (j,), it) for j, it in enumerate(items) if isinstance(it, Loop)}
            best = None
            dbg = os.environ.get("D4_DEBUG") == ename
            for L in range(LEVEL_CAP):
                prev: list = []
                plan = None
                for _ in range(8):
                    plan = run_body(items, ctx1, ctx2, summaries, L, prev)
                    if plan is None or plan["allpush"] == prev:
                        break
                    prev = plan["allpush"]
                if dbg and L < 6:
                    print(f"[D4 {ename}] loop {pos} L={L}: " + ("infeasible" if plan is None else
                          f"end={plan['end']} k={ {k: v for k, v in plan['k_sites'].items() if v} } hoist={plan['hoist']} prime={plan['prime']} pushes={plan['allpush']}"))
                if plan is None:
                    continue
                key = (plan["end"], L)                  # fewest body-end pops, then the smallest level
                if best is None or key < best[0]:
                    best = (key, L, plan)
            assert best is not None, f"{ename}: no consistent FIFO level for loop {pos}"
            _, L, plan = best
            allpush = plan["allpush"]
            prev_iter = [ctx1 + p[len(ctx2):] for p in allpush]
            left = (prev_iter + allpush)[len(prev_iter) + len(allpush) - L:] if L else []
            S = _Summary(L, len(allpush), plan["k_sites"], plan["hoist"], plan["prime"], plan["end"], left,
                         first_iteration_exports(items, ctx1, ctx2, summaries, plan))
            summary_of[pos] = S
            return S

        # ---- top level: straight code with the loops as blocks ----
        items = self.items
        summaries = {j: solve_loop((j,), it) for j, it in enumerate(items) if isinstance(it, Loop)}
        allpush: list = []
        popped = 0
        k_top, hoist_top, prime_top = {}, {}, {}

        def depth_top(p):
            return allpush.index(p) + 1 if p in allpush else 0

        for j, it in enumerate(items):
            pj = (j,)
            if isinstance(it, Loop):
                S = summaries[j]
                level_before = len(allpush) - popped
                h = max(0, level_before - S.L)
                pr = max(0, S.L - level_before)
                for p, c_s in site_needs(items, (), j, summaries):
                    extra = max(0, depth_top(p) - popped - c_s - h)
                    h += extra
                    pr += extra
                if it.extent is None:
                    for j2 in range(j + 1, len(items)):
                        for p, _ in site_needs(items, (), j2, summaries):
                            if p < pj:
                                extra = max(0, depth_top(p) - popped - h)
                                h += extra
                                pr += extra
                hoist_top[pj], prime_top[pj] = h, pr
                popped += h
                allpush.extend([pj + (PRIME,)] * pr)
                popped += S.L
                allpush.extend(S.leftovers)
                continue
            k = max([depth_top(p) - popped for p in by_site.get(pj, [])] + [0])
            popped += k
            k_top[pj] = k
            assert popped <= len(allpush), f"{ename}: pop before its push at {pj}"
            if pj in pushes:
                allpush.append(pj)
        drain = len(allpush) - popped
        assert drain >= 0, (ename, len(allpush), popped)

        # ---- record the placement ----
        push_tok, pop_tok = ("push", x, y), ("pop", y, x)
        emit_push = {p for p in pushes if _canon(p) == p}
        for p in emit_push:
            self.after.setdefault(p, []).append(push_tok)
        k_all, hoist_all, prime_all, end_all = dict(k_top), dict(hoist_top), dict(prime_top), {}
        for pos, S in summary_of.items():
            k_all.update(S.k_sites)
            hoist_all.update(S.hoist)
            prime_all.update(S.prime)
            end_all[pos] = S.end
        for pos, k in k_all.items():
            if k:
                self.before.setdefault(pos, []).extend([pop_tok] * k)
        for pos in sorted(set(hoist_all) | set(prime_all)):
            toks = [push_tok] * prime_all.get(pos, 0) + [pop_tok] * hoist_all.get(pos, 0)
            if toks:
                self.before_loop.setdefault(pos, []).extend(toks)
            self.primed += prime_all.get(pos, 0)
            self.hoisted += hoist_all.get(pos, 0)
        for pos, k in end_all.items():
            if k:
                self.body_end.setdefault(pos, []).extend([pop_tok] * k)
            self.body_end_count += k
        self.drain.extend([pop_tok] * drain)
        self.drained += drain

        # ---- static level check over the two-pass expansion (both iterations of every loop) ----
        self._lvl = 0
        self._maxlvl = 0

        def walk(items, ctx):
            for j, it in enumerate(items):
                pos = ctx + (j,)
                if isinstance(it, Loop):
                    cpos = _canon(pos)
                    self._lvl += prime_all.get(cpos, 0) - hoist_all.get(cpos, 0)
                    assert self._lvl >= 0, f"{ename}: hoisted pops before LOOP {pos} exceed the FIFO level"
                    self._maxlvl = max(self._maxlvl, self._lvl)
                    for ps in (1, 2):
                        walk(it.body, pos + (ps,))
                        self._lvl -= end_all.get(cpos, 0)
                        assert self._lvl >= 0, f"{ename}: body-end pops of LOOP {pos} exceed the FIFO level"
                    continue
                cpos = _canon(pos)
                self._lvl -= k_all.get(cpos, 0)
                assert self._lvl >= 0, f"{ename}: pop before its push at {pos}"
                if cpos in emit_push:
                    self._lvl += 1
                    self._maxlvl = max(self._maxlvl, self._lvl)

        walk(items, ())
        self._lvl -= drain
        assert self._lvl == 0, f"{ename}: {self._lvl} tokens unaccounted for after the drain"
        # above fifo_depth the producer stalls on PUSH until the consumer catches up (ISA §8.1); reported, not refused
        self.max_level[ename] = self._maxlvl


# --------------------------------------------------------------------------
# Phase C: emission
# --------------------------------------------------------------------------

def _emit_tok(p: Program, tok):
    kind, a, b = tok
    if kind == "push":
        p.push(a, b)
    else:
        p.pop(a, b)


def _emit_op(p: Program, op: Op):
    if op.incr:
        getattr(p, op.name)(**op.args, incr=op.incr)
    else:
        getattr(p, op.name)(**op.args)


class _Emitter:
    def __init__(self, toks: _Tokens):
        self.p = Program()
        self.toks = toks
        self.tags: dict[int, int | None] = {}     # pc -> id of the innermost enclosing loop (None outside)
        self.loop_pcs: dict[int, tuple] = {}
        self.loops: list = []                     # (id, "R<n>"|extent, step) in pre-order

    def emit(self, items, ctx) -> list:
        tree = []
        for j, it in enumerate(items):
            pos = ctx + (j,)
            if isinstance(it, Loop):
                for t in self.toks.before_loop.get(pos, []):
                    _emit_tok(self.p, t)
                start = len(self.p.instrs)
                if it.extent is None:
                    self.p.loop(r=it.reg, step=it.step)
                    ent = (it.id, f"R{it.reg}", it.step)
                else:
                    self.p.loop(n=it.extent)
                    ent = (it.id, it.extent, 1)
                self.loops.append(ent)
                children = self.emit(it.body, pos + (2,))
                for t in self.toks.body_end.get(pos, []):
                    _emit_tok(self.p, t)
                self.p.endloop()
                self.loop_pcs[it.id] = (start, len(self.p.instrs) - 1)
                tree.append(ent + (children,))
                continue
            for t in self.toks.before.get(pos, []):
                _emit_tok(self.p, t)
            if not it.is_event():
                self.tags[len(self.p.instrs)] = it.meta.get("loop")
                _emit_op(self.p, it)
            for t in self.toks.after.get(pos, []):
                _emit_tok(self.p, t)
        return tree


def edge_counts(instrs) -> dict:
    """PUSH/POP counts per edge; a static LOOP n body counted n times (nested loops multiply), a LOOP r body once."""
    counts: dict[str, list] = {}
    mults = [1]
    for ins in instrs:
        if ins.op == "LOOP":
            mults.append(mults[-1] * ins.args.get("n", 1))
        elif ins.op == "ENDLOOP":
            mults.pop()
        elif ins.op == "PUSH":
            e = f"{ins.queue.value}->{ins.args['to'].value}"
            counts.setdefault(e, [0, 0])[0] += mults[-1]
        elif ins.op == "POP":
            e = f"{ins.args['from'].value}->{ins.queue.value}"
            counts.setdefault(e, [0, 0])[1] += mults[-1]
    return {k: tuple(v) for k, v in counts.items()}


def tile_tir_to_program(func: tirx.PrimFunc, bx: int, by: int | None = None, dram: dict | None = None,
                        cfg: DeviceConfig | None = None) -> tuple[Program, MappingReport]:
    """Map a cut PrimFunc to an ISA program for tile (bx, by). `cfg` is the device descriptor the slot
    plan is checked against (default: the simulator's v1 configuration)."""
    m = _Mapper(func, bx, by, dram or {}, cfg)
    items = m.postfilter(m.linearize())

    toks = _Tokens(items)
    drained = toks.run()

    em = _Emitter(toks)
    tree = em.emit(items, ())
    for t in toks.drain:
        _emit_tok(em.p, t)
    em.p.finish()
    p = em.p
    p.tags = em.tags

    if len(em.loops) == 1:
        loop_extent = em.loops[0][1]
    elif em.loops:
        loop_extent = [l[1] for l in em.loops]
    else:
        loop_extent = 0
    report = MappingReport(
        slots={k: repr(v) for k, v in m.slots.items()},
        loop_extent=loop_extent,
        fill_wait_pops=toks.fill_wait_pops,
        drained=drained,
        edge_counts=edge_counts(p.instrs),
        n_instrs=len(p.instrs),
        body_end_pops=toks.body_end_count,
        hoisted_pops=toks.hoisted,
        primed_pushes=toks.primed,
        unrolled_loops=m.unrolled,
        shared_slots=m.shared_slots,
        max_fifo_level=dict(toks.max_level),
        loops=em.loops,
        loop_pcs=em.loop_pcs,
        loop_tree=tree,
        collapsed=dict(m.collapsed),
    )
    return p, report


def listing(prog: Program) -> str:
    return "\n".join(f"{ins.queue.value:5s} {ins!r}" + (f"  incr={ins.incr}" if ins.incr else "") for ins in prog.instrs)


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from cut import lower_to_tile_tir
    from fa_kernel import build_fa_kernel
    import moe_data
    import moe_kernel

    f = lower_to_tile_tir(build_fa_kernel())
    prog, rep = tile_tir_to_program(f, bx=0, by=0, dram={"Q": 0, "K": 1 << 19, "V": 1 << 20, "Output": 1 << 21})
    print(listing(prog))
    print(rep)
    f = lower_to_tile_tir(moe_kernel.build_moe_persistent_kernel(), scope_map=moe_kernel.SCOPE_MAP)
    prog, rep = tile_tir_to_program(f, bx=0, dram=moe_data.DRAM)
    print(listing(prog))
    print(rep)
