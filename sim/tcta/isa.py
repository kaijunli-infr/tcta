"""ISA v0.1 definitions: queues, scratchpads, edges, instruction records.

This module is data only. Semantics live in machine.py.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class Q(str, Enum):
    LOAD = "LOAD"
    MMA = "MMA"
    VEC = "VEC"
    STORE = "STORE"
    COMM = "COMM"
    FE = "FE"  # front end (control)


class SP(str, Enum):
    INP = "INP"
    WGT = "WGT"
    ACC = "ACC"
    VEC = "VEC"
    STATE = "STATE"


# Token graph (ISA §8.1). Any PUSH/POP on an edge not listed is BAD_EDGE.
EDGES = {
    (Q.LOAD, Q.MMA), (Q.LOAD, Q.VEC), (Q.MMA, Q.VEC), (Q.VEC, Q.MMA),
    (Q.MMA, Q.LOAD), (Q.VEC, Q.LOAD), (Q.VEC, Q.STORE), (Q.MMA, Q.STORE),
    (Q.STORE, Q.VEC), (Q.STORE, Q.MMA), (Q.COMM, Q.MMA), (Q.COMM, Q.VEC),
    (Q.MMA, Q.COMM), (Q.VEC, Q.COMM), (Q.LOAD, Q.COMM),
    (Q.LOAD, Q.STORE), (Q.STORE, Q.LOAD),
}

# Which queue executes which opcode.
QUEUE_OF = {
    "LOAD.2D": Q.LOAD, "LOAD.3D": Q.LOAD, "LOAD.IDX": Q.LOAD,
    "STORE.2D": Q.STORE, "STORE.IDX": Q.STORE,
    "MMA": Q.MMA, "MMA.T": Q.MMA, "MMA.MASKED": Q.MMA,
    "SEND": Q.COMM, "RECV": Q.COMM, "RECV.ANY": Q.COMM,
    "LOOP": Q.FE, "ENDLOOP": Q.FE, "FINISH": Q.FE,
    "SETSCALE": Q.FE, "SETBASE": Q.FE, "SETMASK": Q.FE,
    # PUSH/POP are tagged with their own queue at construction.
}
VEC_OPS = {
    "VROWMAX", "VROWSUM", "VSUB", "VMUL", "VADD", "VSCALE_ACC", "VGATE", "VCVT",
    "VMASK", "VTOPK", "VCOUNT", "VSEL", "VEXP2", "VEXP2.SUM", "VRCP", "VLOG2",
    "VSIGMOID", "VNEG", "VFILL", "VMAX", "VCOPY",
}
for _op in VEC_OPS:
    QUEUE_OF[_op] = Q.VEC


@dataclass(frozen=True)
class TRef:
    sp: SP
    slot: int

    def __repr__(self) -> str:
        return f"{self.sp.value}[{self.slot}]"


@dataclass
class Region:
    """DRAM region descriptor (ISA §3)."""
    base: int
    dtype: str
    shape: tuple  # (d0, d1[, d2]) in logical elements along each axis
    stride: tuple  # bytes per step along each axis
    scale_t: float = 1.0  # per-tensor scale for f8.t
    scales_base: Optional[int] = None  # for *.b32: address of the UE8M0 scale grid
    scales_stride: int = 0  # bytes per row of the scale grid


@dataclass
class Instr:
    op: str
    queue: Q
    args: dict = field(default_factory=dict)
    pc: int = -1
    # fields whose values are loop-variant: name -> per-iteration increment
    incr: dict = field(default_factory=dict)

    def __repr__(self) -> str:
        a = ", ".join(f"{k}={v}" for k, v in self.args.items())
        return f"{self.op}({a})"


@dataclass
class DeviceConfig:
    """Implementation parameters (ISA §11)."""
    T_M: int = 64
    T_K: int = 128
    T_N: int = 128
    T_V: int = 64
    T_VR: int = 512
    T_S: int = 128
    slots: dict = field(default_factory=lambda: {
        SP.INP: 2, SP.WGT: 4, SP.ACC: 6, SP.VEC: 16, SP.STATE: 1})
    # VEC slots >= this index are row tiles [1, T_VR]; below are 2-D [T_M, T_V]
    vec_row_slot_start: int = 8
    fifo_depth: int = 8                     # v0.1.4: 8 on every edge (4 deadlocks the KDA program)
    t_deadlock: int = 1_000_000
    n_regs: int = 16
    # ---- accumulation orders (ISA §9: implementation-defined, deterministic per configuration; phase 7 fixes
    #      them to what the RTL does so the RTL can be bit-exact against this simulator) ----
    # MMA: each output element sums its K products sequentially in k order from 0 in float32, the mask (if any)
    # is applied to that sum, then the previous accumulator value is added once (accumulate=1).
    # Row reductions (VROWSUM, VEXP2.SUM): `sum_lanes` lanes, lane l sums columns l, l+L, l+2L, ... sequentially
    # in float32 (columns >= cols_valid contribute 0), then the L partials are added as a binary tree.
    sum_lanes: int = 16


# ISA §11 descriptors (v0.1.4). ACC "8 + 2 wide" is 16 slot indices (a wide accumulator is 4 consecutive slots).
DEVICE_KV260 = DeviceConfig()
DEVICE_CONFIG_C = DeviceConfig(slots={SP.INP: 5, SP.WGT: 32, SP.ACC: 16, SP.VEC: 64, SP.STATE: 2}, fifo_depth=8,
                               t_deadlock=10_000_000)


class Fault(Exception):
    def __init__(self, code: str, queue: Q | None = None, pc: int = -1, **aux: Any):
        self.code, self.queue, self.pc, self.aux = code, queue, pc, aux
        super().__init__(f"{code} queue={queue.value if queue else '-'} pc={pc} {aux}")
