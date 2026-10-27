"""DeviceTiming: the parameters of the cycle model (phase 5, `docs/phase5-cycle-model-task.md` §2; phase 6 / ISA v0.1.4
§11: FIFO 8, striping 2, operand broadcast, cores, the Little's-law WGT ring).

Every number here is a planning value, to be calibrated against Verilator later (`docs/cycle-model-report.md`
§"calibration"). Two descriptors are provided: `KV260` (v1: one 16x16 array at 200 MHz, one LPDDR channel
path) and `CONFIG_C` (ISA §11 config C: four 64x64 arrays at 1 GHz, 8 read channels). Sweeps use
`dataclasses.replace(CONFIG_C, vec_lanes=32)` and friends; nothing in `cycle.py` is hard-coded.

Units: one cycle = one clock at `f_mhz`. The cycle model reports cycles and microseconds at that clock.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .isa import SP, Q


@dataclass
class DeviceTiming:
    name: str = "v1-kv260"
    # ---- clock ----
    f_mhz: float = 200.0
    # ---- array: an MMA occupies one array for ceil(M/rows)*ceil(N/cols)*K cycles; its result is visible
    #      (its PUSH fires) mma_latency cycles later; n_arrays MMAs may be in flight ----
    mac_rows: int = 16
    mac_cols: int = 16
    n_arrays: int = 1
    mma_latency: int = 8
    # ---- VEC: ceil(M*N / lanes) cycles per elementwise op over an [M, N] tile (row/column vectors:
    #      ceil(len / lanes)); VCVT and VGATE at the same rate ----
    vec_lanes: int = 16
    # ---- SFU: exp2 / rcp / log2 / sigmoid at ceil(M*N / sfu_lanes); VEXP2.SUM is one pass when fused,
    #      three (subtract, exp2, sum) when not; VEXP2 with a row subtrahend is one or two passes ----
    sfu_lanes: int = 4
    sfu_fused_sub_sum: bool = False
    # ---- DMA: a request occupies its channel for ceil(bytes / bytes_per_cycle) cycles once its data arrives,
    #      dram_latency_cycles after issue; requests on a busy channel queue behind it (the latency overlaps
    #      the previous transfer). LOAD.2D / STORE.2D are one request; LOAD.IDX / STORE.IDX issue one request
    #      per row ("row") or per 64-byte burst ("burst64"), one per cycle, at most max_outstanding_rows in
    #      flight; -1 rows cost nothing ----
    read_channels: int = 1
    write_channels: int = 1
    bytes_per_cycle_per_channel: int = 16
    dram_latency_cycles: int = 100
    row_gather_cost: str = "row"            # "row" | "burst64"
    max_outstanding_rows: int = 16
    # ---- channel striping (phase 5 finding, ISA v0.1.4 §11): a LOAD.2D / STORE.2D request is split over this
    #      many channels (address interleave) and transfers at k * bytes_per_cycle; a gather's rows go
    #      round-robin over k channels. 1 = one request on one channel (the phase-5 rule) ----
    request_channels: int = 1
    # ---- operand broadcast (ISA v0.1.4 §11): the arrays of one core read one A tile at once, so MMAs sharing
    #      an A operand do not contend for its bank's read port ----
    operand_broadcast: bool = False
    # ---- cores (ISA v0.1.4 §11.2): independent instruction streams sharing the DMA channels; the number here
    #      documents the chip, `cycle.run_multicore` takes the programs ----
    cores: int = 1
    # ---- dequant-on-load (f4.b32 / f8.* / f4.r16): a per-channel rate limit on the raw bytes, no extra latency ----
    dequant_bytes_per_cycle: int = 16
    # ---- scratchpads: slot -> bank = slot mod banks; a bank serves ports_per_bank = (readers, writers) per
    #      cycle; an instruction holds its ports for its duration, a later one waits (reported as stall_bank) ----
    banks: dict = field(default_factory=lambda: {SP.INP: 2, SP.WGT: 4, SP.ACC: 4, SP.VEC: 2, SP.STATE: 1})
    ports_per_bank: tuple = (1, 1)
    # ---- token FIFOs: one depth, or a dict {(Q, Q): depth} with a "default" key (v0.1.4: 8; 4 deadlocks KDA) ----
    fifo_depth: int | dict = 8
    # ---- front end: one dispatch per cycle (PUSH/POP free, ISA §8.5); LOOP/ENDLOOP cost loop_overhead_cycles;
    #      LOOP r waits for the VCOUNT writing R[r] plus vcount_to_loop_latency ----
    loop_overhead_cycles: int = 2
    vcount_to_loop_latency: int = 4
    # ---- COMM link (emulated in v0.1; priced, never exercised by the five families) ----
    link_bytes_per_cycle: int = 32
    link_latency_cycles: int = 2000

    # ---- derived ----
    def mma_occupancy(self, M: int, K: int, N: int) -> int:
        return -(-M // self.mac_rows) * -(-N // self.mac_cols) * K

    def mma_issue_cycles(self, M: int, K: int, N: int) -> int:
        """Start to result-visible: array occupancy plus the pipeline latency (task §2)."""
        return self.mma_occupancy(M, K, N) + self.mma_latency

    def fifo_depth_of(self, edge: tuple) -> int:
        if isinstance(self.fifo_depth, dict):
            if edge in self.fifo_depth:
                return self.fifo_depth[edge]
            key = f"{edge[0].value}->{edge[1].value}"
            return self.fifo_depth.get(key, self.fifo_depth.get("default", 4))
        return self.fifo_depth

    @property
    def peak_macs_per_cycle(self) -> int:
        return self.n_arrays * self.mac_rows * self.mac_cols

    @property
    def read_bytes_per_cycle(self) -> int:
        return self.read_channels * self.bytes_per_cycle_per_channel

    def us(self, cycles: int) -> float:
        return cycles / self.f_mhz


KV260 = DeviceTiming()

# Config C after ISA v0.1.4 §11: FIFO 8, request striping 2, operand broadcast, WGT 32 slots (one bank each),
# outstanding window 512 >= a striped gather's demand (one request per cycle x (latency + transfer) ~ 302).
# One stream with the four 64x64 arrays as one MMA unit (the phase-5 view; the phase-6 core sweep uses
# CONFIG_C_CORE for the 4-cores x 1-array view of the same silicon).
CONFIG_C = DeviceTiming(
    name="config-C",
    f_mhz=1000.0,
    mac_rows=64, mac_cols=64, n_arrays=4, mma_latency=12,
    vec_lanes=64,
    sfu_lanes=8, sfu_fused_sub_sum=True,
    read_channels=8, write_channels=2, bytes_per_cycle_per_channel=32, dram_latency_cycles=300,
    row_gather_cost="burst64", max_outstanding_rows=512,
    dequant_bytes_per_cycle=32,
    banks={SP.INP: 8, SP.WGT: 32, SP.ACC: 8, SP.VEC: 4, SP.STATE: 2},
    fifo_depth=8,
    request_channels=2, operand_broadcast=True, cores=1,
)

# The same chip as 4 cores: one 64x64 array per core, the channels shared (ISA v0.1.4 §11.2)
CONFIG_C_CORE = DeviceTiming(
    **{**{f.name: getattr(CONFIG_C, f.name) for f in CONFIG_C.__dataclass_fields__.values()},
       "name": "config-C-core", "n_arrays": 1, "cores": 4})

# The phase-5 descriptor (one request per channel, one reader per bank, FIFO 16, WGT 16), kept so the
# phase-5 tables can be reproduced
CONFIG_C_PHASE5 = DeviceTiming(
    **{**{f.name: getattr(CONFIG_C, f.name) for f in CONFIG_C.__dataclass_fields__.values()},
       "name": "config-C-phase5", "fifo_depth": 16, "request_channels": 1, "operand_broadcast": False,
       "max_outstanding_rows": 256, "banks": {SP.INP: 8, SP.WGT: 16, SP.ACC: 8, SP.VEC: 4, SP.STATE: 2}})

__all__ = ["DeviceTiming", "KV260", "CONFIG_C", "CONFIG_C_CORE", "CONFIG_C_PHASE5", "Q"]
