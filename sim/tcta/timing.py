"""DeviceTiming: the parameters of the cycle model (phase 5, `docs/phase5-cycle-model-task.md` §2; phase 6 / ISA v0.1.4
§11: FIFO 8, striping 2, operand broadcast, cores, the Little's-law WGT ring).

Descriptors: `KV260` (v1: one 16x16 array at 200 MHz, one LPDDR channel path) carries the values measured on the
phase-7 RTL under Verilator (`rtl_rules=True`, docs/calibration-report.md); `KV260_PLAN` is the phase-5 planning
descriptor of the same core; `CONFIG_C` (ISA §11 config C: four 64x64 arrays at 1 GHz, 8 read channels) keeps
planning values. Sweeps use `dataclasses.replace(CONFIG_C, vec_lanes=32)` and friends; nothing in `cycle.py` is
hard-coded.

Units: one cycle = one clock at `f_mhz`. The cycle model reports cycles and microseconds at that clock.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

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
    # ---- v0 RTL rules (phase 7 calibration, `docs/calibration-report.md`). The planning rules above assume a
    #      streaming DMA (one request, latency once, bytes_per_cycle after), a lane-parallel VEC pipe and a free
    #      front end; the v0 core (`rtl/`) is row-serial: `rtl_rules` switches the cycle model to its structure
    #      and the constants below are the values measured on the Verilator trace (never tuned). KV260 carries
    #      them; KV260_PLAN and CONFIG_C keep the planning rules ----
    rtl_rules: bool = False
    #   DMA (phase 8 K1, rows pipelined through a row-buffer ring, dma_rows_in_flight records): a LOAD is a fetch
    #   stage of beats + dma_load_row_overhead cycles per row and a write stage of one slice per 16 columns + 1 per
    #   row, overlapped; the instruction costs setup + latency + rows * max(fetch, write) + min(fetch, write) + tail.
    #   A STORE is a read stage of one slice per 16 columns + 1 per row (f4.r16: dma_store_r16_slice_cycles per
    #   slice + the row's scale chunks + padding beats) and a write stage of beats + dma_store_row_overhead per row.
    #   Small rows are bound by the records in flight: a row holds its record for latency + fetch + write cycles.
    #   (K0: row-serial, latency + beats + slices + 2 per LOAD row, 2 * slices + beats + 1 per STORE row.)
    dma_load_setup_cycles: int = 5
    dma_store_setup_cycles: int = 4
    dma_load_row_overhead: int = 1
    dma_store_row_overhead: int = 1
    dma_tail_cycles: int = 1
    dma_rows_in_flight: int = 8
    dma_store_r16_slice_cycles: int = 9
    dma_idx_slice_cycles: int = 3           # LOAD.IDX / STORE.IDX: the index-slice read every 16 rows (front stage)
    #   VEC: one [row, 16-column] slice per operand read per cycle (a column vector of M elements costs M cycles), a
    #   row-vector operand one read per row, SFU ops sfu_slice_cycles per slice (VEXP2.SUM one more per slice and
    #   one row write), plus vec_op_overhead_cycles per instruction
    vec_op_overhead_cycles: int = 6
    sfu_slice_cycles: int = 6
    sfu_sum_slice_extra: int = 1
    sfu_sum_row_extra: int = 0          # VEXP2.SUM: the partial add on the lane ALUs after a row's last slice (K0.2: 1)
    vec_reduce_extra: int = 0           # VROWMAX / VROWSUM / VEXP2.SUM: the shared reducer's levels after the last slice (K0.2: 4)
    #   MMA (rtl/src/mma_unit.sv, replayed cycle by cycle in cycle._rtl_mma_cycles): mma_setup_cycles from issue to
    #   the run state; operand fills of 16 * ceil16(K) / 16 cycles per A row block and K (MMA) or 16 * n_cg (MMA.T)
    #   per B block into four buffer halves, a half ready mma_fill_ready_latency after its last read and free again
    #   K + mma_release_latency after the last block start that used it; blocks start every P = max(K, mac_rows) cycles
    #   while the result FIFO (mma_rf_rows) has room for a block; a block's row i lands in the FIFO K +
    #   mma_result_latency + i cycles after its start; the converter takes mma_cv_row_cycles per row and the writeback
    #   mma_wb_row_cycles (+ extras for accumulate / masked add / masked mul; padding rows cost the converter only).
    #   Phase 8 L3 (KV260): 8 writeback adders in two lane halves (+1 per accumulated or mask-added row); the converter
    #   keeps 8 lanes (4 lanes would saturate at P = 64).
    #   The queue issues the next instruction the cycle after the previous one completes (no drain overlap).
    mma_setup_cycles: int = 2
    mma_fill_ready_latency: int = 2
    mma_release_latency: int = 20
    mma_result_latency: int = 21
    mma_cv_row_cycles: int = 2
    mma_wb_row_cycles: int = 2
    mma_wb_acc_extra: int = 2
    mma_wb_mask_extra: int = 3
    mma_wb_mask_mul_extra: int = 4
    mma_rf_rows: int = 32
    #   front end: fe_word_cycles per instruction word (fetch, decode, dispatch; PUSH/POP words fe_sync_word_cycles),
    #   fe_ext_word_cycles per extension word plus fe_ext_entry_cycles per loop-variant entry and fe_mod_entry_cycles
    #   per `_mod` entry; LOOP / ENDLOOP loop_overhead_cycles each; LOOP r: vcount_to_loop_latency after the VCOUNT
    #   completes plus loop_r_count_cycles per iteration of the count (ceil(R / step) by repeated subtraction);
    #   dispatch stalls while the target queue holds iq_depth dispatched, unissued instructions (0 = unbounded)
    fe_word_cycles: int = 1
    fe_sync_word_cycles: int = 0
    fe_ext_word_cycles: int = 0
    fe_ext_entry_cycles: int = 0
    fe_mod_entry_cycles: int = 0
    loop_r_count_cycles: int = 0
    iq_depth: int = 0
    #   tokens: a PUSH fired at cycle t is poppable at t + push_visible_cycles
    push_visible_cycles: int = 0

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


# The phase-5 planning descriptor of the v1 KV260 core (the numbers of docs/cycle-model-report.md and the phase-6
# tables): kept for reproducing them
KV260_PLAN = DeviceTiming(name="v1-kv260-plan")

# The v1 KV260 core as built (rtl/, phase 7): every value below is measured on the Verilator trace
# (`rtl/test/calibrate.py`, docs/calibration-report.md §2). dram_latency_cycles is the harness's memory model
# (+MEMLAT, the plan's fixed-latency port assumption), not a measured LPDDR; f_mhz is the plan's clock until the
# synthesis number replaces it.
KV260 = DeviceTiming(
    name="v1-kv260",
    rtl_rules=True,
    dram_latency_cycles=24,
    dma_load_setup_cycles=5, dma_store_setup_cycles=4, dma_load_row_overhead=1, dma_store_row_overhead=1, dma_tail_cycles=1,
    dma_rows_in_flight=8, dma_store_r16_slice_cycles=9, dma_idx_slice_cycles=3,
    vec_op_overhead_cycles=6, sfu_slice_cycles=6, sfu_sum_slice_extra=1, sfu_sum_row_extra=1, vec_reduce_extra=4, sfu_fused_sub_sum=True,
    mma_setup_cycles=2, mma_fill_ready_latency=2, mma_release_latency=20, mma_result_latency=21,
    mma_cv_row_cycles=2, mma_wb_row_cycles=2, mma_wb_acc_extra=3, mma_wb_mask_extra=4, mma_wb_mask_mul_extra=4, mma_rf_rows=32,
    fe_word_cycles=3, fe_sync_word_cycles=3, fe_ext_word_cycles=2, fe_ext_entry_cycles=1, fe_mod_entry_cycles=1,
    loop_overhead_cycles=2, vcount_to_loop_latency=1, loop_r_count_cycles=1, iq_depth=16,
    push_visible_cycles=1,
    # as built: every scratchpad is one group of 16 column banks (all slots share them) with two read ports and
    # one write port per bank; a third reader or a second writer waits (rtl/src/scratchpad.sv)
    banks={SP.INP: 1, SP.WGT: 1, SP.ACC: 1, SP.VEC: 1, SP.STATE: 1},
    ports_per_bank=(2, 1),
)

# Phase 8 L5 measurement: the same core with the 8 x 8 MAC array (rtl `TCTA_MAC8`, `make all MAC8=1`). Blocks start every
# max(K, 8) cycles, the result FIFO holds 16 rows, a result row converts in one pass and the writeback works on one
# adder half (no second mask pass, old + result straight into the write); the fill, release and result latencies
# follow the array size (K + MAC_ROWS + 4, K + MAC_COLS + 5).
KV260_L5 = replace(KV260, name="v1-kv260-mac8", mac_rows=8, mac_cols=8,
                   mma_release_latency=12, mma_result_latency=13, mma_cv_row_cycles=1,
                   mma_wb_acc_extra=2, mma_wb_mask_extra=3, mma_wb_mask_mul_extra=3, mma_rf_rows=16)

# Config C after ISA v0.1.4 §11 (planning rules, not calibrated): FIFO 8, request striping 2, operand broadcast, WGT 32 slots (one bank each),
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

__all__ = ["DeviceTiming", "KV260", "KV260_PLAN", "CONFIG_C", "CONFIG_C_CORE", "CONFIG_C_PHASE5", "Q"]
