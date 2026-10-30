"""H2: sanity tests for the cycle model (`sim/tcta/cycle.py`, `docs/phase5-cycle-model-task.md` §3).

(a) a single MMA costs exactly mma_issue_cycles; (b) two independent MMAs pipeline on different ACC banks and
serialize on the same bank; (c) a LOAD->MMA token gives start(MMA) >= end(LOAD), checked for every popped token
of every family program (the task's one FAIL condition); (d) FIFO depth: a producer that runs ahead is blocked
at depth 1 and free at depth >= its run-ahead, and the FA ring itself never holds more than one LOAD->MMA token
(the ring, not the FIFO, bounds it); (e) LOAD.IDX requests: rows under `row`, ceil(row_bytes/64)*rows under
`burst64`. Plus: the cycle model never changes numerics (same status and DRAM as `Machine`).

Run: python -m pytest tests/test_cycle.py -q
"""
import os
import sys
from dataclasses import replace

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "sim"))

from tcta import ACC, INP, VEC, WGT, DeviceConfig, Machine, Program, Q, Region  # noqa: E402
from tcta import dtypes as dt  # noqa: E402
from tcta.cycle import CycleDeadlock, CycleMachine  # noqa: E402
from tcta.isa import SP  # noqa: E402
from tcta.timing import CONFIG_C, KV260, KV260_PLAN  # noqa: E402

rng = np.random.default_rng(0)


def put_bf16(m, addr, arr):
    arr = np.asarray(arr, np.float32)
    bits = (dt.to_bf16(arr).view(np.uint32) >> 16).astype(np.uint16)
    m.dram_write(addr, bits)
    return Region(addr, "bf16", arr.shape, (arr.shape[1] * 2, 2))


def gemm_program(m, n_pairs=1, accs=(0,)):
    """n_pairs (INP, WGT) tile pairs loaded, all tokens popped, then one MMA per pair into ACC[accs[i]] (so the
    MMAs are token-ready together), then a STORE of the first."""
    A = rng.standard_normal((64, 128)); B = rng.standard_normal((128, 128))
    ra = put_bf16(m, 0, A); rb = put_bf16(m, 1 << 16, B)
    ro = Region(1 << 20, "f32", (64, 128), (128 * 4, 4))
    p = Program()
    for i in range(n_pairs):
        p.load2d(INP(i), ra, 64, 128); p.push(Q.LOAD, Q.MMA)
        p.load2d(WGT(i), rb, 128, 128); p.push(Q.LOAD, Q.MMA)
    for i in range(2 * n_pairs):
        p.pop(Q.MMA, Q.LOAD)
    for i in range(n_pairs):
        p.mma(ACC(accs[i]), INP(i), WGT(i))
    p.push(Q.MMA, Q.STORE); p.pop(Q.STORE, Q.MMA); p.store2d(ACC(accs[0]), ro, 64, 128)
    p.finish()
    return p, A, B, ro


def recs(t, op):
    return [r for r in t.recs if r.op == op]


# ------------------------------------------------------------------ (a)
@pytest.mark.parametrize("T", [KV260_PLAN, CONFIG_C])
def test_single_mma_costs_mma_issue_cycles(T):
    m = CycleMachine(timing=T)
    p, A, B, ro = gemm_program(m)
    st = m.run(p.instrs)
    assert st["ok"] and st["cycles"] == m.timing_report.cycles
    (mma,) = recs(m.timing_report, "MMA")
    assert mma.end - mma.start == T.mma_issue_cycles(64, 128, 128)
    assert mma.busy == T.mma_occupancy(64, 128, 128) and mma.macs == 64 * 128 * 128
    assert mma.info == {"M": 64, "K": 128, "N": 128, "acc": False}


# ------------------------------------------------------------------ (b)
def test_two_mmas_pipeline_on_different_banks_serialize_on_the_same():
    T = replace(KV260_PLAN, n_arrays=2)                       # ACC banks 4: slots 0 and 4 share bank 0
    m = CycleMachine(timing=T)
    p, *_ = gemm_program(m, n_pairs=2, accs=(0, 1))
    m.run(p.instrs)
    a, b = recs(m.timing_report, "MMA")
    assert b.start == a.start + 1 and b.stall_bank == 0 and {a.unit, b.unit} == {0, 1}
    m = CycleMachine(timing=T)
    p, *_ = gemm_program(m, n_pairs=2, accs=(0, 4))
    m.run(p.instrs)
    a, b = recs(m.timing_report, "MMA")
    occ = T.mma_occupancy(64, 128, 128)
    assert b.start == a.start + occ == a.end - T.mma_latency and b.stall_bank > 0   # ACC write port held while a streams
    assert m.timing_report.banks == {"ACC/bank0": b.stall_bank}


# ------------------------------------------------------------------ (c)
def test_token_dependency_load_before_mma():
    m = CycleMachine(timing=KV260_PLAN)
    p, *_ = gemm_program(m)
    m.run(p.instrs)
    t = m.timing_report
    loads, (mma,), (store,) = recs(t, "LOAD.2D"), recs(t, "MMA"), recs(t, "STORE.2D")
    assert mma.start >= max(l.end for l in loads) and mma.stall_token > 0
    assert store.start >= mma.end
    assert t.edges["LOAD->MMA"]["pushes"] == 2 and t.edges["LOAD->MMA"]["max_level"] <= 2


@pytest.mark.parametrize("T", [KV260_PLAN, CONFIG_C])
def test_popped_tokens_were_pushed_first_on_every_family(T):
    tilelang = pytest.importorskip("tilelang")
    sys.path.insert(0, os.path.join(HERE, "..", "compiler", "prototype"))
    sys.path.insert(0, os.path.join(HERE, "..", "ref"))
    import common, gemm_kernel, kda_kernel, mla_kernel  # noqa: E402
    from cut import lower_to_tile_tir  # noqa: E402
    from tile_tir_to_program import tile_tir_to_program  # noqa: E402
    fxd = os.path.join(HERE, "..", "fixtures")
    cases = []
    fx = dict(np.load(os.path.join(fxd, "mla_decode_absorbed", "data.npz")))
    prog, _ = tile_tir_to_program(lower_to_tile_tir(mla_kernel.build_mla_kernel(), scope_map=mla_kernel.SCOPE_MAP), bx=0, dram=mla_kernel.DRAM)
    cases.append((prog, None, mla_kernel.DRAM_BYTES, lambda m: mla_kernel.load_dram(m, fx), T))
    fx2 = dict(np.load(os.path.join(fxd, "kda_chunk64", "data.npz")))
    prog2, _ = tile_tir_to_program(lower_to_tile_tir(kda_kernel.build_kda_kernel(), scope_map=kda_kernel.SCOPE_MAP), bx=0,
                                   dram=kda_kernel.DRAM, cfg=kda_kernel.CONFIG_C)
    cases.append((prog2, kda_kernel.CONFIG_C, kda_kernel.DRAM_BYTES, lambda m: kda_kernel.load_dram(m, fx2), replace(T, fifo_depth=16)))
    for prog, cfg, dram_bytes, load, timing in cases:
        m = CycleMachine(timing=timing, cfg=cfg, dram_bytes=dram_bytes)
        load(m)
        st = m.run(prog.instrs)
        assert st["ok"]
        t = m.timing_report
        # the k-th POP on an edge consumed the k-th PUSH: pop cycle >= push cycle, and the consumer's next work
        # instruction started at or after the pop
        push_t, pop_t = {}, {}
        for cyc, q, op, pc in sorted(t.sync_events, key=lambda x: (x[0], x[2] == "POP")):
            ins = prog.instrs[pc]
            if op == "PUSH":
                push_t.setdefault((q, ins.args["to"]), []).append(cyc)
            else:
                pop_t.setdefault((ins.args["from"], q), []).append(cyc)
        for e, pops in pop_t.items():
            assert len(push_t[e]) == len(pops)
            assert all(pp >= ps for pp, ps in zip(pops, push_t[e])), e
        # consumer start after its pops: for each work instruction, every POP that precedes it in its queue
        by_q = {}
        for cyc, q, op, pc in t.step_trace(include_sync=True):
            by_q.setdefault(q, []).append((cyc, op))
        for q, lst in by_q.items():
            last_pop = -1
            for cyc, op in lst:
                if op == "POP":
                    last_pop = max(last_pop, cyc)
                elif op != "PUSH":
                    assert cyc >= last_pop, (q, op, cyc, last_pop)
                    last_pop = -1


# ------------------------------------------------------------------ (d)
def ring_program(m, n=4):
    """One WGT tile, then n INP tiles into distinct slots (no slot reuse), each with a push; n MMAs each pop
    one token. The INP loads (1024 cycles) are faster than the MMAs (4096), so the LOAD queue runs ahead and
    the FIFO fills while the MMA queue's head waits for the array."""
    A = rng.standard_normal((64, 128)); B = rng.standard_normal((128, 128))
    ra = put_bf16(m, 0, A); rb = put_bf16(m, 1 << 16, B)
    p = Program()
    p.load2d(WGT(0), rb, 128, 128); p.push(Q.LOAD, Q.MMA)
    for i in range(n):
        p.load2d(INP(i), ra, 64, 128); p.push(Q.LOAD, Q.MMA)
    p.pop(Q.MMA, Q.LOAD)
    for i in range(n):
        p.pop(Q.MMA, Q.LOAD)
        p.mma(ACC(i), INP(i), WGT(0))
    p.finish()
    return p


def test_fifo_depth_blocks_a_producer_that_runs_ahead():
    """Depth 1 blocks the LOAD queue on every push while the MMA queue's head waits for the array (tokens are
    consumed when the consumer's POP is reached, i.e. after the previous MMA issued); depth >= 2 covers the
    run-ahead of this program and nothing blocks."""
    cfg = DeviceConfig(slots={SP.INP: 4, SP.WGT: 4, SP.ACC: 6, SP.VEC: 16, SP.STATE: 1}, fifo_depth=4)
    blocked, load_end, level = {}, {}, {}
    for depth in (1, 2, 4):
        T = replace(KV260_PLAN, fifo_depth={"LOAD->MMA": depth, "default": 4}, banks={SP.INP: 4, SP.WGT: 4, SP.ACC: 8, SP.VEC: 2, SP.STATE: 1})
        m = CycleMachine(timing=T, cfg=cfg)
        p = ring_program(m)
        m.run(p.instrs)
        t = m.timing_report
        blocked[depth] = t.edges["LOAD->MMA"]["push_blocked_cycles"]
        load_end[depth] = max(c for c, q, op, pc in t.sync_events if q == Q.LOAD and op == "PUSH")   # the queue's last push accepted
        level[depth] = t.edges["LOAD->MMA"]["max_level"]
        assert level[depth] <= depth
    assert blocked[1] > 0 and blocked[2] == blocked[4] == 0 and t.per_queue["LOAD"]["stall_fifo"] == 0
    assert load_end[1] > load_end[2] == load_end[4]
    assert level[2] == level[4] == 2


def test_fa_ring_holds_one_token_whatever_the_fifo_depth():
    """Finding: the FA kernel's 2-deep K/V rings gate the LOAD queue through the MMA->LOAD releases, so the
    LOAD->MMA FIFO never holds more than one token; depth 1 and depth 4 give the same cycle count."""
    tilelang = pytest.importorskip("tilelang")
    sys.path.insert(0, os.path.join(HERE, "..", "compiler", "prototype"))
    sys.path.insert(0, os.path.join(HERE, "..", "ref"))
    import common  # noqa: E402
    from cut import lower_to_tile_tir  # noqa: E402
    from fa_kernel import build_fa_kernel  # noqa: E402
    from tile_tir_to_program import tile_tir_to_program  # noqa: E402
    DRAM = {"Q": 0, "K": 1 << 19, "V": 1 << 20, "Output": 1 << 21}
    d = dict(np.load(os.path.join(HERE, "..", "fixtures", "flash_attn_gqa", "data.npz")))
    G = d["Q"].shape[0] // d["K"].shape[0]
    kv = [h // G for h in range(d["Q"].shape[0])]
    prog, _ = tile_tir_to_program(lower_to_tile_tir(build_fa_kernel(), cut="inject_pipeline"), bx=0, by=0, dram=DRAM)
    cycles = {}
    for depth in (1, 4):
        m = CycleMachine(timing=replace(KV260_PLAN, fifo_depth={"LOAD->MMA": depth, "default": 4}), dram_bytes=4 << 20)
        m.dram_write(DRAM["Q"], common.bf16_bytes(d["Q"]).ravel())
        m.dram_write(DRAM["K"], common.bf16_bytes(d["K"][kv]).ravel())
        m.dram_write(DRAM["V"], common.bf16_bytes(d["V"][kv]).ravel())
        st = m.run(prog.instrs)
        assert st["ok"]
        t = m.timing_report
        cycles[depth] = t.cycles
        assert t.edges["LOAD->MMA"]["max_level"] <= 1 and t.edges["LOAD->MMA"]["push_blocked_cycles"] == 0
    assert cycles[1] == cycles[4]


# ------------------------------------------------------------------ (e)
@pytest.mark.parametrize("mode,requests", [("row", 64), ("burst64", 4 * 64)])
def test_load_idx_requests(mode, requests):
    m = CycleMachine(timing=replace(KV260_PLAN, row_gather_cost=mode))
    X = rng.standard_normal((256, 128)); rx = put_bf16(m, 0, X)
    p = Program()
    p.vfill(VEC(8), array=np.arange(64, dtype=np.float32).reshape(1, 64)); p.push(Q.VEC, Q.LOAD); p.pop(Q.LOAD, Q.VEC)
    p.loadidx(INP(0), rx, VEC(8), 64, 256, 128)
    p.finish()
    m.run(p.instrs)
    (ld,) = recs(m.timing_report, "LOAD.IDX")
    assert ld.requests == requests and ld.bytes == 64 * 256
    assert ld.busy == 64 * 256 // KV260_PLAN.bytes_per_cycle_per_channel     # channel transfer cycles are the same
    assert ld.end - ld.start >= KV260_PLAN.dram_latency_cycles + ld.busy


def test_skipped_rows_cost_nothing():
    m = CycleMachine(timing=KV260_PLAN)
    X = rng.standard_normal((256, 128)); rx = put_bf16(m, 0, X)
    idx = np.arange(64, dtype=np.float32); idx[32:] = -1
    p = Program()
    p.vfill(VEC(8), array=idx.reshape(1, 64)); p.push(Q.VEC, Q.LOAD); p.pop(Q.LOAD, Q.VEC)
    p.loadidx(INP(0), rx, VEC(8), 64, 256, 128)
    p.finish()
    m.run(p.instrs)
    (ld,) = recs(m.timing_report, "LOAD.IDX")
    assert ld.requests == 32 and ld.bytes == 32 * 256


# ------------------------------------------------------------------ numerics unchanged, deadlock reported
def test_cycle_machine_matches_functional_machine():
    A = rng.standard_normal((64, 128)); B = rng.standard_normal((128, 128))
    outs = []
    for cls, kw in ((Machine, {}), (CycleMachine, {"timing": CONFIG_C})):
        m = cls(**kw)
        ra = put_bf16(m, 0, A); rb = put_bf16(m, 1 << 16, B)
        ro = Region(1 << 20, "f32", (64, 128), (128 * 4, 4))
        p = Program()
        p.load2d(INP(0), ra, 64, 128); p.push(Q.LOAD, Q.MMA)
        p.load2d(WGT(0), rb, 128, 128); p.push(Q.LOAD, Q.MMA)
        p.pop(Q.MMA, Q.LOAD); p.pop(Q.MMA, Q.LOAD)
        p.mma(ACC(0), INP(0), WGT(0)); p.push(Q.MMA, Q.STORE)
        p.pop(Q.STORE, Q.MMA); p.store2d(ACC(0), ro, 64, 128)
        p.finish()
        st = m.run(p.instrs)
        outs.append((st["steps"], st["mma"], m.dram[ro.base: ro.base + 64 * 128 * 4].copy()))
    assert outs[0][0] == outs[1][0] and outs[0][1] == outs[1][1] and np.array_equal(outs[0][2], outs[1][2])


def test_fifo_deadlock_is_reported_not_silent():
    """A producer forced to push 4 tokens before its consumer can pop at depth 1, with the consumer's first pop
    behind a token the producer cannot push: the timing pass raises CycleDeadlock naming the FIFO."""
    cfg = DeviceConfig(slots={SP.INP: 4, SP.WGT: 4, SP.ACC: 6, SP.VEC: 16, SP.STATE: 1}, fifo_depth=4)
    m = CycleMachine(timing=replace(KV260_PLAN, fifo_depth={"LOAD->MMA": 1, "VEC->LOAD": 4, "default": 4}), cfg=cfg)
    A = rng.standard_normal((64, 128)); B = rng.standard_normal((128, 128))
    ra = put_bf16(m, 0, A); rb = put_bf16(m, 1 << 16, B)
    p = Program()
    # LOAD pushes two tokens, then waits for a VEC token; the MMA pops only after the VEC fill, which waits for the LOAD
    p.load2d(INP(0), ra, 64, 128); p.push(Q.LOAD, Q.MMA)
    p.load2d(WGT(0), rb, 128, 128); p.push(Q.LOAD, Q.MMA); p.push(Q.LOAD, Q.VEC)
    p.pop(Q.VEC, Q.LOAD); p.vfill(ACC(1), imm=0.0, rows_valid=64, cols_valid=128); p.push(Q.VEC, Q.MMA)
    p.pop(Q.MMA, Q.VEC); p.pop(Q.MMA, Q.LOAD); p.pop(Q.MMA, Q.LOAD); p.mma(ACC(0), INP(0), WGT(0))
    p.finish()
    with pytest.raises(CycleDeadlock) as ei:
        m.run(p.instrs)
    assert "LOAD" in ei.value.heads and "FIFO full" in ei.value.heads["LOAD"]


@pytest.mark.parametrize("k", [1, 2, 8])
def test_request_striping_shortens_the_transfer_not_the_channel_busy(k):
    """What-if parameter (phase 5 finding): a LOAD.2D striped over k channels transfers k times faster, the
    latency is unchanged and the channel-cycles are the same; k = 1 is the task's rule."""
    T = replace(CONFIG_C, request_channels=k)
    m = CycleMachine(timing=T)
    p, A, B, ro = gemm_program(m)
    m.run(p.instrs)
    t = m.timing_report
    w = [r for r in recs(t, "LOAD.2D") if r.bytes == 64 * 128 * 2][0]          # the first LOAD: no channel queueing
    xfer = 64 * 128 * 2 // (T.bytes_per_cycle_per_channel * k)
    assert w.end - w.start == T.dram_latency_cycles + xfer
    assert w.busy == 64 * 128 * 2 // T.bytes_per_cycle_per_channel
    assert sum(v for key, v in t.channels.items() if key.startswith("read")) == t.per_queue["LOAD"]["busy"]
    assert sum(1 for key, v in t.channels.items() if key.startswith("read") and v) == min(2 * k, T.read_channels)



# ------------------------------------------------------------------ phase 6: broadcast, cores, arbiter

def shared_a_program(m):
    """Two MMAs on different ACC/WGT banks that read the same INP tile (the A operand)."""
    A = rng.standard_normal((64, 128)); B = rng.standard_normal((128, 128))
    ra = put_bf16(m, 0, A); rb = put_bf16(m, 1 << 16, B)
    p = Program()
    p.load2d(INP(0), ra, 64, 128); p.push(Q.LOAD, Q.MMA)
    p.load2d(WGT(0), rb, 128, 128); p.push(Q.LOAD, Q.MMA)
    p.load2d(WGT(1), rb, 128, 128); p.push(Q.LOAD, Q.MMA)
    for _ in range(3):
        p.pop(Q.MMA, Q.LOAD)
    p.mma(ACC(0), INP(0), WGT(0)); p.mma(ACC(1), INP(0), WGT(1))
    p.finish()
    return p


@pytest.mark.parametrize("broadcast", [False, True])
def test_operand_broadcast_lets_two_arrays_read_one_a_tile(broadcast):
    T = replace(KV260_PLAN, n_arrays=2, operand_broadcast=broadcast)
    m = CycleMachine(timing=T)
    m.run(shared_a_program(m).instrs)
    a, b = recs(m.timing_report, "MMA")
    if broadcast:
        assert b.start == a.start + 1 and b.stall_bank == 0
    else:
        assert b.start == a.start + T.mma_occupancy(64, 128, 128) and m.timing_report.banks == {"INP/bank0": b.stall_bank}


def stream_program(m, n_tiles=16):
    """A pure weight stream: n_tiles LOAD.2D of 32 KB into distinct WGT slots, no tokens (saturates the channels)."""
    B = rng.standard_normal((128, 128)); rb = put_bf16(m, 1 << 16, B)
    p = Program()
    for i in range(n_tiles):
        p.load2d(WGT(i), rb, 128, 128)
    p.finish()
    return p


def test_multicore_copies_of_an_idle_channel_program_cost_one():
    """c copies of the GEMM program on c cores take ~1x the time of one when the channels are idle: with the
    config-C write channels (2) the 32 KB f32 STORE of the copies would already contend, so the descriptor here
    has 8 write channels; the read side (8 channels, 2 loads per copy) is idle enough as is."""
    from tcta.cycle import run_multicore
    from tcta.timing import CONFIG_C_CORE
    T = replace(CONFIG_C_CORE, write_channels=8)
    A = rng.standard_normal((64, 128)); B = rng.standard_normal((128, 128))

    def load(m):
        put_bf16(m, 0, A); put_bf16(m, 1 << 16, B)

    def prog():
        m = Machine()
        p, *_ = gemm_program(m)
        return p

    p = prog()
    one, _ = run_multicore(T, [dict(instrs=p.instrs, load=load)])
    two, ms = run_multicore(T, [dict(instrs=p.instrs, load=load)] * 2)
    four, _ = run_multicore(T, [dict(instrs=p.instrs, load=load)] * 4)
    assert two.cycles <= 1.15 * one.cycles and four.cycles <= 1.3 * one.cycles       # channels mostly idle
    assert max(t.cycles for t in two.cores) <= 1.15 * min(t.cycles for t in two.cores)   # identical programs, near-identical timing
    # per-core functional result == single-core result
    single = CycleMachine(timing=T); load(single); single.run(p.instrs)
    for m in ms:
        assert np.array_equal(m.dram, single.dram) and m.status["ok"]


def test_multicore_saturated_channels_scale_with_cores_and_wait_at_the_arbiter():
    from tcta.cycle import run_multicore
    from tcta.timing import CONFIG_C_CORE
    cfg = DeviceConfig(slots={SP.INP: 2, SP.WGT: 16, SP.ACC: 6, SP.VEC: 16, SP.STATE: 1})
    B = rng.standard_normal((128, 128))

    def load(m):
        put_bf16(m, 1 << 16, B)

    p = stream_program(Machine(cfg=cfg))
    one, _ = run_multicore(CONFIG_C_CORE, [dict(instrs=p.instrs, load=load, cfg=cfg)])
    two, _ = run_multicore(CONFIG_C_CORE, [dict(instrs=p.instrs, load=load, cfg=cfg)] * 2)
    assert one.channel_util > 0.8                                                    # one program saturates the 8 channels
    assert two.cycles >= 1.7 * (one.cycles - CONFIG_C_CORE.dram_latency_cycles)      # ~2x the transfer time
    assert all(w > 0 for w in two.arbiter_wait) and max(two.arbiter_wait) <= 1.5 * min(two.arbiter_wait)   # both wait, fairly


# ---------------------------------------------------------------------------------------------------------------
# phase 7 (J5): the calibrated KV260 descriptor reproduces the Verilator measurements (rtl/test/calibrate.py,
# docs/calibration-report.md): row-serial DMA, the array's block schedule, the slice-serial VEC pipe, the front end
# ---------------------------------------------------------------------------------------------------------------

def test_kv260_calibrated_reproduces_rtl_durations():
    m = CycleMachine(timing=KV260)
    A = np.random.default_rng(1).standard_normal((64, 128)).astype(np.float32)
    B = np.random.default_rng(2).standard_normal((128, 128)).astype(np.float32)
    m.dram_write(0, (dt.to_bf16(A).view(np.uint32) >> 16).astype(np.uint16))
    m.dram_write(1 << 16, (dt.to_bf16(B).view(np.uint32) >> 16).astype(np.uint16))
    ra = Region(0, "bf16", (64, 128), (256, 2)); rb = Region(1 << 16, "bf16", (128, 128), (256, 2))
    ro = Region(1 << 20, "f32", (64, 128), (512, 4))
    p = Program()
    p.load2d(INP(0), ra, 64, 128); p.push(Q.LOAD, Q.MMA)
    p.load2d(WGT(0), rb, 128, 128); p.push(Q.LOAD, Q.MMA)
    p.pop(Q.MMA, Q.LOAD); p.pop(Q.MMA, Q.LOAD)
    p.mma(ACC(0), INP(0), WGT(0)); p.mma(ACC(0), INP(0), WGT(0), accumulate=True); p.push(Q.MMA, Q.VEC)
    p.pop(Q.VEC, Q.MMA); p.vrowmax(VEC(0), ACC(0)); p.vsub(ACC(0), vec=VEC(0)); p.push(Q.VEC, Q.STORE)
    p.pop(Q.STORE, Q.VEC); p.store2d(ACC(0), ro, 64, 128)
    p.finish()
    st = m.run(p.instrs)
    assert st["ok"]
    t = m.timing_report
    dur = [(r.op, r.end - r.start) for r in sorted(t.recs, key=lambda r: r.start)]
    ops = [d[0] for d in dur]
    assert ops == ["LOAD.2D", "LOAD.2D", "MMA", "MMA", "VROWMAX", "VSUB", "STORE.2D"]
    # measured on the RTL trace (rtl/test/vectors/{mma,vec}/rtl_trace.txt): LOAD 64x128 bf16 1127, 128x128 2215 (phase 8
    # K1 row-pipelined DMA: beats + 1 per row; K0: 3205 / 6405), MMA 64x128x128 4284 / accumulate 4332 (phase 8 K0.3 unit:
    # block-buffered operands, fixed-point cells; L3: 8 writeback adders, +1 cycle per accumulated row),
    # VROWMAX 64x128 523 (K0.2 reducer), VSUB with a column vector 582, STORE 64x128 f32 2126 (K1; K0: 3140)
    assert dur[0][1] == 1127 and dur[1][1] == 2215
    assert dur[2][1] == 4284 and dur[3][1] == 4332
    assert abs(dur[4][1] - 523) <= 1 and dur[5][1] == 582
    assert dur[6][1] == 2126
    # the front end: 3 cycles per word (PUSH / POP words too), the first LOAD dispatched at cycle 3
    assert t.recs[0].dispatch == 3
