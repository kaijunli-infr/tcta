"""F3: run the persistent MoE program on the simulator with the step trace and report, per expert
boundary, when the LOAD queue issued the first LOAD.IDX of expert e+1 relative to the last MMA of
expert e, and whether the critical queue idles at the boundary. Also runs the combine programs and
checks y against the fixture end to end.

    python compiler/prototype/overlap_report.py [--weights bf16|f4.b32] [--stages N] [--markdown]
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "..", "sim"))
sys.path.insert(0, os.path.join(HERE, "..", "..", "ref"))

import common  # noqa: E402
import moe_data  # noqa: E402
import moe_kernel  # noqa: E402
from cut import lower_to_tile_tir  # noqa: E402
from tcta import Machine, Q  # noqa: E402
from tile_tir_to_program import tile_tir_to_program  # noqa: E402

B, D, E, K = moe_kernel.B, moe_kernel.D, moe_kernel.E, moe_kernel.TOPK
BM = moe_kernel.BLOCK_M


def run_persistent(fx, weight_format="f4.b32", idx=None, w=None, dst=None, num_stages=3, prefetch=True):
    """-> (machine, status, program, report, trace). The trace is [(step, queue, op, pc)]."""
    if idx is None:
        idx, w, dst = moe_data.pad_lists(fx)
    cut = lower_to_tile_tir(moe_kernel.build_moe_persistent_kernel(weight_format=weight_format, num_stages=num_stages, prefetch=prefetch),
                            scope_map=moe_kernel.SCOPE_MAP)
    prog, rep = tile_tir_to_program(cut, bx=0, dram=moe_data.DRAM)
    m = Machine(dram_bytes=moe_data.DRAM_BYTES)
    moe_data.load_dram(m, fx, idx, w, dst, weight_format)
    m.trace = []
    st = m.run(prog.instrs)
    trace, m.trace = m.trace, None
    return m, st, prog, rep, trace


def run_combine(m, fx):
    """Run the four combine tile programs on the machine holding Yp; -> (y, statuses)."""
    cut = lower_to_tile_tir(moe_kernel.build_moe_combine_kernel(), scope_map=moe_kernel.COMBINE_SCOPE_MAP)
    moe_data.load_combine_inputs(m, fx)
    sts = []
    for bx in range(B // moe_kernel.BLOCK_M):
        m.reset()
        prog, rep = tile_tir_to_program(cut, bx=bx, dram=moe_data.DRAM)
        sts.append(m.run(prog.instrs))
    return moe_data.read_rows(m, "y", B, D), sts


def expert_of(prog, rep, trace, counts):
    """Expert index of every trace entry (None for tokens and for instructions outside the token-tile
    loops). Two program shapes: one LOOP r section per expert (loop id = expert), or one static outer
    loop whose body holds K dynamic loops serving experts ee*K + k (v0.1.2 per-level incr); there the
    n-th execution of a body instruction belongs to the n-th token tile of that loop's experts."""
    tiles = [-(-int(c) // BM) for c in counts]
    tags = prog.tags
    dyn_ids = [lid for lid, ext, _ in rep.loops if isinstance(ext, str)]
    outer = [t for t in rep.loop_tree if not isinstance(t[1], str) and t[3]]
    if outer:
        assert len(outer) == 1
        children = [c[0] for c in outer[0][3] if isinstance(c[1], str)]
        Kc = len(children)
        tile_expert = {}
        for k, lid in enumerate(children):
            seq = []
            for e in range(k, E, Kc):
                seq.extend([e] * tiles[e])
            tile_expert[lid] = seq
        occ: dict[int, int] = {}
        out = []
        for step, q, op, pc in trace:
            lid = tags.get(pc)
            if lid in tile_expert:
                n = occ.get(pc, 0)
                occ[pc] = n + 1
                out.append(tile_expert[lid][n] if n < len(tile_expert[lid]) else None)
            else:
                out.append(None)
        return out
    return [tags.get(pc) if tags.get(pc) in dyn_ids else None for step, q, op, pc in trace]


def token_ready(prog, trace):
    """For every work instruction in the trace: the step at which its last token dependency was
    satisfied, i.e. the latest push consumed by the POPs that immediately precede it on its queue
    (-1 if it popped nothing). FIFO tokens are anonymous, so the n-th POP on an edge consumes the n-th
    PUSH; with the free-sync rule the POPs execute in the step of the instruction they gate."""
    push_steps: dict[tuple, list] = {}
    pops: dict[tuple, int] = {}
    pending: dict = {}
    ready = {}
    for i, (step, q, op, pc) in enumerate(trace):
        ins = prog.instrs[pc]
        if op == "PUSH":
            push_steps.setdefault((q, ins.args["to"]), []).append(step)
        elif op == "POP":
            edge = (ins.args["from"], q)
            n = pops.get(edge, 0)
            pops[edge] = n + 1
            ps = push_steps.get(edge, [])
            pending[q] = max(pending.get(q, -1), ps[n] if n < len(ps) else step)
        else:
            ready[i] = pending.pop(q, -1)
    return ready


def boundaries(prog, trace, counts, rep):
    """Per boundary e -> e+1: (e, step of the last MMA of e, step of the first LOAD.IDX of e+1,
    overlap steps, LOAD instrs of e+1 before e's last MMA, MMAs of e at/after that gather,
    LOAD-queue idle steps between e's last and e+1's first LOAD work instruction,
    MMA-queue gap between e's last and e+1's first MMA,
    token slack: last MMA of e minus the step at which the first gather of e+1 was token-ready)."""
    experts = expert_of(prog, rep, trace, counts)
    ready = token_ready(prog, trace)
    last_mma, first_gather, gather_ready, mma_steps = {}, {}, {}, {}
    load_steps: dict[int, list] = {}
    load_ready: dict[int, int] = {}       # latest token the LOAD queue waited for since the previous expert's work
    prev_e = None
    for i, ((step, q, op, pc), e) in enumerate(zip(trace, experts)):
        if q == Q.LOAD and op not in ("PUSH", "POP"):
            # pops between two experts' LOAD work (hoists before LOOP, the prefetch) count for the next expert
            key = e if e is not None else prev_e
            if key is not None:
                load_ready[key] = max(load_ready.get(key, -1), ready[i])
                prev_e = key
        if e is None or op in ("PUSH", "POP"):
            continue
        if q == Q.MMA:
            last_mma[e] = max(last_mma.get(e, -1), step)
            mma_steps.setdefault(e, []).append(step)
        if q == Q.LOAD:
            load_steps.setdefault(e, []).append(step)
            if op == "LOAD.IDX" and e not in first_gather:
                first_gather[e] = step
                gather_ready[e] = load_ready[e]
                load_ready[e] = -1        # from here on, readiness belongs to the expert's own body
    rows = []
    for e in range(E - 1):
        if counts[e] == 0 or counts[e + 1] == 0:
            continue
        ahead = sum(1 for s in load_steps[e + 1] if s < last_mma[e])
        outstanding = sum(1 for s in mma_steps[e] if s >= first_gather[e + 1])
        load_idle = min(load_steps[e + 1]) - max(load_steps[e]) - 1
        mma_gap = min(mma_steps[e + 1]) - last_mma[e]
        slack = last_mma[e] - gather_ready[e + 1] if gather_ready[e + 1] >= 0 else None   # None: no token waited for
        rows.append((e, last_mma[e], first_gather[e + 1], last_mma[e] - first_gather[e + 1], ahead, outstanding,
                     load_idle, mma_gap, slack))
    return rows


def queue_busy(trace):
    """Steps on which each queue executed a work instruction (PUSH/POP cost no issue slot in v0.1.2)."""
    busy = {}
    for step, q, op, pc in trace:
        if op not in ("PUSH", "POP"):
            busy.setdefault(q.value, set()).add(step)
    return {q: len(s) for q, s in busy.items()}


def intra_mma_gap(trace, experts):
    """Median gap between consecutive MMAs inside an expert (the steady-state MMA cadence)."""
    gaps = []
    by_e: dict[int, list] = {}
    for (step, q, op, pc), e in zip(trace, experts):
        if e is not None and q == Q.MMA and op not in ("PUSH", "POP"):
            by_e.setdefault(e, []).append(step)
    for steps in by_e.values():
        gaps.extend(np.diff(sorted(steps)).tolist())
    return float(np.median(gaps)) if gaps else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="f4.b32", choices=["bf16", "f4.b32"])
    ap.add_argument("--markdown", action="store_true")
    ap.add_argument("--stages", type=int, default=3, help="weight ring depth (T.Pipelined stages)")
    ap.add_argument("--no-prefetch", action="store_true", help="count each expert right before its tile loop")
    args = ap.parse_args()
    fx = dict(np.load(os.path.join(HERE, "..", "..", "fixtures", "moe_grouped", "data.npz")))
    m, st, prog, rep, trace = run_persistent(fx, args.weights, num_stages=args.stages, prefetch=not args.no_prefetch)
    parts = moe_data.read_parts(m, B, K, D)
    err_parts = common.rowmax_err(parts.reshape(-1, D), fx["y_parts"].reshape(-1, D))
    y, sts = run_combine(m, fx)
    err_y = common.rowmax_err(y, fx["y"])
    rows = boundaries(prog, trace, fx["counts"], rep)
    busy = queue_busy(trace)
    experts = expert_of(prog, rep, trace, fx["counts"])
    print(f"weights={args.weights} stages={args.stages} prefetch={not args.no_prefetch} instrs={rep.n_instrs} loops={rep.loops} steps={st['steps']} "
          f"mma={st['mma']} work={st['instr']} sync={st.get('sync')} err_parts={err_parts:.3e} err_y={err_y:.3e} "
          f"combine_ok={all(s['ok'] for s in sts)}")
    print(f"tokens: primed={rep.primed_pushes} hoisted={rep.hoisted_pops} body_end={rep.body_end_pops} drained={rep.drained} "
          f"max_fifo_level={rep.max_fifo_level}")
    print(f"queue busy steps: {busy} of {st['steps']}; median intra-expert MMA gap {intra_mma_gap(trace, experts):.1f}")
    ov, ah, ou, li, mg = ([r[i] for r in rows] for i in (3, 4, 5, 6, 7))
    sl = [r[8] for r in rows if r[8] is not None] or [None]
    print(f"boundaries={len(rows)} overlap steps min={min(ov)} max={max(ov)} all_overlapped={all(o > 0 for o in ov)}; "
          f"MMAs of e outstanding at the first gather of e+1: min={min(ou)} max={max(ou)}; "
          f"LOAD instrs of e+1 issued before e's last MMA: min={min(ah)} max={max(ah)}; "
          f"LOAD idle steps at the boundary: min={min(li)} max={max(li)}; MMA gap at the boundary: min={min(mg)} max={max(mg)}; "
          f"token slack of the first gather of e+1 vs the last MMA of e: min={min(sl)} max={max(sl)}")
    if args.markdown:
        print("\n| boundary | last MMA of e | first LOAD.IDX of e+1 | overlap (steps) | MMAs of e outstanding | "
              "e+1 LOAD instrs before e's last MMA | LOAD idle at boundary | MMA gap at boundary | token slack of the gather |")
        print("|---|---|---|---|---|---|---|---|---|")
        for e, lm, fg, o, a, u, idle, gap, slack in rows:
            print(f"| {e} → {e + 1} | {lm} | {fg} | {o} | {u} | {a} | {idle} | {gap} | {slack} |")
    else:
        for e, lm, fg, o, a, u, idle, gap, slack in rows[:4]:
            print(f"  {e}->{e + 1}: last MMA {lm}, first LOAD.IDX {fg}, overlap {o}, MMAs outstanding {u}, "
                  f"LOAD instrs ahead {a}, LOAD idle {idle}, MMA gap {gap}, token slack {slack}")


if __name__ == "__main__":
    main()
