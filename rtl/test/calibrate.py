#!/usr/bin/env python
"""J5: the cycle model against the Verilator trace (phase 7, `docs/phase7-rtl-core-task.md`).

    python rtl/test/calibrate.py                 # every program with vectors/<name>/rtl_trace.txt
    python rtl/test/calibrate.py gemm fa --md    # markdown tables for docs/calibration-report.md

For each program the cycle model runs twice -- the phase-5 planning descriptor (`KV260_PLAN`) and the calibrated
one (`KV260`, `rtl_rules`) -- and the per-instruction issue -> done durations of the RTL trace are paired with the
model's start -> end per pc (in execution order). Reported: total cycles RTL / plan / calibrated and the per
instruction deltas, grouped by (op, shape).
"""
from __future__ import annotations

import os
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gen_vectors as gv  # noqa: E402  (adds sim / model / compiler paths)
from tcta.cycle import CycleMachine  # noqa: E402
import tcta.timing as timing  # noqa: E402
from tcta.timing import KV260, KV260_PLAN  # noqa: E402

VEC_DIR = os.path.join(HERE, "vectors")


def model_run(name, T):
    m = CycleMachine(timing=T)
    prog, _, cfg = gv.PROGRAMS[name](m)
    if cfg is not None:
        m = CycleMachine(timing=T, cfg=cfg)
        prog, _, _ = gv.PROGRAMS[name](m)
    m.run(prog.instrs)
    return m.timing_report


def rtl_run(name):
    """-> (status cycles, [(queue, op, pc, issue, done)] in issue order, fe dispatch gaps)"""
    d = os.path.join(VEC_DIR, name)
    status = open(os.path.join(d, "status.txt")).read().split()
    cycles = int(status[1]) if status[0] == "ok" else int(status[-1])
    open_ev, rows, fe = {}, [], []
    for line in open(os.path.join(d, "rtl_trace.txt")):
        p = line.split()
        if len(p) != 5:
            continue
        cyc, q, kind, op, pc = int(p[0]), p[1], p[2], p[3], int(p[4])
        if kind == "ISSUE":
            open_ev[(q, pc)] = cyc
        elif kind == "DONE":
            rows.append((q, op, pc, open_ev.pop((q, pc)), cyc))
        elif kind == "DISP":
            fe.append((cyc, op, pc))
    rows.sort(key=lambda r: r[3])
    return cycles, rows, fe


def shape_of(r):
    i = r.info
    if "M" in i and "K" in i:
        return f"{i['M']}x{i['K']}x{i['N']}"
    if "M" in i:
        return f"{i['M']}x{i['N']}"
    if "rows" in i:
        return f"{i['rows']}x{i.get('cols', '?')} {i.get('dtype', '')}"
    return ""


def pair(rtl_rows, recs):
    """RTL occurrences and model recs of the same pc, in order."""
    by_pc_m = defaultdict(list)
    for r in sorted(recs, key=lambda r: r.start):
        by_pc_m[r.pc].append(r)
    out = []
    seen = defaultdict(int)
    for q, op, pc, iss, done in rtl_rows:
        k = seen[pc]; seen[pc] += 1
        recs_pc = by_pc_m.get(pc, [])
        rec = recs_pc[k] if k < len(recs_pc) else None
        out.append((q, op, pc, done - iss, rec))
    return out


def report(names, md=False, calibrated=KV260):
    totals = []
    for name in names:
        if not os.path.exists(os.path.join(VEC_DIR, name, "rtl_trace.txt")):
            continue
        rtl_cycles, rtl_rows, fe = rtl_run(name)
        tp = model_run(name, KV260_PLAN)
        tc = model_run(name, calibrated)
        totals.append((name, rtl_cycles, tp.cycles, tc.cycles))
        pp = pair(rtl_rows, tp.recs)
        pc_ = pair(rtl_rows, tc.recs)
        groups = defaultdict(list)
        for (q, op, pc, d_rtl, rp), (_, _, _, _, rc) in zip(pp, pc_):
            key = (op, shape_of(rc) if rc else "", bool(rc and rc.info.get("acc")))
            groups[key].append((d_rtl, (rp.end - rp.start) if rp else None, (rc.end - rc.start) if rc else None))
        print(f"\n## {name}: RTL {rtl_cycles} cycles, model (plan) {tp.cycles} ({_pct(tp.cycles, rtl_cycles)}), "
              f"model (calibrated) {tc.cycles} ({_pct(tc.cycles, rtl_cycles)})\n")
        print("| op | shape | n | RTL issue->done | plan | calibrated | delta |")
        print("|---|---|---|---|---|---|---|")
        for (op, shp, acc), lst in sorted(groups.items(), key=lambda x: (x[0][0], x[0][1])):
            r = sorted(set(x[0] for x in lst)); p = sorted(set(x[1] for x in lst if x[1] is not None)); c = sorted(set(x[2] for x in lst if x[2] is not None))
            rs, ps, cs = _rng(r), _rng(p), _rng(c)
            delta = _pct(c[0], r[0]) if (c and r and len(c) == 1 and len(r) == 1) else (_pct(sum(c) / len(c), sum(r) / len(r)) if c and r else "")
            print(f"| {op}{' (acc)' if acc else ''} | {shp} | {len(lst)} | {rs} | {ps} | {cs} | {delta} |")
        if fe:
            gaps = defaultdict(list)
            for (c0, op0, pc0), (c1, op1, pc1) in zip(fe, fe[1:]):
                gaps[(op1, pc1)].append(c1 - c0)
            print(f"\nfront end: {len(fe)} dispatches; gap to the previous dispatch per (op, pc): " +
                  ", ".join(f"{op}@{pc}:{sorted(set(g))}" for (op, pc), g in sorted(gaps.items(), key=lambda x: x[0][1])[:40]))
    print("\n## totals\n")
    print("| program | RTL cycles | plan | plan / RTL | calibrated | calibrated / RTL |")
    print("|---|---|---|---|---|---|")
    for name, r, p, c in totals:
        print(f"| {name} | {r} | {p} | {_pct(p, r)} | {c} | {_pct(c, r)} |")


def _pct(x, ref):
    return f"{100.0 * (x - ref) / ref:+.1f} %" if ref else ""


def _rng(v):
    if not v:
        return ""
    return str(v[0]) if len(v) == 1 else f"{v[0]}..{v[-1]}"


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    names = args or ["dma_copy", "mma", "mma_mask", "vec", "sfu", "gemm", "fa", "loop"]
    # --timing NAME: the calibrated descriptor (default KV260; KV260_L5 for a `make all MAC8=1` trace)
    cal = KV260
    for a in sys.argv[1:]:
        if a.startswith("--timing="):
            cal = getattr(timing, a.split("=", 1)[1])
    report(names, md="--md" in sys.argv, calibrated=cal)
