#!/usr/bin/env python
"""Compare one RTL run (status.txt, dram_out.hex, rtl_trace.txt in the vector directory) with the simulator's
expectations (expect.json, expect_dram.hex, sim_trace.txt, ref.npy). Exit code 0 = pass.

    python rtl/test/compare.py rtl/test/vectors/gemm
"""
from __future__ import annotations

import json
import os
import struct
import sys
from collections import Counter

import numpy as np

PAGE = 4096


def read_hex_pages(path):
    """Sparse @addr hex dump -> {page_addr: bytes}."""
    out, addr, buf = {}, None, []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith("@"):
                if addr is not None:
                    out[addr] = bytes(buf)
                addr, buf = int(line[1:], 16), []
            else:
                buf.append(int(line, 16))
    if addr is not None:
        out[addr] = bytes(buf)
    return out


def f32(b):
    return struct.unpack("<f", b)[0]


def bf16_to_f32(u16):
    return np.asarray(u16, np.uint32).astype(np.uint32).__lshift__(16).view(np.float32)


def rowmax_err(out, ref):
    """model.common.rowmax_err: max over rows of max|out - ref| / max|ref| in the row."""
    out = np.asarray(out, np.float64); ref = np.asarray(ref, np.float64)
    scale = np.maximum(np.abs(ref).max(axis=1, keepdims=True), 1e-30)
    return float((np.abs(out - ref) / scale).max())


def main(d):
    exp = json.load(open(os.path.join(d, "expect.json")))
    status = open(os.path.join(d, "status.txt")).read().split()
    ok = True
    print(f"[{exp['name']}] rtl status: {' '.join(status)}")
    if exp["ok"]:
        if status[0] != "ok":
            print("  FAIL: simulator finished ok, RTL did not"); ok = False
    else:
        want = exp["fault"]["code"]
        if status[0] != "fault" or status[1] != want:
            print(f"  FAIL: expected fault {want}"); ok = False
        else:
            print(f"  fault code matches ({want})")
            if exp["fault"].get("pc", -1) >= 0 and int(status[3]) != exp["fault"]["pc"]:
                print(f"  note: fault pc {status[3]} vs simulator {exp['fault']['pc']}")
    # trace: the same multiset of (queue, op, pc) events (syncs included)
    sim_ev = Counter()
    for line in open(os.path.join(d, "sim_trace.txt")):
        p = line.split()
        if len(p) == 4:
            sim_ev[(p[1], p[2], int(p[3]))] += 1
    rtl_ev = Counter()
    for line in open(os.path.join(d, "rtl_trace.txt")):
        p = line.split()
        if len(p) == 5 and p[2] in ("SYNC", "ISSUE"):
            rtl_ev[(p[1], p[3], int(p[4]))] += 1
    if exp["ok"]:
        if sim_ev != rtl_ev:
            missing = sim_ev - rtl_ev; extra = rtl_ev - sim_ev
            print(f"  FAIL: trace differs: missing {sum(missing.values())} extra {sum(extra.values())}")
            for k in list(missing)[:5]: print("    missing", k)
            for k in list(extra)[:5]: print("    extra", k)
            ok = False
        else:
            print(f"  trace: {sum(rtl_ev.values())} instruction events match the simulator")
    # DRAM
    if exp["ok"] and exp["pages"]:
        want = read_hex_pages(os.path.join(d, "expect_dram.hex"))
        got = read_hex_pages(os.path.join(d, "dram_out.hex"))
        bad = 0; first = []
        for pg in exp["pages"]:
            w = want.get(pg); g = got.get(pg)
            if g is None:
                print(f"  FAIL: page {pg:#x} missing from the RTL dump"); ok = False; continue
            if w != g:
                diff = [i for i in range(PAGE) if w[i] != g[i]]
                bad += len(diff)
                for i in diff[:3]:
                    a = pg + (i & ~3)
                    first.append((a, w[(i & ~3):(i & ~3) + 4].hex(), g[(i & ~3):(i & ~3) + 4].hex()))
        for c in exp["checks"]:
            if c["kind"] == "bitexact":
                if bad:
                    print(f"  FAIL: {bad} DRAM bytes differ (bit-exact check)"); ok = False
                    for a, w4, g4 in first[:6]:
                        try:
                            print(f"    {a:#x}: sim {w4} ({f32(bytes.fromhex(w4)):.7g})  rtl {g4} ({f32(bytes.fromhex(g4)):.7g})")
                        except Exception:
                            print(f"    {a:#x}: sim {w4} rtl {g4}")
                else:
                    print(f"  DRAM: {len(exp['pages'])} pages bit-exact")
            elif c["kind"] == "rowmax":
                ref = np.load(os.path.join(d, c["ref"]))
                rows, cols = c["rows"], c["cols"]
                raw = bytearray()
                for pg in range(c["addr"] // PAGE * PAGE, c["addr"] + rows * cols * 2, PAGE):
                    raw += got.get(pg, bytes(PAGE))
                off = c["addr"] % PAGE
                u16 = np.frombuffer(bytes(raw[off: off + rows * cols * 2]), np.uint16).reshape(rows, cols)
                out = bf16_to_f32(u16)
                err = rowmax_err(out, ref)
                # the simulator's own result for reference
                raw_s = bytearray()
                for pg in range(c["addr"] // PAGE * PAGE, c["addr"] + rows * cols * 2, PAGE):
                    raw_s += want.get(pg, bytes(PAGE))
                u16s = np.frombuffer(bytes(raw_s[off: off + rows * cols * 2]), np.uint16).reshape(rows, cols)
                err_sim = rowmax_err(bf16_to_f32(u16s), ref)
                print(f"  rowmax err: rtl {err:.3e} (sim {err_sim:.3e}) tol {c['tol']:.3e} -> {'PASS' if err <= c['tol'] else 'FAIL'}")
                if err > c["tol"]:
                    ok = False
                if bad:
                    print(f"  note: {bad} DRAM bytes differ from the simulator (SFU paths are tolerance-checked, not bit-exact)")
    print("  RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
