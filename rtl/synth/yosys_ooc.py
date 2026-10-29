#!/usr/bin/env python3
"""Per-module Yosys `synth_xilinx` estimate of tcta_core for the KV260's XCK26 (UltraScale+ fabric).

    export PATH=~/.local/opt/oss-cad-suite/bin:$PATH     # yosys + the slang plugin
    python rtl/synth/yosys_ooc.py [-j 4] [--out DIR]       # from the repo root -> rtl/synth/yosys_xck26_estimate.md

Why per module: yosys-slang flattens the design by default, and with --keep-hierarchy it still emits one module
per *instance* (256 mma_pe, 387 f32_add, ...); either way the core needs more than 16 GB before ABC starts
(rtl/synth/yosys_xck26.ys, kept for a machine that has it). Here every distinct module (and distinct
parameterization) is synthesized once, out of context, with every other design module replaced by a blackbox
stub generated from its header; `stat` then reports the blackbox instances as cells, which gives the instance
counts, and the hierarchy totals are the per-module cell counts weighted by those counts. Same mapping as the
whole-design script (synth_xilinx -family xcup -nowidelut), no placement or timing: Fmax comes from Vivado
(vivado_ooc.tcl).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "rtl" / "src"
PKGS = ["tcta_isa_pkg.sv", "tcta_pkg.sv"]
FILES = ["fp/f32_add.sv", "fp/f32_mul.sv", "fp/f32_misc.sv", "sfu.sv", "token_fifo.sv", "token_net.sv",
         "scratchpad.sv", "tile_meta.sv", "dequant.sv", "dma_load.sv", "dma_store.sv", "mma_pe.sv",
         "mma_unit.sv", "vec_unit.sv", "front_end.sv", "queue.sv", "tcta_core.sv"]
# token_check.sv is the TOKEN_CHECK=1 debug twin; tcta_core instantiates it only with TOKEN_CHECK, left out.

# jobs: (job name, module, parameter overrides). Children with a parameterization other than the default are
# mapped by CHILD_VARIANT; tcta_core's five scratchpads are listed explicitly (one instance each).
JOBS: list[tuple[str, str, dict[str, str]]] = [
    ("tcta_core", "tcta_core", {}),
    ("front_end", "front_end", {}),
    ("queue", "queue", {}),
    ("token_net", "token_net", {}),
    ("token_fifo", "token_fifo", {}),
    ("tile_meta", "tile_meta", {}),
    ("dma_load", "dma_load", {}),
    ("dma_store", "dma_store", {}),
    ("dequant_elem", "dequant_elem", {}),
    ("mma_unit", "mma_unit", {}),
    ("mma_pe", "mma_pe", {}),
    ("vec_unit", "vec_unit", {}),
    ("sfu_lane", "sfu_lane", {}),
    ("f32_add", "f32_add", {}),
    ("f32_mul", "f32_mul", {}),
    ("f32_mul_bf16", "f32_mul", {"MA": "7", "MB": "7"}),      # mma_pe's multiplier (BF16 mantissas)
    ("f32_max", "f32_max", {}),
    ("f32_to_bf16", "f32_to_bf16", {}),
    ("f32_to_i32_rne", "f32_to_i32_rne", {}),
    ("i32_to_f32", "i32_to_f32", {}),
    ("f32_to_int_trunc", "f32_to_int_trunc", {}),
]
SP_SLOTS = [("INP", "INP_WORDS", 16), ("WGT", "WGT_WORDS", 16), ("ACC", "ACC_WORDS", 32),
            ("VEC", "VEC_WORDS", 32), ("STATE", "STATE_WORDS", 32)]
CHILD_VARIANT = {("mma_pe", "f32_mul"): "f32_mul_bf16"}
# jobs mapped with a short abc script (strash; dretime; &if): dma_store's byte-addressed row buffer (2048 × 8 flops, a 64-source write mux per
# byte and a 16-byte read window at any offset) does not finish synth_xilinx's default abc call within an hour.
FAST_ABC = {"dma_store"}
# (parent job, child module) -> explicit list of (child job, count) replacing stat's count
EXPLICIT_CHILDREN = {("tcta_core", "scratchpad"): [(f"scratchpad_{n}", 1) for n, _, _ in SP_SLOTS]}

LUTS = {"LUT1", "LUT2", "LUT3", "LUT4", "LUT5", "LUT6"}
FFS = {"FDRE", "FDSE", "FDCE", "FDPE", "FDRE_1", "FDSE_1", "FDCE_1", "FDPE_1"}
GROUPS = [("LUT", LUTS), ("MUXF", {"MUXF7", "MUXF8"}), ("FF", FFS), ("CARRY", {"CARRY4", "CARRY8"}),
          ("DSP", {"DSP48E2"}), ("BRAM36", {"RAMB36E2"}), ("BRAM18", {"RAMB18E2"}),
          ("LUTRAM", {"RAM32M16", "RAM32M", "RAM64M8", "RAM64M", "RAM32X1D", "RAM64X1D", "RAM128X1D",
                      "RAM256X1D", "RAM32X1S", "RAM64X1S", "RAM128X1S", "RAM256X1S", "RAM512X1S", "RAM32X2S",
                      "RAM64X2S"}),
          ("SRL", {"SRL16E", "SRLC32E"})]
# KV260 (K26 SOM, XCK26-SFVC784-2LV-C): the xczu5ev-class fabric
BUDGET = {"LUT": 117120, "FF": 234240, "DSP": 1248, "BRAM36": 144}


def pkg_params() -> dict[str, int]:
    """Evaluate the integer localparams of the packages (simple expressions) for the scratchpad sizes."""
    ns: dict[str, int] = {}
    rx = re.compile(r"localparam\s+int\s+(\w+)\s*=\s*([^;]+);")
    for p in PKGS:
        text = re.sub(r"//.*", "", (SRC / p).read_text())
        for name, expr in rx.findall(text):
            e = expr.strip().replace("?", " if ").replace(":", " else ")
            try:
                ns[name] = int(eval(e, {}, dict(ns)))  # noqa: S307 (repo source, integers only)
            except Exception:
                pass
    return ns


def module_headers() -> dict[str, tuple[str, str]]:
    """module name -> (file, header text up to the port list's closing `);`)."""
    out = {}
    for f in FILES:
        lines = (SRC / f).read_text().splitlines()
        i = 0
        while i < len(lines):
            m = re.match(r"\s*module\s+(\w+)", lines[i])
            if m:
                j = i
                while not lines[j].rstrip().endswith(");"):
                    j += 1
                # file-local packages (sfu_tables) are not visible to a stub; the port lists only use tcta_pkg
                hdr = [l for l in lines[i:j + 1] if not re.match(r"\s*import\s+(?!tcta_)\w+::\*;", l)]
                out[m.group(1)] = (f, "\n".join(hdr))
                i = j
            i += 1
    return out


def header_ports(hdr: str) -> tuple[list[str], list[str]]:
    """(output port names, parameter names) of a module header."""
    outs, params = [], []
    for raw in hdr.splitlines():
        line = re.sub(r"//.*", "", raw)
        line = re.sub(r"\[[^\]]*\]", "", line).strip().rstrip(",").rstrip(");").strip()
        if re.match(r"output\b", line):
            for piece in line.split(","):
                toks = re.findall(r"\w+", piece)
                if toks:
                    outs.append(toks[-1])
    for m in re.finditer(r"parameter\s+[^,=()]*?(\w+)\s*=", hdr):
        params.append(m.group(1))
    return outs, params


def run_job(job: str, module: str, params: dict[str, str], headers, out: Path, keep: bool) -> dict:
    f, _ = headers[module]
    # every other design module as an empty blackbox stub, elaborated by slang for the parent's port widths;
    # slang emits the instances but no module for them, so `hierarchy -generate` makes the blackboxes (output
    # ports first, everything else input) and `setparam -unset` drops the instance parameters hierarchy -check
    # would otherwise reject on a non-parametric blackbox.
    others = {m: h for m, (mf, h) in headers.items() if mf != f}
    stubs = "\n".join(f"(* blackbox *)\n{h}\nendmodule\n" for h in others.values())
    stub_file = out / f"stubs_{job}.sv"
    stub_file.write_text(stubs)
    gen = []
    for m, h in others.items():
        outs, ps = header_ports(h)
        gen.append(f"hierarchy -generate {m} " + " ".join(f"o:{o}*" for o in outs) + " i:*")
        if ps:
            gen.append("setparam " + " ".join(f"-unset {p}" for p in ps) + f" t:{m}")
    g = " ".join(f"-G {k}={v}" for k, v in params.items())
    files = " ".join(str(SRC / p) for p in PKGS) + f" {stub_file} {SRC / f}"
    js = out / f"{job}.json"
    script = f"""plugin -i slang
read_slang --empty-blackboxes --top {module} {g} {files}
""" + "\n".join(gen) + f"""
hierarchy -check -top {module}
""" + (f"""synth_xilinx -family xcup -top {module} -nowidelut -noiopad -noclkbuf -run begin:map_luts
opt_expr -mux_undef -noclkinv
abc -luts 2,2,3,6:5 -script +strash;dretime;&get,-n;&if,{{W}};&put
clean
techmap -map +/xilinx/lut_map.v -map +/xilinx/cells_map.v -D LUT_WIDTH=6
clean
""" if job in FAST_ABC else f"""synth_xilinx -family xcup -top {module} -nowidelut -noiopad -noclkbuf
""") + f"""tee -q -o {js} stat -json -top {module}
"""
    ys = out / f"{job}.ys"
    ys.write_text(script)
    log = out / f"{job}.log"
    with log.open("w") as lf:
        rc = subprocess.run(["yosys", "-s", str(ys)], stdout=lf, stderr=subprocess.STDOUT).returncode
    if rc != 0 or not js.exists():
        tail = "".join(log.read_text().splitlines(True)[-15:])
        raise RuntimeError(f"{job}: yosys rc={rc}\n{tail}")
    d = json.load(js.open())
    mod = d["modules"]["\\" + module] if "\\" + module in d["modules"] else d["modules"][module]
    by_type = {k.lstrip("\\"): v for k, v in mod["num_cells_by_type"].items()}
    if not keep:
        stub_file.unlink()
    return by_type


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("-j", type=int, default=4)
    ap.add_argument("--out", default=str(ROOT / "rtl" / "synth" / "ooc"))
    ap.add_argument("--keep", action="store_true", help="keep the generated stub files")
    ap.add_argument("--only", nargs="*", help="run only these jobs (reuse existing json for the rest)")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    headers = module_headers()
    pp = pkg_params()
    jobs = list(JOBS) + [(f"scratchpad_{n}", "scratchpad", {"WORDS": str(pp[w]), "DW": str(dw), "SP_ID": f"4'd{i}"})
                         for i, (n, w, dw) in enumerate(SP_SLOTS)]
    print(f"{len(jobs)} jobs, scratchpad WORDS: " + ", ".join(f"{n}={pp[w]}" for n, w, _ in SP_SLOTS), flush=True)

    design_modules = set(headers)
    cells: dict[str, dict[str, int]] = {}

    def do(j):
        job, module, params = j
        if a.only is not None and job not in a.only:
            if not (out / f"{job}.json").exists():
                return job, {}
            d = json.load((out / f"{job}.json").open())
            mod = d["modules"].get("\\" + module) or d["modules"][module]
            return job, {k.lstrip("\\"): v for k, v in mod["num_cells_by_type"].items()}
        r = run_job(job, module, params, headers, out, a.keep)
        print(f"  {job:18s} done: " + ", ".join(f"{k}={v}" for k, v in sorted(r.items()) if k in LUTS or k in FFS or k == "DSP48E2" or k in design_modules), flush=True)
        return job, r

    with ThreadPoolExecutor(max_workers=a.j) as ex:
        for job, r in ex.map(do, jobs):
            cells[job] = r
    job_module = {job: module for job, module, _ in jobs}

    # hierarchy totals: own cells (non-design types) + sum over children
    def children(job: str) -> list[tuple[str, int]]:
        module = job_module[job]
        kids = []
        for t, n in cells[job].items():
            if t not in design_modules:
                continue
            if (job, t) in EXPLICIT_CHILDREN:
                kids += EXPLICIT_CHILDREN[(job, t)]
            else:
                kids.append((CHILD_VARIANT.get((module, t), t), n))
        return kids

    def own(job: str) -> dict[str, int]:
        return {t: n for t, n in cells[job].items() if t not in design_modules}

    memo: dict[str, dict[str, int]] = {}

    def total(job: str) -> dict[str, int]:
        if job in memo:
            return memo[job]
        acc = dict(own(job))
        for kid, n in children(job):
            for t, c in total(kid).items():
                acc[t] = acc.get(t, 0) + n * c
        memo[job] = acc
        return acc

    # instance counts in the core
    inst: dict[str, int] = {"tcta_core": 1}

    def count(job: str, n: int):
        for kid, k in children(job):
            inst[kid] = inst.get(kid, 0) + n * k
            count(kid, n * k)
    count("tcta_core", 1)

    def grouped(c: dict[str, int]) -> dict[str, int]:
        g = {name: sum(c.get(t, 0) for t in types) for name, types in GROUPS}
        known = set().union(*(types for _, types in GROUPS))
        g["other"] = sum(v for t, v in c.items() if t not in known)
        return g

    cols = [name for name, _ in GROUPS] + ["other"]
    lines = ["# Yosys synth_xilinx estimate of tcta_core (XCK26 / UltraScale+ fabric)", "",
             "Per-module out-of-context synthesis (`rtl/synth/yosys_ooc.py`), the hierarchy totals weighted by the",
             "instance counts. `own` = the module without its children; `total` = with them. No placement, no timing.", "",
             "## Core total", "", "| cell | count | KV260 budget | used |", "|---|---|---|---|"]
    core = grouped(total("tcta_core"))
    for k in cols:
        b = BUDGET.get(k)
        lines.append(f"| {k} | {core[k]} | {b if b else ''} | {f'{100 * core[k] / b:.1f} %' if b else ''} |")
    others = {t: v for t, v in total("tcta_core").items() if t not in set().union(*(ty for _, ty in GROUPS))}
    if others:
        lines += ["", "other cells: " + ", ".join(f"{t} {v}" for t, v in sorted(others.items()))]
    lines += ["", "## Per module", "", "| module | instances | " + " | ".join(f"own {c}" for c in cols[:6]) + " | " +
              " | ".join(f"total {c}" for c in cols[:6]) + " |", "|---|---|" + "---|" * 12]
    order = sorted(cells, key=lambda j: -grouped(total(j))["LUT"] * inst.get(j, 0))
    for job in order:
        o, t = grouped(own(job)), grouped(total(job))
        lines.append(f"| {job} | {inst.get(job, 0)} | " + " | ".join(str(o[c]) for c in cols[:6]) + " | " +
                     " | ".join(str(t[c]) for c in cols[:6]) + " |")
    lines += ["", "## Children per module (instance counts from the blackbox cells)", ""]
    for job in order:
        k = children(job)
        if k:
            lines.append(f"- {job}: " + ", ".join(f"{c} ×{n}" for c, n in k))
    lines += ["", "## How to read it", "",
              "- `synth_xilinx -family xcup -nowidelut -noiopad -noclkbuf`, Yosys of the OSS CAD Suite with the slang",
              "  frontend; each module once, children as blackboxes, so cross-module optimization (constant ports,",
              "  unused outputs) is not applied: an upper bound per module, a straight sum for the core.",
              f"- {', '.join(sorted(FAST_ABC))}: mapped with a short abc script (strash, dretime, &if) because the default",
              "  abc9 call does not finish within an hour on its byte-addressed row buffer; its LUT count is therefore",
              "  less optimized than the others (the structure, 2048 byte registers each behind a 64-source write mux,",
              "  is what the number measures).",
              "- Block RAM: Yosys maps each scratchpad bank as a true-dual-port RAM but does not merge a port's read",
              "  with its write, so it uses twice the bit-minimum (the five scratchpads hold 4.4 Mbit = 122 RAMB36,",
              "  one RAMB18 counting as half); Vivado merges them. LUT-RAM cells (RAM32M16 etc.) are listed",
              "  separately and are not in the LUT column.",
              "- Budget: K26 SOM (XCK26-SFVC784-2LV-C) 117 120 LUT6, 234 240 FF, 1 248 DSP48E2, 144 RAMB36 (5.1 Mbit),",
              "  64 URAM288 (unused by this flow). No timing: Fmax needs the Vivado run (vivado_ooc.tcl)."]
    md = ROOT / "rtl" / "synth" / "yosys_xck26_estimate.md"
    md.write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:16]))
    print(f"\nwritten {md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
