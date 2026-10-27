"""D2: dump the FA kernel's IR after every pass of TileLang's CUDA lowering.

    python compiler/prototype/dump_passes.py [--arch sm_90|sm_80] [--out DIR]

Writes 00_frontend.py (before any pass) and NN_<pass>.py after each pass, plus
passes.tsv (index, pass name, changed?, error) for building the pass table. If
lowering raises, the dumps up to the failing pass are kept and the error is
written to ERROR.txt.
"""

import argparse
import os
import re
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import tilelang  # noqa: E402
from tilelang import tvm  # noqa: E402
from tilelang.engine.lower import lower  # noqa: E402

from fa_kernel import build_fa_kernel  # noqa: E402

PASS_CONFIGS = {tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True}


@tvm.instrument.pass_instrument
class DumpAfterEachPass:
    def __init__(self, out_dir):
        self.out_dir = out_dir
        self.n = 0
        self.prev = None
        self.rows = []
        self.depth = 0

    def enter_pass_ctx(self):
        pass

    def exit_pass_ctx(self):
        pass

    def should_run(self, mod, info):
        return True

    def run_before_pass(self, mod, info):
        self.depth += 1

    def run_after_pass(self, mod, info):
        self.depth -= 1
        # Only record top-level passes: a Sequential's children are reported too,
        # and some TileLang passes call nested passes internally.
        if self.depth > 0:
            return
        self.n += 1
        text = mod.script()
        changed = text != self.prev
        name = re.sub(r"[^A-Za-z0-9_.]", "_", info.name)
        with open(os.path.join(self.out_dir, f"{self.n:02d}_{name}.py"), "w") as f:
            f.write(f"# after pass {self.n}: {info.name}  (changed: {changed})\n")
            f.write(text)
        self.rows.append((self.n, info.name, changed))
        self.prev = text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="sm_90")
    ap.add_argument("--out", default=None)
    ap.add_argument("--num_stages", type=int, default=2)
    ap.add_argument("--no-ws", action="store_true", help="set tl.disable_warp_specialized")
    ap.add_argument("--no-async", action="store_true", help="set tl.enable_async_copy=False")
    a = ap.parse_args()
    variant = a.arch + ("_nows" if a.no_ws else "") + ("_noasync" if a.no_async else "")
    if a.no_ws:
        PASS_CONFIGS[tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED] = True
    if a.no_async:
        PASS_CONFIGS[tilelang.PassConfigKey.TL_ENABLE_ASYNC_COPY] = False
    out = a.out or os.path.join(os.path.dirname(os.path.abspath(__file__)), "dumps", variant)
    os.makedirs(out, exist_ok=True)
    for f in os.listdir(out):
        os.remove(os.path.join(out, f))

    func = build_fa_kernel(num_stages=a.num_stages)
    frontend = func.script()
    with open(os.path.join(out, "00_frontend.py"), "w") as f:
        f.write("# frontend PrimFunc, before any pass\n")
        f.write(frontend)

    inst = DumpAfterEachPass(out)
    inst.prev = tvm.IRModule({"main": func}).script()
    target = tvm.target.Target({"kind": "cuda", "arch": a.arch})
    err = None
    try:
        with tvm.transform.PassContext(config=PASS_CONFIGS, instruments=[inst]), target:
            lower(func, target=target)
    except Exception as e:  # keep dumps up to the failure
        err = e
        with open(os.path.join(out, "ERROR.txt"), "w") as f:
            f.write(f"failed after pass {inst.n} ({inst.rows[-1][1] if inst.rows else 'none'})\n\n")
            f.write("".join(traceback.format_exception(e)))

    with open(os.path.join(out, "passes.tsv"), "w") as f:
        f.write("n\tpass\tchanged\n")
        for n, name, changed in inst.rows:
            f.write(f"{n}\t{name}\t{int(changed)}\n")
    print(f"{a.arch}: {inst.n} passes dumped to {out}" + (f"; FAILED: {type(err).__name__}: {err}"[:300] if err else ""))


if __name__ == "__main__":
    main()
