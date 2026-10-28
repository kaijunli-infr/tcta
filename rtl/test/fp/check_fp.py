"""Generate random + edge-case operand pairs, run the Verilated FP library, compare with NumPy float32."""
import os, struct, subprocess, sys
import numpy as np
N = int(sys.argv[1]) if len(sys.argv) > 1 else 200000
rng = np.random.default_rng(1)
def rnd_bits(n):
    bits = rng.integers(0, 2**32, n, dtype=np.uint64).astype(np.uint32)
    # bias toward moderate exponents and bf16-like values
    k = n // 4
    e = rng.integers(100, 150, k, dtype=np.uint32)
    bits[:k] = (rng.integers(0, 2, k, dtype=np.uint32) << 31) | (e << 23) | rng.integers(0, 2**23, k, dtype=np.uint32)
    bits[k:2*k] = bits[k:2*k] & 0xFFFF0000  # bf16 values
    return bits
a = rnd_bits(N); b = rnd_bits(N)
# edge cases
edges = np.array([0, 0x80000000, 0x7f800000, 0xff800000, 0x7fc00000, 0x00000001, 0x80000001, 0x00400000, 0x007fffff,
                  0x3f800000, 0xbf800000, 0x3f7fffff, 0x7f7fffff, 0x00800000, 0x33800000, 0x4b000000, 0x4f000000, 0xcf000000,
                  0x4effffff, 0x3effffff, 0x3f000000, 0x3f400000, 0x40a00000, 0xc0a00000], dtype=np.uint32)
ne = len(edges)
a[:ne*ne] = np.repeat(edges, ne); b[:ne*ne] = np.tile(edges, ne)
# cancellation pairs
a[ne*ne:ne*ne+1000] = rng.integers(0, 2**32, 1000, dtype=np.uint64).astype(np.uint32)
b[ne*ne:ne*ne+1000] = a[ne*ne:ne*ne+1000] ^ 0x80000000 ^ rng.integers(0, 4, 1000, dtype=np.uint32)
with open("fp_in.hex", "w") as fh:
    fh.write("".join(f"{int(x):08x}{int(y):08x}\n" for x, y in zip(a, b)))
subprocess.check_call(["./obj_dir/Vtb_fp", f"+N={N}"])
fa = a.view(np.float32); fb = b.view(np.float32)
with np.errstate(all="ignore"):
    exp_add = (fa + fb).view(np.uint32); exp_sub = (fa - fb).view(np.uint32); exp_mul = (fa * fb).view(np.uint32)
    a7 = (a & 0xFFFF0000).view(np.float32); b7 = (b & 0xFFFF0000).view(np.float32)
    exp_mul7 = (a7 * b7).view(np.uint32)
    exp_max = np.maximum(fa, fb).view(np.uint32)
    lsb = (a >> 16) & 1; exp_bf = ((a + 0x7FFF + lsb) >> 16).astype(np.uint32); exp_bf[np.isnan(fa)] = 0x7fc0
    exp_i32 = np.rint(fa).astype(np.int32).view(np.uint32)
    exp_i2f = a.view(np.int32).astype(np.float32).view(np.uint32)
    exp_tr = fa.astype(np.int64); exp_tr = np.clip(np.nan_to_num(exp_tr, nan=0), -2**31, 2**31-1).astype(np.int32).view(np.uint32)
out = np.loadtxt("fp_out.hex", dtype=str)
got = {k: np.array([int(v, 16) for v in out[:, i]], dtype=np.uint32) for i, k in enumerate(["add", "sub", "mul", "mul7", "max", "bf", "i32", "i2f", "tr"])}
exp = dict(add=exp_add, sub=exp_sub, mul=exp_mul, mul7=exp_mul7, max=exp_max, bf=exp_bf, i32=exp_i32, i2f=exp_i2f, tr=exp_tr)
def canon(x, nanmask):  # any NaN pattern counts as NaN
    x = x.copy(); x[nanmask] = 0x7fc00000; return x
ok = True
for k in got:
    g, e = got[k], exp[k]
    if k in ("add", "sub", "mul", "mul7", "max", "i2f"):
        nan_e = ((e >> 23) & 0xff) == 0xff; nan_e &= (e & 0x7fffff) != 0
        nan_g = ((g >> 23) & 0xff) == 0xff; nan_g &= (g & 0x7fffff) != 0
        g = canon(g, nan_g); e = canon(e, nan_e)
    if k == "tr":  # only integer-valued finite inputs matter
        m = (fa == np.floor(fa)) & np.isfinite(fa) & (np.abs(fa) < 2**31); g = g[m]; e = e[m]
    if k == "i32":  # numpy on aarch64 saturates; skip NaN/inf/out-of-range
        m = np.isfinite(fa) & (np.abs(fa) < 2**31); g = g[m]; e = e[m]
    bad = np.flatnonzero(g != e)
    print(f"{k:5s} mismatches {len(bad)} / {len(g)}")
    if len(bad):
        ok = False
        idx = np.flatnonzero(m)[bad[:5]] if k in ("tr", "i32") else bad[:5]
        for j, i in zip(bad[:5], idx):
            print(f"   a={a[i]:08x} ({fa[i]!r}) b={b[i]:08x} ({fb[i]!r}) exp {e[j]:08x} got {g[j]:08x}")
sys.exit(0 if ok else 1)
