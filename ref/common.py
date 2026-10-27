"""Shared helpers for reference generators.

Every generator writes one fixture: a directory with `data.npz` (all arrays)
and `meta.json` (shapes, formats, tolerances, layout notes). Fixtures are
consumed by the simulator tests, the RTL testbench and the FPGA driver, so
they contain BOTH logical float32 values and, where a DRAM format is involved,
the exact byte image the DMA will read (ISA §4, Appendix A).

Numerics: the reference computes in float64 from BF16-rounded inputs, so the
tolerance a kernel is held to is purely its own accumulation/approximation
error, not reference noise.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "sim"))
from tcta import dtypes as dt  # noqa: E402

TOL_ROWMAX = 2.0 ** -7   # max |err| <= TOL * row max (plan + ISA §9)


def bf16(x):
    return dt.to_bf16(np.asarray(x, np.float32))


def rowmax_err(got, ref):
    got, ref = np.asarray(got, np.float64), np.asarray(ref, np.float64)
    g2 = got.reshape(-1, got.shape[-1]); r2 = ref.reshape(-1, ref.shape[-1])
    denom = np.maximum(np.abs(r2).max(axis=1, keepdims=True), 1e-30)
    return float((np.abs(g2 - r2) / denom).max())


def bf16_bytes(x: np.ndarray) -> np.ndarray:
    """Row-major BF16 byte image (uint8) of a float32 array."""
    return (bf16(x).view(np.uint32) >> 16).astype(np.uint16).view(np.uint8)


def f8_t_bytes(x: np.ndarray):
    """Per-tensor FP8 E4M3 image: returns (bytes, scale, dequantized values)."""
    x = np.asarray(x, np.float32)
    scale = np.float32(max(np.abs(x).max() / 448.0, 1e-12))
    codes = dt.quant_e4m3(x / scale)
    deq = bf16(dt.decode_e4m3(codes) * scale)
    return codes.reshape(-1), scale, deq


def f4_b32_bytes(w: np.ndarray):
    """Appendix A `f4.b32`: E2M1 packed 2/byte + UE8M0 scale per 32x32 block.

    Returns (payload bytes row-major [K, N/2], scale codes [K/32, N/32], dequantized [K, N]).
    """
    w = np.asarray(w, np.float32)
    K, N = w.shape
    assert K % 32 == 0 and N % 32 == 0
    blocks = w.reshape(K // 32, 32, N // 32, 32).transpose(0, 2, 1, 3)  # [Kb, Nb, 32, 32]
    amax = np.abs(blocks).max(axis=(2, 3))
    # UE8M0: power-of-two scale so that amax/scale <= 6
    e = np.ceil(np.log2(np.maximum(amax / 6.0, 2.0 ** -126))).astype(np.int32)
    scales = np.exp2(e.astype(np.float32))
    codes = dt.quant_e2m1(blocks / scales[:, :, None, None])            # [Kb, Nb, 32, 32]
    deq = (dt._E2M1[codes] * scales[:, :, None, None]).transpose(0, 2, 1, 3).reshape(K, N)
    codes = codes.transpose(0, 2, 1, 3).reshape(K, N)
    payload = (codes[:, 0::2] | (codes[:, 1::2] << 4)).astype(np.uint8)  # [K, N/2]
    scale_codes = (e + 127).astype(np.uint8)                              # [Kb, Nb]
    return payload, scale_codes, bf16(deq)


def f4_r16_rows(x: np.ndarray):
    """Appendix A `f4.r16` KV entries: one 384-byte row per entry of 576 channels."""
    x = np.asarray(x, np.float32)
    rows = np.stack([dt.encode_f4_r16_row(r) for r in x])
    deq = np.stack([dt.dequant_row("f4.r16", r, ncols=x.shape[1]) for r in rows])
    return rows, deq


class Fixture:
    def __init__(self, name: str, root: str):
        self.name = name
        self.dir = os.path.join(root, name)
        self.arrays: dict[str, np.ndarray] = {}
        self.meta: dict = {"name": name, "tol_rowmax": TOL_ROWMAX}

    def add(self, **arrays):
        for k, v in arrays.items():
            self.arrays[k] = np.ascontiguousarray(v)
        return self

    def save(self):
        os.makedirs(self.dir, exist_ok=True)
        np.savez_compressed(os.path.join(self.dir, "data.npz"), **self.arrays)
        self.meta["arrays"] = {k: {"shape": list(v.shape), "dtype": str(v.dtype)} for k, v in self.arrays.items()}
        with open(os.path.join(self.dir, "meta.json"), "w") as f:
            json.dump(self.meta, f, indent=2, default=_json_default)
        return self.dir


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def load(fixture_dir: str):
    data = dict(np.load(os.path.join(fixture_dir, "data.npz")))
    with open(os.path.join(fixture_dir, "meta.json")) as f:
        meta = json.load(f)
    return data, meta
