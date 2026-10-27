"""DRAM layout and host-side data preparation for the MoE programs (E4/E5, phase 3).

The host (here: the test) prepares what the router would hand the device: per-expert
token lists padded with -1 to T_MAX*64 entries, the matching combine weights, the scatter
rows dst[e, i] = token*k + slot (-1 pad; fixture `dst_rows`), and for the combine kernel
the per-slot gather rows rows_idx[s, token] = token*k + s. Expert weights come from the
fixture's f4.b32 images (payload [E, K, N/2] + UE8M0 scales [E, K/32, N/32], Appendix A),
laid out as [W1 | W3] along N so one weight stream covers both gate and up projections;
the bf16 kernel variant uses the fixture's dequantized values W*_deq.
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "sim"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "ref"))
import common  # noqa: E402

MB = 1 << 20
# same addresses for both weight formats; the bf16 and f4 weight images never coexist in one run
DRAM = {"x": 0, "idx": 1 * MB, "dst": 1 * MB + 64 * 1024, "w": 1 * MB + 128 * 1024,
        "rows_idx": 1 * MB + 192 * 1024, "logits": 1 * MB + 256 * 1024, "topk": 1 * MB + 320 * 1024,
        "W13": 2 * MB, "W2": 18 * MB,
        "W13q": 2 * MB, "W13s": 6 * MB, "W2q": 7 * MB, "W2s": 9 * MB,
        "Yp": 26 * MB, "y_shared": 30 * MB, "y": 31 * MB}
DRAM_BYTES = 36 * MB


def pad_lists(fx: dict, t_max: int = 2, block_M: int = 64):
    """-> idx[E+1, t_max*64] int32 (-1 pad; row E is an all -1 pad row so the persistent program's count
    prefetch of expert e+1 stays in bounds), w[E, t_max*64] f32, dst[E, t_max*64] int32 (-1 pad)."""
    E, maxc = fx["idx_lists"].shape
    k = fx["topk_idx"].shape[1]
    rows = t_max * block_M
    assert maxc <= rows, f"fixture needs {maxc} rows per expert, layout has {rows}"
    idx = -np.ones((E + 1, rows), np.int32)
    idx[:E, :maxc] = fx["idx_lists"]
    dst = -np.ones((E, rows), np.int32)
    dst[:, :maxc] = fx["dst_rows"]
    w = np.zeros((E, rows), np.float32)
    valid = idx[:E] >= 0
    slot = np.where(valid, dst - idx[:E] * k, 0)
    w[valid] = fx["topk_w"][idx[:E][valid], slot[valid]]
    return idx, w, dst


def combine_rows(batch: int, k: int) -> np.ndarray:
    """rows_idx[s, token] = token*k + s: the gather rows of the combine kernel."""
    return (np.arange(batch, dtype=np.int32)[None, :] * k + np.arange(k, dtype=np.int32)[:, None]).astype(np.int32)


def expert_weights(fx: dict):
    """-> dict with the device images: W13q/W13s/W2q/W2s (f4.b32) and W13/W2 (dequantized, bf16-valued)."""
    return {
        "W13q": np.concatenate([fx["W1_payload"], fx["W3_payload"]], axis=-1),
        "W13s": np.concatenate([fx["W1_scales"], fx["W3_scales"]], axis=-1),
        "W2q": fx["W2_payload"], "W2s": fx["W2_scales"],
        "W13": np.concatenate([fx["W1_deq"], fx["W3_deq"]], axis=-1), "W2": fx["W2_deq"],
    }


def load_dram(m, fx: dict, idx, w, dst, weight_format: str = "f4.b32"):
    """Write x, lists and weights. Returns the (dequantized) weights the device sees (W13, W2)."""
    m.dram_write(DRAM["x"], common.bf16_bytes(fx["x"]).ravel())
    m.dram_write(DRAM["idx"], idx.astype(np.int32).ravel())
    m.dram_write(DRAM["dst"], dst.astype(np.int32).ravel())
    m.dram_write(DRAM["w"], w.astype(np.float32).ravel())
    W = expert_weights(fx)
    if weight_format == "bf16":
        m.dram_write(DRAM["W13"], common.bf16_bytes(W["W13"]).ravel())
        m.dram_write(DRAM["W2"], common.bf16_bytes(W["W2"]).ravel())
    else:
        for name in ("W13q", "W13s", "W2q", "W2s"):
            m.dram_write(DRAM[name], np.ascontiguousarray(W[name]).ravel())
    return W["W13"], W["W2"]


def load_combine_inputs(m, fx: dict):
    B, k = fx["topk_idx"].shape
    m.dram_write(DRAM["y_shared"], fx["y_shared"].astype(np.float32).ravel())
    m.dram_write(DRAM["rows_idx"], combine_rows(B, k).ravel())


def read_parts(m, batch: int, k: int, d: int) -> np.ndarray:
    """The scatter buffer Yp as [B, k, d] (compare with the fixture's y_parts)."""
    base = DRAM["Yp"]
    return m.dram[base: base + batch * k * d * 4].view(np.float32).reshape(batch, k, d).copy()


def read_rows(m, name: str, rows: int, d: int, row0: int = 0) -> np.ndarray:
    base = DRAM[name] + row0 * d * 4
    return m.dram[base: base + rows * d * 4].view(np.float32).reshape(rows, d).copy()


def reference_expert_output(fx: dict, W13, W2, idx_row, w_row) -> np.ndarray:
    """w[i] * swiglu(x[idx[i]]) per list entry, zeros for -1 rows (float64 math)."""
    d_ff = W2.shape[0]
    x = fx["x"].astype(np.float64)
    rows = np.zeros((idx_row.size, W2.shape[1]))
    valid = idx_row >= 0
    xr = x[idx_row[valid]]
    h1 = xr @ W13[:, :d_ff].astype(np.float64)
    h3 = xr @ W13[:, d_ff:].astype(np.float64)
    h = h1 / (1.0 + np.exp(-h1)) * h3
    rows[valid] = w_row[valid, None] * (h @ W2.astype(np.float64))
    return rows
