"""Element formats and dequantization (ISA §2.3, §4, Appendix A).

The simulator is the numeric reference: every conversion here is the
normative result. All arithmetic is NumPy float32; BF16 is emulated by
round-to-nearest-even truncation of the float32 bit pattern.
"""
from __future__ import annotations

import numpy as np

# --------------------------------------------------------------------------
# BF16 emulation
# --------------------------------------------------------------------------

def to_bf16(x: np.ndarray) -> np.ndarray:
    """Round float32 array to BF16 (RNE) and return as float32 holding BF16 values."""
    x = np.ascontiguousarray(x, dtype=np.float32)
    bits = x.view(np.uint32)
    # RNE: add rounding bias, then clear low 16 bits. NaNs are preserved.
    lsb = (bits >> 16) & 1
    bias = 0x7FFF + lsb
    rounded = (bits + bias) & 0xFFFF0000
    nan_mask = np.isnan(x)
    out = rounded.view(np.float32).copy()
    out[nan_mask] = np.nan
    return out


# --------------------------------------------------------------------------
# FP8 E4M3 / FP4 E2M1 decode tables
# --------------------------------------------------------------------------

def _e4m3_table() -> np.ndarray:
    t = np.zeros(256, dtype=np.float32)
    for b in range(256):
        s = -1.0 if b & 0x80 else 1.0
        e = (b >> 3) & 0xF
        m = b & 0x7
        if e == 0:
            v = s * (m / 8.0) * 2.0 ** (-6)
        elif e == 15 and m == 7:
            v = np.nan
        else:
            v = s * (1 + m / 8.0) * 2.0 ** (e - 7)
        t[b] = v
    return t


_E4M3 = _e4m3_table()
# E2M1: sign, 2-bit exp, 1-bit mantissa. Values: 0, .5, 1, 1.5, 2, 3, 4, 6
_E2M1 = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6, -0, -0.5, -1, -1.5, -2, -3, -4, -6], dtype=np.float32)


def decode_e4m3(b: np.ndarray) -> np.ndarray:
    return _E4M3[np.asarray(b, dtype=np.uint8)]


def decode_e2m1_packed(b: np.ndarray) -> np.ndarray:
    """Unpack bytes with two E2M1 nibbles each (low nibble first) to float32."""
    b = np.asarray(b, dtype=np.uint8)
    lo = b & 0xF
    hi = (b >> 4) & 0xF
    out = np.empty(b.shape[:-1] + (b.shape[-1] * 2,), dtype=np.float32)
    out[..., 0::2] = _E2M1[lo]
    out[..., 1::2] = _E2M1[hi]
    return out


def decode_ue8m0(b: np.ndarray) -> np.ndarray:
    """UE8M0 scale: 2**(e-127); e == 255 reserved -> NaN."""
    e = np.asarray(b, dtype=np.int32)
    out = np.exp2(e.astype(np.float32) - 127.0)
    out[e == 255] = np.nan
    return out


# --------------------------------------------------------------------------
# Qualifiers (ISA §4). Each dequantizer takes raw bytes for one *row* and
# returns a float32 row of logical elements, already rounded to BF16 when
# the destination is INP/WGT.
# --------------------------------------------------------------------------

QUALIFIERS = {"bf16", "f32", "i32", "f8.t", "f8.b32", "f4.b32", "f4.r16"}

ROW_ALIGN = 64  # bytes; mandatory for LOAD.IDX rows (ISA §6.1)


def dequant_row(q: str, raw: np.ndarray, *, ncols: int, scale_t: float = 1.0,
                block_scales: np.ndarray | None = None) -> np.ndarray:
    """Convert one DRAM row (uint8 bytes) to float32 logical values.

    ncols: number of logical elements expected.
    scale_t: per-tensor scale for f8.t.
    block_scales: for f8.b32/f4.b32, the UE8M0 scale codes for this row's
        32-column blocks (shape [ncols/32]).
    """
    if q == "bf16":
        v = raw[: ncols * 2].view(np.uint16).astype(np.uint32) << 16
        return v.view(np.float32)
    if q == "f32":
        return raw[: ncols * 4].view(np.float32).copy()
    if q == "i32":
        return raw[: ncols * 4].view(np.int32).astype(np.float32)
    if q == "f8.t":
        return to_bf16(decode_e4m3(raw[:ncols]) * np.float32(scale_t))
    if q == "f8.b32":
        assert block_scales is not None and ncols % 32 == 0
        v = decode_e4m3(raw[:ncols]).reshape(-1, 32)
        s = decode_ue8m0(block_scales).reshape(-1, 1)
        return to_bf16((v * s).reshape(-1))
    if q == "f4.b32":
        assert block_scales is not None and ncols % 32 == 0
        v = decode_e2m1_packed(raw[: ncols // 2]).reshape(-1, 32)
        s = decode_ue8m0(block_scales).reshape(-1, 1)
        return to_bf16((v * s).reshape(-1))
    if q == "f4.r16":
        # Appendix A: ncols E2M1 packed (ncols/2 bytes) then ncols/16 E4M3 scales.
        assert ncols % 16 == 0
        nb = ncols // 2
        v = decode_e2m1_packed(raw[:nb]).reshape(-1, 16)
        s = decode_e4m3(raw[nb: nb + ncols // 16]).reshape(-1, 1)
        return to_bf16((v * s).reshape(-1))
    raise ValueError(f"unknown qualifier {q}")


def row_bytes_for(q: str, ncols: int) -> int:
    """Natural (unpadded) DRAM bytes per row for a qualifier."""
    if q == "bf16":
        return ncols * 2
    if q in ("f32", "i32"):
        return ncols * 4
    if q in ("f8.t", "f8.b32"):
        return ncols
    if q == "f4.b32":
        return ncols // 2
    if q == "f4.r16":
        return ncols // 2 + ncols // 16
    raise ValueError(q)


def pad64(n: int) -> int:
    return (n + ROW_ALIGN - 1) // ROW_ALIGN * ROW_ALIGN


# --------------------------------------------------------------------------
# Quantizers (for STORE f4.r16 and for building test fixtures)
# --------------------------------------------------------------------------

def quant_e2m1(x: np.ndarray) -> np.ndarray:
    """Nearest E2M1 code for each float32 value (values assumed pre-scaled to |x|<=6)."""
    mags = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6], dtype=np.float32)
    a = np.abs(x)[..., None]
    idx = np.argmin(np.abs(a - mags), axis=-1).astype(np.uint8)
    # ties: argmin picks lower code -> round-half-down; acceptable for fixtures
    return idx | (np.where(x < 0, 8, 0).astype(np.uint8))


_E4M3_MAGS = np.array(sorted({float(abs(v)) for v in _E4M3 if not np.isnan(v)}), dtype=np.float32)
_E4M3_CODE_OF_MAG = {float(abs(v)): i for i, v in reversed(list(enumerate(_E4M3[:128]))) if not np.isnan(v)}
_E4M3_MAG_CODES = np.array([_E4M3_CODE_OF_MAG[float(m)] for m in _E4M3_MAGS], dtype=np.uint8)


def quant_e4m3(x: np.ndarray) -> np.ndarray:
    """Nearest E4M3 code (magnitude via searchsorted; memory-linear in x)."""
    x = np.asarray(x, dtype=np.float32)
    a = np.minimum(np.abs(x), _E4M3_MAGS[-1])
    hi = np.searchsorted(_E4M3_MAGS, a, side="left")
    hi = np.clip(hi, 1, _E4M3_MAGS.size - 1)
    lo = hi - 1
    pick_hi = (a - _E4M3_MAGS[lo]) > (_E4M3_MAGS[hi] - a)
    code = np.where(pick_hi, _E4M3_MAG_CODES[hi], _E4M3_MAG_CODES[lo]).astype(np.uint8)
    return code | np.where(x < 0, np.uint8(0x80), np.uint8(0))


def encode_f4_r16_row(row: np.ndarray) -> np.ndarray:
    """Encode one row of ncols floats as Appendix-A f4.r16 bytes, padded to 64 B."""
    row = np.asarray(row, dtype=np.float32)
    n = row.size
    assert n % 16 == 0
    blocks = row.reshape(-1, 16)
    amax = np.abs(blocks).max(axis=1)
    scale = np.where(amax > 0, amax / 6.0, 1.0).astype(np.float32)
    scode = quant_e4m3(scale)
    sdec = decode_e4m3(scode)
    codes = quant_e2m1(blocks / sdec[:, None]).reshape(-1)
    packed = (codes[0::2] | (codes[1::2] << 4)).astype(np.uint8)
    out = np.zeros(pad64(n // 2 + n // 16), dtype=np.uint8)
    out[: n // 2] = packed
    out[n // 2: n // 2 + n // 16] = scode
    return out
