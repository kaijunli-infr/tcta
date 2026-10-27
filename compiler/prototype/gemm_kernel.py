"""Partial-tile GEMM in TileLang (Gate 1, GEMM family).

Fixture `fixtures/gemm_partial`: C[50, 96] = A[50, 100] @ B[100, 96] inside 64 x 128 / 128 x 128
bf16 tiles. The kernel copies the valid sub-rectangles into one INP and one WGT tile, runs one
`T.gemm`, and stores the [50, 96] accumulator; D4 maps the sliced copies to `LOAD.2D` with
`rows_valid`/`cols_valid`, the gemm to one `MMA` on the partial operands (ISA §6.3: the array
contracts over `A.valid[1]` and writes `A.valid[0] x B.valid[1]`), and the store to `STORE.2D`.
Token graph: LOAD -> MMA (two tiles), MMA -> STORE (ISA §8.1 "plain GEMM output").
"""

import os
import sys

import numpy as np
import tilelang.language as T

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "ref"))
import common  # noqa: E402

M, K, N = 64, 128, 128
M_VALID, K_VALID, N_VALID = 50, 100, 96

SCOPE_MAP = {"A_s": "tcta.inp", "B_s": "tcta.wgt", "C_acc": "tcta.acc"}
DRAM = {"A": 0, "B": 64 * 1024, "C": 128 * 1024}
DRAM_BYTES = 256 * 1024


def build_gemm_kernel(m_valid: int = M_VALID, k_valid: int = K_VALID, n_valid: int = N_VALID, dtype: str = "float16"):
    accum = "float32"

    @T.prim_func
    def main(
        A: T.Tensor([M, K], dtype),
        B: T.Tensor([K, N], dtype),
        C: T.Tensor([m_valid, n_valid], accum),
    ):
        with T.Kernel(1, threads=128) as (bx,):
            A_s = T.alloc_shared([m_valid, k_valid], dtype)
            B_s = T.alloc_shared([k_valid, n_valid], dtype)
            C_acc = T.alloc_fragment([m_valid, n_valid], accum)
            T.copy(A[0:m_valid, 0:k_valid], A_s)
            T.copy(B[0:k_valid, 0:n_valid], B_s)
            T.clear(C_acc)
            T.gemm(A_s, B_s, C_acc)
            T.copy(C_acc, C)

    return main


def load_dram(m, fx: dict) -> None:
    m.dram_write(DRAM["A"], common.bf16_bytes(fx["A"]).ravel())
    m.dram_write(DRAM["B"], common.bf16_bytes(fx["B"]).ravel())


def read_c(m, fx: dict) -> np.ndarray:
    r, c = fx["C"].shape
    return m.dram[DRAM["C"]: DRAM["C"] + 4 * r * c].view(np.float32).reshape(r, c).copy()
