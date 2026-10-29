# rtl/test: the Verilator conformance harness (phase 7, J3–J5)

```
make build                     # Verilate tb_top + the core (obj_dir_tc0; TOKEN_CHECK=1 -> obj_dir_tc1, FORMAL=1 -> *_formal)
make run PROG=gemm             # run vectors/gemm and compare with the simulator's expectations
make all                       # the ten programs below (neg_untokened with TOKEN_CHECK=1)
make all FORMAL=1              # the same with the `ifdef FORMAL properties compiled as runtime assertions
make lint                      # verilator --lint-only -Wall on the core
python gen_vectors.py [names]  # (re)generate vectors/<name>/ from the simulator
python calibrate.py [names]    # cycle model vs rtl_trace.txt per instruction (docs/calibration-report.md)
```

Interpreter: `~/Projects/tilelang-cutoff/.venv/bin/python` (`PYTHON=` in the Makefile).

## Programs (`gen_vectors.py`)

| name | what | check |
|---|---|---|
| `dma_copy` | LOAD.2D f32 64×64 into VEC, STORE.2D back | bit-exact DRAM |
| `mma` | LOAD INP/WGT, MMA plain, accumulate, A from a converted ACC tile (plain, accumulate), VCVT in place, two f32 STOREs | bit-exact |
| `vec` | the FA VEC chain on one 64×128 tile without SFU ops (VFILL, VROWMAX, VMAX, VSUB vec, VMUL, VADD, VROWSUM, VSCALE_ACC, VNEG, VCOPY, VCVT), two STOREs | bit-exact |
| `gemm` | the compiler's GEMM program (`model/sweeps.fam_gemm`, `gemm_partial` fixture) | bit-exact |
| `fa` | the compiler's flash-attention program (`fam_fa`, `flash_attn_gqa`, 4 KV tiles) | §9 row-max tolerance 2⁻⁷ against the fixture (SFU paths), `ref.npy` |
| `loop` | front-end micro-test: VCOUNT → LOOP r, static nested loops, loop-variant `row0` (extension words) | bit-exact; its `FE DISP` trace lines feed `calibrate.py` |
| `gather` | LOAD.IDX of f32 rows at an odd column offset (two -1 rows), of bf16 rows and of f4.r16 rows at an odd, non-16-aligned offset (both MMA B operands), STORE.IDX, an i32 STORE.2D (phase 8 K0.1) | bit-exact |
| `neg_untokened` | second LOAD without a PUSH, the MMA reads it | fault `UNTOKENED_READ MMA 4` (needs `TOKEN_CHECK=1`) |
| `neg_leak` | a PUSH nobody pops, then FINISH | fault `TOKEN_LEAK` |
| `neg_deadlock` | a POP nobody pushes (`t_deadlock` 200) | fault `TOKEN_TIMEOUT` |

The three negative tests are those of `tests/test_sim.py`. TOKEN_LEAK and
TOKEN_TIMEOUT are hardware faults of ISA §10 (the core checks every token FIFO
at FINISH and counts idle cycles on a blocked POP). UNTOKENED_READ /
UNTOKENED_WRITE are not hardware faults (§1: an untokened read returns stale
data); `tcta_core` instantiates `token_check.sv`, a debug twin of the
simulator's `_note_read/_note_write` checks, when built with `TOKEN_CHECK=1`.
It is fed by every granted scratchpad access and by the operand tiles an
instruction names at issue, so it faults at the same instruction as the
simulator (pc 4) even though the MMA, seeing a zero-shaped WGT tile, never
reads its data.

## Vector directory

| file | content |
|---|---|
| `stream.hex` | the §12 instruction words (`encode.py`), one 128-bit hex word per line, extension words included |
| `desc.hex` | the descriptor table, one 384-bit entry per line (`desc[i][63:0]` = base) |
| `dram.hex`, `expect_dram.hex` | initial DRAM and the simulator's DRAM after the run: sparse `$readmemh`, the touched 4 KB pages (`pages.txt`) |
| `expect.json` | status (ok / fault code, queue, pc), `n_words`, `n_desc`, `t_deadlock`, the pages, the checks |
| `sim_trace.txt` | the simulator's step trace `step queue op pc` |
| `rtl_trace.txt` | written by the run: `cycle queue SYNC\|ISSUE\|DONE op pc` per queue event and `cycle FE DISP op pc` per front-end dispatch |
| `status.txt`, `dram_out.hex`, `sp_dump.txt` | written by the run: `ok <cycle>` or `fault <code> <queue> <pc> <cycle>`, the DRAM pages, every scratchpad word |

`compare.py` checks the status, that the RTL retired exactly the simulator's
multiset of (queue, op, pc) events, and the DRAM pages (bit-exact, or the
row-max tolerance for FA; SFU paths differ from the simulator in the last
bits and are reported as a count).

## Testbench (`tb_top.sv`, `tb_main.cpp`)

DRAM is a 64 MB byte array with a fixed read latency (`+MEMLAT`, default 24
cycles, the value `tcta.timing.KV260.dram_latency_cycles` carries) and
16-byte beats at any alignment, one request per cycle, in-order responses;
writes take effect in the cycle they are presented. Plusargs: `+VEC=dir`,
`+OUT=dir`, `+NWORDS`, `+NDESC`, `+TDEADLOCK`, `+MEMLAT`, `+MAXCYC`.

`fp/` holds the FP library unit test (`check_fp.py N`, N ≤ 200000 random
operand pairs against NumPy float32: f32_add, f32_mul, f32_max, to_bf16,
f32_to_i32_rne; 0 mismatches).
