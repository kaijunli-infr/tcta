# Task: phase 7 — RTL core v0 and the Verilator conformance harness (toward Gate 2)

Prerequisites: Gate 1 frozen (ISA v0.1.5). This phase starts the plan's RTL
track: one core of the frozen ISA in RTL, run under Verilator against the
*same instruction streams* the Python simulator runs, with the cycle model
calibrated from it. Gate 2 in the plan is "all four kernels on Verilator,
cycle model within 10 %"; this phase gets GEMM and FA there and sets up the
harness so MoE/MLA/KDA follow mechanically.

## 0. Decisions (apply)

| Item | Value |
|---|---|
| Language | **Chisel** (Scala/sbt), structure borrowed from Gemmini: decoupled load/execute/store controllers over scratchpads. Kevin may override to SystemVerilog before you start; if so, same module boundaries. |
| Scope | One core, v1 KV260 descriptor (16×16 BF16 array, FP32 accumulate, INP 2 / WGT 4 / ACC 2+wide / VEC 16 / STATE 1, FIFO 8, 1 read + 1 write DMA channel, SFU 4 lanes). Config C is a parameter set, not a second design. |
| Instruction input | The simulator's `Program` serialized to a binary stream by a new `sim/tcta/encode.py` implementing ISA §12 (128-bit words, descriptor table for regions). The RTL decodes that; the Python side is the single source of the encoding. |
| Memory | A simple AXI-lite-free model: the testbench owns a byte-addressable DRAM (`dram.bin`) with a fixed-latency, fixed-bandwidth port matching `DeviceTiming` KV260 defaults; real AXI comes with the FPGA bring-up, not here. |
| Numerics | Bit-exact against the Python simulator for every instruction except SFU ops, which must meet ISA §9 (≤ 2 ulp BF16); BF16 rounding is RNE as in `dtypes.to_bf16`. |

## 1. Deliverables

| # | Deliverable | Definition of done |
|---|---|---|
| J1 | `sim/tcta/encode.py` + ISA §12 made normative | `encode(program) -> bytes` and `decode(bytes) -> Program` round-trip every program in `compiler/prototype/dumps/`; §12 rewritten with the final field layout and the descriptor-table format; a `tests/test_encode.py` |
| J2 | `rtl/` Chisel project | Modules: `FrontEnd` (decode, LOOP stack, registers, VCOUNT wait), five `Queue`s with token FIFOs (`TokenFifo(depth)` per edge), `Scratchpads` (banked SRAMs with the §11 slot/bank map), `Dma` (2D/3D stride engine, `LOAD.IDX` row gather with the -1 skip and range fault, dequant for `bf16/f32/i32/f8.t/f4.b32/f4.r16`), `MmaUnit` (16×16 systolic, BF16×BF16→FP32, A from INP or ACC, B transposed form, wide, row0, masks add/mul), `VecUnit` (every §5.3 op; SFU exp2/rcp/log2/sigmoid as a separate pipeline), `Comm` stubbed. Status word and fault codes of §10. |
| J3 | `rtl/test/` Verilator harness | Loads `dram.bin` + instruction stream, runs to FINISH or fault, dumps DRAM and the scratchpads, and a per-instruction `(pc, queue, issue_cycle, complete_cycle)` trace in the cycle model's trace format |
| J4 | Conformance | GEMM and FA programs (from `dump_programs.py`) run on Verilator; DRAM output bit-exact to the Python simulator for GEMM, within §9 for FA; token faults reproduced: the three negative tests of `test_sim.py` (untokened read, token leak, deadlock timeout) fault on RTL with the same codes |
| J5 | Calibration | `DeviceTiming` KV260 values replaced by measured ones from the J3 trace: array latency and issue rate, DMA bytes/cycle and latency, VEC/SFU per-op cycles, loop overhead, VCOUNT→LOOP latency, PUSH visibility latency. Cycle model vs RTL total cycles for GEMM and FA within 10 %; report the per-parameter deltas in `docs/calibration-report.md` and list what still differs |
| J6 | Formal | SymbiYosys bounded model check on `TokenFifo` and the bank arbiter: no lost or duplicated token, no two writers on one bank per cycle, FIFO full blocks the producer and never drops. Depth-8 FIFO, bound ≥ 24 cycles |
| J7 | Phase 7 section in the verdict | PASS/FAIL per §3; synthesis sanity: Vivado out-of-context synthesis of the core for the KV260 part with LUT/DSP/BRAM/URAM counts and the achieved Fmax, against the plan's 200 MHz and the resource budget from the plan review (URAM by port width) |

J1 first (it is a Python task and unblocks everything), then J2 bottom-up:
`TokenFifo` + J6 formal, `Scratchpads`, `Dma`, `MmaUnit`, `VecUnit`, then
`FrontEnd` and the queues, then J3/J4.

## 2. Rules that keep the RTL honest

- No behaviour that the ISA leaves to the microarchitecture may be assumed
  by the compiler: the harness must pass with the Python simulator's
  programs unchanged. If the RTL needs the program changed, that is an ISA
  erratum (§16), not a program fix.
- Token FIFOs are the only inter-queue state. No bypass paths, no
  "optimizations" that read a tile before its POP; the three negative tests
  exist to catch exactly this.
- Dequant on ingress is part of `Dma`, not of `MmaUnit` or `VecUnit`.
- `PUSH`/`POP` must not consume an issue slot (§8.5): a queue retires leading
  syncs in the same cycle as its next work instruction. Measure this (J5).
- Every planning value in `DeviceTiming` that the RTL pins down is replaced,
  never "tuned to match".

## 3. Decision rule

- **PASS**: J1–J7; GEMM bit-exact, FA within §9; negative tests fault with
  the right codes; formal clean; cycle model within 10 % after calibration;
  synthesis closes ≥ 150 MHz on the KV260 part with the core inside the
  resource budget.
- **PASS-WITH-CAVEATS**: within 10 % only after a `DeviceTiming` rule
  change (say which), or Fmax between 100 and 150 MHz, or resources over
  budget by < 25 % (name the block).
- **FAIL**: a program must change to run on RTL (ISA erratum needed), a
  formal counterexample on `TokenFifo`, or the harness cannot reproduce a
  token fault.

## 4. Not in scope

MoE/MLA/KDA on RTL (phase 8, mechanical once J3 works), multi-core, COMM,
AXI/DDR integration, the FPGA board itself, power, ASIC synthesis.
