# Phase 8 amendment: fit the core before fixing its speed

The phase-7 Yosys estimate puts the v0 core at 1.30 M LUTs against the
KV260's 117 k. Phase 8's K1–K4 (speed) are postponed behind a new **K0:
fit**, because a core that does not fit cannot be timed on the board and the
fixes change the same modules. Formal (J6) passed; nothing here touches it.

## Where the 11× comes from, and what each is

| Block | LUTs | Cause | Kind |
|---|---|---|---|
| `dma_store` row buffer | 767 k | 2 KB byte-addressed buffer: 16 k flops each behind a 64-source write mux | implementation bug |
| `mma_unit` accumulators | ~165 k of 290 k | 256 single-cycle IEEE FP32 adders at 643 LUTs each; UltraScale+ has no FP DSP | architecture (numerics) |
| `vec_unit` | 122 k | 95 unshared adders/multipliers for a 16-lane unit | implementation |
| KV260 budget | 117 k | | |

## K0 deliverables (each its own commit with the per-module Yosys number before/after)

**K0.1 `dma_store`: slice-addressed row buffer.** The row buffer is written
in 16-byte slices at slice-aligned offsets (that is what the VEC/ACC read
port produces) and read out as 16-byte beats; implement it as a small
two-port RAM or a 128-entry × 128-bit register file with a one-hot slice
write, never a byte mux. Target < 5 k LUTs for the whole engine. `dma_load`
gets the same review (its row buffer is the mirror).

**K0.2 `vec_unit`: one ALU per lane.** 16 lanes = 16 adders + 16 multipliers
+ the SFU pipe, op-muxed; comparators (`VMAX`, `VROWMAX`) share the adder's
subtract path. Multipliers on DSP48E2. Target < 25 k LUTs.

**K0.3 `mma_unit`: the accumulator decision.** Two options; pick one, the
other stays as a parameter.

- **(A) Keep IEEE FP32 accumulate, cut the rate.** Pipelined FP32 adder
  (~400 LUTs) shared by two cells: 128 adders, the K step takes 2 cycles,
  the array delivers 128 MAC/cycle. ~55 k LUTs + 256 DSPs for the BF16
  products. Bit-exact to the simulator as now. Half the planned throughput;
  the cycle model's `mma_issue_cycles` doubles.
- **(B) Wide fixed-point accumulate per cell** (recommended). Each cell
  keeps a 48-bit accumulator in a DSP48E2 plus a 6-bit exponent; the BF16×BF16
  product (exact, 16-bit mantissa) is aligned into the accumulator's window
  by a barrel shift, and the accumulator renormalizes when a product's
  exponent exceeds the window. Full rate, 256 DSPs (products + accumulate on
  the same DSP in MAC mode where alignment allows), ~150 LUTs per cell
  (~40 k). Numerically it is *more* accurate than FP32 (no rounding per add
  within the window), but it is not bit-exact to NumPy f32. Conformance is
  therefore by ISA §9 tolerance, which was written for this case, and the
  simulator gets an `acc_mode="fixed48"` that models the window exactly so
  RTL stays bit-exact to *that*. ASIC may keep IEEE FP32; §9 makes both
  conformant. Requires **erratum E5** (accumulation format is
  implementation-defined within §9; simulator exposes both).

Decision rule: (B) unless the simulator mode cannot be made bit-exact to the
RTL on GEMM and FA within a day; then (A).

**K0.4 Fit report.** Per-module table after K0.1–K0.3 against the 117 k /
1 248 DSP / 144 BRAM budget, with ≥ 15 % LUT headroom for the AXI/DDR glue
of the FPGA phase. If 16×16 still does not fit with headroom, the fallback
is an 8×8 array parameter (same RTL), stated in the report; the KV260 is a
flow-proof part, not the product.

Then K1–K8 as written, with K1's DMA row pipelining built on the K0.1 buffer.

## Not changed

The programs, the ISA semantics (E5 is a numerics-format erratum only), the
formal properties.
