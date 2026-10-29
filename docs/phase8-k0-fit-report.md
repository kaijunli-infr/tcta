# Phase 8 K0.4: fit report (v1 core after K0.1–K0.3, Yosys estimate)

Per-module `synth_xilinx` estimate of `tcta_core` (`rtl/synth/yosys_ooc.py`
→ `rtl/synth/yosys_xck26_estimate.md`, OSS CAD Suite Yosys 0.70, UltraScale+
cells, children blackboxed and summed, no placement, no timing) against the
KV260's XCK26: **117 120 LUT, 234 240 FF, 1 248 DSP48E2, 144 RAMB36, 64
URAM**. LUT-RAM cells are counted as LUTs (a `RAM32M16` occupies the eight
LUT6 of a SLICEM): `LUT-eq = LUT + 8 · RAM32M16`. The fit budget with the
required 15 % headroom for the AXI/DDR glue is **99 552 LUT-eq**.

## 1. Result

| | phase 7 (J7) | after K0.3 | KV260 | used |
|---|---|---|---|---|
| LUT | 1 296 593 | 200 006 | 117 120 | 171 % |
| LUT-RAM cells (RAM32M16) | 159 | 1 016 (= 8 128 LUT) | | |
| **LUT-eq** | **1 297 865** | **208 134** | **117 120** | **178 %**; 209 % of the 99.5 k fit budget |
| FF | 78 685 | 59 878 | 234 240 | 26 % |
| DSP48E2 | 546 | 624 | 1 248 | 50 % |
| RAMB36 (+ RAMB18 as ½) | 299 | 299 (192 + 214/2) | 144 | 208 % as mapped; the bit-minimum is 122 (85 %) |

**The 16×16 core does not fit the K26 with headroom: 2.1× the fit budget**
(6.2× better than phase 7's 11×). DSPs, flip-flops and, after port merging
or URAM, block RAM fit.

## 2. Per module

| module | LUT | RAM32M16 | LUT-eq | % of core | FF | DSP | RAMB36-eq | phase 7 LUT |
|---|---|---|---|---|---|---|---|---|
| `mma_unit` (16×16 fixed-point array, operand buffers, converter, writeback) | 51 781 | 896 | 58 949 | 28 % | 29 563 | 537 | | 290 180 |
| `vec_unit` (16 lanes, 4 SFU lanes, shared reducer) | 53 581 | 40 | 53 901 | 26 % | 4 862 | 50 | | 122 507 |
| scratchpads ×5 (16 banks each, 2 read + 1 write port, 7 clients) | 40 789 | | 40 789 | 20 % | 225 | | 288 | 40 789 |
| `dma_load` (beat-banked row buffer, 16 dequant paths) | 18 082 | 40 | 18 402 | 9 % | 1 654 | 27 | | 46 758 |
| `queue` ×5 (16 × 128-bit instruction registers, two heads) | 15 920 | | 15 920 | 8 % | 19 845 | | | 15 920 |
| `tcta_core` glue (request routing, grant/rvalid fan-in) | 8 361 | | 8 361 | 4 % | 107 | | | 8 361 |
| `dma_store` (beat-banked row buffer, 16 i32 + 16 bf16 converters) | 7 057 | 40 | 7 377 | 4 % | 1 133 | 7 | | 767 643 |
| `front_end`, `tile_meta`, `token_net` | 4 435 | | 4 435 | 2 % | 2 489 | 3 | | 4 435 |
| **core** | **200 006** | **1 016** | **208 134** | | **59 878** | **624** | **299** | **1 296 593** |

Inside the two big ones:

- `mma_unit`: own logic 25 989 LUT (the 16 lane collectors — a one-hot 16:1
  mux of 99 bits per lane, ~8–12 k — the per-row / per-column operand
  shifters, the writeback muxes, the schedulers), 256 `mma_cell` × 7 LUT +
  2 DSP (1 792 LUT, 512 DSP), 16 `f32_add` 10 288, 8 `f32_mul` 4 328 + 16
  DSP, 8 `fix_to_f32` 4 952, operand buffers 4 432 LUT + 816 RAM32M16 (B:
  16 per-lane RAMs behind the XOR permutation, 512 cells; A: one 256-bit
  word with 16 read ports replicated by Yosys, 304 cells), result FIFO 80
  cells. 29.6 k FF, of which 256 × 82 in the cells (the DSP registers are
  inside the primitive and not counted).
- `vec_unit`: own 18 819 (operand muxes of the one-ALU-per-lane design, the
  sequencer, tag queue, write FIFO), 24 `f32_add` 15 432 (16 lanes + 8
  reducer), 16 `f32_mul` 8 656 + 32 DSP, 4 `sfu_lane` 9 908 + 16 DSP, 16
  `f32_to_bf16` 384, 2 `i32_to_f32` 382.

## 3. Block RAM

Yosys maps every scratchpad bank as a true-dual-port RAM without merging the
write with one read port, so it uses 2× the bit-minimum: INP 16, WGT 64, ACC
96, VEC 80, STATE 32 RAMB36-equivalents (288) for 4.4 Mbit whose minimum is
122 RAMB36. Vivado merges the ports (phase-7 finding); if it does not, ACC +
STATE (3.4 + 1.1 Mbit) go to URAM (64 × 288 Kbit on the part; 16 URAMs) and
the remaining 160 RAMB36-eq at Yosys's 2× fit the 144 after merging the
smaller pads. The 1 016 LUT-RAM cells (operand buffers, result FIFO, DMA row
buffers) could also move to RAMB18s (16 for the operand buffers), −8 k LUT-eq.

## 4. What would make it fit

Nothing below was built; the estimates are structural (cells × count).

| lever | LUT-eq saved | cost | kind |
|---|---|---|---|
| L1 instruction queues in LUT-RAM (16 × 128 bit, two read heads: 20 RAM32M16 ≈ 160 LUT-eq each instead of 3 184) | −13 k | none | implementation |
| L2 scratchpad clients per the ISA §3 access table (today every pad takes all 7 clients on 16 banks × 2 ports; INP/WGT need 3, ACC 5, VEC 4, STATE 3) | −15 k (est.) | none | implementation |
| L3 MMA writeback: 8 adders over two cycles for the accumulate / mask add (−5 k), 4 converter lanes (−2.5 k) | −7.5 k | +1 cycle per accumulated row; +2 per row when P < 64 | implementation |
| L4 operand buffers in RAMB18 | −8 k | 16 RAMB18 | implementation |
| **L1–L4 together** | **−43 k → ~165 k (141 % of budget, 166 % of fit budget)** | | |
| L5 8×8 array (the amendment's fallback): 64 cells, 8-lane collectors, half the operand buffers | −35 k → ~130 k | ¼ of the MACs (MMA 64×128×128 4 284 → ~16 500 cycles); **not a parameter change today**: `BANKS = MAC_COLS = VEC_LANES = 16` (`tcta_pkg.sv`, one row slice per access), so the writeback needs half-slice byte enables and the B buffer 8 lanes (1–2 days) | architecture |
| L6 8 VEC lanes (two cycles per slice) | −12 k → ~118 k | VEC chain 2×; FA's per-tile VEC chain 4.6 k → ~8 k cycles | architecture |
| L7 Vivado instead of Yosys: `-nowidelut` (no MUXF7/F8), no cross-module constant propagation, no LUT combining; mux-heavy logic typically maps 20–35 % smaller | −25 % | the x86 run | measurement |

**Statement of the fallback.** By this estimate the 8×8 array alone does
not make the core fit: with L1–L5 the core is ~130 k LUT-eq, 1.3× the fit
budget, because the lane-proportional logic outside the array (VEC lanes,
scratchpad ports, DMA converters, collectors) is sized by `BANKS = 16`, not
by the array. What fits with headroom is the **8-lane configuration**
(`BANKS = 8`: 8×8 array, 8 VEC lanes, 8-column slices, 8 banks per pad —
roughly half of everything lane-proportional: ≈ 105–110 k LUT-eq before
Vivado's gains, ≈ 80 k after), i.e. L5 + L6 + L1–L4, or the 16-lane core
with L1–L4 **if** Vivado's mapping comes in ≥ 30 % under Yosys (165 k ×
0.7 = 115 k, still above 99.5 k). The KV260 being a flow-proof part, the
recommendation is: run Vivado on the x86 machine first (`rtl/synth/vivado_ooc.tcl`)
to replace L7's assumption with a number, do L1–L4 in any case (no speed
cost, −43 k), and decide the lane width from the Vivado figure; the
16-lane core stays the planning machine (`tcta.timing.KV260`).

## 5. Reproduce

```
export PATH=~/.local/opt/oss-cad-suite/bin:$PATH
python3 rtl/synth/yosys_ooc.py -j 2          # ~20 min on the aarch64 box; per-module JSON in rtl/synth/ooc/
```
