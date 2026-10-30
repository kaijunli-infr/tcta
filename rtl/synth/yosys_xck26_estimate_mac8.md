# Yosys synth_xilinx estimate of tcta_core with the 8 x 8 MAC array (TCTA_MAC8, phase 8 L5 measurement; XCK26)

Per-module out-of-context synthesis (`rtl/synth/yosys_ooc.py`), the hierarchy totals weighted by the
instance counts. `own` = the module without its children; `total` = with them. No placement, no timing.

Only `mma_unit`, `mma_opbuf` and `mma_opbuf_a` were re-synthesised with `--define TCTA_MAC8` (`--out rtl/synth/ooc_mac8`);
every other module is the 16-lane core's result (`yosys_xck26_estimate.md`), which the define does not touch.
Phase 8 L5 measurement (docs/phase8-k0-fit-report.md §4c): core 162 496 -> 145 051 LUT-eq, `mma_unit` 53 199 -> 35 754.

## Core total

| cell | count | KV260 budget | used |
|---|---|---|---|
| LUT | 138763 | 117120 | 118.5 % |
| MUXF | 0 |  |  |
| FF | 23152 | 234240 | 9.9 % |
| CARRY | 8044 |  |  |
| DSP | 238 | 1248 | 19.1 % |
| BRAM36 | 192 | 144 | 133.3 % |
| BRAM18 | 214 |  |  |
| LUTRAM | 786 |  |  |
| SRL | 0 |  |  |
| other | 4813 |  |  |
| **LUT-eq** (LUT + LUT-RAM as LUT6) | **145051** | 117120 | **123.8 %** |

other cells: INV 4813

## Per module

| module | instances | own LUT | own MUXF | own FF | own CARRY | own DSP | own BRAM36 | own LUTRAM | own LUT-eq | total LUT | total MUXF | total FF | total CARRY | total DSP | total BRAM36 | total LUTRAM | total LUT-eq |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| tcta_core | 1 | 3592 | 0 | 107 | 19 | 0 | 0 | 0 | 3592 | 138763 | 0 | 23152 | 8044 | 238 | 192 | 786 | 145051 |
| vec_unit | 1 | 18819 | 0 | 4862 | 1106 | 2 | 0 | 40 | 19139 | 53581 | 0 | 4862 | 3006 | 50 | 0 | 40 | 53901 |
| mma_unit | 1 | 14597 | 0 | 6174 | 1759 | 9 | 0 | 40 | 14917 | 32170 | 0 | 11422 | 2567 | 153 | 0 | 448 | 35754 |
| f32_add | 32 | 643 | 0 | 0 | 37 | 0 | 0 | 0 | 643 | 643 | 0 | 0 | 37 | 0 | 0 | 0 | 643 |
| dma_load | 1 | 5288 | 0 | 1439 | 364 | 9 | 0 | 70 | 5848 | 17865 | 0 | 1439 | 1224 | 25 | 0 | 70 | 18425 |
| f32_mul | 24 | 541 | 0 | 0 | 26 | 2 | 0 | 0 | 541 | 541 | 0 | 0 | 26 | 2 | 0 | 0 | 541 |
| dequant_elem | 16 | 147 | 0 | 0 | 0 | 0 | 0 | 0 | 147 | 747 | 0 | 0 | 45 | 1 | 0 | 0 | 747 |
| dma_store | 1 | 6073 | 0 | 2244 | 451 | 7 | 0 | 48 | 6457 | 11034 | 0 | 2244 | 943 | 7 | 0 | 48 | 11418 |
| sfu_lane | 4 | 2477 | 0 | 0 | 108 | 4 | 0 | 0 | 2477 | 2477 | 0 | 0 | 108 | 4 | 0 | 0 | 2477 |
| f32_mul_e4m3 | 16 | 385 | 0 | 0 | 19 | 1 | 0 | 0 | 385 | 385 | 0 | 0 | 19 | 1 | 0 | 0 | 385 |
| scratchpad_ACC | 1 | 5077 | 0 | 11 | 0 | 0 | 96 | 0 | 5077 | 5077 | 0 | 11 | 0 | 0 | 96 | 0 | 5077 |
| scratchpad_VEC | 1 | 5019 | 0 | 13 | 0 | 0 | 0 | 0 | 5019 | 5019 | 0 | 13 | 0 | 0 | 0 | 0 | 5019 |
| fix_to_f32 | 8 | 619 | 0 | 0 | 38 | 0 | 0 | 0 | 619 | 619 | 0 | 0 | 38 | 0 | 0 | 0 | 619 |
| f32_to_i32_rne | 16 | 271 | 0 | 0 | 22 | 0 | 0 | 0 | 271 | 271 | 0 | 0 | 22 | 0 | 0 | 0 | 271 |
| i32_to_f32 | 18 | 191 | 0 | 0 | 18 | 0 | 0 | 0 | 191 | 191 | 0 | 0 | 18 | 0 | 0 | 0 | 191 |
| front_end | 1 | 2872 | 0 | 1812 | 183 | 3 | 0 | 0 | 2872 | 2872 | 0 | 1812 | 183 | 3 | 0 | 0 | 2872 |
| scratchpad_STATE | 1 | 2839 | 0 | 3 | 0 | 0 | 32 | 0 | 2839 | 2839 | 0 | 3 | 0 | 0 | 32 | 0 | 2839 |
| mma_opbuf | 1 | 1723 | 0 | 0 | 0 | 0 | 0 | 256 | 3771 | 1723 | 0 | 0 | 0 | 0 | 0 | 256 | 3771 |
| queue | 5 | 316 | 0 | 112 | 17 | 0 | 0 | 36 | 604 | 316 | 0 | 112 | 17 | 0 | 0 | 36 | 604 |
| f32_to_bf16 | 64 | 24 | 0 | 0 | 8 | 0 | 0 | 0 | 24 | 24 | 0 | 0 | 8 | 0 | 0 | 0 | 24 |
| tile_meta | 1 | 1343 | 0 | 609 | 0 | 0 | 0 | 0 | 1343 | 1343 | 0 | 609 | 0 | 0 | 0 | 0 | 1343 |
| mma_opbuf_a | 1 | 978 | 0 | 0 | 0 | 0 | 0 | 152 | 2194 | 978 | 0 | 0 | 0 | 0 | 0 | 152 | 2194 |
| scratchpad_WGT | 1 | 804 | 0 | 1 | 0 | 0 | 64 | 0 | 804 | 804 | 0 | 1 | 0 | 0 | 64 | 0 | 804 |
| scratchpad_INP | 1 | 767 | 0 | 1 | 0 | 0 | 0 | 0 | 767 | 767 | 0 | 1 | 0 | 0 | 0 | 0 | 767 |
| f32_to_int_trunc | 2 | 241 | 0 | 0 | 12 | 0 | 0 | 0 | 241 | 241 | 0 | 0 | 12 | 0 | 0 | 0 | 241 |
| mma_cell | 64 | 7 | 0 | 82 | 0 | 2 | 0 | 0 | 7 | 7 | 0 | 82 | 0 | 2 | 0 | 0 | 7 |
| token_net | 1 | 67 | 0 | 0 | 0 | 0 | 0 | 0 | 67 | 220 | 0 | 68 | 17 | 0 | 0 | 0 | 220 |
| token_fifo | 17 | 9 | 0 | 4 | 1 | 0 | 0 | 0 | 9 | 9 | 0 | 4 | 1 | 0 | 0 | 0 | 9 |
| mma_pe | 0 | 33 | 0 | 102 | 0 | 0 | 0 | 0 | 33 | 929 | 0 | 102 | 52 | 1 | 0 | 0 | 929 |
| f32_mul_bf16 | 0 | 253 | 0 | 0 | 15 | 1 | 0 | 0 | 253 | 253 | 0 | 0 | 15 | 1 | 0 | 0 | 253 |
| f32_max | 0 | 123 | 0 | 0 | 19 | 0 | 0 | 0 | 123 | 123 | 0 | 0 | 19 | 0 | 0 | 0 | 123 |

## Children per module (instance counts from the blackbox cells)

- tcta_core: dma_load ×1, dma_store ×1, front_end ×1, mma_unit ×1, queue ×5, scratchpad_INP ×1, scratchpad_WGT ×1, scratchpad_ACC ×1, scratchpad_VEC ×1, scratchpad_STATE ×1, tile_meta ×1, token_net ×1, vec_unit ×1
- vec_unit: f32_add ×24, f32_mul ×16, f32_to_bf16 ×16, i32_to_f32 ×2, sfu_lane ×4
- mma_unit: f32_add ×8, f32_mul ×8, fix_to_f32 ×8, mma_cell ×64, mma_opbuf_a ×1, mma_opbuf ×1
- dma_load: dequant_elem ×16, f32_to_bf16 ×16, f32_to_int_trunc ×1
- dequant_elem: f32_mul_e4m3 ×1, f32_to_bf16 ×1, i32_to_f32 ×1
- dma_store: f32_to_bf16 ×16, f32_to_i32_rne ×16, f32_to_int_trunc ×1
- token_net: token_fifo ×17
- mma_pe: f32_add ×1, f32_mul_bf16 ×1

## How to read it

- `synth_xilinx -family xcup -nowidelut -noiopad -noclkbuf`, Yosys of the OSS CAD Suite with the slang
  frontend; each module once, children as blackboxes, so cross-module optimization (constant ports,
  unused outputs) is not applied: an upper bound per module, a straight sum for the core.
- Block RAM: Yosys maps each scratchpad bank as a true-dual-port RAM but does not merge a port's read
  with its write, so it uses twice the bit-minimum (the five scratchpads hold 4.4 Mbit = 122 RAMB36,
  one RAMB18 counting as half); Vivado merges them. LUT-RAM cells (RAM32M16 etc.) are listed
  separately and are not in the LUT column; LUT-eq adds them as LUT6 (RAM32M16 = 8, RAM32M = 4, ...).
- Budget: K26 SOM (XCK26-SFVC784-2LV-C) 117 120 LUT6, 234 240 FF, 1 248 DSP48E2, 144 RAMB36 (5.1 Mbit),
  64 URAM288 (unused by this flow). No timing: Fmax needs the Vivado run (vivado_ooc.tcl).
