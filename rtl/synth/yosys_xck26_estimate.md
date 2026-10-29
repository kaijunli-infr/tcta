# Yosys synth_xilinx estimate of tcta_core (XCK26 / UltraScale+ fabric)

Per-module out-of-context synthesis (`rtl/synth/yosys_ooc.py`), the hierarchy totals weighted by the
instance counts. `own` = the module without its children; `total` = with them. No placement, no timing.

## Core total

| cell | count | KV260 budget | used |
|---|---|---|---|
| LUT | 438538 | 117120 | 374.4 % |
| MUXF | 0 |  |  |
| FF | 62604 | 234240 | 26.7 % |
| CARRY | 21356 |  |  |
| DSP | 429 | 1248 | 34.4 % |
| BRAM36 | 192 | 144 | 133.3 % |
| BRAM18 | 214 |  |  |
| LUTRAM | 159 |  |  |
| SRL | 0 |  |  |
| other | 7421 |  |  |

other cells: INV 7421

## Per module

| module | instances | own LUT | own MUXF | own FF | own CARRY | own DSP | own BRAM36 | total LUT | total MUXF | total FF | total CARRY | total DSP | total BRAM36 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| tcta_core | 1 | 8361 | 0 | 107 | 19 | 0 | 0 | 438538 | 0 | 62604 | 21356 | 429 | 192 |
| mma_unit | 1 | 23124 | 0 | 6699 | 1300 | 54 | 0 | 290180 | 0 | 32299 | 16212 | 342 | 0 |
| mma_pe | 256 | 33 | 0 | 100 | 0 | 0 | 0 | 929 | 0 | 100 | 52 | 1 | 0 |
| f32_add | 312 | 643 | 0 | 0 | 37 | 0 | 0 | 643 | 0 | 0 | 37 | 0 | 0 |
| f32_mul_bf16 | 256 | 253 | 0 | 0 | 15 | 1 | 0 | 253 | 0 | 0 | 15 | 1 | 0 |
| vec_unit | 1 | 18952 | 0 | 4852 | 1105 | 2 | 0 | 53714 | 0 | 4852 | 3005 | 50 | 0 |
| dma_load | 1 | 5505 | 0 | 1654 | 316 | 11 | 0 | 18082 | 0 | 1654 | 1176 | 27 | 0 |
| f32_mul | 32 | 541 | 0 | 0 | 26 | 2 | 0 | 541 | 0 | 0 | 26 | 2 | 0 |
| queue | 5 | 3184 | 0 | 3969 | 18 | 0 | 0 | 3184 | 0 | 3969 | 18 | 0 | 0 |
| dequant_elem | 16 | 147 | 0 | 0 | 0 | 0 | 0 | 747 | 0 | 0 | 45 | 1 | 0 |
| scratchpad_VEC | 1 | 10448 | 0 | 109 | 0 | 0 | 0 | 10448 | 0 | 109 | 0 | 0 | 0 |
| sfu_lane | 4 | 2477 | 0 | 0 | 108 | 4 | 0 | 2477 | 0 | 0 | 108 | 4 | 0 |
| scratchpad_ACC | 1 | 9687 | 0 | 77 | 0 | 0 | 96 | 9687 | 0 | 77 | 0 | 0 | 96 |
| scratchpad_STATE | 1 | 9676 | 0 | 13 | 0 | 0 | 32 | 9676 | 0 | 13 | 0 | 0 | 32 |
| dma_store | 1 | 2096 | 0 | 1133 | 162 | 7 | 0 | 7057 | 0 | 1133 | 654 | 7 | 0 |
| f32_mul_e4m3 | 16 | 385 | 0 | 0 | 19 | 1 | 0 | 385 | 0 | 0 | 19 | 1 | 0 |
| scratchpad_WGT | 1 | 5560 | 0 | 13 | 0 | 0 | 64 | 5560 | 0 | 13 | 0 | 0 | 64 |
| scratchpad_INP | 1 | 5418 | 0 | 13 | 0 | 0 | 0 | 5418 | 0 | 13 | 0 | 0 | 0 |
| f32_to_i32_rne | 16 | 271 | 0 | 0 | 22 | 0 | 0 | 271 | 0 | 0 | 22 | 0 | 0 |
| i32_to_f32 | 18 | 191 | 0 | 0 | 18 | 0 | 0 | 191 | 0 | 0 | 18 | 0 | 0 |
| front_end | 1 | 2872 | 0 | 1812 | 183 | 3 | 0 | 2872 | 0 | 1812 | 183 | 3 | 0 |
| f32_to_bf16 | 64 | 24 | 0 | 0 | 8 | 0 | 0 | 24 | 0 | 0 | 8 | 0 | 0 |
| tile_meta | 1 | 1343 | 0 | 609 | 0 | 0 | 0 | 1343 | 0 | 609 | 0 | 0 | 0 |
| f32_to_int_trunc | 2 | 241 | 0 | 0 | 12 | 0 | 0 | 241 | 0 | 0 | 12 | 0 | 0 |
| token_net | 1 | 67 | 0 | 0 | 0 | 0 | 0 | 220 | 0 | 68 | 17 | 0 | 0 |
| token_fifo | 17 | 9 | 0 | 4 | 1 | 0 | 0 | 9 | 0 | 4 | 1 | 0 | 0 |
| f32_max | 0 | 123 | 0 | 0 | 19 | 0 | 0 | 123 | 0 | 0 | 19 | 0 | 0 |

## Children per module (instance counts from the blackbox cells)

- tcta_core: dma_load ×1, dma_store ×1, front_end ×1, mma_unit ×1, queue ×5, scratchpad_INP ×1, scratchpad_WGT ×1, scratchpad_ACC ×1, scratchpad_VEC ×1, scratchpad_STATE ×1, tile_meta ×1, token_net ×1, vec_unit ×1
- mma_unit: f32_add ×32, f32_mul ×16, mma_pe ×256
- mma_pe: f32_add ×1, f32_mul_bf16 ×1
- vec_unit: f32_add ×24, f32_mul ×16, f32_to_bf16 ×16, i32_to_f32 ×2, sfu_lane ×4
- dma_load: dequant_elem ×16, f32_to_bf16 ×16, f32_to_int_trunc ×1
- dequant_elem: f32_mul_e4m3 ×1, f32_to_bf16 ×1, i32_to_f32 ×1
- dma_store: f32_to_bf16 ×16, f32_to_i32_rne ×16, f32_to_int_trunc ×1
- token_net: token_fifo ×17

## How to read it

- `synth_xilinx -family xcup -nowidelut -noiopad -noclkbuf`, Yosys of the OSS CAD Suite with the slang
  frontend; each module once, children as blackboxes, so cross-module optimization (constant ports,
  unused outputs) is not applied: an upper bound per module, a straight sum for the core.
- Block RAM: Yosys maps each scratchpad bank as a true-dual-port RAM but does not merge a port's read
  with its write, so it uses twice the bit-minimum (the five scratchpads hold 4.4 Mbit = 122 RAMB36,
  one RAMB18 counting as half); Vivado merges them. LUT-RAM cells (RAM32M16 etc.) are listed
  separately and are not in the LUT column.
- Budget: K26 SOM (XCK26-SFVC784-2LV-C) 117 120 LUT6, 234 240 FF, 1 248 DSP48E2, 144 RAMB36 (5.1 Mbit),
  64 URAM288 (unused by this flow). No timing: Fmax needs the Vivado run (vivado_ooc.tcl).
