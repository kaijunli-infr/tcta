# Yosys synth_xilinx estimate of tcta_core (XCK26 / UltraScale+ fabric)

Per-module out-of-context synthesis (`rtl/synth/yosys_ooc.py`), the hierarchy totals weighted by the
instance counts. `own` = the module without its children; `total` = with them. No placement, no timing.

## Core total

| cell | count | KV260 budget | used |
|---|---|---|---|
| LUT | 158578 | 117120 | 135.4 % |
| MUXF | 0 |  |  |
| FF | 43199 | 234240 | 18.4 % |
| CARRY | 8691 |  |  |
| DSP | 622 | 1248 | 49.8 % |
| BRAM36 | 192 | 144 | 133.3 % |
| BRAM18 | 214 |  |  |
| LUTRAM | 1235 |  |  |
| SRL | 0 |  |  |
| other | 5199 |  |  |
| **LUT-eq** (LUT + LUT-RAM as LUT6) | **168458** | 117120 | **143.8 %** |

other cells: INV 5199

## Per module

| module | instances | own LUT | own MUXF | own FF | own CARRY | own DSP | own BRAM36 | own LUTRAM | own LUT-eq | total LUT | total MUXF | total FF | total CARRY | total DSP | total BRAM36 | total LUTRAM | total LUT-eq |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| tcta_core | 1 | 3592 | 0 | 107 | 19 | 0 | 0 | 0 | 3592 | 158578 | 0 | 43199 | 8691 | 622 | 192 | 1235 | 168458 |
| vec_unit | 1 | 20846 | 0 | 5409 | 1299 | 2 | 0 | 41 | 21174 | 55608 | 0 | 5409 | 3199 | 50 | 0 | 41 | 55936 |
| mma_unit | 1 | 26308 | 0 | 9372 | 2213 | 9 | 0 | 80 | 26948 | 46956 | 0 | 30364 | 3021 | 537 | 0 | 896 | 54124 |
| f32_add | 32 | 643 | 0 | 0 | 37 | 0 | 0 | 0 | 643 | 643 | 0 | 0 | 37 | 0 | 0 | 0 | 643 |
| dma_load | 1 | 5288 | 0 | 1439 | 364 | 9 | 0 | 70 | 5848 | 17865 | 0 | 1439 | 1224 | 25 | 0 | 70 | 18425 |
| f32_mul | 24 | 541 | 0 | 0 | 26 | 2 | 0 | 0 | 541 | 541 | 0 | 0 | 26 | 2 | 0 | 0 | 541 |
| dequant_elem | 16 | 147 | 0 | 0 | 0 | 0 | 0 | 0 | 147 | 747 | 0 | 0 | 45 | 1 | 0 | 0 | 747 |
| dma_store | 1 | 6073 | 0 | 2244 | 451 | 7 | 0 | 48 | 6457 | 11034 | 0 | 2244 | 943 | 7 | 0 | 48 | 11418 |
| sfu_lane | 4 | 2477 | 0 | 0 | 108 | 4 | 0 | 0 | 2477 | 2477 | 0 | 0 | 108 | 4 | 0 | 0 | 2477 |
| scratchpad_VEC | 1 | 8021 | 0 | 571 | 0 | 0 | 0 | 0 | 8021 | 8021 | 0 | 571 | 0 | 0 | 0 | 0 | 8021 |
| f32_mul_e4m3 | 16 | 385 | 0 | 0 | 19 | 1 | 0 | 0 | 385 | 385 | 0 | 0 | 19 | 1 | 0 | 0 | 385 |
| scratchpad_ACC | 1 | 5077 | 0 | 11 | 0 | 0 | 96 | 0 | 5077 | 5077 | 0 | 11 | 0 | 0 | 96 | 0 | 5077 |
| fix_to_f32 | 8 | 619 | 0 | 0 | 38 | 0 | 0 | 0 | 619 | 619 | 0 | 0 | 38 | 0 | 0 | 0 | 619 |
| f32_to_i32_rne | 16 | 271 | 0 | 0 | 22 | 0 | 0 | 0 | 271 | 271 | 0 | 0 | 22 | 0 | 0 | 0 | 271 |
| i32_to_f32 | 18 | 191 | 0 | 0 | 18 | 0 | 0 | 0 | 191 | 191 | 0 | 0 | 18 | 0 | 0 | 0 | 191 |
| front_end | 1 | 2872 | 0 | 1812 | 183 | 3 | 0 | 0 | 2872 | 2872 | 0 | 1812 | 183 | 3 | 0 | 0 | 2872 |
| scratchpad_STATE | 1 | 2839 | 0 | 3 | 0 | 0 | 32 | 0 | 2839 | 2839 | 0 | 3 | 0 | 0 | 32 | 0 | 2839 |
| mma_opbuf | 1 | 2492 | 0 | 0 | 0 | 0 | 0 | 512 | 6588 | 2492 | 0 | 0 | 0 | 0 | 0 | 512 | 6588 |
| mma_opbuf_a | 1 | 1940 | 0 | 0 | 0 | 0 | 0 | 304 | 4372 | 1940 | 0 | 0 | 0 | 0 | 0 | 304 | 4372 |
| mma_cell | 256 | 7 | 0 | 82 | 0 | 2 | 0 | 0 | 7 | 7 | 0 | 82 | 0 | 2 | 0 | 0 | 7 |
| queue | 5 | 316 | 0 | 112 | 17 | 0 | 0 | 36 | 604 | 316 | 0 | 112 | 17 | 0 | 0 | 36 | 604 |
| f32_to_bf16 | 64 | 24 | 0 | 0 | 8 | 0 | 0 | 0 | 24 | 24 | 0 | 0 | 8 | 0 | 0 | 0 | 24 |
| tile_meta | 1 | 1343 | 0 | 609 | 0 | 0 | 0 | 0 | 1343 | 1343 | 0 | 609 | 0 | 0 | 0 | 0 | 1343 |
| scratchpad_WGT | 1 | 804 | 0 | 1 | 0 | 0 | 64 | 0 | 804 | 804 | 0 | 1 | 0 | 0 | 64 | 0 | 804 |
| scratchpad_INP | 1 | 767 | 0 | 1 | 0 | 0 | 0 | 0 | 767 | 767 | 0 | 1 | 0 | 0 | 0 | 0 | 767 |
| f32_to_int_trunc | 2 | 241 | 0 | 0 | 12 | 0 | 0 | 0 | 241 | 241 | 0 | 0 | 12 | 0 | 0 | 0 | 241 |
| token_net | 1 | 67 | 0 | 0 | 0 | 0 | 0 | 0 | 67 | 220 | 0 | 68 | 17 | 0 | 0 | 0 | 220 |
| token_fifo | 17 | 9 | 0 | 4 | 1 | 0 | 0 | 0 | 9 | 9 | 0 | 4 | 1 | 0 | 0 | 0 | 9 |
| mma_pe | 0 | 33 | 0 | 102 | 0 | 0 | 0 | 0 | 33 | 929 | 0 | 102 | 52 | 1 | 0 | 0 | 929 |
| f32_mul_bf16 | 0 | 253 | 0 | 0 | 15 | 1 | 0 | 0 | 253 | 253 | 0 | 0 | 15 | 1 | 0 | 0 | 253 |
| f32_max | 0 | 123 | 0 | 0 | 19 | 0 | 0 | 0 | 123 | 123 | 0 | 0 | 19 | 0 | 0 | 0 | 123 |

## Children per module (instance counts from the blackbox cells)

- tcta_core: dma_load ×1, dma_store ×1, front_end ×1, mma_unit ×1, queue ×5, scratchpad_INP ×1, scratchpad_WGT ×1, scratchpad_ACC ×1, scratchpad_VEC ×1, scratchpad_STATE ×1, tile_meta ×1, token_net ×1, vec_unit ×1
- vec_unit: f32_add ×24, f32_mul ×16, f32_to_bf16 ×16, i32_to_f32 ×2, sfu_lane ×4
- mma_unit: f32_add ×8, f32_mul ×8, fix_to_f32 ×8, mma_cell ×256, mma_opbuf_a ×1, mma_opbuf ×1
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
