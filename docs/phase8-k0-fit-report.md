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
| L5 8×8 array (the amendment's fallback): 64 cells, 8-lane collectors, half the operand buffers | −35 k → ~130 k (**measured 2026-10-29: −17.4 k → 145 k**, §4c) | ¼ of the MACs (MMA 64×128×128 4 284 → ~16 500 cycles); **not a parameter change today**: `BANKS = MAC_COLS = VEC_LANES = 16` (`tcta_pkg.sv`, one row slice per access), so the writeback needs half-slice byte enables and the B buffer 8 lanes (1–2 days) | architecture |
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

## 4a. Re-evaluation after L1–L2 (measured, 2026-10-29)

Requested with the phase-8 go: restate the fallback once L2, the lever that
attacks the lane-proportional logic, is measured.

| step | commit | LUT-eq saved | core LUT-eq | of the K26 | of the fit budget |
|---|---|---|---|---|---|
| after K0 | 9d34f96 | | 208 134 | 178 % | 209 % |
| L1 queues in LUT-RAM | e88d321 | −12 900 (est. −13 k) | 195 234 | 167 % | 196 % |
| L2 scratchpad clients per the access table, one address per request, readers / writers split | f4afb57 | −31 052 (est. −15 k) | 164 182 | 140 % | 165 % |
| L3 MMA writeback: 8 adders in two lane halves (converter kept at 8 lanes: 4 lanes cost FA's K = 64 products +15.6 %) | this commit | −5 750 (est. −5 k for the adders) | 158 432 | 135 % | 159 % |
| K1 row-pipelined DMA engines, `STORE.2D f4.r16` quantiser, block-scaled `LOAD.IDX` (a speed item, not a fit lever: `dma_store` +4.0 k for the egress quantiser, `dma_load` ±0) | e158d5d | +4 064 | 162 496 | 139 % | 163 % |
| K2 VEC pad banked by row (E2): the pad's element-serial path for row access to 2-D VEC tiles +3.0 k, the VEC unit's column mode and 16-lane vector register +2.2 k (a speed item: FA −18.9 %) | 1cf5bfe | +5 203 | 167 699 | 143 % | 169 % |
| K3 SFU slices pipelined (`vec_unit` −0.2 k) and the MMA writeback pipelined through its port (`mma_unit` +0.9 k: the 512-FF operand landing register, its adder-input and request-descriptor muxes); a speed item: FA −17.1 % | K3 commit | +759 | 168 458 | 144 % | 169 % |

L2 came in at twice its estimate because the ISA table separates readers from
writers: only 12 of the 35 (pad, client) pairs read and 12 write, so the
write-data muxes and the response path shrank more than the arbiter did
(the five pads 40 789 → 14 506 LUTs, the core's response mux 8 361 → 3 592).

**Fallback statement, re-run.** With L3 and L4 at their estimates (−15.5 k)
the 16-lane core lands near **149 k LUT-eq**, 1.5× the fit budget. The
8×8 array (L5, −35 k by the K0 estimate) on the 16-lane core would then be
near **114 k**: still above the 99.5 k budget by Yosys, but now inside
Vivado's usual 20–35 % gain over Yosys (it needs 13 %), where before L1–L2
it needed more than the gain could give (130 k, 23 %) and the 16-lane core
with L1–L4 (165 k) needed 40 %. So after L2 the 8×8 fallback is no longer
ruled out: it is the Vivado run that decides between the 16-lane core with
L1–L4 (needs a 33 % gain), the 8×8 array on the 16-lane core (13 %), and the
8-lane configuration (≈ 80 k by the same arithmetic, fits by Yosys alone).
The order of the phase-8 task stands: Vivado first, L3–L4 now, the lane
decision from the Vivado figure; the array is not the problem.

## 4b. L4 (operand buffers in block RAM): analysed, not taken (2026-10-29)

The K0 estimate sized L4 by capacity (2 × 128 kbit = 16 RAMB18, −8 k
LUT-eq). The port structure does not allow that mapping:

- **The buffers are a corner turn.** The fill engines write one row slice
  per cycle (16 elements of one row or, scattered, one element of each of 16
  columns), while the array reads one element of every one of the 16 rows
  per cycle, each row at its own k. Row-major memories (one per row, as the
  LUT-RAM is organised) need a 256-bit write port per memory: 8 RAMB18 per
  row, 128 per buffer. Lane-major memories (one per lane, element k of row r
  at `{r, half, k / 16}`) take the write as one 16-bit word per memory and
  the reads as a permutation across lanes (`k mod 16` differs per row), but
  a block whose K is not a multiple of 16 overlaps the next one on the same
  lanes (GEMM's K = 100), so each lane memory sees up to two reads and the
  fill's write per cycle: three ports.
- **Splitting by buffer-half parity** (halves {0, 2} and {1, 3} in separate
  memories; consecutive blocks use consecutive halves, blocks two apart never
  overlap for K ≥ 16) brings it to one read and one write per memory, the
  RAMB18 SDP shape: 32 RAMB18 per buffer, **64 RAMB18 = 32 RAMB36-eq** for
  the two, plus the lane permutation and its inverse on the address side
  (≈ 4 k LUTs per buffer). Net: ≈ −3 to −4 k LUT-eq for +32 RAMB36-eq.
- **Block RAM is the tighter resource.** The pads need 122 RAMB36 at the
  bit minimum against the K26's 144 (299 as Yosys maps them, which Vivado's
  port merging or the task's URAM move for ACC / STATE has to fix first);
  32 more leaves nothing for the AXI glue and nothing if the merging falls
  short. The 16-RAMB18 version does not exist.

Decision: the operand buffers stay in LUT-RAM (A 1 940 LUT + 304 RAM32M16,
B 2 492 + 592; 11.6 k LUT-eq, 7 % of the core). The lever is reopened only
if ACC and STATE move to URAM (task §0.4) and the Vivado block-RAM count
leaves ≥ 32 RAMB36 spare; then it buys ≈ 3–4 k LUT-eq, not 8 k.

## 5. Reproduce

```
export PATH=~/.local/opt/oss-cad-suite/bin:$PATH
python3 rtl/synth/yosys_ooc.py -j 2          # ~20 min on the aarch64 box; per-module JSON in rtl/synth/ooc/
```

## 4c. L5 (the 8×8 array on the 16-lane core): measured, kept as a build option (2026-10-29)

Requested by the main session before the Vivado run. `TCTA_MAC8` (`tcta_pkg.sv`)
sets `MAC_ROWS = MAC_COLS = 8` with `BANKS = 16`: a block covers half a slice,
so the B fill takes lanes `j0 % 16 .. +7` of each WGT slice, the converter
places a result row at those banks in one pass, the writeback works on one
adder half (no second mask pass, old + result straight into the write) with
the half's bank enables, the result FIFO holds two blocks (16 rows), blocks
start every `max(K, 8)` cycles. `make all MAC8=1`: 10/10 PASS bit-exact;
`calibrate.py --timing=KV260_L5` reproduces every MMA shape exactly.

| | 16×16 (K1 state) | 8×8 (`TCTA_MAC8`) | delta |
|---|---|---|---|
| `mma_unit` own (controllers, prep, collectors, FIFO, converter, writeback) | 26 023 LUT-eq | 14 917 | −11 106 |
| operand buffers (B scatter + A wide) | 6 588 + 4 372 | 3 771 + 2 194 | −4 995 |
| cells (7 LUT + 2 DSP each) | 256 × 7 = 1 792 LUT, 512 DSP | 64 × 7 = 448, 128 DSP | −1 344 LUT, −384 DSP |
| `mma_unit` total | 53 199 LUT-eq, 537 DSP, 29 820 FF | 35 754, 153 DSP, 11 422 FF | **−17 445 (−33 %)** |
| core | 162 496 LUT-eq (139 % of the K26) | **145 051 (124 %)** | −10.7 % |
| MMA 64×128×128 | 4 284 cycles | 16 547 | 3.9× |
| MMA 64×64×64 (FA, acc / .T) | 1 436 / 1 148 | 4 211 / 4 163 | 2.9× / 3.6× |
| GEMM 50×100·100×96 program | 5 917 | 11 875 | +101 % |
| FA program (4 KV tiles) | 32 109 | 55 269 | +72 % |
| mma / vec / gather programs | 23 230 / 14 910 / 22 683 | 72 218 / 27 173 / 48 404 | |

The K0 estimate (−35 k) was twice the measurement: the array's cells are
DSP-based (7 LUTs each, 1.8 k in all), so three quarters of them are worth
1.3 k LUTs, and what the smaller array removes is its lane- and
row-proportional control (collectors, row and column controllers, operand
preparation, the FIFO lanes: −11.1 k) and half of the operand buffers
(−5.0 k). The 109 k of the core outside `mma_unit` (VEC 53.6 k, DMA 29.8 k,
pads 14.5 k) is sized by `BANKS = 16` and does not move. The array runs at
full occupancy under the 8×8 build (blocks back to back at `K` cycles, the B
fill at `K` per block), so the cycle cost is the MAC count and nothing
structural.

**Fallback statement, measured.** On the Yosys numbers the 16-lane core
(162.5 k) fits the 99.5 k budget only if Vivado maps ≥ 39 % under Yosys,
the 8×8 array on the 16-lane core (145.1 k) only if ≥ 31 %; both are at or
beyond the usual 20–35 % gain, and L5 buys its 10.7 % with FA +72 % and
GEMM +101 %. The 8-lane configuration (`BANKS = 8`, ≈ 80 k by the §4
arithmetic) remains the only Yosys-backed fit, and the task's §0.3 rule
stands: 16-lane core if Vivado ≤ 99.5 k, otherwise the 8-lane configuration
for the KV260 build; the 8×8 array alone is not a middle ground worth its
speed. `TCTA_MAC8` stays in the tree as a build option (default off) for the
Vivado run to measure both arrays in one pass.
