# Gate 1 report: five kernel families through TileLang → #19 → ISA → simulator

Closes the compiler/ISA exit condition of the plan: every kernel family has
been authored in TileLang, cut at pass #19 (`tl.InjectSoftwarePipeline`,
`compiler/prototype/cut.py`), mapped to the tcta ISA by the pattern pass
(`compiler/prototype/tile_tir_to_program.py`) and run on the functional
simulator v0.1.2 with token checking, against the fixtures in `fixtures/`.
Phase verdicts: 1 PASS, 2 PASS, 3 PASS, 4 PASS-WITH-CAVEATS (KDA's five
mapping idioms, named in `docs/tilelang-cutoff-verdict.md`, Phase 4). Repo
state at Gate 1: `pytest tests -q` → **58 passed** (sim 14, ref, cut 10, MoE 27,
MLA/KDA/GEMM 13).

**Phase 5 §0 update (ISA v0.1.3, the Gate 1 freeze candidate).** The five
programs were re-emitted on v0.1.3: `VCVT dst=WGT|INP` carries device-computed
operands, `VCVT dst=STATE, row0` moves the state halves into the 128-row STATE
tile, `VCOPY` and `VNEG` replace the copy and negation idioms. KDA drops from
521 to 352 instructions and from 13 DRAM round trips + a 64 KB state reload per
chunk to none (per chunk: 13 LOAD, 27 MMA, 76 VEC, 3 STORE; DRAM reads 230 KB
per chunk, inputs only); numerics unchanged. The masked products stay `VGATE
mul` because `MMA.MASKED` has no transposed-B form (§3). MLA 123 → 122, FA 130
→ 127, MoE persistent 527 → 525 (VCOPY). `pytest tests -q` → **59 passed**.
Numbers in §2 and §4 that are marked v0.1.2 are kept as the record of the Gate
1 state; the cycle model (`docs/cycle-model-report.md`) uses the v0.1.3
programs.

Step counts are the v0.1.2 issue-slot proxy (PUSH/POP free; one instruction
per queue per step when not blocked), not cycles. "busy" is the number of
steps on which a queue issued a work instruction.

## 1. Family summary

| Family | Kernel / fixture | Program | Steps | MMAs | Numerics (rowmax, tol 7.8e-3) | Busiest queue |
|---|---|---|---|---|---|---|
| GEMM | `gemm_kernel.py` / `gemm_partial` | 13 instrs (4 work), no loop | 3 | 1 | C 1.7e-7 | — |
| FA / GQA | `fa_kernel.py` / `flash_attn_gqa` (causal=False) | 127 on v0.1.3 (130) per (64-row tile, head), `LOOP n=2` steady state of the 4-tile pipelined KV loop; 32 programs | 73 (77) | 8 | O 5.05e-3 | MMA (8 of 77 steps busy; token-latency bound at this size) |
| MLA decode | `mla_kernel.py` / `mla_decode_absorbed` | 122 on v0.1.3 (123), `LOOP n=16` over KV tiles | 451 | 144 | O 2.15e-3 | VEC 280 / LOAD 160 / MMA 144 |
| KDA | `kda_kernel.py` / `kda_chunk64` (config C) | 352 on v0.1.3 (521 on v0.1.2), `LOOP n=4` over chunks | 348 (335) | 108 | S checkpoints ≤ 1.46e-3, o 5.99e-3 | VEC 307 / MMA 108 / LOAD 57 / STORE 12 |
| MoE + combine | `moe_kernel.py` / `moe_grouped` (f4.b32) | persistent 525 on v0.1.3 (527) (`LOOP n=16 { LOOP r; LOOP r }`), or 273 (274) without the count prefetch; combine 61 per 64-token tile; router 8 per token | 1092 (1153) | 768 | Yp 3.09e-3, y 1.52e-3 | LOAD 1025 / MMA 768 / VEC 449 / STORE 128 |

## 2. Per family

### 2.1 GEMM (`fixtures/gemm_partial`, C[50, 96] = A[50, 100] B[100, 96])

- **Instructions.** `LOAD.2D INP (50×100)`, `LOAD.2D WGT (100×96)`, `MMA`,
  `STORE.2D f32 (50×96)`, `FINISH`; 8 sync instructions.
- **Token graph.** LOAD→MMA 2, MMA→LOAD 1 (bank release), MMA→STORE 1
  (ISA §8.1 "plain GEMM output"). Pushes = pops; nothing primed or drained.
- **Numerics.** 1.7e-7 against the f64 reference (bf16 inputs, f32
  accumulate).
- **ISA features.** Partial-tile validities on `LOAD.2D`/`STORE.2D`; the
  array's contraction over `A.valid[1]` and output `A.valid[0] × B.valid[1]`
  (ISA §6.3); the bank-release convention.

### 2.2 FlashAttention-2 / GQA (`fixtures/flash_attn_gqa`)

- **Instructions.** 130 per (query tile, head): the `T.Pipelined` KV loop
  cut at #19 keeps its prologue/epilogue fills and the two-deep K/V rings;
  per KV tile 2 LOAD.2D, 2 MMA (`MMA.T` for QKᵀ), the online-softmax VEC
  sequence (`VROWMAX`, `VMAX`, `VSUB`, `VMUL`, `VEXP2.SUM`, `VSCALE_ACC`,
  `VCVT inplace`), `STORE.2D bf16` once. 32 programs cover the fixture.
- **Token graph.** LOAD→MMA 9, MMA→LOAD 8, MMA→VEC 9, VEC→MMA 8,
  VEC→STORE 1; 1 primed, 4 drained; FIFO max LOAD→MMA 5, MMA→LOAD 4
  (`tests/test_tilelang_cut.py::test_token_counts_per_edge`, with the
  hand-schedule comparison in its comments).
- **Numerics.** O 5.05e-3 worst head (bf16 P, bf16 output).
- **ISA features.** `LOOP n`, `tcta.fill_issued`/`fill_wait` events →
  pushes/pops for the weight ring, `MMA.T`, `VEXP2.SUM t, vec`, `VRCP` +
  `VSCALE_ACC`, in-place `VCVT` (P-ready), `STORE.2D bf16`.

### 2.3 MLA decode, absorbed (`fixtures/mla_decode_absorbed`)

- **Instructions.** 123; per KV tile 10 LOAD.2D (5 Q chunks re-streamed, 5 K
  chunks: one 64-wide rope chunk and four 128-wide chunks that double as the V
  tile), 5 `MMA.T` into the `[64, 64]` S tile, the FA softmax sequence with
  one `VSCALE_ACC wide` rescale, 4 `MMA` into the four slots of the `[64,
  512]` wide accumulator; after the loop `VRCP` + `VSCALE_ACC wide`,
  `STORE.2D wide f32`.
- **Token graph.** LOAD→MMA 160, MMA→LOAD 150, MMA→VEC 84, VEC→MMA 32,
  VEC→STORE 1; 10 primed, 0 hoisted, 0 at body end, 6 drained; FIFO max
  MMA→LOAD 6, MMA→VEC 5 (both above the default depth 4: the 4-slot WGT ring
  releases 4 banks per tile plus the rope slot).
- **Numerics.** O 2.15e-3.
- **ISA features.** Wide accumulator (`ACC.W`: 4 consecutive slots,
  `VSCALE_ACC wide`, `STORE.2D wide`), partial-tile `LOAD.2D` + 64-deep
  `MMA.T`, the resident WGT ring used by two products, `VFILL −1e30` for −inf.
- **Shape note.** Q is 4.5 INP tiles; the 2-slot INP forces 16× re-streaming
  (2.36 MB read for a 1.18 MB cache). With ≥ 5 INP slots the Q loads leave
  the loop.

### 2.4 KDA chunkwise (`fixtures/kda_chunk64`, config C)

v0.1.3 form (`dumps/program_kda.txt`): 352 instructions; per chunk 13 LOAD (8
`LOAD.IDX` gathers, the gate rows, beta, two index rows), 27 MMA (unchanged),
76 VEC (11 `VCVT dst=WGT`, 15 `VCVT dst=INP`, 2 `VCVT dst=STATE row0`, 4
kept-source bf16 casts, `VGATE decay/mul/add`, `VSCALE_ACC`, `VNEG`, `VSUB`,
`VROWMAX`, `VEXP2`), 3 STORE (`o` and the two checkpoint halves). Token graph
10 edges, 408 tokens, no LOAD→STORE edge (STORE→LOAD 4: a gather reusing the
slot a checkpoint read). 11 shared ACC slots. The v0.1.2 record follows.


- **Instructions.** 521; per 64-token chunk 27 LOAD (8 `LOAD.IDX` gathers of
  q/k/v/kᵀ/gᵀ rows into ACC, the gate rows and constants into WGT, 13
  `LOAD.2D` reloads of device-computed B operands, 1 `LOAD.2D f32 [128,128]`
  of the state mirror into STATE), 27 MMA (cumsum 1, K̂K̄ᵀ 1, solve 10, K̂S 1,
  U 2, Q̂S 1, Q̂K̄ᵀ 1, WU 2, state update 2×4 incl. decay-split pairs), 72 VEC
  (`VGATE decay/mul/add`, `VSCALE_ACC`, `VCVT` in place and into INP, `VMUL
  −1`, `VSUB`, `VROWMAX`, `VEXP2`, `VFILL`), 14 STORE (13 scratch/checkpoint
  tiles + `o`). No `MMA.MASKED`.
- **Token graph.** 12 of the 12 possible edges, 688 tokens: LOAD→MMA 73,
  MMA→LOAD 111, MMA→VEC 86, VEC→MMA 108, LOAD→VEC 39, VEC→LOAD 38,
  STORE→LOAD 49, LOAD→STORE 42, STORE→MMA 47, MMA→STORE 32, STORE→VEC 38,
  VEC→STORE 24; 15 primed, 3 hoisted, 0 at body end, 16 drained; FIFO max
  MMA→LOAD 8, STORE→MMA 5.
- **Numerics.** `S_chunks` 1.46 / 1.36 / 1.15 / 1.27e-3, `o` 5.99e-3 with hi/lo
  bf16 pairs on U, R and the decayed Kᵀ; 8.2e-3 (fail) with plain bf16
  operands. `kda_recurrent` unchanged; chunk algebra exact in f64.
- **ISA features.** `STATE` as a B operand (`LOAD.2D` f32 into STATE),
  `VGATE decay` with ACC gates, `VGATE mul/add` with 2-D VEC masks, STORE→LOAD
  / LOAD→STORE round trips, `LOAD.IDX` into ACC keeping f32, `VCVT` into INP,
  live-range slot sharing against the config-C descriptor.
- **Shape note.** Needs INP 3, WGT 8, ACC 8 (after sharing 21 buffers),
  VEC 5 two-dimensional + 2 rows, STATE 1, FIFO ≥ 8.

### 2.5 MoE decode + combine (`fixtures/moe_grouped`, E = 32, k = 6)

- **Instructions.** Per-expert program 270 (f4.b32 or bf16); persistent
  program 527 with the count prefetch (`LOOP n=16 { count→R1; LOOP r=0,
  step=64; count→R0; LOOP r=1 }`) or 274 single-register; combine 61 per
  64-token tile (4 programs); router 8 per token. Per token tile: 4 `LOAD.IDX`
  gathers, 24 `LOAD.2D f4.b32` fills + 4 scale loads, 24 MMA, 16 VEC (SwiGLU
  `VGATE mul`, `VSIGMOID`, `VSCALE_ACC`), 4 `STORE.IDX` scatters.
- **Token graph (persistent).** LOAD→MMA 896, MMA→LOAD 771, MMA→VEC 256,
  VEC→MMA 258, VEC→STORE 128, STORE→MMA 130, STORE→VEC 130, LOAD→VEC 65,
  VEC→LOAD 82, LOAD→STORE 32, STORE→LOAD 33; 10 primed, 2 hoisted, 0 at body
  end, 11 drained; FIFO max VEC→MMA 6.
- **Numerics.** `Yp` vs `y_parts` 3.09e-3 (both weight formats), `y` end to
  end 1.52e-3 (f32 gathers in the combine).
- **Overlap (F3, restated under free sync).** The LOAD queue is the critical
  one (32 instructions per tile vs 24 MMA); at all 31 expert boundaries it
  idles 1 step, the MMA gap is 3 steps, the first gather of `e+1` has 5 steps
  of token slack; ring depth 2/3/4 does not change the boundary (1091 / 1092 /
  1125 steps); the prefetch is worth 5.6 % (`docs/overlap-report.md`).
- **ISA features.** `LOAD.IDX` (−1 rows skipped), `STORE.IDX`, `VCOUNT` +
  `LOOP r, step` (zero-trip transparent), per-level `incr`, `LOAD.2D f4.b32`
  dequant-on-load with scale grids, `VTOPK`, `VSIGMOID`, `VGATE mul/add`,
  `LOAD→STORE` index relay, `STORE.2D i32`.

## 3. Consolidated ISA v0.2 candidate list

Closed in v0.1.1/v0.1.2 (for the record): per-level `incr`; LOAD↔STORE
edges; `LOAD.IDX` keeping f32 into ACC/VEC; free sync; `LOOP r, step`;
`STORE.IDX`; `VGATE` with ACC sources.

Closed in v0.1.3 (phase 5 §0): items 1, 3, 4, 7 and 9 below (`VCVT dst=WGT/INP`,
`MMA.MASKED mask_mode=mul` with ACC-resident masks, `row0` on MMA/VCVT into
STATE/ACC, `VCOPY`, `VNEG`). Item 2 is left to the compiler's numerics budget
with a BF16×2 array mode as the v0.2 hardware alternative. New after §0:
`MMA.MASKED` has no transposed-B form (`MMA.T` is where the delta rule's masked
Gram products live), so KDA's two masks are still `VGATE mul`; a
`transpose_b` flag on `MMA.MASKED` would close that. The list as it stood at
Gate 1:

1. **ACC→WGT move** (`VCVT dst=WGT`, or an MMA B operand from ACC). Removes
   KDA's 13 DRAM round trips per chunk (26 instructions, 1.3 MB of writes per
   256 tokens, STORE→LOAD latency on the solve's critical path). Any kernel
   that multiplies two device-computed matrices needs it.
2. **bf16×2 / f32 operand mode on the array.** KDA's plain bf16 schedule
   misses the tolerance; the hi/lo workaround is +7 MMAs per chunk (35 %).
   Alternatively the reference fixes those products in bf16.
3. **Multiplicative `MMA.MASKED`** (zero mask) or documented `VGATE mul` with
   a 2-D VEC mask as the triangular idiom. Today: one VEC instruction per
   masked product.
4. **Row-offset destinations on STATE/ACC, or 128-row state ops.** Would let
   the KDA state live in STATE alone (no mirror, no 64 KB reload per chunk).
5. **Transposed A operand** (`LOAD.2D transpose` or `MMA` A-transpose).
   Removes the host-side `kᵀ`/`gᵀ` layouts of KDA.
6. **`LOAD.COUNT reg, region` or tagged tokens.** Keeps the MoE trip-count
   round trip inside the LOAD queue; today a dynamic loop's entry pops wait on
   the VEC queue (phase-4 §0 finding 1).
7. **`VCOPY` (VEC↔VEC, ACC→ACC f32 copy).** Today `VFILL 0; VADD/VGATE add`
   (two instructions); used in MoE (`w_c`) and KDA (P, Rl/Ul/Ktl).
8. **Fused scale-then-exp2** for the softmax scalar path (5 instructions
   today for `exp2(m_prev·s − m·s)`); the matrix path already has
   `VEXP2.SUM`.
9. **Simulator fix: `VNEG`.** `_op_VNEG` requires an `imm`/`vec` operand the
   assembler does not emit; unusable as is. D4 emits `VMUL imm=−1`.
10. **ISA text:** §5.3 should state that `VGATE`'s source may be a 2-D VEC
    tile (the simulator accepts it); §8.1 should list the STORE↔LOAD edges'
    DRAM-buffer semantics (a round trip is the intended use).

## 4. Open items for the cycle model — answered in phase 5

The functional-simulator evidence (v0.1.2 issue-slot counts) is kept in the
second column; the third column is the cycle model's answer under config C
(`docs/cycle-model-report.md` §4, `model/cycle_results.md`; the programs are
the v0.1.3 ones).

| Question | Evidence from the five families (phase 4) | Cycle model (phase 5, config C) |
|---|---|---|
| **Ring depths** (INP / WGT / ACC) | MoE: WGT ring 3 gives overlap at every boundary, 2 does not change the free-sync boundary, 4 adds prologue only. FA: 2-deep K/V rings. MLA: 4 WGT slots serve K and V, 2 INP slots force Q re-streaming (10 vs 5 LOADs per tile) — price INP 5–6 for decode. KDA: 8 WGT, 8 ACC after sharing. | WGT ring 4 is worth −14.5 % on the MoE tile (11586 vs 13554 cycles; ring 2 +10.1 %): a weight load is 534 cycles against a 268-cycle MMA. INP 5 is worth −20.4 % on the MLA tile (3665 vs 4603) and halves its DRAM reads; INP 3 gives −10.9 %, INP 6 nothing more. **Config C: WGT 4, INP 5.** |
| **FIFO depths** | Per-edge maxima: MoE VEC→MMA 6; MLA MMA→LOAD 6, MMA→VEC 5; KDA MMA→LOAD 8, STORE→MMA 5; FA LOAD→MMA 5. Default depth 4 would stall three of the five families; config C's 16 is never approached. | Depth 4 **deadlocks KDA** in the timing pass; MLA loses 4 cycles; the other three are unchanged. Deepest dynamic level 6 (MLA MMA→LOAD). **Depth 8 on every edge**, both descriptors. |
| **Gather efficiency** | MoE: 4 `LOAD.IDX` per tile (same token rows, four 128-column windows), the LOAD queue issues 32 instructions per 24 MMAs; KDA: 8 gathers per chunk, 2048 index rows total, no skips. Whether a gather costs a row-DMA or a tile-DMA decides the MoE tile's bound. | `row` vs `burst64` is immaterial at config C (MoE 0 %, KDA +0.5 %); on the KV260 `burst64` with 16 outstanding costs KDA +5.3 %. The MoE tile is gather-bound when a gather's requests queue on one channel (4 × 1836 of 13554 cycles); rows spread over channels make it 1181, bound by the 1 request/cycle issue rate. |
| **VEC / SFU sizing** | MLA: 18 VEC per 9 MMA per tile, VEC busy 66 % of steps; KDA: 72 VEC per 27 MMA (splits and converts), VEC busy 87 %; MoE: 16 VEC per 24 MMA; FA: 7 per 2. Decode attention and the delta rule are VEC-issue-bound at equal costs; `VEXP2` throughput and the `VCVT`/`VGATE` rate set the tile time. | 16 → 64 lanes: MLA tile 7865 → 5627 (unfused), 4603 fused; FA 3047 → 2201, 1177 fused. The fused `VEXP2.SUM` is −18 % (MLA) / −47 % (FA). At 64 + fused MLA is 28 % VEC and LOAD-chain-bound; FA stays VEC-bound at every lane count (array 2.7 % busy). **64 lanes + fused SFU for config C; prefill attention needs bigger tiles.** |
| **DRAM round trips** | KDA on v0.1.2: 13 per chunk on the solve path (STORE → DRAM → LOAD). Closed by v0.1.3 (`VCVT dst=WGT`): 0 per chunk. | v0.1.3 chunk 15176 cycles vs 26535 for the v0.1.2 form at latency 300 (−43 %); latency slope 6.5 vs 25.0 cycles per chunk per latency cycle (100 → 600). |
| **State reload** | KDA on v0.1.2: one 64 KB `LOAD.2D` into STATE per chunk plus 32 KB of checkpoint stores. Closed by v0.1.3 (`VCVT dst=STATE, row0`). | Included in the row above (the v0.1.2 form carries both). |
| **Count round trip** | MoE: `VCOUNT` → `LOOP r` costs 3 LOAD-idle steps per expert without the prefetch (5.6 % of steps); with it, the loop-entry pop waits on the VEC queue. | Prefetch form 13554 cycles per tile vs 14498 without (−7.0 %), within 0.1 % of an oracle front end; latency slope 7.0 vs 7.7 cycles per tile per latency cycle. |
| **Q re-streaming vs INP depth** | MLA: 1.18 MB extra DRAM reads per 64-row decode step at INP 2. | 2304 KB → 1224 KB per step at INP 5; see the ring row. |
| **(new) request → channel mapping** | — | One request on one channel (the task's rule) vs a tile striped over 2 channels: MoE −20 %, MLA −17 %, KDA −17 %, FA −5 %; η_bw of the MoE tile 0.146 → 0.182 at 8. The single in-order LOAD stream keeps the 8 channels at ≤ 18 % of peak either way. |
| **(new) operand sharing across arrays** | — | Under one read port per bank the four arrays serialize on the MoE token tile (32640 stall cycles per INP bank); 2 ports −1.9 %, 4 ports −3.8 %. |

## 5. What the real compiler owns after Gate 1

Unchanged from the D6 list, plus: wide-accumulator slot grouping and the
`wide` qualifiers; DRAM scratch allocation for computed B operands (until
item 1 of §3); operand-splitting decisions (hi/lo) from a numerics budget;
live-range slot allocation against a device descriptor; the choice between
the prefetch and single-register MoE forms from the cycle model's count
latency. The pattern pass in `tile_tir_to_program.py` is the specification of
the mapping, not its implementation.
