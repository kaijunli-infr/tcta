# Tile-Centric Tensor Accelerator — ISA Specification v0.1.3 (Gate 1 freeze candidate)

Status: Gate 1 freeze candidate. Date: 2026-10-27. v0.1.3 (after the MLA/KDA prototypes, `docs/gate1-report.md`) adds the ACC→WGT/INP move, `VCOPY`, a multiplicative mask mode and ACC-resident masks for `MMA.MASKED`, and row-offset writes into STATE/ACC. v0.1.1 added `LOOP r, step`, an ACC operand for `VGATE`, and the `i32` store qualifier (MoE prototype); v0.1.2 adds per-level loop increments, the LOAD↔STORE edges, f32 gathers into ACC, and the free-sync issue rule (persistent-MoE prototype). See §15.
Owner: ISA. Consumers: functional simulator, cycle model, compiler (tensorize + token insertion), RTL.

This document defines instruction semantics, scratchpad names, data types and the synchronization model. It does not define latency, bandwidth, bank counts, queue depths or encodings beyond what the compiler must know. Those belong to the microarchitecture and are runtime-discoverable parameters (§11).

---

## 0. Design decisions this version encodes

| # | Decision | Section |
|---|---|---|
| D1 | Decode-first machine; prefill supported, not optimized | §1 |
| D2 | Tile is the unit of addressing and of every dependency token; no byte addressing of scratchpads | §2, §3 |
| D3 | Five decoupled queues (LOAD, MMA, VEC, STORE, COMM); ordering only by explicit PUSH/POP tokens | §2, §8 |
| D4 | `LOAD.IDX` is row-granular (rows are multiples of 64 B); tile-granular gather is the degenerate case | §6.1 |
| D5 | Block-scaled formats (MXFP4, FP8 block, FP4 KV) are dequantized by the DMA on ingress, selected by a dtype qualifier on LOAD/STORE; scales are never a separate operand in v1 | §4, §6.1 |
| D6 | Index lists are VEC scratchpad tiles with their own tokens; the VEC→LOAD edge exists in the token graph | §5.3, §8.1 |
| D7 | MMA may take its A operand from ACC (P tile written in place by VEC) | §6.3 |
| D8 | Triangular solve is a VEC/MMA sequence; no TRSV instruction | §6.4 |
| D9 | Reserved for v2: `MMA.MX` (array-native block scales), scoreboard mode (tokens become assertions) | §13 |

---

## 1. Machine model

The accelerator is a decoupled access/execute tile processor. A kernel is a linear instruction stream, partitioned by the front end into five in-order queues that execute independently:

| Queue | Executes | Reads scratchpads | Writes scratchpads |
|---|---|---|---|
| LOAD | `LOAD.*` | VEC (index tiles only) | INP, WGT, VEC, STATE |
| MMA | `MMA.*` | INP, WGT, ACC, VEC (mask/scale), STATE | ACC, STATE |
| VEC | `V*`, SFU ops | ACC, VEC, STATE | ACC, VEC, STATE, INP |
| STORE | `STORE.*` | ACC, VEC | — |
| COMM | `SEND`, `RECV` | INP, WGT, VEC, ACC | INP, WGT, VEC |

Control instructions (`LOOP`, `ENDLOOP`, `FINISH`, `SET*`) are executed by the front end and affect all queues.

There is **no implicit ordering** between queues. A read of a tile written by another queue without an intervening token pair is a programming error; the functional simulator faults on it (§8.4), the RTL may return stale data.

Within one queue, instructions execute in program order and later instructions observe earlier ones.

---

## 2. Tiles and scratchpads

### 2.1 Tile

A tile is a 2-D array of elements, `[rows, cols]`, stored in one scratchpad at one tile index. Every instruction operand that names scratchpad data is a tile reference:

```
tref := (scratchpad, index)
```

Tile shapes are per scratchpad (§2.2). Partial tiles are expressed by `rows_valid`/`cols_valid` fields on the instruction, not by a different tile shape. Elements outside the valid region are undefined after the instruction.

### 2.2 Scratchpads

| Name | Element type | Tile shape | Purpose | Written by | Read by |
|---|---|---|---|---|---|
| INP | BF16 | `[T_M, T_K]` | A operands: activations, P (if not in ACC), chunk inputs | LOAD, VEC, COMM | MMA, COMM |
| WGT | BF16 | `[T_K, T_N]` | B operands: weights, K, V | LOAD, COMM | MMA, COMM |
| ACC | FP32 | `[T_M, T_N]` | Accumulators; also holds P after in-place VEC write | MMA, VEC | VEC, STORE, MMA (A operand, §6.3), COMM |
| VEC | FP32 or INT32 | `[T_M, T_V]` (2-D) or `[1, T_VR]` (row) | Row statistics, masks, gates, rescale factors, **index lists**, router scores | LOAD, VEC | VEC, MMA (mask/scale port), LOAD (index), STORE, COMM |

VEC has two tile kinds. A **2-D VEC tile** is `[T_M, T_V]` and holds per-row vectors (one value per ACC row, or a mask). A **row tile** is `[1, T_VR]` with `T_VR ≥ T_V` (v1: `T_VR = 512`) and holds one long vector: router scores over all experts, a top-k index list, an accept mask. `VTOPK` writes row tiles; `LOAD.IDX` reads them. A slot is either kind; the kind is fixed per slot by the device descriptor.
| STATE | FP32 | `[T_S, T_S]` | Resident recurrent state (KDA S) | MMA, VEC | MMA, VEC |

`T_M, T_K, T_N, T_V, T_S` and the number of tile slots per scratchpad are implementation parameters (§11). The v1 reference configuration is `T_M = 64, T_K = 128, T_N = 128, T_V = 64, T_S = 128`, with ACC additionally supporting a `[64, 512]` wide tile for MLA decode (`ACC.W`, 4 consecutive ACC slots treated as one tile by `MMA` with `wide=1`).

Bank assignment is invisible to the ISA. The runtime chooses tile indices; the hardware chooses banks.

### 2.3 Element types

| Code | Type | Bits | Where it may appear |
|---|---|---|---|
| `bf16` | BFloat16 | 16 | INP, WGT, DRAM |
| `f32` | IEEE binary32 | 32 | ACC, VEC, STATE, DRAM |
| `i32` | signed 32-bit integer | 32 | VEC (index tiles, counts) |
| `f8` | FP8 E4M3 (per-tensor or block-scaled, §4) | 8 | DRAM only (dequantized on LOAD) |
| `f4` | FP4 E2M1 (block-scaled, §4) | 4 | DRAM only (dequantized on LOAD) |

The array consumes BF16 only in v1. FP8 and FP4 exist as DRAM formats and are converted by the DMA (§4).

---

## 3. Addressing DRAM

DRAM operands are described by a **region descriptor**:

```
region := { base: u64, dtype: dtype_q, shape: [d0, d1, d2], stride: [s0, s1, s2] (bytes) }
```

`LOAD.2D` / `STORE.2D` use `d0 × d1`; `LOAD.3D` uses all three (head gather, paged KV). Strides are in bytes and may be zero (broadcast). `base` is a device virtual address; translation (page tables for paged KV) is done by the DMA and is invisible to the ISA except for the fault codes in §10.

---

## 4. Data-type qualifiers (dequant on ingress)

A `dtype_q` qualifier on every `LOAD.*` and `STORE.*` names the DRAM format and how it is converted to the scratchpad's element type. This is the only place block scales appear in the ISA.

| Qualifier | DRAM layout | Conversion on LOAD | Allowed destinations |
|---|---|---|---|
| `bf16` | BF16 | none | INP, WGT |
| `f32` | FP32 | none; or `f32→bf16` convert if dest is INP/WGT | INP, WGT, VEC, STATE |
| `i32` | INT32 | none | VEC |
| `f8.t` | FP8 E4M3, one FP32 scale per tensor (scale in descriptor) | `x × s` → BF16 | INP, WGT |
| `f8.b32` | FP8 E4M3, UE8M0 scale per 32×32 block, scales stored after the block grid | `x × 2^e` → BF16 | WGT |
| `f4.b32` | FP4 E2M1 packed 2/byte, UE8M0 scale per 32×32 block | `x × 2^e` → BF16 | WGT |
| `f4.r16` | FP4 E2M1 packed, one E4M3 scale per 16 consecutive channels in a row (KV cache layout) | `x × s` → BF16 | WGT (K/V rows via `LOAD.IDX`) |

Rules:
- Scale-block geometry is part of the qualifier, not of the descriptor, so the compiler cannot produce a load that the DMA cannot parse.
- `STORE.*` supports `bf16`, `f32` and `i32` destinations in v1 (`i32` for index rows and counts), plus `f4.r16` for writing new KV entries (quantization on egress, scale computed per 16 channels, round-to-nearest-even).
- The DRAM layout for each block-scaled format is normative and given in Appendix A. Padding to 64-byte rows is mandatory for formats used with `LOAD.IDX` (§6.1).
- Dequantized values are exact BF16 roundings of `x × scale`; the simulator is the reference.

---

## 5. Instruction classes

Operand notation: `tref` tile reference; `region` DRAM descriptor; `imm` immediate; `r` front-end register (32-bit, 16 registers `R0–R15`); `q` a qualifier.

### 5.1 LOAD queue

| Instruction | Operands | Semantics |
|---|---|---|
| `LOAD.2D dst, region, q, rows_valid, cols_valid` | dst ∈ INP/WGT/VEC/STATE | Copy a `[rows_valid, cols_valid]` block from DRAM into `dst`, converting per `q`. |
| `LOAD.3D dst, region, q, rows_valid, cols_valid` | as above | As `LOAD.2D` with a third stride (head/page dimension). |
| `LOAD.IDX dst, region, idx, q, nrows, row_bytes, col0, cols_valid` | dst ∈ INP/WGT/ACC/VEC; idx ∈ VEC (i32 row tile) | Row gather, §6.1. `col0`/`cols_valid` select the column window of each gathered row that lands in `dst`. Rows are rounded to BF16 only for INP/WGT; an ACC/VEC destination keeps FP32 (partial-sum combine). |

### 5.2 STORE queue

| Instruction | Operands | Semantics |
|---|---|---|
| `STORE.2D src, region, q, rows_valid, cols_valid` | src ∈ ACC/VEC | Write a tile to DRAM, converting per `q`. |
| `STORE.IDX src, region, idx, q, nrows, row_bytes` | src ∈ ACC/VEC; idx ∈ VEC | Row scatter; same index semantics as `LOAD.IDX`. No ordering among rows; duplicate indices are undefined. |

### 5.3 VEC queue

Index tiles are ordinary VEC tiles of type `i32` written by `VTOPK` (or loaded by `LOAD.2D ... i32`), consumed by `LOAD.IDX`/`STORE.IDX` after a VEC→LOAD or VEC→STORE token (§8.1).

| Instruction | Operands | Semantics |
|---|---|---|
| `VROWMAX dst, src` | dst ∈ VEC, src ∈ ACC | Per-row max over valid cols. |
| `VROWSUM dst, src` | dst ∈ VEC, src ∈ ACC | Per-row sum. |
| `VSUB src, vec` / `VMUL src, vec` / `VADD src, vec` / `VMAX src, vec` | src ∈ ACC/VEC in place; vec ∈ VEC per-row or immediate | Row-broadcast elementwise; `VMAX` is the online-softmax `m_new = max(m_old, m_tile)`. |
| `VSCALE_ACC acc, vec` | acc ∈ ACC (may be `wide`), vec ∈ VEC | `acc[r, :] *= vec[r]` in place (online-softmax O rescale). |
| `VGATE acc, src, op` | src ∈ VEC (2-D tile) or ACC; op ∈ {mul, add, decay} | Elementwise `acc = acc ∘ src`. With `src ∈ ACC` this is the ACC×ACC multiply the SwiGLU gate needs; with a VEC tile it is KDA per-channel gating. |
| `VCVT dst, src, q, row0` | dst ∈ INP/WGT (BF16), ACC/STATE (FP32 move, or BF16 with `q`); src ∈ ACC | **The device-computed-operand path**: converts an ACC tile to BF16 into INP (A operand) or WGT (B operand), so a product of two on-chip results needs no DRAM round trip. With `dst = src` and `q = bf16.inplace`, writes BF16 P into the ACC tile's storage for `MMA` A-from-ACC (§6.3). `row0` places a `T_M`-row result at a row offset of a taller STATE/ACC tile (a `[128,128]` state is written as two halves). |
| `VCOPY dst, src` | same scratchpad kind (VEC→VEC, ACC→ACC) | Exact copy. |
| `VMASK acc, mask_kind, vec` | mask_kind ∈ {causal, band, tile} | Write `−inf` into `acc` per mask; `band` width and `tile` mask come from `vec`. |
| `VTOPK idx, val, src, k, axis` | idx, val ∈ VEC (i32, f32); src ∈ ACC | Top-k along `axis`; writes indices (row-relative or absolute per `SETBASE`) and values. `k ≤ T_V`. |
| `VCOUNT r, idx, bound` | r ∈ R; idx ∈ VEC | Count entries of `idx` in `[0, bound)`; writes a front-end register for `LOOP`. |
| `VSEL dst, a, b, mask` | VEC/ACC | Select. |
| `VEXP2 t` / `VRCP t` / `VLOG2 t` / `VSIGMOID t` | t ∈ ACC/VEC in place | SFU ops; accuracy ≤ 2 ulp BF16 after convert. `VEXP2` optionally fuses `VSUB` (`VEXP2 t, vec` computes `exp2(t − vec[r])`) and `VROWSUM` (`VEXP2.SUM t, vec, dst`). |

### 5.4 MMA queue

| Instruction | Operands | Semantics |
|---|---|---|
| `MMA acc, a, b, accumulate, wide` | acc ∈ ACC/STATE; a ∈ INP or ACC (§6.3); b ∈ WGT | `acc (+)= a × b`. `wide=1` targets `ACC.W`. |
| `MMA.T acc, a, b, accumulate` | as above | `b` is read transposed (for `Q·Kᵀ`). |
| `MMA.MASKED acc, a, b, mask, accumulate, mode, row0` | mask ∈ VEC (2-D, `[T_M, T_V]`) or ACC (`[T_M, T_N]`); mode ∈ {add, mul} | As `MMA` then applies `mask` before writeback: `add` adds `−inf`/0 (softmax masking), `mul` multiplies by 0/1 (the delta-rule triangular products, which must zero, not −inf). A full-width mask lives in ACC. |
| `MMA acc, a, b, accumulate, row0` (addendum) | acc ∈ STATE or ACC with `row0 ∈ {0, T_M, …}` | Writes a `T_M`-row product into rows `[row0, row0+T_M)` of a taller destination; the tile's valid rows grow to cover the union. |

Accumulation is FP32. Inputs are BF16 (A from ACC is read as the BF16 payload written by `VCVT ... inplace`).

### 5.5 COMM queue

| Instruction | Operands | Semantics |
|---|---|---|
| `SEND src, chip, dst_tref, nbytes_valid` | src ∈ INP/WGT/VEC/ACC | Transmit a tile to `dst_tref` on `chip`. Completion on the sender means the source tile may be reused. |
| `RECV dst, chip, tag` | dst ∈ INP/WGT/VEC | Block until a tile with `tag` arrives from `chip`; then `PUSH` to the consumer exactly as a `LOAD` would. |
| `RECV.ANY dst, tag, r` | r ← source chip | Receive from any chip; used for ragged combine. |

v1 hardware emulates the link through DRAM between two accelerator instances; the simulator models N chips with latency and bandwidth parameters.

### 5.6 SYNC

| Instruction | Semantics |
|---|---|
| `PUSH q_from → q_to` | Enqueue one token on the FIFO from `q_from` to `q_to`. Issued in `q_from`'s stream after the producing instruction. |
| `POP q_to ← q_from` | Dequeue one token; blocks `q_to` until available. Issued in `q_to`'s stream before the consuming instruction. |

Valid edges are listed in §8.1. A token carries no payload; it is a promise that the next tile the consumer was told about is ready (or free).

### 5.7 Control (front end)

| Instruction | Semantics |
|---|---|
| `LOOP n` / `LOOP r, step` … `ENDLOOP` | Hardware loop. Static form: `n` iterations. Register form: runs `⌈R[r] / step⌉` iterations (0 when `R[r] ≤ 0`), so a token count from `VCOUNT` drives a loop over `T_M`-row tiles with `step = T_M` and no host-side `ceil`. Nesting depth ≥ 2. **Loop-variant operands are per level**: an instruction carries up to one increment per enclosing loop level for each variant field (DRAM base, scale-grid base, tile slot, slot-ring modulus); the field's value is the sum of `i_L × step_L` over active levels. This is what lets a static outer loop over experts advance the operands of instructions inside an inner register-count loop without unrolling. |
| `SETSCALE r, imm` | Softmax scale, etc. |
| `SETBASE r, imm64` | Base for absolute indices produced by `VTOPK`. |
| `SETMASK r, imm` | Band width / tile mask select. |
| `FINISH` | All queues drain; all token FIFOs must be empty (§8.3); status written. |

---

## 6. Detailed semantics

### 6.1 `LOAD.IDX` (row gather)

```
LOAD.IDX dst, region, idx, q, nrows, row_bytes, col0, cols_valid
for i in 0 .. nrows-1:
    r = idx[i]                       # i32, read from VEC row tile
    if r == -1:                        dst.row[i] = 0; continue        # skip
    if r < 0 or r >= region.shape[0]:  fault IDX_RANGE (§10)
    src = region.base + r * region.stride[0]
    row = dequant_q( DRAM[src : src + row_bytes] )                     # whole row (scales live in it)
    dst.row[i][0:cols_valid] = row[col0 : col0 + cols_valid]
rows nrows .. T_M-1 of dst are undefined
```

A gathered row wider than one tile (a 576-channel KV entry into `[128, 128]` WGT tiles) is loaded by repeating `LOAD.IDX` with the same index tile and advancing `col0`; the DMA may cache the row across these repeats (microarchitecture), and the index tile is read, not consumed, so one VEC→LOAD token covers the whole sequence.

Constraints:
- `row_bytes` is a multiple of 64 and ≤ the destination tile row in bytes after conversion.
- `nrows ≤ T_M`. The index tile must hold ≥ `nrows` valid entries; `VCOUNT` or the producing `VTOPK` defines validity.
- Rows are written in index order; the DMA may fetch in any order (coalescing is not observable).
- Negative index `−1` is a **skip**: the row is zero-filled, no fault. This encodes ragged tails without branching.
- Every block-scaled format used with `LOAD.IDX` must have its per-row scales inside the same 64-byte-aligned row (Appendix A). The KV entry layout for `f4.r16` is therefore 288 B payload + 36 B scales, padded to 384 B.

Expert token gather is `LOAD.IDX` with `row_bytes = d × 2` (BF16 activation row); KV sparse attention is `row_bytes = 384` with `q = f4.r16`; Engram rows are `row_bytes = 256` with `q = f8.t`.

### 6.2 `LOAD.2D/3D` with partial tiles

`rows_valid`/`cols_valid` bound the DRAM access; elements outside are not read and are undefined in the destination. A consumer `MMA` must use the same valid bounds or accumulate garbage. The compiler is responsible; the simulator checks that an `MMA` never reads a column/row the producing load did not mark valid.

### 6.3 `MMA` with A operand in ACC

`VCVT acc, acc, bf16.inplace` writes the BF16 rounding of each FP32 element into the low 16 bits of the same storage and marks the tile `P-ready`. An `MMA` with `a ∈ ACC` reads those 16 bits. Reading an ACC tile as A that is not `P-ready` is a fault (`ACC_NOT_CONVERTED`). The ACC tile is still an FP32 tile for every other consumer; its FP32 values are destroyed by the in-place convert.

The port cost of this feature is a microarchitecture choice; an implementation that lacks the port must still accept the instruction and may implement it as an internal copy.

### 6.4 Triangular solve (KDA)

Not an instruction. The reference sequence for a `C × C` lower-triangular `(I + L)^{-1}` is a `log2(C)`-step Neumann/blockwise sequence of `MMA.MASKED` + `VSUB`, given in Appendix B. The compiler emits it as a macro.

### 6.5 Wide accumulator

`MMA ... wide=1` treats ACC slots `[i, i+1, i+2, i+3]` as one `[T_M, 4·T_N]` tile. `VSCALE_ACC` and `STORE.2D` accept `wide`. No other instruction does.

---

## 7. Front-end registers and ragged loops

Sixteen 32-bit registers `R0–R15`. Written by `VCOUNT`, `RECV.ANY`, `SET*`, and the host (kernel arguments). Read by `LOOP` counts and by loop-variant operand fields. There is no ALU; arithmetic on registers is limited to the per-iteration increments of `LOOP`.

A `LOOP` whose count register is 0 executes zero iterations, and any `PUSH`/`POP` inside it is skipped. This is the only conditional control flow in the ISA and is what makes zero-token experts legal.

---

## 8. Token rules

### 8.1 Queue graph

Tokens flow only along these edges. Each edge is a FIFO; depth is an implementation parameter, and a `PUSH` to a full FIFO stalls the producer.

| Edge | Meaning | Typical use |
|---|---|---|
| LOAD → MMA | tile in INP/WGT/STATE is full | K/V, weights, activations |
| LOAD → VEC | tile in VEC/STATE is full | masks, gates, loaded index lists |
| MMA → VEC | ACC tile is full | S ready for softmax |
| VEC → MMA | ACC/INP/VEC tile is ready, or WGT/INP bank is free | P ready, mask ready, rescale done |
| MMA → LOAD | INP/WGT bank is free | release K/V bank after use |
| VEC → LOAD | **index tile is ready** (D6) | router top-k, sparse-attention indices |
| VEC → STORE | ACC/VEC tile is final | O tile, logits |
| MMA → STORE | ACC tile is final (no vector pass) | plain GEMM output |
| STORE → VEC, STORE → MMA | ACC tile is free | release accumulator |
| COMM → MMA, COMM → VEC | received tile is full | expert inputs, combine partials |
| MMA → COMM, VEC → COMM | tile ready to send / bank free to receive | dispatch, combine |
| LOAD → COMM | tile loaded for send | — |
| LOAD → STORE, STORE → LOAD | a loaded index row or tile is consumed directly by `STORE.IDX`/`STORE.2D`; STORE releases it | scatter index rows, pass-through copies |

Any `PUSH`/`POP` on an edge not in this table is an encoding error.

### 8.2 The two obligations

1. **Full before read.** A queue may read a tile only after `POP`ping on the edge from the tile's most recent writer, once per write.
2. **Free before overwrite.** A queue may overwrite a tile only after `POP`ping on the edge from every reader that the compiler scheduled against the previous contents, once per such reader.

Tokens are counted, not named. The compiler guarantees that the k-th `POP` on an edge corresponds to the k-th `PUSH`, and that the tile each side has in mind is the same. The hardware does not check this; the simulator does (§8.4).

3. **Per-edge FIFO order.** Because tokens carry no tile identity, the consumer's `POP`s on an edge must occur in the same order as the producer's `PUSH`es of the tokens they stand for. A dependency analysis that emits pops in consumer program order can violate this when one producer feeds two consumer sites in reverse order (found in the TileLang FA prototype: "S ready (k+1)" popped before "O ready (k)"); the compiler must reorder such pops earlier. Moving a pop earlier is always safe; moving it later is not. Tile-tagged dependencies (scoreboard mode, §13) remove this obligation in v2.

### 8.3 Termination and deadlock

- At `FINISH`, every token FIFO must be empty. A non-empty FIFO is a fault (`TOKEN_LEAK`).
- A `POP` that waits longer than `T_DEADLOCK` cycles (implementation parameter; simulator default 10⁶) faults (`TOKEN_TIMEOUT`) with the queue and edge in the status word. This covers the three known deadlock shapes: a `RECV` that never arrives, a `LOAD.IDX` whose index tile is never produced, and a zero-iteration `LOOP` on one queue mismatched with a non-zero one on another.
- Deadlock freedom for a static schedule is a compiler obligation; the bounded-model-checking target is the FIFO protocol and the bank arbiter, not the schedule.

### 8.5 Issue cost of SYNC

`PUSH` and `POP` do not occupy an issue slot. A queue retires every leading `PUSH`/`POP` that can proceed in the same cycle as its next work instruction, and a `POP` costs time only while it blocks. In hardware this is the queue's dependency check, not an instruction; an encoding may fold the sync fields into the work instruction (§12). The functional simulator follows this rule so that its step count is a first-order issue-slot proxy (MMA queue: one GEMM per step when not blocked).

### 8.4 Simulator obligations

The functional simulator tracks, per tile slot, its last writer queue and a set of outstanding readers, and faults on:
- a read without a preceding matching `POP` (`UNTOKENED_READ`);
- a write while readers are outstanding (`UNTOKENED_WRITE`);
- an `MMA` consuming columns/rows beyond the producing load's valid region (`VALID_MISMATCH`);
- an `MMA` A-from-ACC on a tile not `P-ready` (`ACC_NOT_CONVERTED`);
- `LOAD.IDX` / `STORE.IDX` with an index out of range or an index tile with fewer valid entries than `nrows` (`IDX_RANGE`, `IDX_SHORT`).

A kernel that passes the simulator with these checks enabled is **conformant**; RTL is tested only with conformant streams.

---

## 9. Numerics

- MMA: BF16 × BF16 products exact in FP32; accumulation order within a tile is implementation-defined but must be deterministic for a given configuration. Cross-tile accumulation order is program order.
- SFU: `VEXP2`, `VRCP`, `VLOG2`, `VSIGMOID` within 2 ulp of the BF16 rounding of the IEEE double result over the BF16 input domain; `VEXP2` must return exactly 0 for inputs ≤ −127 and must not produce NaN for `−inf`.
- Dequant: exact BF16 rounding (RNE) of `x × scale` computed in FP32.
- Conformance tolerance for whole kernels against the PyTorch reference: max abs error ≤ 2⁻⁷ × row max, as in the plan.

---

## 10. Faults and status

`FINISH` writes a status word: `{ok, fault_code, queue, pc, edge, aux}`. Fault codes:

| Code | Raised by |
|---|---|
| `TOKEN_LEAK`, `TOKEN_TIMEOUT` | §8.3 |
| `UNTOKENED_READ`, `UNTOKENED_WRITE`, `VALID_MISMATCH`, `ACC_NOT_CONVERTED` | simulator only, §8.4 |
| `IDX_RANGE`, `IDX_SHORT` | `LOAD.IDX`, `STORE.IDX` (hardware and simulator) |
| `PAGE_FAULT` | DMA translation |
| `BAD_EDGE`, `BAD_QUALIFIER`, `BAD_SHAPE` | decode |

Hardware faults stop all queues; the host reads status and resets.

---

## 11. Implementation parameters (runtime-discoverable)

The compiler reads these from a device descriptor; the ISA does not fix them.

| Parameter | v1 KV260 | 22FDX config C (model) |
|---|---|---|
| `T_M, T_K, T_N, T_V, T_VR, T_S` | 64, 128, 128, 64, 512, 128 | same |
| Slots: INP / WGT / ACC / VEC / STATE | 2 / 4 / 2 + 1 wide / 16 / 1 | 8 / 16 / 8 + 2 wide / 64 / 2 |
| FIFO depth per edge | 4 | 16 |
| DMA channels (read / write) | 1 / 1 | 8 / 2 |
| `LOAD.IDX` rows in flight | 16 | 256 |
| Front-end registers | 16 | 16 |
| `T_DEADLOCK` | 10⁶ | 10⁷ |
| Chips (COMM) | 2 (emulated) | N (simulated) |
| Supported qualifiers | all of §4 | all of §4 |

---

## 12. Encoding sketch (non-normative)

128-bit instruction words. The front end needs only: queue id (3 b), opcode (6 b), and whether any field is loop-variant. The rest is queue-private.

```
[127:125] queue   [124:119] opcode   [118:112] qualifier   [111:96] flags (accumulate, wide, inplace, rows_valid/cols_valid class)
[95:64]   operand A (scratchpad 3 b | slot 8 b | loop-variant 1 b | reg-indirect 1 b | ...)
[63:32]   operand B / index tile / mask
[31:0]    immediate: nrows, row_bytes, k, count, or DRAM descriptor id
```

DRAM region descriptors (`region`) live in a 64-entry descriptor table written by the host; instructions reference them by id. Loop-variant fields add `stride_i` from a per-loop increment table.

---

## 13. Reserved for v2

- `MMA.MX acc, a, b, scale_a, scale_b`: array consumes FP4/FP8 with block scales from VEC scale tiles; `f4.b32` and `f8.b32` become legal destinations for WGT. Added without changing any v1 instruction.
- **Scoreboard mode**: a device capability bit under which `PUSH`/`POP` are checked rather than required; the front end derives RAW/WAR hazards from tile indices. v1 compilers that emit full tokens remain correct.
- `SEND`/`RECV` over a physical link with remote scratchpad writes; `RECV` gains a timeout operand.
- Three-bit expert formats (`f3.b32`) if the model side adopts them.

---

## 15. Findings from the TileLang prototypes (FA, MoE)

Recorded here so the freeze reflects them; details in `docs/tilelang-cutoff-verdict.md`.

- Software pipelining in the instruction stream (prologue/steady/epilogue) is a single-thread-stream artefact. On this machine, overlap across iterations of a dynamic-count loop comes from the decoupled queues themselves: with a 2-slot INP ring and the release token issued after the last MMA on a slot, the LOAD queue gathers tile `i+1` while MMA runs tile `i`. The compiler therefore does not pipeline dynamic loops; it allocates rings.
- Dequant-on-load changes the pipeline stage of the dequant: it is producer work. A scheduler that keeps it with the consumer refills the ring slot before the GEMM reads it.
- MoE combine is a pure scatter plus a reduce: expert outputs go to a `[B, k, d]` buffer by `STORE.IDX` at row `token·k + slot`, and a second kernel sums the `k` rows per token with the routing weights (`LOAD.IDX` + VEC). No scatter-add in the DMA.
- The per-edge FIFO ordering rule (§8.2, rule 3) is now the construction rule of the token-insertion pass, not a repair.
- Persistent MoE (phase 3): with rings on INP (2), WGT (3) and ACC (2) and a release token per slot, the next expert's gather is issued while two GEMMs of the current expert are still queued, with no prologue/epilogue in the instruction stream. Overlap depth equals ring depth minus one GEMM; it is a resource number, not a schedule. Two things must still be in the stream: double-buffered output tiles (a single-buffered scale→`STORE.IDX` round trip serializes each N-tile) and a flattened W2 stream (a ring restarted per loop re-primes).
- The expert loop had to be unrolled (8,025 instructions for 32 experts) because loop-variant operands were innermost-only; fixed in v0.1.2 by per-level increments.
- KDA (phase 4): with LOAD as the only writer of WGT, every product of two device-computed matrices cost a STORE→DRAM→LOAD round trip (13 per chunk); fixed in v0.1.3 by `VCVT dst=WGT/INP`. The additive `MMA.MASKED` was the wrong primitive for the delta rule (needs 0, not −inf); fixed by the `mul` mode. A 128-row state did not fit the 64-row write path; fixed by `row0`. Plain BF16 operands miss the KDA tolerance by 5%; the hi/lo split costs +35% MMAs and is left to the compiler's numerics budget, with a BF16×2 array mode as the v0.2 hardware alternative.
- Gate 1 closes with five kernel families (GEMM, FA/GQA, MLA, KDA, MoE+combine) through TileLang → #19 → ISA → simulator; the remaining questions (ring depths, FIFO depths, gather cost, VEC/SFU sizing, DRAM round-trip latency) are the cycle model's (`docs/gate1-report.md` §4).

## 14. Open issues for v0.2

Candidates (consolidated in `docs/gate1-report.md` §3; closed items removed):
- BF16×2 (hi/lo) or FP32 operand mode on the array, for KDA-class numerics.
- Transposed A operand (`LOAD.2D transpose` or `MMA` A-transpose), to drop host-side kᵀ/gᵀ layouts.
- `LOAD.COUNT reg, region` or tagged tokens, to keep the MoE trip-count round trip inside the LOAD queue.
- Fused scale-then-exp2 for a row vector (`VEXP2 t, vec, scale`).
- `STORE.2D` f32 output preferred by the compiler when the output dtype is wider than bf16.

Open questions:

1. ~~`VTOPK` over 384 router scores~~ — resolved: VEC row tiles with `T_VR = 512` (§2.2).
2. `RECV.ANY` ordering guarantees across ragged combine; may need a sequence number in the tag.
3. Whether `LOAD.IDX` should accept a second index tile (2-D gather: expert id × token) or the compiler flattens on the host.
4. Exact `band` mask semantics for SWA with replay (window 128 across tile boundaries).
5. Whether `LOAD.IDX` should take a destination tile *group* so a 576-wide row lands in 4.5 tiles with one instruction and one DRAM read, instead of the `col0` repeat form; decide from the cycle model's gather-efficiency numbers.

---

## Appendix A. Normative DRAM layouts for block-scaled formats

**`f8.b32` / `f4.b32` (weights).** Matrix `[K, N]` stored as a grid of `32×32` blocks, row-major over blocks, each block row-major; payload array followed by a `[K/32, N/32]` array of UE8M0 scales. For `f4`, two elements per byte, low nibble first.

**`f4.r16` (KV entry).** One entry = 576 channels (512 nope + 64 rope) of E2M1 packed 2/byte = 288 B, followed by 36 E4M3 scales (one per 16 channels), followed by 60 B of zero padding, total 384 B, 64-byte aligned. Entries are contiguous with `stride[0] = 384`.

**`f8.t` (Engram rows).** Row of 256 E4M3 values, FP32 scale in the descriptor, 256 B, 64-byte aligned.

## Appendix B. Triangular-solve macro (KDA)

For `C = 64` and `A = I + L` with `L` strictly lower-triangular, computed as `A^{-1} = Σ_{j=0}^{5} (−L)^j` by repeated squaring:

```
T0 = -L                          (VEC: negate masked ACC tile)
P  = I + T0                      (VEC add identity from VEC tile)
for s in 1..log2(C):             (6 steps)
    T = T × T                    MMA.MASKED (lower mask), ACC
    P = P × (I + T)              MMA, then VADD
```

Exact for strictly lower-triangular `L` of size ≤ 64. Cost: 12 MMAs + 8 VEC ops per chunk; promote to an instruction only if the cycle model shows it on the critical path.
