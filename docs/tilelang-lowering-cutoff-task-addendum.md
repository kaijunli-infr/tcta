# Addendum after D1/D2 review: decisions and the D3–D6 spec

Read with `tilelang-lowering-cutoff-task.md`. D1 and D2 are accepted. The
pass table, the sm_90/sm_80 finding and the three deciding-pass descriptions
were checked against `dumps/sm_80/{00,18,19}` and match.

## Decisions

**1. Shell strip is allowed; §6 is amended.** The `T.Kernel` launch shell
(`launch_thread` for `blockIdx.*`/`threadIdx.*`) is CUDA's launch model, not
the tile body. Replace §6 PASS with:

> PASS: D3 is produced by TileLang's own passes up to the cut, with at most
> (a) a scope-rewrite pass, (b) skipping of named passes, (c) a shell-strip
> pass that removes the launch attributes and rebinds `bx, by` as PrimFunc
> parameters, and (d) the event rewrite of decision 2. The body must contain
> no read of `threadIdx.*`; a body that does is still REJECT.

**2. Cut at #19 `InjectSoftwarePipeline`, with `ptx_commit_group` /
`ptx_wait_group` rewritten to abstract pipeline events.** This changes the
earlier "#18 vs #13" question. `19_tl.InjectSoftwarePipeline.py` is already
the schedule the ISA wants:

- `K_shared`, `V_shared` → `[2,64,64]` rings indexed `k % 2`;
- a 2-fill prologue, a steady-state loop of `n − 2` with the fill of
  `(k+2) % 2` placed *after* the `gemm` that read bank `k % 2`, and an
  epilogue;
- `ptx_commit_group()` after each copy = "fill issued";
- `ptx_wait_group(n)` before each `gemm` = "the fill this gemm needs has landed".

Those two intrinsics are the token insertion points. `commit_group` is a
`PUSH LOAD→MMA`; `wait_group(n)` after `c` commits is `c − n − (pops so far)`
`POP`s, which is static at every program point (1 per wait in steady state,
3/2/1/0 in the epilogue). The bank-release direction (free before overwrite)
is also already encoded: every ring fill is placed in program order after the
gemm that last read that bank. D4 turns that placement into `PUSH MMA→LOAD`
after each gemm that reads a WGT bank and `POP LOAD←MMA` before each fill,
except the prologue fills (bank never used yet).

Keep #18 as the fallback cut: if the event rewrite or the scope rewrite fails
on the #19 PrimFunc, D3 returns #18's output and D4 falls back to the fixed
pattern of the original §5.

**3. #5 `MaterializeKernelLaunch` is ACCEPT.** Decision 1 makes the shell
irrelevant; the verdict in the table stands.

**4. Target config for D3**: `sm_80`, or `sm_90` with
`tl.disable_warp_specialized=True`; the IR is identical. Record which was
used in D6.

**5. The TVM commit cannot be pinned**: accepted. `PINNED.md` keeps the wheel
hash (`pip download` + sha256) as the pin instead.

## D3 spec (revised)

`lower_to_tile_tir(prim_func, cut="inject_pipeline") -> PrimFunc` runs, in order:

1. passes 1–19 from `CUDAPassPipelineBody`, with the sm_80 target (or the
   sm_90 config above);
2. **scope rewrite**: `Q_shared → tcta.inp`, `K_shared/V_shared → tcta.wgt`,
   `acc_s/acc_s_cast/acc_o → tcta.acc`, `scores_*/logsum → tcta.vec`.
   Apply it *after* #19: `InjectSoftwarePipeline` re-declares the ring buffers
   (`T.Buffer((2,64,64), scope="shared.dyn")` + `sblock_alloc_buffer(..., data=K_shared.data)`),
   so a scope set before it may not survive. Test both orders and record which
   works in D6; either is allowed by §4.4;
3. **event rewrite**: `T.ptx_commit_group()` → `T.evaluate(T.call_extern("tcta.fill_issued"))`,
   `T.ptx_wait_group(n)` → `T.evaluate(T.call_extern("tcta.fill_wait", n))`,
   and drop `no_implicit_async_commit_wait=1`;
4. **shell strip**: remove the six `launch_thread` statements and the
   `tl.launch_threads` attr; turn `bx`, `by` into `T.int32` PrimFunc params;
   drop `bz` (already folded to 0).

The D3 pytest asserts: no `launch_thread`, no `threadIdx`, no `ptx_*`, no
`mbarrier`/`tma`/`cp_async`, every `sblock_alloc_buffer`/`T.Buffer` carries a
`tcta.*` scope, and the number of `tcta.fill_issued` events equals the number
of `T.copy` into `tcta.wgt`.

## D4 spec (revised): queue partition + event-driven tokens

`tile_tir_to_program(prim_func) -> tcta.Program`, three phases.

**Phase A, statement → (queue, instruction).** Walk the body in program
order, flattening `sblock`s and the prologue/loop/epilogue structure, and
map:

| TIR | Queue | Instruction |
|---|---|---|
| `T.copy(global → tcta.inp/wgt)` | LOAD | `load2d(dst_slot, region, 64, 64)`; `region.base` from the `K[0, by, (k+2)*64, 0]` offset; `dst_slot` from the ring index expression (`k % 2`, `FloorMod(c, 2)`) |
| `T.copy(tcta.acc → global)` | STORE | `store2d` |
| `T.gemm(A, B, C, transA=False, transB=True)` | MMA | `mma_t(ACC, A, B)`; `transB=False` → `mma(..., accumulate=True)`; A in `tcta.acc` → the in-place P (see `acc_s_cast` below) |
| `T.fill(acc_s, 0)` immediately before a non-accumulating gemm | — | dropped (`accumulate=False` on that MMA) |
| `T.fill(vec, x)` | VEC | `vfill(imm=x)` (`-inf` → `-1e30`) |
| `T.fill(acc_o, 0)` | VEC | `vfill(ACC, 0)` |
| `T.reduce(acc, vec, "max", 1, clear=False)` | VEC | `vrowmax` (the preceding `T.fill(scores_max, -inf)` is dropped; `clear=False` with a fresh `-inf` is a plain row max) |
| `T.reduce(acc, vec, "sum", 1, clear=True)` | VEC | `vrowsum`, or fused into `vexp2_sum` when it directly follows the exp2 loop |
| `T.copy(vec → vec)` | VEC | `vfill(dst, 0)` + `vadd(dst, vec=src)` (no VCOPY in v0.1; note as an ISA v0.2 candidate) |
| `T.parallel` loop `v[i] = max(v[i], w[i])` | VEC | `vmax` |
| `T.parallel` loop `v[i] = exp2(a[i]*s − b[i]*s)` | VEC | `vfill(t,0); vadd(t,vec=a); vsub(t,vec=b); vmul(t,imm=s); vexp2(t)` |
| `T.parallel` 2-D loop `acc[i,j] = exp2(acc[i,j]*s − v[i]*s)` | VEC | `vmul(acc, imm=s)` once (hoist the scale), then `vexp2_sum(acc, sum, vec=v_scaled)` where `v_scaled = v*s` |
| `T.parallel` loop `l[i] = l[i]*a[i] + b[i]` | VEC | `vmul(l, vec=a); vadd(l, vec=b)` |
| `T.copy(acc_s → acc_s_cast)` | VEC | `vcvt(acc_s, acc_s, inplace=True)`; `acc_s_cast` is aliased to `acc_s` thereafter |
| `T.parallel` 2-D loop `acc_o[i,j] *= v[i]` | VEC | `vscale_acc` |
| `T.parallel` 2-D loop `acc_o[i,j] /= l[i]` | VEC | `vrcp(l); vscale_acc(acc_o, l)` |
| `tcta.fill_issued` / `tcta.fill_wait(n)` | — | consumed in phase B |

Pattern matching on the `T.parallel` bodies is by expression shape; there
are exactly six shapes in this kernel and all are listed.

**Phase B, tokens from events and from bank reuse.**

- Every `tcta.fill_issued` → `push(LOAD, MMA)` in the LOAD stream right after
  the preceding `load2d`.
- `tcta.fill_wait(n)` → emit `pop(MMA, LOAD)` `(c − n − p)` times in the MMA
  stream before the next MMA, where `c` = fills issued so far in program
  order and `p` = pops emitted so far. Assert the count is ≥ 0.
- Bank release: after each MMA that reads a `tcta.wgt` slot, `push(MMA, LOAD)`;
  before each `load2d` into a `tcta.wgt` slot that has been read before,
  `pop(LOAD, MMA)`. Prologue fills get no pop. Program order in the TIR
  guarantees each pop's token was pushed earlier.
- ACC/VEC hand-offs between MMA and VEC follow the same reader/writer rule on
  `tcta.acc` slots: `push(MMA, VEC)` after an MMA that writes ACC and the next
  reader is VEC; `push(VEC, MMA)` after the last VEC write to an ACC tile the
  next MMA reads (P ready, O rescaled); matching `pop`s before the readers.
  This is derivable from the `T.reads/T.writes` of consecutive sblocks and
  reproduces the two VEC→MMA pushes before the PV gemm.
- STORE: `push(VEC, STORE)` after the final `vscale_acc`, `pop(STORE, VEC)` before `store2d`.

The result must be token-identical in count per edge to
`test_flash_attention_inner_loop` (order within a queue may differ). Add that
as an assertion in D5: compare per-edge push/pop counts.

**Phase C, loop structure.** The steady-state `for k in T.serial(2, ...)`
becomes `LOOP n=2`/`ENDLOOP` with `incr` on the K/V `region.base` and `_mod`
on the ring slots; prologue and epilogue are emitted straight-line. Multiple
query tiles / heads: run one `(bx, by)` per `Machine.run` in D5 (the shell
strip made them parameters), or wrap in an outer `LOOP` later.

## D5 and D6

D5 as specified, plus the per-edge token-count assertion above, and run for
all 8 heads against `fixtures/flash_attn_gqa` with KV head `h // 4`.

D6 verdict must now also state: whether the scope rewrite had to go after
#19; whether every `fill_wait` count came out ≥ 0 (if not, the event model
and the FIFO model disagree and that is a finding, not a bug to patch);
and one sentence on what the real compiler owns after this result (expected
answer: scope inference, the event→token rewrite generalized beyond FA, and
everything LayoutInference/LowerTileOp do for the GPU, which we replace with
tensorize-to-ISA).

## Not changed

Everything in §7 of the task still applies. No ISA or simulator changes; the
two idioms the ISA lacks (`VCOPY`, a fused `scale-then-exp2` for the rescale
vector) are recorded in D6 as v0.2 candidates, not added.
