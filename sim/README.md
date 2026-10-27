# TCTA functional simulator (ISA v0.1)

Python reference model for the tile-centric tensor accelerator. It is the
contract between the compiler and the RTL: a kernel is *conformant* when it
runs here with token checking on and matches the PyTorch reference.

```
sim/tcta/dtypes.py    element formats, BF16 emulation, dequant for every §4 qualifier
sim/tcta/isa.py       queues, scratchpads, token-graph edges, instruction/record types
sim/tcta/machine.py   execution: five queues, token FIFOs, tile-level token checker, front end
sim/tcta/asm.py       Python DSL for writing instruction streams
tests/test_sim.py     conformance tests (GEMM, FA inner loop, LOAD.IDX, VTOPK/VCOUNT/LOOP, faults)
```

Run:

```
pip install numpy pytest
PYTHONPATH=sim python -m pytest tests -q
```

## What it checks (ISA §8.4)

| Fault | Raised when |
|---|---|
| `UNTOKENED_READ` | a queue reads a tile written by another queue with no POP on that edge since the write, or reads a never-written tile |
| `UNTOKENED_WRITE` | a queue overwrites a tile that another queue read, with no POP on the reader→writer edge since that read |
| `VALID_MISMATCH` | MMA operand valid regions disagree, or accumulate into an ACC tile of a different valid shape |
| `ACC_NOT_CONVERTED` | MMA takes A from an ACC tile that was not `VCVT ... inplace`'d |
| `IDX_RANGE` / `IDX_SHORT` | `LOAD.IDX` index out of the region, or index tile shorter than `nrows` |
| `TOKEN_LEAK` | a token FIFO is non-empty at FINISH |
| `TOKEN_TIMEOUT` | no queue makes progress for `t_deadlock` steps |
| `BAD_EDGE` | PUSH/POP on an edge not in §8.1 |

The token checker is tile-level and time-based: it proves that *a* token
crossed the edge after the write/read it is protecting, which catches every
missing PUSH/POP and every ring-buffer overrun, but not a schedule that pairs
the right number of tokens with the wrong tiles. The cycle model and RTL
testbench get the exact pairing from the compiler's token insertion pass.

## Writing a kernel

See `test_flash_attention_inner_loop` in `tests/test_sim.py` for the FA
inner loop with a 2-stage K/V ring buffer, online softmax, P read from ACC in
place, and bank release tokens. The pattern for a ring of `n` banks:

1. the consumer pre-`PUSH`es `n` release tokens before the loop;
2. the producer `POP`s one release before every fill;
3. the consumer `PUSH`es one release after every use;
4. the producer drains the last `n` releases after the loop.

Loop-variant operands use `incr=`: `{"region.base": bytes}` advances a DRAM
base per iteration, `{"dst_mod": n}` / `{"b_mod": n}` rotate a tile slot
through an `n`-entry ring.

## Not yet implemented (open for v0.2)

- `LOAD.3D` is an alias of `LOAD.2D`; head/page dimension must be folded by the assembler.
- `STORE.IDX`, `RECV.ANY`, N-chip COMM with latency and bandwidth (single-machine mailbox only).
- Nested-loop increments (innermost loop only).
- `VTOPK` over 2-D tiles (`axis=0`).
- Cycle model: `stats` counts instructions, bytes and gathers; no timing.

Step trace (phase 3, F3): set `m.trace = []` before `m.run(...)` and the machine
appends `(step, queue, op, pc)` for every executed instruction, syncs included.
No semantic effect.

## Cycle model (phase 5)

`tcta/timing.py` holds `DeviceTiming` (every timing parameter, named and defaulted
for the v1 KV260 and for config C; all planning values) and `tcta/cycle.py` the
`CycleMachine(Machine)`: the same functional run, then a discrete-event timing
pass over the recorded per-queue instruction streams (in-order queues, tokens at
the producer's completion, FIFO depth, array/DMA-channel/VEC units, scratchpad
bank ports, front-end dispatch). `m.timing_report` is a `Timing` with cycles,
per-queue busy/stall cycles, per-edge FIFO levels and blocked cycles, channel
utilization, bank conflicts and a Gantt trace (`.trace()`, `.step_trace()` in
the `Machine.trace` shape). It never changes numerics; `tests/test_cycle.py`
and `model/sweeps.py` use it. Rules: `docs/cycle-model-report.md` §1.

Phase 6 (ISA v0.1.4 §11): `DeviceTiming` gained `request_channels` (a tile
request striped over k channels), `operand_broadcast` (the arrays of one core
read one A tile at once) and `cores`; `CONFIG_C` carries the §11 numbers
(FIFO 8, WGT 32, striping 2, broadcast), `CONFIG_C_CORE` is the same chip as
four one-array cores, `CONFIG_C_PHASE5` the phase-5 descriptor.
`cycle.run_multicore(timing, [dict(instrs=..., load=..., cfg=..., dram_bytes=...), ...])`
replays c programs on c cores (own queues, scratchpads, FIFOs, arrays, VEC
pipe) against the shared DMA channels through an arrival-order arbiter (ties
rotate across cores) and returns a `MultiTiming` (chip cycles, η_bw, η_mfu,
channel utilization, per-core `Timing` with `stall_arbiter`). The multi-core
runs check every core's functional result against the single-core run
(`tests/test_cycle.py`, `model/sweeps_mlp.py`).

