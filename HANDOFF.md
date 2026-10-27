# Handoff: TileLang cut-off task, phases 1–6 complete (2026-10-27)

For the session steering `docs/tilelang-lowering-cutoff-task.md`, its
addendum, `docs/tilelang-phase2-moe-task.md`, `docs/tilelang-phase3-task.md`,
`docs/tilelang-phase4-task.md`, `docs/phase5-cycle-model-task.md` and
`docs/phase6-mlp-task.md`. All deliverables of the six phases exist; the whole
repo is green on the v0.1.4 simulator (`pytest tests -q`: **84 passed**). The
ISA changed only where the main session sent deltas (v0.1.4 = §11, the device
descriptor); the simulator gained the phase-3 trace list (approved), the
phase-5 cycle model (`sim/tcta/timing.py`, `sim/tcta/cycle.py`) and the
phase-6 multi-core replay in the same two modules; `machine.py` is untouched
beyond the trace list. **This repo is canonical** (process change, phase 6):
the main session sends spec and doc deltas only, no `sim/` copies. Remote:
`github.com/kaijunli-infr/tcta` (private).

**Phase 1: PASS. Phase 2: PASS. Phase 3: PASS. Phase 4: PASS-WITH-CAVEATS
(closed by v0.1.3, Gate 1 closed). Phase 5: PASS. Phase 6: PASS.** The
phase-6 verdict section and `docs/phase6-mlp-report.md` §7 carry eight
findings for the main session; the two that matter most are the descriptor
decision (config C = 4 cores × one 64×64 array, 2.1–2.9× the 4-array single
core on the same silicon) and that the recommended configuration lifts the
MoE tile's η_bw from 0.146 to 0.644, which moves Module 0 from 255 to 58
chips per replica ($0.249 → $0.096 per Mtoken), within 10 % of the planning
placeholders.

## Phase 6 in one paragraph

ISA v0.1.4 §11 applied (FIFO 8, INP 5, WGT 32, striping 2, A-operand
broadcast, cores). `DeviceTiming` gained `request_channels`,
`operand_broadcast`, `cores`; `CONFIG_C` carries §11 as one stream with a
4-array MMA unit, `CONFIG_C_CORE` the same chip as four one-array cores,
`CONFIG_C_PHASE5` the phase-5 descriptor (so `model/sweeps.py` still
reproduces phase 5). `cycle.run_multicore` replays c programs on c cores
against the shared channels through an arrival-order arbiter (ties rotate),
reports per-core `stall_arbiter`, and checks every core's functional result.
I1: the MoE ring knee is 12 (6481 cycles per tile, η_bw 0.304; ring 16 is
worse) and the MLA K-ring knee 10 (2177, 0.155); the knee is the tile's own
serial chains (SFU 3331 cycles, 6144 array-cycles of accumulation per MoE
tile), not Little's law; rings deeper than a loop's trip count are not rings,
so `[W1|W3]` is streamed as one 16-tile loop (`flat_w13`). I3: 4 cores × 8
experts give 3063 cycles per tile chip-wide, η_bw 0.644, η_mfu 0.502,
channels 52 % busy, per-core completion within 1.4 %; MLA copies 758 per KV
tile, η_bw 0.446, with an arrival-order convoy (+4.8 % per core index). I4:
striping 2 is −5.7 % at 1–2 cores and 0 at 4; 8-way costs 27 %. I5: Module 0
reads `module0_recommended` by default. I6: 4 cores × 1 array, INP 5, WGT 32,
FIFO 8, striping 2.

## What to read first

1. `docs/phase6-mlp-report.md`: §0 the result, §2 ring sweep, §4 cores, §5
   striping, §6 Module 0, §7 the descriptor decision and the lever-by-lever
   η_bw argument, findings.
2. `docs/tilelang-cutoff-verdict.md`, Phase 6 section: verdict, the answers in
   one table, findings for the main session, calibration additions.
3. `docs/cycle-model-report.md`: the timing model (§1, with the phase-6 rules
   for striping, broadcast, cores and the arbiter at the end), phase-5 sweeps
   and calibration list.
4. `model/mlp_results.md` (phase-6 tables), `model/cycle_results.md` (phase-5
   tables), `model/README.md` (the η rows before and after).
5. `compiler/prototype/PINNED.md` "Phase 6 notes": rings vs trip counts, the
   expert-subset programs, the K-ring span, FIFO ≥ r·(d − 1), the multi-core
   rules.

## Deliverables (phase 6; earlier phases unchanged, see the verdict)

| # | Path | Status |
|---|---|---|
| §0 | `isa/isa-v0.1.md` (v0.1.4 §11 as received), `docs/phase6-mlp-task.md`; `sim/tcta/isa.py` (`DEVICE_CONFIG_C`, FIFO 8); `sim/tcta/timing.py` (`CONFIG_C`, `CONFIG_C_CORE`, `CONFIG_C_PHASE5`) | done |
| I1 | `moe_kernel.build_moe_persistent_kernel(num_stages, flat_w13)`, `mla_kernel.build_mla_kernel(inp_slots=5, kv_unroll)`; `model/sweeps_mlp.py --only ring`; dumps `program_moe_flat_ring12.txt`, `program_mla_kv2.txt` | done |
| I2 | `cycle.run_multicore`, `MultiTiming`, `_Core`, arbiter, `stall_arbiter[_gather]`, `gather_bound`; `tests/test_cycle.py` (+4) | 20 passed |
| I3 | `build_moe_persistent_kernel(expert_base, n_experts)`, `sweeps.fam_moe(..., expert_base, n_experts)`; `sweeps_mlp.py --only cores` | done |
| I4 | `sweeps_mlp.py --only striping` | done |
| I5 | keys `module0_recommended`, `module0_1core` in `model/cycle_results.json`; `module0.py` default key + fallback; `model/README.md` | done |
| I6 | `docs/phase6-mlp-report.md`; verdict Phase 6 section | done |
| + | `model/sweeps.py` pinned to `CONFIG_C_PHASE5` and merging into the results file; `docs/cycle-model-report.md` §1 phase-6 rules; `sim/README.md` | done |

## Key numbers (config C, cycles at 1 GHz; phase-5 numbers in the Phase 5 section of the verdict)

- Ring sweep, MoE token tile: 13554 (phase 5) → 9974 (§1 descriptor, ring 3)
  → 8561 (flat `[W1|W3]`, 3) → 7116 (6) → 6481 (12) → 6750 (16); η_bw 0.146 →
  0.304. MLA KV tile: 3665 → 2772 (K ring 4) → 2177 (10) = 2177 (20, FIFO 36
  needed).
- Cores: MoE per tile chip-wide 9314 / 4812 / 3063 on 1 / 2 / 4 one-array
  cores (η_bw 0.212 / 0.410 / 0.644, η_mfu 0.165 / 0.319 / 0.502, channels
  17 / 33 / 52 %); the 4-array single core 6481 (0.304 / 0.237). MLA 2561 /
  1356 / 758 (η_bw 0.132 / 0.249 / 0.446); 4-array single core 2177.
- Striping × cores (MoE): k = 2 −5.7 % / −5.6 % / +0.4 % at 1 / 2 / 4 cores;
  k = 8 −4.8 % / +10.5 % / +26.6 %.
- Module 0: η_bw 0.644 / η_mfu 0.502 → N = 58, $0.096/Mtoken ($0.036 at B*);
  phase 5 255 / $0.249; 1 core × 4 arrays 122 / $0.146; placeholders 53 /
  $0.093. 12LP+ with HBM2e: 76 → 21 chips.

## Decisions taken while implementing (phase 6)

- Rings are set by `num_stages`, capped at the pipelined loop's trip count;
  to reach ring 12 the MoE W13 stream had to be flattened (`flat_w13`), with
  x resident in 2 INP slots. This is a program-shape change beyond "ring
  depth" and is recorded as such (report §7 finding 5).
- The MLA K ring is deepened by spanning `kv_unroll` KV tiles (5 slots each);
  `u = 1` is byte-identical to the phase-5 form.
- The ring sweep runs at FIFO 8 and 32 to separate ring effects from FIFO
  effects; the core and striping sweeps and the Module 0 keys use FIFO 8 (the
  §1 decision), which costs nothing at the recommended rings.
- Arbiter: arrival order with a rotating tie-break (the event loop replays
  all cores' heads in ready-time order). The wait includes queueing behind
  the core's own requests, so single-core runs report it too.
- The idle-channel multi-core test uses 8 write channels: with config C's 2
  the GEMM's 32 KB f32 stores already contend at 2 copies.
- `MultiTiming.eta_mfu` charges all arrays of the run; `eta_mfu_chip(4)` is
  used in every table so 1-core and 4-core rows compare.
- `model/sweeps.py` now runs on `CONFIG_C_PHASE5` and merges into
  `cycle_results.json`, so the phase-5 tables and keys reproduce and a rerun
  does not drop the phase-6 keys.
- `max_outstanding_rows` 512 (not §11's 256): the window already binds on 16 %
  of gather bursts at 4 cores.

## Environment

- No GPU, aarch64. Interpreter: `~/Projects/tilelang-cutoff/.venv/bin/python`
  (tilelang 0.1.15, bundled TVM 0.25.dev0, Python 3.14); `pip install
  tilelang==0.1.15 numpy pytest` recreates it (sha256 in PINNED.md).
- Run everything: `python -m pytest tests -q` from the repo root (~30 s).
- Regenerate: `python ref/kernels.py fixtures` (fixtures, not in git),
  `python compiler/prototype/dump_programs.py` (program listings),
  `python compiler/prototype/cut.py`, `python compiler/prototype/overlap_report.py --markdown`,
  `python model/sweeps.py` (phase-5 tables, ~65 s), `python model/sweeps_mlp.py`
  (phase-6 tables, ~100 s), `python model/module0.py [--sweep] [--cycle-key ...]`.

## Not done / out of scope

Power, area, RTL, Verilator, COMM between cores or chips, V4.1-Flash shapes,
FP8/MX attention, DSpark, autotuning, the MLA K-split reduction (moot for a
one-array core). Open for the main session: the arbiter's grant policy
(convoy on identical programs), `sfu_lanes` (now on the MoE critical path),
and the request-burst queueing that keeps η_bw at 0.64 with 52 % channel
utilization.
