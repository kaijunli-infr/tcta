# Handoff: TileLang cut-off task, phases 1–5 complete (2026-10-27)

For the session steering `docs/tilelang-lowering-cutoff-task.md`, its
addendum, `docs/tilelang-phase2-moe-task.md`, `docs/tilelang-phase3-task.md`,
`docs/tilelang-phase4-task.md` and `docs/phase5-cycle-model-task.md`. All
deliverables of the five phases exist; the whole repo is green on the v0.1.3
simulator (`pytest tests -q`: **80 passed**). No ISA change was made in any
phase; the simulator gained the phase-3 trace list (approved) and, in phase 5,
two new modules (`sim/tcta/timing.py`, `sim/tcta/cycle.py`) that leave
`machine.py` untouched.

**Phase 1: PASS. Phase 2: PASS. Phase 3: PASS. Phase 4: PASS-WITH-CAVEATS
(closed by v0.1.3, Gate 1 closed). Phase 5: PASS.** The phase-5 verdict
section and `docs/cycle-model-report.md` §7 carry eight findings for the main
session; the two that matter most are that the request → channel mapping and
the arrays' operand sharing (both outside the task's rule set) decide the
config-C numbers, and that one in-order program keeps the eight config-C
channels at 13–18 % of peak, which moves Module 0 from 53 to 204–255 chips per
replica.

## Phase 5 in one paragraph

§0: v0.1.3 unpacked, the trace list re-applied, KDA re-emitted with `VCVT
dst=WGT|INP` for computed operands and `VCVT dst=STATE, row0` for the state
halves: 521 → 352 instructions, 13 DRAM round trips + 64 KB state reload per
chunk → none, numerics identical; `MMA.MASKED mul` unusable there because the
masked products are transposed (`MMA.T`). H1: `DeviceTiming` (every parameter
of task §2, `KV260` and `CONFIG_C`) and `CycleMachine`, a post-mortem
discrete-event replay of the functional run's recorded dispatch order and
per-queue streams (in-order queues, tokens at the producer's completion, FIFO
depth, arrays / DMA channels / VEC pipe, bank ports, front end); it reports
cycles, per-queue stalls, per-edge FIFO levels, channel and bank utilization
and a Gantt trace, and raises `CycleDeadlock` when the modelled FIFO depth
leaves no head able to proceed. H2: 16 sanity tests. H3: eight sweeps (the six
of the task plus two what-ifs) in `model/sweeps.py`, 150 timed runs each
checked against its fixture, tables in `model/cycle_results.md`, the report in
`docs/cycle-model-report.md`. H4: `model/module0.py` reads η_bw / η_mfu from
`model/cycle_results.json`. H5: verdict section, Gate 1 report §4 answered,
calibration list.

## What to read first

1. `docs/cycle-model-report.md`: the model as implemented (§1), the answers
   with numbers and recommended values (§4), Module 0 (§5), calibration (§6),
   findings (§7).
2. `docs/tilelang-cutoff-verdict.md`, Phase 5 section: verdict, the answers
   in one table, findings for the main session.
3. `docs/gate1-report.md` §4 (answered), §3 (v0.2 candidates, with the
   v0.1.3 closures).
4. `model/cycle_results.md` (every sweep table) and `model/README.md` (the
   η change).
5. `compiler/prototype/PINNED.md` "Phase 5 notes": the v0.1.3 D4 forms, the
   two modelling rules that matter when reading numbers.

## Deliverables (phase 5; earlier phases unchanged, see the verdict)

| # | Path | Status |
|---|---|---|
| §0 | `isa/`, `sim/` core, `tests/test_sim.py` from the v0.1.3 sync; `tile_tir_to_program.py` (vcvt_operand / vcvt_state / vcvt_copy / vcopy / vneg / mma_masked_mul); `kda_kernel.py` (v0.1.3 form); `kda_kernel_roundtrip.py` (v0.1.2 form kept); `mla_kernel.py` (`inp_slots`); dumps regenerated (+ `program_kda_roundtrip.txt`, `program_mla_inp6.txt`); Gate 1 report header + KDA rows | done |
| H1 | `sim/tcta/timing.py`, `sim/tcta/cycle.py`; `sim/README.md` paragraph | done |
| H2 | `tests/test_cycle.py` | 16 passed |
| H3 | `model/sweeps.py`, `model/cycle_results.json`, `model/cycle_results.md`, `docs/cycle-model-report.md` | done |
| H4 | `model/module0.py` (`apply_cycle_model`, `--cycle-key`, `--placeholders`), `model/README.md` | done |
| H5 | verdict Phase 5 section; `docs/gate1-report.md` §4 | done |
| + | `tests/test_tilelang_mla_kda.py`: v0.1.3 KDA shape/round-trip tests, the round-trip form and the INP variants still match | 18 passed |

## Key numbers (config C unless stated; cycles at 1 GHz)

- Baseline per unit: GEMM 2013 (program); FA 1177 / KV tile; MLA 4603 / KV
  tile (η_mfu 0.059, η_bw 0.132); KDA 15176 / chunk (0.048, 0.085); MoE
  13554 / token tile (0.113, 0.146). KV260: MoE η_mfu 0.946, MLA 0.675, FA
  0.294, GEMM 0.333; KDA deadlocks at FIFO 4 (321303 cycles at depth 16).
- INP 5: MLA tile −20.4 %, DRAM 2304 → 1224 KB. WGT ring 4: MoE tile −14.5 %
  (ring 2 +10.1 %). FIFO 8 everywhere (4 deadlocks KDA). `row` vs `burst64`
  ≤ 0.5 % at config C, +5.3 % on KDA at KV260. 64 lanes + fused SFU: MLA
  4603 vs 7865 at 16 unfused; FA 1177 vs 3047, still 70 % VEC. KDA v0.1.3 vs
  v0.1.2: 15176 vs 26535 per chunk at latency 300. MoE prefetch −7 %.
- What-ifs: tile striped over 2 channels MoE −20 %, MLA −17 %, KDA −17 %;
  4 read ports MoE −3.8 %.
- Module 0: η_bw 0.146 / η_mfu 0.113 → N = 255, $0.249/Mtoken (placeholders
  53, $0.093; striped 204, $0.210).

## Decisions taken while implementing (phase 5)

- A PUSH never holds its queue unless the FIFO is full (task §1 "free";
  §4 "multiple outstanding LOADs"); the first model version did and
  serialized every LOAD stream — the numbers in this handoff are after the
  fix.
- Bank-port holds are intervals; a hold that starts later than the cycle
  considered does not block (queues are replayed in ready-time order).
  Port conflicts are therefore a lower bound.
- MMA ports are held for the array occupancy, not through `mma_latency`
  (accumulate forwarding, task §4); a gather holds its index vector's read
  port only while issuing requests.
- `request_channels` and `ports_per_bank` readers > 1 added as what-ifs
  because the sweeps showed those rules set the config-C numbers; both
  default to the task's rules, and Module 0 records the task-rule values.
- KDA's `VGATE mul` masks kept (no transposed `MMA.MASKED`); `kda_kernel_roundtrip.py`
  duplicates the v0.1.2 authoring rather than flagging the new kernel, so the
  before-case is a real program.
- The MLA `inp_slots` variants Python-unroll the chunk loop (static slot
  indices); the default form is byte-identical to phase 4.

## Environment

- No GPU, aarch64. Interpreter: `~/Projects/tilelang-cutoff/.venv/bin/python`
  (tilelang 0.1.15, bundled TVM 0.25.dev0, Python 3.14). Not in the tarball;
  `pip install tilelang==0.1.15 numpy pytest` recreates it (sha256 in PINNED.md).
- Run everything: `python -m pytest tests -q` from the repo root (~30 s).
- Regenerate: `python ref/kernels.py fixtures` (fixtures),
  `python compiler/prototype/dump_passes.py --arch sm_80` (D2),
  `python compiler/prototype/cut.py` (cut dumps),
  `python compiler/prototype/dump_programs.py` (program listings),
  `python compiler/prototype/overlap_report.py --markdown ...` (F3),
  `python model/sweeps.py [--only baseline,inp,...]` (cycle-model tables, ~65 s),
  `python model/module0.py [--cycle-key module0_striped | --placeholders]`.

## Not done / out of scope

Power, area, RTL, Verilator, multi-chip timing (the COMM link is priced,
never exercised), V4.1-Flash shapes, FP8/MX attention, DSpark, autotuning.
The one-program-per-chip question behind the η_bw finding is the main
session's. No ISA change; the findings are in the verdict (Phase 5) and
`docs/cycle-model-report.md` §7.
