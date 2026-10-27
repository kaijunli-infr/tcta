# TCTA

A decoupled tile accelerator for LLM decode, as code: the ISA
(`isa/isa-v0.1.md`, v0.1.4), a functional simulator with token checking
(`sim/`), a cycle model on top of it (`sim/tcta/cycle.py`), the NumPy
reference kernels and fixtures (`ref/`), the TileLang cut-off compiler
prototype that maps five kernel families (GEMM, FlashAttention/GQA, MLA
decode, KDA chunkwise, MoE decode) from TileLang's pass #19 to the ISA
(`compiler/prototype/`), and the workload/chip/TCO model that the cycle model
feeds (`model/`).

Start with `HANDOFF.md` (state of the work, what to read in which order), then
`docs/tilelang-cutoff-verdict.md` (one section per phase, with the findings)
and `docs/cycle-model-report.md` (the timing model, its sweeps and the
calibration list), then `docs/phase6-mlp-report.md` (rings, cores and the
η values Module 0 uses).

## Running it

No GPU. Python 3.14 on aarch64 is what the work was done on; the pinned
TileLang wheel and its sha256 are in `compiler/prototype/PINNED.md`.

```
python -m venv .venv && . .venv/bin/activate
pip install tilelang==0.1.15 numpy pytest
python ref/kernels.py fixtures        # fixtures/ is not in git (433 MB); regenerates it
python -m pytest tests -q             # 84 passed
```

Regenerate the derived artifacts:

```
python compiler/prototype/dump_programs.py      # ISA program listings in compiler/prototype/dumps/
python compiler/prototype/cut.py                # cut PrimFuncs (TileLang pass #19 output)
python compiler/prototype/overlap_report.py --markdown
python model/sweeps.py                          # phase-5 cycle-model sweeps -> model/cycle_results.{json,md}
python model/sweeps_mlp.py                      # phase-6 ring / core / striping sweeps -> model/mlp_results.md + keys in cycle_results.json
python model/module0.py                         # headline chip/TCO config with the measured eta_bw / eta_mfu
```

## Layout

| Path | What |
|---|---|
| `isa/` | ISA specification (v0.1.4: Gate 1 freeze candidate, §11 device descriptor revised after the cycle model) |
| `sim/tcta/` | functional simulator (`machine.py`, `asm.py`, `isa.py`, `dtypes.py`); cycle model (`timing.py` descriptors, `cycle.py` single- and multi-core) |
| `ref/` | NumPy reference kernels and the fixture generator |
| `compiler/prototype/` | TileLang kernels for the five families, the pass-#19 cut, the pattern pass `tile_tir_to_program.py`, program dumps, `PINNED.md` |
| `model/` | `module0.py` (workload, chip, TCO), `sweeps.py` (phase 5 sweeps), `sweeps_mlp.py` (phase 6: rings, cores, striping) and their results |
| `docs/` | task statements per phase, verdicts, Gate 1 report, cycle-model report, expressivity and pass notes |
| `tests/` | simulator, reference, cut, MoE, MLA/KDA/GEMM and cycle-model tests |
