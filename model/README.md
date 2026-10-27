# Module 0: workload / chip / TCO model

`module0.py` is the "Module 0" doc as code. Standard library only.

```
python model/module0.py                       # headline config: C @ 22FDX, 8 x LPDDR5X, B=256, 30 ms
python model/module0.py --sweep               # array x channels, batch, alpha, eta_bw, node
python model/module0.py --chip C-12lp-hbm --B 512 --alpha 0.4
python model/module0.py --json
```

Structure:

- `Workload`: DeepSeek V4.1-Flash from its published config; derives active params
  (16.1 B / 8.1 B, matching the model card), resident bytes (298 GB), FLOP/token,
  bytes/step as a function of batch (expert-touch formula), KV read/token.
- `Chip` + `NODES_APPROX` + `MEM_APPROX`: array, scratchpad, memory PHY, node
  densities, shuttle price, pJ/MAC. Everything in `*_APPROX` is a planning estimate
  to be replaced by PDK/vendor data; model parameters and shuttle prices are sourced.
- `size_replica()`: N chips from the bandwidth, compute and capacity constraints;
  step time, MFU, B*, throughput at B and at B*, TTFT for a 128 K prefill.
- `tco()`: capex + amortized NRE + 3-year energy -> $/Mtoken at B and at B*.

Reading the two $/Mtoken numbers: the first assumes the scheduler assembles exactly
the configured batch; the second assumes it can fill B*, where the array stops
idling. For a bandwidth-bound decode machine the gap between them is the value of
the serving system, not of the chip.

What to calibrate from the cycle model, in order of leverage: `eta_bw` under
random 384-byte gathers, then `eta_mfu` on the MoE grouped GEMM at ragged
`tokens_e`, then `pj_per_mac`. Nothing else in the model moves the answer by
more than the uncertainty in those three.

## Phase 5: η from the cycle model (H4)

`module0.py` now reads `eta_bw` / `eta_mfu` for the config-C presets from
`model/cycle_results.json` (written by `python model/sweeps.py`, the cycle-model
sweeps of `docs/cycle-model-report.md`) instead of the 0.7 / 0.5 placeholders.
The values are the MoE decode tile's (weight stream + token gathers) under the
config-C timing descriptor; the MLA tile's are printed alongside.

```
python model/module0.py                              # cycle-model η, task's DMA rule (one request, one channel)
python model/module0.py --cycle-key module0_striped  # what-if: a tile request striped over the 8 channels
python model/module0.py --placeholders               # the 0.7 / 0.5 planning values
```

Headline (C @ 22FDX, 8 × LPDDR5X, B = 256, 30 ms):

| η source | η_bw | η_mfu | N chips / replica | $/Mtoken at B = 256 | at B* = 751 |
|---|---|---|---|---|---|
| placeholders | 0.70 | 0.50 | 53 | 0.093 | 0.038 |
| cycle model (task rule) | 0.146 | 0.113 | 255 | 0.249 | 0.093 |
| cycle model (striped) | 0.181 | 0.141 | 204 | 0.210 | 0.079 |

Read before using: these are the efficiencies of one in-order program whose
LOAD stream keeps the eight channels at 15–18 % of peak (report §3, §4.7). If a
config-C chip runs several such programs against its channels, η_bw scales
with their number until the channels saturate; the descriptor does not say,
so the one-program value is what the model records. `--sweep` keeps the η_bw
sweep (0.3–0.9) for that question.
