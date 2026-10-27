# Reference generators

NumPy golden references for the five kernel families plus the DeepSeek
V4.1-Flash decoder-layer slice. Each fixture is `fixtures/<name>/data.npz` +
`meta.json`, holding logical BF16-rounded float32 values, the exact DRAM byte
images for every quantized format (ISA §4 / Appendix A), the expected outputs,
and the tolerance (`tol_rowmax = 2^-7` of each output row's max).

```
python ref/kernels.py fixtures      # ~3 min, ~440 MB; deterministic (seeded)
PYTHONPATH=sim python -m pytest tests -q
```

`fixtures/` is generated, not committed.

| Fixture | Shape | Validates | Size |
|---|---|---|---|
| `gemm_partial` | 64×128 × 128×128, valid 50/100/96 | partial-tile valid regions | 0.1 MB |
| `flash_attn_gqa` | 8 q-heads / 2 kv-heads, L=256, d=64, causal | FA prefill, GQA head stacking; `m`,`l` per row for debugging the online pass | 1 MB |
| `mla_decode_absorbed` | 16 heads × 4 rows, L=1024, d_k=576, d_v=512 | 9-tile K reduction, `[64,512]` wide ACC | 3 MB |
| `kda_chunk64` | T=256, d=128, C=64 | recurrent delta rule (normative) + per-chunk state checkpoints | 0.6 MB |
| `moe_grouped` | B=256, d=512, 32 experts, top-6, sqrtsoftplus | `VTOPK` indices, ragged `counts`, `idx_lists` with `-1` skips, grouped SwiGLU, combine weights | 25 MB |
| `v41_flash_decoder_slice` | B=64, L=4096, d=5120, 64 heads, 576/512, 32 experts × 512, top-512 sparse | every kernel at real tile shapes and real formats: `f4.r16` KV rows, `f4.b32` MXFP4 experts, `f8.t` attention weights | 409 MB |

Notes on the slice: expert weights exist only as payload + scales (the
dequantization is normative and is what the reference consumed); `d_ff` is
reduced 2304 → 512 to bound size without changing tile shapes; pre-norms are
omitted; the indexer is synthetic (random distinct top-512). Real-weight
numerics are a level-3 test, not this fixture.

KDA: the recurrent form in `kda_recurrent` is frozen as the reference. If the
pinned Kimi Linear commit places the gate differently, change that function
and regenerate; the chunkwise kernel is derived from it, never the reverse.

## Change log forced by the generators

- `LOAD.IDX` gained `col0`/`cols_valid` (ISA §6.1): a 576-wide KV entry does
  not fit one `[128,128]` WGT tile, so a gather is repeated per column window
  on the same index tile. Whether this becomes a tile-group destination is
  ISA open issue 5.
- `quant_e4m3` was rewritten to be memory-linear (the table-broadcast version
  needed 48 GB on `W_qb`).
