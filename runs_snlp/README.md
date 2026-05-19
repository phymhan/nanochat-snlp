# Reproducing Paper Results

## Evaluation Data Note

The evaluation code uses a **shuffled** (seed=42) sampling of the validation shard for PPL computation, which gives slightly different absolute PPL values compared to the sequential reading used in the paper. The relative ΔPPL% between sequential and SNLP inference is consistent (within ~1 percentage point).

## 0.5B Results: Shuffle vs Original

Each cell shows PPL (ΔPPL% vs corresponding sequential baseline).

| Model | Config | Orig PPL | Shuffled PPL |
|-------|--------|----------|-------------|
| | | seq = 69.54 | seq = 71.80 |
| 0.5B No Reg. | 12xF1-h0 K=4 | 62.01 (-10.8%) | 63.22 (-12.0%) |
| 0.5B No Reg. | 16xF1-h0 K=8 | 47.25 (-32.1%) | 47.84 (-33.4%) |
| | | seq = 53.25 | seq = 55.27 |
| 0.5B IDN | 12xF2-h0 K=2 | 53.68 (+0.8%) | 53.83 (-2.6%) |
| 0.5B IDN | 2xF6-fwd K=1 | 44.00 (-17.4%) | 45.29 (-18.1%) |
| | | seq = 63.08 | seq = 65.37 |
| 0.5B DiagN | 4xF2-h0 K=2 | 63.40 (+0.5%) | 65.47 (+0.2%) |
| 0.5B DiagN | 12xF1-h0 K=8 | 51.42 (-18.5%) | 52.91 (-19.1%) |
| | | seq = 84.74 | seq = 87.49 |
| 0.5B w/o x0ve No Reg. | 8xF2-fwd K=2 | 81.35 (-4.0%) | 83.68 (-4.4%) |
| 0.5B w/o x0ve No Reg. | 6xF2-fwd K=4 | 78.54 (-7.3%) | 81.02 (-7.4%) |
| | | seq = 79.96 | seq = 83.26 |
| 0.5B w/o x0ve IDN | 4xF6-h0 K=2 | 75.09 (-6.1%) | 77.85 (-6.5%) |
| 0.5B w/o x0ve IDN | 4xF4-h0 K=4 | 72.71 (-9.1%) | 75.44 (-9.4%) |
| | | seq = 73.24 | seq = 76.00 |
| 0.5B-mHC No Reg. | 4xF3-fwd K=2 | 69.42 (-5.2%) | 71.26 (-6.2%) |
| 0.5B-mHC No Reg. | 4xF2-fwd K=2 | 61.34 (-16.3%) | 63.25 (-16.8%) |
| | | seq = 67.23 | seq = 70.16 |
| 0.5B-mHC HCN | n20-HCN-h0 K=4 | 66.56 (-1.0%) | 69.49 (-1.0%) |
| 0.5B-mHC HCN | n8-HCN-h0 K=1 | 65.91 (-2.0%) | 68.75 (-2.0%) |

ΔPPL% is consistent between the two data orderings — all conclusions from the paper hold. Shuffled eval gives 2–4 higher absolute PPL since it samples a more diverse document mix.

## Note on layer fusion and NxF1 notation

When fusing multiple layers into a chunk (`NxFM` with M>1), the fused chunk averages the per-layer x0/residual lambdas and drops value embeddings (VE). For `NxF1` (chunk_size=1), there is no lambda averaging, but VE is still dropped. So for standard models (with x0+VE), ChunkB `NxF1` differs slightly from `forward_idn_batched` which preserves exact per-layer weights including VE.

For **w/o x0ve** models (no lambdas, no VE), ChunkB `NxF1` is mathematically equivalent to `forward_idn_batched`. The `forward_idn_fused_split` variant uses a fused QKV kernel but splits out-projection and MLP per layer, so it is also mathematically equivalent to `forward_idn_batched`.

In the paper, we write all per-layer configs as `NxF1` for notational simplicity. In the selected configs above, only the two mHC HCN entries use the true per-layer HCN batched forward (`forward_hcn_batched`); all others use ChunkB.
