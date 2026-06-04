# Reproducing SNLP Runs

The shell scripts in this directory reproduce the selected training runs and
SNLP evaluation configurations. Baseline models have two selected eval configs;
regularized models have three.

## Training Scripts

| Model | Script | Checkpoint tag |
|-------|--------|----------------|
| Nanochat-3B baseline | `train_d32_base.sh` | `d32_baseline` |
| Nanochat-3B IDN | `train_d32_idn.sh` | `d32_idn05_npar8` |
| Nanochat-0.5B baseline | `train_d32s_base.sh` | `d32s_baseline_4800` |
| Nanochat-0.5B IDN | `train_d32s_idn.sh` | `d32s_idn00625_npar24_s3` |
| Nanochat-0.5B w/o x0ve baseline | `train_d32s_nox0ve_base.sh` | `d32s_nox0ve_baseline` |
| Nanochat-0.5B w/o x0ve IDN | `train_d32s_nox0ve_idn.sh` | `d32s_nox0ve_idn05_npar24_s6_4800` |
| Nanochat-0.5B-mHC baseline | `train_d32s_mhc_base.sh` | `d32s_mhc4_x0ve_baseline` |
| Nanochat-0.5B-mHC HCN | `train_d32s_mhc_hcn.sh` | `d32s_mhc4_x0ve_newton05` |

## Evaluation Scripts

Each cell shows PPL and speedup for the selected SNLP config.

| Model | Script | Seq PPL | Selected configs |
|-------|--------|--------:|------------------|
| Nanochat-3B baseline | `eval_idn_d32_base.sh` | 10.10 | `IDN_batched n8 K4 h0`: 10.65 (0.95x)<br>`IDN_batched n8 K8 h0`: 10.13 (0.77x) |
| Nanochat-3B IDN | `eval_idn_d32_idn.sh` | 10.07 | `IDN_batched n8 K1 h0`: 10.46 (1.20x)<br>`IDN_batched n8 K1 batch_fwd`: 10.43 (1.10x)<br>`IDN_batched n8 K4 h0`: 10.09 (0.95x) |
| Nanochat-0.5B baseline | `eval_idn_d32s_base.sh` | 15.21 | `IDN_batched n8 K4 h0`: 15.42 (0.99x)<br>`IDN_batched n8 K4 batch_fwd`: 15.35 (0.93x) |
| Nanochat-0.5B IDN | `eval_idn_d32s_idn.sh` | 15.36 | `IDN_batched n24 K2 h0`: 18.09 (1.88x)<br>`IDN_batched n24 K4 h0`: 16.42 (1.33x)<br>`IDN_batched n12 K2 batch_fwd`: 15.59 (1.14x) |
| Nanochat-0.5B w/o x0ve baseline | `eval_idn_d32s_nox0ve_base.sh` | 17.65 | `IDN_batched n20 K4 batch_fwd`: 18.91 (1.27x)<br>`IDN_batched n8 K4 h0`: 17.75 (0.98x) |
| Nanochat-0.5B w/o x0ve IDN | `eval_idn_d32s_nox0ve_idn.sh` | 17.57 | `ChunkB_12xF2 n24 K1 h0`: 20.55 (2.58x)<br>`ChunkB_12xF2 n24 K2 h0`: 18.46 (2.09x)<br>`Fused_1xF12 n12 K1 h0`: 17.56 (1.40x) |
| Nanochat-0.5B-mHC baseline | `eval_idn_d32s_mhc_base.sh` | 15.16 | `mHC-Newton n8 K2 batch_fwd`: 15.65 (1.10x)<br>`mHC-Newton n8 K4 h0`: 15.38 (1.00x) |
| Nanochat-0.5B-mHC HCN | `eval_idn_d32s_mhc_hcn.sh` | 15.52 | `mHC-Newton n16 K1 h0`: 16.93 (1.60x)<br>`mHC-Newton n16 K4 h0`: 15.71 (1.14x)<br>`mHC-Newton n8 K4 h0`: 15.60 (1.00x) |

## Notes

PPL is evaluated with 2048-token contexts over the fixed validation stream. The
single-config evaluator defaults to `--seq-len 2048`; pass `--seq-len` explicitly
to reproduce older short-context experiments.
