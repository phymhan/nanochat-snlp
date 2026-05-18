# SNLP: Structured Newton Layer Parallelism

Code for the paper *Layer-Parallel Inference via Structured Newton Corrections*.

<!-- [Paper](https://arxiv.org/abs/XXXX.XXXXX) -->

![Overview](dev/snlp.png)

## Overview

Autoregressive language models execute Transformer layers sequentially, creating a latency bottleneck not removed by tensor or pipeline parallelism. We introduce **Structured Newton Layer Parallelism (SNLP)**, a training and inference framework that replaces exact layer Jacobians with cheap structured surrogates to enable layer-parallel inference.

- **Identity Newton (IDN)**: For residual Transformers, the correction reduces to additive prefix-style propagation over depth.
- **HC Newton (HCN)**: For mHC-style architectures, uses the learned residual mixing matrix as the Newton surrogate.
- **Diagonal Newton (DiagN)**: Uses the diagonal of the layer Jacobian, solved via associative prefix scan.

With SNLP-aware regularization and chunkwise layer fusion, trained-from-scratch Nanochat models reach up to **2.3x speedup** with comparable or lower perplexity. SNLP regularization also improves sequential PPL by **4.7%--23.4%** across model variants.

## Project Structure

```
nanochat-snlp/
├── nanochat/                    # Core model library (GPT, tokenizer, training infra)
│   ├── gpt.py                   # GPT model with mHC, IDN/HCN reg support
│   └── jacobi_forward.py        # Jacobi iteration primitives, preheat calibration
├── scripts/                     # Training scripts
│   ├── base_train.py            # Main training entry point
│   └── base_eval.py             # Core evaluation
├── snlp/                        # SNLP inference and evaluation
│   ├── inference_idn.py         # IDN forward functions (batched, fused, chunkwise)
│   ├── inference_hcn.py         # HCN forward functions for mHC models
│   ├── inference_diagn.py       # DiagN forward functions with associative scan
│   ├── inference_idn_ots.py     # IDN for off-the-shelf HF models (Qwen, TinyLlama, Gemma)
│   ├── eval_snlp.py             # Evaluation entry point (PPL, timing, top-1, cos_sim)
│   ├── eval_snlp_ots.py         # Evaluation entry for off-the-shelf models
│   └── demo_snlp_deer_qwen.py   # Standalone demo: exact/diagonal Newton on Qwen
└── runs_snlp/                   # Shell scripts to reproduce paper results
    ├── train_*.sh               # Training scripts for each model variant
    └── eval_idn_*.sh            # Evaluation scripts for selected configs
```

## SNLP Inference Methods

| Function | File | Description | Paper notation |
|----------|------|-------------|----------------|
| `forward_idn_batched` | `inference_idn.py` | IDN per-layer batched + prefix-sum correction | IDN NxF1 |
| `forward_idn_fused` | `inference_idn.py` | Fully fused mega-block (no inter-layer correction) | IDN 1xFN |
| `forward_idn_chunkwise` | `inference_idn.py` | Chunkwise fused + inter-chunk IDN correction | IDN CxFM |
| `forward_hcn_batched` | `inference_hcn.py` | HCN per-layer batched + H^res Newton correction | HCN NxF1 |
| `forward_hcn_chunkwise` | `inference_hcn.py` | Chunkwise fused + inter-chunk HCN correction | HCN CxFM |
| `forward_diagn_batched` | `inference_diagn.py` | Diagonal Newton + Hutchinson estimation + prefix scan | DiagN |

**Config notation**: `CxFM-init` = C parallel chunks, each fusing M layers, with initialization `h0` (prefix state) or `batch_fwd` (one-shot forward). K = number of Newton iterations.

## Training Configurations

| Model | Tag | Key flags |
|-------|-----|-----------|
| Nanochat-3B baseline | `d32_baseline` | `--depth=32 --fp8` |
| Nanochat-3B IDN | `d32_idn01_npar24` | `+ --identity-newton-reg=0.1` |
| Nanochat-3B DiagN | `d32_diag01_npar24_stride3` | `+ --diag-newton-reg=0.1 --idn-stride=3 --idn-detach-target` |
| Nanochat-0.5B baseline | `d32s_baseline_4800` | `--depth=32 --aspect-ratio=20` |
| Nanochat-0.5B IDN | `d32s_idn05_npar24_s3` | `+ --identity-newton-reg=0.5 --idn-stride=3` |
| Nanochat-0.5B DiagN | `d32s_diag01_npar24_stride3_9600` | `+ --diag-newton-reg=0.1 --idn-stride=3 --idn-detach-target` |
| Nanochat-0.5B w/o x0ve baseline | `d32s_nox0ve_baseline_9600` | `+ --no-x0-resid --no-ve` |
| Nanochat-0.5B w/o x0ve IDN | `d32s_nox0ve_idn05_npar24_s6_4800` | `+ --identity-newton-reg=0.5 --idn-stride=6` |
| Nanochat-0.5B-mHC baseline | `d32s_mhc4_x0ve_baseline` | `+ --use-mhc --mhc-num-streams=4` |
| Nanochat-0.5B-mHC HCN | `d32s_mhc4_x0ve_newton05` | `+ --mhc-newton-reg=0.5 --idn-detach-target` |

## Setup

```bash
# Install dependencies
uv sync

# Download training data (~15GB)
export NANOCHAT_BASE_DIR=$(pwd)/cache
uv run python -m nanochat.dataset -n 170

# Train tokenizer (~1 min)
uv run python -m scripts.tok_train
```

## Training

```bash
# Example: Train Nanochat-0.5B with IDN regularization (4 GPUs)
bash runs_snlp/train_d32s_idn.sh

# Or run directly:
NANOCHAT_BASE_DIR=cache uv run torchrun --standalone --nproc_per_node=4 \
    -m scripts.base_train -- \
    --depth=32 --aspect-ratio=20 --device-batch-size=8 \
    --num-iterations=4800 \
    --identity-newton-reg=0.5 --idn-stride=3 --jacobi-reg-warmup=0.2 \
    --n-par-configs=8,16,24 \
    --model-tag=d32s_idn05_npar24_s3
```

See `runs_snlp/train_*.sh` for all model training scripts.

## Inference Evaluation

```bash
# Evaluate a specific SNLP config (single-config evaluator)
CUDA_VISIBLE_DEVICES=0 NANOCHAT_BASE_DIR=cache uv run python -m snlp.eval_snlp \
    --model-tag d32s_idn05_npar24_s3 \
    --n-par 24 --method ChunkB --chunks 12 --K 2 --init h0

# Evaluate mHC model with HCN correction
CUDA_VISIBLE_DEVICES=0 NANOCHAT_BASE_DIR=cache uv run python -m snlp.eval_snlp \
    --model-tag d32s_mhc4_x0ve_newton05 \
    --n-par 20 --method mHC-Newton --K 4 --init h0 --cache-hc

# Evaluate with DiagN correction
CUDA_VISIBLE_DEVICES=0 NANOCHAT_BASE_DIR=cache uv run python -m snlp.eval_snlp \
    --model-tag d32s_idn05_npar24_s3 \
    --n-par 24 --method DiagN --K 4 --init batch_fwd --jvp-method vjp

# Reproduce paper configs for a model variant
bash runs_snlp/eval_idn_d32s_idn.sh
```

### Off-the-Shelf Models

Evaluate SNLP on pretrained HuggingFace models (Qwen 2.5, TinyLlama, Gemma 3):

```bash
HF_HOME=/path/to/hf_cache CUDA_VISIBLE_DEVICES=0 uv run python -m snlp.eval_snlp_ots \
    --model Qwen/Qwen2.5-0.5B-Instruct --n-par 4 8
```

### Demo: Exact Newton Sanity Check

`demo_snlp_deer_qwen.py` is a standalone demo that implements all Newton variants on Qwen 2.5 0.5B. With exact (full Jacobian) Newton and enough iterations, the output converges to sequential for any off-the-shelf model — even with all 24 layers parallel:

```bash
# Exact Newton — all 24 layers parallel, converges to sequential output
HF_HOME=/path/to/hf_cache uv run python -m snlp.demo_snlp_deer_qwen \
    --jacobian full --jvp jvp --layer-start 0 --parallel-iters 16

# Diagonal Newton (VJP Hutchinson + associative scan) — 8 parallel layers
HF_HOME=/path/to/hf_cache uv run python -m snlp.demo_snlp_deer_qwen \
    --jacobian diag --jvp vjp --scan --layer-start 16 --parallel-iters 8
```

## Citation

Coming soon.

<!--
```bibtex
@article{han2026snlp,
  title={Layer-Parallel Inference via Structured Newton Corrections},
  author={Han, Ligong and Xu, Kai and Wang, Hao and Srivastava, Akash},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2026}
}
```
-->

## Acknowledgement

This codebase builds on several excellent open-source projects:

- [Nanochat](https://github.com/karpathy/nanochat) by Andrej Karpathy — base model architecture, training, and data infrastructure
- [ELK](https://github.com/lindermanlab/elk) by Gonzalez et al. — quasi-Newton methods for parallelizing nonlinear recurrences
- [mHC](https://github.com/tokenbender/mHC-manifold-constrained-hyper-connections) by Xie et al. — manifold-constrained HyperConnections
- [SJD](https://github.com/tyshiwo1/Accelerating-T2I-AR-with-SJD/) by Song et al. — Jacobi decoding for accelerating autoregressive models

We thank the authors for generously open-sourcing their work.
