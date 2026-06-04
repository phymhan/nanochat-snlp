#!/bin/bash
# Eval selected IDN configs for d32 IDN 0.5 npar8 (seq PPL@2048=10.07)
# IDN_batched n8 K=1 h0 -> PPL 10.46 (1.20x)
# IDN_batched n8 K=1 batch_fwd -> PPL 10.43 (1.10x)
# IDN_batched n8 K=4 h0 -> PPL 10.09 (0.95x)
set -e; export OMP_NUM_THREADS=1; export NANOCHAT_BASE_DIR=${NANOCHAT_BASE_DIR:-cache}; GPU=${GPU:-0}
CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 uv run python -m snlp.eval_snlp \
    --model-tag d32_idn05_npar8 --n-par 8 --method IDN_batched --K 1 --init h0
CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 uv run python -m snlp.eval_snlp \
    --model-tag d32_idn05_npar8 --n-par 8 --method IDN_batched --K 1 --init batch_fwd
CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 uv run python -m snlp.eval_snlp \
    --model-tag d32_idn05_npar8 --n-par 8 --method IDN_batched --K 4 --init h0
