#!/bin/bash
# Eval selected IDN configs for d32 baseline (seq PPL@2048=10.10)
# IDN_batched n8 K=4 h0 -> PPL 10.65 (0.95x)
# IDN_batched n8 K=8 h0 -> PPL 10.13 (0.77x)
set -e; export OMP_NUM_THREADS=1; export NANOCHAT_BASE_DIR=${NANOCHAT_BASE_DIR:-cache}; GPU=${GPU:-0}
CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 uv run python -m snlp.eval_snlp \
    --model-tag d32_baseline --n-par 8 --method IDN_batched --K 4 --init h0
CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 uv run python -m snlp.eval_snlp \
    --model-tag d32_baseline --n-par 8 --method IDN_batched --K 8 --init h0
