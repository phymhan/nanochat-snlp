#!/bin/bash
# Eval selected IDN configs for d32s baseline (seq PPL@2048=15.21)
# IDN_batched n8 K=4 h0 -> PPL 15.42 (0.99x)
# IDN_batched n8 K=4 batch_fwd -> PPL 15.35 (0.93x)
set -e; export OMP_NUM_THREADS=1; export NANOCHAT_BASE_DIR=${NANOCHAT_BASE_DIR:-cache}; GPU=${GPU:-0}
CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 uv run python -m snlp.eval_snlp \
    --model-tag d32s_baseline_4800 --n-par 8 --method IDN_batched --K 4 --init h0
CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 uv run python -m snlp.eval_snlp \
    --model-tag d32s_baseline_4800 --n-par 8 --method IDN_batched --K 4 --init batch_fwd
