#!/bin/bash
# Eval selected IDN configs for d32s nox0ve baseline (seq PPL@2048=17.65)
# IDN_batched n20 K=4 batch_fwd -> PPL 18.91 (1.27x)
# IDN_batched n8 K=4 h0 -> PPL 17.75 (0.98x)
set -e; export OMP_NUM_THREADS=1; export NANOCHAT_BASE_DIR=${NANOCHAT_BASE_DIR:-cache}; GPU=${GPU:-0}
CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 uv run python -m snlp.eval_snlp \
    --model-tag d32s_nox0ve_baseline --n-par 20 --method IDN_batched --K 4 --init batch_fwd
CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 uv run python -m snlp.eval_snlp \
    --model-tag d32s_nox0ve_baseline --n-par 8 --method IDN_batched --K 4 --init h0
