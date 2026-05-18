#!/bin/bash
# Eval selected IDN configs for d32 baseline (seq PPL=37.16)
# c1: n4 K=2 ChunkB_4xF1 batch_fwd → PPL 32.27 (0.94x)
# c2: n12 K=8 ChunkB_12xF1 h0 → PPL 31.45 (0.59x)
set -e; export OMP_NUM_THREADS=1; export NANOCHAT_BASE_DIR=${NANOCHAT_BASE_DIR:-cache}; GPU=${GPU:-0}
CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 uv run python -m snlp.eval_snlp \
    --model-tag d32_baseline --n-par 4 --method ChunkB --chunks 4 --K 2 --init batch_fwd
CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 uv run python -m snlp.eval_snlp \
    --model-tag d32_baseline --n-par 12 --method ChunkB --chunks 12 --K 8 --init h0
