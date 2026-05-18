#!/bin/bash
# Eval selected IDN configs for d32s baseline (seq PPL=69.54)
# c8: n12 K=4 ChunkB_12xF1 h0 → PPL 62.01 (1.11x)
# c9: n16 K=8 ChunkB_16xF1 h0 → PPL 47.25 (0.78x)
set -e; export OMP_NUM_THREADS=1; export NANOCHAT_BASE_DIR=${NANOCHAT_BASE_DIR:-cache}; GPU=${GPU:-0}
CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 uv run python -m snlp.eval_snlp \
    --model-tag d32s_baseline_4800 --n-par 12 --method ChunkB --chunks 12 --K 4 --init h0
CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 uv run python -m snlp.eval_snlp \
    --model-tag d32s_baseline_4800 --n-par 16 --method ChunkB --chunks 16 --K 8 --init h0
