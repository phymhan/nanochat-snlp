#!/bin/bash
# Eval selected IDN configs for d32s nox0ve IDN 0.5 s6 (seq PPL=79.96)
# n3: n24 K=2 ChunkB_4xF6 h0 → PPL 75.09 (2.32x)
# n6: n16 K=4 ChunkB_4xF4 h0 → PPL 72.71 (1.24x)
set -e; export OMP_NUM_THREADS=1; export NANOCHAT_BASE_DIR=${NANOCHAT_BASE_DIR:-cache}; GPU=${GPU:-0}
CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 uv run python -m snlp.eval_snlp \
    --model-tag d32s_nox0ve_idn05_npar24_s6_4800 --n-par 24 --method ChunkB --chunks 4 --K 2 --init h0
CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 uv run python -m snlp.eval_snlp \
    --model-tag d32s_nox0ve_idn05_npar24_s6_4800 --n-par 16 --method ChunkB --chunks 4 --K 4 --init h0
