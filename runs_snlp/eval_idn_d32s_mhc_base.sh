#!/bin/bash
# Eval selected configs for d32s mHC baseline (seq PPL=73.24)
# m2: n12 K=2 ChunkB_4xF3 batch_fwd → PPL 69.42 (1.31x)
# m3: n8 K=2 ChunkB_4xF2 batch_fwd → PPL 61.34 (1.13x)
set -e; export OMP_NUM_THREADS=1; export NANOCHAT_BASE_DIR=${NANOCHAT_BASE_DIR:-cache}; GPU=${GPU:-0}
CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 uv run python -m snlp.eval_snlp \
    --model-tag d32s_mhc4_x0ve_baseline --n-par 12 --method ChunkB --chunks 4 --K 2 --init batch_fwd --cache-hc
CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 uv run python -m snlp.eval_snlp \
    --model-tag d32s_mhc4_x0ve_baseline --n-par 8 --method ChunkB --chunks 4 --K 2 --init batch_fwd --cache-hc
