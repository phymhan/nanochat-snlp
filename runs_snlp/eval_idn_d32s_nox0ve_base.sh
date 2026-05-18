#!/bin/bash
# Eval selected IDN configs for d32s nox0ve baseline (seq PPL=84.74)
# c17: n16 K=2 ChunkB_8xF2 batch_fwd → PPL 81.35 (1.39x)
# c18: n12 K=4 ChunkB_6xF2 batch_fwd → PPL 78.54 (1.05x)
set -e; export OMP_NUM_THREADS=1; export NANOCHAT_BASE_DIR=${NANOCHAT_BASE_DIR:-cache}; GPU=${GPU:-0}
CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 uv run python -m snlp.eval_snlp \
    --model-tag d32s_nox0ve_baseline_9600 --n-par 16 --method ChunkB --chunks 8 --K 2 --init batch_fwd
CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 uv run python -m snlp.eval_snlp \
    --model-tag d32s_nox0ve_baseline_9600 --n-par 12 --method ChunkB --chunks 6 --K 4 --init batch_fwd
