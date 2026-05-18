#!/bin/bash
# Eval selected IDN configs for d32s DiagN 0.1 (seq PPL=63.08)
# c13: n8 K=2 ChunkB_4xF2 h0 → PPL 63.40 (1.17x)
# c16: n12 K=8 ChunkB_12xF1 h0 → PPL 51.42 (0.88x)
set -e; export OMP_NUM_THREADS=1; export NANOCHAT_BASE_DIR=${NANOCHAT_BASE_DIR:-cache}; GPU=${GPU:-0}
CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 uv run python -m snlp.eval_snlp \
    --model-tag d32s_diag01_npar24_stride3_9600 --n-par 8 --method ChunkB --chunks 4 --K 2 --init h0
CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 uv run python -m snlp.eval_snlp \
    --model-tag d32s_diag01_npar24_stride3_9600 --n-par 12 --method ChunkB --chunks 12 --K 8 --init h0
