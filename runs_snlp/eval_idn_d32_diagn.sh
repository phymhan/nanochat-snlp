#!/bin/bash
# Eval selected IDN configs for d32 DiagN 0.1 (seq PPL=35.41)
# c6: n4 K=2 ChunkB_4xF1 batch_fwd → PPL 31.52 (0.92x)
# c7: n12 K=8 ChunkB_12xF1 h0 → PPL 31.39 (0.57x)
set -e; export OMP_NUM_THREADS=1; export NANOCHAT_BASE_DIR=${NANOCHAT_BASE_DIR:-cache}; GPU=${GPU:-0}
CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 uv run python -m snlp.eval_snlp \
    --model-tag d32_diag01_npar24_stride3 --n-par 4 --method ChunkB --chunks 4 --K 2 --init batch_fwd
CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 uv run python -m snlp.eval_snlp \
    --model-tag d32_diag01_npar24_stride3 --n-par 12 --method ChunkB --chunks 12 --K 8 --init h0
