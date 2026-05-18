#!/bin/bash
# Eval selected IDN configs for d32s IDN 0.5 s3 (seq PPL=53.25)
# a2: n24 K=2 ChunkB_12xF2 h0 → PPL 53.68 (2.37x)
# a4: n12 K=1 ChunkB_2xF6 batch_fwd → PPL 44.00 (1.37x)
set -e; export OMP_NUM_THREADS=1; export NANOCHAT_BASE_DIR=${NANOCHAT_BASE_DIR:-cache}; GPU=${GPU:-0}
CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 uv run python -m snlp.eval_snlp \
    --model-tag d32s_idn05_npar24_s3 --n-par 24 --method ChunkB --chunks 12 --K 2 --init h0
CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 uv run python -m snlp.eval_snlp \
    --model-tag d32s_idn05_npar24_s3 --n-par 12 --method ChunkB --chunks 2 --K 1 --init batch_fwd
