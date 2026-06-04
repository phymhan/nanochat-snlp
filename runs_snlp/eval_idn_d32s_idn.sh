#!/bin/bash
# Eval selected IDN configs for d32s IDN 0.0625 s3 (seq PPL@2048=15.36)
# IDN_batched n24 K=2 h0 -> PPL 18.09 (1.88x)
# IDN_batched n24 K=4 h0 -> PPL 16.42 (1.33x)
# IDN_batched n12 K=2 batch_fwd -> PPL 15.59 (1.14x)
set -e; export OMP_NUM_THREADS=1; export NANOCHAT_BASE_DIR=${NANOCHAT_BASE_DIR:-cache}; GPU=${GPU:-0}
CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 uv run python -m snlp.eval_snlp \
    --model-tag d32s_idn00625_npar24_s3 --n-par 24 --method IDN_batched --K 2 --init h0
CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 uv run python -m snlp.eval_snlp \
    --model-tag d32s_idn00625_npar24_s3 --n-par 24 --method IDN_batched --K 4 --init h0
CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 uv run python -m snlp.eval_snlp \
    --model-tag d32s_idn00625_npar24_s3 --n-par 12 --method IDN_batched --K 2 --init batch_fwd
