#!/bin/bash
# Eval selected configs for d32s mHC HCN 0.5 (seq PPL=67.23)
# m6: n20 K=4 mHC-Newton h0 → PPL 66.56 (1.22x)
# m7: n8 K=1 mHC-Newton h0 → PPL 65.91 (0.97x)
set -e; export OMP_NUM_THREADS=1; export NANOCHAT_BASE_DIR=${NANOCHAT_BASE_DIR:-cache}; GPU=${GPU:-0}
CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 uv run python -m snlp.eval_snlp \
    --model-tag d32s_mhc4_x0ve_newton05 --n-par 20 --method mHC-Newton --K 4 --init h0 --cache-hc
CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 uv run python -m snlp.eval_snlp \
    --model-tag d32s_mhc4_x0ve_newton05 --n-par 8 --method mHC-Newton --K 1 --init h0 --cache-hc
