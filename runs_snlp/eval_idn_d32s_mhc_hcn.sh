#!/bin/bash
# Eval selected configs for d32s mHC HCN 0.5 (seq PPL@2048=15.52)
# mHC-Newton n16 K=1 h0 -> PPL 16.93 (1.60x)
# mHC-Newton n16 K=4 h0 -> PPL 15.71 (1.14x)
# mHC-Newton n8 K=4 h0 -> PPL 15.60 (1.00x)
set -e; export OMP_NUM_THREADS=1; export NANOCHAT_BASE_DIR=${NANOCHAT_BASE_DIR:-cache}; GPU=${GPU:-0}
CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 uv run python -m snlp.eval_snlp \
    --model-tag d32s_mhc4_x0ve_newton05 --n-par 16 --method mHC-Newton --K 1 --init h0 --cache-hc
CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 uv run python -m snlp.eval_snlp \
    --model-tag d32s_mhc4_x0ve_newton05 --n-par 16 --method mHC-Newton --K 4 --init h0 --cache-hc
CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 uv run python -m snlp.eval_snlp \
    --model-tag d32s_mhc4_x0ve_newton05 --n-par 8 --method mHC-Newton --K 4 --init h0 --cache-hc
