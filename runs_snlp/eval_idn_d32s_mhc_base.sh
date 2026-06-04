#!/bin/bash
# Eval selected configs for d32s mHC baseline (seq PPL@2048=15.16)
# mHC-Newton n8 K=2 batch_fwd -> PPL 15.65 (1.10x)
# mHC-Newton n8 K=4 h0 -> PPL 15.38 (1.00x)
set -e; export OMP_NUM_THREADS=1; export NANOCHAT_BASE_DIR=${NANOCHAT_BASE_DIR:-cache}; GPU=${GPU:-0}
CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 uv run python -m snlp.eval_snlp \
    --model-tag d32s_mhc4_x0ve_baseline --n-par 8 --method mHC-Newton --K 2 --init batch_fwd --cache-hc
CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 uv run python -m snlp.eval_snlp \
    --model-tag d32s_mhc4_x0ve_baseline --n-par 8 --method mHC-Newton --K 4 --init h0 --cache-hc
