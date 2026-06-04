#!/bin/bash
# Eval selected IDN configs for d32s nox0ve IDN 0.5 s6 (seq PPL@2048=17.57)
# ChunkB_12xF2 n24 K=1 h0 -> PPL 20.55 (2.58x)
# ChunkB_12xF2 n24 K=2 h0 -> PPL 18.46 (2.09x)
# Fused_1xF12 n12 K=1 h0 -> PPL 17.56 (1.40x)
set -e; export OMP_NUM_THREADS=1; export NANOCHAT_BASE_DIR=${NANOCHAT_BASE_DIR:-cache}; GPU=${GPU:-0}
CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 uv run python -m snlp.eval_snlp \
    --model-tag d32s_nox0ve_idn05_npar24_s6_4800 --n-par 24 --method ChunkB --chunks 12 --K 1 --init h0
CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 uv run python -m snlp.eval_snlp \
    --model-tag d32s_nox0ve_idn05_npar24_s6_4800 --n-par 24 --method ChunkB --chunks 12 --K 2 --init h0
CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 uv run python -m snlp.eval_snlp \
    --model-tag d32s_nox0ve_idn05_npar24_s6_4800 --n-par 12 --method Fused --K 1 --init h0
