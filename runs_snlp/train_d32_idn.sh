#!/bin/bash
# Train: Nanochat-3B IDN reg (d32, x0+VE, lambda=0.5, stride=0, no-detach)
# Model tag: d32_idn05_npar8, 9600 steps
set -e
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR=${NANOCHAT_BASE_DIR:-cache}

PYTHONUNBUFFERED=1 torchrun --standalone --nproc_per_node=${NGPU:-8} \
    -m scripts.base_train -- \
    --depth=32 --device-batch-size=8 --fp8 \
    --num-iterations=9600 --save-every=-1 --eval-every=500 --sample-every=-1 --core-metric-every=-1 \
    --identity-newton-reg=0.5 --jacobi-reg-warmup=0.2 \
    --n-par-configs=8 \
    --model-tag=d32_idn05_npar8 \
    --run=dummy \
    2>&1 | tee "$NANOCHAT_BASE_DIR/d32_idn05_npar8_train.log"
