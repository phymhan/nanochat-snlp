#!/bin/bash
# Train: Nanochat-0.5B IDN reg w/o x0ve (d32s, nox0ve, lambda=0.5, stride=6, no-detach)
# Model tag: d32s_nox0ve_idn05_npar24_s6_4800, 4800 steps
set -e
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR=${NANOCHAT_BASE_DIR:-cache}

PYTHONUNBUFFERED=1 torchrun --standalone --nproc_per_node=${NGPU:-4} \
    -m scripts.base_train -- \
    --depth=32 --aspect-ratio=20 --device-batch-size=8 \
    --num-iterations=4800 --save-every=-1 --eval-every=500 --sample-every=-1 --core-metric-every=-1 \
    --no-x0-resid --no-ve \
    --identity-newton-reg=0.5 --idn-stride=6 --jacobi-reg-warmup=0.2 \
    --n-par-configs=8,16,24 \
    --model-tag=d32s_nox0ve_idn05_npar24_s6_4800 \
    --run=dummy \
    2>&1 | tee "$NANOCHAT_BASE_DIR/d32s_nox0ve_idn05_npar24_s6_4800_train.log"
