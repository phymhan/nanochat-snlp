#!/bin/bash
# Train: Nanochat-0.5B baseline w/o x0ve (d32s, nox0ve, no reg)
# Model tag: d32s_nox0ve_baseline_9600, 9600 steps
set -e
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR=${NANOCHAT_BASE_DIR:-cache}

PYTHONUNBUFFERED=1 torchrun --standalone --nproc_per_node=${NGPU:-4} \
    -m scripts.base_train -- \
    --depth=32 --aspect-ratio=20 --device-batch-size=8 \
    --num-iterations=9600 --save-every=-1 --eval-every=500 --sample-every=-1 --core-metric-every=-1 \
    --no-x0-resid --no-ve \
    --jacobi-reg-warmup=0.2 \
    --model-tag=d32s_nox0ve_baseline_9600 \
    --run=dummy \
    2>&1 | tee "$NANOCHAT_BASE_DIR/d32s_nox0ve_baseline_9600_train.log"
