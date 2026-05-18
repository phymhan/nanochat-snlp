#!/bin/bash
# Train: Nanochat-0.5B DiagN reg (d32s, x0+VE, lambda=0.1, stride=3, detach)
# Model tag: d32s_diag01_npar24_stride3_9600, 9600 steps
set -e
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR=${NANOCHAT_BASE_DIR:-cache}

PYTHONUNBUFFERED=1 torchrun --standalone --nproc_per_node=${NGPU:-4} \
    -m scripts.base_train -- \
    --depth=32 --aspect-ratio=20 --device-batch-size=8 \
    --num-iterations=9600 --save-every=-1 --eval-every=500 --sample-every=-1 --core-metric-every=-1 \
    --diag-newton-reg=0.1 --idn-stride=3 --idn-detach-target --jacobi-reg-warmup=0.2 \
    --n-par-configs=8,16,24 \
    --model-tag=d32s_diag01_npar24_stride3_9600 \
    --run=dummy \
    2>&1 | tee "$NANOCHAT_BASE_DIR/d32s_diag01_npar24_stride3_9600_train.log"
