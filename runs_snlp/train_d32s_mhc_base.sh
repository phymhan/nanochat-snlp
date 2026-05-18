#!/bin/bash
# Train: Nanochat-0.5B mHC baseline (d32s, mHC x0+VE, no reg)
# Model tag: d32s_mhc4_x0ve_baseline, 4800 steps
set -e
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR=${NANOCHAT_BASE_DIR:-cache}

PYTHONUNBUFFERED=1 torchrun --standalone --nproc_per_node=${NGPU:-4} \
    -m scripts.base_train -- \
    --depth=32 --aspect-ratio=20 --device-batch-size=4 \
    --num-iterations=4800 --save-every=-1 --eval-every=500 --sample-every=-1 --core-metric-every=-1 \
    --use-mhc --mhc-num-streams=4 \
    --jacobi-reg-warmup=0.2 \
    --model-tag=d32s_mhc4_x0ve_baseline \
    --run=dummy \
    2>&1 | tee "$NANOCHAT_BASE_DIR/d32s_mhc4_x0ve_baseline_train.log"
