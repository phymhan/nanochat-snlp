#!/bin/bash
# Train: Nanochat-0.5B IDN reg + MTP-1 (d32s, x0+VE, lambda=0.0625, stride=3, num_mtp_steps=1)
# Model tag: d32s_idn00625_npar24_s3_mtp1_4800, 4800 steps
set -e
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR=${NANOCHAT_BASE_DIR:-cache}

PYTHONUNBUFFERED=1 torchrun --standalone --nproc_per_node=${NGPU:-4} \
    -m scripts.base_train -- \
    --depth=32 --aspect-ratio=20 --device-batch-size=8 \
    --num-iterations=4800 --save-every=-1 --eval-every=500 --sample-every=-1 --core-metric-every=-1 \
    --identity-newton-reg=0.0625 --idn-stride=3 --jacobi-reg-warmup=0.2 \
    --n-par-configs=8,16,24 \
    --num-mtp-steps=1 --mtp-loss-weight=0.3 \
    --model-tag=d32s_idn00625_npar24_s3_mtp1_4800 \
    --run=dummy \
    2>&1 | tee "$NANOCHAT_BASE_DIR/d32s_idn00625_npar24_s3_mtp1_4800_train.log"
