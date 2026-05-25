#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

python3 main.py \
    --model=TigerDiff \
    --dataset=AmazonReviews2014 \
    --category=Beauty \
    --run_id=tigerdiff_beauty \
    --lr=0.0005 \
    --weight_decay=0.01 \
    --train_batch_size=256 \
    --eval_batch_size=64 \
    --max_item_seq_len=50 \
    --embd_pdrop=0.1 \
    --attn_pdrop=0.1 \
    --diffloss_w=512 \
    --diffloss_d=3 \
    --num_sampling_steps=20 \
    --diffusion_batch_mul=1 \
    --diff_temperature=1.0 \
    --use_rectified_flow=True \
    --rectified_flow_steps=1000 \
    --temperature=0.07 \
    "$@"
