#!/usr/bin/env bash
# 脚本有命令失败即退出而非继续进行
set -euo pipefail 
# project root
cd "$(dirname "$0")/.."


export CUDA_VISIBLE_DEVICES="1"

CKPT_PATH=ckpt/AmazonReviews2014/TigerDiff/Beauty/tigerdiff_Beauty_qwen3_hstu/best_tigerdiff_Beauty_qwen3_hstu.pth


# transformer model
# uv run train.py \
# --config=config_transformer.py \
# --encoder_type=transformer \
# --category=Beauty \
# --run_id="eval_only" \
# --eval-only \
# --checkpoint="$CKPT_PATH" \
# --eval_sampling_mode=knn


# transformer model
# uv run train.py \
# --config=config_transformer.py \
# --encoder_type=transformer \
# --category=Beauty \
# --run_id="eval_only" \
# --eval-only \
# --checkpoint="$CKPT_PATH" \
# --eval_sampling_mode=diffusion_multi \
# --eval_num_samples=8 \
# --eval_candidates_per_sample=20 \
# --eval_candidate_metric=l2 \
# --eval_candidate_ranking=support \
# --eval_support_temperature=1.0 \
# --eval_log_sampling_stats=true


# # hstu model
uv run train.py \
--config=config.py \
--encoder_type=hstu \
--category=Beauty \
--run_id="eval_only" \
--eval-only \
--checkpoint="$CKPT_PATH" \
--eval_sampling_mode=diffusion_multi \
--eval_num_samples=32 \
--eval_candidates_per_sample=1 \
--eval_candidate_metric=l2 \
--eval_candidate_ranking=support \
--eval_support_temperature=1.0 \
--eval_log_sampling_stats=true