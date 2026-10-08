#!/usr/bin/env bash
# 脚本有命令失败即退出而非继续进行
set -euo pipefail 
# project root
cd "$(dirname "$0")/.."

export CUDA_VISIBLE_DEVICES="0"


CATEGORY="Beauty"
# CATEGORY="Sports_and_Outdoors"
# CATEGORY="Toys_and_Games"
# CATEGORY="CDs_and_Vinyl"


RUN_ID="${RUN_ID:-tigerdiff_${CATEGORY}_qwen3_hstu_geo-ae-no-relation-new-eval}"

# encoder_type: hstu | transformer
# hstu_attention_scaling: max_seq_len | input_length 
# supervision_mode：last_item | all_positions
# --config=config_transformer.py
# 恢复训练: --resume
uv run train.py \
  --category="${CATEGORY}" \
  --run_id="${RUN_ID}" \
  --encoder_type="hstu" \
  --supervision_mode=all_positions \
  "$@" 
