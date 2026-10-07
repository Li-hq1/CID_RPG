#!/usr/bin/env bash
# 脚本有命令失败即退出而非继续进行
set -euo pipefail 

# project root
cd "$(dirname "$0")/.."

export CUDA_VISIBLE_DEVICES="4"


CATEGORY="Beauty"
# CATEGORY="Sports_and_Outdoors"
# CATEGORY="Toys_and_Games"
# CATEGORY="CDs_and_Vinyl"




ENCODER_TYPE="${ENCODER_TYPE:-hstu}"
RUN_ID="${RUN_ID:-tigerdiff_${CATEGORY}_qwen3_${ENCODER_TYPE}_all_positions_warmup0-1_50epoch}"

# supervision_mode 可选：last_item | all_positions
# 恢复训练时，在命令中加入 --resume
# "$@" 将脚本收到的额外参数传给 train.py
uv run train.py \
  --category="${CATEGORY}" \
  --run_id="${RUN_ID}" \
  --encoder_type="${ENCODER_TYPE}" \
  --supervision_mode=all_positions \
  "$@" 
