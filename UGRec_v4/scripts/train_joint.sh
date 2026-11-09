#!/usr/bin/env bash
# 脚本有命令失败即退出而非继续进行
set -euo pipefail
# project root
cd "$(dirname "$0")/.."

export CUDA_VISIBLE_DEVICES="${GPU_ID:-0}"

CATEGORY="${CATEGORY:-Beauty}"
# CATEGORY="Sports_and_Outdoors"
# CATEGORY="Toys_and_Games"
# CATEGORY="CDs_and_Vinyl"

RUN_ID="${RUN_ID:-ugrec_v3_${CATEGORY}_joint}"

# layernorm，无 contrastive
bash scripts/python.sh -m joint_training \
  --category="${CATEGORY}" \
  --device="cuda:0" \
  --run-id="${RUN_ID}" \
  "$@"


# 单个 run
# tensorboard --logdir outputs/joint_training/Beauty/<run_id>/tensorboard --port 6006
# 多 run 对比
# tensorboard --logdir outputs/joint_training/Beauty --port 6006
