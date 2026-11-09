#!/usr/bin/env bash
# 脚本有命令失败即退出而非继续进行
set -euo pipefail
# project root
cd "$(dirname "$0")/.."

export CUDA_VISIBLE_DEVICES="1"

CATEGORY="Beauty"
# CATEGORY="Sports_and_Outdoors"
# CATEGORY="Toys_and_Games"
# CATEGORY="CDs_and_Vinyl"

RUN_ID="${RUN_ID:-joint_${CATEGORY}_layernorm_no-infoNCE_no-init-test}"

# layernorm，无 contrastive
uv run python -m joint_training \
  --category="${CATEGORY}" \
  --device="cuda:0" \
  --run-id="${RUN_ID}"


# 单个 run
# tensorboard --logdir ckpt/joint_training/Beauty/<run_id>/tensorboard --port 6006
# 多 run 对比
# tensorboard --logdir ckpt/joint_training/Beauty --port 6006
