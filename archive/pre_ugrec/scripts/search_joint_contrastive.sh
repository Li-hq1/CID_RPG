#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

export CUDA_VISIBLE_DEVICES="${GPU_ID:-0}"

CATEGORY="${CATEGORY:-Beauty}"
STUDY="${STUDY:-joint_${CATEGORY}_contrastive_search}"

# 搜索空间在 joint_training/search_space.json。N_TRIALS 是这一次新跑的 trial 数；
# 同一个 STUDY 再执行会接着搜，不会重跑已经完成的 trial。
# 缩短单个 trial 可追加：--set training.epochs=30 --set evaluation.patience=10
uv run python -m joint_training.search \
  --study-name="${STUDY}" \
  --category="${CATEGORY}" \
  --device="cuda:0" \
  --contrastive \
  "$@"
