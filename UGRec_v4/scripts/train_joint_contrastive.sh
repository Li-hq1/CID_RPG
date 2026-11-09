#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

export CUDA_VISIBLE_DEVICES="${GPU_ID:-0}"

CATEGORY="${CATEGORY:-Beauty}"
CONTRASTIVE_WEIGHT="${CONTRASTIVE_WEIGHT:-0.01}"
NEGATIVE_COUNT="${NEGATIVE_COUNT:-64}"
RUN_ID="${RUN_ID:-ugrec_v3_${CATEGORY}_contrastive_w${CONTRASTIVE_WEIGHT}}"

bash scripts/python.sh -m joint_training \
  --category="${CATEGORY}" \
  --device="cuda:0" \
  --run-id="${RUN_ID}" \
  --contrastive \
  --set "contrastive.weight=${CONTRASTIVE_WEIGHT}" \
  --set "contrastive.num_negatives=${NEGATIVE_COUNT}" \
  "$@"
