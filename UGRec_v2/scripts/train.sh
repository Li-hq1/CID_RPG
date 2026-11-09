#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES="${GPU_ID:-0}"
CATEGORY="${CATEGORY:-Beauty}"
ENCODER_TYPE="${ENCODER_TYPE:-hstu}"
RUN_ID="${RUN_ID:-ugrec_v2_${CATEGORY}_${ENCODER_TYPE}}"
exec bash scripts/python.sh train.py --category="$CATEGORY" --run_id="$RUN_ID" --encoder_type="$ENCODER_TYPE" "$@"
