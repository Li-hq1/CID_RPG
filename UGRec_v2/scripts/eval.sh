#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES="${GPU_ID:-0}"
if [[ $# -lt 1 ]]; then
  echo "Usage: bash scripts/eval.sh <checkpoint> [matching training/config options]" >&2
  exit 2
fi
UGREC_CHECKPOINT="$1"
shift
exec bash scripts/python.sh train.py --eval-only --checkpoint="$UGREC_CHECKPOINT" --run_id=eval_only "$@"
