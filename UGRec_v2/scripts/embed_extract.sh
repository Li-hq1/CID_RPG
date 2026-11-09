#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES="${GPU_ID:-0}"
exec bash scripts/python.sh -m data.extract_embeddings --category="${CATEGORY:-Beauty}" "$@"
