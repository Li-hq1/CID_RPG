#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES="${GPU_ID:-0}"
exec bash scripts/python.sh -m compression.geometry_ae --category="${CATEGORY:-Beauty}" "$@"
