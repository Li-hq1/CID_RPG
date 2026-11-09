#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" uv run python -m compression.geometry_ae --category="${CATEGORY:-Beauty}" "$@"
