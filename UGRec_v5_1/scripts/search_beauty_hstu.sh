#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

export CUDA_VISIBLE_DEVICES="${GPU_ID:-4}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"

# First-round space: HSTU structure, epoch, and recommender.lr on Beauty.
# The model is direct CE on the HSTU head, not the flow control.
# n_trials in the JSON is this invocation's new trials; the same study name resumes.
bash scripts/python.sh -m joint_training.search \
  --search-config joint_training/search_space.json \
  --config configs/direct_ce.json \
  --category Beauty \
  --device cuda:0 \
  --set training.output_dir=outputs/search \
  "$@"
