#!/usr/bin/env bash
set -euo pipefail
set -o noclobber
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES="${GPU_ID:-5}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
mkdir -p outputs/initial_20261106
# Two declared trials, sequential; cap this process at 20% of device memory.
# Existing jobs are neither signalled nor reconfigured.
for preset in ce_only serial_shared; do
  run_id="beauty_${preset}_s2024_20261106"
  bash scripts/python.sh -c 'import runpy, torch; torch.cuda.set_per_process_memory_fraction(0.20, 0); runpy.run_path("run.py", run_name="__main__")' \
    --config "configs/${preset}.json" --device cuda:0 --category Beauty \
    --run-id "$run_id" --audit full --eval-users 4096 \
    > "outputs/initial_20261106/${run_id}.log" 2>&1
done
