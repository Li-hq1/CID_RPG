#!/usr/bin/env bash
# 脚本有命令失败即退出而非继续进行
set -euo pipefail 
# project root
cd "$(dirname "$0")/.."

# ContRec anti-collapse loss is on by default: recon + beta*mu + gamma*twins
# (beta=0.5, gamma=0.5, twins_lambda=0.005). New loss weights create a new
# artifact directory; pass --force only when overwriting the same fingerprint.
CUDA_VISIBLE_DEVICES=0 uv run python -m compression.contrec_vae --category=Beauty "$@"