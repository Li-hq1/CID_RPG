#!/usr/bin/env bash
# 脚本有命令失败即退出而非继续进行
set -euo pipefail 
# project root
cd "$(dirname "$0")/.."

export CUDA_VISIBLE_DEVICES="2"

uv run python -m data.extract_embeddings --category=Sports_and_Outdoors
uv run python -m data.extract_embeddings --category=Toys_and_Games
uv run python -m data.extract_embeddings --category=CDs_and_Vinyl