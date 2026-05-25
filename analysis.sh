#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

# Analysis template for TigerDiff.
#
# Examples:
#   bash analysis.sh
#   CATEGORY=Beauty ANALYSIS=embedding bash analysis.sh
#   CATEGORY=Beauty ANALYSIS=attention CHECKPOINT=ckpt/your_model.pth bash analysis.sh
#   CATEGORY=Beauty ANALYSIS=all CHECKPOINT=ckpt/your_model.pth METHOD=umap bash analysis.sh
#
# Notes:
# - embedding analysis reads cached sentence embeddings under cache/AmazonReviews2014/<CATEGORY>/processed
# - attention analysis can run without CHECKPOINT (random weights), but to analyze a trained model,
#   pass a .pth checkpoint produced by training (default ckpt/).

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

CATEGORY="${CATEGORY:-Beauty}"
ANALYSIS="${ANALYSIS:-embedding}"   # embedding | attention | all
OUTPUT_DIR="${OUTPUT_DIR:-vis_results/analysis}"
METHOD="${METHOD:-tsne}"           # tsne | umap (umap requires umap-learn)
MAX_ITEMS="${MAX_ITEMS:-5000}"
N_CLUSTERS="${N_CLUSTERS:-8}"
N_SAMPLES="${N_SAMPLES:-3}"
CHECKPOINT="${CHECKPOINT:-}"

usage() {
    cat <<'USAGE'
Usage:
    CATEGORY=Beauty ANALYSIS=embedding bash analysis.sh

Environment variables:
    CATEGORY      Amazon category (Beauty, Sports_and_Outdoors, Toys_and_Games, CDs_and_Vinyl)
    ANALYSIS      embedding | attention | all
    OUTPUT_DIR   Output directory for figures (default: vis_results/analysis)
    METHOD       tsne | umap
    MAX_ITEMS    Subsample size for embedding visualization
    N_CLUSTERS   KMeans clusters for coloring in embedding plot
    N_SAMPLES    Number of attention heatmap samples
    CHECKPOINT   Path to .pth checkpoint (recommended for attention)

Tip:
    Run training once to generate cache + checkpoints.
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

if [[ "${ANALYSIS}" == "embedding" || "${ANALYSIS}" == "all" ]]; then
    if [[ ! -d "cache/AmazonReviews2014/${CATEGORY}/processed" ]]; then
        echo "[analysis.sh] Missing cache for CATEGORY='${CATEGORY}' (required for embedding plots)." >&2
        echo "            Expected: cache/AmazonReviews2014/${CATEGORY}/processed" >&2
        echo "            Run training once (see train_tigerdiff_*.sh or: python3 main.py --model=TigerDiff --category=${CATEGORY})" >&2
        echo "            Or set CATEGORY=Beauty (already present in this workspace)." >&2
        exit 2
    fi

    for required in \
        "cache/AmazonReviews2014/${CATEGORY}/processed/id_mapping.json" \
        "cache/AmazonReviews2014/${CATEGORY}/processed/all_item_seqs.json" \
        "cache/AmazonReviews2014/${CATEGORY}/processed/sentence-t5-base.sent_emb"; do
        if [[ ! -f "${required}" ]]; then
            echo "[analysis.sh] Missing required file for embedding analysis: ${required}" >&2
            exit 2
        fi
    done
fi

if [[ "${ANALYSIS}" == "attention" || "${ANALYSIS}" == "all" ]]; then
    if [[ -z "${CHECKPOINT}" ]]; then
        echo "[analysis.sh] CHECKPOINT is empty; attention analysis will use random weights." >&2
        echo "            To analyze a trained model, set CHECKPOINT=ckpt/<file>.pth" >&2
    elif [[ ! -f "${CHECKPOINT}" ]]; then
        echo "[analysis.sh] CHECKPOINT not found: ${CHECKPOINT}" >&2
        exit 2
    fi
fi

mkdir -p "${OUTPUT_DIR}"

cmd=(
    python3 analysis_tigerdiff.py
    --analysis "${ANALYSIS}"
    --category "${CATEGORY}"
    --output_dir "${OUTPUT_DIR}"
    --method "${METHOD}"
    --max_items "${MAX_ITEMS}"
    --n_clusters "${N_CLUSTERS}"
    --n_samples "${N_SAMPLES}"
)

if [[ -n "${CHECKPOINT}" ]]; then
    cmd+=(--checkpoint "${CHECKPOINT}")
fi

echo "[analysis.sh] Running: ${cmd[*]}"
"${cmd[@]}"
