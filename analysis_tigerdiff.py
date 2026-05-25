"""
TigerDiff Analysis Script
=========================
Analysis 3: Item Embedding t-SNE / UMAP Visualization
Analysis 12: Attention Weight Heatmap Visualization

Usage:
    # t-SNE embedding visualization (no checkpoint needed)
    python analysis_tigerdiff.py --analysis embedding --category Beauty

    # Attention heatmap (requires checkpoint)
    python analysis_tigerdiff.py --analysis attention --category Beauty --checkpoint ckpt/your_model.pth

    # Both analyses
    python analysis_tigerdiff.py --analysis all --category Beauty --checkpoint ckpt/your_model.pth
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.colors import LinearSegmentedColormap
from sklearn.decomposition import PCA

matplotlib.rcParams["font.family"] = ["DejaVu Sans", "Arial", "sans-serif"]
matplotlib.rcParams["axes.unicode_minus"] = False

# ──────────────────────────────────────────────
# Helpers to load data WITHOUT the full Pipeline
# ──────────────────────────────────────────────

def load_cached_data(category: str, cache_root: str = "cache/AmazonReviews2014"):
    """Load pre-processed dataset artifacts from cache."""
    base = os.path.join(cache_root, category, "processed")
    with open(os.path.join(base, "id_mapping.json")) as f:
        id_mapping = json.load(f)
    with open(os.path.join(base, "all_item_seqs.json")) as f:
        all_item_seqs = json.load(f)
    meta_path = os.path.join(base, "metadata.sentence.json")
    item2meta = None
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            item2meta = json.load(f)
    return id_mapping, all_item_seqs, item2meta


def load_sentence_embeddings(category: str, emb_dim: int = 768,
                             cache_root: str = "cache/AmazonReviews2014"):
    """Load raw sentence-t5 embeddings from cache."""
    path = os.path.join(cache_root, category, "processed", "sentence-t5-base.sent_emb")
    return np.fromfile(path, dtype=np.float32).reshape(-1, emb_dim)


def get_train_mask(all_item_seqs: dict, item2id: dict, n_items: int):
    """Build boolean mask for items that appear in training sequences."""
    train_items = set()
    for user, seq in all_item_seqs.items():
        for item in seq[:-2]:  # leave-one-out: last 2 are val/test
            train_items.add(item)
    mask = np.zeros(n_items - 1, dtype=bool)
    for item in train_items:
        if item in item2id:
            mask[item2id[item] - 1] = True
    return mask


def apply_pca_and_normalize(sent_embs: np.ndarray, train_mask: np.ndarray,
                            n_components: int = 128):
    """Apply PCA + z-score normalization (same as tokenizer)."""
    pca = PCA(n_components=n_components, whiten=True)
    pca.fit(sent_embs[train_mask])
    transformed = pca.transform(sent_embs)
    train_part = transformed[train_mask]
    mean = np.mean(train_part, axis=0)
    std = np.std(train_part, axis=0)
    std = np.where(std < 1e-6, 1e-6, std)
    return ((transformed - mean) / std).astype(np.float32), pca


def extract_item_categories(item2meta: dict, id2item: list):
    """Extract simple category labels from metadata text for coloring."""
    categories = []
    for i in range(1, len(id2item)):
        raw_item = id2item[i]
        meta_text = item2meta.get(raw_item, "") if item2meta else ""
        categories.append(meta_text[:80])  # truncate for label
    return categories


# ──────────────────────────────────────────────
# Analysis 3: t-SNE / UMAP Embedding Visualization
# ──────────────────────────────────────────────

def run_embedding_analysis(category: str, output_dir: str, method: str = "tsne",
                           max_items: int = 5000, n_clusters: int = 8):
    """
    Visualize item embeddings in 2D using t-SNE or UMAP.
    Colors items by KMeans cluster on the PCA embeddings.
    """
    print(f"[Analysis 3] Loading data for {category}...")
    id_mapping, all_item_seqs, item2meta = load_cached_data(category)
    item2id = id_mapping["item2id"]
    id2item = id_mapping["id2item"]
    n_items = len(id2item)

    raw_embs = load_sentence_embeddings(category)
    train_mask = get_train_mask(all_item_seqs, item2id, n_items)
    pca_embs, pca_model = apply_pca_and_normalize(raw_embs, train_mask, n_components=128)

    print(f"  Total items: {pca_embs.shape[0]}, PCA dim: {pca_embs.shape[1]}")

    # Subsample for visualization speed
    n_total = pca_embs.shape[0]
    if n_total > max_items:
        indices = np.random.choice(n_total, max_items, replace=False)
        indices.sort()
        vis_embs = pca_embs[indices]
    else:
        indices = np.arange(n_total)
        vis_embs = pca_embs

    # Cluster for coloring
    from sklearn.cluster import KMeans
    print(f"  Clustering into {n_clusters} groups...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(vis_embs)

    # Dimensionality reduction to 2D
    if method == "umap":
        try:
            from umap import UMAP
            print("  Running UMAP...")
            reducer = UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
            coords_2d = reducer.fit_transform(vis_embs)
        except ImportError:
            print("  UMAP not installed, falling back to t-SNE")
            method = "tsne"

    if method == "tsne":
        from sklearn.manifold import TSNE
        print("  Running t-SNE (this may take a minute)...")
        tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000,
                     learning_rate="auto", init="pca")
        coords_2d = tsne.fit_transform(vis_embs)

    # ── Plot ──
    fig, ax = plt.subplots(figsize=(10, 8))

    cmap = plt.cm.get_cmap("Spectral", n_clusters)
    scatter = ax.scatter(
        coords_2d[:, 0], coords_2d[:, 1],
        c=cluster_labels, cmap=cmap,
        s=6, alpha=0.6, edgecolors="none", rasterized=True
    )

    cbar = plt.colorbar(scatter, ax=ax, shrink=0.7, pad=0.02)
    cbar.set_label("Cluster ID", fontsize=11)

    method_name = "t-SNE" if method == "tsne" else "UMAP"
    ax.set_title(
        f"Item Embedding Visualization ({method_name})\n"
        f"Amazon {category} — {len(vis_embs):,} items, Sentence-T5 + PCA(128)",
        fontsize=13, fontweight="bold", pad=12
    )
    ax.set_xlabel(f"{method_name} Dimension 1", fontsize=11)
    ax.set_ylabel(f"{method_name} Dimension 2", fontsize=11)
    ax.tick_params(labelsize=9)
    ax.grid(True, alpha=0.15)

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"embedding_{method}_{category}.pdf")
    plt.savefig(out_path, bbox_inches="tight", dpi=300)
    print(f"  ✅ Saved: {out_path}")
    plt.close()

    # ── Also plot PCA explained variance ──
    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    variance_ratio = pca_model.explained_variance_ratio_
    cumulative = np.cumsum(variance_ratio)

    ax1.bar(range(1, len(variance_ratio) + 1), variance_ratio,
            color="#4C72B0", alpha=0.7, width=1.0)
    ax1.set_xlabel("Principal Component", fontsize=11)
    ax1.set_ylabel("Explained Variance Ratio", fontsize=11)
    ax1.set_title("Individual Explained Variance", fontsize=12, fontweight="bold")
    ax1.set_xlim(0, len(variance_ratio) + 1)
    ax1.grid(True, alpha=0.2, axis="y")

    ax2.plot(range(1, len(cumulative) + 1), cumulative,
             color="#C44E52", linewidth=2)
    ax2.fill_between(range(1, len(cumulative) + 1), cumulative,
                     alpha=0.15, color="#C44E52")
    ax2.axhline(y=0.95, color="gray", linestyle="--", alpha=0.6, label="95% threshold")
    ax2.axhline(y=0.99, color="gray", linestyle=":", alpha=0.6, label="99% threshold")
    n95 = np.searchsorted(cumulative, 0.95) + 1
    ax2.annotate(f"{n95} PCs for 95%", xy=(n95, 0.95),
                 xytext=(n95 + 10, 0.88), fontsize=9,
                 arrowprops=dict(arrowstyle="->", color="gray"))
    ax2.set_xlabel("Number of Principal Components", fontsize=11)
    ax2.set_ylabel("Cumulative Explained Variance", fontsize=11)
    ax2.set_title("Cumulative Explained Variance", fontsize=12, fontweight="bold")
    ax2.set_xlim(1, len(cumulative))
    ax2.set_ylim(0, 1.02)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.2)

    fig2.suptitle(
        f"PCA Variance Analysis — Sentence-T5 (768d → 128d)\nAmazon {category}",
        fontsize=13, fontweight="bold", y=1.02
    )
    plt.tight_layout()
    pca_path = os.path.join(output_dir, f"pca_variance_{category}.pdf")
    plt.savefig(pca_path, bbox_inches="tight", dpi=300)
    print(f"  ✅ Saved: {pca_path}")
    plt.close()


# ──────────────────────────────────────────────
# Analysis 12: Attention Weight Heatmap
# ──────────────────────────────────────────────

def build_model_for_analysis(category: str, checkpoint_path: str = None):
    """Build TigerDiff model and optionally load checkpoint for analysis."""
    from accelerate import Accelerator
    from genrec.utils import get_config, get_dataset, get_tokenizer, get_model, init_seed, init_logger

    accelerator = Accelerator()
    config = get_config(
        model_name="TigerDiff",
        dataset_name="AmazonReviews2014",
        config_file=None,
        config_dict={"category": category}
    )
    config["device"] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config["use_ddp"] = False
    config["accelerator"] = accelerator

    init_seed(config["rand_seed"], config["reproducibility"])
    init_logger(config)

    dataset = get_dataset("AmazonReviews2014")(config)
    _ = dataset.split()
    tokenizer = get_tokenizer("TigerDiff")(config, dataset)
    model = get_model("TigerDiff")(config, dataset, tokenizer)

    if checkpoint_path is not None:
        state_dict = torch.load(checkpoint_path, map_location=config["device"])
        model.load_state_dict(state_dict)
        print(f"  Loaded checkpoint: {checkpoint_path}")
    else:
        print("  ⚠️  No checkpoint provided — using randomly initialized weights")

    model.to(config["device"])
    model.eval()

    return model, tokenizer, dataset, config


def extract_attention_weights(model, batch, device):
    """Forward pass through encoder, capturing attention weights from each layer."""
    model.eval()
    batch = {k: v.to(device) for k, v in batch.items()}

    input_embs = model.item_id2embs[batch["input_ids"]]
    input_embs = model.input_proj(input_embs)
    seq_len = input_embs.shape[1]
    positions = torch.arange(seq_len, device=device).unsqueeze(0)
    input_embs = input_embs + model.position_emb(positions)

    attention_mask = batch["attention_mask"].bool()
    hidden_states = input_embs * attention_mask.unsqueeze(-1).float()

    all_attn_weights = []
    for layer in model.encoder.layers:
        normed = layer.attn_norm(hidden_states)
        # Extract attention weights manually
        B, S, _ = normed.shape
        sa = layer.self_attn
        qkv = sa.qkv(normed).view(B, S, 3, sa.num_heads, sa.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(dim=0)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * sa.scale
        key_mask = attention_mask[:, None, None, :].bool()
        attn_scores = attn_scores.masked_fill(~key_mask, float("-inf"))
        attn_probs = torch.softmax(attn_scores, dim=-1)
        attn_probs = attn_probs.masked_fill(~key_mask, 0.0)
        all_attn_weights.append(attn_probs.detach().cpu())

        # Continue forward pass
        context = torch.matmul(attn_probs, v)
        context = context.transpose(1, 2).contiguous().view(B, S, sa.hidden_size)
        context = sa.out_proj(context)
        context = sa.resid_dropout(context)
        hidden_states = hidden_states + context
        hidden_states = hidden_states + layer.ffn(layer.ffn_norm(hidden_states))
        hidden_states = hidden_states * attention_mask.unsqueeze(-1).float()

    return all_attn_weights  # list of [B, n_heads, S, S]


def plot_attention_heatmap(attn_weights, seq_len, item_names, output_path,
                           sample_idx=0, category="Beauty"):
    """
    Plot attention heatmaps for all layers and heads.
    attn_weights: list of [B, n_heads, S, S] tensors, one per layer.
    """
    n_layers = len(attn_weights)
    n_heads = attn_weights[0].shape[1]

    # ── Figure 1: Per-head attention for each layer ──
    fig, axes = plt.subplots(n_layers, n_heads,
                             figsize=(3.5 * n_heads, 3 * n_layers),
                             squeeze=False)

    cmap = LinearSegmentedColormap.from_list("custom",
        ["#FFFFFF", "#FFF7EC", "#FEE8C8", "#FDD49E", "#FDBB84",
         "#FC8D59", "#EF6548", "#D7301F", "#B30000", "#7F0000"])

    for layer_idx in range(n_layers):
        for head_idx in range(n_heads):
            ax = axes[layer_idx][head_idx]
            attn = attn_weights[layer_idx][sample_idx, head_idx, :seq_len, :seq_len].numpy()

            im = ax.imshow(attn, cmap=cmap, aspect="auto", vmin=0, vmax=attn.max())
            ax.set_title(f"L{layer_idx+1} H{head_idx+1}", fontsize=9, fontweight="bold")

            if layer_idx == n_layers - 1:
                ax.set_xlabel("Key Position", fontsize=8)
            if head_idx == 0:
                ax.set_ylabel("Query Position", fontsize=8)
            ax.tick_params(labelsize=6)

            if seq_len <= 15 and item_names:
                short_names = [n[:12] for n in item_names[:seq_len]]
                ax.set_xticks(range(seq_len))
                ax.set_xticklabels(short_names, rotation=45, ha="right", fontsize=5)
                ax.set_yticks(range(seq_len))
                ax.set_yticklabels(short_names, fontsize=5)

    fig.suptitle(
        f"TigerDiff Attention Weights — Amazon {category}\n"
        f"Sequence length = {seq_len}",
        fontsize=14, fontweight="bold", y=1.01
    )
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", dpi=300)
    print(f"  ✅ Saved: {output_path}")
    plt.close()

    # ── Figure 2: Head-averaged attention per layer ──
    fig2, axes2 = plt.subplots(1, n_layers, figsize=(4 * n_layers, 3.5))
    if n_layers == 1:
        axes2 = [axes2]

    for layer_idx in range(n_layers):
        ax = axes2[layer_idx]
        avg_attn = attn_weights[layer_idx][sample_idx, :, :seq_len, :seq_len].mean(dim=0).numpy()
        im = ax.imshow(avg_attn, cmap=cmap, aspect="auto", vmin=0)
        ax.set_title(f"Layer {layer_idx+1} (head-averaged)", fontsize=11, fontweight="bold")
        ax.set_xlabel("Key Position", fontsize=10)
        if layer_idx == 0:
            ax.set_ylabel("Query Position", fontsize=10)
        plt.colorbar(im, ax=ax, shrink=0.8)

    fig2.suptitle(
        f"Head-Averaged Attention — Amazon {category}",
        fontsize=13, fontweight="bold", y=1.02
    )
    plt.tight_layout()
    avg_path = output_path.replace(".pdf", "_averaged.pdf")
    plt.savefig(avg_path, bbox_inches="tight", dpi=300)
    print(f"  ✅ Saved: {avg_path}")
    plt.close()


def run_attention_analysis(category: str, checkpoint_path: str = None,
                           output_dir: str = "vis_results/analysis",
                           n_samples: int = 3):
    """Run attention weight visualization on sample sequences."""
    if checkpoint_path:
        print(f"[Analysis 12] Loading model from {checkpoint_path}...")
    else:
        print(f"[Analysis 12] Building model with random weights (no checkpoint)...")
    model, tokenizer, dataset, config = build_model_for_analysis(category, checkpoint_path)
    device = config["device"]

    # Get test data
    split_data = dataset.split()
    tokenized = tokenizer.tokenize(split_data)
    from torch.utils.data import DataLoader
    test_loader = DataLoader(
        tokenized["test"], batch_size=1, shuffle=False,
        collate_fn=tokenizer.collate_fn["test"]
    )

    os.makedirs(output_dir, exist_ok=True)
    id2item = dataset.id_mapping["id2item"]
    item2meta = dataset.item2meta

    count = 0
    for batch in test_loader:
        if count >= n_samples:
            break

        seq_len = batch["seq_lens"][0].item()
        if seq_len < 5:  # skip very short sequences
            continue

        with torch.no_grad():
            attn_weights = extract_attention_weights(model, batch, device)

        # Get item names for labels
        input_ids = batch["input_ids"][0].cpu().tolist()
        item_names = []
        for iid in input_ids[:seq_len]:
            if 0 < iid < len(id2item):
                raw_id = id2item[iid]
                if item2meta and raw_id in item2meta:
                    title = item2meta[raw_id].split(".")[0][:30]
                    item_names.append(title)
                else:
                    item_names.append(f"item_{iid}")
            else:
                item_names.append("[PAD]")

        suffix = "_random" if checkpoint_path is None else ""
        out_path = os.path.join(output_dir, f"attention_sample{count}_{category}{suffix}.pdf")
        plot_attention_heatmap(
            attn_weights, seq_len, item_names, out_path,
            sample_idx=0, category=category
        )
        count += 1

    print(f"  ✅ Generated {count} attention heatmaps")


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="TigerDiff Analysis")
    parser.add_argument("--analysis", type=str, default="all",
                        choices=["embedding", "attention", "all"],
                        help="Which analysis to run")
    parser.add_argument("--category", type=str, default="Beauty",
                        help="Amazon dataset category")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to model checkpoint (required for attention analysis)")
    parser.add_argument("--output_dir", type=str, default="vis_results/analysis",
                        help="Output directory for figures")
    parser.add_argument("--method", type=str, default="tsne",
                        choices=["tsne", "umap"],
                        help="Dimensionality reduction method for embedding analysis")
    parser.add_argument("--max_items", type=int, default=5000,
                        help="Max items to visualize in embedding plot")
    parser.add_argument("--n_clusters", type=int, default=8,
                        help="Number of KMeans clusters for coloring")
    parser.add_argument("--n_samples", type=int, default=3,
                        help="Number of attention heatmap samples")
    args = parser.parse_args()

    if args.analysis in ("embedding", "all"):
        run_embedding_analysis(
            category=args.category,
            output_dir=args.output_dir,
            method=args.method,
            max_items=args.max_items,
            n_clusters=args.n_clusters,
        )

    if args.analysis in ("attention", "all"):
        run_attention_analysis(
            category=args.category,
            checkpoint_path=args.checkpoint,
            output_dir=args.output_dir,
            n_samples=args.n_samples,
        )


if __name__ == "__main__":
    main()
