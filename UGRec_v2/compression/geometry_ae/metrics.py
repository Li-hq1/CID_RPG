"""Collapse diagnostics and reproducible neighborhood comparisons."""

import numpy as np


def geometry_metrics(values):
    centered = values.astype(np.float64) - values.mean(0, dtype=np.float64)
    std = np.sqrt(np.mean(centered ** 2, axis=0))
    cov = centered.T @ centered / len(values)
    eigenvalues = np.maximum(np.linalg.eigvalsh(cov), 0)[::-1]
    total = eigenvalues.sum()
    probabilities = eigenvalues / total if total > 0 else np.zeros_like(eigenvalues)
    positive = probabilities[probabilities > 0]
    normalized = centered / np.maximum(std, 1e-8)
    correlation = normalized.T @ normalized / len(values)
    off = correlation[~np.eye(values.shape[1], dtype=bool)]
    return {
        "effective_rank": float(np.exp(-(positive * np.log(positive)).sum())) if total > 0 else 0.,
        "participation_rank": float(1 / np.sum(probabilities ** 2)) if total > 0 else 0.,
        "top1_variance_fraction": float(probabilities[0]),
        "std_min": float(std.min()), "std_median": float(np.median(std)),
        "std_max": float(std.max()), "per_dim_std": std.tolist(),
        "low_std_dimensions": int((std < 1e-6).sum()),
        "off_diagonal_correlation_rms": float(np.sqrt(np.mean(off ** 2))) if off.size else 0.,
    }


def neighbor_indices(values, candidates, queries, k):
    normalized = values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)
    similarity = normalized[queries] @ normalized[candidates].T
    similarity[queries[:, None] == candidates[None, :]] = -np.inf
    return np.argpartition(-similarity, k - 1, axis=1)[:, :k]


def neighbor_comparison(x, z, pca_features, config):
    if len(x) < 2:
        return {"raw_neighbor_overlap": 0., "pca_neighbor_overlap": 0.}
    rng = np.random.default_rng(config.seed)
    candidates = rng.choice(len(x), min(len(x), config.neighbor_candidates), replace=False)
    queries = candidates[:min(config.neighbor_queries, len(candidates))]
    k = min(config.neighbor_k, len(candidates) - 1)
    student = neighbor_indices(z, candidates, queries, k)

    def overlap(teacher):
        return float(np.mean([len(set(a) & set(b)) / k for a, b in zip(student, teacher)]))

    return {"raw_neighbor_overlap": overlap(neighbor_indices(x, candidates, queries, k)),
            "pca_neighbor_overlap": overlap(neighbor_indices(pca_features, candidates, queries, k)),
            "candidate_count": len(candidates), "query_count": len(queries), "k": k}
