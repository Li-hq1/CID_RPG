"""Random autoencoder initialization for the frozen scheme."""

from logging import getLogger

import numpy as np
import torch

from compression.geometry_ae.model import GeometryAE


def fit_input_statistics(embeddings, indices):
    selected = embeddings[indices].astype(np.float64)
    mean = selected.mean(0)
    scale = float(np.sqrt(np.mean((selected - mean) ** 2)))
    if not np.isfinite(scale) or scale <= 1e-8:
        raise ValueError("Compression training inputs have no usable variance.")
    return torch.from_numpy(mean.astype(np.float32)), torch.tensor(scale, dtype=torch.float32)


def initialize_random_ae(config, raw, train_ids):
    """Fit input statistics on training items, then use the default linear initialization."""
    if raw.ndim != 2 or raw.dtype != np.float32 or min(raw.shape) <= 0 or not np.isfinite(raw).all():
        raise ValueError("Random AE initialization requires finite float32 embeddings.")
    indices = train_ids.detach().cpu().numpy().astype(np.int64, copy=False) - 1
    if (indices.ndim != 1 or len(indices) < 2 or np.unique(indices).size != len(indices)
            or indices.min() < 0 or indices.max() >= len(raw)):
        raise ValueError("Random AE initialization needs at least two unique in-range training items.")
    mean, scale = fit_input_statistics(raw, indices)
    # Seed only the CPU generator used by Linear.reset_parameters, and restore it
    # so the recommender keeps the RNG stream established by seed_run.
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(config.training.seed)
        model = GeometryAE(raw.shape[1], config.ae, mean, scale)
    getLogger(__name__).info("Random AE initialization on %d training items, input_scale=%.6g",
                             len(indices), float(scale))
    return model.eval()


def fit_input_codes(raw, levels, size, iterations=25, seed=0, device="cpu"):
    """Residual k-means on the fixed content embedding: [items, levels] codes.

    Uses every catalog item's content vector and no interaction labels, as with
    TIGER/RPG semantic IDs. Its own generator leaves the training RNG untouched.
    Distances use exact fp32 (no TF32), so CPU and GPU give the same codes.
    """
    if levels <= 0:
        raise ValueError("Input codes need at least one level.")
    precision = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")
    try:
        return _fit_input_codes(raw, levels, size, iterations, seed, device)
    finally:
        torch.set_float32_matmul_precision(precision)


def _fit_input_codes(raw, levels, size, iterations, seed, device):
    x = torch.as_tensor(raw, dtype=torch.float32, device=device)
    if size > len(x):
        raise ValueError("Code size cannot exceed the number of items.")
    generator = torch.Generator().manual_seed(seed)
    residual, codes = x.clone(), []
    for _ in range(levels):
        centers = residual[torch.randperm(len(residual), generator=generator)[:size].to(x.device)].clone()
        for _ in range(iterations):
            assign = torch.cdist(residual, centers).argmin(1)
            total = torch.zeros_like(centers).index_add_(0, assign, residual)
            count = torch.bincount(assign, minlength=size).float()[:, None]
            centers = torch.where(count > 0, total / count.clamp_min(1), centers)
        assign = torch.cdist(residual, centers).argmin(1)
        residual = residual - centers[assign]
        codes.append(assign)
    return torch.stack(codes, 1).cpu()
