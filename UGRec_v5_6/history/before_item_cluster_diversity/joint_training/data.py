"""Random autoencoder initialization for the frozen scheme."""

from logging import getLogger

import numpy as np
import torch

from compression.geometry_ae.model import GeometryAE
from compression.geometry_ae.cluster_model import ClusterGeometryAE


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
        model_type = ClusterGeometryAE if config.ae.cluster_count else GeometryAE
        model = model_type(raw.shape[1], config.ae, mean, scale)
    getLogger(__name__).info("Random AE initialization on %d training items, input_scale=%.6g",
                             len(indices), float(scale))
    return model.eval()
