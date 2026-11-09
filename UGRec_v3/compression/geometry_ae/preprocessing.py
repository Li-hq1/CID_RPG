"""Deterministic item splits, batches and latent standardization."""

import numpy as np
import torch

def split_items(train_mask, config):
    config.validate()
    rng = np.random.default_rng(config.seed)
    indices = rng.permutation(np.flatnonzero(train_mask))
    if config.smoke_items is not None:
        indices = indices[:config.smoke_items]
    if len(indices) < 4:
        raise ValueError("AE needs at least four training items (two train, one valid, one test).")
    test_size = max(1, int(len(indices) * config.test_ratio))
    remaining = len(indices) - test_size
    valid_size = max(1, int(remaining * config.valid_ratio))
    if remaining - valid_size < 2:
        raise ValueError("AE split leaves fewer than two training items for training.")
    return {"test": indices[:test_size],
            "valid": indices[test_size:test_size + valid_size],
            "train": indices[test_size + valid_size:]}


def batch_indices(indices, batch_size, *, generator=None, training=False):
    if training:
        if batch_size < 2 or len(indices) < 2:
            raise ValueError("Training batches require at least two items for training.")
        indices = indices[torch.randperm(len(indices), generator=generator).numpy()]
    batches = [indices[start:start + batch_size] for start in range(0, len(indices), batch_size)]
    if training and len(batches) > 1 and len(batches[-1]) == 1:
        batches[-2] = np.concatenate((batches[-2], batches[-1]))
        batches.pop()
    return batches


def standardize_latents(latents, train_mask):
    if train_mask.shape != (len(latents),) or not train_mask.any():
        raise ValueError("Invalid training mask for latent standardization.")
    selected = latents[train_mask]
    mean = selected.mean(axis=0)
    std = np.maximum(selected.std(axis=0), 1e-6)
    features = ((latents - mean) / std).astype(np.float32)
    if not np.isfinite(features).all():
        raise ValueError("Non-finite standardized AE features.")
    return features, mean, std

