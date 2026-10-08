"""Item-only pretraining, reconstruction evaluation, and latent export."""

import csv
import io
import logging
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .cache import (
    ARTIFACT_FILES, FORMAT_VERSION, artifact_directory, atomic_write,
    experiment_key, file_hash, json_hash, read_manifest, source_metadata, write_json,
)
from .config import VAEConfig
from .model import SigmaVae, configure_loss


logger = logging.getLogger(__name__)


def split_items(train_mask, config):
    config.validate()
    rng = np.random.default_rng(config.seed)
    indices = rng.permutation(np.flatnonzero(train_mask))
    if config.smoke_items is not None:
        indices = indices[:config.smoke_items]
    if len(indices) < 4:
        raise ValueError("VAE needs at least four training items (two train, one valid, one test).")
    test_size = max(1, int(len(indices) * config.test_ratio))
    remaining = len(indices) - test_size
    valid_size = max(1, int(remaining * config.valid_ratio))
    if remaining - valid_size < 2:
        raise ValueError("VAE split leaves fewer than two training items for BatchNorm.")
    return {"test": indices[:test_size],
            "valid": indices[test_size:test_size + valid_size],
            "train": indices[test_size + valid_size:]}


def batch_indices(indices, batch_size, *, generator=None, training=False):
    if training:
        if batch_size < 2 or len(indices) < 2:
            raise ValueError("Training batches require at least two items for BatchNorm.")
        indices = indices[torch.randperm(len(indices), generator=generator).numpy()]
    batches = [indices[start:start + batch_size] for start in range(0, len(indices), batch_size)]
    if training and len(batches) > 1 and len(batches[-1]) == 1:
        batches[-2] = np.concatenate((batches[-2], batches[-1]))
        batches.pop()
    return batches


@torch.no_grad()
def evaluate_reconstruction(model, embeddings, indices, batch_size, device):
    model.eval()
    squared_error, cosine, elements, rows = 0.0, 0.0, 0, 0
    for batch in batch_indices(indices, batch_size):
        x = torch.from_numpy(embeddings[batch]).to(device)
        reconstructed = model.valid(x)[0]
        squared_error += F.mse_loss(reconstructed, x, reduction="sum").item()
        cosine += F.cosine_similarity(reconstructed, x, dim=1).sum().item()
        elements += x.numel()
        rows += len(x)
    metrics = {"mse": squared_error / elements, "cosine_similarity": cosine / rows}
    if not all(np.isfinite(value) for value in metrics.values()):
        raise ValueError("Non-finite VAE reconstruction metrics.")
    return metrics


def standardize_latents(latents, train_mask):
    if train_mask.shape != (len(latents),) or not train_mask.any():
        raise ValueError("Invalid training mask for latent standardization.")
    selected = latents[train_mask]
    mean = selected.mean(axis=0)
    std = np.maximum(selected.std(axis=0), 1e-6)
    features = ((latents - mean) / std).astype(np.float32)
    if not np.isfinite(features).all():
        raise ValueError("Non-finite standardized VAE features.")
    return features, mean, std


def write_history(path, history):
    stream = io.StringIO(newline="")
    fieldnames = [
        "epoch", "train_mse", "train_mu", "train_twins", "train_loss", "valid_mse",
    ]
    writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(history)
    atomic_write(path, lambda output: output.write(stream.getvalue().encode()))


def load_best(directory, source, device="cpu"):
    checkpoint = torch.load(Path(directory) / "best.pt", map_location="cpu", weights_only=True)
    if checkpoint.get("format_version") != FORMAT_VERSION or checkpoint.get("source") != source:
        raise ValueError("VAE checkpoint format or input source mismatch.")
    config = VAEConfig(**checkpoint["vae_config"])
    settings = config.to_dict()
    if checkpoint.get("experiment_key") != experiment_key(source, settings):
        raise ValueError("VAE checkpoint experiment identity mismatch.")
    model = SigmaVae(source["input_shape"][1], config.latent_dim, 2,
                     source["input_shape"][0], config.mask_ratio, config.sigma, config.seed)
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()
    return model, checkpoint, config


@torch.no_grad()
def export_artifact(directory, embeddings, dataset, data_config, train_mask, *, device="cpu"):
    directory = Path(directory)
    source = source_metadata(embeddings, dataset, data_config, train_mask)
    model, checkpoint, config = load_best(directory, source, device)
    splits = {key: values.numpy() for key, values in checkpoint["splits"].items()}
    expected_splits = split_items(train_mask, config)
    if set(splits) != set(expected_splits) or any(
            not np.array_equal(splits[key], expected_splits[key]) for key in expected_splits):
        raise ValueError("VAE checkpoint item splits do not match the configured training population.")
    directory.joinpath("manifest.json").unlink(missing_ok=True)
    metrics = {split: evaluate_reconstruction(model, embeddings, indices, config.eval_batch_size, device)
               for split, indices in splits.items()}
    metrics.update(best_epoch=checkpoint["epoch"],
                   split_sizes={split: len(indices) for split, indices in splits.items()})
    latents = np.empty((len(embeddings), config.latent_dim), dtype=np.float32)
    for indices in batch_indices(np.arange(len(embeddings)), config.eval_batch_size):
        x = torch.from_numpy(embeddings[indices]).to(device)
        latents[indices] = model.latent(x).cpu().numpy()
    if not np.isfinite(latents).all():
        raise ValueError("Non-finite raw VAE latents.")
    features, mean, std = standardize_latents(latents, train_mask)
    atomic_write(directory / "raw_latents.npy", lambda stream: np.save(stream, latents, allow_pickle=False))
    atomic_write(directory / "latents.npy", lambda stream: np.save(stream, features, allow_pickle=False))
    atomic_write(directory / "statistics.npz", lambda stream: np.savez(stream, mean=mean, std=std))
    atomic_write(directory / "splits.npz", lambda stream: np.savez(stream, **splits))
    if not (directory / "loss.csv").is_file():
        write_history(directory / "loss.csv", checkpoint["history"])
    write_json(directory / "metrics.json", metrics)
    hashes = {name: file_hash(directory / name) for name in ARTIFACT_FILES}
    key = checkpoint["experiment_key"]
    manifest = {
        "format_version": FORMAT_VERSION, "experiment_key": key,
        "source": source, "vae_config": config.to_dict(),
        "artifact_id": json_hash({"experiment_key": key, "files": hashes}),
        "files": hashes, "feature_shape": list(features.shape),
        "row_order": "row i corresponds to non-padding item ID i + 1",
        "standardization_population": "recommendation_training_items",
    }
    # Only this final file marks the entire export as usable.
    write_json(directory / "manifest.json", manifest)
    logger.info("Exported %s VAE features to %s", features.shape, directory)
    return directory


def train_artifact(embeddings, dataset, data_config, train_mask, config=None, *, device="cpu", force=False):
    config = config or VAEConfig()
    settings = config.to_dict()
    source = source_metadata(embeddings, dataset, data_config, train_mask)
    directory = artifact_directory(dataset.cache_dir, source, settings)
    if (directory / "manifest.json").exists() and not force:
        manifest = read_manifest(directory)
        if manifest["source"] != source or manifest["vae_config"] != settings:
            raise ValueError("Cached VAE experiment does not match its source or training configuration.")
        logger.info("Reusing complete VAE artifact: %s", directory)
        return directory
    directory.mkdir(parents=True, exist_ok=True)
    directory.joinpath("manifest.json").unlink(missing_ok=True)
    directory.joinpath("loss.csv").unlink(missing_ok=True)
    splits = split_items(train_mask, config)
    atomic_write(directory / "splits.npz", lambda stream: np.savez(stream, **splits))
    torch.manual_seed(config.seed)
    generator = torch.Generator().manual_seed(config.seed)
    model = SigmaVae(embeddings.shape[1], config.latent_dim, 2, len(embeddings),
                     config.mask_ratio, config.sigma, config.seed).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    best_loss, stale, history = float("inf"), 0, []
    for epoch in range(1, config.epochs + 1):
        model.train()
        totals = {"reconstruction": 0.0, "mu": 0.0, "twins": 0.0, "loss": 0.0}
        batches = 0
        for indices in batch_indices(splits["train"], config.train_batch_size,
                                     generator=generator, training=True):
            x = torch.from_numpy(embeddings[indices]).to(device)
            optimizer.zero_grad(set_to_none=True)
            reconstructed, mus, encoded = model(x)
            terms = configure_loss(
                x, reconstructed, mus, encoded,
                beta=config.beta, gamma=config.gamma, twins_lambda=config.twins_lambda,
            )
            loss = terms["loss"]
            if not torch.isfinite(loss):
                raise ValueError("Non-finite VAE training loss.")
            loss.backward()
            optimizer.step()
            totals["reconstruction"] += terms["reconstruction"].item()
            totals["mu"] += terms["mu"].item()
            totals["twins"] += terms["twins"].item()
            totals["loss"] += loss.item()
            batches += 1
        scale = max(batches, 1)
        valid_loss = evaluate_reconstruction(
            model, embeddings, splits["valid"], config.eval_batch_size, device)["mse"]
        history.append({
            "epoch": epoch,
            "train_mse": totals["reconstruction"] / scale,
            "train_mu": totals["mu"] / scale,
            "train_twins": totals["twins"] / scale,
            "train_loss": totals["loss"] / scale,
            "valid_mse": valid_loss,
        })
        if valid_loss < best_loss:
            best_loss, stale = valid_loss, 0
            checkpoint = {
                "format_version": FORMAT_VERSION, "experiment_key": experiment_key(source, settings),
                "source": source, "vae_config": settings, "model": model.state_dict(),
                "epoch": epoch, "valid_mse": valid_loss, "history": history,
                "splits": {key: torch.from_numpy(values) for key, values in splits.items()},
            }
            atomic_write(directory / "best.pt", lambda stream: torch.save(checkpoint, stream))
        else:
            stale += 1
        if epoch == 1 or epoch % 10 == 0 or epoch == config.epochs:
            row = history[-1]
            logger.info(
                "VAE epoch %d/%d train recon %.6g mu %.6g twins %.6g loss %.6g valid MSE %.6g",
                epoch, config.epochs, row["train_mse"], row["train_mu"],
                row["train_twins"], row["train_loss"], valid_loss,
            )
        if stale >= config.patience:
            logger.info("VAE early stopping at epoch %d", epoch)
            break
    write_history(directory / "loss.csv", history)
    del optimizer, model
    if torch.device(device).type == "cuda":
        with torch.cuda.device(device):
            torch.cuda.empty_cache()
    return export_artifact(directory, embeddings, dataset, data_config, train_mask, device=device)
