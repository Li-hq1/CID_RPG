"""Versioned artifacts isolated from legacy SigmaVae experiments."""

import json
from pathlib import Path
import numpy as np

from compression.contrec_vae.cache import (
    atomic_write, file_hash, json_hash, source_metadata, training_item_mask, write_json,
)
from data.embeddings import read_sentence_embeddings, sentence_embedding_cache_path

FORMAT_VERSION = 1
METHOD = "geometry_ae"
ARTIFACT_FILES = ("best.pt", "splits.npz", "raw_latents.npy", "latents.npy", "statistics.npz",
                  "metrics.json", "loss.csv", "diagnostics.json")


def experiment_key(source, settings):
    return json_hash({"method": METHOD, "format_version": FORMAT_VERSION,
                      "source": source, "ae_config": settings})


def artifact_directory(cache_dir, source, settings):
    root = Path(cache_dir) / "Geometry_AE"
    if settings.get("smoke_items") is not None:
        root /= "smoke"
    elif settings.get("train_items") is not None:
        root /= "ablation"
    return root / experiment_key(source, settings)[:20]


def read_manifest(directory):
    directory = Path(directory)
    with (directory / "manifest.json").open() as stream:
        manifest = json.load(stream)
    if manifest.get("method") != METHOD or manifest.get("format_version") != FORMAT_VERSION:
        raise ValueError("Unsupported geometry AE artifact.")
    key = experiment_key(manifest["source"], manifest["ae_config"])
    if manifest.get("experiment_key") != key:
        raise ValueError("Geometry AE experiment identity mismatch.")
    hashes = manifest.get("files", {})
    if set(hashes) != set(ARTIFACT_FILES):
        raise ValueError("Incomplete geometry AE file manifest.")
    for name, digest in hashes.items():
        if file_hash(directory / name) != digest:
            raise ValueError(f"Geometry AE checksum mismatch: {name}.")
    if manifest.get("artifact_id") != json_hash({"experiment_key": key, "files": hashes}):
        raise ValueError("Geometry AE artifact identity mismatch.")
    return manifest


def load_features(directory, dataset, config):
    directory = Path(directory).expanduser().resolve()
    if not directory.is_relative_to((Path(dataset.cache_dir) / "Geometry_AE").resolve()):
        raise ValueError("Geometry AE artifacts must be under the dataset Geometry_AE directory.")
    manifest = read_manifest(directory)
    if manifest["ae_config"].get("smoke_items") is not None and not config.get("smoke_test", False):
        raise ValueError("Smoke geometry AE artifacts require --smoke-test.")
    path = sentence_embedding_cache_path(config, dataset.cache_dir)
    embeddings = read_sentence_embeddings(path, config["sent_emb_dim"], dataset.n_items - 1)
    if manifest["source"] != source_metadata(embeddings, dataset, config, training_item_mask(dataset)):
        raise ValueError("Geometry AE source mismatch: features, item order or training population changed.")
    values = np.load(directory / "latents.npy", allow_pickle=False)
    shape = (dataset.n_items - 1, manifest["ae_config"]["latent_dim"])
    if values.shape != shape or values.dtype != np.float32 or not np.isfinite(values).all():
        raise ValueError(f"Invalid geometry AE features; expected finite float32 {shape}.")
    return values, manifest["artifact_id"]
