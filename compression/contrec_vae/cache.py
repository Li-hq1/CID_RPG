"""Artifact identities, atomic persistence, and strict feature loading."""

import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np

from data.embeddings import embedding_settings, read_sentence_embeddings, sentence_embedding_cache_path


FORMAT_VERSION = 1
ARTIFACT_FILES = (
    "best.pt", "splits.npz", "raw_latents.npy", "latents.npy",
    "statistics.npz", "metrics.json", "loss.csv",
)


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_hash(values):
    return hashlib.sha256(memoryview(np.ascontiguousarray(values)).cast("B")).hexdigest()


def training_item_mask(dataset):
    splits = dataset.split()
    mask = np.zeros(dataset.n_items - 1, dtype=bool)
    for sequence in splits["train"]["item_seq"]:
        for item in sequence:
            mask[dataset.item2id[item] - 1] = True
    return mask


def source_metadata(embeddings, dataset, config, train_mask):
    if embeddings.shape != (dataset.n_items - 1, config["sent_emb_dim"]):
        raise ValueError("Raw embedding shape does not match the dataset and encoder settings.")
    if embeddings.dtype != np.float32 or not np.isfinite(embeddings).all():
        raise ValueError("Raw embeddings must contain finite float32 values.")
    if train_mask.shape != (len(embeddings),) or not train_mask.any():
        raise ValueError("No matching recommendation training items found.")
    return {
        "dataset": config.get("dataset", "AmazonReviews2014"),
        "category": config["category"],
        "embedding_settings": embedding_settings(config),
        "input_shape": list(embeddings.shape),
        "embeddings_sha256": array_hash(embeddings),
        "item_order_sha256": json_hash(dataset.id_mapping["id2item"]),
        "train_mask_sha256": array_hash(train_mask),
    }


def experiment_key(source, vae_config):
    return json_hash({"format_version": FORMAT_VERSION, "source": source, "vae_config": vae_config})


def artifact_directory(dataset_cache, source, vae_config):
    root = Path(dataset_cache) / "ContRec_VAE"
    if vae_config.get("smoke_items") is not None:
        root /= "smoke"
    return root / experiment_key(source, vae_config)[:20]


def atomic_write(path, writer):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
    try:
        with temporary.open("wb") as stream:
            writer(stream)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_json(path, value):
    atomic_write(path, lambda stream: stream.write(
        (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()))


def read_manifest(directory):
    directory = Path(directory)
    path = directory / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"Incomplete VAE artifact: {path}. Train or export the VAE first.")
    with path.open() as stream:
        manifest = json.load(stream)
    if manifest.get("format_version") != FORMAT_VERSION:
        raise ValueError("Unsupported VAE artifact format.")
    key = experiment_key(manifest["source"], manifest["vae_config"])
    if manifest.get("experiment_key") != key:
        raise ValueError("VAE manifest experiment identity is invalid.")
    hashes = manifest.get("files", {})
    if set(hashes) != set(ARTIFACT_FILES):
        raise ValueError("Incomplete VAE artifact file manifest.")
    for name in ARTIFACT_FILES:
        path = directory / name
        if not path.is_file():
            raise FileNotFoundError(f"Incomplete VAE artifact: {path}.")
        if file_hash(path) != hashes[name]:
            raise ValueError(f"VAE artifact checksum mismatch: {path}.")
    identity = json_hash({"experiment_key": key, "files": hashes})
    if manifest.get("artifact_id") != identity:
        raise ValueError("VAE artifact identity is invalid.")
    return manifest


def load_features(directory, dataset, config):
    directory = Path(directory).expanduser().resolve()
    root = (Path(dataset.cache_dir) / "ContRec_VAE").resolve()
    if not directory.is_relative_to(root):
        raise ValueError(f"VAE artifacts must be under {root}.")
    manifest = read_manifest(directory)
    if manifest["vae_config"].get("smoke_items") is not None and not config.get("smoke_test", False):
        raise ValueError("Smoke VAE artifacts may only be loaded with --smoke-test.")
    path = sentence_embedding_cache_path(config, dataset.cache_dir)
    if not Path(path).is_file():
        raise FileNotFoundError(f"Raw Qwen cache required to verify VAE source: {path}.")
    embeddings = read_sentence_embeddings(path, config["sent_emb_dim"], dataset.n_items - 1)
    source = source_metadata(embeddings, dataset, config, training_item_mask(dataset))
    if manifest["source"] != source:
        raise ValueError("VAE source mismatch: encoder settings, features, item order or training items changed.")
    features = np.load(directory / "latents.npy", allow_pickle=False)
    shape = (dataset.n_items - 1, manifest["vae_config"]["latent_dim"])
    if features.shape != shape or features.dtype != np.float32 or not np.isfinite(features).all():
        raise ValueError(f"Invalid VAE features: expected finite float32 {shape}.")
    return features, manifest["artifact_id"]
