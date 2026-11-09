"""Artifact persistence and source identities; hashes preserve existing experiments."""

import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np

from data.embeddings import embedding_settings

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

