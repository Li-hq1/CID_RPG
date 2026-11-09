"""Item-ID samples and AE initialization for joint training."""

from dataclasses import asdict
from logging import getLogger
from pathlib import Path

import numpy as np
import torch

from misc.artifacts import json_hash, source_metadata, training_item_mask
from compression.geometry_ae.model import GeometryAE
from compression.geometry_ae.pipeline import fit_input_statistics
from data.amazon_reviews import AmazonReviews2014
from data.embeddings import load_item_sentence_embeddings
from data.tigerdiff_data import TigerDiffTokenizer


class JointTokenizer(TigerDiffTokenizer):
    def __init__(self, config, dataset):
        self.config = config
        self.logger = getLogger(__name__)
        self.eos_token = None
        self.collate_fn = {split: None for split in ("train", "val", "test")}
        self.padding_token = 0
        self.cache_dir = dataset.cache_dir
        self.item2id = dataset.item2id
        self.user2id = dataset.user2id
        self.id2item = dataset.id_mapping["id2item"]
        self.item2tokens = range(dataset.n_items - 1)
        self.REGENERATE_SAMPLE = False
        self.ignored_label = -100
        self._sample_source = {
            "item_order": self.id2item,
            "split": config["split"],
            "max_item_seq_len": self.max_token_seq_len,
            "policy": "first_window_all_next_then_last_next_v1",
        }

    def _tokenized_cache_path(self):
        return str(Path(self.cache_dir) / "JointTraining" / "samples" / json_hash(self._sample_source)[:20])

    def tokenize(self, datasets):
        source = {**self._sample_source,
                  "splits": {key: values.to_dict() for key, values in datasets.items()}}
        self._sample_source = source
        return super().tokenize(datasets)


def data_settings(config, accelerator):
    settings = asdict(config.data)
    device = config.training.device
    settings["device"] = "cuda:0" if device == "auto" and torch.cuda.is_available() else "cpu" if device == "auto" else device
    settings["accelerator"] = accelerator
    return settings


def load_data(config, accelerator):
    settings = data_settings(config, accelerator)
    dataset = AmazonReviews2014(settings)
    mask = training_item_mask(dataset)
    raw = load_item_sentence_embeddings(dataset, settings)
    source = source_metadata(raw, dataset, settings, mask)
    train_ids = torch.from_numpy(np.flatnonzero(mask).astype(np.int64) + 1)
    return dataset, JointTokenizer(settings, dataset), raw, train_ids, source


def initialize_random_ae(config, raw, train_ids):
    """Match GeometryAE pretraining: fit input statistics, then default Linear init."""
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
