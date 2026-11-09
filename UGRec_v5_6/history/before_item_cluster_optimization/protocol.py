"""Data windows and validation scoring for the frozen scheme.

The window rule matches the training samples used to reproduce the frozen
v4_2 checkpoint. Validation targets are the penultimate item of each sequence.
The final item is the test label and is never read.
"""

import json
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parent
SELECTION_SEED = 20261028
WINDOW = 51
CATEGORIES = {"beauty": "Beauty", "sports": "Sports_and_Outdoors",
              "toys": "Toys_and_Games", "cds": "CDs_and_Vinyl"}


def canonical_category(category):
    for alias, canonical in CATEGORIES.items():
        if category.lower() in {alias, canonical.lower()}:
            return canonical
    raise ValueError(f"Unknown dataset {category!r}; use Beauty, sports, toys, CDs.")


def corpus(category, cache_dir=None):
    directory = Path(cache_dir or ROOT / "cache") / "AmazonReviews2014" / canonical_category(category) / "processed"
    mapping = json.loads((directory / "id_mapping.json").read_text())
    sequences = json.loads((directory / "all_item_seqs.json").read_text())
    n_items = len(mapping["id2item"]) - 1
    positive_ids = [value for value in mapping["item2id"].values() if value != 0]
    if sorted(positive_ids) != list(range(1, n_items + 1)):
        raise ValueError("Item mapping must be contiguous, with zero reserved for padding.")
    if any(mapping["id2item"][value] != key for key, value in mapping["item2id"].items()):
        raise ValueError("Item mapping directions disagree.")
    if not sequences or any(len(seq) < 3 for seq in sequences.values()):
        raise ValueError("Leave-one-out requires nonempty users and at least three items per user.")
    # The shared file contains test items; exclude the final element before
    # even looking it up. Test targets never enter training or validation.
    ids = [[mapping["item2id"][item] for item in seq[:-1]] for seq in sequences.values()]
    if any(item <= 0 for seq in ids for item in seq):
        raise ValueError("Padding cannot be an observed interaction.")
    train = [seq[:-1] for seq in ids]
    histories = [seq[:-1][-50:] for seq in ids]
    targets = np.array([seq[-1] for seq in ids], dtype=np.int64)
    return directory, mapping, train, histories, targets


def raw_features(directory, n_items):
    path = directory / "Qwen3-Embedding-4B_d14cc3313ecc.sent_emb"
    expected = n_items * 2560 * np.dtype(np.float32).itemsize
    if path.stat().st_size != expected:
        raise ValueError(f"Embedding byte count mismatch: {path}, expected {expected}.")
    features = np.memmap(path, dtype=np.float32, mode="c", shape=(n_items, 2560))
    for start in range(0, n_items, 4096):
        if not np.isfinite(features[start:start + 4096]).all():
            raise ValueError("Non-finite content embeddings.")
    return features


def training_windows(train):
    """First window supervises every next-item; later windows supervise only the last."""
    inputs, labels = [], []
    for seq in train:
        windows = [seq[:WINDOW]] + [seq[start:start + WINDOW] for start in range(1, max(len(seq) - 50, 1))]
        for index, window in enumerate(windows):
            if len(window) < 2:
                continue
            x = window[:-1] + [0] * (WINDOW - len(window))
            y = window[1:] + [-100] * (WINDOW - len(window))
            if index > 0:
                y[:len(window) - 2] = [-100] * (len(window) - 2)
            inputs.append(x)
            labels.append(y)
    return inputs, labels


def training_dataset(train):
    inputs, labels = training_windows(train)
    return torch.utils.data.TensorDataset(torch.tensor(inputs), torch.tensor(labels))


def validation_batches(histories, targets, batch_size, indices):
    for start in range(0, len(indices), batch_size):
        chosen = indices[start:start + batch_size]
        selected = [histories[i] for i in chosen]
        ids = torch.zeros(len(chosen), 50, dtype=torch.long)
        for row, seq in enumerate(selected):
            ids[row, :len(seq)] = torch.tensor(seq)
        yield chosen, {
            "input_ids": ids,
            "attention_mask": ids > 0,
            "seq_lens": (ids > 0).sum(1),
            "labels": torch.tensor(targets[chosen])[:, None],
        }


def ranking_metrics(predictions, targets):
    predictions = np.asarray(predictions)
    hit = predictions == np.asarray(targets)[:, None]
    output = {}
    for k in (5, 10):
        output[f"recall@{k}"] = float(hit[:, :k].any(1).mean())
        output[f"ndcg@{k}"] = float((hit[:, :k] / np.log2(np.arange(k) + 2)).sum(1).mean())
    return output
