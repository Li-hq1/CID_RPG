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


def corpus(category):
    directory = ROOT / "cache/AmazonReviews2014" / category / "processed"
    mapping = json.loads((directory / "id_mapping.json").read_text())
    sequences = json.loads((directory / "all_item_seqs.json").read_text())
    ids = [[mapping["item2id"][item] for item in seq] for seq in sequences.values()]
    train = [seq[:-2] for seq in ids]
    histories = [seq[:-2][-50:] for seq in ids]
    targets = np.array([seq[-2] for seq in ids], dtype=np.int64)
    return directory, mapping, train, histories, targets


def raw_features(directory, n_items):
    path = directory / "Qwen3-Embedding-4B_d14cc3313ecc.sent_emb"
    features = np.fromfile(path, dtype=np.float32).reshape(n_items, 2560)
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
