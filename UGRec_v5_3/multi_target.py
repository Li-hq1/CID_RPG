"""Chronological future windows. Test suffix values are never mapped or scored."""

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from protocol import ROOT, canonical_category


@dataclass
class Split:
    directory: Path
    mapping: dict
    train: list
    histories: list
    targets: np.ndarray
    users: list
    audit: dict


def split_sequences(sequences, mapping, *, horizon=3, protocol="future_window", min_train_items=2):
    if horizon < 1 or min_train_items < 2 or protocol not in {"future_window", "leave_one_out"}:
        raise ValueError("Invalid chronological split settings.")
    width = horizon if protocol == "future_window" else 1
    train, histories, targets, users = [], [], [], []
    n_items = len(mapping["id2item"]) - 1
    if sorted(v for v in mapping["item2id"].values() if v) != list(range(1, n_items + 1)):
        raise ValueError("Catalog IDs must be contiguous, with zero reserved for padding.")
    if any(mapping["id2item"][v] != k for k, v in mapping["item2id"].items()):
        raise ValueError("Item mapping directions disagree.")
    lengths = []
    for user, sequence in sequences.items():
        lengths.append(len(sequence))
        if len(sequence) < 2 * width + min_train_items:
            continue
        # Do not inspect or look up ANY value in the reserved test suffix.
        development = sequence[:-width]
        ids = [mapping["item2id"][item] for item in development]
        if any(i <= 0 for i in ids):
            raise ValueError("Observed interactions cannot be padding.")
        prefix, heldout = ids[:-width], ids[-width:]
        train.append(prefix)
        histories.append(prefix[-50:])
        targets.append(heldout)
        users.append(user)
    targets = np.asarray(targets, dtype=np.int64).reshape(-1, width)
    audit = {"protocol": protocol, "horizon": horizon, "validation_width": width,
             "test_suffix_reserved": width, "test_targets_accessed": False,
             "split_type": "per_user_contiguous_not_global_time",
             "original_users": len(sequences), "eligible_users": len(users),
             "retention": len(users) / max(len(sequences), 1),
             "min_train_items": min_train_items,
             "training_interactions": sum(map(len, train)),
             "training_items": len({x for seq in train for x in seq}),
             "validation_unique_targets_mean": float(np.mean([len(set(t)) for t in targets])) if len(targets) else 0.,
             "original_length_quantiles": np.quantile(lengths, [0, .25, .5, .75, 1]).tolist() if lengths else []}
    return train, histories, targets, users, audit


def corpus(category, cache_dir=None, *, horizon=3, protocol="future_window", min_train_items=2):
    directory = Path(cache_dir or ROOT / "cache") / "AmazonReviews2014" / canonical_category(category) / "processed"
    mapping = json.loads((directory / "id_mapping.json").read_text())
    sequences = json.loads((directory / "all_item_seqs.json").read_text())
    return Split(directory, mapping, *split_sequences(sequences, mapping, horizon=horizon,
                 protocol=protocol, min_train_items=min_train_items))


class FutureDataset(Dataset):
    """Same next-item windows as v5_1; future labels are strictly train-only."""
    def __init__(self, sequences, horizon):
        if horizon < 1:
            raise ValueError("horizon must be positive")
        self.sequences, self.horizon = sequences, horizon
        self.windows = []
        self.next_queries = self.auxiliary_queries = 0
        for user, seq in enumerate(sequences):
            starts = [0] + list(range(1, max(len(seq) - 50, 1)))
            for start in starts:
                length = min(50, len(seq) - start - 1)
                if length <= 0:
                    continue
                self.windows.append((user, start))
                positions = range(length) if start == 0 else [length - 1]
                for p in positions:
                    self.next_queries += 1
                    self.auxiliary_queries += int(start + p + 1 + horizon <= len(seq))

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, index):
        user, start = self.windows[index]
        seq = self.sequences[user]
        length = min(50, len(seq) - start - 1)
        ids = torch.zeros(50, dtype=torch.long)
        labels = torch.full((50,), -100, dtype=torch.long)
        future = torch.zeros(50, self.horizon, dtype=torch.long)
        ids[:length] = torch.tensor(seq[start:start + length])
        positions = range(length) if start == 0 else [length - 1]
        for p in positions:
            offset = start + p + 1
            labels[p] = seq[offset]
            if offset + self.horizon <= len(seq):
                future[p] = torch.tensor(seq[offset:offset + self.horizon])
        return {"input_ids": ids, "labels": labels, "future_labels": future,
                "attention_mask": ids > 0, "seq_lens": torch.tensor(length)}


def validation_batches(histories, targets, batch_size, indices):
    for start in range(0, len(indices), batch_size):
        chosen = indices[start:start + batch_size]
        ids = torch.zeros(len(chosen), 50, dtype=torch.long)
        for row, index in enumerate(chosen):
            seq = histories[index]
            ids[row, :len(seq)] = torch.tensor(seq)
        # labels are used by scoring only; the model never receives them.
        yield chosen, {"input_ids": ids, "attention_mask": ids > 0,
                       "seq_lens": (ids > 0).sum(1)}, targets[chosen]


def ranking_metrics(predictions, targets, topk=(5, 10)):
    predictions, targets = np.asarray(predictions), np.asarray(targets)
    if targets.ndim == 1:
        targets = targets[:, None]
    if predictions.ndim != 2 or len(predictions) != len(targets) or not len(predictions):
        raise ValueError("Need nonempty aligned predictions and targets.")
    if (predictions <= 0).any() or (targets <= 0).any():
        raise ValueError("Padding cannot be scored.")
    if any(len(set(row)) != len(row) for row in predictions):
        raise ValueError("The recommendation list must contain unique items.")
    relevant = (predictions[:, :, None] == targets[:, None, :]).any(-1)
    counts = np.array([len(set(row)) for row in targets])
    metrics = {}
    for k in topk:
        if predictions.shape[1] < k:
            raise ValueError("Not enough predictions for the requested K.")
        discount = 1 / np.log2(np.arange(k) + 2)
        ideal = np.array([discount[:min(k, n)].sum() for n in counts])
        metrics[f"recall@{k}"] = float((relevant[:, :k].sum(1) / counts).mean())
        metrics[f"ndcg@{k}"] = float(((relevant[:, :k] * discount).sum(1) / ideal).mean())
    for step in range(targets.shape[1]):
        hit = predictions == targets[:, step, None]
        for k in topk:
            metrics[f"step{step + 1}/recall@{k}"] = float(hit[:, :k].any(1).mean())
            metrics[f"step{step + 1}/ndcg@{k}"] = float((hit[:, :k] / np.log2(np.arange(k) + 2)).sum(1).mean())
    return metrics
