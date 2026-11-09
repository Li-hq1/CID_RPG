"""Read existing data without changing shared caches or accessing test targets."""

from dataclasses import fields
import json
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
SEED = 20261028


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def corpus(category):
    directory = ROOT / "cache/AmazonReviews2014" / category / "processed"
    mapping = json.loads((directory / "id_mapping.json").read_text())
    sequences = json.loads((directory / "all_item_seqs.json").read_text())
    ids = [[mapping["item2id"][item] for item in seq] for seq in sequences.values()]
    # Never use the final item in any fitting, query, metric or model selection.
    train = [seq[:-2] for seq in ids]
    histories = [seq[:-2][-50:] for seq in ids]
    targets = np.array([seq[-2] for seq in ids], dtype=np.int64)
    return directory, mapping, train, histories, targets


def raw_features(directory, n_items):
    path = directory / "Qwen3-Embedding-4B_d14cc3313ecc.sent_emb"
    features = np.fromfile(path, dtype=np.float32).reshape(n_items, 2560)
    return features


def batches(histories, targets, batch_size=64, indices=None):
    indices = np.arange(len(targets)) if indices is None else indices
    for start in range(0, len(indices), batch_size):
        ix = indices[start:start + batch_size]
        selected = [histories[i] for i in ix]
        ids = torch.zeros(len(ix), 50, dtype=torch.long)
        for row, seq in enumerate(selected):
            ids[row, :len(seq)] = torch.tensor(seq)
        yield ix, {"input_ids": ids, "attention_mask": ids > 0,
                   "seq_lens": (ids > 0).sum(1),
                   "labels": torch.tensor(targets[ix])[:, None]}


def per_user(predictions, targets):
    predictions = np.asarray(predictions)
    hit = predictions == np.asarray(targets)[:, None]
    output = {}
    for k in (5, 10):
        output[f"recall@{k}"] = hit[:, :k].any(1).astype(float)
        output[f"ndcg@{k}"] = (hit[:, :k] / np.log2(np.arange(k) + 2)).sum(1)
    return output


def summarize(predictions, targets):
    return {key: float(values.mean()) for key, values in per_user(predictions, targets).items()}


def paired_interval(first, second, seed=SEED, draws=2000):
    delta = np.asarray(first, dtype=float) - np.asarray(second, dtype=float)
    rng = np.random.default_rng(seed)
    means = [delta[rng.integers(len(delta), size=len(delta))].mean() for _ in range(draws)]
    return {"difference": float(delta.mean()),
            "bootstrap_95pct": np.quantile(means, [.025, .975]).tolist()}


def load_joint(path, device):
    from joint_training.checkpoint import read_checkpoint
    from joint_training.config import JointConfig
    from joint_training.model import JointModel
    from misc.artifacts import array_hash, json_hash

    checkpoint = read_checkpoint(path)
    defaults = JointConfig()
    clean = {section: {key: value for key, value in values.items()
                        if key in {f.name for f in fields(getattr(defaults, section))}}
             for section, values in checkpoint["config"].items() if hasattr(defaults, section)}
    config = JointConfig.from_dict(clean)
    directory, mapping, train, histories, targets = corpus(config.data.category)
    raw = raw_features(directory, len(mapping["id2item"]) - 1)
    mask = np.zeros(len(raw), dtype=bool)
    mask[np.array(sorted({i for seq in train for i in seq})) - 1] = True
    expected = checkpoint["source"]
    assert expected["embeddings_sha256"] == array_hash(raw), "Embedding source mismatch"
    assert expected["item_order_sha256"] == json_hash(mapping["id2item"]), "Item order mismatch"
    assert expected["train_mask_sha256"] == array_hash(mask), "Training mask mismatch"
    model = JointModel(config, raw).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    model.refresh_catalog()
    return model, checkpoint, histories, targets
