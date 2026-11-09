"""Fixed-alpha memory complement diagnostic; no training or coefficient search."""

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
import torch

from joint_training.checkpoint import read_checkpoint
from joint_training.config import JointConfig
from joint_training.model import JointModel
from misc.artifacts import array_hash, json_hash
from .common import ROOT, batches, corpus, raw_features, per_user, paired_interval, summarize, write_json


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", default="Beauty")
    parser.add_argument("--device", default="cuda:3")
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("high")
    source_dir, mapping, train, histories, targets = corpus(args.category)
    raw = raw_features(source_dir, len(mapping["id2item"]) - 1)
    counts = Counter((a, b) for seq in train for a, b in zip(seq[:-1], seq[1:]))
    by_item = {}
    for (a, b), count in counts.items():
        by_item.setdefault(a, []).append((b, count))
    root = ROOT / "outputs/iterative_20261028"
    runs = []
    for seed in (2024, 2025, 2026):
        directory = root / args.category / f"r1_ce_only_s{seed}_e40"
        result = json.loads((directory / "results.json").read_text())
        assert result["source"]["raw_sha256"] == array_hash(raw)
        assert result["source"]["train_sha256"] == json_hash(train)
        assert result["source"]["item_order_sha256"] == json_hash(mapping["id2item"])
        model = JointModel(JointConfig.from_dict(result["config"]), raw).to(args.device)
        model.load_state_dict(read_checkpoint(directory / "best_ranking.pt")["model"], strict=True)
        model.cache_features()
        model.eval()
        model.refresh_catalog()
        unmixed, mixed = [], []
        for indices, batch in batches(histories, targets, 64):
            batch = {key: value.to(model.device) for key, value in batch.items()}
            _, condition = model.encode_history(batch, ae_grad=False)
            probabilities = model.ranking_logits(condition, model.catalog[1:]).softmax(-1)
            transitions = torch.zeros_like(probabilities)
            for row, user in enumerate(indices):
                values = by_item.get(histories[user][-1], [])
                if values:
                    ids, weights = zip(*values)
                    transitions[row, torch.tensor(ids, device=model.device) - 1] = torch.tensor(weights, device=model.device, dtype=torch.float32)
            transitions /= transitions.sum(-1, keepdim=True).clamp_min(1)
            unmixed.append((probabilities.topk(10).indices + 1).cpu().numpy())
            mixed.append(((probabilities + transitions).topk(10).indices + 1).cpu().numpy())
        unmixed, mixed = np.concatenate(unmixed), np.concatenate(mixed)
        reference = np.load(directory / "full_selected_ranking_decoded_ranking.npz")
        np.testing.assert_array_equal(targets, reference["targets"])
        np.testing.assert_array_equal(unmixed, reference["predictions"])
        selection = np.load(directory / "selection_ranking.npz")["indices"]
        complement = np.ones(len(targets), dtype=bool)
        complement[selection] = False
        np.savez_compressed(directory / "memory_complement_alpha05.npz", targets=targets, predictions=mixed)
        runs.append({"seed": seed, "mixed": summarize(mixed, targets), "unmixed": summarize(unmixed, targets),
                     "validation_complement": summarize(mixed[complement], targets[complement]),
                     "mixed_minus_ce": paired_interval(per_user(mixed, targets)["ndcg@10"],
                                                       per_user(unmixed, targets)["ndcg@10"])})
    write_json(root / f"memory_complement_{args.category}.json", {
        "split": "validation", "test_accessed": False, "alpha": .5, "runs": runs,
        "alpha_origin": "Fixed .5 from the preceding v4 Beauty study; no new coefficient search.",
        "note": "This is an inference-only complement diagnostic, not a new generator or claimed research contribution. Zero-support transition rows leave the CE ranking unchanged."})
    print(json.dumps(runs, indent=2))


if __name__ == "__main__":
    main()
