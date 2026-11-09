"""Validation-only mixture of a content ranker and train-only transition counts.

The mixing coefficient is reported on a grid; it is not a held-out test claim.
This tests complementarity, not a new flow-matching algorithm.
"""

import argparse
from collections import Counter
import json

import numpy as np
import torch

from .common import ROOT, SEED, batches, corpus, summarize, write_json
from .compact_ablation import CompactModel


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:4")
    parser.add_argument("--category", default="Beauty")
    args = parser.parse_args()
    torch.set_num_threads(2)
    directory, mapping, train, histories, targets = corpus(args.category)
    root = ROOT / "outputs/research_20261028"
    content = torch.tensor(np.load(root / "prepared" / args.category / "pca128.npy"))
    model = CompactModel(content, "adapter", "transformer", "ce").to(args.device).eval()
    checkpoint = root / "compact" / f"{args.category}_adapter_transformer_ce_s2024_e40/best.pt"
    model.load_state_dict(torch.load(checkpoint, map_location=args.device, weights_only=True))
    counts = Counter((a, b) for seq in train for a, b in zip(seq[:-1], seq[1:]))
    by_item = {}
    for (a, b), value in counts.items():
        by_item.setdefault(a, []).append((b, value))
    alphas = [0., .01, .03, .1, .3, .5, 1.]
    predictions = {str(alpha): [] for alpha in alphas}
    table = model.catalog()
    for ix, batch in batches(histories, targets, 128):
        ids = batch["input_ids"].to(args.device)
        condition = model.encode(ids, table)
        condition = condition[torch.arange(len(ix), device=args.device), batch["seq_lens"].to(args.device) - 1]
        probabilities = model.logits(condition, table).softmax(-1)
        transitions = torch.zeros_like(probabilities)
        for row, user in enumerate(ix):
            values = by_item.get(histories[user][-1], [])
            if values:
                item_ids, weights = zip(*values)
                transitions[row, torch.tensor(item_ids, device=args.device) - 1] = torch.tensor(weights, dtype=torch.float32, device=args.device)
        transitions = transitions / transitions.sum(-1, keepdim=True).clamp_min(1)
        for alpha in alphas:
            scores = (1 - alpha) * probabilities + alpha * transitions
            predictions[str(alpha)].append((scores.topk(10).indices + 1).cpu().numpy())
    selection = np.random.default_rng(SEED).permutation(len(targets))[:4096]
    complement = np.setdiff1d(np.arange(len(targets)), selection)
    predictions = {key: np.concatenate(value) for key, value in predictions.items()}
    scores = {key: {"full_validation": summarize(value, targets),
                    "selection_subset": summarize(value[selection], targets[selection]),
                    "validation_complement": summarize(value[complement], targets[complement])}
              for key, value in predictions.items()}
    best = max(scores, key=lambda key: scores[key]["selection_subset"]["ndcg@10"])
    result = {"checkpoint": str(checkpoint), "split": "validation", "coefficients": scores,
              "selected_alpha_on_4096_users": best,
              "audit_on_remaining_validation": scores[best]["validation_complement"],
              "note": "Exploratory mixture and validation tuning; not test results. Alpha=1 is unsmoothed transitions with arbitrary zero-score ties, unlike the popularity-tied nonparametric baseline."}
    suffix = "" if args.category == "Beauty" else f"_{args.category}"
    write_json(root / f"hybrid_rank{suffix}.json", result)
    np.savez_compressed(root / f"hybrid_rank_predictions{suffix}.npz", targets=targets, **predictions)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
