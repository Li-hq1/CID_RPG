"""Audit train-only transition coverage and validation timestamp ties."""

import argparse
import ast
from collections import Counter, defaultdict
import gzip
import json

import numpy as np

from .common import ROOT, corpus, per_user, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", required=True)
    args = parser.parse_args()
    directory, mapping, train, histories, targets = corpus(args.category)
    counts = Counter((a, b) for seq in train for a, b in zip(seq[:-1], seq[1:]))
    frequency = Counter(i for seq in train for i in seq)
    support = np.array([counts[(h[-1], int(t))] for h, t in zip(histories, targets)])
    target_frequency = np.array([frequency[int(t)] for t in targets])
    by_user = defaultdict(list)
    raw = directory.parent / "raw" / f"reviews_{args.category}_5.json.gz"
    with gzip.open(raw, "rt") as stream:
        for line in stream:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                row = ast.literal_eval(line)
            by_user[row["reviewerID"]].append((int(row["unixReviewTime"]), row["asin"]))
    # Match the production stable timestamp sort. Read the validation pair, never the last test pair.
    ordered_users = list(json.loads((directory / "all_item_seqs.json").read_text()))
    gaps = []
    for user, history, target in zip(ordered_users, histories, targets):
        records = sorted(by_user[user], key=lambda x: x[0])
        assert mapping["item2id"][records[-2][1]] == target
        assert mapping["item2id"][records[-3][1]] == history[-1]
        gaps.append(records[-2][0] - records[-3][0])
    gaps = np.array(gaps)
    directory_out = ROOT / "outputs/research_20261028"
    predictions = {}
    baseline = np.load(directory_out / "nonparametric" / f"{args.category}_markov_last_predictions.npz")
    predictions["markov_last"] = baseline["predictions"]
    for representation in ("pca", "id", "adapter"):
        path = directory_out / "compact" / f"{args.category}_{representation}_transformer_ce_s2024_e40/full_validation_predictions.npz"
        if path.exists():
            predictions[representation] = np.load(path)["predictions"]
    strata = {"seen_transition": support > 0, "unseen_transition": support == 0,
              "same_timestamp": gaps == 0, "different_timestamp": gaps > 0,
              "target_train_count_0_to_5": target_frequency <= 5,
              "target_train_count_6_to_20": (target_frequency > 5) & (target_frequency <= 20),
              "target_train_count_gt20": target_frequency > 20}
    result = {"category": args.category, "split": "validation", "users": len(targets),
              "train_transition_coverage": float((support > 0).mean()),
              "validation_last_pair_same_timestamp_fraction": float((gaps == 0).mean()),
              "warning": "Same-time events have no observed order; an order is induced by input-record order. This is a benchmark property, not proof of data leakage.",
              "strata": {}}
    for name, mask in strata.items():
        result["strata"][name] = {"users": int(mask.sum()),
                                  "fraction": float(mask.mean()),
                                  "scores": {method: {metric: float(value[mask].mean())
                                                       for metric, value in per_user(prediction, targets).items()}
                                             for method, prediction in predictions.items()}}
    write_json(directory_out / "data_audit" / f"{args.category}.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
