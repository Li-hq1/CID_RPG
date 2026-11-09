"""Behavioral-transfer and frequency strata for already audited predictions."""

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from .common import ROOT, corpus, per_user, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directories", type=Path, nargs="+")
    args = parser.parse_args()
    categories = {}
    rows = []
    for directory in args.directories:
        result = json.loads((directory / "results.json").read_text())
        category = result["config"]["data"]["category"]
        if category not in categories:
            _, _, train, histories, targets = corpus(category)
            pairs = Counter((a, b) for seq in train for a, b in zip(seq[:-1], seq[1:]))
            frequency = Counter(i for seq in train for i in seq)
            seen = np.array([pairs[(h[-1], int(t))] > 0 for h, t in zip(histories, targets)])
            counts = np.array([frequency[int(t)] for t in targets])
            categories[category] = (targets, {"seen_transition": seen, "unseen_transition": ~seen,
                "target_count_0_to_5": counts <= 5, "target_count_6_to_20": (counts > 5) & (counts <= 20),
                "target_count_gt20": counts > 20})
        targets, masks = categories[category]
        record = {"run": directory.name, "category": category, "heads": {}}
        for name in result["best"]:
            prediction_path = directory / f"full_selected_{name}_decoded_{name}.npz"
            if not prediction_path.exists():
                continue
            data = np.load(prediction_path)
            np.testing.assert_array_equal(targets, data["targets"])
            metrics = per_user(data["predictions"], targets)
            selected = np.load(directory / f"selection_{name}.npz")["indices"]
            complement = np.ones(len(targets), dtype=bool)
            complement[selected] = False
            record["heads"][name] = {"catalog_coverage_top10": int(len(np.unique(data["predictions"]))),
                "strata": {key: {"users": int(mask.sum()),
                                  "ndcg@10": float(metrics["ndcg@10"][mask].mean()),
                                  "recall@10": float(metrics["recall@10"][mask].mean())}
                           for key, mask in {**masks, "validation_complement": complement}.items()}}
        rows.append(record)
    output = ROOT / "outputs/iterative_20261028/stratified_audits.json"
    write_json(output, {"split": "validation", "runs": rows,
                        "note": "Post-hoc target-based strata are unavailable at inference; unseen transition is not cold item."})
    print(output)


if __name__ == "__main__":
    main()
