"""Score the already selected frozen checkpoints on the leave-one-out test item.

Checkpoint choice stays the validation selection. This script does not train,
retune, or rewrite results.json.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .audit_unified import load
from .common import ROOT, SEED, corpus, heldout_examples, item_sequences, paired_interval, per_user, write_json
from .unified import evaluate


FROZEN = [
    (category, name, seed)
    for category in ("Beauty", "Sports_and_Outdoors")
    for name in ("frozen_flow", "frozen_direct")
    for seed in (2024, 2025, 2026)
]


def test_examples(category):
    _, _, ids = item_sequences(category)
    _, _, _, histories, targets = corpus(category)
    check_histories, check_targets = heldout_examples(ids, "validation")
    if check_histories != histories or not np.array_equal(check_targets, targets):
        raise RuntimeError("Validation reconstruction does not match corpus().")
    test_histories, test_targets = heldout_examples(ids, "test")
    if len(test_histories) != len(histories) or not all(history and history[-1] == int(target)
                                                       for history, target in zip(test_histories, targets)):
        raise RuntimeError("Test history does not end at the validation target.")
    if np.array_equal(test_targets, targets):
        raise RuntimeError("Test targets match validation targets.")
    return test_histories, test_targets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--base", type=Path, default=ROOT / "outputs/unified_20261028")
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("high")
    rows = []
    by_category = {}
    cached = {}
    for category, name, seed in FROZEN:
        directory = args.base / category / f"{name}_s{seed}_e40"
        if category not in cached:
            cached[category] = test_examples(category)
        histories, targets = cached[category]
        model, result, _, validation_targets = load(directory, args.device)
        if len(validation_targets) != len(targets):
            raise RuntimeError(f"User count mismatch for {directory}.")
        metrics, predictions = evaluate(model, histories, targets, np.arange(len(targets)))
        np.savez_compressed(directory / "test_flow.npz", targets=targets, predictions=predictions)
        record = {
            "category": category,
            "run": directory.name,
            "seed": seed,
            "head": name,
            "checkpoint_epoch": result["best"]["epoch"],
            "validation_ndcg@10": result["audits"]["flow"]["ndcg@10"],
            "test": metrics,
            "users": int(len(targets)),
            "protocol": "leave-one-out final item; history is the preceding 50 items; same generate() as the validation audit",
            "eval_seed": SEED,
            "selected_on": "validation",
        }
        write_json(directory / "test_metrics.json", record)
        rows.append(record)
        by_category.setdefault(category, {}).setdefault(name, []).append(
            (seed, metrics, per_user(predictions, targets)["ndcg@10"]))
        print(json.dumps({"run": f"{category}/{directory.name}", "test": metrics}, ensure_ascii=False), flush=True)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    summary = {"split": "test", "selected_on": "validation", "eval_seed": SEED,
               "runs": rows, "comparisons": {}}
    for category, groups in by_category.items():
        comparison = {}
        user_means = {}
        for name, values in groups.items():
            values.sort(key=lambda item: item[0])
            scores = [item[1]["ndcg@10"] for item in values]
            comparison[name] = {
                "seeds": [item[0] for item in values],
                "ndcg@10": scores,
                "recall@10": [item[1]["recall@10"] for item in values],
                "recall@5": [item[1]["recall@5"] for item in values],
                "ndcg@5": [item[1]["ndcg@5"] for item in values],
                "mean_ndcg@10": float(np.mean(scores)),
                "sample_std_ndcg@10": float(np.std(scores, ddof=1)),
                "metrics_mean": {key: float(np.mean([item[1][key] for item in values])) for key in values[0][1]},
            }
            user_means[name] = np.stack([item[2] for item in values]).mean(0)
        comparison["flow_minus_direct"] = paired_interval(user_means["frozen_flow"], user_means["frozen_direct"])
        summary["comparisons"][category] = comparison
    write_json(args.base / "test_summary.json", summary)
    print(json.dumps(summary["comparisons"], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
