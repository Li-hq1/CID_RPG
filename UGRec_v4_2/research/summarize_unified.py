"""Rebuild v4_2 tables and paired validation comparisons from saved artifacts."""

import json
from pathlib import Path

import numpy as np

from .common import ROOT, paired_interval, per_user, write_json


def main():
    base = ROOT / "outputs/unified_20261028"
    records = []
    for path in sorted(base.glob("*/*/results.json")):
        result = json.loads(path.read_text())
        config = result["config"]
        metric = result["audits"].get("flow", {}).get("ndcg@10")
        records.append({"category": config["data"]["category"], "run": path.parent.name,
            "seed": config["training"]["seed"], "epochs": config["training"]["epochs"],
            "best_epoch": result["best"]["epoch"], "selection_ndcg10": result["best"]["score"],
            "full_validation_ndcg10": metric, "settings": config["flow_ranking"],
            "training_seconds": result["seconds_training"],
            "training_seconds_per_epoch": result["seconds_training"] / config["training"]["epochs"],
            "peak_gpu_allocated_mb": result["peak_gpu_allocated_mb"], "directory": str(path.parent)})
    comparisons = {}
    for category in ["Beauty", "Sports_and_Outdoors"]:
        groups, user_scores = {}, {}
        for name in ["frozen_flow", "frozen_direct"]:
            selected = [r for r in records if r["category"] == category and r["run"].startswith(name + "_")]
            if len(selected) != 3 or any(r["full_validation_ndcg10"] is None for r in selected):
                continue
            selected.sort(key=lambda r: r["seed"])
            scores, users = [], []
            reference_targets = None
            for row in selected:
                data = np.load(Path(row["directory"]) / "full_flow.npz")
                if reference_targets is None:
                    reference_targets = data["targets"]
                np.testing.assert_array_equal(reference_targets, data["targets"])
                values = per_user(data["predictions"], data["targets"])["ndcg@10"]
                users.append(values)
                scores.append(values.mean())
            groups[name] = {"seeds": [r["seed"] for r in selected], "scores": scores,
                            "mean": float(np.mean(scores)), "sample_std": float(np.std(scores, ddof=1))}
            user_scores[name] = np.stack(users).mean(0)
        comparisons[category] = groups
        if len(user_scores) == 2:
            comparisons[category]["flow_minus_direct"] = paired_interval(user_scores["frozen_flow"], user_scores["frozen_direct"])
            data = np.load(next(Path(r["directory"]) / "selection_flow.npz" for r in records
                                if r["category"] == category and r["run"].startswith("frozen_flow_")))
            complement = np.ones(len(user_scores["frozen_flow"]), dtype=bool)
            complement[data["indices"]] = False
            comparisons[category]["outside_selection"] = {name: float(values[complement].mean()) for name, values in user_scores.items()}
    summary = {"split": "validation", "test_accessed": False, "completed_training_runs": len(records),
               "records": records, "three_seed_comparisons": comparisons,
               "note": "20-epoch subset screening and 40-epoch full audits are separate. Paired intervals condition on trained models."}
    write_json(base / "summary.json", summary)
    lines = ["# v4_2 实验索引", "", "全部为 validation；20轮筛查与40轮完整审计不可直接比较。", "",
             "| 数据集 | 实验 | epochs | 子集 N@10 | 完整 N@10 |", "|---|---|---:|---:|---:|"]
    for row in records:
        full = f"{row['full_validation_ndcg10']:.6f}" if row['full_validation_ndcg10'] is not None else "—"
        lines.append(f"| {row['category']} | {row['run']} | {row['epochs']} | {row['selection_ndcg10']:.6f} | {full} |")
    (base / "results_table.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"completed": len(records), "comparisons": comparisons}), flush=True)


if __name__ == "__main__":
    main()
