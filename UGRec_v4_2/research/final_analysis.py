"""Paired seed-matched validation analysis for the frozen transfer candidate."""

import ast
from collections import Counter, defaultdict
import gzip
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .common import ROOT, corpus, paired_interval, per_user, write_json

OUT = ROOT / "outputs/iterative_20261028"


def temporal_masks(directory, category, mapping, histories, targets):
    by_user = defaultdict(list)
    raw = directory.parent / "raw" / f"reviews_{category}_5.json.gz"
    with gzip.open(raw, "rt") as stream:
        for line in stream:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                row = ast.literal_eval(line)
            by_user[row["reviewerID"]].append((int(row["unixReviewTime"]), row["asin"]))
    users = list(json.loads((directory / "all_item_seqs.json").read_text()))
    gaps = []
    for user, history, target in zip(users, histories, targets):
        records = sorted(by_user[user], key=lambda x: x[0])
        assert mapping["item2id"][records[-2][1]] == target
        assert mapping["item2id"][records[-3][1]] == history[-1]
        # The final test pair is never queried or scored.
        gaps.append(records[-2][0] - records[-3][0])
    gaps = np.asarray(gaps)
    return {"same_timestamp": gaps == 0, "different_timestamp": gaps > 0}


def load_prediction(directory, head, chosen=None):
    data = np.load(directory / f"full_selected_{chosen or head}_decoded_{head}.npz")
    return data["targets"], data["predictions"]


def main():
    manifest = json.loads((OUT / "frozen_candidate.json").read_text())
    name = manifest["name"]
    output = {"candidate": manifest, "split": "validation", "test_accessed": False,
              "note": "Paired intervals resample users after averaging fitted seeds; they exclude fitting/selection uncertainty.",
              "categories": {}, "ablations": {}, "profiles": {}, "marginal_exploration": {}}
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for axis, category in zip(axes, ("Beauty", "Sports_and_Outdoors")):
        directory, mapping, train, histories, targets = corpus(category)
        pairs = Counter((a, b) for seq in train for a, b in zip(seq[:-1], seq[1:]))
        frequency = Counter(i for seq in train for i in seq)
        seen = np.array([pairs[(h[-1], int(t))] > 0 for h, t in zip(histories, targets)])
        counts = np.array([frequency[int(t)] for t in targets])
        masks = {"seen_transition": seen, "unseen_transition": ~seen,
                 "target_count_0_to_5": counts <= 5,
                 "target_count_6_to_20": (counts > 5) & (counts <= 20),
                 "target_count_gt20": counts > 20}
        masks.update(temporal_masks(directory, category, mapping, histories, targets))
        models = {"flow": [], "ranking": [], "ce_reference": []}
        predictions, details = {key: [] for key in models}, []
        for seed in (2024, 2025, 2026):
            candidate = OUT / category / f"{name}_s{seed}_e40"
            ce = OUT / category / f"r1_ce_only_s{seed}_e40"
            result = json.loads((candidate / "results.json").read_text())
            for key, path, head in (("flow", candidate, "flow"), ("ranking", candidate, "ranking"),
                                    ("ce_reference", ce, "ranking")):
                truth, values = load_prediction(path, head)
                np.testing.assert_array_equal(targets, truth)
                models[key].append(per_user(values, targets))
                predictions[key].append(values)
            details.append({"seed": seed, "best": result["best"], "source": result["source"]})
            truth, values = load_prediction(candidate, "ranking", chosen="flow")
            np.testing.assert_array_equal(targets, truth)
            models.setdefault("ranking_at_flow_checkpoint", []).append(per_user(values, targets))
            prior_path = candidate / "full_selected_flow_decoded_prior_control.npz"
            if prior_path.exists():
                prior = np.load(prior_path)
                models.setdefault("same_noise_prior", []).append(per_user(prior["predictions"], targets))
        markov_path = ROOT.parent / "UGRec_v4/outputs/research_20261028/nonparametric" / f"{category}_markov_last_predictions.npz"
        markov = np.load(markov_path)
        np.testing.assert_array_equal(targets, markov["targets"])
        models["markov"] = [per_user(markov["predictions"], targets)]
        # Contextual ID reference from the earlier compact study, not an FMRec
        # reproduction or a parameter-count-matched module ablation.
        for seed in (2024, 2025, 2026):
            id_path = (ROOT.parent / "UGRec_v4/outputs/research_20261028/compact" /
                       f"{category}_id_transformer_ce_s{seed}_e40/full_validation_predictions.npz")
            if id_path.exists():
                data = np.load(id_path)
                np.testing.assert_array_equal(targets, data["targets"])
                models.setdefault("earlier_compact_id_ce", []).append(per_user(data["predictions"], targets))
        means = {key: {metric: np.mean([run[metric] for run in runs], axis=0)
                       for metric in runs[0]} for key, runs in models.items()}
        selected = np.load(OUT / category / f"{name}_s2024_e40/selection_flow.npz")["indices"]
        complement = np.ones(len(targets), dtype=bool)
        complement[selected] = False
        masks["validation_complement"] = complement
        record = {"users": len(targets), "seeds": details, "models": {}, "paired": {}}
        for key, values in means.items():
            seeds = [float(run["ndcg@10"].mean()) for run in models[key]]
            record["models"][key] = {
                "metrics_mean": {metric: float(v.mean()) for metric, v in values.items()},
                "ndcg10_per_seed": seeds,
                "ndcg10_seed_sample_std": float(np.std(seeds, ddof=1)) if len(seeds) > 1 else None,
                "strata": {label: {"users": int(mask.sum()),
                                   **{metric: float(v[mask].mean()) for metric, v in values.items()}}
                           for label, mask in masks.items()}}
            if key in predictions:
                record["models"][key]["coverage_top10_mean"] = float(np.mean([len(np.unique(v)) for v in predictions[key]])) / (len(mapping["id2item"]) - 1)
        for reference in ("ce_reference", "ranking", "ranking_at_flow_checkpoint", "markov",
                          "same_noise_prior", "earlier_compact_id_ce"):
            if reference not in means:
                continue
            record["paired"][f"flow_minus_{reference}"] = paired_interval(means["flow"]["ndcg@10"], means[reference]["ndcg@10"])
            record["paired"][f"flow_minus_{reference}_complement"] = paired_interval(
                means["flow"]["ndcg@10"][complement], means[reference]["ndcg@10"][complement])
        output["categories"][category] = record
        labels = ("Markov", "Content AE + CE", "Trajectory + RF")
        keys = ("markov", "ce_reference", "flow")
        heights = [record["models"][key]["metrics_mean"]["ndcg@10"] for key in keys]
        error = [record["models"][key]["ndcg10_seed_sample_std"] or 0 for key in keys]
        bars = axis.bar(labels, heights, yerr=error, capsize=4, color=("#999999", "#4c78a8", "#f58518"))
        for bar, height in zip(bars, heights):
            axis.text(bar.get_x() + bar.get_width() / 2, height + .002, f"{height:.4f}", ha="center", fontsize=10)
        axis.set_title(category.replace("_and_Outdoors", ""))
        axis.set_ylabel("Full validation NDCG@10")
        axis.set_ylim(0, max(heights) * 1.25)
        axis.grid(axis="y", alpha=.2)
        axis.set_axisbelow(True)
    fig.suptitle("Frozen configuration; neural models use 3 seeds (error bars: sample SD)", fontsize=11)
    fig.savefig(OUT / "validation_comparison.png", dpi=180)
    plt.close(fig)
    for label, run in manifest.get("ablations", {}).items():
        path = OUT / "Beauty" / f"{run}_s2024_e40"
        result = json.loads((path / "results.json").read_text())
        record = {"run": run, "audits": result["audits"], "best": result["best"]}
        truth, values = load_prediction(path, "flow")
        reference_truth, reference = load_prediction(OUT / "Beauty" / f"{name}_s2024_e40", "flow")
        np.testing.assert_array_equal(truth, reference_truth)
        record["minus_main_flow"] = paired_interval(per_user(values, truth)["ndcg@10"],
                                                       per_user(reference, truth)["ndcg@10"])
        output["ablations"][label] = record
    for label, run in manifest.get("profiles", {}).items():
        path = OUT / "Beauty" / f"{run}_s2024_e40"
        result = json.loads((path / "results.json").read_text())
        record = {"run": run, "audits": result["audits"], "best": result["best"], "seeds": [2024]}
        if label == "hstu":
            truth, values = load_prediction(path, "flow")
            ref_path = OUT / "Beauty" / f"{manifest['matched_hstu_ce_reference']}_s2024_e40"
            reference_truth, reference = load_prediction(ref_path, "ranking")
            np.testing.assert_array_equal(truth, reference_truth)
            record["matched_ce_ndcg10"] = float(per_user(reference, truth)["ndcg@10"].mean())
            record["flow_minus_matched_hstu_ce"] = paired_interval(per_user(values, truth)["ndcg@10"],
                                                                   per_user(reference, truth)["ndcg@10"])
        output["profiles"][label] = record
    sample_reference = OUT / "Beauty/r7_sample4_s2024_e40"
    for path in sorted((OUT / "Beauty").glob("r7*")):
        if not (path / "full_selected_flow_decoded_flow.npz").exists():
            continue
        result = json.loads((path / "results.json").read_text())
        record = {"audits": result["audits"], "best": result["best"], "config": result["config"]["ranking"]}
        truth, values = load_prediction(path, "flow")
        for label, reference_path in (("sample4", sample_reference), ("main_sample1", OUT / "Beauty" / f"{name}_s2024_e40")):
            if (reference_path / "full_selected_flow_decoded_flow.npz").exists():
                reference_truth, reference = load_prediction(reference_path, "flow")
                np.testing.assert_array_equal(truth, reference_truth)
                record[f"minus_{label}"] = paired_interval(per_user(values, truth)["ndcg@10"],
                                                            per_user(reference, truth)["ndcg@10"])
        probe = path / "generation_probe_flow.json"
        if probe.exists():
            record["generation_probe"] = json.loads(probe.read_text())
        output["marginal_exploration"][path.name] = record
    write_json(OUT / "paired_final_analysis.json", output)
    print(json.dumps({k: {m: v["metrics_mean"]["ndcg@10"] for m, v in r["models"].items()}
                      for k, r in output["categories"].items()}, indent=2))


if __name__ == "__main__":
    main()
