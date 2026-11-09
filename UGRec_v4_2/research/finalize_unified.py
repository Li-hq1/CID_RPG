"""Post-hoc validation diagnostics; never selects models or accesses test targets."""

from collections import Counter
import json
from pathlib import Path

import numpy as np

from .common import ROOT, corpus, paired_interval, per_user, write_json


def main():
    base = ROOT / "outputs/unified_20261028"
    output = {"split": "validation", "test_accessed": False, "categories": {},
              "note": "Post-hoc target strata are unavailable at inference. User bootstrap conditions on saved models, not training or selection uncertainty."}
    sources = {}
    for category in ["Beauty", "Sports_and_Outdoors"]:
        _, mapping, train, histories, targets = corpus(category)
        pairs = Counter((a, b) for seq in train for a, b in zip(seq[:-1], seq[1:]))
        frequency = Counter(i for seq in train for i in seq)
        seen = np.array([pairs[h[-1], int(y)] > 0 for h, y in zip(histories, targets)])
        count = np.array([frequency[int(y)] for y in targets])
        masks = {"seen_transition": seen, "unseen_transition": ~seen,
                 "target_count_0_to_5": count <= 5,
                 "target_count_6_to_20": (count > 5) & (count <= 20), "target_count_gt20": count > 20}
        groups, rows = {}, {}
        for name in ["frozen_flow", "frozen_direct"]:
            scores, coverage, times, memories, candidates = [], [], [], [], []
            for seed in [2024, 2025, 2026]:
                directory = base / category / f"{name}_s{seed}_e40"
                result = json.loads((directory / "results.json").read_text())
                experiment = json.loads((directory / "experiment.json").read_text())
                sources[str(directory.relative_to(base))] = experiment["code_sha256"]
                assert result["config"]["ranking"]["teacher_weight"] == 0
                assert not result["test_accessed"]
                data = np.load(directory / "full_flow.npz")
                np.testing.assert_array_equal(targets, data["targets"])
                values = per_user(data["predictions"], targets)["ndcg@10"]
                np.testing.assert_allclose(values.mean(), result["audits"]["flow"]["ndcg@10"], atol=1e-12)
                scores.append(values)
                coverage.append(len(np.unique(data["predictions"])))
                times.append(result["seconds_training"] / 40)
                memories.append(result["peak_gpu_allocated_mb"])
                log = json.loads((directory / "metrics.json").read_text())
                candidates.append(float(np.mean([r["sampled"]["candidates"] for r in log])))
            groups[name] = np.stack(scores).mean(0)
            rows[name] = {"mean_catalog_coverage_top10": float(np.mean(coverage)),
                          "mean_training_seconds_per_epoch": float(np.mean(times)),
                          "mean_peak_gpu_allocated_mb": float(np.mean(memories)),
                          "mean_encoded_rank_candidates_per_step": float(np.mean(candidates)),
                          "strata": {key: {"users": int(mask.sum()), "ndcg@10": float(groups[name][mask].mean()),
                                           "seed_scores": [float(values[mask].mean()) for values in scores]}
                                     for key, mask in masks.items()}}
        differences = {key: paired_interval(groups["frozen_flow"][mask], groups["frozen_direct"][mask])
                       for key, mask in masks.items()}
        controls = {}
        directory = base / category / "frozen_flow_s2024_e40"
        reference = per_user(np.load(directory / "full_flow.npz")["predictions"], targets)["ndcg@10"]
        for name in ["zero_noise", "single", "prior", "ranking"]:
            path = directory / f"full_{name}.npz"
            if not path.exists():
                continue
            data = np.load(path)
            np.testing.assert_array_equal(targets, data["targets"])
            values = per_user(data["predictions"], targets)["ndcg@10"]
            controls[name] = {"ndcg@10": float(values.mean()), "flow_minus_control": paired_interval(reference, values)}
        output["categories"][category] = {"users": len(targets), "catalog_size": len(mapping["id2item"]) - 1,
            "groups": rows, "flow_minus_direct_by_stratum": differences,
            "seed2024_controls": controls,
            "timing_note": "Observed research time with shared GPU concurrency; not a controlled speed comparison."}
    first = next(iter(sources.values()))
    # Auxiliary analysis scripts may change while jobs run. The model, trainer,
    # AE, input processing, sampling and inference code must be identical.
    prefixes = ("joint_training/", "models/", "compression/", "data/", "misc/", "engine/")
    experiment_files = {"research/unified.py", "research/common.py", "research/compact_ablation.py"}
    computation = {key: value for key, value in first.items()
                   if key.startswith(prefixes) or key in experiment_files}
    for run, hashes in sources.items():
        assert {key: value for key, value in hashes.items()
                if key.startswith(prefixes) or key in experiment_files} == computation, run
    output["identical_model_and_training_source_across_frozen_runs"] = True
    output["computation_source_sha256"] = computation
    output["legacy_beauty_seed2024"] = {}
    for encoder, old_name, new_name in [
        ("transformer", "v42_legacy_no_teacher_s2024_e40", "s2_history_endpoint_uniform_corrected_s2024_e40"),
        ("hstu", "v42_legacy_hstu_no_teacher_s2024_e40", "frozen_flow_s2024_e40")]:
        old = np.load(ROOT / "outputs/iterative_20261028/Beauty" / old_name / "full_selected_flow_decoded_flow.npz")
        new = np.load(base / "Beauty" / new_name / "full_flow.npz")
        np.testing.assert_array_equal(old["targets"], new["targets"])
        a = per_user(new["predictions"], new["targets"])["ndcg@10"]
        b = per_user(old["predictions"], old["targets"])["ndcg@10"]
        output["legacy_beauty_seed2024"][encoder] = {"old_ndcg@10": float(b.mean()),
            "new_ndcg@10": float(a.mean()), "new_minus_old": paired_interval(a, b)}
    k1024 = np.load(base / "Beauty/frozen_flow_s2024_e40/full_flow.npz")
    k4096 = np.load(base / "Beauty/followup_hstu_k4096_s2024_e40/full_flow.npz")
    np.testing.assert_array_equal(k1024["targets"], k4096["targets"])
    a = per_user(k4096["predictions"], k4096["targets"])["ndcg@10"]
    b = per_user(k1024["predictions"], k1024["targets"])["ndcg@10"]
    output["posthoc_beauty_seed2024_negative_budget"] = {"k1024_ndcg@10": float(b.mean()),
        "k4096_ndcg@10": float(a.mean()), "k4096_minus_k1024": paired_interval(a, b)}
    write_json(base / "final_diagnostics.json", output)
    print(json.dumps({"source_identical": True, "categories": output["categories"]}), flush=True)


if __name__ == "__main__":
    main()
