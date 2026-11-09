"""Paired seed summaries and fixed-seed diversity audits; never choose epochs."""

import argparse
import hashlib
import json

import numpy as np

from misc.artifacts import json_hash, write_json
from protocol import ROOT, ranking_metrics


BASE = ROOT / "outputs/item_cluster_diversity"
HISTORICAL = {
    "next_k1_s2024": ROOT / "outputs/item_cluster_followup/runs/Beauty/beauty_scaled_k1_s2024",
    "residual_k4_linear_s2024": ROOT / "outputs/item_cluster_simplification/runs/Beauty/beauty_residual_k4_linear_s2024",
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit_run(directory, *, probe_directory):
    result, exp, cfg, hist, hashes = [json.loads((directory / f).read_text()) for f in
        ["results.json", "experiment.json", "config.json", "metrics.json", "source_hashes.json"]]
    assert result["benchmark_eligible"] and exp["benchmark_eligible"] and not result["test_accessed"]
    assert result["checkpoint_reload_verified"] and not exp["has_fm_module"]
    assert not cfg["training"]["evaluate_test"] and cfg["data"]["category"] == "Beauty"
    assert exp["seed"] == cfg["training"]["seed"] and exp["seed"] in {2024, 2025, 2026}
    assert result["steps"] == 4040 and result["completed_epochs"] == len(hist) == 40
    assert cfg["recommender"]["n_layer"] == 2 and cfg["recommender"]["train_batch_size"] == 256
    assert cfg["recommender"]["n_embd"] == 128 and cfg["ae"]["latent_dim"] == 128
    assert cfg["flow_ranking"]["path"] == "direct_ce" and exp["selection_users"] == 4096
    assert json_hash(hashes) == exp["source_snapshot_sha256"]
    assert all(sha(directory / "source_snapshot" / p) == h for p, h in hashes.items())
    for file, key, count in [("selection_flow.npz", "validation_selection", 4096), ("full_flow.npz", "validation", 22363)]:
        with np.load(directory / file) as d:
            assert len(d["targets"]) == count
            for k, v in ranking_metrics(d["predictions"], d["targets"]).items():
                assert abs(v - result[key][k]) < 1e-12
    points = [(h["validation_selection"]["ndcg@10"], h["epoch"]) for h in hist if "validation_selection" in h]
    score = max(v for v, _ in points)
    assert result["best"] == {"epoch": next(e for v, e in points if v == score), "score": score}
    p = json.loads((probe_directory / "followup_probe.json").read_text())
    assert p["initial_state_reproduced"] and p["reload_predictions_verified"] and not p["test_accessed"]
    assert json_hash(p["initial_parameter_hashes"]) == exp["initial_state_sha256"]
    assert sha(directory / "best_flow.pt") == p["checkpoint_sha256"]
    assert sha(probe_directory / "followup_probe_source.py") == p["probe_script_sha256"]
    for stage in ["initial", "trained"]:
        assert abs(p[stage]["hstu_input_token_norm_mean"] - 128**.5) < 1e-4
    runtime = {n: h for n, h in hashes.items() if n.startswith(("models/", "compression/", "joint_training/", "misc/"))
               or n in {"run.py", "protocol.py"}}
    return {**result, "experiment": exp, "config": cfg, "history": hist, "probe": p,
            "runtime_hashes": runtime, "directory": str(directory)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--methods", nargs="*", default=[])
    parser.add_argument("--name", required=True)
    args = parser.parse_args()
    output = BASE / f"{args.name}.json"
    if output.exists(): raise FileExistsError(output)
    records = {name: audit_run(path, probe_directory=BASE / "reference_probes" / name)
               for name, path in HISTORICAL.items()}
    ref = records["next_k1_s2024"]
    for name in args.methods:
        r = audit_run(BASE / "runs/Beauty" / f"beauty_{name}", probe_directory=BASE / "probes" / name)
        cfg = r["config"]
        for key in ["source", "selection_indices_sha256", "data"]:
            assert r["experiment"][key] == ref["experiment"][key], (name, key)
        for section, keys in {
            "ae": ["lr", "hidden_dims", "latent_dim", "cosine_weight", "variance_weight", "correlation_weight"],
            "recommender": ["lr", "n_embd", "n_layer", "n_head", "train_batch_size", "weight_decay"],
            "flow_ranking": ["path", "weight", "local_weight"],
            "evaluation": ["eval_interval", "val_metric"]}.items():
            for key in keys: assert cfg[section][key] == ref["config"][section][key]
        prdir = BASE / "preflight/runs/Beauty" / f"beauty_{name}_preflight"
        pr, pe, ph = [json.loads((prdir / f).read_text()) for f in
                      ["results.json", "experiment.json", "source_hashes.json"]]
        assert pr["steps"] == 4 and pr["checkpoint_reload_verified"] and not pr["test_accessed"]
        assert not pe["model_reduced"] and not pr["benchmark_eligible"]
        for key in ["initial_state_sha256", "shared_initial_state_sha256", "source", "data"]:
            assert pe[key] == r["experiment"][key], (name, key)
        assert json_hash(ph) == pe["source_snapshot_sha256"]
        assert all(sha(prdir / "source_snapshot" / p) == h for p, h in ph.items())
        assert all(ph[p] == h for p, h in r["runtime_hashes"].items())
        pm = json.loads((prdir / "metrics.json").read_text())[0]["train"]
        assert pm["ae_joint_grad_norm"] > 0 and pm["item_vlad_ce_grad_norm"] > 0
        if cfg["recommender"]["item_vlad_mapping"] == "linear": assert pm["item_mapping_ce_grad_norm"] > 0
        r["preflight_verified"] = True
        records[name] = r
    seed_summary = {}
    complete_seeds = all(f"{method}_s{seed}" in records for method in ["next_k1", "residual_k4_linear"] for seed in [2024, 2025, 2026])
    if complete_seeds:
        for method in ["next_k1", "residual_k4_linear"]:
            seed_summary[method] = {"seeds": [2024, 2025, 2026], "statistics": {}}
            for scope in ["validation_selection", "validation"]:
                seed_summary[method]["statistics"][scope] = {}
                for metric in ["ndcg@10", "recall@10", "nll"]:
                    values = [records[f"{method}_s{s}"][scope][metric] for s in [2024, 2025, 2026]]
                    seed_summary[method]["statistics"][scope][metric] = {
                        "values": values, "mean": float(np.mean(values)), "sample_std": float(np.std(values, ddof=1))}
        paired = {}
        for scope in ["validation_selection", "validation"]:
            a = np.array(seed_summary["next_k1"]["statistics"][scope]["ndcg@10"]["values"])
            b = np.array(seed_summary["residual_k4_linear"]["statistics"][scope]["ndcg@10"]["values"])
            for seed in [2024, 2025, 2026]:
                assert records[f"next_k1_s{seed}"]["experiment"]["shared_initial_state_sha256"] == records[f"residual_k4_linear_s{seed}"]["experiment"]["shared_initial_state_sha256"]
            paired[scope] = {"difference_k4_minus_k1": (b-a).tolist(), "mean_difference": float((b-a).mean()),
                "sample_std_difference": float((b-a).std(ddof=1)), "k4_wins": int((b>a).sum()),
                "ratio_of_means_gain_percent": float(100*(b.mean()/a.mean()-1))}
        seed_summary["paired"] = paired
    comparisons = {}
    for name in args.methods:
        r = records[name]
        if r["config"]["training"]["seed"] != 2024: continue
        assert r["experiment"]["shared_initial_state_sha256"] == ref["experiment"]["shared_initial_state_sha256"]
        comparisons[name] = {scope: 100*(r[scope]["ndcg@10"]/records["residual_k4_linear_s2024"][scope]["ndcg@10"]-1)
                             for scope in ["validation_selection", "validation"]}
    if "orth_k4_linear_s2024" in records:
        a, b = records["orth_k4_linear_s2024"], records["residual_k4_linear_s2024"]
        assert a["experiment"]["initial_state_sha256"] == b["experiment"]["initial_state_sha256"]
        assert a["probe"]["trained"]["token_orthogonality"] >= 0.
        first = a["history"][0]["train"]
        assert "loss/item_orthogonality" in first
    if all(n in records for n in ["sequential_k4_linear_s2024", "parallel_k4_linear_s2024"]):
        a, b = [records[n] for n in ["sequential_k4_linear_s2024", "parallel_k4_linear_s2024"]]
        assert a["experiment"]["initial_state_sha256"] == b["experiment"]["initial_state_sha256"]
        assert a["runtime_hashes"] == b["runtime_hashes"]
        comparisons["sequential_vs_parallel"] = {scope: 100*(a[scope]["ndcg@10"]/b[scope]["ndcg@10"]-1)
                                                for scope in ["validation_selection", "validation"]}
    old = json.loads((ROOT / "docs/item_cluster_diversity/previous_artifacts.json").read_text())
    assert all((ROOT / p).is_file() and sha(ROOT / p) == h for p, h in old.items())
    archive = ROOT / "history/before_item_cluster_diversity"
    ah = json.loads((archive / "source_hashes.json").read_text())
    assert all(sha(archive / p) == h for p, h in ah.items())
    protected = json.loads((ROOT / "docs/item_cluster_diversity/protected_versions.json").read_text())
    changes = {p: {"before": h, "after": sha(ROOT.parent / p) if (ROOT.parent / p).exists() else None}
               for p, h in protected.items() if not (ROOT.parent / p).exists() or sha(ROOT.parent / p) != h}
    audit = {"all_sources_initializations_data_selection_reload_metrics_verified": True,
        "all_new_full_width_preflights_verified": True, "paired_seed_backbone_initializations_match": complete_seeds,
        "old_outputs_unchanged": True, "old_output_count": len(old), "archive_files_verified": len(ah),
        "protected_other_files": len(protected), "observed_concurrent_changes": changes,
        "all_other_versions_unchanged": not changes}
    write_json(output, {"runs": records, "seed_summary": seed_summary, "comparisons": comparisons,
        "audit": audit, "test_accessed": False,
        "note": "Three paired seeds for two requested candidates only; diversity candidates remain single-seed exploration. No multi-seed no-VLAD baseline claim."})
    for name, r in records.items():
        print(json.dumps({"method": name, "selection": r["validation_selection"]["ndcg@10"],
            "full": r["validation"]["ndcg@10"], "epoch": r["best"]["epoch"]}), flush=True)
    print(json.dumps({"seed_summary": seed_summary, "comparisons": comparisons, "audit": audit}), flush=True)


if __name__ == "__main__": main()
