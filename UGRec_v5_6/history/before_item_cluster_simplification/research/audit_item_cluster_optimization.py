"""Audit fixed-reference scale-matched parameter and item-pooling experiments."""

import argparse
import hashlib
import json

from misc.artifacts import json_hash, write_json
from protocol import ROOT
from research.audit_item_cluster_followup import audit_run


BASE = ROOT / "outputs/item_cluster_optimization"
REFERENCES = {
    "baseline": ROOT / "outputs/item_clusters/runs/Beauty/beauty_item_vlad_baseline_s2024",
    "k1_g2": ROOT / "outputs/item_cluster_followup/runs/Beauty/beauty_scaled_k1_s2024",
    "k4_g2": ROOT / "outputs/item_cluster_followup/runs/Beauty/beauty_scaled_k4_s2024",
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_probe(record, directory):
    p = json.loads((directory / "followup_probe.json").read_text())
    assert p["initial_state_reproduced"] and p["reload_predictions_verified"] and not p["test_accessed"]
    assert sha(directory / "followup_probe_source.py") == p["probe_script_sha256"]
    assert sha(record["path"] / "best_flow.pt") == p["checkpoint_sha256"]
    assert json_hash(p["initial_parameter_hashes"]) == record["experiment"]["initial_state_sha256"]
    multiplier = record["config"]["recommender"].get("item_vlad_scale_multiplier", 1.)
    assert abs(p["trained"]["hstu_input_token_norm_mean"] - multiplier * 128 ** .5) < 1e-4
    return p


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--methods", required=True, nargs="+")
    parser.add_argument("--name", required=True)
    args = parser.parse_args()
    output = BASE / f"{args.name}.json"
    if output.exists(): raise FileExistsError(output)
    refs = {name: {**audit_run(path, probe=False), "path": path} for name, path in REFERENCES.items()}
    for name in ["k1_g2", "k4_g2"]:
        refs[name]["probe"] = verify_probe(refs[name], BASE / "reference_probes" / name)
    branch_directory = BASE / "reference_probes/k1_g2_branch"
    if branch_directory.exists():
        refs["k1_g2"]["residual_balance_probe"] = verify_probe(refs["k1_g2"], branch_directory)
    reference = refs["k1_g2"]
    records = {}
    for name in args.methods:
        directory = BASE / "runs/Beauty" / f"beauty_{name}_s2024"
        r = {**audit_run(directory), "path": directory}
        r["probe"] = verify_probe(r, directory)
        cfg = r["config"]
        assert cfg["ae"].get("cluster_count", 0) == 0
        assert cfg["recommender"]["item_vlad_scale"] == "sqrt_dim"
        for key in ["shared_initial_state_sha256", "source", "data", "selection_indices_sha256"]:
            assert r["experiment"][key] == reference["experiment"][key], (name, key)
        if name.startswith("k1_g2_scale_"):
            assert r["experiment"]["initial_state_sha256"] == reference["experiment"]["initial_state_sha256"]
            assert r["probe"]["initial_parameter_hashes"] == reference["probe"]["initial_parameter_hashes"]
        preflight = BASE / "preflight/runs/Beauty" / f"beauty_{name}_s2024_preflight"
        pr, pe, pc, ph = [json.loads((preflight / f).read_text()) for f in
                         ["results.json", "experiment.json", "config.json", "source_hashes.json"]]
        assert pr["steps"] == 4 and pr["checkpoint_reload_verified"] and not pr["test_accessed"]
        assert not pr["benchmark_eligible"] and not pe["model_reduced"]
        assert pc["recommender"]["train_batch_size"] == 256 and pc["recommender"]["n_embd"] == 128
        for key in ["initial_state_sha256", "shared_initial_state_sha256", "source", "data"]:
            assert pe[key] == r["experiment"][key]
        assert json_hash(ph) == pe["source_snapshot_sha256"]
        for filename, digest in ph.items(): assert sha(preflight / "source_snapshot" / filename) == digest
        assert all(ph[filename] == digest for filename, digest in r["runtime_hashes"].items())
        pm = json.loads((preflight / "metrics.json").read_text())[0]["train"]
        assert pm["ae_joint_grad_norm"] > 0
        mode = cfg["recommender"].get("item_vlad_pooling", "sum")
        assert mode == r["experiment"]["item_vlad"]["aggregation"]
        if mode != "sum":
            assert pm["item_vlad_ce_grad_norm"] > 0
            if mode == "context": assert pm["item_pool_ce_grad_norm"] > 0
        r["preflight_verified"] = {"full_width": True, "steps": 4, "initial_and_runtime_match": True,
                                   "gradients": {k: v for k, v in pm.items() if k.endswith("grad_norm")}}
        for target, target_run in [("positive_reference", reference), ("baseline", refs["baseline"])]:
            r["ndcg10_gain_percent_vs_" + target] = {
                scope: 100 * (r[scope]["ndcg@10"] / target_run[scope]["ndcg@10"] - 1)
                for scope in ["validation_selection", "validation"]}
        records[name] = r
    comparisons = {}
    for name, r in records.items():
        rec = r["config"]["recommender"]
        if rec.get("item_vlad_pooling", "sum") == "sum": continue
        summed = next((s for s in list(records.values()) + list(refs.values())
                       if s["config"]["recommender"].get("item_vlad_pooling", "sum") == "sum"
                       and all(s["config"]["recommender"][k] == rec[k]
                               for k in ["item_vlad_clusters", "item_vlad_groups", "item_vlad_expansion",
                                         "item_vlad_assignment_temperature", "item_vlad_scale"])), None)
        assert summed is not None, (name, "missing matched sum control")
        a, b = r["probe"]["initial_parameter_hashes"], summed["probe"]["initial_parameter_hashes"]
        assert all(a[k] == value for k, value in b.items()), (name, "unmatched common initialization")
        if rec["item_vlad_pooling"] == "mass":
            assert a == b
        else:
            assert r["probe"]["initial"]["pooling"]["query_norm"] == 0.
            assert r["probe"]["initial"]["pooling"]["pooled_vs_sum_relative_l2"] == 0.
        comparisons[name + "_vs_matched_sum"] = {
            scope: 100 * (r[scope]["ndcg@10"] / summed[scope]["ndcg@10"] - 1)
            for scope in ["validation_selection", "validation"]}
    old = json.loads((ROOT / "docs/item_cluster_optimization/previous_artifacts.json").read_text())
    assert all((ROOT / p).is_file() and sha(ROOT / p) == h for p, h in old.items())
    archive = ROOT / "history/before_item_cluster_optimization"
    ah = json.loads((archive / "source_hashes.json").read_text())
    assert all(sha(archive / p) == h for p, h in ah.items())
    protected = json.loads((ROOT / "docs/item_cluster_optimization/protected_versions.json").read_text())
    changes = {p: {"before": h, "after": sha(ROOT.parent / p) if (ROOT.parent / p).exists() else None}
               for p, h in protected.items() if not (ROOT.parent / p).exists() or sha(ROOT.parent / p) != h}
    for r in list(records.values()) + list(refs.values()): r["path"] = str(r["path"])
    audit = {"all_run_and_preflight_snapshots_verified": True, "all_initial_states_reproduced": True,
             "all_shared_backbones_and_data_match": True, "all_reload_predictions_verified": True,
             "all_ranking_metrics_recomputed": True, "all_pooling_common_initial_parameters_match": True,
             "previous_artifacts_unchanged": True, "previous_artifact_count": len(old),
             "archive_verified": True, "archive_files": len(ah), "protected_files": len(protected),
             "observed_protected_changes": changes, "all_other_versions_unchanged": not changes,
             "runtime_note": "Every run is audited against its own snapshot; source can evolve across stages."}
    write_json(output, {"runs": records, "references": refs, "comparisons": comparisons,
                        "audit": audit, "test_accessed": False})
    for name, r in records.items():
        print(json.dumps({"method": name, "selection": r["validation_selection"]["ndcg@10"],
                          "full": r["validation"]["ndcg@10"], "best_epoch": r["best"]["epoch"],
                          "gain_vs_positive_reference": r["ndcg10_gain_percent_vs_positive_reference"]}), flush=True)
    print(json.dumps({"comparisons": comparisons, "audit": audit}), flush=True)


if __name__ == "__main__":
    main()
