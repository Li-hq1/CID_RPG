"""Audit compact center directions and shared mapping against fixed references."""

import argparse
import hashlib
import json

import numpy as np

from misc.artifacts import array_hash, json_hash, write_json
from protocol import ROOT, ranking_metrics
from research.audit_item_cluster_followup import audit_run


BASE = ROOT / "outputs/item_cluster_simplification"
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
    for stage in ["initial", "trained"]:
        assert abs(p[stage]["hstu_input_token_norm_mean"] - 128**.5) < 1e-4
        mapping = p[stage]["token_mapping"]
        assert abs(mapping["mapped_vs_unmapped_relative_l2"]**2
                   - 2*(1-mapping["mapped_vs_unmapped_cosine"])) < 1e-5
    return p


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--methods", required=True, nargs="+")
    parser.add_argument("--name", required=True)
    args = parser.parse_args()
    output = BASE / f"{args.name}.json"
    if output.exists(): raise FileExistsError(output)
    refs = {name: {**audit_run(path, probe=False), "path": path} for name, path in REFERENCES.items()}
    refs["k1_g2"]["probe"] = verify_probe(refs["k1_g2"], BASE / "reference_probes/k1_g2_v2")
    reference = refs["k1_g2"]
    records = {}
    for name in args.methods:
        directory = BASE / "runs/Beauty" / f"beauty_{name}_s2024"
        r = {**audit_run(directory, probe=False), "path": directory}
        r["probe"] = verify_probe(r, BASE / "probes" / name)
        for key in ["shared_initial_state_sha256", "source", "selection_indices_sha256", "data"]:
            assert r["experiment"][key] == reference["experiment"][key], (name, key)
        assert r["config"]["recommender"]["item_vlad_scale"] == "sqrt_dim"
        assert r["config"]["recommender"]["item_vlad_pooling"] == "sum"
        for section, keys in {
            "ae": ["lr", "hidden_dims", "latent_dim", "cosine_weight", "variance_weight", "correlation_weight"],
            "recommender": ["lr", "n_embd", "n_layer", "n_head", "train_batch_size", "weight_decay"],
            "flow_ranking": ["path", "weight", "local_weight"],
            "evaluation": ["eval_interval", "val_metric", "temperature"]}.items():
            for key in keys: assert r["config"][section][key] == reference["config"][section][key]
        scores = [(h["validation_selection"]["ndcg@10"], h["epoch"])
                  for h in r["history"] if "validation_selection" in h]
        best_score = max(v for v, _ in scores)
        best_epoch = next(e for v, e in scores if v == best_score)
        assert r["best"] == {"score": best_score, "epoch": best_epoch}
        prdir = BASE / "preflight/runs/Beauty" / f"beauty_{name}_s2024_preflight"
        pr, pe, pc, ph = [json.loads((prdir / file).read_text()) for file in
                         ["results.json", "experiment.json", "config.json", "source_hashes.json"]]
        assert pr["steps"] == 4 and pr["checkpoint_reload_verified"] and not pr["test_accessed"]
        assert not pe["model_reduced"] and not pr["benchmark_eligible"]
        assert pc["recommender"]["train_batch_size"] == 256 and pc["ae"]["latent_dim"] == 128
        for key in ["initial_state_sha256", "shared_initial_state_sha256", "source", "data"]:
            assert pe[key] == r["experiment"][key], (name, "preflight", key)
        assert json_hash(ph) == pe["source_snapshot_sha256"]
        assert all(sha(prdir / "source_snapshot" / p) == h for p, h in ph.items())
        assert all(ph[p] == h for p, h in r["runtime_hashes"].items())
        pm = json.loads((prdir / "metrics.json").read_text())[0]["train"]
        assert pm["ae_joint_grad_norm"] > 0
        rec = r["config"]["recommender"]
        if rec["item_vlad_variant"] != "normalized_input": assert pm["item_vlad_ce_grad_norm"] > 0
        if rec["item_vlad_mapping"] == "linear":
            assert pm["item_mapping_ce_grad_norm"] > 0
            assert r["probe"]["initial"]["token_mapping"]["weight_relative_distance_from_identity"] == 0.
            assert r["probe"]["trained"]["token_mapping"]["weight_relative_distance_from_identity"] > 0.
        parameters = r["experiment"]["trainable_parameters"] - refs["baseline"]["experiment"]["trainable_parameters"]
        expected = {"norm_k1": 0, "residual_k1": 128, "residual_k1_linear": 16512,
                    "next_k1_linear": 50564, "residual_k4": 512, "residual_k4_linear": 16896}[name]
        assert parameters == expected == r["experiment"]["item_vlad"]["module_parameters"]
        r["preflight_verified"] = {"full_width": True, "steps": 4, "reload": True,
            "initial_state_and_runtime_match_official_run": True,
            "gradients": {k: v for k, v in pm.items() if "grad_norm" in k}}
        r["added_parameters"] = parameters
        for refname in ["baseline", "k1_g2"]:
            r["gains_vs_" + refname] = {scope: 100*(r[scope]["ndcg@10"] / refs[refname][scope]["ndcg@10"]-1)
                                      for scope in ["validation_selection", "validation"]}
        records[name] = r
    comparisons = {}
    pairs = [("residual_k1_linear", "residual_k1"), ("residual_k4_linear", "residual_k4"),
             ("next_k1_linear", "k1_g2"), ("residual_k1", "norm_k1"),
             ("residual_k4", "residual_k1"), ("residual_k4_linear", "residual_k1_linear")]
    all_records = {**refs, **records}
    for r in all_records.values():
        with np.load(r["path"] / "selection_flow.npz") as selected, np.load(r["path"] / "full_flow.npz") as full:
            # run.py writes full predictions in original validation-user order.
            assert np.array_equal(full["targets"][selected["indices"]], selected["targets"])
            kept = ~np.isin(np.arange(len(full["targets"])), selected["indices"])
            assert int(kept.sum()) == 18267
            r["validation_outside_selection"] = ranking_metrics(full["predictions"][kept], full["targets"][kept])
            r["validation_outside_selection_users"] = int(kept.sum())
    for a, b in pairs:
        if a not in all_records or b not in all_records: continue
        ra, rb = all_records[a], all_records[b]
        if a.endswith("linear") and not ("k4" in a and "k1" in b):
            ah, bh = ra["probe"]["initial_parameter_hashes"], rb["probe"]["initial_parameter_hashes"]
            assert all(ah[k] == h for k, h in bh.items()), (a, b, "common initial parameters")
            # CUDA high matmul precision rounds the identity product's input
            # (~2e-4 observed); shared weights/RNG are exact, activations are not.
            assert ra["probe"]["initial"]["token_mapping"]["mapped_vs_unmapped_relative_l2"] < 1e-3
        comparisons[a + "_vs_" + b] = {scope: 100*(ra[scope]["ndcg@10"]/rb[scope]["ndcg@10"]-1)
                                               for scope in ["validation_selection", "validation"]}
    mechanisms = {}
    for name in ["residual_k4", "residual_k4_linear"]:
        if name not in records: continue
        r = records[name]
        directory = BASE / "mechanisms" / name
        p = json.loads((directory / "center_diversity_probe.json").read_text())
        assert p["original_predictions_verified"] and p["parameters_restored_exactly"] and not p["test_accessed"]
        assert sha(r["path"] / "best_flow.pt") == p["checkpoint_sha256"]
        assert sha(directory / "probe_source.py") == p["script_sha256"]
        with np.load(directory / "predictions.npz") as data, np.load(r["path"] / "selection_flow.npz") as saved:
            assert array_hash(data["indices"]) == p["indices_sha256"]
            assert np.array_equal(data["original"], saved["predictions"][:512])
            assert np.array_equal(data["targets"], saved["targets"][:512])
            for key, field in [("original_metrics", "original"), ("collapsed_metrics", "collapsed")]:
                assert ranking_metrics(data[field], data["targets"]) == p[key]
        mechanisms[name] = p
    old = json.loads((ROOT / "docs/item_cluster_simplification/previous_artifacts.json").read_text())
    assert all((ROOT / p).is_file() and sha(ROOT / p) == h for p, h in old.items())
    archive = ROOT / "history/before_item_cluster_simplification"
    ah = json.loads((archive / "source_hashes.json").read_text())
    assert all(sha(archive / p) == h for p, h in ah.items())
    protected = json.loads((ROOT / "docs/item_cluster_simplification/protected_versions.json").read_text())
    changes = {p: {"before": h, "after": sha(ROOT.parent / p) if (ROOT.parent / p).exists() else None}
               for p, h in protected.items() if not (ROOT.parent / p).exists() or sha(ROOT.parent / p) != h}
    for r in list(records.values()) + list(refs.values()): r["path"] = str(r["path"])
    audit = {"all_run_and_preflight_snapshots_verified": True, "all_initial_states_reproduced": True,
             "all_shared_backbones_and_data_match": True, "all_reload_predictions_verified": True,
             "all_ranking_metrics_recomputed": True, "all_mapping_common_initial_parameters_match": True,
             "all_checkpoint_epochs_selected_by_fixed_subset": True,
             "identity_mapping_numerics": "Identity weights and common parameters exact; CUDA high matmul produces ~2e-4 activation drift, bounded below 1e-3. No bitwise activation-equivalence claim.",
             "previous_artifacts_unchanged": True, "previous_artifact_count": len(old),
             "archive_verified": True, "archive_files": len(ah), "protected_files": len(protected),
             "observed_protected_changes": changes, "all_other_versions_unchanged": not changes,
             "runtime_note": "Each run audited against its own snapshot; runtime matches its preflight."}
    write_json(output, {"runs": records, "references": refs, "comparisons": comparisons, "mechanisms": mechanisms,
                        "audit": audit, "test_accessed": False})
    for name, r in records.items():
        print(json.dumps({"method": name, "selection": r["validation_selection"]["ndcg@10"],
            "full": r["validation"]["ndcg@10"], "best_epoch": r["best"]["epoch"],
            "added_parameters": r["added_parameters"], "gains_vs_k1_g2": r["gains_vs_k1_g2"]}), flush=True)
    print(json.dumps({"comparisons": comparisons, "audit": audit}), flush=True)


if __name__ == "__main__": main()
