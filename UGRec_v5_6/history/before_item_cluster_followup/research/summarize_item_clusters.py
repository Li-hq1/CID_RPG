"""Audit the fixed Beauty experiment and summarize token diversity and cost."""

import hashlib
import json
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from misc.artifacts import json_hash, write_json
from protocol import ROOT, ranking_metrics


def main():
    base = ROOT / "outputs/item_clusters"
    summary_path = base / "summary.json"
    if summary_path.exists(): raise FileExistsError(summary_path)
    runs, identities, runtimes = {}, [], []
    for name,k in [("baseline",0),("k1",1),("k4",4)]:
        directory = base / "runs/Beauty" / f"beauty_item_vlad_{name}_s2024"
        read = lambda filename:json.loads((directory / filename).read_text())
        result, experiment, config, history, diag = [read(filename) for filename in
            ["results.json","experiment.json","config.json","metrics.json","item_cluster_diagnostics.json"]]
        assert result["benchmark_eligible"] and result["checkpoint_reload_verified"]
        assert result["completed_epochs"] == 40 and len(history) == 40
        assert result["steps"] == 40 * math.ceil(experiment["data"]["training_windows"] / 256) == 4040
        assert not result["test_accessed"] and not experiment["test_accessed"] and not diag["test_accessed"]
        assert diag["checkpoint_reload_predictions_verified"] and diag["initial_state_reproduced"]
        assert experiment["selection_users"] == 4096 and experiment["seed"] == 2024
        assert config["recommender"]["n_layer"] == 2 and config["recommender"]["lr"] == .001
        assert config["recommender"]["item_vlad_clusters"] == k
        assert config["flow_ranking"]["path"] == "direct_ce" and not experiment["has_fm_module"]
        assert hashlib.sha256((directory / "best_flow.pt").read_bytes()).hexdigest() == diag["checkpoint_sha256"]
        assert hashlib.sha256((directory / "item_cluster_diagnostics_source.py").read_bytes()).hexdigest() == diag["diagnostic_script_sha256"]
        for filename,key,count in [("selection_flow.npz","validation_selection",4096),
                                   ("full_flow.npz","validation",22363)]:
            with np.load(directory / filename) as saved:
                assert len(saved["targets"]) == count
                recalculated = ranking_metrics(saved["predictions"],saved["targets"])
                for metric,value in recalculated.items():
                    assert abs(value-result[key][metric]) < 1e-12
        hashes = read("source_hashes.json")
        assert json_hash(hashes) == experiment["source_snapshot_sha256"]
        for relative,digest in hashes.items():
            assert hashlib.sha256((directory / "source_snapshot" / relative).read_bytes()).hexdigest() == digest
        runtime = {p:h for p,h in hashes.items() if p.startswith(("models/","joint_training/","compression/","misc/"))
                   or p in {"run.py","protocol.py","research/item_cluster_campaign.py"}}
        for relative,digest in runtime.items():
            assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == digest
        runtimes.append(runtime)
        identities.append({key:experiment[key] for key in
            ["source","shared_initial_state_sha256","selection_indices_sha256","data"]})
        runs[name] = {**result,"parameters":experiment["trainable_parameters"],
                      "history":history,"diagnostics":diag,"directory":str(directory)}
    assert identities[0] == identities[1] == identities[2]
    assert runtimes[0] == runtimes[1] == runtimes[2]
    for name in ["k1","k4"]:
        for scope in ["validation_selection","validation"]:
            runs[name][scope+"_ndcg10_gain_percent_vs_baseline"] = 100*(
                runs[name][scope]["ndcg@10"] / runs["baseline"][scope]["ndcg@10"] - 1)
    source_manifest = json.loads((ROOT / "history/before_item_clusters/source_hashes.json").read_text())
    for relative,digest in source_manifest.items():
        assert hashlib.sha256((ROOT / "history/before_item_clusters" / relative).read_bytes()).hexdigest() == digest
    protected = json.loads((ROOT / "docs/item_clusters/protected_versions.json").read_text())
    changes = {path:{"before":digest,"after":hashlib.sha256((ROOT.parent/path).read_bytes()).hexdigest()
                       if (ROOT.parent/path).exists() else None}
               for path,digest in protected.items() if not (ROOT.parent/path).exists()
               or hashlib.sha256((ROOT.parent/path).read_bytes()).hexdigest() != digest}
    previous = json.loads((ROOT / "outputs/fm_ablation/runs/Beauty/beauty_direct_ce_s2024/results.json").read_text())
    legacy_delta = runs["baseline"]["validation"]["ndcg@10"]-previous["validation"]["ndcg@10"]
    audit={"shared_initialization_data_and_selection_matched":True,"same_runtime_across_three_runs":True,
        "saved_source_hashes_verified":True,"prediction_metrics_recomputed":True,
        "all_checkpoint_reloads_verified":True,"before_source_archive_verified":True,
        "protected_files":len(protected),"observed_protected_changes":changes,
        "old_versions_unchanged":not changes,"legacy_baseline_full_ndcg10_difference":legacy_delta,
        "runtime_source_sha256":json_hash(runtimes[0])}
    summary={"category":"Beauty","seed":2024,"test_accessed":False,"runs":runs,"audit":audit,
        "k4_vs_k1_full_ndcg10_gain_percent":100*(runs["k4"]["validation"]["ndcg@10"]/runs["k1"]["validation"]["ndcg@10"]-1),
        "k4_vs_k1_selection_ndcg10_gain_percent":100*(runs["k4"]["validation_selection"]["ndcg@10"]/runs["k1"]["validation_selection"]["ndcg@10"]-1)}
    write_json(summary_path,summary)
    lines=["# Item VLAD | Beauty | seed2024", "", "40 epochs, 4040 updates; fixed 4096-user selection; full validation; no test.","",
        "| Method | Selection NDCG10 | Full NDCG10 | Full Recall10 | Full NLL | Best epoch | Params | Peak train MiB | Seconds |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name,r in runs.items():
        lines.append(f"| {name} | {r['validation_selection']['ndcg@10']:.8f} | {r['validation']['ndcg@10']:.8f} | {r['validation']['recall@10']:.8f} | {r['validation']['nll']:.5f} | {r['best']['epoch']} | {r['parameters']} | {r['peak_memory_allocated_mib']:.1f} | {r['seconds']:.1f} |")
    lines += ["", "Shared-device training times are descriptive; the runs trained concurrently.",
              "", "K4/K1 full NDCG10 change: "+f"{summary['k4_vs_k1_full_ndcg10_gain_percent']:+.3f}%", "",
              "Audit: "+json.dumps(audit,ensure_ascii=False)]
    (base / "summary.md").write_text("\n".join(lines)+"\n")
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout="constrained")
    for name,r in runs.items():
        selected=[row for row in r["history"] if "validation_selection" in row]
        axes[0].plot([row["epoch"] for row in selected],
                     [row["validation_selection"]["ndcg@10"] for row in selected],marker="o",label=name)
    axes[0].set(xlabel="Epoch",ylabel="Selection NDCG@10");axes[0].legend();axes[0].grid(alpha=.2)
    values=[r["validation"]["ndcg@10"] for r in runs.values()]
    axes[1].bar(list(runs),values,color=["#4c78a8","#72b7b2","#f58518"])
    for i,v in enumerate(values):axes[1].text(i,v+max(values)*.02,f"{v:.5f}",ha="center")
    axes[1].set(ylabel="Full validation NDCG@10",ylim=(0,max(values)*1.16))
    fig.suptitle("Per-item VLAD tokens | Beauty, seed2024 | no test")
    fig.savefig(base / "comparison.png",dpi=180);fig.savefig(base / "comparison.pdf");plt.close(fig)
    print("\n".join(lines))


if __name__ == "__main__": main()
