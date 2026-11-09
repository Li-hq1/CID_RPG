"""Recompute metrics and verify source, data, initialized weights and probes."""

import argparse
import hashlib
import json
import math

import numpy as np

from misc.artifacts import json_hash,write_json
from protocol import ROOT,ranking_metrics


BASE = ROOT / "outputs/item_cluster_followup"
REFERENCE = ROOT / "outputs/item_clusters/runs/Beauty/beauty_item_vlad_baseline_s2024"


def audit_run(directory, *, probe=True):
    read=lambda name:json.loads((directory/name).read_text())
    result,experiment,config,history,hashes=[read(name)for name in
        ["results.json","experiment.json","config.json","metrics.json","source_hashes.json"]]
    assert result["benchmark_eligible"] and experiment["benchmark_eligible"]
    assert result["checkpoint_reload_verified"] and not result["test_accessed"]
    assert not experiment["test_accessed"] and not experiment["has_fm_module"]
    assert experiment["seed"]==2024 and experiment["selection_users"]==4096
    assert config["data"]["category"]=="Beauty" and config["flow_ranking"]["path"]=="direct_ce"
    assert result["completed_epochs"]==40 and len(history)==40
    assert result["steps"]==40*math.ceil(experiment["data"]["training_windows"]/256)==4040
    assert config["recommender"]["n_layer"]==2
    assert not config["training"]["evaluate_test"]
    for filename,key,count in [("selection_flow.npz","validation_selection",4096),("full_flow.npz","validation",22363)]:
        with np.load(directory/filename)as saved:
            assert len(saved["targets"])==count
            metrics=ranking_metrics(saved["predictions"],saved["targets"])
            for metric,value in metrics.items():assert abs(value-result[key][metric])<1e-12
    assert json_hash(hashes)==experiment["source_snapshot_sha256"]
    for filename,digest in hashes.items():
        assert hashlib.sha256((directory/"source_snapshot"/filename).read_bytes()).hexdigest()==digest
    runtime={name:digest for name,digest in hashes.items()if name.startswith(
        ("models/","compression/","joint_training/","misc/"))or name in {"run.py","protocol.py"}}
    record={**result,"experiment":experiment,"config":config,"history":history,
            "runtime_hashes":runtime,"directory":str(directory)}
    if probe:
        p=read("followup_probe.json")
        assert p["initial_state_reproduced"] and p["reload_predictions_verified"] and not p["test_accessed"]
        assert hashlib.sha256((directory/"best_flow.pt").read_bytes()).hexdigest()==p["checkpoint_sha256"]
        assert hashlib.sha256((directory/"followup_probe_source.py").read_bytes()).hexdigest()==p["probe_script_sha256"]
        assert json_hash(p["initial_parameter_hashes"])==experiment["initial_state_sha256"]
        record["probe"]=p
    return record


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--methods",required=True,nargs="+")
    parser.add_argument("--name",required=True)
    args=parser.parse_args()
    output=BASE/f"{args.name}.json"
    if output.exists():raise FileExistsError(output)
    reference=audit_run(REFERENCE,probe=False)
    records={}
    for name in args.methods:
        directory=BASE/"runs/Beauty"/f"beauty_{name}_s2024"
        r=audit_run(directory)
        for key in ["shared_initial_state_sha256","source","selection_indices_sha256","data"]:
            assert r["experiment"][key]==reference["experiment"][key],(name,key)
        preflight=BASE/"preflight/runs/Beauty"/f"beauty_{name}_s2024_preflight"
        pr,pe,pc,ph=[json.loads((preflight/filename).read_text())for filename in
                     ["results.json","experiment.json","config.json","source_hashes.json"]]
        assert pr["steps"]==4 and pr["checkpoint_reload_verified"] and not pr["test_accessed"]
        assert not pe["model_reduced"] and not pr["benchmark_eligible"]
        assert pc["recommender"]["train_batch_size"]==256 and pc["ae"]["latent_dim"]==128
        for key in ["initial_state_sha256","shared_initial_state_sha256","source","data"]:
            assert pe[key]==r["experiment"][key],(name,"preflight",key)
        assert json_hash(ph)==pe["source_snapshot_sha256"]
        for filename,digest in ph.items():
            assert hashlib.sha256((preflight/"source_snapshot"/filename).read_bytes()).hexdigest()==digest
        assert all(ph[filename]==digest for filename,digest in r["runtime_hashes"].items())
        pm=json.loads((preflight/"metrics.json").read_text())[0]["train"]
        assert pm["ae_joint_grad_norm"]>0
        if name.startswith("ae_"):
            assert pm["cluster_ce_grad_norm"]>0 and pm["cluster_geometry_grad_norm"]>0
        r["preflight_verified"]={"full_width":True,"steps":4,"reload":True,
                                 "initial_state_and_runtime_match_official_run":True,
                                 "gradients":{k:pm[k]for k in ["ae_joint_grad_norm","cluster_ce_grad_norm",
                                                             "cluster_geometry_grad_norm"]if k in pm}}
        for scope in ["validation_selection","validation"]:
            r[scope+"_ndcg10_gain_percent_vs_baseline"]=100*(r[scope]["ndcg@10"]/reference[scope]["ndcg@10"]-1)
        records[name]=r
    old=json.loads((ROOT/"docs/item_cluster_followup/previous_artifacts.json").read_text())
    old_changes=[name for name,digest in old.items()if not (ROOT/name).exists()
                or hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest]
    assert not old_changes,old_changes
    archive=ROOT/"history/before_item_cluster_followup"
    archive_hashes=json.loads((archive/"source_hashes.json").read_text())
    for name,digest in archive_hashes.items():assert hashlib.sha256((archive/name).read_bytes()).hexdigest()==digest
    protection=json.loads((ROOT/"docs/item_cluster_followup/protected_versions.json").read_text())
    observed_changes={name:{"before":digest,"after":hashlib.sha256((ROOT.parent/name).read_bytes()).hexdigest()
        if (ROOT.parent/name).exists()else None}for name,digest in protection.items()
        if not (ROOT.parent/name).exists()or hashlib.sha256((ROOT.parent/name).read_bytes()).hexdigest()!=digest}
    comparisons={}
    for name,r in records.items():
        if name.startswith("scaled_"):
            k=name.removeprefix("scaled_")
            old_run=audit_run(ROOT/f"outputs/item_clusters/runs/Beauty/beauty_item_vlad_{k}_s2024",probe=False)
            assert r["experiment"]["initial_state_sha256"]==old_run["experiment"]["initial_state_sha256"]
            comparisons[name+"_vs_unit"]={scope:100*(r[scope]["ndcg@10"]/old_run[scope]["ndcg@10"]-1)
                                         for scope in ["validation_selection","validation"]}
    if all(name in records for name in ["ae_sum_multi","ae_sum_merged"]):
        a,b=records["ae_sum_multi"],records["ae_sum_merged"]
        assert a["experiment"]["initial_state_sha256"]==b["experiment"]["initial_state_sha256"]
        assert a["runtime_hashes"]==b["runtime_hashes"]
        comparisons["ae_multi_vs_merged"]={scope:100*(a[scope]["ndcg@10"]/b[scope]["ndcg@10"]-1)
                                          for scope in ["validation_selection","validation"]}
    if all(name in records for name in ["ae_linear_multi","ae_sum_multi"]):
        linear,summed=records["ae_linear_multi"],records["ae_sum_multi"]
        ah,bh=linear["probe"]["initial_parameter_hashes"],summed["probe"]["initial_parameter_hashes"]
        assert all(ah[key]==value for key,value in bh.items())
        comparisons["ae_linear_vs_sum"]={scope:100*(linear[scope]["ndcg@10"]/summed[scope]["ndcg@10"]-1)
                                         for scope in ["validation_selection","validation"]}
    if all(name in records for name in ["ae_linear_multi","ae_linear_merged"]):
        a,b=records["ae_linear_multi"],records["ae_linear_merged"]
        assert a["experiment"]["initial_state_sha256"]==b["experiment"]["initial_state_sha256"]
        assert a["runtime_hashes"]==b["runtime_hashes"]
        comparisons["ae_linear_multi_vs_merged"]={scope:100*(a[scope]["ndcg@10"]/b[scope]["ndcg@10"]-1)
                                                 for scope in ["validation_selection","validation"]}
    mechanisms={}
    for name,r in records.items():
        if not name.startswith("ae_"):continue
        directory=BASE/"mechanisms"/name
        p=json.loads((directory/"aggregation_probe.json").read_text())
        assert p["initial_state_reproduced"] and not p["test_accessed"]
        assert p["checkpoint_sha256"]==r["probe"]["checkpoint_sha256"]
        assert p["training_item_ids_sha256"]==r["probe"]["training_item_ids_sha256"]
        assert hashlib.sha256((directory/"probe_source.py").read_bytes()).hexdigest()==p["script_sha256"]
        for stage in ["initial","trained"]:
            assert p[stage]["content_sum_assignment_cancellation_verified"]
            assert p[stage]["implementation_and_formula_match"]
        mechanisms[name]=p
    audit={"all_snapshot_hashes_verified":True,"all_prediction_metrics_recomputed":True,
        "all_reload_predictions_verified":True,"all_shared_backbone_initializations_match_baseline":True,
        "all_full_width_preflights_verified":True,
        "all_data_and_selection_match":True,"previous_artifacts_unchanged":True,"previous_artifacts":len(old),
        "archive_verified":True,"protected_files":len(protection),"observed_protected_changes":observed_changes,
        "all_other_versions_unchanged":not observed_changes,
        "runtime_note":"Source can evolve between stages; each actual run is verified against its own saved snapshot."}
    write_json(output,{"runs":records,"reference":reference,"comparisons":comparisons,
                       "mechanisms":mechanisms,"audit":audit,"test_accessed":False})
    for name,r in records.items():
        print(json.dumps({"method":name,"selection":r["validation_selection"]["ndcg@10"],
            "full":r["validation"]["ndcg@10"],"best_epoch":r["best"]["epoch"],
            "full_gain":r["validation_ndcg10_gain_percent_vs_baseline"]}),flush=True)
    print(json.dumps({"comparisons":comparisons,"audit":audit}),flush=True)


if __name__=="__main__":main()
