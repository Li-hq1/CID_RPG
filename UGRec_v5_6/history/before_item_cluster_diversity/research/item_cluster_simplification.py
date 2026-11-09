"""One explicitly selected hypothesis at a time; fixed Beauty/seed2024 protocol."""

import argparse
import json
import os
import subprocess
import sys
import time

from misc.artifacts import write_json
from protocol import ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", required=True)
    parser.add_argument("--stage", required=True, choices=["preflight", "train"])
    parser.add_argument("--device", required=True)
    args = parser.parse_args()
    config_path = ROOT / "configs/item_cluster_simplification" / f"{args.method}.json"
    config = json.loads(config_path.read_text())
    if config["training"]["seed"] != 2024 or config["data"]["category"] != "Beauty":
        raise ValueError("This exploration fixes Beauty and seed2024.")
    base = ROOT / "outputs/item_cluster_simplification" / args.stage
    base.mkdir(parents=True, exist_ok=True)
    status_path = base / f"status_{args.method}.json"
    if status_path.exists(): raise FileExistsError(status_path)
    if args.stage == "train":
        preflight = ROOT / "outputs/item_cluster_simplification/preflight" / f"status_{args.method}.json"
        if json.loads(preflight.read_text())["state"] != "complete":
            raise RuntimeError("Complete this method's full-width preflight first.")
    config["training"]["device"] = args.device
    if args.stage == "preflight":
        config["training"].update(epochs=1,max_steps=4,output_dir="outputs/item_cluster_simplification/preflight/runs",
                                  run_id=config["training"]["run_id"]+"_preflight")
    resolved = base / f"{args.method}.json"
    if resolved.exists(): raise FileExistsError(resolved)
    write_json(resolved,config)
    status={"state":"running","method":args.method,"stage":args.stage,
            "device":args.device,"started_unix":time.time(),"test_accessed":False}
    write_json(status_path,status)
    command=[sys.executable,str(ROOT / "run.py"),"--config",str(resolved)]
    if args.stage == "preflight": command += ["--eval-users","8","--audit","subset"]
    with (base / f"{args.method}.log").open("x") as log:
        completed=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
            env={**os.environ,"OMP_NUM_THREADS":"2","MKL_NUM_THREADS":"2"})
    status.update(state="complete" if completed.returncode==0 else "failed",
                  returncode=completed.returncode,finished_unix=time.time())
    if not completed.returncode:
        directory=ROOT / config["training"]["output_dir"] / "Beauty" / config["training"]["run_id"]
        status["run_directory"]=str(directory)
        status["results"]=json.loads((directory / "results.json").read_text())
        if args.stage=="preflight":
            history=json.loads((directory / "metrics.json").read_text())
            assert status["results"]["steps"]==4 and status["results"]["checkpoint_reload_verified"]
            assert history[0]["train"]["ae_joint_grad_norm"] > 0
            rec = config["recommender"]
            if rec["item_vlad_variant"] != "normalized_input":
                assert history[0]["train"]["item_vlad_ce_grad_norm"] > 0
            if rec["item_vlad_mapping"] == "linear":
                assert history[0]["train"]["item_mapping_ce_grad_norm"] > 0
            mode = rec.get("item_vlad_pooling", "sum")
            if mode != "sum":
                assert history[0]["train"]["item_vlad_ce_grad_norm"] > 0
            if mode == "context" and config["recommender"]["item_vlad_clusters"] > 1:
                assert history[0]["train"]["item_pool_ce_grad_norm"] > 0
    write_json(status_path,status)
    print(json.dumps(status),flush=True)
    if completed.returncode: raise SystemExit(completed.returncode)


if __name__=="__main__":main()
