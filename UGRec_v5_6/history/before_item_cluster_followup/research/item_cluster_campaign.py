"""Fixed Beauty baseline/K1/K4 campaign. Exploration uses only seed2024."""

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
    parser.add_argument("--stage", required=True, choices=["preflight", "train"])
    parser.add_argument("--method", choices=["baseline", "k1", "k4", "all"], default="all")
    parser.add_argument("--device", required=True)
    args = parser.parse_args()
    if args.stage == "train":
        preflight = json.loads((ROOT / "outputs/item_clusters/preflight/status_all.json").read_text())
        if preflight["state"] != "complete":
            raise RuntimeError("Complete the three full-width real-cache preflight runs first.")
    base = ROOT / "outputs/item_clusters" / args.stage
    base.mkdir(parents=True, exist_ok=True)
    status_path = base / f"status_{args.method}.json"
    if status_path.exists():
        raise FileExistsError(status_path)
    status = {"state": "running", "device": args.device, "stage": args.stage,
              "seed": 2024, "started_unix": time.time(), "runs": []}
    write_json(status_path, status)
    for method in (["baseline", "k1", "k4"] if args.method == "all" else [args.method]):
        cfg = json.loads((ROOT / f"configs/item_clusters/{method}.json").read_text())
        cfg["training"]["device"] = args.device
        if args.stage == "preflight":
            cfg["training"].update(epochs=1, max_steps=4, output_dir="outputs/item_clusters/preflight/runs",
                                   run_id=f"beauty_item_vlad_{method}_s2024_preflight")
        target = base / f"{method}.json"
        log_path = base / f"{method}.log"
        if target.exists() or log_path.exists():
            raise FileExistsError(method)
        write_json(target, cfg)
        entry = {"method": method, "state": "running", "started_unix": time.time()}
        status["runs"].append(entry)
        write_json(status_path, status)
        command = [sys.executable, str(ROOT / "run.py"), "--config", str(target)]
        if args.stage == "preflight":
            command += ["--eval-users", "8", "--audit", "subset"]
        with log_path.open("x") as log:
            completed = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                env={**os.environ, "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"})
        entry.update(state="complete" if completed.returncode == 0 else "failed",
                     returncode=completed.returncode, finished_unix=time.time())
        write_json(status_path, status)
        if completed.returncode:
            status["state"] = "failed"; write_json(status_path, status)
            raise SystemExit(completed.returncode)
        directory = ROOT / cfg["training"]["output_dir"] / "Beauty" / cfg["training"]["run_id"]
        entry["results"] = json.loads((directory / "results.json").read_text())
        if args.stage == "preflight":
            history = json.loads((directory / "metrics.json").read_text())
            assert entry["results"]["steps"] == 4
            assert entry["results"]["checkpoint_reload_verified"]
            assert history[0]["train"]["ae_joint_grad_norm"] > 0
        write_json(status_path, status)
        print(json.dumps({"method": method, "results": entry["results"]}), flush=True)
    status.update(state="complete", finished_unix=time.time())
    write_json(status_path, status)


if __name__ == "__main__": main()
