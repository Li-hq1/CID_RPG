"""Bounded, sequential Beauty/seed2024 loss experiments; no parameter grid."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from misc.artifacts import write_json
from protocol import ROOT


def specification(name):
    config = json.loads((ROOT / "configs/flowrec.json").read_text())
    if name == "control":
        config = json.loads((ROOT / "configs/control.json").read_text())
    elif name == "prior":
        config["flow_ranking"].update(cfm_weight=0., align_weight=0.)
        config["evaluation"]["prediction_source"] = "prior"
    elif name == "no_cfm":
        config["flow_ranking"]["cfm_weight"] = 0.
    elif name == "low_cfm":
        config["flow_ranking"]["cfm_weight"] = 1.
    elif name != "full":
        raise ValueError(name)
    config["data"]["category"] = "Beauty"
    config["training"].update(seed=2024, epochs=40, max_steps=None, smoke_test=False,
                              output_dir="outputs/flowrec/runs",
                              run_id=f"flowrec_{name}_s2024")
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["initial", "no_cfm", "low_cfm"], required=True)
    parser.add_argument("--device", required=True)
    args = parser.parse_args()
    names = ["prior", "full", "control"] if args.stage == "initial" else [args.stage]
    base = ROOT / "outputs/flowrec"
    configs = base / "configs"
    configs.mkdir(parents=True, exist_ok=True)
    status_path = base / f"status_{args.stage}.json"
    if status_path.exists():
        raise FileExistsError(f"Stage already started: {status_path}")
    status = {"stage": args.stage, "device": args.device, "seed": 2024,
              "state": "running", "started_unix": time.time(), "runs": []}
    write_json(status_path, status)
    for name in names:
        config = specification(name)
        config["training"]["device"] = args.device
        run_id = config["training"]["run_id"]
        config_path = configs / f"{run_id}.json"
        if config_path.exists():
            raise FileExistsError(config_path)
        write_json(config_path, config)
        log_path = base / f"{run_id}.log"
        command = [sys.executable, str(ROOT / "run.py"), "--config", str(config_path)]
        entry = {"name": name, "command": command, "log": str(log_path), "state": "running"}
        status["runs"].append(entry)
        write_json(status_path, status)
        print(f"START {name}", flush=True)
        with log_path.open("x") as log:
            result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                    env={**os.environ, "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"})
        entry.update(state="complete" if result.returncode == 0 else "failed", returncode=result.returncode)
        if result.returncode:
            status["state"] = "failed"
            write_json(status_path, status)
            raise SystemExit(result.returncode)
        run_dir = ROOT / config["training"]["output_dir"] / "Beauty" / run_id
        entry["results"] = json.loads((run_dir / "results.json").read_text())
        write_json(status_path, status)
        print(f"DONE {name}: {entry['results']['validation']}", flush=True)
    status.update(state="complete", finished_unix=time.time())
    write_json(status_path, status)


if __name__ == "__main__":
    main()
