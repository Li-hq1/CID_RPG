"""Finite sequential experiments; followups require a recorded hypothesis."""

import argparse
import json
import os
import subprocess
import sys
import time

from misc.artifacts import write_json
from protocol import ROOT


def specification(name):
    config = json.loads((ROOT / "configs" / ("control.json" if name == "control" else "clean_endpoint.json")).read_text())
    settings = config["flow_ranking"]
    if name == "ce_only":
        settings["fm_weight"] = 0.
    elif name == "fm01":
        settings["fm_weight"] = .1
    elif name == "fm5":
        settings["fm_weight"] = 5.
    elif name == "fm_only":
        settings["ce_weight"] = 0.
    elif name == "source_ce":
        settings.update(fm_weight=0., endpoint_training="source")
    elif name not in {"control", "joint"}:
        raise ValueError(name)
    config["training"].update(seed=2024, epochs=40, max_steps=None, smoke_test=False,
                              output_dir="outputs/clean_endpoint/runs", run_id=f"endpoint_{name}_s2024")
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=["initial", "fm01", "fm5", "source_ce", "fm_only"])
    parser.add_argument("--device", required=True)
    parser.add_argument("--hypothesis", default="")
    args = parser.parse_args()
    if args.stage != "initial" and not args.hypothesis.strip():
        parser.error("Followup requires --hypothesis based on observed selection/loss diagnostics.")
    base = ROOT / "outputs/clean_endpoint"
    configs = base / "configs"
    configs.mkdir(parents=True, exist_ok=True)
    status_path = base / f"status_{args.stage}.json"
    if status_path.exists():
        raise FileExistsError(status_path)
    status = {"stage": args.stage, "hypothesis": args.hypothesis, "device": args.device,
              "state": "running", "started_unix": time.time(), "runs": []}
    write_json(status_path, status)
    names = ["control", "ce_only", "joint"] if args.stage == "initial" else [args.stage]
    for name in names:
        config = specification(name)
        config["training"]["device"] = args.device
        run_id = config["training"]["run_id"]
        config_path, log_path = configs / f"{run_id}.json", base / f"{run_id}.log"
        if config_path.exists() or log_path.exists():
            raise FileExistsError(run_id)
        write_json(config_path, config)
        entry = {"name": name, "state": "running", "started_unix": time.time()}
        status["runs"].append(entry)
        write_json(status_path, status)
        print(f"START {name}", flush=True)
        with log_path.open("x") as log:
            result = subprocess.run([sys.executable, str(ROOT / "run.py"), "--config", str(config_path)],
                cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                env={**os.environ, "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"})
        entry.update(state="complete" if result.returncode == 0 else "failed", returncode=result.returncode)
        if result.returncode:
            status["state"] = "failed"
            write_json(status_path, status)
            raise SystemExit(result.returncode)
        run_dir = ROOT / config["training"]["output_dir"] / "Beauty" / run_id
        entry["results"] = json.loads((run_dir / "results.json").read_text())
        write_json(status_path, status)
        print(f"DONE {name}: {entry['results']['validation_selection']}", flush=True)
    status.update(state="complete", finished_unix=time.time())
    write_json(status_path, status)


if __name__ == "__main__":
    main()
