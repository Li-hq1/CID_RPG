"""Finite v4_2 plans; explicitly allocated GPUs only, no unrelated processes."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys

from .common import ROOT, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("plan", type=Path)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    logdir = ROOT / "outputs/unified_20261028/logs"
    logdir.mkdir(parents=True, exist_ok=True)

    def run(entry):
        path = logdir / f"{entry['category']}_{entry['name']}_s{entry['seed']}_e{entry['epochs']}.log"
        command = [sys.executable, "-m", "research.unified"]
        for key, value in entry.items():
            command.extend(["--" + key.replace("_", "-"), str(value)])
        print(json.dumps({"started": entry, "log": str(path)}), flush=True)
        with path.open("w") as stream:
            result = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                 env={**os.environ, "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"}, check=False)
        record = {"entry": entry, "exit_code": result.returncode, "log": str(path)}
        print(json.dumps(record), flush=True)
        return record

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(run, plan))
    write_json(logdir / f"{args.plan.stem}_status.json", results)
    if any(x["exit_code"] for x in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
