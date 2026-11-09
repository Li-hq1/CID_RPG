"""Run a finite full-validation audit plan, one checkpoint queue per GPU."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .common import ROOT, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("plan", type=Path)
    args = parser.parse_args()
    queues = {}
    for entry in json.loads(args.plan.read_text()):
        queues.setdefault(entry["device"], []).append(entry)
    logdir = ROOT / "outputs/iterative_20261028/logs"

    def run_queue(entries):
        records = []
        for entry in entries:
            directory = ROOT / entry["directory"]
            if entry.get("wait_for"):
                dependency = ROOT / entry["wait_for"]
                started = time.monotonic()
                while not dependency.exists():
                    if time.monotonic() - started > 7200:
                        raise TimeoutError(dependency)
                    time.sleep(2)
            if not (directory / "results.json").exists():
                raise FileNotFoundError(f"Training must finish before audit: {directory}")
            command = [sys.executable, "-m", "research.audit_iteration", str(directory),
                       "--device", entry["device"], "--heads", *entry.get("heads", ["flow", "ranking"])]
            log = logdir / f"audit_{directory.parent.name}_{directory.name}.log"
            environment = os.environ.copy()
            environment.update(OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
            with log.open("w") as stream:
                process = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                                         env=environment, check=False)
            record = {**entry, "exit_code": process.returncode, "log": str(log)}
            print(json.dumps(record), flush=True)
            records.append(record)
        return records

    with ThreadPoolExecutor(max_workers=len(queues)) as pool:
        groups = list(pool.map(run_queue, queues.values()))
    records = [record for group in groups for record in group]
    write_json(logdir / f"{args.plan.stem}_status.json", records)
    if any(record["exit_code"] for record in records):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
