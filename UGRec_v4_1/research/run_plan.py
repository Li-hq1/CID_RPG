"""Run a finite, reviewable experiment plan, with one training process per GPU."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
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
    plan = json.loads(args.plan.read_text())
    logdir = ROOT / "outputs/iterative_20261028/logs"
    logdir.mkdir(parents=True, exist_ok=True)
    # A slot is assigned a whole GPU queue; never concurrently place two jobs there.
    queues = {}
    for entry in plan:
        queues.setdefault(entry["device"], []).append(entry)

    def run_queue(entries):
        results = []
        for entry in entries:
            if entry.get("wait_for"):
                dependency = ROOT / entry["wait_for"]
                waiting_since = time.monotonic()
                while not dependency.exists():
                    if time.monotonic() - waiting_since > 7200:
                        raise TimeoutError(f"Dependency did not finish: {dependency}")
                    time.sleep(2)
            command = [sys.executable, "-m", "research.iterate"]
            for key, value in entry.items():
                if key == "wait_for":
                    continue
                command.extend(["--" + key.replace("_", "-"), str(value)])
            path = logdir / f"{entry['category']}_{entry['name']}_s{entry['seed']}.log"
            print(json.dumps({"started": entry, "log": str(path)}), flush=True)
            environment = os.environ.copy()
            environment.update(OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
            with path.open("w") as stream:
                process = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                                         env=environment, check=False)
            record = {"entry": entry, "exit_code": process.returncode, "log": str(path)}
            print(json.dumps(record), flush=True)
            results.append(record)
        return results

    results = []
    with ThreadPoolExecutor(max_workers=len(queues)) as pool:
        futures = [pool.submit(run_queue, entries) for entries in queues.values()]
        for future in as_completed(futures):
            results.extend(future.result())
            write_json(logdir / f"{args.plan.stem}_status.json", results)
    if any(result["exit_code"] != 0 for result in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
