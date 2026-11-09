"""Move only not-yet-started Sports jobs to two GPUs after Beauty completes."""

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from .common import ROOT, write_json


def main():
    unit = "ugrec-v42-confirmation-20261029.service"
    pid = int(subprocess.check_output(["systemctl", "--user", "show", unit, "--property=MainPID", "--value"], text=True))
    command = Path(f"/proc/{pid}/cmdline").read_bytes()
    if b"research.run_unified" not in command:
        raise RuntimeError("Refusing to pause a process other than this experiment runner.")
    # Pause the runner's threads, not its already-running child trainings.
    os.kill(pid, signal.SIGSTOP)
    try:
        directories = [ROOT / f"outputs/unified_20261028/Beauty/frozen_{name}_s2026_e40/results.json" for name in ["flow", "direct"]]
        deadline = time.monotonic() + 1800
        while not all(p.exists() for p in directories):
            if time.monotonic() > deadline:
                raise TimeoutError("Waiting for the current Beauty jobs exceeded 30 minutes.")
            time.sleep(10)
        time.sleep(2)
        # Kill only our paused queue process, without giving its threads a
        # chance to start another job. Completed children have committed output.
        os.kill(pid, signal.SIGKILL)
        subprocess.run(["systemctl", "--user", "stop", unit], check=False)
    except BaseException:
        try:
            os.kill(pid, signal.SIGCONT)
        except ProcessLookupError:
            pass
        raise
    original = json.loads((ROOT / "research/v4_2_confirmation.json").read_text())
    sports = [e for e in original if e["category"] == "Sports_and_Outdoors"]
    for i, entry in enumerate(sports):
        entry["device"] = f"cuda:{i % 2}"
        directory = ROOT / "outputs/unified_20261028" / entry["category"] / f"{entry['name']}_s{entry['seed']}_e40"
        if directory.exists():
            raise RuntimeError(f"Sports unexpectedly started before redistribution: {directory}")
    plan = ROOT / "research/v4_2_sports_parallel.json"
    write_json(plan, sports)
    subprocess.run([sys.executable, "-u", "-m", "research.run_unified", str(plan), "--workers", "2"], cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
