"""Run finite-update real-cache engineering checks, keeping all model widths."""

import argparse
import json
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--id", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = root / "outputs" / "readiness" / args.id
    output.mkdir(parents=True, exist_ok=False)
    jobs = [(f"Beauty_{arm}", "Beauty", config, []) for arm, config in
            [("A", "A_ce"), ("B", "B_next_fm"), ("C", "C_future_ce"), ("D", "D_future_fm")]]
    jobs += [("Beauty_LOO", "Beauty", "A_ce", ["--protocol", "leave_one_out"]),
             ("CDs_C", "CDs", "C_future_ce", [])]
    rows = []
    for name, category, config, extra in jobs:
        run_id = f"{args.id}_{name}"
        command = ["bash", "scripts/python.sh", "run.py", "--config", f"configs/{config}.json",
                   "--category", category, "--device", args.device, "--check", "--run-id", run_id] + extra
        with (output / f"{name}.log").open("w") as log:
            completed = subprocess.run(command, cwd=root, stdout=log, stderr=subprocess.STDOUT)
        if completed.returncode:
            raise RuntimeError(f"{name} failed; see {output / (name + '.log')}")
        canonical = "CDs_and_Vinyl" if category == "CDs" else category
        path = root / "outputs/experiments" / canonical / run_id / "results.json"
        result = json.loads(path.read_text())
        assert result["engineering_check"] and not result["benchmark_eligible"]
        assert result["checkpoint_reload_verified"] and not result["test_accessed"]
        rows.append({"name": name, "results": str(path.relative_to(root)), "steps": result["steps"],
                     "parameters": result["trainable_parameters"], "source_sha256": result["source"]["source_sha256"],
                     "checkpoint_reload_verified": True})
        print(f"{name}: {result['steps']} updates, checkpoint reload verified", flush=True)
    assert len({r["source_sha256"] for r in rows}) == 1
    (output / "summary.json").write_text(json.dumps({"engineering_check": True, "test_accessed": False,
                                                     "runs": rows}, indent=2) + "\n")


if __name__ == "__main__":
    main()
