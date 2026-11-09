"""Compare matched completed runs; reject protocol, source or budget mismatches."""

import argparse
import copy
import json
from pathlib import Path

from misc.artifacts import write_json


def compare(left, right):
    left, right = Path(left), Path(right)
    a, b = [json.loads((p / "results.json").read_text()) for p in (left, right)]
    if not all(r["benchmark_eligible"] and r["checkpoint_reload_verified"] and not r["test_accessed"] for r in (a, b)):
        raise ValueError("Need completed benchmark-eligible runs without test access.")
    for key in ("source", "selection_indices_sha256", "selection_users", "selection_metric", "steps", "completed_epochs"):
        if a[key] != b[key]:
            raise ValueError(f"Unmatched {key}")
    ca, cb = [copy.deepcopy(json.loads((p / "config.json").read_text())) for p in (left, right)]
    for c in (ca, cb):
        c["training"].pop("run_id")
        c["experiment"].pop("arm")
    if ca != cb:
        raise ValueError("Non-arm configuration differences; explicitly define a new comparison.")
    metrics = ["ndcg@10", "recall@10", "step1/ndcg@10", "step2/ndcg@10", "step3/ndcg@10"]
    return {"left": str(left), "right": str(right), "left_arm": a["arm"], "right_arm": b["arm"],
            "matched_protocol_and_budget": True, "test_accessed": False, "single_seed_trend_only": True,
            "metrics": {k: {"left": a["validation"][k], "right": b["validation"][k],
                              "relative_change": b["validation"][k] / a["validation"][k] - 1 if a["validation"][k] else None}
                        for k in metrics if k in a["validation"] and k in b["validation"]},
            "left_nfe_per_user": a["nfe_per_user"], "right_nfe_per_user": b["nfe_per_user"],
            "right_diagnostics": b["diagnostics"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("left", type=Path)
    parser.add_argument("right", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = compare(args.left, args.right)
    write_json(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
