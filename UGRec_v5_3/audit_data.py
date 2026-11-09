"""Audit window eligibility without mapping any test target."""

import argparse
from pathlib import Path

from misc.artifacts import write_json
from multi_target import FutureDataset, corpus


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--category", default="Beauty")
    parser.add_argument("--horizons", type=int, nargs="+", default=[1, 3, 5])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    rows = []
    for horizon in args.horizons:
        split = corpus(args.category, horizon=horizon)
        dataset = FutureDataset(split.train, horizon)
        rows.append({**split.audit, "training_windows": len(dataset),
                     "next_queries": dataset.next_queries, "auxiliary_queries": dataset.auxiliary_queries})
    write_json(args.output, {"category": args.category, "model_metrics_used": False,
                            "test_targets_accessed": False, "windows": rows})
    print(args.output.read_text())


if __name__ == "__main__":
    main()
