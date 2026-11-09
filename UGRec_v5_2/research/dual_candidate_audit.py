"""Read-only candidate-pool audit using saved full-validation top-10 lists.

This measures list-level complementarity, not per-draw FM candidate voting.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from misc.artifacts import write_json
from protocol import ROOT, ranking_metrics


def load_predictions(run, filename):
    with np.load(run / filename) as values:
        return values["predictions"].copy(), values["targets"].copy()


def quota_list(ce, fm, ce_slots, k=10):
    fm_slots = k - ce_slots
    ordered = []
    for row_ce, row_fm in zip(ce, fm):
        selected = list(row_ce[:ce_slots]) + list(row_fm[:fm_slots])
        seen = set()
        selected = [int(item) for item in selected if not (item in seen or seen.add(int(item)))]
        for item in list(row_ce) + list(row_fm):
            if len(selected) >= k:
                break
            if int(item) not in seen:
                selected.append(int(item))
                seen.add(int(item))
        if len(selected) != k:
            raise ValueError("Saved lists do not contain enough unique candidates to backfill.")
        ordered.append(selected)
    return np.asarray(ordered, dtype=np.int64)


def hit_counts(predictions, targets):
    return predictions == targets[:, None]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ce-run", default="beauty_ce_only_s2024_20261106")
    parser.add_argument("--fm-run", default="beauty_frozen_ce_flow_s2024_20261106")
    parser.add_argument("--output", default="outputs/dual_candidate_audit_20261106")
    args = parser.parse_args()

    ce_dir = ROOT / "outputs/Beauty" / args.ce_run
    fm_dir = ROOT / "outputs/Beauty" / args.fm_run
    ce, ce_targets = load_predictions(ce_dir, "full_flow.npz")
    fm, fm_targets = load_predictions(fm_dir, "full_flow.npz")
    np.testing.assert_array_equal(ce_targets, fm_targets)
    targets = ce_targets

    ce_hits, fm_hits = hit_counts(ce, targets), hit_counts(fm, targets)
    ce_hit = ce_hits.any(1)
    fm_hit = fm_hits.any(1)
    union_hit = np.logical_or(ce_hit, fm_hit)
    both = ce_hit & fm_hit
    result = {
        "protocol": "Frozen CE checkpoint and its 32-draw/four-step FM aggregate top-10; full validation, one seed.",
        "candidate_semantics": "The saved FM list ranks the average per-draw softmax probability. It is not a per-draw item-vote list.",
        "validation_users": int(len(targets)),
        "test_accessed": False,
        "ce_only_top10": ranking_metrics(ce, targets),
        "fm_only_top10": ranking_metrics(fm, targets),
        "top10_pool_coverage": {
            "ce": float(ce_hit.mean()),
            "fm": float(fm_hit.mean()),
            "union_of_both_top10": float(union_hit.mean()),
            "fm_only_additional_hits_over_ce": int((fm_hit & ~ce_hit).sum()),
            "ce_only_hits_over_fm": int((ce_hit & ~fm_hit).sum()),
            "both_hit": int(both.sum()),
        },
        "quota_lists": {},
    }
    for ce_slots in (10, 8, 6, 5, 4, 2, 0):
        predictions = quota_list(ce, fm, ce_slots)
        result["quota_lists"][f"ce_{ce_slots}_fm_{10-ce_slots}"] = ranking_metrics(predictions, targets)

    output = ROOT / args.output
    output.mkdir(parents=True, exist_ok=False)
    result["source_runs"] = {"ce": str(ce_dir), "fm": str(fm_dir)}
    write_json(output / "summary.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
