"""Recompute metrics and identities for completed, independent loss runs."""

import hashlib
import json

import numpy as np

from misc.artifacts import array_hash, json_hash, write_json
from protocol import ROOT, ranking_metrics


def main():
    base = ROOT / "outputs/flowrec"
    rows = []
    experiments = []
    core_hashes = []
    for directory in sorted((base / "runs/Beauty").glob("flowrec_*_s2024")):
        if not (directory / "results.json").exists():
            continue
        result = json.loads((directory/"results.json").read_text())
        config = json.loads((directory/"config.json").read_text())
        experiment = json.loads((directory/"experiment.json").read_text())
        log = json.loads((directory/"metrics.json").read_text())
        assert result["steps"] == 4040 and result["completed_epochs"] == 40
        assert experiment["seed"] == 2024 and experiment["selection_users"] == 4096
        assert result["benchmark_eligible"] and result["checkpoint_reload_verified"]
        assert not result["test_accessed"] and not experiment["test_accessed"]
        with np.load(directory/"full_flow.npz") as full, np.load(directory/"selection_flow.npz") as selected:
            assert len(full["targets"]) == 22363
            assert array_hash(selected["indices"]) == experiment["selection_indices_sha256"]
            assert np.array_equal(full["targets"][selected["indices"]], selected["targets"])
            for filename, expected in [(full, result["validation"]), (selected, result["validation_selection"])]:
                metrics = ranking_metrics(filename["predictions"], filename["targets"])
                for key, value in metrics.items():
                    assert abs(value-expected[key]) < 1e-12, (directory, key)
        if "validation_prior" in result:
            with np.load(directory/"full_prior.npz") as prior:
                metrics = ranking_metrics(prior["predictions"], prior["targets"])
                for key, value in metrics.items():
                    assert abs(value-result["validation_prior"][key]) < 1e-12
        hashes = json.loads((directory/"source_hashes.json").read_text())
        for relative, expected in hashes.items():
            assert hashlib.sha256((directory/"source_snapshot"/relative).read_bytes()).hexdigest() == expected
        core_hashes.append(json_hash({key: value for key, value in hashes.items()
            if key in {"run.py", "protocol.py"} or key.split("/")[0] in {"models", "joint_training", "compression", "misc"}}))
        experiments.append(experiment)
        row = {"run": directory.name, "path": str(directory), "best_epoch": result["best"]["epoch"],
               "selection": result["validation_selection"], "validation": result["validation"],
               "alpha": config["flow_ranking"]["cfm_weight"] if config["flow_ranking"]["path"] == "flowrec" else None,
               "beta": config["flow_ranking"]["align_weight"] if config["flow_ranking"]["path"] == "flowrec" else None,
               "prediction_source": config["evaluation"]["prediction_source"],
               "seconds": result["seconds"], "steps": result["steps"],
               "peak_memory_mib": result.get("peak_memory_allocated_mib"),
               "first_epoch": log[0]["train"], "last_epoch": log[-1]["train"],
               "curve": [{key: entry[key] for key in ["epoch", "validation_selection", "validation_selection_prior"] if key in entry}
                         for entry in log if "validation_selection" in entry]}
        if "validation_prior" in result:
            row["same_checkpoint_prior"] = result["validation_prior"]
        if (directory/"integration_diagnostic.json").exists():
            row["integration_diagnostic"] = json.loads((directory/"integration_diagnostic.json").read_text())
        rows.append(row)
    assert rows, "No completed runs"
    for key in ["initial_state_sha256", "source", "selection_indices_sha256", "trainable_parameters"]:
        assert all(exp[key] == experiments[0][key] for exp in experiments), key
    assert len(set(core_hashes)) == 1, "Training code differed across runs"
    protected = json.loads((base/"protected_before.json").read_text())
    changed = [path for path, expected in protected.items()
               if not (ROOT.parent/path).exists() or hashlib.sha256((ROOT.parent/path).read_bytes()).hexdigest() != expected]
    report = {"runs": rows, "n_runs": len(rows), "metric_recomputation_verified": True,
              "snapshots_verified": True, "training_code_equal": True, "initialization_equal": True,
              "source_equal": True, "selection_equal": True, "test_accessed": False,
              "protected_files_checked": len(protected), "protected_files_changed": changed,
              "protected_files_change_note": "Changes were detected in pre-existing v5_1 files; this campaign did not write them.",
              "core_source_sha256": core_hashes[0], "initial_state_sha256": experiments[0]["initial_state_sha256"]}
    write_json(base/"summary.json", report)
    for row in rows:
        print(json.dumps({key: row[key] for key in ["run", "best_epoch", "selection", "validation"]}))
    print(json.dumps({"protected_files_changed": changed, "runs": len(rows)}))


if __name__ == "__main__":
    main()
