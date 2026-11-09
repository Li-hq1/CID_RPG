"""Validation-only integration-step diagnostic for one already selected checkpoint."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from joint_training.checkpoint import read_checkpoint
from joint_training.config import JointConfig
from joint_training.model import JointModel
from misc.artifacts import array_hash, json_hash, write_json
from protocol import ROOT, corpus, raw_features
from run import evaluate, evaluate_prior


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--steps", type=int, nargs="+", default=[1, 4, 8])
    args = parser.parse_args()
    output = args.run / "integration_diagnostic.json"
    if output.exists():
        raise FileExistsError(output)
    config = JointConfig.from_dict(json.loads((args.run / "config.json").read_text()))
    config.training.device = args.device
    if config.flow_ranking.path != "flowrec" or config.evaluation.prediction_source != "flow":
        raise ValueError("Diagnostic expects a FlowRec generation checkpoint.")
    torch.set_num_threads(config.training.cpu_threads)
    torch.set_float32_matmul_precision("high")
    cache = Path(config.data.cache_dir)
    directory, mapping, train, histories, targets = corpus(config.data.category, cache if cache.is_absolute() else ROOT/cache)
    raw = raw_features(directory, len(mapping["id2item"])-1)
    checkpoint_path = args.run/"best_flow.pt"
    checkpoint = read_checkpoint(checkpoint_path)
    actual = {"category": config.data.category, "raw_sha256": array_hash(raw), "train_sha256": json_hash(train),
              "item_order_sha256": json_hash(mapping["id2item"]),
              "validation_histories_sha256": json_hash(histories), "validation_targets_sha256": array_hash(targets),
              "input": "raw", "teacher_kind": None}
    if actual != checkpoint["source"]:
        raise ValueError("Diagnostic source differs from trained checkpoint.")
    model = JointModel(config, raw).to(args.device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.cache_features()
    with np.load(args.run/"selection_flow.npz") as saved:
        indices = saved["indices"].copy()
        expected = saved["predictions"].copy()
    result = {"kind": "secondary validation-selection integration diagnostic",
              "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
              "device": args.device, "selection_users": len(indices), "test_accessed": False,
              "selection_indices_sha256": array_hash(indices), "steps": {}}
    for steps in args.steps:
        if steps <= 0:
            raise ValueError("steps must be positive")
        config.flow_ranking.steps = steps
        config.recommender.num_sampling_steps = steps+1
        config.validate()
        metrics, predictions = evaluate(model, histories, targets, indices)
        if steps == 1:
            result["original_predictions_equal"] = bool(np.array_equal(expected, predictions))
        result["steps"][str(steps)] = metrics
        print(json.dumps({"steps": steps, "metrics": metrics}), flush=True)
    result["prior"], _ = evaluate_prior(model, histories, targets, indices)
    write_json(output, result)


if __name__ == "__main__":
    main()
