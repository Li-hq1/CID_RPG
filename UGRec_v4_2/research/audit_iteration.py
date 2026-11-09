"""Audit selected checkpoints after screening, without any test evaluation."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from joint_training.checkpoint import read_checkpoint
from joint_training.config import JointConfig
from joint_training.model import JointModel
from misc.artifacts import array_hash, json_hash
from .common import corpus, raw_features, write_json
from .compact_ablation import prepare
from .iterate import evaluate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--heads", nargs="+", default=["flow", "one_step", "ranking"])
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("high")
    result = json.loads((args.directory / "results.json").read_text())
    category = result["config"]["data"]["category"]
    content, _, histories, targets = prepare(category)
    directory, mapping, train, _, _ = corpus(category)
    raw = raw_features(directory, len(content)) if result["source"]["input"] == "raw" else content.numpy()
    assert array_hash(raw) == result["source"]["raw_sha256"]
    assert json_hash(train) == result["source"]["train_sha256"]
    assert json_hash(mapping["id2item"]) == result["source"]["item_order_sha256"]
    config = JointConfig.from_dict(result["config"])
    model = JointModel(config, raw).to(args.device)
    model.cache_features()
    for chosen in args.heads:
        if chosen not in result["best"] or f"selected_by_{chosen}" in result["audits"]:
            continue
        checkpoint = read_checkpoint(args.directory / f"best_{chosen}.pt")
        model.load_state_dict(checkpoint["model"], strict=True)
        metrics, predictions = evaluate(model, histories, targets, np.arange(len(targets)), flow=True,
                                         only=None if chosen == "flow" else chosen)
        result["audits"][f"selected_by_{chosen}"] = metrics
        for name, prediction in predictions.items():
            np.savez_compressed(args.directory / f"full_selected_{chosen}_decoded_{name}.npz",
                                targets=targets, predictions=prediction)
        write_json(args.directory / "results.json", result)
        print(chosen, json.dumps(metrics), flush=True)


if __name__ == "__main__":
    main()
