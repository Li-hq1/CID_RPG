"""Full validation and same-checkpoint controls; never evaluates test targets."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from compression.geometry_ae.metrics import geometry_metrics
from joint_training.checkpoint import read_checkpoint
from joint_training.config import JointConfig
from joint_training.model import JointModel
from misc.artifacts import array_hash, json_hash
from .common import corpus, raw_features, write_json
from .unified import evaluate


def load(directory, device):
    result = json.loads((directory / "results.json").read_text())
    config = JointConfig.from_dict(result["config"])
    source_directory, mapping, train, histories, targets = corpus(config.data.category)
    raw = raw_features(source_directory, len(mapping["id2item"]) - 1)
    assert array_hash(raw) == result["source"]["raw_sha256"]
    assert json_hash(train) == result["source"]["train_sha256"]
    assert json_hash(mapping["id2item"]) == result["source"]["item_order_sha256"]
    model = JointModel(config, raw).to(device)
    model.load_state_dict(read_checkpoint(directory / "best_flow.pt")["model"], strict=True)
    model.cache_features()
    model.eval()
    return model, result, histories, targets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--controls", action="store_true")
    parser.add_argument("--ann", action="store_true")
    parser.add_argument("--eval-users", type=int)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("high")
    model, result, histories, targets = load(args.directory, args.device)
    indices = np.arange(len(targets))[:args.eval_users]
    audits = result["audits"] if args.eval_users is None else {}
    if "flow" not in audits:
        audits["flow"], prediction = evaluate(model, histories, targets, indices)
        np.savez_compressed(args.directory / ("full_flow.npz" if args.eval_users is None else "probe_flow.npz"),
                            targets=targets[indices], predictions=prediction)
    if args.controls:
        variants = ["zero_noise", "single"]
        if model.config.flow_ranking.source == "history":
            variants.extend(["prior", "ranking"])
        for name in variants:
            previous_prior = model.flow_prior
            previous_samples = model.eval_config["eval_num_samples"]
            if name == "zero_noise":
                model.flow_prior = lambda h, noise=None: previous_prior(h, torch.zeros(len(h), model.latent_dim, device=h.device, dtype=h.dtype))
            if name == "single":
                # Change only draw count: keep the same metric, candidate
                # discovery and support aggregation as the 32-draw baseline.
                model.eval_config["eval_num_samples"] = 1
            try:
                audits[name], prediction = evaluate(model, histories, targets, indices,
                    source=name if name in {"prior", "ranking"} else "flow")
                np.savez_compressed(args.directory / f"{'full' if args.eval_users is None else 'probe'}_{name}.npz",
                                    targets=targets[indices], predictions=prediction)
            finally:
                model.flow_prior = previous_prior
                model.eval_config["eval_num_samples"] = previous_samples
    if args.ann:
        model.config.evaluation.retrieval_backend = "faiss_hnsw"
        audits["ann"], prediction = evaluate(model, histories, targets, indices)
        np.savez_compressed(args.directory / f"{'full' if args.eval_users is None else 'probe'}_ann.npz",
                            targets=targets[indices], predictions=prediction)
    if args.eval_users is None:
        result["audits"] = audits
        write_json(args.directory / "results.json", result)
    else:
        write_json(args.directory / "probe_audits.json", audits)
    model.refresh_catalog()
    values = model.catalog[1:].cpu().numpy()
    geometry = geometry_metrics(values)
    geometry.pop("per_dim_std")
    unit = values / np.linalg.norm(values, axis=-1, keepdims=True).clip(1e-12)
    geometry["mean_direction_squared_norm"] = float(np.square(unit.mean(0)).sum())
    write_json(args.directory / "geometry.json", geometry)
    print(json.dumps({"audits": audits, "geometry": geometry}), flush=True)


if __name__ == "__main__":
    main()
