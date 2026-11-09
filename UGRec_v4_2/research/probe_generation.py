"""Diagnose selected generators on validation histories without refitting.

Single-draw, mean-vector and zero-noise decoders are post-hoc diagnostics,
not new selection criteria. Sample spread is not a calibrated entropy measure.
"""

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from joint_training.checkpoint import read_checkpoint
from joint_training.config import JointConfig
from joint_training.evaluation import evaluation_randomness
from joint_training.model import JointModel
from misc.artifacts import array_hash, json_hash
from models.evaluation_sampling import _nearest_candidates, aggregate_diffusion_candidates
from .common import SEED, batches, corpus, raw_features, summarize, write_json
from .compact_ablation import prepare


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--head", choices=("flow", "one_step"), default="flow")
    parser.add_argument("--users", type=int, default=1024)
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
    assert config.recommender.ode_solver == "euler" and config.recommender.tiger_guidance_weight == 0
    model = JointModel(config, raw).to(args.device)
    model.load_state_dict(read_checkpoint(args.directory / f"best_{args.head}.pt")["model"], strict=True)
    model.cache_features()
    model.eval()
    model.refresh_catalog()
    # A recorded prefix of the screening subset: diagnostic, not a holdout.
    indices = np.load(args.directory / f"selection_{args.head}.npz")["indices"][:args.users]
    predictions = {k: [] for k in ("standard32", "single_draw", "mean_vector", "zero_noise", "ranking")}
    stats = {k: [] for k in ("spread_before", "spread_after", "mean_cosine_displacement",
                              "unique_top1_before", "unique_top1_after")}
    generation_seconds = 0.
    steps = config.recommender.num_sampling_steps - 1
    count = config.evaluation.eval_num_samples

    def decode(initial, condition):
        if args.head == "flow":
            return model.transport(initial, condition, steps)
        times = torch.full((len(initial),), model.diffloss.train_diffusion.num_timesteps - 1,
                           device=model.device, dtype=torch.long)
        return initial - model.diffloss.net(initial, times, condition)

    with evaluation_randomness(model.device, SEED):
        for _, batch in batches(histories, targets, 64, indices):
            batch = {k: v.to(model.device) for k, v in batch.items()}
            _, condition = model.encode_history(batch, ae_grad=False)
            expanded = condition.repeat_interleave(count, 0)
            base = model.flow_prior(expanded)
            torch.cuda.synchronize(model.device)
            started = time.perf_counter()
            generated = decode(base, expanded)
            torch.cuda.synchronize(model.device)
            generation_seconds += time.perf_counter() - started
            initial = base.reshape(len(condition), count, -1)
            samples = generated.reshape_as(initial)
            zero_noise = torch.zeros(len(condition), model.latent_dim, device=model.device, dtype=condition.dtype)
            deterministic = decode(model.flow_prior(condition, zero_noise), condition)
            items = model.catalog[1:]
            predicted, _ = aggregate_diffusion_candidates(samples, items, 10, model.eval_config)
            predictions["standard32"].append(predicted.squeeze(-1).cpu().numpy())
            for key, vectors in (("single_draw", samples[:, 0]), ("mean_vector", samples.mean(1)),
                                 ("zero_noise", deterministic)):
                ids, _ = _nearest_candidates(vectors, items, 10, config.evaluation.eval_candidate_metric)
                predictions[key].append(ids.cpu().numpy())
            predictions["ranking"].append((model.ranking_logits(condition, items).topk(10).indices + 1).cpu().numpy())
            for key, values in (("before", initial), ("after", samples)):
                spread = (values - values.mean(1, keepdim=True)).square().mean((1, 2)).sqrt()
                stats[f"spread_{key}"].extend(spread.cpu().tolist())
                ids, _ = _nearest_candidates(values.flatten(0, 1), items, 1, config.evaluation.eval_candidate_metric)
                ids = ids.reshape(len(condition), count)
                stats[f"unique_top1_{key}"].extend([len(row.unique()) for row in ids])
            displacement = 1 - F.cosine_similarity(initial, samples, dim=-1).mean(1)
            stats["mean_cosine_displacement"].extend(displacement.cpu().tolist())
    output = {"run": args.directory.name, "head": args.head, "checkpoint_epoch": result["best"][args.head]["epoch"],
              "split": "validation_selection_prefix", "users": len(indices), "draws": count,
              "test_accessed": False, "generation_only_ms_per_user": generation_seconds * 1000 / len(indices),
              "metrics": {k: summarize(np.concatenate(v), targets[indices]) for k, v in predictions.items()},
              "sample_statistics": {k: float(np.mean(v)) for k, v in stats.items()},
              "note": "Decoder diagnostics are post-hoc; no model or configuration is selected using these values. Timing excludes encoding/retrieval."}
    write_json(args.directory / f"generation_probe_{args.head}.json", output)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
