"""Change support temperatures while reusing exactly the same generated validation draws."""

import argparse

import numpy as np
import torch

from models.evaluation_sampling import _distances, _nearest_candidates
from .common import ROOT, load_joint, paired_interval, per_user, summarize, write_json


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--device", default="cuda:7")
    args = parser.parse_args()
    torch.set_num_threads(2)
    directory = ROOT / "outputs/research_20261028/checkpoints" / args.name
    import json
    result = json.loads((directory / "results.json").read_text())
    model, _, histories, _ = load_joint(result["checkpoint"], args.device)
    data = np.load(directory / "generated_samples.npz")
    baseline = np.load(directory / "paired_predictions.npz")["support_k1"]
    temperatures = [.25, 1., 4., 16., 64., 1e6]
    outputs = {str(t): [] for t in temperatures}
    filtered = []
    for user, samples in enumerate(data["samples"]):
        samples = torch.tensor(samples, device=args.device)
        catalog = model.catalog[1:]
        nearest_ids, nearest_distances = _nearest_candidates(samples, catalog, 10, "l2")
        depth = 1
        while nearest_ids[:, :depth].unique().numel() < 10:
            depth = min(10, max(depth + 1, depth * 2))
        ids = nearest_ids[:, :depth]
        candidates, inverse = ids.flatten().unique(sorted=True, return_inverse=True)
        min_distances = _distances(samples, catalog[candidates - 1], "l2").amin(0)
        distance_order = min_distances.argsort(stable=True)
        for temperature in temperatures:
            weights = torch.full((32, len(candidates)), -torch.inf,
                                 dtype=torch.float64, device=args.device)
            weights.scatter_(1, inverse.reshape_as(ids), -nearest_distances[:, :depth].double() / temperature)
            support = torch.logsumexp(weights, dim=0)
            order = distance_order[support[distance_order].argsort(descending=True, stable=True)]
            outputs[str(temperature)].append(candidates[order[:10]].cpu().numpy())
        # Masking is applied at retrieval, preserving candidate expansion and aggregation.
        masked = catalog.clone()
        masked[torch.tensor(histories[data["indices"][user]], device=args.device) - 1] = 1e6
        from models.evaluation_sampling import aggregate_diffusion_candidates
        prediction, _ = aggregate_diffusion_candidates(
            samples[None], masked, 10, {"eval_candidates_per_sample": 1,
                                       "eval_candidate_ranking": "support",
                                       "eval_candidate_metric": "l2",
                                       "eval_support_temperature": 1.})
        filtered.append(prediction.flatten().cpu().numpy())
        if user % 512 == 0:
            print(f"reranking {user}/{len(data['samples'])}", flush=True)
    outputs = {name: np.stack(value) for name, value in outputs.items()}
    # This assertion makes any changed sampling, catalog, tie handling or distance implementation visible.
    np.testing.assert_array_equal(outputs["1.0"], baseline)
    outputs["mask_history_t1"] = np.stack(filtered)
    targets = data["targets"]
    scores = {name: summarize(prediction, targets) for name, prediction in outputs.items()}
    intervals = {name: paired_interval(per_user(prediction, targets)["ndcg@10"],
                                       per_user(baseline, targets)["ndcg@10"])
                 for name, prediction in outputs.items()}
    result = {"split": "validation", "users": len(targets), "same_draws_verified": True,
              "scores": scores, "paired_ndcg10_against_temperature1": intervals,
              "note": "Temperature grid is exploratory validation tuning; it is not a test result."}
    write_json(directory / "reranking.json", result)
    np.savez_compressed(directory / "reranking_predictions.npz", targets=targets, **outputs)
    print(result, flush=True)


if __name__ == "__main__":
    main()
