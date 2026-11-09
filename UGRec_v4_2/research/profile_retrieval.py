"""Compare HNSW and exact retrieval on the same generated validation vectors."""

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from joint_training.evaluation import evaluation_randomness
from joint_training.retrieval import CatalogIndex
from models.evaluation_sampling import aggregate_diffusion_candidates, _nearest_candidates
from .audit_unified import load
from .common import SEED, batches, summarize, write_json


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--users", type=int, default=1024)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("high")
    model, _, histories, targets = load(args.directory, args.device)
    model.refresh_catalog()
    items = model.catalog[1:]
    selected = np.arange(min(args.users, len(targets)))
    generated = []
    with evaluation_randomness(model.device, SEED):
        for _, batch in batches(histories, targets, 64, selected):
            batch = {k: v.to(model.device) for k, v in batch.items()}
            _, h = model.encode_history(batch, ae_grad=False)
            expanded = h.repeat_interleave(32, 0)
            generated.append(model._sample_condition(expanded).reshape(len(h), 32, -1))
    samples = torch.cat(generated)
    queries = samples.flatten(0, 1)[:4096]
    exact_neighbors, _ = _nearest_candidates(queries, items, 10, "l2")
    rows = []
    exact_predictions = None
    for ef in [None, 128, 256]:
        torch.cuda.synchronize(model.device)
        start = time.perf_counter()
        index = CatalogIndex(items, "l2", 32, ef) if ef else None
        build_seconds = time.perf_counter() - start
        if index:
            ann_neighbors, _ = index.search(queries, 10, "l2")
            neighbor_recall = float((exact_neighbors[:, :, None] == ann_neighbors[:, None, :]).any(-1).float().mean())
        else:
            neighbor_recall = 1.
        predictions = []
        torch.cuda.synchronize(model.device)
        start = time.perf_counter()
        for block in samples.split(64):
            prediction, _ = aggregate_diffusion_candidates(block, items, 10, model.eval_config,
                candidate_lookup=index.search if index else None)
            predictions.append(prediction.squeeze(-1).cpu().numpy())
        torch.cuda.synchronize(model.device)
        search_seconds = time.perf_counter() - start
        predictions = np.concatenate(predictions)
        if exact_predictions is None:
            exact_predictions = predictions
        overlap = (predictions[:, :, None] == exact_predictions[:, None, :]).any(-1).mean()
        rows.append({"backend": "exact_gpu" if ef is None else "hnsw_cpu", "ef_search": ef,
            "build_seconds": build_seconds, "retrieval_aggregation_seconds": search_seconds,
            "nearest_recall_at10": neighbor_recall, "recommendation_top10_overlap": float(overlap),
            "metrics": summarize(predictions, targets[selected])})
    result = {"validation_users": len(selected), "samples_per_user": 32, "catalog_size": len(items),
              "same_generated_vectors": True, "rows": rows,
              "note": "CPU HNSW vs GPU exact, including aggregation and transfer. Small-catalog speed is empirical, not an asymptotic guarantee."}
    write_json(args.directory / "retrieval_profile.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
