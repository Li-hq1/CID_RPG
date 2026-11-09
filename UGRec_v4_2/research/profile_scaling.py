"""Isolated candidate-training cost, with equal query count and AE width.

This is a computation benchmark, not a recommendation accuracy experiment.
Run on an otherwise idle device; does not extrapolate accuracy to larger data.
"""

import argparse
import gc
import json
import time

import torch
from torch.nn import functional as F

from compression.geometry_ae.model import GeometryAE
from joint_training.config import JointConfig
from joint_training.flow_ranking import ranking_loss
from .common import ROOT, write_json


def measure(ae, features, queries, positives, sampled, repeats, negatives):
    device = features.device
    ids = sampled if sampled is not None else torch.arange(len(features), device=device)
    labels = torch.searchsorted(ids, positives) if sampled is not None else positives
    cfg = JointConfig().flow_ranking
    cfg.sampling = "uniform_corrected"
    negative_columns = torch.randint(len(ids), (negatives,), device=device)
    def step():
        ae.zero_grad(set_to_none=True)
        queries.grad = None
        keys = F.layer_norm(ae.latent(features[ids]), (128,), eps=1e-6)
        loss = (F.cross_entropy(F.normalize(queries, dim=-1) @ F.normalize(keys, dim=-1).T / .07, labels)
                if sampled is None else ranking_loss(queries, keys, labels, cfg, negative_columns, len(features)))
        loss.backward()
    for _ in range(3):
        step()
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    for _ in range(repeats):
        step()
    torch.cuda.synchronize(device)
    return {"milliseconds": (time.perf_counter() - started) * 1000 / repeats,
            "peak_allocated_mb": torch.cuda.max_memory_allocated(device) / 2**20,
            "encoded_candidates": len(ids)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--negatives", type=int, default=1024)
    parser.add_argument("--sizes", type=int, nargs="+", default=[12101, 50000, 100000])
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("high")
    torch.manual_seed(2024)
    cfg = JointConfig().ae
    cfg.hidden_dims = (512, 256)
    rows = []
    for size in args.sizes:
        features = torch.randn(size, 2560, device=args.device)
        queries = torch.randn(1024, 128, device=args.device, requires_grad=True)
        positives = torch.randint(size, (1024,), device=args.device)
        sampled = torch.cat((positives, torch.randint(size, (args.negatives,), device=args.device))).unique(sorted=True)
        ae = GeometryAE(2560, cfg).to(args.device)
        for name, ids in (("full_catalog", None), ("sampled", sampled)):
            rows.append({"catalog_size": size, "method": name,
                         **measure(ae, features, queries, positives, ids, args.repeats, args.negatives)})
        del features, queries, positives, sampled, ae
        gc.collect()
        torch.cuda.empty_cache()
    result = {"device": torch.cuda.get_device_name(torch.device(args.device)), "queries": 1024,
              "raw_dim": 2560, "latent_dim": 128, "teacher": False, "uniform_negatives": args.negatives,
              "note": "Candidate AE forward + cosine matching + CE backward only; feature storage still O(N). Random synthetic data, no accuracy claim.",
              "rows": rows}
    write_json(ROOT / "outputs/unified_20261028/profile_scaling.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
