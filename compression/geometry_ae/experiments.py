"""Fixed-budget ablations, full training and paired recommendation experiments."""

import argparse
from copy import deepcopy
from dataclasses import replace
import json
import logging
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch
from torch.nn import functional as F
from accelerate import Accelerator

from config import CONFIG
from data.amazon_reviews import AmazonReviews2014
from data.embeddings import load_item_sentence_embeddings
from .cache import json_hash, read_manifest, source_metadata, training_item_mask, write_json
from .config import AEConfig
from .metrics import geometry_metrics, neighbor_comparison
from .model import relation_kl
from .pipeline import (batch_indices, fit_input_statistics, fit_pca, pca_features, split_items,
                       train_artifact)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def ablation_configs(config):
    base = replace(config, cosine_weight=0., variance_weight=0., correlation_weight=0., relation_weight=0.)
    return {
        "linear_reconstruction": replace(base, architecture="linear"),
        "mlp_reconstruction": replace(base, architecture="mlp"),
        "mlp_cosine": replace(base, cosine_weight=.1),
        "mlp_statistics": replace(base, cosine_weight=.1, variance_weight=.1, correlation_weight=1.),
        "mlp_relations": replace(base, cosine_weight=.1, variance_weight=.1, correlation_weight=1., relation_weight=.1),
    }


def baseline_metrics(x, rec, z, scale, config):
    mse = float(np.mean((x - rec) ** 2))
    cosine = float(F.cosine_similarity(torch.from_numpy(rec), torch.from_numpy(x), dim=1).mean())
    relation, rows = 0., 0
    indices_batches = (batch_indices(np.arange(len(x)), config.eval_batch_size)
                       if config.relation_mode != "none" else [])
    for indices in indices_batches:
        if len(indices) > 1:
            relation += float(relation_kl(torch.from_numpy(x[indices]), torch.from_numpy(z[indices]),
                                         config.relation_temperature)) * len(indices)
            rows += len(indices)
    relation /= max(rows, 1)
    nmse = mse / float(scale.square())
    return {"mse": mse, "normalized_mse": nmse, "cosine_similarity": cosine,
            "relation_kl": relation, "score": nmse + .1 * (1 - cosine) + .1 * relation}


def run_ablation(embeddings, dataset, data_config, mask, config, *, device="cpu", force=False):
    source = source_metadata(embeddings, dataset, data_config, mask)
    configs = ablation_configs(config)
    report_id = json_hash({"source": source, "candidates": {k: v.to_dict() for k, v in configs.items()}})[:20]
    directory = Path(dataset.cache_dir) / "Geometry_AE" / "experiments" / report_id
    splits = split_items(mask, config)
    tr, valid = splits["train"], splits["valid"]
    mean, scale = fit_input_statistics(embeddings, tr)
    pca = fit_pca(embeddings, tr, config)
    pc = pca_features(pca, embeddings, tr)
    x = embeddings[valid]
    rec = pca.inverse_transform(pca.transform(x))
    report = {
        "source": source, "selection": "validation score only; test excluded",
        "split_sizes": {k: len(v) for k, v in splits.items()},
        "mean": baseline_metrics(x, np.broadcast_to(mean.numpy(), x.shape).copy(),
                                 np.zeros((len(x), config.latent_dim), dtype=np.float32), scale, config),
        "pca": {**baseline_metrics(x, rec, pc[valid], scale, config),
                "geometry": geometry_metrics(pc[tr]),
                "neighbors": neighbor_comparison(x, pc[valid], pc[valid], config)},
        "candidates": {},
    }
    for name, settings in configs.items():
        artifact = train_artifact(embeddings, dataset, data_config, mask, settings, device=device, force=force)
        metrics = json.loads((artifact / "metrics.json").read_text())
        report["candidates"][name] = {"artifact": str(artifact.resolve()), "config": settings.to_dict(),
                                      "valid": metrics["valid"], "geometry": metrics["geometry"],
                                      "neighbors": metrics["neighbors"]["valid"]}
        write_json(directory / "ablation.json", report)
    report["winner"] = min(report["candidates"], key=lambda key: report["candidates"][key]["valid"]["score"])
    report["winner_config"] = configs[report["winner"]].to_dict()
    write_json(directory / "ablation.json", report)
    return directory, report


def recommendation_commands(artifact, category, cache_dir, ckpt_dir, prefix):
    commands = []
    for seed in (2024, 2025, 2026):
        for method in ("pca", "geometry_ae"):
            run_id = f"{prefix}_{method}_{seed}"
            command = [sys.executable, str(PROJECT_ROOT / "train.py"), "--config=config.py",
                       f"--category={category}", f"--cache_dir={cache_dir}", f"--ckpt_dir={ckpt_dir}",
                       f"--run_id={run_id}", f"--rand_seed={seed}", "--epochs=100", "--patience=100",
                       "--encoder_type=hstu", "--supervision_mode=all_positions", "--sent_emb_pca=128",
                       "--sent_emb_model=Qwen/Qwen3-Embedding-4B", "--sent_emb_dim=2560",
                       "--sent_emb_normalize=true", "--sent_emb_dtype=auto", "--sent_emb_max_length=512",
                       "--eval_sampling_mode=knn", "--tiger_knn_metric=cosine", "--use_rectified_flow=true",
                       f"--sent_emb_compression={method}"]
            if method == "geometry_ae":
                command.append(f"--geometry_ae_cache={artifact}")
            commands.append((method, seed, run_id, command))
    return commands


def summarize_recommendation(records):
    summary = {}
    for method in ("pca", "geometry_ae"):
        results = [r["test_results"] for r in records if r["method"] == method]
        summary[method] = {metric: {"mean": float(np.mean([r[metric] for r in results])),
                                    "std": float(np.std([r[metric] for r in results]))}
                           for metric in ("ndcg@10", "recall@10")}
    summary["accepted"] = all(summary["geometry_ae"][metric]["mean"] >= summary["pca"][metric]["mean"]
                              for metric in ("ndcg@10", "recall@10"))
    summary["recommended_default"] = "geometry_ae" if summary["accepted"] else "pca"
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--category", default="Beauty")
    parser.add_argument("--cache_dir", default="cache/")
    parser.add_argument("--ckpt_dir", default="ckpt/")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--full", action="store_true", help="Train winner on full compression training split.")
    parser.add_argument("--recommendation", action="store_true", help="Also run six paired recommendation jobs.")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    if args.recommendation and (not args.full or args.smoke_test):
        parser.error("Recommendation comparisons require --full and cannot use smoke artifacts.")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; use --device=cpu for smoke checks.")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if device.type == "cpu":
        torch.set_num_threads(min(torch.get_num_threads(), 4))
    accelerator = Accelerator(cpu=device.type == "cpu")
    if accelerator.num_processes != 1:
        raise RuntimeError("Geometry AE experiments require one process.")
    data_config = deepcopy(CONFIG)
    data_config.update(category=args.category, cache_dir=args.cache_dir, dataset="AmazonReviews2014",
                       device=device, accelerator=accelerator, sent_emb_model="Qwen/Qwen3-Embedding-4B",
                       sent_emb_dim=2560, sent_emb_normalize=True, sent_emb_dtype="auto", sent_emb_max_length=512)
    try:
        dataset = AmazonReviews2014(data_config)
        mask = training_item_mask(dataset)
        embeddings = load_item_sentence_embeddings(dataset, data_config)
        config = AEConfig(train_items=4096, epochs=100, patience=100)
        if args.smoke_test:
            config = replace(config, smoke_items=64, train_items=None, epochs=2,
                             train_batch_size=16, eval_batch_size=64)
        directory, report = run_ablation(embeddings, dataset, data_config, mask, config,
                                        device=device, force=args.force)
        print(f"Ablation report: {directory / 'ablation.json'}; winner: {report['winner']}")
        if args.full:
            winner = replace(AEConfig(**report["winner_config"]), train_items=None, epochs=200, patience=30)
            artifact = train_artifact(embeddings, dataset, data_config, mask, winner, device=device, force=args.force)
            report["full_artifact"] = str(artifact.resolve())
            write_json(directory / "ablation.json", report)
    finally:
        accelerator.end_training()
    if args.recommendation:
        records = []
        identity = read_manifest(artifact)["artifact_id"]
        for method, seed, run_id, command in recommendation_commands(
                artifact.resolve(), args.category, args.cache_dir, args.ckpt_dir, f"geometry_{directory.name}"):
            result_path = Path(args.ckpt_dir) / "AmazonReviews2014" / "TigerDiff" / args.category / run_id / "results.json"
            if not result_path.exists():
                if result_path.parent.exists():
                    raise RuntimeError(f"Incomplete recommendation run at {result_path.parent}; resume it before retrying.")
                subprocess.run(command, cwd=PROJECT_ROOT, check=True)
            result = json.loads(result_path.read_text())
            expected_identity = identity if method == "geometry_ae" else None
            if (result.get("sent_emb_compression") != method or result.get("rand_seed") != seed
                    or result.get("feature_artifact_id") != expected_identity):
                raise ValueError(f"Cached recommendation result does not match current experiment: {result_path}")
            records.append({"method": method, "seed": seed, "run_id": run_id,
                            "test_results": result["test_results"]})
            write_json(directory / "recommendation.json", {"records": records})
        write_json(directory / "recommendation.json", {"records": records,
                                                       "summary": summarize_recommendation(records)})


if __name__ == "__main__":
    main()
