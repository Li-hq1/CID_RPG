"""Validation-only v4_2 experiments: RF + exactly one sampled ranking loss."""

import argparse
import hashlib
import json
import io
from pathlib import Path
import platform
import tarfile
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from joint_training.config import JointConfig
from joint_training.data import initialize_random_ae
from joint_training.model import JointModel
from joint_training.training import JointTrainer
from joint_training.evaluation import evaluation_randomness
from joint_training.__main__ import seed_run
from joint_training.checkpoint import save_checkpoint, read_checkpoint, load_checkpoint
from misc.artifacts import array_hash, json_hash
from .common import ROOT, SEED, batches, corpus, raw_features, summarize, write_json
from .compact_ablation import prepare


OUT = ROOT / "outputs/unified_20261028"


@torch.no_grad()
def evaluate(model, histories, targets, indices, source="flow"):
    predictions = []
    previous = model.config.evaluation.prediction_source
    with evaluation_randomness(model.device, SEED):
        model.eval()
        model.refresh_catalog()
        model.config.evaluation.prediction_source = source
        try:
            for _, batch in batches(histories, targets, 64, indices):
                batch = {key: value.to(model.device) for key, value in batch.items()}
                predictions.append(model.generate(batch, n_return_sequences=10).squeeze(-1).cpu().numpy())
        finally:
            model.config.evaluation.prediction_source = previous
    predictions = np.concatenate(predictions)
    return summarize(predictions, targets[indices]), predictions


def configuration(args):
    config = JointConfig()
    rec = config.recommender
    rec.encoder_type, rec.n_embd, rec.n_layer, rec.n_inner = "transformer", 128, 2, 256
    rec.layer_norm_epsilon = 1e-6
    rec.hstu_hidden_dim = rec.hstu_attention_dim = 32
    rec.diffloss_w, rec.diffloss_d = 256, 2
    rec.tiger_uncond_prob = rec.tiger_guidance_weight = 0.
    rec.lr = 1e-3
    config.ae.hidden_dims, config.ae.lr = (512, 256), 1e-4
    config.flow_ranking.enabled = True
    config.evaluation.eval_batched_retrieval = True
    if args.config:
        values = config.to_dict()
        for key, patch in json.loads(args.config.read_text()).items():
            values[key].update(patch)
        # Steps must be derived before validation of the partial patch.
        f = values["flow_ranking"]
        values["recommender"]["num_sampling_steps"] = (f["steps"] if f["path"] in {"rollout", "metric_rf"} else 1) + 1
        config = JointConfig.from_dict(values)
    config.recommender.num_sampling_steps = (config.flow_ranking.steps if config.flow_ranking.path in {"rollout", "metric_rf"} else 1) + 1
    config.training.device, config.training.seed = args.device, args.seed
    config.training.epochs, config.training.evaluate_test = args.epochs, False
    config.data.category = args.category
    assert config.ranking.teacher_weight == 0, "No teachers in v4_2"
    config.validate()
    return config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--category", default="Beauty")
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--eval-users", type=int, default=4096)
    parser.add_argument("--audit", choices=("subset", "full"), default="subset")
    args = parser.parse_args()
    directory = OUT / args.category / f"{args.name}_s{args.seed}_e{args.epochs}"
    directory.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("high")
    config = configuration(args)
    config.training.run_id = directory.name
    config.training.output_dir = str(OUT)
    content, dataset, histories, targets = prepare(args.category)
    source_directory, mapping, train, _, _ = corpus(args.category)
    raw = raw_features(source_directory, len(content))
    train_ids = torch.tensor(sorted({i for seq in train for i in seq}))
    source = {"category": args.category, "raw_sha256": array_hash(raw),
              "train_sha256": json_hash(train), "item_order_sha256": json_hash(mapping["id2item"]),
              "input": "raw", "teacher_kind": None}
    seed_run(args.seed, True)
    model = JointModel(config, raw, initialize_random_ae(config, raw, train_ids)).to(args.device)
    model.cache_features()
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(dataset, batch_size=config.recommender.train_batch_size,
                        shuffle=True, generator=generator)
    trainer = JointTrainer(model, config, train_ids, source, args.epochs * len(loader), generator)
    write_json(directory / "config.json", config.to_dict())
    code_files = sorted(p for name in ["joint_training", "models", "compression", "data", "misc", "engine", "research"]
                        for p in (ROOT / name).rglob("*.py"))
    code = {str(p.relative_to(ROOT)): p.read_bytes() for p in code_files}
    hashes = {name: hashlib.sha256(value).hexdigest() for name, value in code.items()}
    with tarfile.open(directory / "source.tar.gz", "w:gz") as archive:
        for name, value in code.items():
            member = tarfile.TarInfo(name)
            member.size = len(value)
            archive.addfile(member, io.BytesIO(value))
    write_json(directory / "experiment.json", {"args": vars(args) | {"config": str(args.config)},
               "source": source, "test_accessed": False, "selection_users": args.eval_users,
               "matmul_precision": "high", "code_sha256": hashes,
               "environment": {"python": platform.python_version(), "torch": torch.__version__,
                               "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(model.device)},
               "supervision": "RF + single sampled rank + local AE; direct is a non-generative control"})
    selected = np.random.default_rng(SEED).permutation(len(targets))[:args.eval_users]
    best = {"score": -1., "epoch": 0}
    log, training_seconds = [], 0.
    started = time.time()
    torch.cuda.reset_peak_memory_stats(model.device) if model.device.type == "cuda" else None
    for epoch in range(1, args.epochs + 1):
        begin, terms, stats = time.time(), [], []
        for ids, labels in loader:
            batch = {"input_ids": ids.to(args.device), "labels": labels.to(args.device),
                     "attention_mask": (ids > 0).to(args.device), "seq_lens": (ids > 0).sum(-1).to(args.device)}
            terms.append(trainer.step(batch, epoch))
            stats.append(model.last_training_stats)
        if model.device.type == "cuda":
            torch.cuda.synchronize(model.device)
        elapsed = time.time() - begin
        training_seconds += elapsed
        row = {"epoch": epoch, "seconds": time.time() - started, "training_seconds": elapsed,
               "train": {key: float(np.mean([v[key] for v in terms])) for key in terms[0]},
               "sampled": {key: float(np.mean([v[key] for v in stats])) for key in stats[0]}}
        if epoch % config.evaluation.eval_interval == 0 or epoch == args.epochs:
            metrics, predictions = evaluate(model, histories, targets, selected)
            row["validation"] = metrics
            if metrics["ndcg@10"] > best["score"]:
                best = {"score": metrics["ndcg@10"], "epoch": epoch}
                save_checkpoint(directory / "best_flow.pt", trainer, epoch=epoch,
                                best_epoch=epoch, best_score=best["score"], stale_epochs=0)
                np.savez_compressed(directory / "selection_flow.npz", indices=selected,
                                    targets=targets[selected], predictions=predictions)
        log.append(row)
        write_json(directory / "metrics.json", log)
        print(json.dumps(row), flush=True)
    result = {"config": config.to_dict(), "source": source, "best": best, "audits": {},
              "test_accessed": False, "seconds_training": training_seconds,
              "peak_gpu_allocated_mb": torch.cuda.max_memory_allocated(model.device) / 2**20 if model.device.type == "cuda" else None,
              "seconds_training_and_selection": time.time() - started,
              "note": "Validation only. Fixed 4096-user selection subset; full audit includes it."}
    if args.audit == "full":
        load_checkpoint(read_checkpoint(directory / "best_flow.pt"), trainer)
        metrics, predictions = evaluate(model, histories, targets, np.arange(len(targets)))
        result["audits"]["flow"] = metrics
        np.savez_compressed(directory / "full_flow.npz", targets=targets, predictions=predictions)
    result["seconds_total"] = time.time() - started
    write_json(directory / "results.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
