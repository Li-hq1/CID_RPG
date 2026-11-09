"""Train v5_3 next-item CE and future-window controls without accessing test."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import time
import traceback

import numpy as np
import torch
from torch.utils.data import DataLoader

from joint_training.checkpoint import evaluation_randomness, load_checkpoint, read_checkpoint, save_checkpoint
from joint_training.config import parse_config
from joint_training.data import initialize_random_ae
from joint_training.multi_target_model import MultiTargetModel
from joint_training.training import JointTrainer
from misc.artifacts import array_hash, json_hash, write_json
from multi_target import FutureDataset, corpus, ranking_metrics, validation_batches
from protocol import ROOT, SELECTION_SEED, canonical_category, raw_features


def source_snapshot(directory):
    files = {}
    for folder in (ROOT, ROOT / "joint_training", ROOT / "compression", ROOT / "models", ROOT / "misc", ROOT / "scripts", ROOT / "configs"):
        candidates = folder.glob("*.py") if folder == ROOT else folder.rglob("*")
        for path in candidates:
            if not path.is_file() or path.suffix not in {".py", ".sh", ".json"} or "__pycache__" in path.parts:
                continue
            relative = path.relative_to(ROOT)
            files[str(relative)] = hashlib.sha256(path.read_bytes()).hexdigest()
            target = directory / "source" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
    write_json(directory / "source_manifest.json", {"sha256": json_hash(files), "files": files})
    return json_hash(files)


def seed_run(seed, deterministic):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic


def load_config(args):
    argv = ["--config", str(args.config)]
    for flag, value in (("--category", args.category), ("--device", args.device), ("--run-id", args.run_id)):
        if value is not None:
            argv += [flag, value]
    for value in args.set:
        argv += ["--set", value]
    if args.protocol is not None:
        argv += ["--set", "data.split=" + args.protocol, "--set", "experiment.protocol=" + args.protocol]
    if args.horizon is not None:
        argv += ["--set", "experiment.horizon=" + str(args.horizon)]
    config, _ = parse_config(argv)
    if args.epochs is not None:
        config.training.epochs = args.epochs
    if config.training.seed != 2024:
        raise ValueError("Exploration is fixed to seed2024; no seed scan is authorized.")
    if not config.experiment.enabled:
        raise ValueError("Use loo_run.py for legacy control reproduction.")
    if config.data.max_item_seq_len != 50 or config.data.sent_emb_dim != 2560:
        raise ValueError("History length=50 and raw Qwen dimension=2560 are fixed.")
    if args.check or args.smoke_test:
        config.training.max_steps = 4
        config.training.epochs = 2
        config.recommender.train_batch_size = 2
        config.flow_ranking.max_queries = 8
        config.ae.batch_size = 8
        config.ae.scheduler = config.recommender.scheduler = "constant"
        config.evaluation.eval_interval = 1
        args.eval_users = min(args.eval_users, 4)
        args.audit = "subset"
    if args.smoke_test:
        config.training.smoke_test = True
        rec = config.recommender
        rec.n_embd, rec.n_layer, rec.n_head, rec.n_inner = 32, 1, 4, 64
        rec.hstu_hidden_dim = rec.hstu_attention_dim = 8
        rec.diffloss_w, rec.diffloss_d = 32, 1
        config.evaluation.eval_num_samples = 4
    config.validate()
    return config


@torch.no_grad()
def evaluate(model, split, indices, readout=None):
    predictions, nll = [], []
    with evaluation_randomness(model.device, SELECTION_SEED):
        model.eval()
        model.refresh_catalog()
        for _, batch, targets in validation_batches(split.histories, split.targets, model.config.evaluation.eval_batch_size, indices):
            batch = {k: v.to(model.device) for k, v in batch.items()}
            logp = model.log_probabilities(batch, readout)
            predictions.append((logp.topk(10, dim=-1).indices + 1).cpu().numpy())
            relevant = torch.as_tensor(targets, device=model.device) - 1
            # NLL weights observed future interactions; ranking deduplicates targets.
            nll.append(-logp.gather(1, relevant).mean(1).cpu())
    predictions = np.concatenate(predictions)
    metrics = ranking_metrics(predictions, split.targets[indices])
    metrics["nll"] = float(torch.cat(nll).mean())
    return metrics, predictions


def execute(args, config, directory):
    started = time.time()
    torch.set_num_threads(config.training.cpu_threads)
    torch.set_float32_matmul_precision("high")
    if torch.device(config.training.device).type == "cuda":
        torch.cuda.set_per_process_memory_fraction(args.memory_fraction, device=config.training.device)
    code_hash = source_snapshot(directory)
    cache = Path(config.data.cache_dir)
    exp = config.experiment
    split = corpus(config.data.category, cache if cache.is_absolute() else ROOT / cache,
                   horizon=exp.horizon, protocol=exp.protocol, min_train_items=exp.min_train_items)
    if not len(split.targets):
        raise ValueError("No users survive the chronological split.")
    raw = raw_features(split.directory, len(split.mapping["id2item"]) - 1)
    train_ids = torch.tensor(sorted({item for seq in split.train for item in seq}))
    dataset = FutureDataset(split.train, exp.horizon)
    if not len(dataset) or (exp.arm != "ce" and not dataset.auxiliary_queries):
        raise ValueError("Insufficient training prefixes with complete future supervision.")
    source = {"category": config.data.category, "raw_sha256": array_hash(raw),
              "train_sha256": json_hash(split.train), "item_order_sha256": json_hash(split.mapping["id2item"]),
              "validation_histories_sha256": json_hash(split.histories),
              "validation_targets_sha256": array_hash(split.targets), "users_sha256": json_hash(split.users),
              "protocol": exp.protocol, "horizon": exp.horizon, "source_sha256": code_hash}
    write_json(directory / "data_audit.json", {**split.audit, "training_windows": len(dataset),
               "next_queries": dataset.next_queries, "auxiliary_queries": dataset.auxiliary_queries})
    seed_run(config.training.seed, config.training.deterministic)
    model = MultiTargetModel(config, raw, initialize_random_ae(config, raw, train_ids)).to(config.training.device)
    model.cache_features()
    generator = torch.Generator().manual_seed(config.training.seed)
    loader = DataLoader(dataset, batch_size=config.recommender.train_batch_size, shuffle=True, generator=generator)
    total_steps = min(config.training.epochs * len(loader), config.training.max_steps or config.training.epochs * len(loader))
    trainer = JointTrainer(model, config, train_ids, source, total_steps, generator)
    indices = np.random.default_rng(SELECTION_SEED).permutation(len(split.targets))[:min(args.eval_users, len(split.targets))]
    engineering = args.check or args.smoke_test or config.training.max_steps is not None
    eligible = not engineering and args.audit == "full" and config.training.epochs == 40
    flow = exp.arm in {"next_fm", "future_fm"}
    metadata = {"source": source, "arm": exp.arm, "protocol": exp.protocol,
                "engineering_check": engineering, "benchmark_eligible": eligible, "test_accessed": False,
                "selection_indices_sha256": array_hash(indices), "selection_users": len(indices),
                "selection_metric": "ndcg@10", "selection_seed": SELECTION_SEED,
                "model_reduced": args.smoke_test, "gpu_memory_fraction_limit": args.memory_fraction, "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
                "nfe_per_user": config.evaluation.eval_num_samples * exp.sampling_steps if flow else 0,
                "runtime": {"torch": str(torch.__version__), "cuda": torch.version.cuda,
                            "device": str(model.device), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}}
    write_json(directory / "experiment.json", metadata)
    log, best, steps = [], {"score": -1., "epoch": 0}, 0
    if model.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(model.device)
    for epoch in range(1, config.training.epochs + 1):
        terms = []
        for batch in loader:
            if config.training.max_steps is not None and steps >= config.training.max_steps:
                break
            batch = {key: value.to(model.device) for key, value in batch.items()}
            terms.append(trainer.step(batch, epoch))
            steps += 1
        if not terms:
            break
        row = {"epoch": epoch, "steps": steps,
               "train": {key: float(np.mean([t[key] for t in terms])) for key in terms[0]}}
        reached = config.training.max_steps is not None and steps >= config.training.max_steps
        if epoch % config.evaluation.eval_interval == 0 or epoch == config.training.epochs or reached:
            metrics, predictions = evaluate(model, split, indices)
            row["validation_selection"] = metrics
            if metrics["ndcg@10"] > best["score"]:
                best = {"score": metrics["ndcg@10"], "epoch": epoch}
                save_checkpoint(directory / "best.pt", trainer, epoch=epoch, best_epoch=epoch,
                                best_score=best["score"], stale_epochs=0)
                np.savez_compressed(directory / "selection.npz", indices=indices,
                                    targets=split.targets[indices], predictions=predictions)
        log.append(row)
        write_json(directory / "metrics.json", log)
        print(json.dumps(row), flush=True)
        if reached:
            break
    load_checkpoint(read_checkpoint(directory / "best.pt"), trainer)
    restored, predictions = evaluate(model, split, indices)
    with np.load(directory / "selection.npz") as saved:
        if not np.array_equal(saved["predictions"], predictions) or abs(restored["ndcg@10"] - best["score"]) > 1e-12:
            raise RuntimeError("Checkpoint reload changed selection predictions or score.")
    result = {**metadata, "best": best, "steps": steps, "completed_epochs": log[-1]["epoch"],
              "checkpoint_reload_verified": True, "validation_selection": restored,
              "benchmark_eligible": eligible and log[-1]["epoch"] == 40,
              "validation_includes_selection_users": True}
    audit_indices = np.arange(len(split.targets)) if args.audit == "full" else indices
    metrics, predictions = evaluate(model, split, audit_indices)
    result["validation" if args.audit == "full" else "validation_subset"] = metrics
    np.savez_compressed(directory / "audit.npz", indices=audit_indices,
                        targets=split.targets[audit_indices], predictions=predictions)
    result["diagnostics"] = {}
    for readout in (["ranking", "prior"] if flow else ["ranking"] if exp.arm == "future_ce" else []):
        diagnostic, _ = evaluate(model, split, audit_indices, readout)
        result["diagnostics"][readout] = diagnostic
    result["seconds"] = time.time() - started
    if model.device.type == "cuda":
        result["peak_memory_reserved_mib"] = torch.cuda.max_memory_reserved(model.device) / 2**20
    write_json(directory / "results.json", result)
    print(json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/final.json")
    parser.add_argument("--category")
    parser.add_argument("--device")
    parser.add_argument("--run-id")
    parser.add_argument("--protocol", choices=("leave_one_out", "future_window"))
    parser.add_argument("--horizon", type=int)
    parser.add_argument("--memory-fraction", type=float, default=.2)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--eval-users", type=int, default=4096)
    parser.add_argument("--audit", choices=("full", "subset"), default="full")
    parser.add_argument("--check", action="store_true", help="Full-width finite-update engineering check")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--set", action="append", default=[])
    args = parser.parse_args()
    if not 0 < args.memory_fraction <= 1:
        parser.error("memory-fraction must be in (0,1]")
    if args.eval_users < 1:
        parser.error("eval-users must be positive")
    config = load_config(args)
    config.data.category = canonical_category(config.data.category)
    output = Path(config.training.output_dir)
    directory = (output if output.is_absolute() else ROOT / output) / config.data.category / config.training.run_id
    directory.mkdir(parents=True, exist_ok=False)
    write_json(directory / "config.json", config.to_dict())
    try:
        execute(args, config, directory)
    except Exception:
        write_json(directory / "failure.json", {"traceback": traceback.format_exc(), "test_accessed": False})
        raise


if __name__ == "__main__":
    main()
