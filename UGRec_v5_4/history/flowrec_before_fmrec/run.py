"""Train UGRec v5.4. Default: FlowRec's three recommendation losses.

Content AE and HSTU remain fixed. Default recommendation objective is Prior CE +
5*CFM + Align CE on one random-time bridge, with deterministic history-source
Euler inference. configs/control.json retains the original marginal likelihood.
Checkpoint selection uses a fixed 4096-user validation subset. The reported
number is a later pass over the full validation split. Test labels are not read.
"""

import argparse
import hashlib
import json
import os
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from joint_training.checkpoint import evaluation_randomness, load_checkpoint, read_checkpoint, save_checkpoint
from joint_training.config import parse_config
from joint_training.data import initialize_random_ae
from joint_training.model import JointModel
from joint_training.training import JointTrainer
from misc.artifacts import array_hash, json_hash, write_json
from protocol import ROOT, SELECTION_SEED, canonical_category, corpus, ranking_metrics, raw_features, training_dataset, validation_batches


def seed_run(seed, deterministic):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic


def smoke_configuration(config):
    # History length stays at the protocol's 50: training windows and validation
    # histories are built at that length in protocol.py.
    rec = config.recommender
    rec.n_embd, rec.n_layer, rec.n_head, rec.n_inner = 32, 1, 4, 64
    rec.hstu_hidden_dim = rec.hstu_attention_dim = 8
    rec.diffloss_w, rec.diffloss_d = 32, 1
    rec.num_sampling_steps = (config.flow_ranking.steps + 1
        if config.flow_ranking.enabled and config.flow_ranking.path in {"rollout", "metric_rf", "flowrec"} else 2)
    config.validate()


def load_config(args):
    argv = ["--config", str(args.config)]
    if args.category:
        argv += ["--category", args.category]
    if args.device:
        argv += ["--device", args.device]
    if args.run_id:
        argv += ["--run-id", args.run_id]
    if args.smoke_test:
        argv.append("--smoke-test")
    for item in args.set:
        argv += ["--set", item]
    config, _ = parse_config(argv)
    if args.seed is not None:
        config.training.seed = args.seed
    if args.epochs is not None:
        config.training.epochs = args.epochs
    config.training.evaluate_test = False
    config.data.category = canonical_category(config.data.category)
    if config.data.max_item_seq_len != 50 or config.data.sent_emb_dim != 2560:
        raise ValueError("The benchmark protocol fixes history length=50 and Qwen embedding dimension=2560.")
    if args.smoke_test:
        smoke_configuration(config)
        if config.training.run_id.startswith("ugrec_v5"):
            config.training.run_id = config.training.run_id + "_smoke"
    config.validate()
    return config


@torch.no_grad()
def evaluate(model, histories, targets, indices):
    predictions, nll = [], []
    with evaluation_randomness(model.device, SELECTION_SEED):
        model.eval()
        model.refresh_catalog()
        for _, batch in validation_batches(histories, targets, model.config.evaluation.eval_batch_size, indices):
            batch = {key: value.to(model.device) for key, value in batch.items()}
            if model.config.flow_ranking.path == "flowrec":
                _, condition = model.encode_history(batch, ae_grad=False)
                query = model._sample_condition(condition)
                log_p = (torch.nn.functional.normalize(query, dim=-1) @ model.unit_catalog.T
                         / model.config.flow_ranking.temperature).log_softmax(-1)
                predictions.append((log_p.topk(10, -1).indices + 1).cpu().numpy())
                nll.append(-log_p.gather(1, batch["labels"] - 1)[:, 0].cpu())
            elif model.config.evaluation.eval_sampling_mode == "mixture":
                log_p = model.mixture_log_probabilities(batch)
                predictions.append((log_p.topk(10, -1).indices + 1).cpu().numpy())
                nll.append(-log_p.gather(1, batch["labels"] - 1)[:, 0].cpu())
            else:
                predictions.append(model.generate(batch, n_return_sequences=10).squeeze(-1).cpu().numpy())
    predictions = np.concatenate(predictions)
    metrics = ranking_metrics(predictions, targets[indices])
    if nll:
        metrics["nll"] = float(torch.cat(nll).mean())
    return metrics, predictions


def evaluate_prior(model, histories, targets, indices):
    original = model.config.evaluation.prediction_source
    try:
        model.config.evaluation.prediction_source = "prior"
        return evaluate(model, histories, targets, indices)
    finally:
        model.config.evaluation.prediction_source = original


def snapshot_source(directory):
    hashes = {}
    for folder in ("joint_training", "models", "compression", "misc", "configs", "research", "scripts", "tests", "docs"):
        for path in sorted((ROOT / folder).rglob("*")):
            if path.is_file() and path.suffix in {".py", ".json", ".sh", ".md"}:
                relative = path.relative_to(ROOT)
                hashes[str(relative)] = hashlib.sha256(path.read_bytes()).hexdigest()
                dest = directory / "source_snapshot" / relative
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, dest)
    for name in ("run.py", "protocol.py", "AGENTS.md", "README.md"):
        path = ROOT / name
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        shutil.copyfile(path, directory / "source_snapshot" / name)
    write_json(directory / "source_hashes.json", hashes)
    return json_hash(hashes)


def main():
    process_started = time.time()
    parser = argparse.ArgumentParser(description="Train UGRec v5.4 (default: FlowRec three losses).")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/final.json")
    parser.add_argument("--category")
    parser.add_argument("--device", help="Override training.device; otherwise use the resolved config.")
    parser.add_argument("--run-id")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--eval-users", type=int, default=4096)
    parser.add_argument("--audit", choices=("full", "subset"), default="full")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--set", action="append", default=[], metavar="SECTION.FIELD=VALUE")
    args = parser.parse_args()
    if args.eval_users <= 0:
        parser.error("--eval-users must be positive.")
    config = load_config(args)
    directory = Path(config.training.output_dir)
    if not directory.is_absolute():
        directory = ROOT / directory
    directory = directory / config.data.category / config.training.run_id
    directory.mkdir(parents=True, exist_ok=False)
    source_hash = snapshot_source(directory)
    write_json(directory / "config.json", config.to_dict())
    torch.set_num_threads(config.training.cpu_threads)
    torch.set_float32_matmul_precision("high")
    cache = Path(config.data.cache_dir)
    source_directory, mapping, train, histories, targets = corpus(config.data.category, cache if cache.is_absolute() else ROOT / cache)
    raw = raw_features(source_directory, len(mapping["id2item"]) - 1)
    train_ids = torch.tensor(sorted({item for seq in train for item in seq}))
    source = {
        "category": config.data.category,
        "raw_sha256": array_hash(raw),
        "train_sha256": json_hash(train),
        "item_order_sha256": json_hash(mapping["id2item"]),
        "validation_histories_sha256": json_hash(histories),
        "validation_targets_sha256": array_hash(targets),
        "input": "raw",
        "teacher_kind": None,
    }
    dataset = training_dataset(train)
    seed_run(config.training.seed, config.training.deterministic)
    model = JointModel(config, raw, initialize_random_ae(config, raw, train_ids)).to(config.training.device)
    model.cache_features()
    initial_state_hash = json_hash({name: array_hash(value.detach().cpu().numpy())
                                   for name, value in model.state_dict().items()})
    generator = torch.Generator().manual_seed(config.training.seed)
    loader = DataLoader(dataset, batch_size=config.recommender.train_batch_size, shuffle=True, generator=generator)
    budget = min(config.training.epochs * len(loader), config.training.max_steps or config.training.epochs * len(loader))
    trainer = JointTrainer(model, config, train_ids, source, budget, generator)
    write_json(directory / "config.json", config.to_dict())
    eval_users = min(args.eval_users, len(targets))
    if args.smoke_test:
        eval_users = min(eval_users, 8)
        args.audit = "subset"
    selected = np.random.default_rng(SELECTION_SEED).permutation(len(targets))[:eval_users]
    write_json(directory / "experiment.json", {
        "runtime": {"torch": torch.__version__, "cuda": torch.version.cuda,
                    "device": str(model.device),
                    "device_name": torch.cuda.get_device_name(model.device) if model.device.type == "cuda" else "cpu"},
        "scheme": config.flow_ranking.path if config.flow_ranking.enabled else "legacy",
        "source_snapshot_sha256": source_hash,
        "initial_state_sha256": initial_state_hash,
        "config": str(args.config),
        "seed": config.training.seed,
        "epochs": config.training.epochs,
        "selection_users": int(len(selected)),
        "selection_seed": SELECTION_SEED,
        "audit": args.audit,
        "test_accessed": False,
        "source": source,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "data": {"users": len(targets), "items": len(raw), "training_windows": len(dataset)},
        "selection_indices_sha256": array_hash(selected),
        "evaluation_noise_seed": SELECTION_SEED,
        "model_reduced": config.training.smoke_test,
        "benchmark_eligible": not config.training.smoke_test and config.training.max_steps is None and args.audit == "full",
        "vlad": {"clusters": config.flow_ranking.vlad_clusters,
                 "groups": config.flow_ranking.vlad_groups,
                 "expansion": config.flow_ranking.vlad_expansion,
                 "condition": config.flow_ranking.vlad_condition,
                 "condition_dim": model.condition_dim,
                 "assignment_temperature": config.flow_ranking.vlad_assignment_temperature,
                 "draws_per_component": config.flow_ranking.draws},
    })
    best = {"score": -1.0, "epoch": 0}
    log, steps, started = [], 0, time.time()
    if model.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(model.device)
    for epoch in range(1, config.training.epochs + 1):
        terms, stats = [], []
        for ids, labels in loader:
            if config.training.max_steps is not None and steps >= config.training.max_steps:
                break
            batch = {
                "input_ids": ids.to(config.training.device),
                "labels": labels.to(config.training.device),
                "attention_mask": (ids > 0).to(config.training.device),
                "seq_lens": (ids > 0).sum(-1).to(config.training.device),
            }
            terms.append(trainer.step(batch, epoch))
            stats.append(model.last_training_stats)
            steps += 1
        if not terms:
            break
        row = {
            "epoch": epoch,
            "steps": steps,
            "train": {key: float(np.mean([term[key] for term in terms])) for key in terms[0]},
            "sampled": {key: float(np.mean([stat[key] for stat in stats])) for key in stats[0]},
        }
        reached_limit = config.training.max_steps is not None and steps >= config.training.max_steps
        if epoch % config.evaluation.eval_interval == 0 or epoch == config.training.epochs or reached_limit:
            metrics, predictions = evaluate(model, histories, targets, selected)
            row["validation_selection"] = metrics
            if config.flow_ranking.path == "flowrec" and config.evaluation.prediction_source == "flow":
                prior_metrics, _ = evaluate_prior(model, histories, targets, selected)
                row["validation_selection_prior"] = prior_metrics
            if metrics["ndcg@10"] > best["score"]:
                best = {"score": metrics["ndcg@10"], "epoch": epoch}
                save_checkpoint(directory / "best_flow.pt", trainer, epoch=epoch,
                                best_epoch=epoch, best_score=best["score"], stale_epochs=0)
                np.savez_compressed(directory / "selection_flow.npz", indices=selected,
                                    targets=targets[selected], predictions=predictions)
        log.append(row)
        write_json(directory / "metrics.json", log)
        print(json.dumps(row), flush=True)
        if config.training.max_steps is not None and steps >= config.training.max_steps:
            break
    if not (directory / "best_flow.pt").is_file():
        raise RuntimeError("No validation checkpoint was saved.")
    load_checkpoint(read_checkpoint(directory / "best_flow.pt"), trainer)
    restored_metrics, restored_predictions = evaluate(model, histories, targets, selected)
    with np.load(directory / "selection_flow.npz") as saved:
        if not np.array_equal(saved["predictions"], restored_predictions):
            raise RuntimeError("Checkpoint reload changed validation predictions.")
    if abs(restored_metrics["ndcg@10"] - best["score"]) > 1e-12:
        raise RuntimeError("Checkpoint reload changed the selection score.")
    result = {
        "best": best,
        "test_accessed": False,
        "seconds": time.time() - started,
        "selection_metric": "ndcg@10 on the fixed validation subset",
        "validation_selection": restored_metrics,
        "checkpoint_reload_verified": True,
        "steps": steps,
        "completed_epochs": log[-1]["epoch"],
        "benchmark_eligible": not config.training.smoke_test and config.training.max_steps is None and args.audit == "full",
    }
    if args.audit == "full":
        metrics, predictions = evaluate(model, histories, targets, np.arange(len(targets)))
        result["validation"] = metrics
        np.savez_compressed(directory / "full_flow.npz", targets=targets, predictions=predictions)
        if config.flow_ranking.path == "flowrec" and config.evaluation.prediction_source == "flow":
            metrics, predictions = evaluate_prior(model, histories, targets, np.arange(len(targets)))
            result["validation_prior"] = metrics
            np.savez_compressed(directory / "full_prior.npz", targets=targets, predictions=predictions)
    if model.device.type == "cuda":
        result["peak_memory_allocated_mib"] = torch.cuda.max_memory_allocated(model.device) / 2**20
        result["peak_memory_reserved_mib"] = torch.cuda.max_memory_reserved(model.device) / 2**20
    result["seconds"] = time.time() - started
    result["process_seconds"] = time.time() - process_started
    write_json(directory / "results.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
