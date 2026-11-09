"""Train UGRec v5_2: history CE and independent conditional flow matching.
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
from joint_training.fm import build_model
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
    rec.n_embd, rec.n_layer, rec.n_head, rec.n_inner = 32, (2 if config.fm.enabled else 1), 4, 64
    if config.fm.enabled:
        rec.n_layer = max(rec.n_layer, config.fm.tail_layers + 1)
    rec.hstu_hidden_dim = rec.hstu_attention_dim = 8
    rec.diffloss_w, rec.diffloss_d = 32, 1
    if config.fm.enabled:
        rec.num_sampling_steps = config.fm.sampling_steps + 1
    elif config.flow_ranking.enabled and config.flow_ranking.path in {"rollout", "metric_rf"}:
        rec.num_sampling_steps = config.flow_ranking.steps + 1
    else:
        rec.num_sampling_steps = 2
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
    config.data.category = canonical_category(config.data.category)
    if args.seed is not None:
        config.training.seed = args.seed
    if args.epochs is not None:
        config.training.epochs = args.epochs
    config.training.evaluate_test = False
    if args.smoke_test:
        smoke_configuration(config)
        if config.training.run_id.startswith("ugrec_v5"):
            config.training.run_id = config.training.run_id + "_smoke"
    config.validate()
    return config


@torch.no_grad()
def evaluate(model, histories, targets, indices, source=None):
    predictions, nll = [], []
    with evaluation_randomness(model.device, SELECTION_SEED):
        model.eval()
        model.refresh_catalog()
        for _, batch in validation_batches(histories, targets, model.config.evaluation.eval_batch_size, indices):
            batch = {key: value.to(model.device) for key, value in batch.items()}
            if model.config.evaluation.eval_sampling_mode == "mixture":
                log_p = (model.mixture_log_probabilities(batch, source=source) if model.config.fm.enabled
                         else model.mixture_log_probabilities(batch))
                predictions.append((log_p.topk(10, -1).indices + 1).cpu().numpy())
                nll.append(-log_p.gather(1, batch["labels"] - 1)[:, 0].cpu())
            else:
                predictions.append(model.generate(batch, n_return_sequences=10).squeeze(-1).cpu().numpy())
    predictions = np.concatenate(predictions)
    metrics = ranking_metrics(predictions, targets[indices])
    if nll:
        metrics["nll"] = float(torch.cat(nll).mean())
    return metrics, predictions


def snapshot_source(directory):
    paths = [ROOT / name for name in ("run.py", "protocol.py")]
    for name in ("models", "joint_training", "compression", "misc", "configs", "scripts"):
        paths.extend(p for p in (ROOT / name).rglob("*")
                     if p.is_file() and p.suffix in {".py", ".json", ".sh"})
    hashes = {}
    for path in sorted(paths):
        relative = path.relative_to(ROOT)
        target = directory / "source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        hashes[str(relative)] = hashlib.sha256(target.read_bytes()).hexdigest()
    write_json(directory / "source_sha256.json", hashes)
    return json_hash(hashes)


def main():
    parser = argparse.ArgumentParser(description="Train UGRec v5_2 (default: parallel history CE + CFM).")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/final.json")
    parser.add_argument("--category")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--run-id")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--eval-users", type=int, default=4096)
    parser.add_argument("--audit", choices=("full", "subset"), default="full")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--set", action="append", default=[], metavar="SECTION.FIELD=VALUE")
    args = parser.parse_args()
    config = load_config(args)
    directory = ROOT / "outputs" / config.data.category / config.training.run_id
    directory.mkdir(parents=True, exist_ok=False)
    source_sha256 = snapshot_source(directory)
    torch.set_num_threads(config.training.cpu_threads)
    torch.set_float32_matmul_precision("high")
    source_directory, mapping, train, histories, targets = corpus(config.data.category)
    raw = raw_features(source_directory, len(mapping["id2item"]) - 1)
    train_ids = torch.tensor(sorted({item for seq in train for item in seq}))
    source = {
        "category": config.data.category,
        "raw_sha256": array_hash(raw),
        "train_sha256": json_hash(train),
        "item_order_sha256": json_hash(mapping["id2item"]),
        "input": "raw",
        "teacher_kind": None,
    }
    dataset = training_dataset(train)
    seed_run(config.training.seed, True)
    model = build_model(config, raw, initialize_random_ae(config, raw, train_ids)).to(config.training.device)
    model.cache_features()
    generator = torch.Generator().manual_seed(config.training.seed)
    loader = DataLoader(dataset, batch_size=config.recommender.train_batch_size, shuffle=True, generator=generator)
    trainer = JointTrainer(model, config, train_ids, source, config.training.epochs * len(loader), generator)
    write_json(directory / "config.json", config.to_dict())
    eval_users = min(args.eval_users, len(targets))
    if args.smoke_test:
        eval_users = min(eval_users, 8)
        args.audit = "subset"
    selected = np.random.default_rng(SELECTION_SEED).permutation(len(targets))[:eval_users]
    write_json(directory / "experiment.json", {
        "scheme": ("history_ce_only" if config.fm.mse_weight == 0 else "decoupled_cfm") if config.fm.enabled else config.flow_ranking.path if config.flow_ranking.enabled else "legacy",
        "fm": config.to_dict()["fm"],
        "config_sha256": json_hash(config.to_dict()),
        "source_sha256": source_sha256,
        "config": str(args.config),
        "seed": config.training.seed,
        "epochs": config.training.epochs,
        "selection_users": int(len(selected)),
        "selection_seed": SELECTION_SEED,
        "selection_indices_sha256": array_hash(selected),
        "training_windows": len(dataset),
        "expected_updates": config.training.epochs * len(loader),
        "engineering_check": config.training.smoke_test or config.training.max_steps is not None,
        "audit": args.audit,
        "test_accessed": False,
        "source": source,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "vlad": {"clusters": config.flow_ranking.vlad_clusters,
                 "groups": config.flow_ranking.vlad_groups,
                 "expansion": config.flow_ranking.vlad_expansion,
                 "condition": config.flow_ranking.vlad_condition,
                 "condition_dim": model.condition_dim,
                 "assignment_temperature": config.flow_ranking.vlad_assignment_temperature,
                 "draws_per_component": config.flow_ranking.draws},
    })
    best = {"score": -1.0, "epoch": 0}
    best_ranking = {"score": -1.0, "epoch": 0}
    track_ranking = config.fm.enabled and config.fm.mse_weight > 0
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
        if epoch % config.evaluation.eval_interval == 0 or epoch == config.training.epochs:
            metrics, predictions = evaluate(model, histories, targets, selected)
            row["validation_selection"] = metrics
            if metrics["ndcg@10"] > best["score"]:
                best = {"score": metrics["ndcg@10"], "epoch": epoch}
                save_checkpoint(directory / "best_flow.pt", trainer, epoch=epoch,
                                best_epoch=epoch, best_score=best["score"], stale_epochs=0)
                np.savez_compressed(directory / "selection_flow.npz", indices=selected,
                                    targets=targets[selected], predictions=predictions)
            if track_ranking:
                ranking_metrics, ranking_predictions = evaluate(model, histories, targets, selected, source="ranking")
                row["validation_ranking_selection"] = ranking_metrics
                if ranking_metrics["ndcg@10"] > best_ranking["score"]:
                    best_ranking = {"score": ranking_metrics["ndcg@10"], "epoch": epoch}
                    save_checkpoint(directory / "best_ranking.pt", trainer, epoch=epoch,
                                    best_epoch=epoch, best_score=best_ranking["score"], stale_epochs=0)
                    np.savez_compressed(directory / "selection_ranking.npz", indices=selected,
                                        targets=targets[selected], predictions=ranking_predictions)
        log.append(row)
        write_json(directory / "metrics.json", log)
        print(json.dumps(row), flush=True)
        if config.training.max_steps is not None and steps >= config.training.max_steps:
            break
    if not (directory / "best_flow.pt").is_file():
        raise RuntimeError("No validation checkpoint was saved.")
    result = {
        "best": best,
        "test_accessed": False,
        "training_seconds": time.time() - started,
        "selection_metric": "ndcg@10 on the fixed validation subset",
    }
    load_checkpoint(read_checkpoint(directory / "best_flow.pt"), trainer)
    reloaded_metrics, reloaded_predictions = evaluate(model, histories, targets, selected)
    with np.load(directory / "selection_flow.npz") as saved:
        if not np.array_equal(reloaded_predictions, saved["predictions"]):
            raise RuntimeError("Checkpoint reload changed validation predictions.")
    if reloaded_metrics["ndcg@10"] != best["score"]:
        raise RuntimeError("Checkpoint reload changed the selection score.")
    result["checkpoint_reload_verified"] = True
    result["actual_updates"] = steps
    result["engineering_check"] = config.training.smoke_test or config.training.max_steps is not None
    if args.audit == "full":
        metrics, predictions = evaluate(model, histories, targets, np.arange(len(targets)))
        result["validation"] = metrics
        np.savez_compressed(directory / "full_flow.npz", targets=targets, predictions=predictions)
    if config.fm.enabled:
        audit_indices = np.arange(len(targets)) if args.audit == "full" else selected
        result["diagnostics"] = {}
        for source in ("ranking", "prior"):
            metrics, predictions = evaluate(model, histories, targets, audit_indices, source=source)
            result["diagnostics"][source] = metrics
            np.savez_compressed(directory / f"diagnostic_{source}.npz", indices=audit_indices,
                                targets=targets[audit_indices], predictions=predictions)
        prediction_source = config.evaluation.prediction_source
        draws = 1 if prediction_source == "ranking" else config.evaluation.eval_num_samples
        nfe = (0 if prediction_source in {"ranking", "prior"} else
               1 if prediction_source == "one_step" else config.fm.sampling_steps)
        result["inference_budget"] = {"draws": draws, "velocity_nfe_per_draw": nfe,
                                      "total_velocity_queries_per_user": draws * nfe}
        if track_ranking:
            load_checkpoint(read_checkpoint(directory / "best_ranking.pt"), trainer)
            rank_metrics, rank_predictions = evaluate(model, histories, targets, selected, source="ranking")
            with np.load(directory / "selection_ranking.npz") as saved:
                if not np.array_equal(rank_predictions, saved["predictions"]) or rank_metrics["ndcg@10"] != best_ranking["score"]:
                    raise RuntimeError("Ranking checkpoint reload changed validation predictions or score.")
            result["ranking_best"] = best_ranking
            result["ranking_checkpoint_reload_verified"] = True
            rank_metrics, rank_predictions = evaluate(model, histories, targets, audit_indices, source="ranking")
            result["ranking_selected_validation"] = rank_metrics
            np.savez_compressed(directory / "ranking_selected.npz", indices=audit_indices,
                                targets=targets[audit_indices], predictions=rank_predictions)
            load_checkpoint(read_checkpoint(directory / "best_flow.pt"), trainer)
    if model.device.type == "cuda":
        result["peak_memory_allocated_mib"] = torch.cuda.max_memory_allocated(model.device) / 2**20
        result["peak_memory_reserved_mib"] = torch.cuda.max_memory_reserved(model.device) / 2**20
    result["seconds"] = time.time() - started
    write_json(directory / "results.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
