"""Train UGRec v5. Default: the final scheme in configs/final.json.

Final scheme: content AE, residual content codes on the HSTU input, one-step
flow from a history source, mixture likelihood over source draws, and mixture
decoding. configs/frozen.json and configs/matched.json reproduce earlier runs.
Checkpoint selection uses a fixed 4096-user validation subset. The reported
number is a later pass over the full validation split. Test labels are not read.
"""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from joint_training.checkpoint import evaluation_randomness, load_checkpoint, read_checkpoint, save_checkpoint
from joint_training.config import parse_config
from joint_training.data import fit_input_codes, initialize_random_ae
from joint_training.model import JointModel
from joint_training.training import JointTrainer
from misc.artifacts import array_hash, json_hash, write_json
from protocol import ROOT, SELECTION_SEED, corpus, ranking_metrics, raw_features, training_dataset, validation_batches


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
        if config.flow_ranking.enabled and config.flow_ranking.path in {"rollout", "metric_rf"} else 2)
    rec.input_code_iterations = min(rec.input_code_iterations, 2)
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
    if args.smoke_test:
        smoke_configuration(config)
        if config.training.run_id.startswith("ugrec_v5"):
            config.training.run_id = config.training.run_id + "_smoke"
    config.validate()
    return config


@torch.no_grad()
def evaluate(model, histories, targets, indices):
    predictions = []
    with evaluation_randomness(model.device, SELECTION_SEED):
        model.eval()
        model.refresh_catalog()
        for _, batch in validation_batches(histories, targets, model.config.evaluation.eval_batch_size, indices):
            batch = {key: value.to(model.device) for key, value in batch.items()}
            predictions.append(model.generate(batch, n_return_sequences=10).squeeze(-1).cpu().numpy())
    predictions = np.concatenate(predictions)
    return ranking_metrics(predictions, targets[indices]), predictions


def main():
    parser = argparse.ArgumentParser(description="Train UGRec v5 (default: the final scheme).")
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
    codes = None
    rec = config.recommender
    if rec.input_code_levels:
        codes = fit_input_codes(raw, rec.input_code_levels, rec.input_code_size, rec.input_code_iterations,
                                rec.input_code_seed, device=config.training.device)
    seed_run(config.training.seed, True)
    model = JointModel(config, raw, initialize_random_ae(config, raw, train_ids),
                       input_codes=codes).to(config.training.device)
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
        "scheme": config.flow_ranking.path if config.flow_ranking.enabled else "legacy",
        "config": str(args.config),
        "seed": config.training.seed,
        "epochs": config.training.epochs,
        "selection_users": int(len(selected)),
        "selection_seed": SELECTION_SEED,
        "audit": args.audit,
        "test_accessed": False,
        "source": source,
        "input_codes": None if codes is None else {
            "levels": int(codes.shape[1]), "size": rec.input_code_size,
            "unique_paths": int(codes.unique(dim=0).shape[0]), "sha256": array_hash(codes.numpy())},
    })
    if codes is not None:
        np.save(directory / "input_codes.npy", codes.numpy())
    best = {"score": -1.0, "epoch": 0}
    log, steps, started = [], 0, time.time()
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
        "seconds": time.time() - started,
        "selection_metric": "ndcg@10 on the fixed validation subset",
    }
    if args.audit == "full":
        load_checkpoint(read_checkpoint(directory / "best_flow.pt"), trainer)
        metrics, predictions = evaluate(model, histories, targets, np.arange(len(targets)))
        result["validation"] = metrics
        np.savez_compressed(directory / "full_flow.npz", targets=targets, predictions=predictions)
    write_json(directory / "results.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
