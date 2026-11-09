"""Validation-only experiments using the actual JointModel and JointTrainer.

Both generative and auxiliary ranking checkpoints are selected independently.
All reported full validation audits state which selection rule produced them.
"""

import argparse
import hashlib
import json
from pathlib import Path
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
from joint_training.checkpoint import save_checkpoint, read_checkpoint, load_checkpoint, random_state, restore_random_state
from misc.artifacts import array_hash, json_hash
from .common import ROOT, SEED, batches, corpus, raw_features, summarize, write_json
from .compact_ablation import prepare


OUT = ROOT / "outputs/iterative_20261028"


def fixed_teacher(category, content, kind):
    if kind == "content":
        return content
    from scipy import sparse
    from sklearn.decomposition import TruncatedSVD
    _, _, train, _, _ = corpus(category)
    rows, cols = zip(*[(u, i - 1) for u, seq in enumerate(train) for i in set(seq)])
    matrix = sparse.csr_matrix((np.ones(len(rows), np.float32), (rows, cols)),
                              shape=(len(train), len(content)))
    counts = np.asarray(matrix.sum(0)).ravel()
    graph = (matrix.T @ matrix).tocsr()
    graph.setdiag(0)
    graph.eliminate_zeros()
    inv = sparse.diags(1 / np.sqrt(counts.clip(1)))
    graph = (inv @ graph @ inv).tocsr()
    cache = OUT / "teachers" / f"{category}_behavior128.npy"
    if cache.exists():
        values = np.load(cache)
    else:
        values = TruncatedSVD(128, random_state=2024, n_iter=7).fit_transform(graph).astype(np.float32)
        cache.parent.mkdir(parents=True, exist_ok=True)
        values[np.linalg.norm(values, axis=-1) < 1e-9] = content.numpy()[np.linalg.norm(values, axis=-1) < 1e-9]
        np.save(cache, values)
    behavior = torch.from_numpy(values)
    if kind == "behavior":
        return behavior
    return torch.cat((torch.nn.functional.normalize(content, dim=-1),
                      torch.nn.functional.normalize(behavior, dim=-1)), dim=-1)


@torch.no_grad()
def evaluate(model, histories, targets, indices, flow=True, only=None):
    names = (["flow"] if flow else []) + (["ranking"] if model.config.ranking.weight > 0 else [])
    if model.config.ranking.endpoint_weight > 0:
        names.insert(1, "one_step")
    if only is not None:
        names = [only]
    values = {key: [] for key in names}
    reference = next((name for name in names if name in {"flow", "one_step"}), None)
    control = reference is not None and model.config.ranking.prior_mode == "history"
    if control:
        values["prior_control"] = []
    previous = model.config.evaluation.prediction_source
    with evaluation_randomness(model.device, SEED):
        model.eval()
        model.refresh_catalog()
        for ix, batch in batches(histories, targets, 64, indices):
            batch = {key: value.to(model.device) for key, value in batch.items()}
            base_state = random_state(model.device)
            for name in names:
                if name in {"flow", "one_step"}:
                    restore_random_state(base_state, model.device)
                model.config.evaluation.prediction_source = name
                before = random_state(model.device) if control and name == reference else None
                prediction = model.generate(batch, n_return_sequences=10)
                values[name].append(prediction.squeeze(-1).cpu().numpy())
                if before is not None:
                    after = random_state(model.device)
                    try:
                        restore_random_state(before, model.device)
                        model.config.evaluation.prediction_source = "prior"
                        prior = model.generate(batch, n_return_sequences=10)
                        values["prior_control"].append(prior.squeeze(-1).cpu().numpy())
                    finally:
                        restore_random_state(after, model.device)
    model.config.evaluation.prediction_source = previous
    predictions = {key: np.concatenate(value) for key, value in values.items()}
    return {key: summarize(value, targets[indices]) for key, value in predictions.items()}, predictions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--category", default="Beauty")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--teacher", choices=("content", "behavior", "mixed"), default="content")
    parser.add_argument("--input", choices=("raw", "pca"), default="raw")
    parser.add_argument("--eval-users", type=int, default=4096)
    parser.add_argument("--audit", choices=("subset", "full"), default="subset")
    args = parser.parse_args()
    directory = OUT / args.category / f"{args.name}_s{args.seed}_e{args.epochs}"
    directory.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("high")
    config = JointConfig()
    config.ranking.enabled = True
    config.recommender.encoder_type = "transformer"
    config.recommender.n_embd = 128
    config.recommender.n_layer = 2
    config.recommender.n_inner = 256
    config.recommender.layer_norm_epsilon = 1e-6
    config.recommender.hstu_hidden_dim = config.recommender.hstu_attention_dim = 32
    config.recommender.diffloss_w = 256
    config.recommender.diffloss_d = 2
    config.recommender.tiger_uncond_prob = 0.
    config.recommender.tiger_guidance_weight = 0.
    config.recommender.lr = 1e-3
    config.ae.hidden_dims = (512, 256) if args.input == "raw" else (256,)
    config.ae.lr = 1e-4
    config.training.epochs = args.epochs
    config.training.evaluate_test = False
    config.training.device = args.device
    config.training.seed = args.seed
    config.training.run_id = directory.name
    config.evaluation.eval_batched_retrieval = True
    config.data.category = args.category
    if args.config:
        patch = json.loads(args.config.read_text())
        base = config.to_dict()
        for section, values in patch.items():
            base[section].update(values)
        config = JointConfig.from_dict(base)
    config.training.device, config.training.seed = args.device, args.seed
    config.training.epochs, config.training.evaluate_test = args.epochs, False
    config.data.category = args.category
    if args.teacher == "mixed":
        config.ranking.teacher_dim = 256
    config.validate()
    content, dataset, histories, targets = prepare(args.category)
    source_directory, mapping, train, _, _ = corpus(args.category)
    raw = raw_features(source_directory, len(content)) if args.input == "raw" else content.numpy()
    train_ids = torch.tensor(sorted({i for seq in train for i in seq}))
    source = {"category": args.category, "raw_sha256": array_hash(raw),
              "train_sha256": json_hash(train), "item_order_sha256": json_hash(mapping["id2item"]),
              "input": args.input, "teacher_kind": args.teacher if config.ranking.teacher_weight > 0 else None}
    seed_run(args.seed, True)
    model = JointModel(config, raw, initialize_random_ae(config, raw, train_ids)).to(args.device)
    model.cache_features()
    if config.ranking.teacher_weight > 0:
        teacher = fixed_teacher(args.category, content, args.teacher)
        model.set_teacher(teacher)
        source["teacher_sha256"] = array_hash(teacher.numpy())
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(dataset, batch_size=config.recommender.train_batch_size,
                        shuffle=True, generator=generator)
    trainer = JointTrainer(model, config, train_ids, source, args.epochs * len(loader), generator)
    write_json(directory / "config.json", config.to_dict())
    code_files = sorted((ROOT / "joint_training").glob("*.py")) + [
        ROOT / "models/evaluation_sampling.py", ROOT / "models/diffusion/diffloss.py",
        ROOT / "models/diffusion/rectified_flow.py", Path(__file__).resolve()]
    code_hashes = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in code_files}
    write_json(directory / "experiment.json", {"args": vars(args) | {"config": str(args.config)}, "source": source,
                                               "test_accessed": False, "selection_users": args.eval_users,
                                               "matmul_precision": "high", "code_sha256": code_hashes})
    selected = np.random.default_rng(SEED).permutation(len(targets))[:args.eval_users]
    flow = max(config.ranking.rf_weight, config.ranking.prediction_weight,
               config.ranking.endpoint_weight, config.ranking.rollout_weight) > 0
    names = (["flow"] if flow else []) + (["ranking"] if config.ranking.weight > 0 else [])
    if config.ranking.endpoint_weight > 0:
        names.insert(1, "one_step")
    best = {key: {"score": -1., "epoch": 0} for key in names}
    log = []
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        terms = []
        for ids, labels in loader:
            batch = {"input_ids": ids.to(args.device), "labels": labels.to(args.device),
                     "attention_mask": (ids > 0).to(args.device), "seq_lens": (ids > 0).sum(-1).to(args.device)}
            terms.append(trainer.step(batch, epoch))
        row = {"epoch": epoch, "seconds": time.time() - started,
               "train": {key: float(np.mean([v[key] for v in terms])) for key in terms[0]}}
        if epoch % config.evaluation.eval_interval == 0 or epoch == args.epochs:
            metrics, predictions = evaluate(model, histories, targets, selected, flow)
            row["validation"] = metrics
            for name in best:
                score = metrics[name]
                if score["ndcg@10"] > best[name]["score"]:
                    best[name] = {"score": score["ndcg@10"], "epoch": epoch}
                    save_checkpoint(directory / f"best_{name}.pt", trainer, epoch=epoch,
                                    best_epoch=epoch, best_score=score["ndcg@10"], stale_epochs=0)
                    np.savez_compressed(directory / f"selection_{name}.npz", indices=selected,
                                        targets=targets[selected], predictions=predictions[name])
        log.append(row)
        write_json(directory / "metrics.json", log)
        print(json.dumps(row), flush=True)
    result = {"config": config.to_dict(), "source": source, "best": best,
              "test_accessed": False, "seconds_training_and_selection": time.time() - started,
              "audits": {}, "note": "All metrics are validation. Each head selects its own checkpoint on a fixed subset. Full audits include that subset."}
    for chosen in (best if args.audit == "full" else []):
        load_checkpoint(read_checkpoint(directory / f"best_{chosen}.pt"), trainer)
        # Full joint audit at the flow checkpoint; auxiliary checkpoints need only
        # their own decoder. Avoid duplicating expensive full-catalog flow audits.
        metrics, predictions = evaluate(model, histories, targets, np.arange(len(targets)), flow,
                                         only=None if chosen == "flow" else chosen)
        result["audits"][f"selected_by_{chosen}"] = metrics
        for name, prediction in predictions.items():
            np.savez_compressed(directory / f"full_selected_{chosen}_decoded_{name}.npz",
                                targets=targets, predictions=prediction)
    result["seconds_total"] = time.time() - started
    write_json(directory / "results.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
