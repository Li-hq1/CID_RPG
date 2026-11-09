"""Fit only a CFM field on a frozen, validation-selected CE representation.

This is a second training stage, not an equal-total-budget system comparison.
The frozen model stays in evaluation mode, including its HSTU dropout.
"""

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import time

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from joint_training.checkpoint import evaluation_randomness, random_state, read_checkpoint
from joint_training.config import JointConfig
from joint_training.contrastive import select_positions
from joint_training.data import initialize_random_ae
from joint_training.fm import build_model
from joint_training.training import scheduler_for
from misc.artifacts import array_hash, atomic_write, json_hash, write_json
from protocol import ROOT, SELECTION_SEED, corpus, raw_features, training_dataset
from run import evaluate, seed_run, snapshot_source


def state_hash(state):
    return json_hash({name: array_hash(value.detach().cpu().numpy())
                      for name, value in state.items()})


def freeze_except_field(model):
    model.requires_grad_(False)
    model.diffloss.net.requires_grad_(True)
    model.eval()
    return list(model.diffloss.net.parameters())


def frozen_state(model):
    return {name: value for name, value in model.state_dict().items()
            if not name.startswith("diffloss.net.")}


def frozen_cfm_loss(model, batch):
    if model.training or any(module.training for module in model.modules()):
        raise RuntimeError("Frozen CE stage requires evaluation mode (no history dropout).")
    with torch.no_grad():
        selection = select_positions(batch, model.fm_config.max_queries)
        states, _ = model.encode_history(batch, ae_grad=False)
        condition = model.condition_proj(states[selection.rows, selection.positions])
        clean = model.catalog[selection.target_ids]
        base = model.flow_prior(condition)
        fraction = torch.rand(len(clean), 1, device=model.device)
        state, target = model.bridge(clean, base, fraction)
        times = fraction[:, 0] * (model.diffloss.train_diffusion.num_timesteps - 1)
    predicted = model.diffloss.net(state, times, condition)
    return F.mse_loss(predicted, target)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--ce-run", default="beauty_ce_only_s2024_20261106")
    parser.add_argument("--run-id", default="beauty_frozen_ce_flow_s2024_20261106")
    parser.add_argument("--engineering-steps", type=int)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("high")
    if args.device.startswith("cuda"):
        torch.cuda.set_per_process_memory_fraction(.20, int(args.device.split(":")[-1]))
    parent = ROOT / "outputs/Beauty" / args.ce_run
    hashes = json.loads((parent / "source_sha256.json").read_text())
    for relative, digest in hashes.items():
        if hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"Training source differs from CE parent: {relative}")
    checkpoint = read_checkpoint(parent / "best_flow.pt")
    cfg = JointConfig.from_dict(checkpoint["config"])
    assert cfg.fm.architecture == "serial" and cfg.fm.mse_weight == 0
    cfg.training.device = args.device
    seed_run(2024, True)
    directory, mapping, train, histories, targets = corpus("Beauty")
    raw = raw_features(directory, len(mapping["id2item"]) - 1)
    source = {"category": "Beauty", "raw_sha256": array_hash(raw),
              "train_sha256": json_hash(train), "item_order_sha256": json_hash(mapping["id2item"]),
              "input": "raw", "teacher_kind": None}
    if checkpoint["source"] != source:
        raise RuntimeError("CE checkpoint data identity mismatch.")
    train_ids = torch.tensor(sorted({item for seq in train for item in seq}))
    model = build_model(cfg, raw, initialize_random_ae(cfg, raw, train_ids)).to(args.device)
    # A fresh field, before loading the CE-only checkpoint's unused network.
    initial_field = deepcopy(model.diffloss.net.state_dict())
    model.load_state_dict(checkpoint["model"], strict=True)
    model.diffloss.net.load_state_dict(initial_field)
    del initial_field
    parameters = freeze_except_field(model)
    before = state_hash(frozen_state(model))
    initial_field_hash = state_hash(model.diffloss.net.state_dict())
    model.cache_features()
    model.refresh_catalog()
    selected = np.random.default_rng(SELECTION_SEED).permutation(len(targets))[:4096]
    dataset = training_dataset(train)
    generator = torch.Generator().manual_seed(2024)
    loader = DataLoader(dataset, batch_size=cfg.recommender.train_batch_size,
                        shuffle=True, generator=generator)
    total_steps = cfg.training.epochs * len(loader)
    optimizer = torch.optim.AdamW(parameters, lr=cfg.recommender.lr,
                                  weight_decay=cfg.recommender.weight_decay)
    scheduler = scheduler_for(optimizer, cfg.recommender, total_steps)
    output = ROOT / "outputs/Beauty" / args.run_id
    output.mkdir(parents=True, exist_ok=False)
    code_hash = snapshot_source(output)
    shutil.copy2(__file__, output / "source/frozen_ce_flow.py")
    script_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    write_json(output / "parent_config.json", cfg.to_dict())
    write_json(output / "experiment.json", {
        "scheme": "frozen_ce_then_cfm", "parent": str(parent), "parent_epoch": checkpoint["epoch"],
        "parent_checkpoint_sha256": hashlib.sha256((parent / "best_flow.pt").read_bytes()).hexdigest(),
        "source": source, "source_sha256": code_hash, "research_script_sha256": script_hash,
        "frozen_state_sha256": before, "initial_field_sha256": initial_field_hash,
        "seed": 2024, "epochs": cfg.training.epochs, "expected_updates": total_steps,
        "selection_users": len(selected), "selection_indices_sha256": array_hash(selected),
        "selection_seed": SELECTION_SEED, "test_accessed": False,
        "engineering_check": args.engineering_steps is not None,
        "trainable_parameters": sum(p.numel() for p in parameters),
        "frozen_representation_mode": "eval", "train_draws": 1,
        "optimizer": "fresh AdamW, LR/weight decay/scheduler copied from CE recommender",
        "clipping": "field parameters only", "ce_and_local_losses": "not computed",
        "sampling_steps": 4, "inference_draws": 32,
        "budget_note": "Extra 40-epoch field stage after selected CE pretraining; not matched total training cost",
    })
    audit_indices = selected[:8] if args.engineering_steps else np.arange(len(targets))
    fixed_ce, fixed_predictions = evaluate(model, histories, targets, audit_indices, source="ranking")
    with np.load(parent / "full_flow.npz") as saved:
        np.testing.assert_array_equal(fixed_predictions, saved["predictions"][audit_indices])
    fixed_prior, _ = evaluate(model, histories, targets, audit_indices, source="prior")
    # Evaluation restores training RNG; restart here to make this stage explicit.
    seed_run(2024, True)
    best, rows, steps = {"score": -1., "epoch": 0}, [], 0
    started = time.time()
    if model.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(model.device)
    for epoch in range(1, cfg.training.epochs + 1):
        losses = []
        for ids, labels in loader:
            batch = {"input_ids": ids.to(args.device), "labels": labels.to(args.device),
                     "attention_mask": (ids > 0).to(args.device),
                     "seq_lens": (ids > 0).sum(-1).to(args.device)}
            optimizer.zero_grad(set_to_none=True)
            loss = frozen_cfm_loss(model, batch)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite frozen-stage loss.")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, cfg.recommender.max_grad_norm, error_if_nonfinite=True)
            optimizer.step()
            scheduler.step()
            steps += 1
            losses.append(float(loss.detach()))
            if args.engineering_steps and steps >= args.engineering_steps:
                break
        row = {"epoch": epoch, "steps": steps, "train_mse": float(np.mean(losses))}
        if epoch % 5 == 0 or epoch == cfg.training.epochs or args.engineering_steps:
            selection = selected[:8] if args.engineering_steps else selected
            metrics, predictions = evaluate(model, histories, targets, selection, source="flow")
            row["validation_selection"] = metrics
            if metrics["ndcg@10"] > best["score"]:
                best = {"score": metrics["ndcg@10"], "epoch": epoch}
                saved = {"method": "frozen_ce_cfm_research", "model": model.state_dict(),
                         "parent_config": cfg.to_dict(), "source": source, "epoch": epoch,
                         "best": best, "steps": steps, "optimizer": optimizer.state_dict(),
                         "scheduler": scheduler.state_dict(), "rng": random_state(model.device),
                         "loader_rng": generator.get_state(), "frozen_state_sha256": before}
                atomic_write(output / "best_flow.pt", lambda stream: torch.save(saved, stream))
                np.savez_compressed(output / "selection_flow.npz", indices=selection,
                                    predictions=predictions, targets=targets[selection])
        rows.append(row)
        write_json(output / "metrics.json", rows)
        print(json.dumps(row), flush=True)
        if args.engineering_steps:
            break
    training_seconds = time.time() - started
    assert before == state_hash(frozen_state(model)), "Frozen representation changed."
    assert initial_field_hash != state_hash(model.diffloss.net.state_dict()), "Field did not update."
    saved = torch.load(output / "best_flow.pt", weights_only=True, map_location="cpu")
    model.load_state_dict(saved["model"], strict=True)
    assert before == state_hash(frozen_state(model))
    with np.load(output / "selection_flow.npz") as expected:
        metrics, predictions = evaluate(model, histories, targets, expected["indices"], source="flow")
        np.testing.assert_array_equal(expected["predictions"], predictions)
        assert metrics["ndcg@10"] == best["score"]
    final, predictions = evaluate(model, histories, targets, audit_indices, source="flow")
    np.savez_compressed(output / "full_flow.npz", indices=audit_indices,
                        targets=targets[audit_indices], predictions=predictions)
    ce_after, predictions = evaluate(model, histories, targets, audit_indices, source="ranking")
    np.testing.assert_array_equal(predictions, fixed_predictions)
    assert ce_after == fixed_ce
    result = {"best": best, "validation": final, "frozen_ce": fixed_ce,
              "frozen_prior": fixed_prior, "actual_updates": steps,
              "frozen_state_verified": True, "ce_predictions_unchanged": True,
              "checkpoint_reload_verified": True, "training_seconds": training_seconds,
              "seconds_including_audits": time.time() - started,
              "engineering_check": args.engineering_steps is not None, "test_accessed": False,
              "fm_vs_ce_relative_ndcg10": final["ndcg@10"] / fixed_ce["ndcg@10"] - 1
              if fixed_ce["ndcg@10"] else None,
              "fm_vs_prior_relative_ndcg10": final["ndcg@10"] / fixed_prior["ndcg@10"] - 1
              if fixed_prior["ndcg@10"] else None}
    if model.device.type == "cuda":
        result["peak_memory_reserved_mib"] = torch.cuda.max_memory_reserved(model.device) / 2**20
    write_json(output / "results.json", result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
