"""Predeclared diagnostics for the two Beauty CE/CFM trials; no parameter updates."""

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from joint_training.checkpoint import evaluation_randomness, read_checkpoint
from joint_training.config import JointConfig
from joint_training.fm import build_model
from misc.artifacts import array_hash, json_hash, write_json
from protocol import ROOT, SELECTION_SEED, corpus, raw_features, training_dataset, validation_batches
from run import evaluate


def per_user_scores(predictions, targets):
    hits = predictions[:, :10] == targets[:, None]
    return (hits / np.log2(np.arange(10) + 2)).sum(1), hits.any(1)


def paired(first, second, targets):
    a, hit_a = per_user_scores(first, targets)
    b, hit_b = per_user_scores(second, targets)
    overlap = (first[:, :10, None] == second[:, None, :10]).any(2).sum(1)
    return {
        "ndcg10_first": float(a.mean()), "ndcg10_second": float(b.mean()),
        "absolute_delta": float((a - b).mean()),
        "relative_delta": float(a.mean() / b.mean() - 1),
        "first_better_users": int((a > b).sum()), "second_better_users": int((b > a).sum()),
        "equal_users": int((a == b).sum()), "first_only_hits": int((hit_a & ~hit_b).sum()),
        "second_only_hits": int((hit_b & ~hit_a).sum()), "both_hits": int((hit_a & hit_b).sum()),
        "mean_top10_overlap": float(overlap.mean()),
        "same_top1_fraction": float((first[:, 0] == second[:, 0]).mean()),
    }


def sample_statistics(samples, keys, temperature):
    unit = F.normalize(samples, dim=-1)
    count = unit.shape[1]
    pair = ((unit @ unit.transpose(1, 2)).sum((1, 2)) - count) / (count * (count - 1))
    variance = unit.var(1, unbiased=False).sum(-1)
    logp = (unit @ keys.T / temperature).log_softmax(-1)
    log_mean = torch.logsumexp(logp, 1) - math.log(count)
    h_mean = -(log_mean.exp() * log_mean).sum(-1)
    h_each = -(logp.exp() * logp).sum(-1).mean(1)
    top1 = logp.argmax(-1).sort(1).values
    unique = 1 + (top1[:, 1:] != top1[:, :-1]).sum(1)
    return {"pairwise_cosine": pair, "unit_variance": variance,
            "top1_unique": unique.float(), "sample_probability_js_nats": h_mean - h_each}


@torch.no_grad()
def distribution_diagnostics(model, histories, targets):
    indices = np.random.default_rng(SELECTION_SEED).permutation(len(targets))[:512]
    samples = {"prior": {}, "flow": {}, "relations": {}}
    velocity = {str(t): {"mse": [], "zero_field_mse": []} for t in (.1, .5, .9)}
    keys = F.normalize(model.catalog[1:], dim=-1)
    model.eval()
    with evaluation_randomness(model.device, SELECTION_SEED):
        for _, batch in validation_batches(histories, targets, 32, indices):
            batch = {k: v.to(model.device) for k, v in batch.items()}
            rank_all, fm_all = model.history_branches(batch)
            rows = torch.arange(len(batch["seq_lens"]), device=model.device)
            last = batch["seq_lens"] - 1
            rank, condition = rank_all[rows, last], fm_all[rows, last]
            base = model.flow_prior(rank.repeat_interleave(32, 0))
            endpoint = model.transport(base, condition.repeat_interleave(32, 0), 4)
            base = base.reshape(len(rows), 32, -1)
            endpoint = endpoint.reshape_as(base)
            for name, values in (("prior", base), ("flow", endpoint)):
                for key, value in sample_statistics(values, keys, model.fm_config.temperature).items():
                    samples[name].setdefault(key, []).extend(value.cpu().tolist())
            anchor = F.normalize(model.ranking_head(rank), dim=-1)
            relations = {
                "mean_flow_direction_vs_ce_cosine": F.cosine_similarity(endpoint.mean(1), anchor, dim=-1),
                "flow_radius_rms": endpoint.square().mean((1, 2)).sqrt(),
                "prior_radius_rms": base.square().mean((1, 2)).sqrt(),
                "transport_displacement_rms": (endpoint - base).square().mean((1, 2)).sqrt(),
            }
            for key, value in relations.items():
                samples["relations"].setdefault(key, []).extend(value.cpu().tolist())
            clean = model.encode_ids(batch["labels"][:, 0], grad=False)
            for t in (.1, .5, .9):
                state, target = model.bridge(clean, base[:, 0], clean.new_full((len(rows), 1), t))
                time = clean.new_full((len(rows),), t * (model.diffloss.train_diffusion.num_timesteps - 1))
                predicted = model.diffloss.net(state, time, condition)
                velocity[str(t)]["mse"].extend((predicted - target).square().mean(-1).cpu().tolist())
                velocity[str(t)]["zero_field_mse"].extend(target.square().mean(-1).cpu().tolist())
    summary = {name: {key: float(np.mean(values)) for key, values in group.items()}
               for name, group in samples.items()}
    summary["users"] = len(indices)
    summary["indices_sha256"] = array_hash(indices)
    summary["velocity"] = {t: {key: float(np.mean(values)) for key, values in group.items()}
                            for t, group in velocity.items()}
    for group in summary["velocity"].values():
        group["relative_to_zero_field"] = group["mse"] / group["zero_field_mse"]
    return summary


def gradient_diagnostics(model, train):
    dataset = training_dataset(train)
    order = np.random.default_rng(2024).permutation(len(dataset))[:4 * 256]
    parameters = [(n, p) for n, p in model.named_parameters()
                  if n.startswith(("input_proj.", "encoder.", "condition_proj."))]
    records = []
    model.train()
    with evaluation_randomness(model.device, 2024):
        for start in range(0, len(order), 256):
            ids, labels = (t[order[start:start + 256]].to(model.device) for t in dataset.tensors)
            batch = {"input_ids": ids, "labels": labels, "attention_mask": ids > 0,
                     "seq_lens": (ids > 0).sum(-1)}
            _, terms = model.joint_objectives(batch)
            ce = torch.autograd.grad(terms["history_ce"], [p for _, p in parameters], retain_graph=True, allow_unused=True)
            mse = torch.autograd.grad(terms["rf"], [p for _, p in parameters], allow_unused=True)
            groups = {}
            for prefix in ("all", "input_proj.", "encoder.", "condition_proj."):
                pairs = [(a, b) for (n, _), a, b in zip(parameters, ce, mse)
                         if (prefix == "all" or n.startswith(prefix)) and a is not None and b is not None]
                aa = sum(a.square().sum() for a, _ in pairs)
                bb = sum(b.square().sum() for _, b in pairs)
                ab = sum((a * b).sum() for a, b in pairs)
                groups[prefix] = {"cosine": float(ab / (aa * bb).sqrt().clamp_min(1e-20)),
                                  "ce_norm": float(aa.sqrt()), "mse_norm": float(bb.sqrt()),
                                  "mse_to_ce_norm": float((bb / aa.clamp_min(1e-20)).sqrt())}
            records.append(groups)
    model.eval()
    return {"mode": "training, fixed dropout and sampling RNG", "batches": records,
            "mean_cosine": float(np.mean([r["all"]["cosine"] for r in records])),
            "mean_mse_to_ce_norm": float(np.mean([r["all"]["mse_to_ce_norm"] for r in records]))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--ce-run", default="beauty_ce_only_s2024_20261106")
    parser.add_argument("--fm-run", default="beauty_serial_shared_s2024_20261106")
    parser.add_argument("--output", default="outputs/initial_analysis_20261106")
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("high")
    if args.device.startswith("cuda"):
        torch.cuda.set_per_process_memory_fraction(.20, int(args.device.split(":")[-1]))
    output = ROOT / args.output
    output.mkdir(parents=True, exist_ok=False)
    ce_dir, fm_dir = (ROOT / "outputs/Beauty" / name for name in (args.ce_run, args.fm_run))
    ce_result, fm_result = (json.loads((p / "results.json").read_text()) for p in (ce_dir, fm_dir))
    experiments = [json.loads((p / "experiment.json").read_text()) for p in (ce_dir, fm_dir)]
    assert experiments[0]["source_sha256"] == experiments[1]["source_sha256"]
    for directory in (ce_dir, fm_dir):
        assert json.loads((directory / "results.json").read_text())["actual_updates"] == 4040
        for relative, digest in json.loads((directory / "source_sha256.json").read_text()).items():
            assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == digest, relative
    predictions = {}
    for name, path in {"ce_only": ce_dir / "full_flow.npz", "fm4": fm_dir / "full_flow.npz",
                       "joint_ce_same_checkpoint": fm_dir / "diagnostic_ranking.npz",
                       "prior": fm_dir / "diagnostic_prior.npz", "joint_ce_own_checkpoint": fm_dir / "ranking_selected.npz"}.items():
        with np.load(path) as saved:
            predictions[name] = saved["predictions"]
            if name == "ce_only":
                targets = saved["targets"]
            else:
                np.testing.assert_array_equal(targets, saved["targets"])
    source_dir, mapping, train, histories, current_targets = corpus("Beauty")
    np.testing.assert_array_equal(targets, current_targets)
    raw = raw_features(source_dir, len(mapping["id2item"]) - 1)
    checkpoint = read_checkpoint(fm_dir / "best_flow.pt")
    assert checkpoint["source"]["raw_sha256"] == array_hash(raw)
    assert checkpoint["source"]["train_sha256"] == json_hash(train)
    config = JointConfig.from_dict(checkpoint["config"])
    config.training.device = args.device
    model = build_model(config, raw).to(args.device)
    model.load_state_dict(checkpoint["model"])
    model.cache_features()
    model.eval()
    model.refresh_catalog()
    one_step_metrics, predictions["fm1"] = evaluate(model, histories, targets, np.arange(len(targets)), source="one_step")
    np.savez_compressed(output / "one_step.npz", targets=targets, predictions=predictions["fm1"])
    comparisons = {}
    for first, second in (("fm4", "ce_only"), ("joint_ce_own_checkpoint", "ce_only"),
                          ("fm4", "joint_ce_same_checkpoint"), ("fm4", "prior"), ("fm1", "fm4")):
        comparisons[f"{first}_vs_{second}"] = paired(predictions[first], predictions[second], targets)
    print(json.dumps({"comparisons": comparisons}, indent=2), flush=True)
    summary = {"ce_run": args.ce_run, "fm_run": args.fm_run, "users": len(targets),
               "source_sha256": experiments[0]["source_sha256"], "test_accessed": False,
               "analysis_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "ce_result": ce_result, "fm_result": fm_result, "one_step": one_step_metrics,
               "comparisons": comparisons, "distribution": distribution_diagnostics(model, histories, targets),
               "gradients": gradient_diagnostics(model, train)}
    write_json(output / "summary.json", summary)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for directory, name, key in ((ce_dir, "CE only", "validation_selection"),
                                 (fm_dir, "CE+CFM: FM output", "validation_selection"),
                                 (fm_dir, "CE+CFM: CE output", "validation_ranking_selection")):
        rows = json.loads((directory / "metrics.json").read_text())
        selected = [row for row in rows if key in row]
        axes[0].plot([row["epoch"] for row in selected], [row[key]["ndcg@10"] for row in selected], marker="o", label=name)
    axes[0].set(xlabel="Epoch", ylabel="Selection validation NDCG@10")
    axes[0].legend(fontsize=8)
    rows = json.loads((fm_dir / "metrics.json").read_text())
    axes[1].plot([r["epoch"] for r in rows], [r["train"]["loss/rf"] for r in rows], label="Train CFM MSE")
    axes[1].set(xlabel="Epoch", ylabel="Velocity MSE")
    axes[1].legend()
    for axis in axes:
        axis.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(output / "learning_curves.png", dpi=170)
    plt.close(fig)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
