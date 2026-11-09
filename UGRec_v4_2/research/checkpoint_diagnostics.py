"""Paired decoding and target-shortcut diagnostics on existing checkpoints.

All experiments use validation only. Different decoding rules reuse identical
generated samples. Teacher-forced noisy targets are never ranking evaluations.
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from compression.geometry_ae.model import loss_terms
from joint_training.contrastive import infonce_loss, sample_negative_ids, select_positions
from joint_training.training import frozen
from models.evaluation_sampling import aggregate_diffusion_candidates
from .common import ROOT, SEED, batches, load_joint, paired_interval, per_user, summarize, write_json


@torch.no_grad()
def decode(model, histories, targets, indices, device, output):
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    predictions = {name: [] for name in ("support_k1", "support_k5", "support_k10", "min_distance_k10",
                                         "single_sample", "mean_sample", "zero_noise", "shuffled_history")}
    saved_samples = []
    norm_samples, nearest_distances, unique_nearest = [], [], []
    catalog = model.catalog[1:]
    cfg = model.eval_config.copy()
    for counter, (ix, batch) in enumerate(batches(histories, targets, 64, indices)):
        batch = {k: v.to(device) for k, v in batch.items()}
        _, condition = model.encode_history(batch, ae_grad=False)
        samples = model._sample_condition(condition.repeat_interleave(32, 0)).reshape(len(ix), 32, -1)
        saved_samples.append(samples.cpu().numpy())
        for depth in (1, 5, 10):
            cfg.update(eval_candidates_per_sample=depth, eval_candidate_ranking="support")
            pred, _ = aggregate_diffusion_candidates(samples, catalog, 10, cfg)
            predictions[f"support_k{depth}"].append(pred.squeeze(-1).cpu().numpy())
        cfg.update(eval_candidates_per_sample=10, eval_candidate_ranking="min_distance")
        pred, _ = aggregate_diffusion_candidates(samples, catalog, 10, cfg)
        predictions["min_distance_k10"].append(pred.squeeze(-1).cpu().numpy())
        for name, value in (("single_sample", samples[:, 0]), ("mean_sample", samples.mean(1))):
            distances = torch.cdist(value, catalog).square()
            predictions[name].append((distances.topk(10, largest=False).indices + 1).cpu().numpy())
        old_temperature = model.diff_temperature
        model.diff_temperature = 0.
        zero = model._sample_condition(condition)
        model.diff_temperature = old_temperature
        predictions["zero_noise"].append((torch.cdist(zero, catalog).topk(10, largest=False).indices + 1).cpu().numpy())
        # Permute complete histories, lengths and masks together; keep targets fixed.
        permutation = torch.randperm(len(ix), device=device)
        shuffled_condition = condition[permutation]
        shuffled = model._sample_condition(shuffled_condition.repeat_interleave(32, 0)).reshape(len(ix), 32, -1)
        cfg.update(eval_candidates_per_sample=1, eval_candidate_ranking="support")
        pred, _ = aggregate_diffusion_candidates(shuffled, catalog, 10, cfg)
        predictions["shuffled_history"].append(pred.squeeze(-1).cpu().numpy())
        distances = torch.cdist(samples.flatten(0, 1), catalog).square().reshape(len(ix), 32, -1)
        norm_samples.extend(samples.norm(dim=-1).flatten().cpu().tolist())
        nearest_distances.extend(distances.amin(-1).sqrt().flatten().cpu().tolist())
        unique_nearest.extend([len(x.unique()) for x in distances.argmin(-1)])
        if counter % 10 == 0:
            print(f"decoding {counter * 64}/{len(indices)}", flush=True)
    predictions = {name: np.concatenate(values) for name, values in predictions.items()}
    labels = targets[indices]
    np.savez_compressed(output / "paired_predictions.npz", indices=indices, targets=labels, **predictions)
    np.savez_compressed(output / "generated_samples.npz", indices=indices, targets=labels,
                        samples=np.concatenate(saved_samples))
    scores = {name: summarize(value, labels) for name, value in predictions.items()}
    intervals = {name: paired_interval(per_user(predictions[name], labels)["ndcg@10"],
                                     per_user(predictions["support_k1"], labels)["ndcg@10"])
                 for name in predictions if name != "support_k1"}
    return {"scores": scores, "paired_ndcg10_against_support_k1": intervals,
            "geometry": {"catalog_norm_mean": float(catalog.norm(dim=-1).mean()),
                         "sample_norm_quantiles": np.quantile(norm_samples, [.1, .5, .9]).tolist(),
                         "sample_nearest_distance_quantiles": np.quantile(nearest_distances, [.1, .5, .9]).tolist(),
                         "unique_top1_per_32_draws": float(np.mean(unique_nearest))}}


def gradient_vector(loss, params):
    values = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
    return torch.cat([(torch.zeros_like(p) if g is None else g).flatten().detach()
                      for p, g in zip(params, values)])


def shortcut(model, histories, targets, indices, device):
    _, batch = next(batches(histories, targets, 256, indices))
    batch = {k: v.to(device) for k, v in batch.items()}
    selection = select_positions(batch)
    rows = selection.rows
    params = list(model.ae.encoder.parameters())
    torch.manual_seed(SEED)
    noise = torch.randn(len(rows), model.latent_dim, device=device)
    negatives = sample_negative_ids(batch, rows, selection.target_ids, model.n_items - 1, 64, device)
    results = []
    with frozen(model.rec_parameters()):
        _, c = model.encode_history(batch, ae_grad=True)
        clean = model.encode_ids(selection.target_ids, grad=False)
        positives = model.encode_ids(selection.target_ids)
        negative_values = model.encode_ids(negatives)
        shuffled_c = c.roll(1, 0)
        for fraction in (0., .1, .25, .5, .75, .9, 1.):
            timestep = torch.full((len(rows),), round(fraction * 999), device=device, dtype=torch.long)
            noisy = model.diffloss.train_diffusion.q_sample(clean, timestep, noise)
            t = timestep[:, None].float() / 999
            row = {"t_noise_fraction": fraction}
            for name, condition in (("correct_history", c), ("shuffled_history", shuffled_c),
                                     ("unconditional", model.uncond_condition[None].expand_as(c))):
                predicted = noisy - t * model.diffloss.net(noisy, timestep, condition)
                loss = infonce_loss(predicted, positives, negative_values,
                                   temperature=model.config.contrastive.temperature)
                # The t=0 gradient below can only reach AE keys; the anchor is the detached target.
                gradient = gradient_vector(loss, params)
                row[name] = {"infonce": float(loss.detach()), "ae_encoder_grad_norm": float(gradient.norm()),
                             "target_cosine": float(F.cosine_similarity(predicted, clean).mean().detach())}
            baseline = infonce_loss(noisy, positives, negative_values,
                                    temperature=model.config.contrastive.temperature)
            row["no_network_infonce"] = float(baseline.detach())
            results.append(row)
        # Explicitly distinguish the existing RF route from its optional history-only AE route.
        no_route = model.rf_loss(batch, ae_grad=False)
        no_route_has_graph = no_route.requires_grad
        routed = model.rf_loss(batch, ae_grad=True)
        rf_grad = gradient_vector(routed, params)
        prediction = model.rf_loss(batch, return_prediction=True, ae_grad=True)[1]
        contrastive = infonce_loss(prediction, positives, negative_values,
                                  temperature=model.config.contrastive.temperature)
        info_grad = gradient_vector(model.config.contrastive.weight * contrastive, params)
        ids = torch.arange(1, min(513, model.n_items), device=device)
        x = model.raw_features[ids.cpu() - 1].to(device)
        reconstructed, z = model.ae(x)
        settings = model.config.ae
        terms = loss_terms(x, reconstructed, z, model.ae.input_scale, batch_relation=False)
        local_loss = terms["reconstruction"] + settings.cosine_weight * terms["cosine"] + settings.variance_weight * terms["variance"] + settings.correlation_weight * terms["correlation"]
        local_grad = gradient_vector(local_loss, params)
        gradients = {"existing_rf_has_ae_graph_with_rec_frozen": no_route_has_graph,
                     "history_routed_rf_encoder_grad_norm": float(rf_grad.norm()),
                     "weighted_infonce_encoder_grad_norm": float(info_grad.norm()),
                     "local_ae_encoder_grad_norm": float(local_grad.norm()),
                     "cosine_local_vs_infonce": float(F.cosine_similarity(local_grad[None], info_grad[None])),
                     "cosine_local_vs_history_rf": float(F.cosine_similarity(local_grad[None], rf_grad[None]))}
    return {"queries": len(rows), "negative_count": negatives.shape[1],
            "warning": "Teacher-forced noisy-target analysis, not recommendation metrics or test leakage.",
            "by_time": results, "gradient_routes": gradients}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--device", default="cuda:4")
    parser.add_argument("--users", type=int, default=4096)
    args = parser.parse_args()
    torch.set_num_threads(2)
    output = ROOT / "outputs/research_20261028/checkpoints" / args.name
    output.mkdir(parents=True, exist_ok=True)
    model, checkpoint, histories, targets = load_joint(args.checkpoint, args.device)
    indices = np.random.default_rng(SEED).permutation(len(targets))[:args.users]
    result = {"checkpoint": str(Path(args.checkpoint).resolve()), "seed": SEED,
              "split": "validation", "users": len(indices), "checkpoint_epoch": checkpoint["epoch"]}
    result["shortcut"] = shortcut(model, histories, targets, indices, args.device)
    write_json(output / "shortcut.json", result)
    print(result["shortcut"], flush=True)
    result["decoding"] = decode(model, histories, targets, indices, args.device, output)
    write_json(output / "results.json", result)
    print(result["decoding"], flush=True)


if __name__ == "__main__":
    main()
