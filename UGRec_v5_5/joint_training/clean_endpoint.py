"""One clean-item output receives both FM regression and catalog CE.

The reverse clock s goes from history+noise (1) to a clean item (0).
All endpoint and history graphs remain live, matching end-to-end control.
"""

import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .contrastive import select_positions
from .flow_ranking import candidate_ids, _exact_target_logp, _sampled_target_logp


def clean_endpoint_objectives(model, batch):
    cfg = model.config.flow_ranking
    selected = select_positions(batch, cfg.max_queries)
    if not len(selected):
        raise ValueError("Clean-endpoint training requires supervised positions.")
    encoded, _ = model.encode_history(batch, ae_grad=True)
    rows, positions = selected.rows.to(model.device), selected.positions.to(model.device)
    condition = model.condition_proj(encoded[rows, positions])
    targets = selected.target_ids.to(model.device)
    exact = cfg.catalog == "exact" or (cfg.catalog == "auto" and model.n_items - 1 <= cfg.max_exact_items)
    if exact:
        keys = model.encode_ids(torch.arange(1, model.n_items, device=model.device), grad=True)
        labels, negatives = targets - 1, None
    else:
        ids, labels, negatives = candidate_ids(targets, model.n_items, cfg.negatives, return_negative=True)
        keys = model.encode_ids(ids, grad=True)
    condition = condition.repeat_interleave(cfg.draws, 0)
    labels = labels.repeat_interleave(cfg.draws)
    clean = keys[labels]
    base = model.flow_prior(condition)
    clock_steps = model.diffloss.train_diffusion.num_timesteps
    # Draw for all ablations, including source-only, preserving random-number usage.
    times = torch.randint(clock_steps, (len(clean),), device=model.device)
    if cfg.endpoint_training == "source":
        times = torch.full_like(times, clock_steps - 1)
    fraction = times[:, None].to(base.dtype) / (clock_steps - 1)
    state = (1 - fraction) * clean + fraction * base
    prediction = model.diffloss.net(state, times, condition)
    fm = F.mse_loss(prediction, clean)
    ce = prediction.new_zeros(())
    if cfg.ce_weight:
        queries, unit_keys = F.normalize(prediction, dim=-1), F.normalize(keys, dim=-1)
        if exact:
            logp = checkpoint(_exact_target_logp, queries, unit_keys, labels,
                              cfg.temperature, use_reentrant=False)
        else:
            logp = _sampled_target_logp(queries, unit_keys, labels, negatives,
                                      cfg.temperature, model.n_items - 1)
        ce = -logp.mean()
    with torch.no_grad():
        distance = (prediction - clean).square().mean(-1)
        model.last_training_stats = {
            "queries": len(targets), "candidates": len(keys),
            "available_queries": selected.available_queries,
            "source_rms": float(base.square().mean().sqrt()),
            "target_rms": float(clean.square().mean().sqrt()),
            "prediction_rms": float(prediction.square().mean().sqrt()),
            "bridge_endpoint_cosine": float(F.cosine_similarity(prediction, clean).mean()),
            "catalog_mean_cosine": float(F.normalize(keys, dim=-1).mean(0).square().sum()),
            "catalog_coordinate_std": float(keys.std(0, unbiased=False).mean()),
        }
        for name, mask in (("near_data", fraction[:, 0] < .2), ("near_source", fraction[:, 0] >= .8)):
            model.last_training_stats[f"fm_{name}"] = float(distance[mask].mean()) if mask.any() else 0.
    return cfg.weight * (cfg.ce_weight * ce + cfg.fm_weight * fm), {
        "ce": ce, "fm": fm, "weighted_ce": cfg.ce_weight * ce, "weighted_fm": cfg.fm_weight * fm,
    }
