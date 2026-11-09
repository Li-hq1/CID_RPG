"""Sampled ranking of actual flow outputs, with no train-time catalog cache.

All trainable computations are live except the sampled RF regression data and
its velocity labels. The same shared AE encodes histories and sampled keys.
There are no history, denoising, or teacher classification branches here.
"""

import math

import torch
from torch.nn import functional as F

from .contrastive import select_positions


def candidate_ids(targets, n_items, negatives, return_negative=False):
    """Unique batch positives plus O(K) uniform draws, never an O(N) mask."""
    random_ids = torch.randint(1, n_items, (negatives,), device=targets.device)
    ids = torch.cat((targets, random_ids)).unique(sorted=True)
    labels = torch.searchsorted(ids, targets)
    if return_negative:
        return ids, labels, torch.searchsorted(ids, random_ids)
    return ids, labels


def ranking_loss(generated, keys, labels, cfg, negative_columns=None, catalog_size=None):
    if cfg.sampling != "batch":
        if negative_columns is None:
            raise ValueError("Uniform ranking requires the sampled draw columns.")
        positive_keys, negative_keys = keys[labels], keys[negative_columns]
        if cfg.objective == "flow_distance":
            pos = -(generated - positive_keys).square().mean(-1, keepdim=True)
            neg = -(generated.square().sum(-1, keepdim=True)
                    + negative_keys.square().sum(-1)[None] - 2 * generated @ negative_keys.T) / generated.shape[-1]
        else:
            queries = F.normalize(generated, dim=-1)
            pos = (queries * F.normalize(positive_keys, dim=-1)).sum(-1, keepdim=True)
            neg = queries @ F.normalize(negative_keys, dim=-1).T
        neg = neg.masked_fill(labels[:, None] == negative_columns[None], -torch.inf)
        if cfg.objective == "contrastive":
            logits = torch.cat((pos, neg), dim=1) / cfg.temperature
            if cfg.sampling == "uniform_corrected":
                # Denominator estimator: exp(s_y) + N/K sum_k 1[n_k != y]
                # exp(s_nk). Draws are uniform with replacement; duplicates are
                # retained. Partition estimate is unbiased, its log is not.
                logits[:, 1:] = logits[:, 1:] + math.log(catalog_size / len(negative_columns))
            return F.cross_entropy(logits, torch.zeros(len(labels), device=labels.device, dtype=torch.long))
        width = min(cfg.hard_negatives or len(negative_columns), len(negative_columns))
        return F.softplus((neg.topk(width, dim=1, sorted=False).values - pos + cfg.margin) / cfg.temperature).mean()
    if cfg.objective == "flow_distance":
        # Squared endpoint error / dimension: also the t^2-weighted velocity
        # residual energy for a straight-path RF clean-state estimate.
        logits = -(generated.square().sum(-1, keepdim=True)
                   + keys.square().sum(-1)[None] - 2 * generated @ keys.T) / generated.shape[-1]
    else:
        logits = F.normalize(generated, dim=-1) @ F.normalize(keys, dim=-1).T
    if len(keys) < 2:
        return logits.sum() * 0
    if cfg.objective == "contrastive":
        return F.cross_entropy(logits / cfg.temperature, labels)
    positive = logits.gather(1, labels[:, None])
    negative = logits.masked_fill(F.one_hot(labels, len(keys)).bool(), -torch.inf)
    width = min(cfg.hard_negatives or len(keys) - 1, len(keys) - 1)
    hardest = negative.topk(width, dim=1, sorted=False).values
    return F.softplus((hardest - positive + cfg.margin) / cfg.temperature).mean()


def directional_velocity_loss(error, positive, keys, labels, count):
    """Penalize RF error along directions to confusable item neighbors.

    A geometry-weighted regression alternative, not a calibrated ranking loss
    or a claim of equivalence to unmodified Euclidean flow matching.
    """
    positive, keys = positive.detach(), keys.detach()
    distance2 = (positive.square().sum(-1, keepdim=True) + keys.square().sum(-1)[None]
                 - 2 * positive @ keys.T).clamp_min(1e-6)
    forbidden = F.one_hot(labels, len(keys)).bool()
    count = min(count or len(keys) - 1, len(keys) - 1)
    if count <= 0:
        return error.sum() * 0
    neighbors = distance2.masked_fill(forbidden, torch.inf).topk(count, largest=False).indices
    projection = (error * positive).sum(-1, keepdim=True) - error @ keys.T
    return (projection.gather(1, neighbors).square() / distance2.gather(1, neighbors)).mean()


def unified_objectives(model, batch):
    cfg = model.config.flow_ranking
    selection = select_positions(batch, cfg.max_queries)
    if not len(selection):
        raise ValueError("Flow ranking needs a supervised history position.")
    encoded, _ = model.encode_history(batch, ae_grad=True)
    condition = model.condition_proj(encoded[selection.rows, selection.positions])
    targets = selection.target_ids.to(model.device)
    ids, labels, negatives = candidate_ids(targets, model.n_items, cfg.negatives, return_negative=True)
    keys = model.encode_ids(ids, grad=True)
    clean = keys[labels].detach()
    base = model.flow_prior(condition)
    if cfg.path == "direct":
        generated = model.ranking_head(condition)
        rf = generated.sum() * 0
    elif cfg.path == "matched":
        # The training query is the inference map: one prior draw and one Euler
        # step. Ranking is the only generative loss, so there is no second
        # velocity-regression forward.
        generated = model.transport(base, condition, 1)
        rf = None
    else:
        # Regression samples and labels define the RF problem; gradients do not
        # move those data points to make the velocity regression artificially easy.
        steps = model.diffloss.train_diffusion.num_timesteps
        times = torch.randint(steps, (len(targets),), device=model.device)
        if cfg.path == "coupled":
            endpoint = torch.rand(len(times), device=model.device) < cfg.endpoint_probability
            times = torch.where(endpoint, steps - 1, times)
        fraction = times[:, None] / (steps - 1)
        noisy = (1 - fraction) * clean + fraction * base.detach()
        velocity = model.diffloss.net(noisy, times, condition)
        rf_per_query = (velocity - (base.detach() - clean)).square().mean(-1)
        if cfg.path == "metric_rf":
            rf = rf_per_query.mean()
            rank = directional_velocity_loss(fraction * (velocity - (base.detach() - clean)),
                                            clean, keys, labels, cfg.hard_negatives)
            model.last_training_stats = {"queries": len(targets), "candidates": len(ids),
                                         "available_queries": selection.available_queries}
            return rf + cfg.weight * rank, {"rf": rf, "directional_rank": rank}
        if cfg.path == "coupled":
            # Importance weighting preserves the original uniform-time RF MSE
            # while sharing one network call with target-free endpoint ranking.
            probability_ratio = (1 - cfg.endpoint_probability) + cfg.endpoint_probability * steps * (times == steps - 1)
            rf = (rf_per_query / probability_ratio).mean()
            selected = times == steps - 1
            generated = (noisy - fraction * velocity)[selected]
            labels = labels[selected]
        elif cfg.path == "random_time":
            rf = rf_per_query.mean()
            generated = noisy - fraction * velocity
        else:
            rf = rf_per_query.mean()
            steps = cfg.steps if cfg.path == "rollout" else 1
            if cfg.depth_dropout:
                # One expectation over computation depths; depth zero is
                # explicitly history/prior ranking supervision, not its removal.
                active = (torch.rand(len(base), device=base.device) >= cfg.depth_dropout).nonzero(as_tuple=True)[0]
                generated = base
                if len(active):
                    transported = model.transport(base[active], condition[active], steps)
                    generated = base.index_copy(0, active, transported)
            else:
                generated = model.transport(base, condition, steps)
    rank = (ranking_loss(generated, keys, labels, cfg, negatives, model.n_items - 1)
            if len(generated) else keys.sum() * 0)
    # Direct is a labelled non-generative reference. Matched keeps the
    # inference transport and drops the separate velocity regression.
    terms = {"sampled_rank": rank} if rf is None else {"rf": rf, "sampled_rank": rank}
    if rf is None:
        rf = rank.sum() * 0
    model.last_training_stats = {"queries": len(targets), "candidates": len(ids),
                                 "available_queries": selection.available_queries}
    return rf + cfg.weight * rank, terms
