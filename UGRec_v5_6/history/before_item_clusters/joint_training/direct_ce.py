"""Deterministic history prediction using the control's existing projection head."""

import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .contrastive import select_positions
from .flow_ranking import candidate_ids, _exact_target_logp, _sampled_target_logp


def direct_objectives(model, batch):
    cfg = model.config.flow_ranking
    selected = select_positions(batch, cfg.max_queries)
    if not len(selected):
        raise ValueError("Direct CE needs supervised positions.")
    encoded, _ = model.encode_history(batch, ae_grad=True)
    rows, positions = selected.rows.to(model.device), selected.positions.to(model.device)
    conditions = model.condition_proj(encoded[rows, positions])
    targets = selected.target_ids.to(model.device)
    exact = cfg.catalog == "exact" or (cfg.catalog == "auto" and model.n_items - 1 <= cfg.max_exact_items)
    if exact:
        keys = model.encode_ids(torch.arange(1, model.n_items, device=model.device), grad=True)
        labels, negatives = targets - 1, None
    else:
        ids, labels, negatives = candidate_ids(targets, model.n_items, cfg.negatives, return_negative=True)
        keys = model.encode_ids(ids, grad=True)
    prediction = model.ranking_head(conditions)
    queries, unit_keys = F.normalize(prediction, dim=-1), F.normalize(keys, dim=-1)
    if exact:
        logp = checkpoint(_exact_target_logp, queries, unit_keys, labels, cfg.temperature, use_reentrant=False)
    else:
        logp = _sampled_target_logp(queries, unit_keys, labels, negatives, cfg.temperature, model.n_items - 1)
    ce = -logp.mean()
    model.last_training_stats = {"queries": len(targets), "candidates": len(keys),
        "available_queries": selected.available_queries}
    return cfg.weight * ce, {"ce": ce}


def direct_log_probabilities(model, batch):
    """Target-free direct head readout, also valid as a control-checkpoint probe."""
    if model.training or len(model.catalog) != model.n_items:
        raise RuntimeError("Call eval and refresh_catalog before direct prediction.")
    _, condition = model.encode_history(batch, ae_grad=False)
    prediction = model.ranking_head(condition)
    return (F.normalize(prediction, dim=-1) @ F.normalize(model.catalog[1:], dim=-1).T
            / model.config.flow_ranking.temperature).log_softmax(-1)
