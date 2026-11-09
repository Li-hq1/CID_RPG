"""FlowRec's Prior + alpha CFM + beta Align on the existing UGRec network.

Our RF clock s=1-t runs from source (s=1) to item (s=0). Thus its velocity is
x0-x1 and xhat1=xs-s*v. CFM and Align share exactly one random-time forward.
The control's conditional network, normalized anchor and cosine/T CE remain.
This is a loss adaptation, not an official FlowRec reproduction.
"""

import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .contrastive import select_positions
from .flow_ranking import candidate_ids, _exact_target_logp, _sampled_target_logp


def bridge_terms(base, target, fraction, velocity):
    state = (1 - fraction) * target + fraction * base
    cfm = (velocity - (base - target)).square().mean()
    endpoint = state - fraction * velocity
    return state, cfm, endpoint


def flowrec_objectives(model, batch):
    cfg = model.config.flow_ranking
    selection = select_positions(batch, cfg.max_queries)
    if not len(selection):
        raise ValueError("FlowRec needs a supervised history position.")
    encoded, _ = model.encode_history(batch, ae_grad=True)
    condition = model.condition_proj(encoded[selection.rows, selection.positions])
    targets = selection.target_ids.to(model.device)
    exact = cfg.catalog == "exact" or (cfg.catalog == "auto" and model.n_items - 1 <= cfg.max_exact_items)
    if exact:
        keys = model.encode_ids(torch.arange(1, model.n_items, device=model.device), grad=True)
        labels, negatives = targets - 1, None
    else:
        ids, labels, negatives = candidate_ids(targets, model.n_items, cfg.negatives, return_negative=True)
        keys = model.encode_ids(ids, grad=True)
    unit_keys = F.normalize(keys, dim=-1)

    def ce(query):
        unit_query = F.normalize(query, dim=-1)
        if exact:
            return -checkpoint(_exact_target_logp, unit_query, unit_keys, labels,
                               cfg.temperature, use_reentrant=False).mean()
        return -_sampled_target_logp(unit_query, unit_keys, labels, negatives,
                                    cfg.temperature, model.n_items - 1).mean()

    base = model.flow_prior(condition)
    prior = ce(base)
    zero = prior.new_zeros(())
    cfm = align = zero
    state = endpoint = base
    velocity = torch.zeros_like(base)
    if cfg.cfm_weight > 0 or cfg.align_weight > 0:
        steps = model.diffloss.train_diffusion.num_timesteps
        times = torch.randint(steps, (len(targets),), device=model.device)
        fraction = times[:, None].to(base.dtype) / (steps - 1)
        target = keys[labels]
        state = (1 - fraction) * target + fraction * base
        velocity = model.diffloss.net(state, times, condition)
        _, cfm, endpoint = bridge_terms(base, target, fraction, velocity)
        if cfg.align_weight > 0:
            align = ce(endpoint)
    # No endpoint detaches: all three signals jointly train the shared AE/encoder.
    loss = prior + cfg.cfm_weight * cfm + cfg.align_weight * align
    with torch.no_grad():
        model.last_training_stats = {
            "queries": len(targets), "candidates": len(keys),
            "available_queries": selection.available_queries,
            "prior_rms": float(base.square().mean().sqrt()),
            "target_rms": float(keys[labels].square().mean().sqrt()),
            "velocity_rms": float(velocity.square().mean().sqrt()),
            "endpoint_rms": float(endpoint.square().mean().sqrt()),
            "prior_target_cosine": float(F.cosine_similarity(base, keys[labels]).mean()),
            "bridge_endpoint_cosine": float(F.cosine_similarity(endpoint, keys[labels]).mean()),
        }
    return cfg.weight * loss, {"prior": prior, "cfm": cfm, "align": align,
                               "weighted_cfm": cfg.cfm_weight * cfm,
                               "weighted_align": cfg.align_weight * align}
