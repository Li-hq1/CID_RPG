"""Fixed-budget diffusion candidate retrieval and aggregation."""

import math

import torch
import torch.nn.functional as F


def validate_eval_sampling(config):
    choices = {
        "eval_sampling_mode": ("knn", {"knn", "diffusion_multi"}),
        "eval_candidate_metric": ("l2", {"l2", "cosine", "inner_product"}),
        "eval_candidate_ranking": ("support", {"support", "first_seen", "min_distance"}),
    }
    for key, (default, allowed) in choices.items():
        if config.get(key, default) not in allowed:
            raise ValueError(f"{key} must be one of: {', '.join(sorted(allowed))}.")
    for key, default in (("eval_num_samples", 8), ("eval_candidates_per_sample", 20)):
        value = config.get(key, default)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{key} must be a positive integer.")
    temperature = config.get("eval_support_temperature", 1.0)
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("eval_support_temperature must be finite and positive.")
    if not isinstance(config.get("eval_log_sampling_stats", False), bool):
        raise ValueError("eval_log_sampling_stats must be boolean.")


def _distances(samples, items, metric):
    samples, items = samples.float(), items.float()
    if metric == "l2":
        return torch.cdist(samples, items, p=2).square()
    if metric == "cosine":
        return 1 - F.normalize(samples, dim=-1) @ F.normalize(items, dim=-1).T
    return -(samples @ items.T)


def _nearest_candidates(samples, items, depth, metric):
    # Stable block merges resolve equal distances by ascending item ID.
    nearest_ids = torch.empty((len(samples), 0), device=samples.device, dtype=torch.long)
    nearest_distances = torch.empty((len(samples), 0), device=samples.device)
    for start in range(0, len(items), 4096):
        distances = _distances(samples, items[start:start + 4096], metric)
        if not torch.isfinite(distances).all():
            raise ValueError("Evaluation candidate distances must be finite.")
        ids = torch.arange(start + 1, start + 1 + distances.shape[1], device=samples.device)
        ids = ids.expand(len(samples), -1)
        distances = torch.cat((nearest_distances, distances), dim=1)
        ids = torch.cat((nearest_ids, ids), dim=1)
        order = distances.argsort(dim=1, stable=True)[:, :depth]
        nearest_distances = distances.gather(1, order)
        nearest_ids = ids.gather(1, order)
    return nearest_ids, nearest_distances


def aggregate_diffusion_candidates(samples, items, max_k, config):
    """Return unique item IDs [B, min(K, N), 1] and per-user statistics.

    Items excludes padding; sample axes are [user, independent draw, embedding].
    """
    n_items = len(items)
    if n_items == 0:
        raise ValueError("Diffusion evaluation requires at least one non-padding item.")
    k = min(max_k, n_items)
    initial_depth = min(config.get("eval_candidates_per_sample", 20), n_items)
    retrieval_limit = max(initial_depth, k)
    metric = config.get("eval_candidate_metric", "l2")
    ranking = config.get("eval_candidate_ranking", "support")
    temperature = config.get("eval_support_temperature", 1.0)
    predictions = []
    statistics = []
    for user_samples in samples:
        nearest_ids, nearest_distances = _nearest_candidates(
            user_samples, items, retrieval_limit, metric
        )
        depth = initial_depth
        discovery = [nearest_ids[:, :depth].reshape(-1)]
        while nearest_ids[:, :depth].unique().numel() < k:
            next_depth = min(retrieval_limit, max(depth + 1, depth * 2))
            discovery.append(nearest_ids[:, depth:next_depth].reshape(-1))
            depth = next_depth
        local_ids = nearest_ids[:, :depth]
        candidate_ids, inverse = local_ids.reshape(-1).unique(sorted=True, return_inverse=True)
        min_distances = _distances(user_samples, items[candidate_ids - 1], metric).amin(dim=0)
        # Candidate IDs start sorted, providing the final tie-breaker for all ranks.
        order = min_distances.argsort(stable=True)
        if ranking == "support":
            log_weights = torch.full(
                (len(user_samples), len(candidate_ids)), -torch.inf,
                device=samples.device, dtype=torch.float64,
            )
            log_weights.scatter_(
                1, inverse.reshape_as(local_ids),
                -nearest_distances[:, :depth].double() / temperature,
            )
            log_support = torch.logsumexp(log_weights, dim=0)
            order = order[log_support[order].argsort(descending=True, stable=True)]
        elif ranking == "first_seen":
            discovered = torch.cat(discovery)
            positions = torch.arange(len(discovered), device=samples.device)
            first_positions = torch.full_like(candidate_ids, len(discovered))
            first_positions.scatter_reduce_(
                0, torch.searchsorted(candidate_ids, discovered), positions,
                reduce="amin", include_self=True,
            )
            order = first_positions.argsort(stable=True)
        predictions.append(candidate_ids[order[:k]])
        occurrences = len(user_samples) * depth
        statistics.append((len(candidate_ids), 1 - len(candidate_ids) / occurrences,
                           depth, len(user_samples), occurrences))
    values = torch.tensor(statistics, device=samples.device, dtype=torch.float32)
    stats = dict(zip(
        ("unique_candidates", "deduplication_rate", "retrieval_depth", "num_samples",
         "candidate_occurrences"), values.unbind(dim=1),
    ))
    return torch.stack(predictions).unsqueeze(-1), stats
