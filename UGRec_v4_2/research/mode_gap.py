"""Where a deterministic query sits relative to training-successor modes.

Successor sets are counted on training sequences only. The validation target
is used afterwards to name the correct item and to mark whether that
transition was already observed. The final test item is never read.

The primary population is a seen transition with at least two distinct
training successors. There the frequency-weighted mean can fall between real
items. One successor has no between-mode geometry, and an unseen target is
not one of the modes being averaged.
"""

from collections import Counter, defaultdict
import argparse
from pathlib import Path

import numpy as np
import torch

from .common import ROOT, corpus, per_user, write_json


CLASSES = ("correct", "gap", "wrong_mode", "off_support")
OUTPUT = ROOT / "outputs/unified_20261028/mode_gap"


def transition_counts(train):
    counts = defaultdict(Counter)
    for seq in train:
        for left, right in zip(seq[:-1], seq[1:]):
            counts[left][right] += 1
    return counts


def successor_statistics(counts, catalog):
    """Frequency-weighted and uniform means, in the same latent space as retrieval."""
    statistics = {}
    for item, counter in counts.items():
        ids = np.fromiter(counter.keys(), dtype=np.int64)
        weight = np.fromiter(counter.values(), dtype=np.float64)
        vectors = np.asarray(catalog[ids], dtype=np.float64)
        probability = weight / weight.sum()
        entropy = float(-(probability * np.log(probability)).sum()) if len(ids) > 1 else 0.0
        statistics[int(item)] = {
            "ids": ids,
            "weighted_mean": (vectors * probability[:, None]).sum(0),
            "uniform_mean": vectors.mean(0),
            "entropy": entropy,
        }
    return statistics


def assign_tertiles(entropy, eligible):
    """Low / mid / high on the eligible users. Cuts are a property of the data."""
    values = np.asarray(entropy, dtype=np.float64)
    chosen = np.asarray(eligible, dtype=bool)
    if int(chosen.sum()) < 3:
        raise ValueError("Entropy tertiles need at least three eligible users.")
    low_cut, high_cut = np.quantile(values[chosen], [1 / 3, 2 / 3])
    labels = np.full(len(values), "ineligible", dtype=object)
    labels[chosen & (values <= low_cut)] = "low"
    labels[chosen & (values > low_cut) & (values <= high_cut)] = "mid"
    labels[chosen & (values > high_cut)] = "high"
    return labels, {"low_max_nats": float(low_cut), "high_min_exclusive_nats": float(high_cut)}


def classify(n_successors, nearest_is_target, nearest_in_support, d_mean, d_nearest):
    """Partition queries. Correct retrieval is decided before the geometry."""
    n_successors = np.asarray(n_successors)
    nearest_is_target = np.asarray(nearest_is_target, dtype=bool)
    nearest_in_support = np.asarray(nearest_in_support, dtype=bool)
    closer_to_mean = np.asarray(d_mean) < np.asarray(d_nearest)
    labels = np.full(len(n_successors), "ineligible", dtype=object)
    eligible = n_successors >= 2
    labels[eligible & nearest_is_target] = "correct"
    missed = eligible & ~nearest_is_target
    labels[missed & closer_to_mean] = "gap"
    labels[missed & ~closer_to_mean & nearest_in_support] = "wrong_mode"
    labels[missed & ~closer_to_mean & ~nearest_in_support] = "off_support"
    return labels


def nearest_ids(queries, items, block=4096):
    """Squared L2 argmin. Equal distances keep the smaller item id."""
    if queries.ndim != 2 or items.ndim != 2 or queries.shape[1] != items.shape[1]:
        raise ValueError("Queries and items must be matrices with the same width.")
    best_distance = torch.full((len(queries),), torch.inf, device=queries.device)
    best_id = torch.ones(len(queries), dtype=torch.long, device=queries.device)
    for start in range(0, len(items), block):
        chunk = items[start:start + block]
        distance = torch.cdist(queries.float(), chunk.float(), p=2).square()
        if not torch.isfinite(distance).all():
            raise ValueError("Nearest-item distances must be finite.")
        value, local = distance.min(dim=1)
        better = value < best_distance
        best_distance = torch.where(better, value, best_distance)
        best_id = torch.where(better, local + start + 1, best_id)
    return best_id, best_distance.sqrt()


def _rates(labels):
    labels = np.asarray(labels)
    total = int(len(labels))
    counts = {name: int((labels == name).sum()) for name in ("ineligible", *CLASSES)}
    shares = {name: (counts[name] / total if total else 0.0) for name in counts}
    errors = total - counts["correct"] - counts["ineligible"]
    error_shares = {
        name: (counts[name] / errors if errors else 0.0) for name in ("gap", "wrong_mode", "off_support")
    }
    return {"users": total, "counts": counts, "shares": shares, "errors": errors, "error_shares": error_shares}


def summarize_labels(labels, tertiles, closer_to_target_mean):
    labels = np.asarray(labels)
    tertiles = np.asarray(tertiles)
    closer = np.asarray(closer_to_target_mean, dtype=bool)
    output = {"all_eligible": _rates(labels[labels != "ineligible"])}
    eligible = labels != "ineligible"
    output["all_eligible"]["closer_to_mean_than_target"] = float(closer[eligible].mean()) if eligible.any() else 0.0
    for name in ("low", "mid", "high"):
        chosen = tertiles == name
        output[name] = _rates(labels[chosen])
        output[name]["closer_to_mean_than_target"] = float(closer[chosen].mean()) if chosen.any() else 0.0
    return output


def deterministic_query(model, condition):
    """One-step map with the anchor noise removed, or the direct ranking head."""
    if model.unified_config.path == "direct":
        return model.ranking_head(condition)
    zeros = torch.zeros(len(condition), model.latent_dim, device=condition.device, dtype=condition.dtype)
    return model.transport(model.flow_prior(condition, zeros), condition, 1)


def _paired_distance(query, reference):
    return (query.float() - reference.float()).norm(dim=-1)


def _encode(model, histories, targets, batch_size):
    from .common import batches

    conditions = []
    for _, batch in batches(histories, targets, batch_size):
        batch = {key: value.to(model.device) for key, value in batch.items()}
        _, condition = model.encode_history(batch, ae_grad=False)
        conditions.append(condition)
    return torch.cat(conditions)


def _sample_nearest(model, condition, items, draws, batch_users):
    unique = np.ones(len(condition), dtype=np.int32)
    if model.unified_config.path == "direct":
        return unique
    cpu_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state(model.device) if model.device.type == "cuda" else None
    torch.manual_seed(20261028)
    if model.device.type == "cuda":
        torch.cuda.manual_seed(20261028)
    try:
        for start in range(0, len(condition), batch_users):
            chosen = condition[start:start + batch_users]
            samples = model._sample_condition(chosen.repeat_interleave(draws, 0))
            flat = samples.reshape(len(chosen) * draws, model.latent_dim)
            ids, _ = nearest_ids(flat, items)
            grouped = ids.reshape(len(chosen), draws).cpu().numpy()
            unique[start:start + len(chosen)] = [len(set(row.tolist())) for row in grouped]
    finally:
        torch.set_rng_state(cpu_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state(cuda_state, model.device)
    return unique


def _user_table(histories, targets, statistics, width):
    n = len(targets)
    weighted = np.zeros((n, width), dtype=np.float64)
    uniform = np.zeros((n, width), dtype=np.float64)
    entropy = np.zeros(n, dtype=np.float64)
    support = np.zeros(n, dtype=np.int32)
    seen = np.zeros(n, dtype=bool)
    lasts = np.array([history[-1] for history in histories], dtype=np.int64)
    for index, (last, target) in enumerate(zip(lasts, targets)):
        record = statistics.get(int(last))
        if record is None:
            continue
        support[index] = len(record["ids"])
        entropy[index] = record["entropy"]
        weighted[index] = record["weighted_mean"]
        uniform[index] = record["uniform_mean"]
        seen[index] = int(target) in set(record["ids"].tolist())
    return lasts, support, seen, entropy, weighted, uniform


def _directional_gap(queries, lasts, support, statistics, catalog):
    """Direction of q is closer to the mean than to every successor direction."""
    flagged = np.zeros(len(queries), dtype=bool)
    groups = defaultdict(list)
    eligible = np.flatnonzero(support >= 2)
    for index in eligible:
        groups[int(lasts[index])].append(int(index))
    for last, indices in groups.items():
        indices = np.asarray(indices)
        vectors = catalog[statistics[last]["ids"]]
        vectors = vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
        chosen = queries[indices]
        chosen = chosen / np.maximum(np.linalg.norm(chosen, axis=1, keepdims=True), 1e-12)
        mean = statistics[last]["weighted_mean"]
        mean = mean / max(float(np.linalg.norm(mean)), 1e-12)
        flagged[indices] = (chosen @ mean) > (chosen @ vectors.T).max(axis=1)
    return flagged


def _slice_rate(mask, values):
    chosen = np.asarray(mask, dtype=bool)
    return float(np.asarray(values)[chosen].mean()) if chosen.any() else 0.0


def diagnose_checkpoint(directory, device, draws=32):
    from .audit_unified import load

    directory = Path(directory)
    model, result, histories, targets = load(directory, device)
    if result.get("test_accessed"):
        raise ValueError("This diagnostic refuses a result that already read test labels.")
    if model.unified_config.path not in {"endpoint", "direct"}:
        raise ValueError("Mode-gap diagnostic expects the frozen endpoint or direct path.")
    model.refresh_catalog()
    items = model.catalog[1:]
    catalog = model.catalog.detach().float().cpu().numpy()
    _, _, train, _, _ = corpus(model.config.data.category)
    statistics = successor_statistics(transition_counts(train), catalog)
    lasts, support, seen, entropy, weighted, uniform = _user_table(
        histories, targets, statistics, catalog.shape[1])
    primary = seen & (support >= 2)
    tertiles, cuts = assign_tertiles(entropy, primary)

    with torch.inference_mode():
        condition = _encode(model, histories, targets, 64)
        query = deterministic_query(model, condition)
        nearest, nearest_distance = nearest_ids(query, items)
        targets_t = torch.as_tensor(targets, device=model.device)
        target_distance = _paired_distance(query, model.catalog[targets_t])
        mean_distance = _paired_distance(query, torch.as_tensor(weighted, device=model.device, dtype=torch.float32))
        uniform_distance = _paired_distance(query, torch.as_tensor(uniform, device=model.device, dtype=torch.float32))
        unique = _sample_nearest(model, condition, items, draws, 32)
    nearest = nearest.cpu().numpy()
    query_np = query.detach().float().cpu().numpy()
    query_norm = np.linalg.norm(query_np, axis=1)
    nearest_distance = nearest_distance.cpu().numpy()
    target_distance = target_distance.cpu().numpy()
    mean_distance = mean_distance.cpu().numpy()
    uniform_distance = uniform_distance.cpu().numpy()
    successor_sets = {item: set(record["ids"].tolist()) for item, record in statistics.items()}
    nearest_in_support = np.array([
        int(item) in successor_sets.get(int(last), ()) for item, last in zip(nearest, lasts)
    ])
    labels = classify(support, nearest == targets, nearest_in_support, mean_distance, nearest_distance)
    uniform_labels = classify(support, nearest == targets, nearest_in_support, uniform_distance, nearest_distance)
    directional = _directional_gap(query_np, lasts, support, statistics, catalog)

    reference_name = "full_flow.npz" if model.unified_config.path == "direct" else "full_zero_noise.npz"
    match = None
    if (directory / reference_name).exists():
        saved = np.load(directory / reference_name)
        np.testing.assert_array_equal(saved["targets"], targets)
        match = float((nearest == saved["predictions"][:, 0]).mean())
        # Saved files can differ on near-ties. A large miss means the query itself changed.
        if match < 0.99:
            raise ValueError(f"Deterministic nearest item disagrees with {reference_name}: match {match:.6f}.")

    item_norm = np.linalg.norm(catalog[1:], axis=1)
    masked = np.where(primary, labels, "ineligible")
    summary = summarize_labels(masked, tertiles, mean_distance < target_distance)
    uniform_summary = summarize_labels(
        np.where(primary, uniform_labels, "ineligible"), tertiles, uniform_distance < target_distance)
    unseen = (support >= 2) & ~seen
    predictions = np.load(directory / "full_flow.npz")
    np.testing.assert_array_equal(predictions["targets"], targets)
    ndcg = per_user(predictions["predictions"], targets)["ndcg@10"]
    primary_errors = primary & (labels != "correct")
    high_errors = (tertiles == "high") & (labels != "correct")
    report = {
        "directory": str(directory.relative_to(ROOT / "outputs/unified_20261028")),
        "path": model.unified_config.path,
        "noise_scale": model.unified_config.noise_scale,
        "zero_noise_top1_match": match,
        "query": "ranking head" if model.unified_config.path == "direct" else "anchor minus one Euler step, noise 0",
        "users": int(len(targets)),
        "primary_users": int(primary.sum()),
        "primary_fraction": float(primary.mean()),
        "unseen_multi_users": int(unseen.sum()),
        "single_successor_users": int((support == 1).sum()),
        "no_successor_users": int((support == 0).sum()),
        "entropy_cuts": cuts,
        "item_norm_median": float(np.median(item_norm)),
        "query_norm_median_primary": float(np.median(query_norm[primary])),
        "distances_primary": {
            "target_median": float(np.median(target_distance[primary])),
            "weighted_mean_median": float(np.median(mean_distance[primary])),
            "nearest_median": float(np.median(nearest_distance[primary])),
        },
        "labels": summary,
        "uniform_mean_error_shares": {
            "all_eligible": uniform_summary["all_eligible"]["error_shares"],
            "high": uniform_summary["high"]["error_shares"],
        },
        "unseen_multi": _rates(labels[unseen]),
        "directional_closer_to_mean_than_every_successor": {
            "primary": _slice_rate(primary, directional),
            "primary_errors": _slice_rate(primary_errors, directional),
            "high": _slice_rate(tertiles == "high", directional),
            "high_errors": _slice_rate(high_errors, directional),
        },
        "unique_nearest": {
            "draws": 1 if model.unified_config.path == "direct" else draws,
            "primary_mean": _slice_rate(primary, unique),
            "high_mean": _slice_rate(tertiles == "high", unique),
        },
        "ndcg@10_saved_predictions": {
            "all": float(ndcg.mean()),
            "primary": float(ndcg[primary].mean()),
            "high": float(ndcg[tertiles == "high"].mean()),
            "low": float(ndcg[tertiles == "low"].mean()),
        },
    }
    arrays = {
        "targets": targets, "support": support, "seen": seen, "entropy": entropy,
        "primary": primary, "nearest": nearest, "d_target": target_distance,
        "d_mean": mean_distance, "d_nearest": nearest_distance, "query_norm": query_norm,
        "unique_nearest": unique, "directional_gap": directional,
    }
    del model, query, condition, items
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return report, arrays


def _mean_of(reports, getter):
    values = [getter(report) for report in reports]
    return {"mean": float(np.mean(values)), "seeds": [float(value) for value in values]}


def aggregate_seeds(reports):
    def shares(name, stratum):
        return _mean_of(reports, lambda report: report["labels"][stratum]["error_shares"][name])

    return {
        "primary_fraction": _mean_of(reports, lambda report: report["primary_fraction"]),
        "primary_error_shares": {name: shares(name, "all_eligible") for name in ("gap", "wrong_mode", "off_support")},
        "high_error_shares": {name: shares(name, "high") for name in ("gap", "wrong_mode", "off_support")},
        "low_error_shares": {name: shares(name, "low") for name in ("gap", "wrong_mode", "off_support")},
        "closer_to_mean_than_target": {
            "primary": _mean_of(reports, lambda report: report["labels"]["all_eligible"]["closer_to_mean_than_target"]),
            "high": _mean_of(reports, lambda report: report["labels"]["high"]["closer_to_mean_than_target"]),
        },
        "directional_high_errors": _mean_of(
            reports, lambda report: report["directional_closer_to_mean_than_every_successor"]["high_errors"]),
        "unique_nearest_high": _mean_of(reports, lambda report: report["unique_nearest"]["high_mean"]),
        "ndcg@10": {
            "high": _mean_of(reports, lambda report: report["ndcg@10_saved_predictions"]["high"]),
            "low": _mean_of(reports, lambda report: report["ndcg@10_saved_predictions"]["low"]),
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:7")
    parser.add_argument("--category", action="append")
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    categories = args.category or ["Beauty", "Sports_and_Outdoors"]
    base = ROOT / "outputs/unified_20261028"
    output = {
        "split": "validation",
        "test_accessed": False,
        "definitions": {
            "successors": "Training-sequence transitions out of the last validation-history item.",
            "mean": "Frequency-weighted Euclidean mean of those successors in the checkpoint's own latent space.",
            "primary": "Validation target is one of those successors, and there are at least two distinct successors.",
            "gap": "Nearest catalog item is not the target, and the query is closer to the mean than to that nearest item.",
            "wrong_mode": "Nearest catalog item is a different training successor, and it is at least as close as the mean.",
            "off_support": "Nearest catalog item is outside the successor set, and it is at least as close as the mean.",
            "entropy": "Shannon entropy in nats of the training successor distribution. Tertiles are taken inside the primary population.",
            "directional_gap": "After normalizing, the query is closer in angle to the mean than to every successor.",
            "query": "Flow uses the zero-noise one-step map. Direct uses the ranking head. 32-draw counts use the evaluation noise.",
        },
        "categories": {},
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for category in categories:
        models = {}
        for name in ("frozen_flow", "frozen_direct"):
            reports = []
            for seed in (2024, 2025, 2026):
                directory = base / category / f"{name}_s{seed}_e40"
                print(f"diagnosing {directory}", flush=True)
                report, arrays = diagnose_checkpoint(directory, args.device)
                reports.append(report)
                np.savez_compressed(OUTPUT / f"{category}_{name}_s{seed}.npz", **arrays)
                write_json(OUTPUT / f"{category}_{name}_s{seed}.json", report)
            models[name] = {"seeds": {str(2024 + offset): report for offset, report in enumerate(reports)},
                            "across_seeds": aggregate_seeds(reports)}
        output["categories"][category] = models
        write_json(OUTPUT / "summary.json", output)
    print(json_brief(output), flush=True)


def json_brief(output):
    brief = {}
    for category, models in output["categories"].items():
        brief[category] = {
            name: model["across_seeds"]["high_error_shares"] for name, model in models.items()
        }
    return brief


if __name__ == "__main__":
    main()
