"""Decode each frozen-CFM draw to an item and audit candidate-pool recall."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from joint_training.config import JointConfig
from joint_training.fm import build_model
from misc.artifacts import write_json
from protocol import ROOT, corpus, raw_features, ranking_metrics, validation_batches
from run import evaluate


@torch.no_grad()
def sample_item_lists(model, histories, targets, indices, draws, seed):
    generator = torch.Generator(device=model.device).manual_seed(seed)
    catalog = F.normalize(model.catalog[1:].float(), dim=-1)
    sample_lists, score_lists = [], []
    model.eval()
    for _, batch in validation_batches(histories, targets, 32, indices):
        batch = {name: value.to(model.device) for name, value in batch.items()}
        rank_all, fm_all = model.history_branches(batch)
        lengths = batch["seq_lens"].long()
        rows = torch.arange(len(lengths), device=model.device)
        rank = rank_all[rows, lengths - 1]
        condition = fm_all[rows, lengths - 1]
        expanded_rank = rank.repeat_interleave(draws, dim=0)
        expanded_condition = condition.repeat_interleave(draws, dim=0)
        noise = torch.randn((len(expanded_rank), model.latent_dim),
                            generator=generator, device=model.device)
        state = model.flow_prior(expanded_rank, noise)
        state = model.transport(state, expanded_condition, model.fm_config.sampling_steps)
        queries = F.normalize(state.float(), dim=-1)
        ids, scores = [], []
        for start in range(0, len(queries), 256):
            similarity = queries[start:start + 256] @ catalog.T
            value, index = similarity.max(dim=-1)
            ids.append(index + 1)
            scores.append(value)
        ids = torch.cat(ids).view(len(lengths), draws)
        scores = torch.cat(scores).view(len(lengths), draws)
        ranked_ids, ranked_scores = [], []
        for item_row, score_row in zip(ids, scores):
            unique, inverse, counts = item_row.unique(return_inverse=True, return_counts=True, sorted=True)
            sums = torch.zeros(len(unique), device=model.device).scatter_add_(0, inverse, score_row)
            means = sums / counts
            order = torch.arange(len(unique), device=model.device)
            order = order[unique[order].argsort(stable=True)]
            order = order[means[order].argsort(descending=True, stable=True)]
            order = order[counts[order].argsort(descending=True, stable=True)]
            ranked_ids.append(unique[order])
            ranked_scores.append(counts[order].float())
        sample_lists.extend([row.cpu().numpy() for row in ranked_ids])
        score_lists.extend([row.cpu().numpy() for row in ranked_scores])
    width = max(map(len, sample_lists))
    out_ids = np.zeros((len(sample_lists), width), dtype=np.int64)
    out_scores = np.zeros((len(sample_lists), width), dtype=np.float32)
    for row, (items, scores) in enumerate(zip(sample_lists, score_lists)):
        out_ids[row, :len(items)] = items
        out_scores[row, :len(scores)] = scores
    return out_ids, out_scores


def quota_list(ce, fm, ce_slots, k=10):
    output = []
    for row_ce, row_fm in zip(ce, fm):
        selected, seen = [], set()
        for item in list(row_ce[:ce_slots]) + list(row_fm[:k - ce_slots]) + list(row_ce) + list(row_fm):
            item = int(item)
            if item > 0 and item not in seen:
                selected.append(item)
                seen.add(item)
            if len(selected) == k:
                break
        output.append(selected)
    return np.asarray(output, dtype=np.int64)


def pool_recall(ce, fm, targets, ce_depth, fm_depth):
    ce_hit = (ce[:, :ce_depth] == targets[:, None]).any(1)
    fm_hit = (fm[:, :fm_depth] == targets[:, None]).any(1)
    return {
        "ce_recall": float(ce_hit.mean()),
        "fm_recall": float(fm_hit.mean()),
        "union_recall": float((ce_hit | fm_hit).mean()),
        "fm_only_hits": int((fm_hit & ~ce_hit).sum()),
        "ce_only_hits": int((ce_hit & ~fm_hit).sum()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--run-id", default="beauty_frozen_ce_flow_rerun_s2024_20261106")
    parser.add_argument("--draws", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20261028)
    parser.add_argument("--output", default="outputs/dual_candidate_samples_20261106")
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("high")
    if args.device.startswith("cuda"):
        torch.cuda.set_per_process_memory_fraction(.20, int(args.device.split(":")[-1]))

    run = ROOT / "outputs/Beauty" / args.run_id
    checkpoint_path = run / "best_flow.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    cfg = JointConfig.from_dict(checkpoint["parent_config"])
    cfg.training.device = args.device
    directory, mapping, _, histories, targets = corpus("Beauty")
    raw = raw_features(directory, len(mapping["id2item"]) - 1)
    model = build_model(cfg, raw).to(args.device).eval()
    model.load_state_dict(checkpoint["model"], strict=True)
    model.refresh_catalog()

    ce_run = ROOT / "outputs/Beauty/beauty_ce_only_s2024_20261106"
    with np.load(ce_run / "full_flow.npz") as saved:
        ce = saved["predictions"].copy()
        np.testing.assert_array_equal(saved["targets"], targets)
    indices = np.arange(len(targets))
    fm_items, fm_counts = sample_item_lists(model, histories, targets, indices, args.draws, args.seed)
    fm_items = np.where(fm_counts > 0, fm_items, 0)

    ce_top42, reranked_union = [], []
    normalized_catalog = F.normalize(model.catalog[1:].float(), dim=-1)
    offset = 0
    for _, batch in validation_batches(histories, targets, 32, indices):
        batch = {name: value.to(model.device) for name, value in batch.items()}
        rank_all, _ = model.history_branches(batch)
        lengths = batch["seq_lens"].long()
        rows = torch.arange(len(lengths), device=model.device)
        query = F.normalize(model.ranking_head(rank_all[rows, lengths - 1]).float(), dim=-1)
        scores = query @ normalized_catalog.T
        ce_top42.extend((scores.topk(42, dim=-1).indices + 1).cpu().numpy())
        for row_index in range(len(batch["input_ids"])):
            candidate_ids = fm_items[offset + row_index]
            pool = np.unique(np.concatenate((ce[offset + row_index, :10], candidate_ids[candidate_ids > 0])))
            candidate_tensor = torch.as_tensor(pool - 1, device=model.device)
            ranked = scores[row_index].index_select(0, candidate_tensor).topk(min(10, len(pool))).indices
            reranked_union.append(pool[ranked.cpu().numpy()])
        offset += len(batch["input_ids"])
    ce_top42 = np.asarray(ce_top42, dtype=np.int64)
    reranked_union = np.asarray(reranked_union, dtype=np.int64)

    aggregate_metrics, aggregate = evaluate(model, histories, targets, indices, source="flow")
    ce_metrics, ce_check = evaluate(model, histories, targets, indices, source="ranking")
    np.testing.assert_array_equal(ce, ce_check)
    fm_top10 = aggregate[:, :10]
    output = ROOT / args.output
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output / "sample_candidates.npz", targets=targets,
                        ce_top10=ce[:, :10], fm_sample_items=fm_items,
                        fm_sample_counts=fm_counts, fm_aggregate_top10=fm_top10,
                        ce_top42=ce_top42, reranked_union_top10=reranked_union)

    summary = {
        "run_id": args.run_id,
        "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
        "draws": args.draws,
        "sampling_seed": args.seed,
        "validation_users": len(targets),
        "test_accessed": False,
        "ce_top10": ce_metrics,
        "fm_aggregate_top10": aggregate_metrics,
        "sample_top1_pool": {
            "unique_items_mean": float((fm_counts > 0).sum(1).mean()),
            "unique_items_median": float(np.median((fm_counts > 0).sum(1))),
            "ce10_plus_fm_unique_draw_items_recall": pool_recall(ce, fm_items, targets, 10, args.draws),
            "ce10_plus_fm_frequency_top10_recall": pool_recall(ce, fm_items, targets, 10, 10),
        },
        "candidate_expansion_reranked_by_ce": {
            "unique_pool_size_mean": float(np.mean([
                len(set(ce[row, :10]) | set(fm_items[row][fm_items[row] > 0]))
                for row in range(len(targets))
            ])),
            "ce_top42": ranking_metrics(ce_top42, targets),
            "ce_top10_plus_fm_sample_pool_reranked_by_ce": ranking_metrics(reranked_union, targets),
            "interpretation": "Equal maximum candidate budget; both lists ranked by the CE scoring head, so differences test candidate availability rather than FM final ranking.",
        },
        "candidate_recall_equalized_by_pool_size": {},
        "quota_lists": {},
        "draw_semantics": "Each Euler endpoint is decoded independently by exact cosine nearest neighbor; FM candidates are ordered by sample frequency then mean matched cosine.",
        "limitations": "Single next-item validation target; no claim of multi-positive coverage or independent test performance.",
    }
    budgets = np.array([
        len(set(ce[row, :10]) | set(fm_items[row][fm_items[row] > 0]))
        for row in range(len(targets))
    ])
    ce_equal_budget_hit = np.array([
        bool(np.any(ce_top42[row, :budget] == targets[row]))
        for row, budget in enumerate(budgets)
    ])
    union_equal_budget_hit = np.array([
        bool(targets[row] in set(ce[row, :10]) | set(fm_items[row][fm_items[row] > 0]))
        for row in range(len(targets))
    ])
    summary["candidate_recall_equalized_by_pool_size"] = {
        "mean_unique_candidate_budget": float(budgets.mean()),
        "ce_top_same_number_candidates_recall": float(ce_equal_budget_hit.mean()),
        "ce10_plus_all_fm_draw_items_recall": float(union_equal_budget_hit.mean()),
        "additional_fm_hits_over_equal_budget_ce": int((union_equal_budget_hit & ~ce_equal_budget_hit).sum()),
        "ce_hits_lost_by_fm_pool_over_equal_budget_ce": int((ce_equal_budget_hit & ~union_equal_budget_hit).sum()),
    }
    for ce_slots in (10, 8, 6, 5, 4, 2, 0):
        prediction = quota_list(ce, fm_items, ce_slots)
        summary["quota_lists"][f"ce_{ce_slots}_fm_{10-ce_slots}"] = ranking_metrics(prediction, targets)
    write_json(output / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
