"""Paired per-draw candidate audit with exact unique-candidate budget controls."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time

import numpy as np
import torch
import torch.nn.functional as F

from joint_training.config import JointConfig
from joint_training.fm import build_model
from misc.artifacts import array_hash, json_hash, write_json
from protocol import ROOT, corpus, ranking_metrics, raw_features, validation_batches
from research.frozen_ce_flow import frozen_state, state_hash


def rank_draws(items, similarities):
    """Frequency, mean nearest-item cosine, then ID; padding is excluded."""
    output = np.zeros_like(items)
    for row, (ids, cosines) in enumerate(zip(items, similarities)):
        unique, inverse, counts = np.unique(ids, return_inverse=True, return_counts=True)
        means = np.bincount(inverse, weights=cosines) / counts
        order = np.lexsort((unique, -means, -counts))
        output[row, :len(unique)] = unique[order]
    return output


def build_pool(ce, additional, size, protected=10):
    """Protect CE prefix, admit unique generated IDs, then CE-fill to size."""
    output = []
    for base, extra in zip(ce, additional):
        seen, selected = set(), []
        for item in list(base[:protected]) + list(extra) + list(base):
            item = int(item)
            if item > 0 and item not in seen:
                selected.append(item)
                seen.add(item)
            if len(selected) == size:
                break
        if len(selected) != size:
            raise ValueError('Insufficient distinct catalog items for requested budget')
        output.append(selected)
    return np.asarray(output, dtype=np.int64)


def paired_hits(candidate, baseline, targets):
    new = (candidate == targets[:, None]).any(1)
    old = (baseline == targets[:, None]).any(1)
    delta = new.astype(float) - old.astype(float)
    return {'recall': float(new.mean()), 'ce_recall': float(old.mean()),
            'recall_difference': float(delta.mean()), 'gained_users': int((new & ~old).sum()),
            'lost_users': int((old & ~new).sum()),
            'paired_user_standard_error': float(delta.std(ddof=1) / np.sqrt(len(delta)))}


@torch.no_grad()
def infer(model, histories, targets, indices, draws=32, seed=20261028):
    noise_rng = torch.Generator(device=model.device).manual_seed(seed)
    category_rng = torch.Generator(device=model.device).manual_seed(seed)
    keys = F.normalize(model.catalog[1:].float(), dim=-1)
    stored = {key: [] for key in ('ce42', 'prior_ids', 'prior_cos', 'fm4_ids', 'fm4_cos',
                                  'fm16_ids', 'fm16_cos', 'categorical_ids', 'fm4_aggregate',
                                  'fm16_aggregate')}
    for _, batch in validation_batches(histories, targets, 32, indices):
        batch = {name: value.to(model.device) for name, value in batch.items()}
        # No use of labels or future targets in generating any candidate.
        logp = model.mixture_log_probabilities(batch, source='ranking')
        ce10 = logp.topk(10, -1).indices + 1
        top42 = logp.topk(42, -1).indices + 1
        # Lock the original CE top-10 prefix, including numerical tie ordering.
        ce42 = build_pool(top42.cpu().numpy(), ce10.cpu().numpy(), 42, protected=0)
        stored['ce42'].append(ce42)
        sampled = torch.multinomial(logp.exp(), draws, replacement=True, generator=category_rng) + 1
        stored['categorical_ids'].append(sampled.cpu().numpy())
        rank_all, fm_all = model.history_branches(batch)
        lengths = batch['seq_lens'].long()
        rows = torch.arange(len(lengths), device=model.device)
        rank = rank_all[rows, lengths - 1].repeat_interleave(draws, 0)
        cond = fm_all[rows, lengths - 1].repeat_interleave(draws, 0)
        noise = torch.randn((len(rank), model.latent_dim), device=model.device, generator=noise_rng)
        source = model.flow_prior(rank, noise)
        for name, state in [('prior', source), ('fm4', model.transport(source, cond, 4)),
                            ('fm16', model.transport(source, cond, 16))]:
            queries = F.normalize(state.float(), dim=-1).reshape(len(rows), draws, -1)
            ids, cosines, probability_sum = [], [], None
            for start in range(0, draws, 8):
                similarity = queries[:, start:start + 8] @ keys.T
                value, index = similarity.max(-1)
                ids.append(index + 1)
                cosines.append(value)
                if name != 'prior':
                    part = (similarity / model.fm_config.temperature).softmax(-1).sum(1)
                    probability_sum = part if probability_sum is None else probability_sum + part
            stored[f'{name}_ids'].append(torch.cat(ids, 1).cpu().numpy())
            stored[f'{name}_cos'].append(torch.cat(cosines, 1).cpu().numpy())
            if name != 'prior':
                stored[f'{name}_aggregate'].append((probability_sum.topk(10, -1).indices + 1).cpu().numpy())
        if sum(map(len, stored['ce42'])) % 4096 == 0:
            print(json.dumps({'inferred_users': sum(map(len, stored['ce42']))}), flush=True)
    return {name: np.concatenate(values) for name, values in stored.items()}


def summarize(arrays, targets):
    ce = arrays['ce42']
    results = {'ce_top10': ranking_metrics(ce[:, :10], targets), 'methods': {}}
    for name in ('prior', 'fm4', 'fm16', 'categorical'):
        ids = arrays[f'{name}_ids']
        cos = arrays.get(f'{name}_cos', np.zeros_like(ids, dtype=float))
        ranked = rank_draws(ids, cos)
        unique_count = (ranked > 0).sum(1)
        union_hits, ce_size_hits, sizes = [], [], []
        for row, target in enumerate(targets):
            pool = set(ce[row, :10]) | set(ranked[row][ranked[row] > 0])
            sizes.append(len(pool))
            union_hits.append(target in pool)
            ce_size_hits.append(target in ce[row, :len(pool)])
        a, b = np.asarray(union_hits), np.asarray(ce_size_hits)
        entry = {'unique_draw_items_mean': float(unique_count.mean()),
                 'variable_union_pool': {'mean_unique_candidates': float(np.mean(sizes)),
                      'recall': float(a.mean()), 'ce_same_user_budget_recall': float(b.mean()),
                      'gained_users': int((a & ~b).sum()), 'lost_users': int((b & ~a).sum())},
                 'fixed_unique_budgets': {}, 'fixed_final10_ce8_plus_samples2': {}}
        for size in (20, 42):
            pool = build_pool(ce, ranked, size)
            entry['fixed_unique_budgets'][str(size)] = paired_hits(pool, ce[:, :size], targets)
        final = build_pool(ce, ranked, 10, protected=8)
        entry['fixed_final10_ce8_plus_samples2'] = ranking_metrics(final, targets)
        if name != 'categorical' and name != 'prior':
            entry['aggregate_softmax_top10'] = ranking_metrics(arrays[f'{name}_aggregate'], targets)
        results['methods'][name] = entry
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--run-id', default='beauty_frozen_ce_flow_rerun_s2024_20261106')
    parser.add_argument('--output', required=True)
    parser.add_argument('--engineering-users', type=int)
    args = parser.parse_args()
    output = ROOT / args.output
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('high')
    if args.device.startswith('cuda'):
        torch.cuda.set_per_process_memory_fraction(.2, int(args.device.split(':')[-1]))
    run = ROOT / 'outputs/Beauty' / args.run_id
    for relative, digest in json.loads((run / 'source_sha256.json').read_text()).items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == digest, relative
    checkpoint = torch.load(run / 'best_flow.pt', map_location='cpu', weights_only=True)
    cfg = JointConfig.from_dict(checkpoint['parent_config'])
    cfg.training.device = args.device
    directory, mapping, train, histories, targets = corpus('Beauty')
    raw = raw_features(directory, len(mapping['id2item']) - 1)
    assert checkpoint['source']['raw_sha256'] == array_hash(raw)
    assert checkpoint['source']['train_sha256'] == json_hash(train)
    assert checkpoint['source']['item_order_sha256'] == json_hash(mapping['id2item'])
    model = build_model(cfg, raw).to(args.device).eval()
    model.load_state_dict(checkpoint['model'], strict=True)
    assert state_hash(frozen_state(model)) == checkpoint['frozen_state_sha256']
    model.refresh_catalog()
    indices = np.arange(args.engineering_users or len(targets))
    shutil.copy2(__file__, output / 'candidate_budget_study.py')
    started = time.time()
    arrays = infer(model, histories, targets, indices)
    ce_run = ROOT / 'outputs/Beauty/beauty_ce_only_s2024_20261106'
    with np.load(ce_run / 'full_flow.npz') as parent:
        np.testing.assert_array_equal(targets, parent['targets'])
        np.testing.assert_array_equal(arrays['ce42'][:, :10], parent['predictions'][indices])
    np.savez_compressed(output / 'predictions.npz', targets=targets[indices], indices=indices, **arrays)
    result = summarize(arrays, targets[indices])
    result.update({'run_id': args.run_id, 'selected_epoch': checkpoint['epoch'],
                   'checkpoint_sha256': hashlib.sha256((run / 'best_flow.pt').read_bytes()).hexdigest(),
                   'research_script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   'selection_note': 'Checkpoint selected only by existing FM4 aggregate validation NDCG@10.',
                   'candidate_budgets': [20, 42], 'draws': 32, 'steps': [4, 16],
                   'source_noise_paired': True, 'test_accessed': False, 'seed': 2024,
                   'sampling_seed': 20261028, 'validation_users': len(indices),
                   'engineering_check': bool(args.engineering_users),
                   'ce_predictions_verified': True, 'seconds': time.time() - started,
                   'cost': {'fm4_nfe_per_user': 128, 'fm16_nfe_per_user': 512,
                            'prior_and_categorical_nfe_per_user': 0},
                   'limitations': 'Single-seed full validation includes selection users. One target/user. User-level SE is not seed stability. No multi-interest or test claim.'})
    write_json(output / 'summary.json', result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
