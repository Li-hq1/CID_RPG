"""A bounded low-learning-rate continuation of a frozen-CE velocity field."""

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from joint_training.checkpoint import random_state, restore_random_state
from joint_training.config import JointConfig
from joint_training.fm import build_model
from joint_training.training import scheduler_for
from misc.artifacts import array_hash, atomic_write, json_hash, write_json
from protocol import ROOT, SELECTION_SEED, corpus, raw_features, training_dataset
from research.frozen_ce_flow import freeze_except_field, frozen_cfm_loss, frozen_state, state_hash
from run import evaluate, snapshot_source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--parent-run', default='beauty_frozen_ce_flow_rerun_s2024_20261106')
    parser.add_argument('--run-id', default='beauty_frozen_ce_flow_continue80_s2024_20261106')
    parser.add_argument('--engineering-steps', type=int)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('high')
    if args.device.startswith('cuda'):
        torch.cuda.set_per_process_memory_fraction(.2, int(args.device.split(':')[-1]))
    output = ROOT / 'outputs/Beauty' / args.run_id
    output.mkdir(parents=True, exist_ok=False)
    parent = ROOT / 'outputs/Beauty' / args.parent_run
    for relative, digest in json.loads((parent / 'source_sha256.json').read_text()).items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == digest, relative
    saved = torch.load(parent / 'best_flow.pt', map_location='cpu', weights_only=True)
    cfg = JointConfig.from_dict(saved['parent_config'])
    cfg.training.device = args.device
    directory, mapping, train, histories, targets = corpus('Beauty')
    raw = raw_features(directory, len(mapping['id2item']) - 1)
    assert saved['source']['raw_sha256'] == array_hash(raw)
    assert saved['source']['train_sha256'] == json_hash(train)
    assert saved['source']['item_order_sha256'] == json_hash(mapping['id2item'])
    model = build_model(cfg, raw).to(args.device)
    model.load_state_dict(saved['model'], strict=True)
    parameters = freeze_except_field(model)
    model.cache_features()
    model.refresh_catalog()
    before = state_hash(frozen_state(model))
    assert before == saved['frozen_state_sha256']
    selected = np.random.default_rng(SELECTION_SEED).permutation(len(targets))[:4096]
    audit = selected[:8] if args.engineering_steps else np.arange(len(targets))
    selection = selected[:8] if args.engineering_steps else selected
    fixed_ce, fixed_predictions = evaluate(model, histories, targets, audit, source='ranking')
    baseline, predictions = evaluate(model, histories, targets, selection, source='flow')
    generator = torch.Generator()
    generator.set_state(saved['loader_rng'])
    loader = DataLoader(training_dataset(train), batch_size=cfg.recommender.train_batch_size,
                        shuffle=True, generator=generator)
    optimizer = torch.optim.AdamW(parameters, lr=3e-4, weight_decay=cfg.recommender.weight_decay)
    optimizer.load_state_dict(saved['optimizer'])
    for group in optimizer.param_groups:
        group['lr'] = group['initial_lr'] = 3e-4
    settings = deepcopy(cfg.recommender)
    settings.lr = 3e-4
    scheduler = scheduler_for(optimizer, settings, 80 * len(loader))
    restore_random_state(saved['rng'], model.device)
    source_hash = snapshot_source(output)
    shutil.copy2(__file__, output / 'source/continue_frozen_flow.py')
    write_json(output / 'experiment.json', {
        'scheme': 'frozen_velocity_continue80', 'parent': str(parent), 'seed': 2024,
        'parent_checkpoint_sha256': hashlib.sha256((parent / 'best_flow.pt').read_bytes()).hexdigest(),
        'source_sha256': source_hash, 'research_script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'source': saved['source'], 'frozen_state_sha256': before,
        'additional_epochs': 80, 'additional_expected_updates': 80 * len(loader),
        'lr': 3e-4, 'optimizer_state_resumed': True, 'rng_resumed': True,
        'selection_indices_sha256': array_hash(selection),
        'engineering_check': bool(args.engineering_steps), 'test_accessed': False,
        'budget_note': 'Additional 80 velocity epochs after CE and 40 field epochs; restart low-LR cosine schedule.'})
    best = {'score': baseline['ndcg@10'], 'epoch': 40}
    saved['best'] = best
    atomic_write(output / 'best_flow.pt', lambda stream: torch.save(saved, stream))
    np.savez_compressed(output / 'selection_flow.npz', indices=selection, predictions=predictions)
    records, steps = [], 0
    started = time.time()
    for epoch in range(41, 121):
        losses = []
        for ids, labels in loader:
            batch = {'input_ids': ids.to(args.device), 'labels': labels.to(args.device),
                     'attention_mask': (ids > 0).to(args.device),
                     'seq_lens': (ids > 0).sum(-1).to(args.device)}
            optimizer.zero_grad(set_to_none=True)
            loss = frozen_cfm_loss(model, batch)
            if not torch.isfinite(loss):
                raise RuntimeError('Non-finite continuation loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, cfg.recommender.max_grad_norm, error_if_nonfinite=True)
            optimizer.step()
            scheduler.step()
            losses.append(float(loss.detach()))
            steps += 1
            if args.engineering_steps and steps >= args.engineering_steps:
                break
        row = {'epoch': epoch, 'additional_updates': steps, 'train_mse': float(np.mean(losses))}
        if epoch % 5 == 0 or args.engineering_steps:
            metrics, predictions = evaluate(model, histories, targets, selection, source='flow')
            row['validation_selection'] = metrics
            if metrics['ndcg@10'] > best['score']:
                best = {'score': metrics['ndcg@10'], 'epoch': epoch}
                checkpoint = {'method': 'frozen_ce_cfm_research', 'model': model.state_dict(),
                              'parent_config': cfg.to_dict(), 'source': saved['source'], 'epoch': epoch,
                              'best': best, 'steps': saved['steps'] + steps,
                              'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
                              'rng': random_state(model.device), 'loader_rng': generator.get_state(),
                              'frozen_state_sha256': before}
                atomic_write(output / 'best_flow.pt', lambda stream: torch.save(checkpoint, stream))
                np.savez_compressed(output / 'selection_flow.npz', indices=selection, predictions=predictions)
        records.append(row)
        write_json(output / 'metrics.json', records)
        print(json.dumps(row), flush=True)
        if args.engineering_steps:
            break
    atomic_write(output / 'last_model.pt', lambda stream: torch.save(model.state_dict(), stream))
    assert state_hash(frozen_state(model)) == before
    last_selection, _ = evaluate(model, histories, targets, selection, source='flow')
    chosen = torch.load(output / 'best_flow.pt', map_location='cpu', weights_only=True)
    model.load_state_dict(chosen['model'], strict=True)
    assert state_hash(frozen_state(model)) == before
    with np.load(output / 'selection_flow.npz') as expected:
        metrics, predictions = evaluate(model, histories, targets, expected['indices'], source='flow')
        np.testing.assert_array_equal(predictions, expected['predictions'])
        assert metrics['ndcg@10'] == best['score']
    final, predictions = evaluate(model, histories, targets, audit, source='flow')
    np.savez_compressed(output / 'full_flow.npz', indices=audit, targets=targets[audit], predictions=predictions)
    ce_after, predictions = evaluate(model, histories, targets, audit, source='ranking')
    np.testing.assert_array_equal(fixed_predictions, predictions)
    assert ce_after == fixed_ce
    result = {'best': best, 'validation': final, 'frozen_ce': fixed_ce,
              'baseline_selection': baseline, 'last_selection': last_selection,
              'additional_updates': steps, 'total_velocity_updates': saved['steps'] + steps,
              'frozen_state_verified': True, 'ce_predictions_unchanged': True,
              'checkpoint_reload_verified': True, 'engineering_check': bool(args.engineering_steps),
              'test_accessed': False, 'seconds': time.time() - started}
    write_json(output / 'results.json', result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
