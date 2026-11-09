"""Prepare, execute and audit finite, hypothesis-driven campaigns.

New campaigns use Beauty, Sports, and Toys. Historical campaigns that already
contain CDs keep that recorded set when they are summarized.
"""

import argparse
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
from datetime import datetime, timezone

import numpy as np

from misc.artifacts import array_hash, json_hash, write_json
from protocol import CATEGORIES, ROOT, SELECTION_SEED, corpus, raw_features
from .schema import ACTIVE_DATASETS, declared_epochs, read_json, resolve_configs, target_aliases, validate_spec

CAMPAIGNS = ROOT / 'outputs/tuning'


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def source_files():
    files = []
    for directory in ('joint_training', 'models', 'compression', 'misc', 'tuning', 'scripts'):
        files.extend(p for p in (ROOT / directory).rglob('*')
                     if p.is_file() and p.suffix in {'.py', '.sh'} and '__pycache__' not in p.parts)
    files.extend(ROOT / name for name in ('run.py', 'protocol.py', 'configs/final.json', 'configs/control.json'))
    return sorted(files)


def source_identity():
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files()}
    return {'sha256': json_hash(hashes), 'files': hashes}


def preflight():
    result = {}
    for alias, category in CATEGORIES.items():
        directory, mapping, train, histories, targets = corpus(category)
        raw = raw_features(directory, len(mapping['id2item']) - 1)
        result[alias] = {
            'category': category, 'users': len(targets), 'items': len(raw),
            'training_windows': sum(max(1, len(seq) - 50) for seq in train if len(seq) >= 2),
            'raw_sha256': array_hash(raw), 'train_sha256': json_hash(train),
            'item_order_sha256': json_hash(mapping['id2item']),
            'validation_histories_sha256': json_hash(histories),
            'validation_targets_sha256': array_hash(targets),
            'selection_indices_sha256': array_hash(np.random.default_rng(SELECTION_SEED).permutation(len(targets))[:4096]),
            'embedding_path': str(directory / 'Qwen3-Embedding-4B_d14cc3313ecc.sent_emb'),
            'test_accessed': False,
        }
    return result


def campaign_path(value):
    path = Path(value).expanduser()
    if path.is_dir():
        return path.resolve()
    if path.name != value or not value or value in {'.', '..'}:
        raise ValueError('Use an existing campaign directory or its campaign id.')
    return CAMPAIGNS / value


def checked_manifest(directory):
    manifest = read_json(directory / 'manifest.json')
    digest = manifest.pop('manifest_sha256', None)
    if json_hash(manifest) != digest:
        raise ValueError(f'Campaign manifest changed: {directory}')
    manifest['manifest_sha256'] = digest
    for run in manifest['runs']:
        if json_hash(read_json(directory / run['config'])) != run['config_sha256']:
            raise ValueError(f"Resolved config changed: {run['config']}")
    return manifest


def is_model_file(relative):
    return relative in {'run.py', 'protocol.py', 'configs/final.json', 'configs/control.json'} or relative.startswith(
        ('joint_training/', 'models/', 'compression/', 'misc/'))


def assert_parent_runtime(previous, current):
    """Allow a harness-only revision to reuse measurements whose model code is unchanged."""
    if previous['sha256'] == current['sha256']:
        return
    previous_model = {name: digest for name, digest in previous.get('files', {}).items() if is_model_file(name)}
    current_model = {name: digest for name, digest in current.get('files', {}).items() if is_model_file(name)}
    if not previous_model or previous_model != current_model:
        raise ValueError('Model/runtime code changed since the parent; rebuild the matched baseline before continuing.')


def comparison_aliases(manifest):
    present = {run['dataset'] for run in manifest['runs']}
    ordered = tuple(alias for alias in CATEGORIES if alias in present)
    rule = manifest.get('protocol', {}).get('selection_rule', 'mean_relative_ndcg10_guard_2pct')
    if rule == 'target_ndcg10':
        if len(ordered) != 1:
            raise ValueError('target_ndcg10 requires exactly one recorded dataset.')
        return ordered
    if ordered not in {tuple(CATEGORIES), ACTIVE_DATASETS}:
        raise ValueError('A campaign must contain the historical four datasets or Beauty/Sports/Toys only.')
    return ordered


def completed_parent(value, mode):
    directory = campaign_path(value)
    manifest = checked_manifest(directory)
    if manifest['mode'] != mode:
        raise ValueError('Check and benchmark campaigns cannot be mixed.')
    summary = summarize(directory)
    if not summary['complete'] or (mode == 'benchmark' and not summary['benchmark_eligible']):
        raise ValueError('A parent/reference must have every recorded dataset completed and audited.')
    return directory, manifest


def prepare(spec_path, mode, device):
    spec = validate_spec(read_json(spec_path))
    identity = source_identity()
    parent_configs = None
    reference = parent = None
    previous = None
    if spec.get('parent'):
        parent, previous = completed_parent(spec['parent'], mode)
        assert_parent_runtime(previous['source'], identity)
        parent_configs = {}
        for alias in target_aliases(spec):
            run = next(r for r in previous['runs'] if r['dataset'] == alias)
            parent_configs[alias] = read_json(parent / run['config'])
    if spec.get('reference'):
        reference, ref = completed_parent(spec['reference'], mode)
        assert_parent_runtime(ref['source'], identity)
    seeds = spec.get('confirmation_seeds') if spec['stage'] == 'final' else [2024]
    if mode == 'check' and spec['stage'] == 'final':
        raise ValueError('Final multi-seed confirmation is benchmark-only.')
    resolved = resolve_configs(spec, parent_configs, seeds)
    data = preflight()
    for value in (parent, reference):
        if value and checked_manifest(value)['data'] != data:
            raise ValueError('Data changed since the parent/reference.')
    directory = CAMPAIGNS / spec['id']
    directory.mkdir(parents=True, exist_ok=False)
    runs = []
    for alias, by_seed in resolved.items():
        for seed, config in by_seed.items():
            config['training'].update(device=device, output_dir=str(directory / 'runs'))
            if mode == 'check':
                # Retain all model dimensions, objective, MC budget and catalog policy.
                config['training'].update(epochs=1, max_steps=2)
                config['recommender']['train_batch_size'] = 8
                config['ae']['batch_size'] = 16
                config['flow_ranking']['max_queries'] = 16
            reused = None
            if previous and spec['stage'] in {'starting_point', 'dataset'}:
                prior = next((r for r in previous['runs'] if r['dataset'] == alias and r['seed'] == seed), None)
                if prior:
                    old_config = read_json(parent / prior['config'])
                    if comparable_config(config) == comparable_config(old_config):
                        # Keep the original configuration and artifact identities.
                        # This is explicitly a reference to a prior audited run,
                        # never presented as a new independent measurement.
                        config = old_config
                        reused = {'campaign': str(parent), 'manifest_sha256': previous['manifest_sha256'],
                                  'dataset': alias, 'seed': seed}
            relative = f'configs/{alias}_s{seed}.json'
            write_json(directory / relative, config)
            runs.append({'dataset': alias, 'category': CATEGORIES[alias], 'seed': seed,
                         'config': relative, 'config_sha256': json_hash(config),
                         'output': f"runs/{CATEGORIES[alias]}/{config['training']['run_id']}",
                         'reused': reused})
    manifest = {
        'format_version': 1, 'created_utc': timestamp(), 'mode': mode, 'spec': spec,
        'source': identity, 'data': data, 'runs': runs,
        'parent': str(parent) if parent else None,
        'reference': str(reference) if reference else None,
        'selection_users': 4096 if mode == 'benchmark' else 8,
        'audit': 'full' if mode == 'benchmark' else 'subset',
        'python': sys.version, 'python_executable': sys.executable,
        'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
        'protocol': {'primary': 'ndcg@10', 'selection_seed': SELECTION_SEED,
                     'selection_rule': spec.get('selection_rule', 'mean_relative_ndcg10_guard_2pct'),
                     'comparison_datasets': list(target_aliases(spec)),
                     'declared_epochs': declared_epochs(spec),
                     'max_dataset_regression': None if spec.get('selection_rule') == 'target_ndcg10' else 0.02,
                     'test_accessed': False},
    }
    manifest['manifest_sha256'] = json_hash(manifest)
    write_json(directory / 'manifest.json', manifest)
    write_json(directory / 'spec.json', spec)
    for relative in identity['files']:
        target = directory / 'source' / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, target)
    return directory


def comparable_config(config):
    config = json.loads(json.dumps(config))
    for field in ('run_id', 'output_dir', 'device'):
        config['training'].pop(field, None)
    return config


def run_output(directory, run):
    reused = run.get('reused')
    if not reused:
        return directory / run['output']
    parent = Path(reused['campaign'])
    manifest = checked_manifest(parent)
    if manifest['manifest_sha256'] != reused['manifest_sha256']:
        raise ValueError('Reused parent manifest changed.')
    original = next(r for r in manifest['runs'] if r['dataset'] == reused['dataset'] and r['seed'] == reused['seed'])
    if original['config_sha256'] != run['config_sha256']:
        raise ValueError('Reused config does not match the original measurement.')
    return run_output(parent, original)


@contextmanager
def execution_lock(directory):
    path = directory / 'running.lock'
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as error:
        raise ValueError('Campaign already locked. Inspect the recorded PID before removing a stale lock.') from error
    try:
        with os.fdopen(descriptor, 'w') as stream:
            stream.write(json.dumps({'pid': os.getpid(), 'started_utc': timestamp()}))
        yield
    finally:
        path.unlink(missing_ok=True)


def validate_run(directory, manifest, run):
    output = run_output(directory, run)
    result = read_json(output / 'results.json')
    config = read_json(output / 'config.json')
    experiment = read_json(output / 'experiment.json')
    if json_hash(config) != run['config_sha256']:
        raise ValueError('Run used a different resolved config.')
    if (result.get('test_accessed') is not False or experiment.get('test_accessed') is not False
            or not result.get('checkpoint_reload_verified') or not (output / 'best_flow.pt').is_file()):
        raise ValueError('Missing checkpoint verification or invalid split provenance.')
    expected = manifest['data'][run['dataset']]
    for key in ('raw_sha256', 'train_sha256', 'item_order_sha256',
                'validation_histories_sha256', 'validation_targets_sha256'):
        if experiment['source'][key] != expected[key]:
            raise ValueError(f'Data identity changed: {key}.')
    metric_name = 'validation' if manifest['mode'] == 'benchmark' else 'validation_selection'
    metrics = result[metric_name]
    for name in ('ndcg@5', 'ndcg@10', 'recall@5', 'recall@10', 'nll'):
        value = metrics[name]
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f'Non-finite or missing {name}.')
        if name != 'nll' and not 0 <= value <= 1:
            raise ValueError(f'Invalid ranking metric {name}.')
    if manifest['mode'] == 'benchmark':
        epochs = config['training']['epochs']
        recorded = manifest.get('protocol', {}).get('declared_epochs', epochs)
        expected_steps = math.ceil(expected['training_windows'] / config['recommender']['train_batch_size']) * epochs
        if (not result.get('benchmark_eligible') or not experiment.get('benchmark_eligible')
                or epochs != recorded or result['completed_epochs'] != epochs or result['steps'] != expected_steps
                or experiment['selection_users'] != min(4096, expected['users'])
                or experiment['selection_indices_sha256'] != expected['selection_indices_sha256']):
            raise ValueError('Benchmark budget, audit or selection subset does not match the contract.')
    return {**metrics, 'seed': run['seed'], 'dataset': run['dataset'],
            'best_epoch': result['best']['epoch'], 'steps': result['steps'],
            'seconds': result['seconds'], 'output': str(output),
            'config_sha256': run['config_sha256'],
            'parameters': experiment['trainable_parameters'],
            'reused_from': run.get('reused'),
            'peak_memory_allocated_mib': result.get('peak_memory_allocated_mib')}


def execute(directory):
    manifest = checked_manifest(directory)
    if source_identity()['sha256'] != manifest['source']['sha256']:
        raise ValueError('Runtime changed after prepare; create a new campaign for this revision.')
    if preflight() != manifest['data']:
        raise ValueError('Data changed after prepare.')
    with execution_lock(directory):
        for run in manifest['runs']:
            output = run_output(directory, run)
            if (output / 'results.json').is_file():
                validate_run(directory, manifest, run)
                print(f"Skip audited completed run: {run['dataset']} seed={run['seed']}", flush=True)
                continue
            if output.exists():
                raise ValueError(f'Incomplete run exists: {output}. Preserve it and use a new campaign id; no silent overwrite/retry.')
            command = [sys.executable, str(ROOT / 'run.py'), '--config', str(directory / run['config']),
                       '--eval-users', str(manifest['selection_users']), '--audit', manifest['audit']]
            write_json(directory / 'status.json', {'state': 'running', 'run': run, 'command': command, 'updated_utc': timestamp()})
            log = directory / 'logs' / f"{run['dataset']}_s{run['seed']}.log"
            log.parent.mkdir(exist_ok=True)
            print(f"Run {run['dataset']} seed={run['seed']}; log={log}", flush=True)
            with log.open('x') as stream:
                completed = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
            if completed.returncode:
                write_json(directory / 'status.json', {'state': 'failed', 'run': run, 'returncode': completed.returncode, 'log': str(log)})
                raise RuntimeError(f'Training failed; see {log}')
            validate_run(directory, manifest, run)
            summarize(directory)
        write_json(directory / 'status.json', {'state': 'complete', 'updated_utc': timestamp()})
    return summarize(directory)


def aggregate(rows):
    result = {}
    for alias in CATEGORIES:
        group = [row for row in rows if row['dataset'] == alias]
        if not group:
            continue
        result[alias] = {'seeds': [r['seed'] for r in group], 'count': len(group)}
        for name in ('ndcg@5', 'ndcg@10', 'recall@5', 'recall@10', 'nll'):
            values = [row[name] for row in group]
            result[alias][name] = {'mean': statistics.mean(values),
                                  'std': statistics.stdev(values) if len(values) > 1 else None}
    return result


def compare_target(current, reference, alias):
    """One dataset, no cross-dataset mean and no 2% guard."""
    baseline = reference[alias]['ndcg@10']['mean']
    if baseline <= 0:
        raise ValueError('Reference NDCG@10 is zero; relative ranking is undefined.')
    relative = current[alias]['ndcg@10']['mean'] / baseline - 1
    return {'relative_ndcg10': {alias: relative}, 'mean_relative_ndcg10': relative,
            'worst_relative_ndcg10': relative, 'regression_guard_passed': None,
            'eligible_for_promotion': current[alias]['ndcg@10']['mean'] > baseline,
            'selection_rule': 'target_ndcg10',
            'interpretation': 'single-dataset single-seed trend; promotion requires full-validation NDCG@10 above the fixed reference and not below the parent',
            'matched_confirmation_seeds': current[alias]['count'] > 1 and sorted(current[alias]['seeds']) == sorted(reference[alias]['seeds'])}


def compare(current, reference, aliases=None):
    aliases = tuple(CATEGORIES if aliases is None else aliases)
    if aliases not in {tuple(CATEGORIES), ACTIVE_DATASETS}:
        raise ValueError('Comparison must use the historical four datasets or Beauty/Sports/Toys.')
    relative = {}
    for alias in aliases:
        baseline = reference[alias]['ndcg@10']['mean']
        if baseline <= 0:
            raise ValueError('Reference NDCG@10 is zero; relative ranking is undefined.')
        relative[alias] = current[alias]['ndcg@10']['mean'] / baseline - 1
    mean = statistics.mean(relative.values())
    guard = all(value >= -0.02 for value in relative.values())
    return {'relative_ndcg10': relative, 'mean_relative_ndcg10': mean,
            'worst_relative_ndcg10': min(relative.values()), 'regression_guard_passed': guard,
            'eligible_for_promotion': mean > 0 and guard,
            'interpretation': 'single-seed trend unless both sides have the same explicit multiple seeds',
            'matched_confirmation_seeds': all(current[a]['count'] > 1 and sorted(current[a]['seeds']) == sorted(reference[a]['seeds']) for a in aliases)}


def summarize(directory):
    manifest = checked_manifest(directory)
    rows, missing = [], []
    for run in manifest['runs']:
        if not (run_output(directory, run) / 'results.json').is_file():
            missing.append(f"{run['dataset']}_s{run['seed']}")
        else:
            rows.append(validate_run(directory, manifest, run))
    grouped = aggregate(rows)
    complete = not missing
    summary = {'campaign': manifest['spec']['id'], 'stage': manifest['spec']['stage'],
               'mode': manifest['mode'], 'complete': complete, 'missing': missing,
               'benchmark_eligible': complete and manifest['mode'] == 'benchmark',
               'test_accessed': False, 'rows': rows, 'datasets': grouped}
    if complete and manifest['reference']:
        ref_dir = Path(manifest['reference'])
        ref_manifest = checked_manifest(ref_dir)
        reference_rows = [validate_run(ref_dir, ref_manifest, run) for run in ref_manifest['runs']]
        if manifest['mode'] == 'benchmark' and manifest.get('protocol', {}).get('selection_rule') == 'target_ndcg10':
            alias = comparison_aliases(manifest)[0]
            summary['comparison'] = compare_target(grouped, aggregate(reference_rows), alias)
            parent_dir = Path(manifest['parent'])
            parent_manifest = checked_manifest(parent_dir)
            parent_grouped = aggregate([validate_run(parent_dir, parent_manifest, run) for run in parent_manifest['runs']])
            delta = grouped[alias]['ndcg@10']['mean'] - parent_grouped[alias]['ndcg@10']['mean']
            summary['dataset_comparison_to_parent'] = {'absolute_ndcg10_changes': {alias: delta}, 'passed': delta >= 0}
            summary['comparison']['eligible_for_promotion'] &= delta >= 0
        elif manifest['mode'] == 'benchmark':
            summary['comparison'] = compare(grouped, aggregate(reference_rows), comparison_aliases(manifest))
            if manifest['spec']['stage'] in {'starting_point', 'dataset'}:
                parent_dir = Path(manifest['parent'])
                parent_manifest = checked_manifest(parent_dir)
                parent_grouped = aggregate([validate_run(parent_dir, parent_manifest, run) for run in parent_manifest['runs']])
                changed = [alias for alias, patch in manifest['spec'].get('datasets', {}).items() if patch]
                differences = {alias: grouped[alias]['ndcg@10']['mean'] - parent_grouped[alias]['ndcg@10']['mean']
                               for alias in changed}
                passes = bool(differences) and all(value >= 0 for value in differences.values()) and any(value > 0 for value in differences.values())
                summary['dataset_comparison_to_parent'] = {'absolute_ndcg10_changes': differences, 'passed': passes}
                summary['comparison']['eligible_for_promotion'] &= passes
        else:
            summary['comparison'] = {'eligible_for_promotion': False,
                                     'interpretation': 'Engineering checks are not compared for performance.'}
    write_json(directory / 'summary.json', summary)
    lines = [f"# {summary['campaign']}", '',
             f"模式：{summary['mode']}；完整：{complete}；可作为 benchmark：{summary['benchmark_eligible']}。test 未用于评估。", '',
             f"假设：{manifest['spec']['hypothesis']}", '',
             '| 数据集 | seed | NDCG@10 | Recall@10 | NLL | 最佳轮 | 更新数 | 测量来源 | 配置 SHA256 |',
             '|---|---:|---:|---:|---:|---:|---:|---|---|']
    for row in rows:
        origin = row['reused_from']['campaign'] if row.get('reused_from') else '本轮新运行'
        lines.append(f"| {row['dataset']} | {row['seed']} | {row['ndcg@10']:.8f} | {row['recall@10']:.8f} | {row['nll']:.5f} | {row['best_epoch']} | {row['steps']} | {origin} | {row['config_sha256']} |")
    if any(value['count'] > 1 for value in grouped.values()):
        lines += ['', '多种子使用样本标准差（ddof=1）；完整逐种子及所有指标见 summary.json。', '',
                  '| 数据集 | seeds | NDCG@10 均值 ± 标准差 |', '|---|---|---:|']
        for alias, values in grouped.items():
            score = values['ndcg@10']
            if score['std'] is not None:
                lines.append(f"| {alias} | {values['seeds']} | {score['mean']:.8f} ± {score['std']:.8f} |")
    comparison = summary.get('comparison', {})
    if comparison.get('selection_rule') == 'target_ndcg10':
        alias, relative = next(iter(comparison['relative_ndcg10'].items()))
        lines += ['', f"{alias} 相对固定 reference 的 NDCG@10：{relative:+.3%}。",
                  f"满足数值晋级条件：{comparison['eligible_for_promotion']}。这是单数据集单种子趋势，还要对照 parent、复杂度和显存后再写入决策。"]
    elif 'mean_relative_ndcg10' in comparison:
        scope = '四数据集' if comparison_aliases(manifest) == tuple(CATEGORIES) else 'Beauty/Sports/Toys'
        lines += ['', f"{scope}平均相对 NDCG@10：{comparison['mean_relative_ndcg10']:+.3%}；最差：{comparison['worst_relative_ndcg10']:+.3%}。",
                  f"满足数值晋级条件：{comparison['eligible_for_promotion']}。机制、复杂度及对照是否充分仍需写入决策记录。"]
    if missing:
        rule = manifest.get('protocol', {}).get('selection_rule')
        note = '目标数据集尚未完成，不能下结论。' if rule == 'target_ndcg10' else '禁止按局部结果选共享结构。'
        lines += ['', f"尚缺：{', '.join(missing)}。{note}"]
    (directory / 'summary.md').write_text('\n'.join(lines) + '\n')
    return summary


def record_decision(directory, outcome, evidence, next_hypothesis):
    summary = summarize(directory)
    if not summary['benchmark_eligible']:
        raise ValueError('Only complete benchmark campaigns support a research decision.')
    if outcome == 'promote' and summary.get('comparison', {}).get('eligible_for_promotion') is not True:
        raise ValueError('Promotion requires positive mean relative NDCG@10 and the per-dataset guard.')
    if not evidence.strip() or not next_hypothesis.strip():
        raise ValueError('Decision evidence and next hypothesis must be explicit.')
    path = directory / 'decision.json'
    if path.exists():
        raise ValueError('Decision is immutable; record a new iteration rather than rewriting history.')
    write_json(path, {'outcome': outcome, 'evidence': evidence, 'next_hypothesis': next_hypothesis,
                      'summary_sha256': json_hash(summary), 'recorded_utc': timestamp()})


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    commands = parser.add_subparsers(dest='command', required=True)
    check = commands.add_parser('preflight')
    check.add_argument('--output', type=Path)
    prep = commands.add_parser('prepare')
    prep.add_argument('--spec', type=Path, required=True)
    prep.add_argument('--mode', choices=('check', 'benchmark'), default='check')
    prep.add_argument('--device', default='cpu')
    for name in ('run', 'summarize'):
        sub = commands.add_parser(name)
        sub.add_argument('campaign')
    decision = commands.add_parser('decide')
    decision.add_argument('campaign')
    decision.add_argument('--outcome', choices=('promote', 'retain', 'reject', 'inconclusive'), required=True)
    decision.add_argument('--evidence', required=True)
    decision.add_argument('--next-hypothesis', required=True)
    args = parser.parse_args()
    try:
        if args.command == 'preflight':
            result = preflight()
            if args.output:
                write_json(args.output, result)
        elif args.command == 'prepare':
            result = {'campaign_directory': str(prepare(args.spec, args.mode, args.device))}
        elif args.command == 'run':
            result = execute(campaign_path(args.campaign))
        elif args.command == 'summarize':
            result = summarize(campaign_path(args.campaign))
        else:
            record_decision(campaign_path(args.campaign), args.outcome, args.evidence, args.next_hypothesis)
            result = {'decision': str(campaign_path(args.campaign) / 'decision.json')}
        print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
    except (ValueError, OSError, RuntimeError, KeyError, StopIteration) as error:
        parser.exit(1, f'Error: {error}\n')
