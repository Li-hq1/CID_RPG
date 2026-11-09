"""Parameter scopes and immutable benchmark contract for UGRec v5.1."""

import copy
import json
import re
from pathlib import Path

from joint_training.config import JointConfig
from protocol import CATEGORIES, ROOT, canonical_category

SHARED = {
    'ae.architecture', 'ae.hidden_dims', 'ae.latent_dim', 'ae.hidden_norm',
    'recommender.n_embd', 'recommender.n_layer', 'recommender.n_head',
    'recommender.hstu_hidden_dim', 'recommender.hstu_attention_dim',
    'recommender.diffloss_w', 'recommender.diffloss_d',
    'recommender.input_projection', 'recommender.condition_projection',
    'recommender.condition_mlp_depth', 'recommender.condition_mlp_norm',
    'recommender.anchor_projection', 'recommender.flow_condition_projection',
}
DATASET = {
    'ae.lr', 'ae.weight_decay', 'ae.warmup_ratio',
    'recommender.lr', 'recommender.weight_decay', 'recommender.warmup_ratio',
    'recommender.hstu_output_dropout', 'recommender.hstu_attention_dropout',
    'flow_ranking.temperature', 'flow_ranking.local_weight',
}
STAGES = ('baseline', 'starting_point', 'structure', 'shared', 'dataset', 'final')
# 2026-11-09 protocol: new campaigns train and compare Beauty, Sports, and Toys.
# CDs stays loadable for historical runs, but it is outside the active set.
ACTIVE_DATASETS = ('beauty', 'sports', 'toys')
SELECTION_RULES = ('mean_relative_ndcg10_guard_2pct', 'target_ndcg10')


def read_json(path):
    return json.loads(Path(path).read_text())


def patch_config(config, patch, allowed):
    if not isinstance(patch, dict):
        raise ValueError('Parameter patches must be objects of section.field: value.')
    for name, value in patch.items():
        if name not in allowed:
            raise ValueError(f'{name}: outside the permitted parameter scope; ask the user before adding a new option.')
        section, field = name.split('.')
        config[section][field] = copy.deepcopy(value)
    return config


def declared_epochs(spec):
    """Benchmark length. The default remains 40; a longer multiple of 5 is explicit."""
    epochs = spec.get('epochs', 40)
    if type(epochs) is not int or epochs < 40 or epochs % 5:
        raise ValueError('Benchmark epochs must be an integer, a multiple of the 5-epoch selection interval, and at least 40.')
    return epochs


def target_aliases(spec):
    """Datasets this campaign trains. Omitting the field keeps the historical active set."""
    raw = spec.get('target_datasets')
    if raw is None:
        return ACTIVE_DATASETS
    if (not isinstance(raw, list) or not raw or len(set(raw)) != len(raw)
            or any(alias not in ACTIVE_DATASETS for alias in raw)):
        raise ValueError('target_datasets must list one or more of beauty, sports, and toys without duplicates.')
    return tuple(alias for alias in ACTIVE_DATASETS if alias in raw)


def validate_spec(spec):
    allowed = {'id', 'stage', 'hypothesis', 'change_reason', 'parent', 'reference',
               'shared', 'datasets', 'seed', 'confirmation_seeds', 'selection_rule',
               'target_datasets', 'epochs'}
    if not isinstance(spec, dict) or set(spec) - allowed:
        raise ValueError('Unknown campaign fields.')
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,95}', spec.get('id', '')):
        raise ValueError('Campaign id must be a safe directory name, at most 96 characters.')
    if spec.get('stage') not in STAGES:
        raise ValueError(f'stage must be one of {STAGES}.')
    for field in ('hypothesis', 'change_reason'):
        if not isinstance(spec.get(field), str) or not spec[field].strip():
            raise ValueError(f'{field} is required for an attributable iteration.')
    if spec.get('seed', 2024) != 2024 or type(spec.get('seed', 2024)) is not int:
        raise ValueError('Exploration fixes seed=2024; confirmation seeds belong only to final.')
    rule = spec.get('selection_rule', 'mean_relative_ndcg10_guard_2pct')
    if rule not in SELECTION_RULES:
        raise ValueError('Changing the declared selection rule requires an explicit protocol revision.')
    aliases = target_aliases(spec)
    if spec['stage'] == 'baseline' and ('target_datasets' in spec or 'epochs' in spec or rule != 'mean_relative_ndcg10_guard_2pct'):
        raise ValueError('Baseline keeps the untouched control, 40 epochs, and the historical comparison.')
    if rule == 'target_ndcg10' and len(aliases) != 1:
        raise ValueError('target_ndcg10 is only for a campaign that names exactly one target dataset.')
    if 'target_datasets' in spec and rule != 'target_ndcg10':
        raise ValueError('A restricted target dataset list must use selection_rule=target_ndcg10.')
    declared_epochs(spec)
    shared = spec.get('shared', {})
    patch_config({'ae': {}, 'recommender': {}}, shared, SHARED)
    datasets = spec.get('datasets', {})
    if not isinstance(datasets, dict) or not set(datasets) <= set(CATEGORIES):
        raise ValueError('Dataset patches use beauty/sports/toys/cds.')
    if set(datasets) - set(aliases):
        raise ValueError('Dataset patches must belong to this campaign target set.')
    for patch in datasets.values():
        patch_config({'ae': {}, 'recommender': {}, 'flow_ranking': {}}, patch, DATASET)
    if spec['stage'] == 'baseline' and (shared or any(datasets.values()) or spec.get('parent') or spec.get('reference')):
        raise ValueError('Baseline is the untouched v5 control, without patches or parent/reference.')
    if spec['stage'] != 'baseline' and (not spec.get('parent') or not spec.get('reference')):
        raise ValueError('Each iteration must name its parent and fixed reference campaign.')
    if spec['stage'] in {'starting_point', 'dataset'} and shared:
        raise ValueError('Dataset tuning cannot change shared structure.')
    if spec['stage'] == 'starting_point' and datasets.get('beauty'):
        raise ValueError('Starting-point calibration leaves Beauty fixed.')
    if spec['stage'] in {'structure', 'shared'} and any(datasets.values()):
        raise ValueError('Shared comparisons freeze all dataset-specific settings.')
    if spec['stage'] == 'final' and (shared or any(datasets.values())):
        raise ValueError('Final confirmation cannot retune the frozen parent.')
    seeds = spec.get('confirmation_seeds', [])
    if 'confirmation_seeds' in spec and not seeds:
        raise ValueError('Omit confirmation_seeds during exploration; final requires a nonempty list.')
    if seeds:
        if (spec['stage'] != 'final' or not isinstance(seeds, list) or len(seeds) < 2
                or any(type(seed) is not int or seed < 0 for seed in seeds)
                or len(set(seeds)) != len(seeds)):
            raise ValueError('Only an explicitly configured final may have distinct confirmation_seeds (at least two).')
    if spec['stage'] == 'final' and not seeds:
        raise ValueError('Final requires explicit confirmation_seeds; no automatic seed sweep.')
    return spec


def resolve_configs(spec, parent_configs=None, seeds=None):
    validate_spec(spec)
    epochs = declared_epochs(spec)
    configs = {}
    for alias in target_aliases(spec):
        category = CATEGORIES[alias]
        base = (copy.deepcopy(parent_configs[alias]) if parent_configs is not None
                else JointConfig.from_dict(read_json(ROOT / 'configs/final.json')).to_dict())
        patch_config(base, spec.get('shared', {}), SHARED)
        patch_config(base, spec.get('datasets', {}).get(alias, {}), DATASET)
        base['data']['category'] = category
        fixed_flow = {'enabled': True, 'path': 'mixture', 'source': 'history',
                      'objective': 'contrastive', 'sampling': 'uniform_corrected',
                      'likelihood': 'marginal', 'noise_scale': 1.0, 'draws': 8,
                      'catalog': 'auto', 'max_exact_items': 20000, 'negatives': 4096,
                      'vlad_clusters': 0}
        if (any(base['flow_ranking'][name] != value for name, value in fixed_flow.items())
                or base['ranking']['enabled'] or base['contrastive']['enabled']
                or base['recommender']['encoder_type'] != 'hstu'):
            raise ValueError('The tuning contract is fixed to v5 control; changing the method requires a protocol revision.')
        configs[alias] = {}
        for seed in seeds or [spec.get('seed', 2024)]:
            config = copy.deepcopy(base)
            config['training'].update(seed=seed, run_id=f"{spec['id']}_s{seed}",
                                      evaluate_test=False, smoke_test=False)
            # All benchmark controls are fixed independently of inherited runtime paths.
            config['training'].update(epochs=epochs, max_steps=None, deterministic=True)
            config['evaluation'].update(eval_interval=5, eval_batch_size=32,
                                        eval_num_samples=32, random_seed=20261028,
                                        val_metric='ndcg@10', retrieval_backend='exact')
            JointConfig.from_dict(config)
            configs[alias][seed] = config
    return configs
