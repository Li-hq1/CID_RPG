"""Tuning isolation, objective equivalence and artifacts are research-critical."""

import copy
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np
import torch
from torch.nn import functional as F

from joint_training.config import JointConfig, parse_config
from joint_training.data import fit_input_statistics
from joint_training.model import JointModel
from protocol import ROOT, canonical_category, corpus, training_windows
from tests.test_final_scheme import fixture
from tests.test_joint_model import make_batch
from tuning.campaign import aggregate, compare
from tuning.schema import resolve_configs, validate_spec


def baseline():
    return {'id': 'test_baseline', 'stage': 'baseline', 'hypothesis': 'test', 'change_reason': 'test'}


def candidate(**extra):
    return {**baseline(), 'stage': 'structure', 'parent': 'baseline', 'reference': 'baseline', **extra}


class TuningContractTest(unittest.TestCase):
    def test_shared_changes_apply_identically_to_active_datasets(self):
        configs = resolve_configs(candidate(shared={'ae.latent_dim': 64, 'recommender.n_layer': 3}))
        self.assertEqual(set(configs), {'beauty', 'sports', 'toys'})
        for group in configs.values():
            self.assertEqual(group[2024]['ae']['latent_dim'], 64)
            self.assertEqual(group[2024]['recommender']['n_layer'], 3)

    def test_dataset_changes_are_isolated_and_structural_leaks_fail(self):
        spec = candidate(stage='dataset', datasets={'sports': {'recommender.lr': 0.0003}})
        configs = resolve_configs(spec)
        self.assertEqual(configs['sports'][2024]['recommender']['lr'], .0003)
        self.assertEqual(configs['beauty'][2024]['recommender']['lr'], .001)
        for field in ('ae.latent_dim', 'recommender.n_layer', 'flow_ranking.draws', 'training.epochs'):
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_spec(candidate(stage='dataset', datasets={'sports': {field: 2}}))
        with self.assertRaises(ValueError):
            validate_spec(candidate(shared={'flow_ranking.noise_scale': .3}))

    def test_seeds_and_stages_are_explicit(self):
        for spec in (baseline() | {'seed': 2025}, baseline() | {'confirmation_seeds': [2024, 2025]},
                     candidate(stage='final'), candidate(stage='starting_point', datasets={'beauty': {'ae.lr': .0002}})):
            with self.subTest(spec=spec), self.assertRaises(ValueError):
                validate_spec(spec)
        validate_spec(candidate(stage='final', confirmation_seeds=[2024, 2025, 2026]))
        slow = candidate(stage='dataset', target_datasets=['beauty'], selection_rule='target_ndcg10',
                         epochs=80, datasets={'beauty': {'recommender.lr': .0005}})
        resolved = resolve_configs(slow)
        self.assertEqual(set(resolved), {'beauty'})
        self.assertEqual(resolved['beauty'][2024]['training']['epochs'], 80)
        self.assertEqual(resolved['beauty'][2024]['recommender']['lr'], .0005)
        for invalid in (candidate(epochs=30), candidate(epochs=42), candidate(epochs=40.0),
                        candidate(target_datasets=['beauty']),
                        candidate(target_datasets=['beauty', 'sports'], selection_rule='target_ndcg10'),
                        baseline() | {'epochs': 80},
                        candidate(stage='dataset', target_datasets=['beauty'], selection_rule='target_ndcg10',
                                  datasets={'sports': {'recommender.lr': .0005}})):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                validate_spec(invalid)

    def test_default_fields_match_v5(self):
        old = json.loads((ROOT.parent / 'UGRec_v5/configs/final.json').read_text())
        new = JointConfig.from_dict(old).to_dict()
        for section, values in old.items():
            for name, value in values.items():
                self.assertEqual(json.loads(json.dumps(new[section][name])), value)

    def test_method_changes_cannot_be_inherited_as_tuning(self):
        parents = {alias: group[2024] for alias, group in resolve_configs(baseline()).items()}
        parents['sports']['flow_ranking']['noise_scale'] = .3
        with self.assertRaises(ValueError):
            resolve_configs(candidate(), parents)

    def test_dimension_invalid_identity_is_rejected(self):
        for patch in ({'flow_condition_projection': 'identity', 'diffloss_w': 256}, {'n_embd': 64, 'anchor_projection': 'identity'},
                      {'n_embd': 64, 'input_projection': 'identity'}):
            with self.subTest(patch=patch), self.assertRaises(ValueError):
                JointConfig.from_dict({'ae': {'latent_dim': 128}, 'recommender': patch})

    def test_no_partial_aggregate_and_guard_against_dataset_regression(self):
        ref = {alias: {'count': 1, 'seeds': [2024], 'ndcg@10': {'mean': .1}}
               for alias in ('beauty', 'sports', 'toys', 'cds')}
        current = copy.deepcopy(ref)
        current['beauty']['ndcg@10']['mean'] = .12
        current['cds']['ndcg@10']['mean'] = .097
        value = compare(current, ref)
        self.assertGreater(value['mean_relative_ndcg10'], 0)
        self.assertFalse(value['eligible_for_promotion'])
        with self.assertRaises(KeyError):
            compare({'beauty': current['beauty']}, ref)
        three = ('beauty', 'sports', 'toys')
        promoted = compare(current, ref, three)
        self.assertGreater(promoted['mean_relative_ndcg10'], 0)
        self.assertTrue(promoted['eligible_for_promotion'])
        self.assertNotIn('cds', promoted['relative_ndcg10'])
        with self.assertRaises(ValueError):
            compare(current, ref, ('beauty', 'sports'))
        from tuning.campaign import compare_target
        above = compare_target(current, ref, 'beauty')
        self.assertGreater(above['relative_ndcg10']['beauty'], 0)
        self.assertTrue(above['eligible_for_promotion'])
        self.assertEqual(above['selection_rule'], 'target_ndcg10')
        tied = copy.deepcopy(current)
        tied['beauty']['ndcg@10']['mean'] = .1
        self.assertFalse(compare_target(tied, ref, 'beauty')['eligible_for_promotion'])
        self.assertIsNone(aggregate([{'dataset': 'beauty', 'seed': 2024, **dict.fromkeys(
            ('ndcg@5', 'ndcg@10', 'recall@5', 'recall@10', 'nll'), .1)}])['beauty']['ndcg@10']['std'])

    def test_aliases(self):
        for alias in ('sports', 'Sports_and_Outdoors', 'SPORTS'):
            self.assertEqual(canonical_category(alias), 'Sports_and_Outdoors')

    def test_runner_preserves_resolved_device(self):
        from run import load_config
        args = SimpleNamespace(config=ROOT / 'configs/final.json', category=None,
                               device=None, run_id=None, smoke_test=False,
                               seed=None, epochs=None, set=['training.device=cpu'])
        self.assertEqual(load_config(args).training.device, 'cpu')

    def test_protocol_ignores_test_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            cache = Path(temporary)
            folder = cache / 'AmazonReviews2014/Beauty/processed'
            folder.mkdir(parents=True)
            (folder / 'id_mapping.json').write_text(json.dumps({'id2item': ['[PAD]', 'a', 'b'],
                                                              'item2id': {'[PAD]': 0, 'a': 1, 'b': 2}}))
            (folder / 'all_item_seqs.json').write_text(json.dumps({'user': ['a', 'b', 'UNKNOWN_TEST_ITEM']}))
            _, _, train, history, targets = corpus('beauty', cache)
            self.assertEqual((train, history, targets.tolist()), ([[1]], [[1]], [2]))


class OptimizationEquivalenceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_chunked_input_statistics(self):
        values = np.random.default_rng(5).normal(size=(2200, 17)).astype(np.float32)
        indices = np.arange(2101)
        mean, scale = fit_input_statistics(values, indices)
        expected = values[indices].astype(np.float64)
        np.testing.assert_allclose(mean, expected.mean(0), rtol=1e-6, atol=1e-8)
        self.assertAlmostEqual(float(scale), float(np.sqrt(((expected - expected.mean(0)) ** 2).mean())), places=6)

    def test_chunked_probabilities_equal_materialized_formula(self):
        for clusters in (0, 2):
            model = fixture(vlad_clusters=clusters)
            model.eval()
            model.refresh_catalog()
            batch = make_batch()
            for chunk in (1, 3, 256):
                model.config.evaluation.mixture_query_chunk_size = chunk
                torch.manual_seed(11)
                actual = model.mixture_log_probabilities(batch)
                torch.manual_seed(11)
                encoded, _ = model.encode_history(batch, ae_grad=False)
                conditions, log_pi = model.mixture_conditions(encoded, batch['attention_mask'], torch.arange(2), torch.ones(2, dtype=torch.long))
                b, k, _ = conditions.shape
                per = model.config.evaluation.eval_num_samples // k
                expanded = conditions.reshape(b*k, -1).repeat_interleave(per, 0)
                samples = model._sample_condition(expanded)
                scores = F.normalize(samples, dim=-1) @ F.normalize(model.catalog[1:], dim=-1).T / model.config.flow_ranking.temperature
                log_p = scores.log_softmax(-1).reshape(b, k, per, -1)
                expected = torch.logsumexp(log_pi[..., None] + torch.logsumexp(log_p, 2) - math.log(per), 1)
                torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)
                torch.testing.assert_close(actual.exp().sum(-1), torch.ones(2))

    def test_projection_candidates_have_finite_joint_gradients(self):
        for mode, depth, norm in (('identity', 1, 'none'), ('norm', 1, 'none'), ('linear', 1, 'none'),
                                  ('mlp', 2, 'none'), ('mlp', 2, 'layer')):
            model = fixture()
            from models.projections import condition_projection
            model.condition_proj = condition_projection(8, mode, depth, norm)
            loss, _ = model.joint_objectives(make_batch())
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
            self.assertGreater(model.input_proj.linear.weight.grad.norm().item(), 0)

    def test_equal_width_projection_removals_and_ae_norm_train(self):
        config = fixture().config.to_dict()
        config['ae'].update(architecture='mlp', hidden_dims=[8], latent_dim=8, hidden_norm='layer')
        config['recommender'].update(n_embd=8, diffloss_w=8, anchor_projection='identity',
                                     input_projection='norm', flow_condition_projection='identity')
        model = JointModel(JointConfig.from_dict(config), torch.randn(5, 6))
        loss, _ = model.joint_objectives(make_batch())
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(model.ae.encoder[0].weight.grad.norm().item(), 0)
        self.assertGreater(model.diffloss.net.final_layer.linear.weight.grad.norm().item(), 0)

    def test_native_default_model_matches_v5_initialization_and_objective(self):
        # Separate processes avoid the unqualified imports shared by all versions.
        code = '''
import torch
from joint_training.config import parse_config
from joint_training.model import JointModel
from misc.artifacts import array_hash, json_hash
from tests.test_joint_model import make_batch
cfg, _ = parse_config(['--config','configs/final.json'])
cfg.training.device='cpu'
cfg.flow_ranking.max_queries=4
cfg.flow_ranking.catalog='exact'
torch.set_num_threads(2)
torch.manual_seed(2024)
model=JointModel(cfg, torch.randn(12,2560))
weights={k: array_hash(v.detach().numpy()) for k,v in model.state_dict().items()}
loss,_=model.joint_objectives(make_batch())
loss.backward()
print(json_hash(weights))
print(float(loss.detach()))
print(float(model.input_proj.linear.weight.grad.norm()))
'''
        outputs = []
        for version in ('UGRec_v5', 'UGRec_v5_1'):
            result = subprocess.run([sys.executable, '-c', code], cwd=ROOT.parent / version,
                                    capture_output=True, text=True, check=True)
            outputs.append(result.stdout.splitlines())
        self.assertEqual(outputs[0], outputs[1])


class CampaignArtifactTest(unittest.TestCase):
    """Isolated synthetic fixtures in temporary directories; no real results are edited."""

    def setUp(self):
        from unittest.mock import patch
        from tuning import campaign
        self.campaign = campaign
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        data = {}
        for alias in ('beauty', 'sports', 'toys', 'cds'):
            data[alias] = {'users': 5000, 'training_windows': 8, **dict.fromkeys(
                ('raw_sha256', 'train_sha256', 'item_order_sha256', 'validation_histories_sha256',
                 'validation_targets_sha256', 'selection_indices_sha256'), 'isolated-synthetic-fixture')}
        for name, value in (('CAMPAIGNS', self.directory), ('preflight', lambda: data),
                            ('source_identity', lambda: {'sha256': 'isolated-test-source', 'files': {}})):
            context = patch.object(campaign, name, value)
            context.start()
            self.addCleanup(context.stop)

    def prepare(self, spec):
        path = self.directory / f"{spec['id']}.json"
        path.write_text(json.dumps(spec))
        return self.campaign.prepare(path, 'benchmark', 'cpu')

    def simulate_audited_outputs(self, directory, datasets=None, ndcg10=None):
        # Labeled, deliberately synthetic artifacts, used only by this unit test.
        campaign = self.campaign
        manifest = campaign.checked_manifest(directory)
        for run in manifest['runs']:
            if datasets is not None and run['dataset'] not in datasets:
                continue
            output = campaign.run_output(directory, run)
            output.mkdir(parents=True)
            cfg = campaign.read_json(directory / run['config'])
            epochs = cfg['training']['epochs']
            steps = math.ceil(manifest['data'][run['dataset']]['training_windows'] / cfg['recommender']['train_batch_size']) * epochs
            metrics = dict.fromkeys(('ndcg@5', 'ndcg@10', 'recall@5', 'recall@10', 'nll'), .1)
            if isinstance(ndcg10, dict):
                metrics['ndcg@10'] = ndcg10.get(run['dataset'], metrics['ndcg@10'])
            elif ndcg10 is not None:
                metrics['ndcg@10'] = ndcg10
            campaign.write_json(output / 'config.json', cfg)
            campaign.write_json(output / 'experiment.json', {
                'source': manifest['data'][run['dataset']], 'test_accessed': False,
                'benchmark_eligible': True, 'selection_users': 4096,
                'selection_indices_sha256': 'isolated-synthetic-fixture', 'trainable_parameters': 10})
            campaign.write_json(output / 'results.json', {
                'validation': metrics,
                'test_accessed': False, 'checkpoint_reload_verified': True,
                'benchmark_eligible': True, 'completed_epochs': epochs, 'steps': steps,
                'best': {'epoch': 5}, 'seconds': 1})
            (output / 'best_flow.pt').write_bytes(b'ISOLATED SYNTHETIC CHECKPOINT FIXTURE; NOT A REAL MODEL')

    def test_dataset_iteration_reuses_unchanged_audited_runs(self):
        first = self.prepare(baseline())
        self.simulate_audited_outputs(first)
        second = self.prepare(candidate(id='sports_lr', stage='dataset', parent=str(first), reference=str(first),
                                        datasets={'sports': {'recommender.lr': .0005}}))
        manifest = self.campaign.checked_manifest(second)
        for run in manifest['runs']:
            self.assertEqual(run['reused'] is None, run['dataset'] == 'sports')
        summary = self.campaign.summarize(second)
        self.assertEqual(summary['missing'], ['sports_s2024'])
        self.assertFalse(summary['benchmark_eligible'])
        self.simulate_audited_outputs(second, {'sports'})
        summary = self.campaign.execute(second)
        self.assertTrue(summary['benchmark_eligible'])
        self.assertEqual(sum(row['reused_from'] is not None for row in summary['rows']), 2)
        self.assertTrue(self.campaign.execute(second)['complete'])

    def test_manifest_and_resolved_configuration_cannot_drift(self):
        directory = self.prepare(baseline())
        manifest = self.campaign.checked_manifest(directory)
        path = directory / manifest['runs'][0]['config']
        config = json.loads(path.read_text())
        config['recommender']['lr'] = .2
        path.write_text(json.dumps(config))
        with self.assertRaises(ValueError):
            self.campaign.checked_manifest(directory)

    def test_incomplete_results_cannot_be_promoted(self):
        directory = self.prepare(baseline())
        with self.assertRaises(ValueError):
            self.campaign.record_decision(directory, 'promote', 'synthetic test', 'synthetic test')
        self.simulate_audited_outputs(directory)
        with self.assertRaises(ValueError):
            self.campaign.record_decision(directory, 'promote', 'synthetic test', 'synthetic test')
        self.campaign.record_decision(directory, 'retain', 'synthetic test', 'synthetic test')
        with self.assertRaises(ValueError):
            self.campaign.record_decision(directory, 'retain', 'synthetic test', 'synthetic test')

    def test_single_dataset_budget_promotes_only_above_reference_and_parent(self):
        base = self.prepare(baseline())
        self.simulate_audited_outputs(base)
        raised = self.prepare(candidate(id='raised_beauty', stage='dataset', parent=str(base), reference=str(base),
                                        datasets={'beauty': {'recommender.lr': .0004}}))
        self.simulate_audited_outputs(raised, {'beauty'}, ndcg10=.3)
        self.assertTrue(self.campaign.summarize(raised)['benchmark_eligible'])
        common = {'stage': 'dataset', 'parent': str(raised), 'reference': str(base),
                  'target_datasets': ['beauty'], 'selection_rule': 'target_ndcg10', 'epochs': 80,
                  'datasets': {'beauty': {'recommender.lr': .0005}}}
        below = self.prepare(candidate(id='below_parent', **common))
        manifest = self.campaign.checked_manifest(below)
        self.assertEqual([run['dataset'] for run in manifest['runs']], ['beauty'])
        self.assertEqual(manifest['protocol']['declared_epochs'], 80)
        self.assertIsNone(manifest['protocol']['max_dataset_regression'])
        self.simulate_audited_outputs(below, ndcg10=.2)
        summary = self.campaign.summarize(below)
        self.assertEqual(summary['rows'][0]['steps'], 80)
        self.assertFalse(summary['comparison']['eligible_for_promotion'])
        self.assertLess(summary['dataset_comparison_to_parent']['absolute_ndcg10_changes']['beauty'], 0)
        with self.assertRaises(ValueError):
            self.campaign.record_decision(below, 'promote', 'synthetic test', 'synthetic test')
        above = self.prepare(candidate(id='above_both', **common))
        self.simulate_audited_outputs(above, ndcg10=.4)
        summary = self.campaign.summarize(above)
        self.assertTrue(summary['comparison']['eligible_for_promotion'])
        self.assertGreater(summary['comparison']['relative_ndcg10']['beauty'], 0)
        self.campaign.record_decision(above, 'promote', 'synthetic test', 'synthetic test')
