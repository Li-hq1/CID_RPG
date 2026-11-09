import math
import os
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader

from engine.evaluate import evaluate
from engine.metrics import Evaluator
from misc.config import parse_config, validate_config
from models.evaluation_sampling import aggregate_diffusion_candidates
from models.tigerdiff import TigerDiff
from tests.test_tigerdiff import DummyDataset, DummyTokenizer, make_batch, make_config


class CandidateAggregationTest(unittest.TestCase):
    def test_three_rankings_share_candidates_but_use_different_evidence(self):
        items = torch.tensor([[0.0], [1.0], [3.0]])
        samples = torch.tensor([[[0.9], [0.0], [1.1]]])
        expected = {"support": [2, 1], "first_seen": [2, 1], "min_distance": [1, 2]}
        for ranking, ids in expected.items():
            with self.subTest(ranking=ranking):
                preds, stats = aggregate_diffusion_candidates(samples, items, 2, {
                    "eval_candidates_per_sample": 1, "eval_candidate_ranking": ranking,
                })
                self.assertEqual(preds.tolist(), [[[i] for i in ids]])
                self.assertEqual(stats["num_samples"].tolist(), [3])
                self.assertAlmostEqual(stats["deduplication_rate"].item(), 1 / 3, places=6)

    def test_support_matches_exponential_sum_not_normalized_votes(self):
        items = torch.tensor([[0.0], [1.0], [2.0]])
        samples = torch.tensor([[[0.2], [1.4]]])
        temperature = 0.3
        for metric in ("l2", "cosine", "inner_product"):
            config = {"eval_candidates_per_sample": 3, "eval_candidate_metric": metric,
                      "eval_support_temperature": temperature}
            distances = {
                "l2": [[(s - i) ** 2 for i in (0, 1, 2)] for s in (0.2, 1.4)],
                "cosine": [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                "inner_product": [[-s * i for i in (0, 1, 2)] for s in (0.2, 1.4)],
            }[metric]
            supports = [sum(math.exp(-row[i] / temperature) for row in distances)
                        for i in range(3)]
            expected = sorted(range(3), key=lambda i: (-supports[i],
                              min(row[i] for row in distances), i))
            preds, _ = aggregate_diffusion_candidates(samples, items, 3, config)
            self.assertEqual(preds.flatten().tolist(), [i + 1 for i in expected])

    def test_expand_existing_neighbors_without_changing_discovery_prefix(self):
        items = torch.tensor([[0.0], [1.0], [2.0], [10.0]])
        samples = torch.tensor([[[0.0], [0.0]]])
        preds, stats = aggregate_diffusion_candidates(samples, items, 3, {
            "eval_candidates_per_sample": 1, "eval_candidate_ranking": "first_seen",
        })
        self.assertEqual(preds.flatten().tolist(), [1, 2, 3])
        self.assertEqual(stats["retrieval_depth"].item(), 3)
        self.assertEqual(stats["unique_candidates"].item(), 3)
        self.assertEqual(stats["candidate_occurrences"].item(), 6)

    def test_metric_changes_nearest_candidate(self):
        items = torch.tensor([[2.0, 0.0], [0.8, 0.3], [0.0, 1.0]])
        samples = torch.tensor([[[1.0, 0.0]]])
        for metric, expected in (("l2", 2), ("cosine", 1), ("inner_product", 1)):
            preds, _ = aggregate_diffusion_candidates(samples, items, 1, {
                "eval_candidates_per_sample": 1, "eval_candidate_metric": metric,
            })
            self.assertEqual(preds.item(), expected)

    def test_expansion_keeps_initial_discoveries_ahead_of_new_neighbors(self):
        preds, _ = aggregate_diffusion_candidates(
            torch.tensor([[[0.0], [10.0]]]), torch.tensor([[0.0], [1.0], [9.0], [10.0]]), 4,
            {"eval_candidates_per_sample": 1, "eval_candidate_ranking": "first_seen"},
        )
        self.assertEqual(preds.flatten().tolist(), [1, 4, 2, 3])

    def test_short_catalog_and_ties_across_retrieval_blocks(self):
        for ranking in ("support", "first_seen", "min_distance"):
            preds, _ = aggregate_diffusion_candidates(
                torch.ones(2, 2, 1), torch.ones(2, 1), 10,
                {"eval_candidate_ranking": ranking},
            )
            self.assertEqual(preds.tolist(), [[[1], [2]], [[1], [2]]])
        preds, _ = aggregate_diffusion_candidates(
            torch.ones(1, 2, 1), torch.ones(4100, 1), 3,
            {"eval_candidates_per_sample": 1},
        )
        self.assertEqual(preds.flatten().tolist(), [1, 2, 3])

    def test_support_remains_finite_under_extreme_distances(self):
        for metric, items in (("l2", torch.tensor([[1000.0], [1001.0]])),
                              ("inner_product", torch.tensor([[1000.0], [1001.0]]))):
            preds, _ = aggregate_diffusion_candidates(torch.ones(1, 2, 1), items, 2, {
                "eval_candidates_per_sample": 2, "eval_candidate_metric": metric,
                "eval_support_temperature": 0.001,
            })
            self.assertEqual(preds.flatten().tolist(), [1, 2] if metric == "l2" else [2, 1])

    def test_equal_support_breaks_ties_by_minimum_distance_then_item_id(self):
        items = torch.tensor([[0.0], [1.0]])
        samples = torch.tensor([[[1.0], [0.1]]])
        # Each candidate receives one vote; a very large temperature makes the
        # double-precision support scores equal, exposing the distance tie-break.
        preds, _ = aggregate_diffusion_candidates(samples, items, 2, {
            "eval_candidates_per_sample": 1, "eval_support_temperature": 1e30,
        })
        self.assertEqual(preds.flatten().tolist(), [2, 1])


class MultiSamplingModelTest(unittest.TestCase):
    def make_model(self, **overrides):
        config = make_config()
        config.update(eval_sampling_mode="diffusion_multi", eval_num_samples=2,
                      eval_candidates_per_sample=1)
        config.update(overrides)
        return TigerDiff(config, DummyDataset(), DummyTokenizer()).eval()

    def test_batched_sampling_preserves_users_cfg_pairs_and_fixed_budget(self):
        model = self.make_model(tiger_guidance_weight=2.0, eval_log_sampling_stats=True)
        conditions = torch.arange(16, dtype=torch.float32).reshape(2, 8)
        model.forward = Mock(return_value=SimpleNamespace(final_states=conditions))

        def sample(z, temperature, cfg):
            torch.testing.assert_close(z[:4], conditions.repeat_interleave(2, dim=0))
            torch.testing.assert_close(z[4:], model.uncond_condition.expand(4, -1))
            self.assertEqual(cfg, 3.0)
            return model.item_id2embs[[1, 2, 3, 3, 2, 2, 2, 2]]

        model.diffloss.sample = Mock(side_effect=sample)
        preds = model.generate({}, n_return_sequences=3)
        self.assertEqual(preds.shape, (2, 3, 1))
        self.assertEqual(preds[0, :2, 0].tolist(), [1, 2])
        self.assertEqual(preds[1, 0, 0].item(), 3)
        model.diffloss.sample.assert_called_once()
        self.assertEqual(model.last_generation_stats["num_samples"].tolist(), [2, 2])

    def test_single_sample_degenerates_to_squared_l2_knn_and_checkpoint_loads(self):
        old_config = make_config()
        old_config["tiger_knn_metric"] = "l2"
        old = TigerDiff(old_config, DummyDataset(), DummyTokenizer()).eval()
        new = self.make_model(eval_num_samples=1, eval_candidates_per_sample=2)
        new.load_state_dict(old.state_dict(), strict=True)
        for model in (old, new):
            model.diffloss.sample = Mock(return_value=torch.tensor([[0.2, 0.9, 0.1]]))
        torch.testing.assert_close(old.generate(make_batch(), 2), new.generate(make_batch(), 2))
        self.assertIsNone(new.last_generation_stats)
        self.assertEqual(set(old.state_dict()), set(new.state_dict()))

    def test_seed_reproduces_actual_flow_and_gaussian_samples(self):
        for use_flow in (False, True):
            model = self.make_model(use_rectified_flow=use_flow)
            torch.manual_seed(123)
            first = model.generate(make_batch(), 2)
            torch.manual_seed(123)
            second = model.generate(make_batch(), 2)
            torch.testing.assert_close(first, second)

    def test_small_catalog_returns_only_valid_unique_items(self):
        for mode in ("knn", "diffusion_multi"):
            model = self.make_model(eval_sampling_mode=mode)
            model.diffloss.sample = Mock(return_value=model.item_id2embs[[1, 1]])
            preds = model.generate(make_batch(), 10)
            self.assertEqual(preds.shape, (1, 3, 1))
            self.assertEqual(sorted(preds.flatten().tolist()), [1, 2, 3])


class EvaluationIntegrationTest(unittest.TestCase):
    def test_real_model_and_evaluator_handle_small_catalog_and_stats(self):
        config = make_config()
        config.update(eval_sampling_mode="diffusion_multi", eval_num_samples=2,
                      eval_candidates_per_sample=1, eval_log_sampling_stats=True,
                      use_ddp=False, topk=[5, 10], metrics=["ndcg", "recall"])
        tokenizer = DummyTokenizer()
        model = TigerDiff(config, DummyDataset(), tokenizer).eval()
        model.diffloss.sample = Mock(return_value=model.item_id2embs[[3, 2]])
        batch = make_batch()
        batch["labels"] = torch.tensor([[3]])
        accelerator = Mock(device=torch.device("cpu"))
        accelerator.unwrap_model.return_value = model
        results = evaluate(model, [batch], config, tokenizer, accelerator)
        self.assertEqual(results["recall@10"], 1.0)
        self.assertAlmostEqual(results["ndcg@10"], 1 / math.log2(3), places=6)
        self.assertEqual(results["sampling_num_samples"], 2)
        self.assertEqual(results["sampling_unique_candidates"], 3)

    def test_gather_trims_predictions_labels_and_per_user_stats_together(self):
        config = {"use_ddp": True, "topk": [5, 10], "metrics": ["recall", "ndcg"],
                  "eval_log_sampling_stats": True}
        model = Mock()
        model.generate.return_value = torch.tensor([[[2], [1]], [[1], [2]]])
        model.last_generation_stats = {"unique_candidates": torch.tensor([2., 99.])}
        accelerator = Mock(device=torch.device("cpu"))
        accelerator.unwrap_model.return_value = model

        def gather(payload):
            preds, labels, stats = payload
            return preds[:1], labels[:1], {key: value[:1] for key, value in stats.items()}

        accelerator.gather_for_metrics.side_effect = gather
        tokenizer = SimpleNamespace(eos_token=None, item2tokens=[None, None])
        batch = {"labels": torch.tensor([[1], [2]])}
        results = evaluate(model, [batch], config, tokenizer, accelerator)
        self.assertEqual(results["recall@10"], 1.0)
        self.assertAlmostEqual(results["ndcg@10"], 1 / math.log2(3), places=6)
        self.assertEqual(results["sampling_unique_candidates"], 2)
        accelerator.gather_for_metrics.assert_called_once()
        config["eval_log_sampling_stats"] = False
        config["use_ddp"] = False
        results = evaluate(model, [batch], config, tokenizer, accelerator)
        self.assertFalse(any(key.startswith("sampling_") for key in results))

    def test_short_predictions_leave_missing_ranks_as_misses(self):
        evaluator = Evaluator({"topk": [5, 10], "metrics": ["ndcg", "recall"]},
                              SimpleNamespace(eos_token=None, item2tokens=[1, 2]))
        results = evaluator.calculate_metrics(torch.tensor([[[2], [1]]]), torch.tensor([[3]]))
        self.assertEqual(results["recall@10"].item(), 0)
        self.assertEqual(results["ndcg@10"].item(), 0)


@unittest.skipUnless(int(os.environ.get("WORLD_SIZE", "1")) > 1, "Requires torchrun with two workers")
class DistributedSamplingTest(unittest.TestCase):
    def test_real_gather_removes_dataloader_padding_from_stats(self):
        accelerator = Accelerator(cpu=True)
        config = {"use_ddp": True, "topk": [5, 10], "metrics": ["recall", "ndcg"],
                  "eval_log_sampling_stats": True}

        class DistributedModel(torch.nn.Module):
            def generate(self, batch, n_return_sequences=1):
                self.last_generation_stats = {"unique_candidates": batch["count"].float()}
                return torch.ones(len(batch["labels"]), 2, 1, dtype=torch.long,
                                  device=batch["labels"].device)

        loader = DataLoader([{"labels": torch.tensor([1]), "count": torch.tensor(i)}
                             for i in (1, 2, 9)], batch_size=2)
        loader = accelerator.prepare(loader)
        tokenizer = SimpleNamespace(eos_token=None, item2tokens=[1, 2])
        try:
            results = evaluate(DistributedModel(), loader, config, tokenizer, accelerator)
            self.assertEqual(results["recall@10"], 1.0)
            self.assertEqual(results["ndcg@10"], 1.0)
            self.assertEqual(results["sampling_unique_candidates"], 4.0)
        finally:
            accelerator.end_training()


class SamplingConfigTest(unittest.TestCase):
    def test_both_configs_default_to_legacy_and_accept_cli_overrides(self):
        keys = ("eval_sampling_mode", "eval_num_samples", "eval_candidates_per_sample",
                "eval_candidate_metric", "eval_candidate_ranking",
                "eval_support_temperature", "eval_log_sampling_stats")
        original, _ = parse_config([])
        transformer, _ = parse_config(["--config=config_transformer.py"])
        self.assertEqual({key: original[key] for key in keys},
                         {key: transformer[key] for key in keys})
        self.assertEqual(original["eval_sampling_mode"], "knn")
        self.assertFalse(original["eval_log_sampling_stats"])
        changed, _ = parse_config([
            "--eval_sampling_mode=diffusion_multi", "--eval_num_samples=4",
            "--eval_candidates_per_sample=2", "--eval_candidate_metric=cosine",
            "--eval_candidate_ranking=first_seen", "--eval_support_temperature=0.5",
            "--eval_log_sampling_stats=true",
        ])
        self.assertEqual([changed[key] for key in keys],
                         ["diffusion_multi", 4, 2, "cosine", "first_seen", 0.5, True])

    def test_invalid_settings_fail_validation(self):
        original, _ = parse_config([])
        for key, value in (("eval_num_samples", 0), ("eval_num_samples", True),
                           ("eval_num_samples", 1.5), ("eval_candidates_per_sample", -1),
                           ("eval_support_temperature", 0), ("eval_support_temperature", math.nan),
                           ("eval_support_temperature", math.inf), ("eval_sampling_mode", "bad"),
                           ("eval_candidate_metric", "bad"), ("eval_candidate_ranking", "bad"),
                           ("eval_log_sampling_stats", "false")):
            with self.subTest(key=key, value=value), self.assertRaisesRegex(ValueError, key):
                validate_config(dict(original, **{key: value}))

    def test_old_config_without_sampling_fields_keeps_working(self):
        config, _ = parse_config([])
        config = {key: value for key, value in config.items()
                  if not key.startswith("eval_candidate") and key not in {
                      "eval_sampling_mode", "eval_num_samples", "eval_support_temperature",
                      "eval_log_sampling_stats",
                  }}
        validate_config(config)
        model = TigerDiff(make_config(), DummyDataset(), DummyTokenizer()).eval()
        model.diffloss.sample = Mock(return_value=model.item_id2embs[[2]])
        self.assertEqual(model.generate(make_batch(), 1).tolist(), [[[2]]])
        model.diffloss.sample.assert_called_once()


if __name__ == "__main__":
    unittest.main()
