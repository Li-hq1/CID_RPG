import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from joint_training.config import JointConfig
from joint_training.flow_ranking import candidate_ids, ranking_loss, directional_velocity_loss
from joint_training.model import JointModel
from joint_training.training import JointTrainer
from joint_training.checkpoint import save_checkpoint, read_checkpoint, load_checkpoint
from tests.test_joint_model import make_batch


def fixture(**settings):
    cfg = JointConfig.from_dict({
        "data": {"max_item_seq_len": 4},
        "ae": {"architecture": "linear", "latent_dim": 3, "batch_size": 5, "warmup_ratio": 0.},
        "flow_ranking": {"enabled": True, "max_queries": 4, "negatives": 2, "steps": 2, **settings},
        "recommender": {"encoder_type": "transformer", "n_embd": 8, "n_layer": 1,
            "n_head": 2, "n_inner": 16, "diffloss_w": 16, "diffloss_d": 1,
            "num_sampling_steps": 3 if settings.get("path", "rollout") in {"rollout", "metric_rf"} else 2,
            "rectified_flow_steps": 10, "embd_pdrop": 0., "attn_pdrop": 0.,
            "tiger_uncond_prob": 0., "tiger_guidance_weight": 0., "warmup_ratio": 0.}})
    torch.manual_seed(13)
    model = JointModel(cfg, torch.randn(5, 6))
    trainer = JointTrainer(model, cfg, torch.arange(1, 6), {"fixture": True}, 4)
    return model, trainer


class UnifiedFlowRankingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_sampler_scales_with_queries_and_draws_not_catalog(self):
        targets = torch.tensor([7, 7, 19, 31])
        ids, labels = candidate_ids(targets, 1_000_000_001, 16)
        self.assertLessEqual(len(ids), targets.unique().numel() + 16)
        torch.testing.assert_close(ids[labels], targets)
        self.assertEqual(len(ids), len(ids.unique()))

    def test_single_rank_updates_ae_history_and_flow_without_full_catalog(self):
        model, _ = fixture(source="history")
        with patch.object(model, "refresh_catalog", side_effect=AssertionError("full catalog")):
            objective, terms = model.joint_objectives(make_batch())
        self.assertEqual(set(terms), {"rf", "sampled_rank"})
        objective.backward()
        for parameter in (model.ae.encoder[0].weight, model.input_proj.linear.weight,
                          model.ranking_head.weight, model.diffloss.net.final_layer.linear.weight):
            self.assertIsNotNone(parameter.grad)
            self.assertGreater(parameter.grad.norm().item(), 0)
        self.assertFalse(hasattr(model, "teacher"))

    def test_rf_uses_fixed_regression_endpoints_but_live_history(self):
        model, _ = fixture(source="history")
        _, terms = model.joint_objectives(make_batch())
        params = (model.ranking_head.weight, model.ae.encoder[0].weight,
                  model.diffloss.net.final_layer.linear.weight)
        gradients = torch.autograd.grad(terms["rf"], params, allow_unused=True)
        self.assertIsNone(gradients[0])
        self.assertGreater(gradients[-1].norm().item(), 0)

    def test_rollout_input_does_not_contain_target_label(self):
        model, _ = fixture(source="history")
        model.eval()
        starts = []
        original = model.transport
        def traced(initial, condition, steps):
            starts.append(initial.detach().clone())
            return original(initial, condition, steps)
        with patch.object(model, "transport", side_effect=traced):
            torch.manual_seed(123)
            model.joint_objectives(make_batch())
            other = make_batch()
            other["labels"][other["labels"] > 0] = 4
            torch.manual_seed(123)
            model.joint_objectives(other)
        torch.testing.assert_close(starts[0], starts[1], rtol=0, atol=0)

    def test_rollout_is_identical_at_inference(self):
        model, _ = fixture(source="history")
        model.eval()
        with torch.no_grad():
            model.diffloss.net.final_layer.linear.weight.normal_(std=.1)
        _, h = model.encode_history(make_batch())
        torch.manual_seed(91)
        expected = model.transport(model.flow_prior(h), h, 2)
        torch.manual_seed(91)
        actual = model._sample_condition(h)
        torch.testing.assert_close(expected, actual, rtol=0, atol=0)

    def test_pairwise_loss_rewards_better_positive_separation(self):
        model, _ = fixture(objective="pairwise")
        keys = torch.eye(3)
        labels = torch.tensor([0])
        good = ranking_loss(torch.tensor([[1., .1, .1]]), keys, labels, model.config.flow_ranking)
        bad = ranking_loss(torch.tensor([[.1, 1., .1]]), keys, labels, model.config.flow_ranking)
        self.assertLess(good.item(), bad.item())

    def test_resume_reproduces_sampling_and_joint_updates(self):
        model, trainer = fixture(source="history")
        trainer.step(make_batch(), 1)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "state.pt"
            save_checkpoint(path, trainer, epoch=1, best_epoch=1, best_score=.1, stale_epochs=0)
            trainer.step(make_batch(), 2)
            expected = {k: v.clone() for k, v in model.state_dict().items()}
            restored, second = fixture(source="history")
            load_checkpoint(read_checkpoint(path), second, resume=True)
            second.step(make_batch(), 2)
            for key, value in restored.state_dict().items():
                torch.testing.assert_close(value, expected[key], rtol=0, atol=0)

    def test_local_ae_sampling_avoids_catalog_sized_permutations(self):
        _, trainer = fixture()
        trainer.train_ids = torch.arange(1, 1_000_001)
        original = torch.randperm
        def bounded(n, **kwargs):
            self.assertLess(n, 4096)
            return original(n, **kwargs)
        with patch("torch.randperm", side_effect=bounded):
            ids = trainer.sample_items(512)
        self.assertEqual(len(ids.unique()), 512)
        self.assertTrue(((ids > 0) & (ids <= 1_000_000)).all())

    def test_mixed_objectives_and_teacher_are_rejected(self):
        model, _ = fixture()
        model.config.ranking.teacher_weight = .1
        with self.assertRaisesRegex(ValueError, "Teacher"):
            model.config.validate()

    def test_corrected_uniform_partition_equals_full_ce_when_all_ids_drawn_once(self):
        model, _ = fixture(sampling="uniform_corrected")
        keys = torch.eye(3)
        generated = torch.tensor([[1., .1, .1], [.3, .4, .2]])
        labels = torch.tensor([0, 2])
        sampled = ranking_loss(generated, keys, labels, model.config.flow_ranking,
                               torch.arange(3), 3)
        model.config.flow_ranking.sampling = "batch"
        full = ranking_loss(generated, keys, labels, model.config.flow_ranking)
        torch.testing.assert_close(sampled, full)

    def test_matched_trains_the_inference_map_without_a_regression_forward(self):
        model, _ = fixture(path="matched", source="history")
        seen = []
        original = model.transport

        def traced(initial, condition, steps):
            seen.append(steps)
            return original(initial, condition, steps)

        with patch.object(model.diffloss.net, "forward", wraps=model.diffloss.net.forward) as call:
            with patch.object(model, "transport", side_effect=traced):
                loss, terms = model.joint_objectives(make_batch())
        self.assertEqual(set(terms), {"sampled_rank"})
        self.assertEqual(seen, [1])
        self.assertEqual(call.call_count, 1)
        loss.backward()
        for parameter in (model.ae.encoder[0].weight, model.ranking_head.weight,
                          model.diffloss.net.final_layer.linear.weight):
            self.assertIsNotNone(parameter.grad)
            self.assertGreater(parameter.grad.norm().item(), 0)

    def test_coupled_uses_one_network_call_and_retains_rf(self):
        model, _ = fixture(path="coupled", max_queries=4)
        with patch.object(model.diffloss.net, "forward", wraps=model.diffloss.net.forward) as call:
            loss, terms = model.joint_objectives(make_batch())
        self.assertEqual(call.call_count, 1)
        self.assertGreater(terms["rf"].item(), 0)
        loss.backward()
        self.assertGreater(model.diffloss.net.final_layer.linear.weight.grad.norm().item(), 0)

    def test_directional_rf_focuses_on_confusable_neighbor_error(self):
        keys = torch.tensor([[1., 0., 0.], [.9, .1, 0.], [-1., 0., 0.]])
        positive, labels = keys[:1], torch.tensor([0])
        orthogonal = directional_velocity_loss(torch.tensor([[0., 0., 1.]]), positive, keys, labels, 1)
        relevant = directional_velocity_loss(torch.tensor([[1., 0., 0.]]), positive, keys, labels, 1)
        self.assertEqual(orthogonal.item(), 0.)
        self.assertGreater(relevant.item(), 0.)
        model, _ = fixture(path="metric_rf", objective="directional")
        with patch.object(model.diffloss.net, "forward", wraps=model.diffloss.net.forward) as call:
            loss, terms = model.joint_objectives(make_batch())
        self.assertEqual(call.call_count, 1)
        self.assertEqual(set(terms), {"rf", "directional_rank"})
        loss.backward()

    def test_hnsw_retrieval_and_support_aggregation_preserve_small_exact_results(self):
        import importlib.util
        if importlib.util.find_spec("faiss") is None:
            self.skipTest("Optional faiss-cpu is unavailable")
        from joint_training.retrieval import CatalogIndex
        from models.evaluation_sampling import aggregate_diffusion_candidates
        generator = torch.Generator().manual_seed(22)
        items = torch.randn(80, 8, generator=generator)
        samples = torch.randn(3, 4, 8, generator=generator)
        index = CatalogIndex(items, ef_search=128)
        cfg = {"eval_candidate_metric": "l2", "eval_candidate_ranking": "support",
               "eval_candidates_per_sample": 1}
        exact, _ = aggregate_diffusion_candidates(samples, items, 10, cfg)
        ann, _ = aggregate_diffusion_candidates(samples, items, 10, cfg, candidate_lookup=index.search)
        torch.testing.assert_close(exact, ann)


if __name__ == "__main__":
    unittest.main()
