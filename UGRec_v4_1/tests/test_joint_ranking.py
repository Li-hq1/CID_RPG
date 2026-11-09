"""Meaningful controls for the new gradient paths and evaluation reproducibility."""

import random
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from joint_training.config import JointConfig
from joint_training.model import JointModel
from joint_training.training import JointTrainer
from joint_training.checkpoint import load_checkpoint, read_checkpoint, save_checkpoint
from joint_training.evaluation import evaluation_randomness
from joint_training.teacher import initialize_teacher
from tests.test_joint_model import make_batch


def fixture(**ranking):
    config = JointConfig.from_dict({
        "data": {"max_item_seq_len": 4},
        "ae": {"architecture": "linear", "latent_dim": 3, "batch_size": 5, "warmup_ratio": 0.},
        "ranking": {"enabled": True, "teacher_dim": 3, **ranking},
        "recommender": {"encoder_type": "transformer", "n_embd": 8, "n_layer": 1, "n_head": 2,
                        "n_inner": 16, "diffloss_w": 16, "diffloss_d": 1,
                        "num_sampling_steps": ranking.get("rollout_steps", 4) + 1 if ranking.get("rollout_weight", 0) else 2,
                        "rectified_flow_steps": 10, "embd_pdrop": 0., "attn_pdrop": 0.,
                        "tiger_uncond_prob": 0., "tiger_guidance_weight": 0., "warmup_ratio": 0.}})
    torch.manual_seed(7)
    model = JointModel(config, torch.randn(5, 6))
    if config.ranking.teacher_weight > 0:
        model.set_teacher(torch.randn(5, 3))
    trainer = JointTrainer(model, config, torch.arange(1, 6), {"fixture": True}, 4)
    return model, trainer


class JointRankingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_cli_smoke_keeps_training_and_inference_rollout_steps_matched(self):
        from joint_training.__main__ import smoke_configuration
        model, _ = fixture(rollout_weight=1.)
        smoke_configuration(model.config)
        self.assertEqual(model.config.recommender.num_sampling_steps,
                         model.config.ranking.rollout_steps + 1)

    def test_history_ce_updates_both_ae_and_sequence_encoder(self):
        model, _ = fixture(rf_weight=0.)
        objective, _ = model.joint_objectives(make_batch())
        objective.backward()
        self.assertGreater(model.ae.encoder[0].weight.grad.norm().item(), 0.)
        self.assertGreater(model.input_proj.linear.weight.grad.norm().item(), 0.)
        self.assertTrue(all(p.grad is None for p in model.diffloss.parameters()))

    def test_rf_and_prediction_ce_cannot_update_ae(self):
        model, _ = fixture(weight=0., prediction_weight=1., prediction_min_time=0.)
        with torch.no_grad():
            model.diffloss.net.final_layer.linear.weight.normal_(std=.1)
            model.diffloss.net.final_layer.adaLN_modulation[-1].weight.normal_(std=.1)
        objective, terms = model.joint_objectives(make_batch())
        self.assertIn("prediction", terms)
        objective.backward()
        self.assertTrue(all(p.grad is None for p in model.ae.parameters()))
        self.assertGreater(model.diffloss.net.final_layer.linear.weight.grad.norm().item(), 0.)

    def test_alignment_reaches_ae_denoiser_and_projection_but_not_teacher(self):
        model, _ = fixture(weight=0., rf_weight=0., teacher_weight=1., teacher_min_time=.25)
        objective, _ = model.joint_objectives(make_batch())
        objective.backward()
        self.assertGreater(model.ae.encoder[0].weight.grad.norm().item(), 0.)
        self.assertGreater(model.diffloss.net.input_proj.weight.grad.norm().item(), 0.)
        self.assertGreater(model.teacher_projection.weight.grad.norm().item(), 0.)
        self.assertFalse(model.teacher.requires_grad)

    def test_checkpoint_resume_is_exact_for_joint_updates(self):
        model, trainer = fixture(prediction_weight=.2, teacher_weight=.1, rollout_weight=.2)
        trainer.step(make_batch(), 1)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "joint.pt"
            save_checkpoint(path, trainer, epoch=1, best_epoch=1, best_score=.1, stale_epochs=0)
            trainer.step(make_batch(), 2)
            expected = {key: value.clone() for key, value in model.state_dict().items()}
            restored, second = fixture(prediction_weight=.2, teacher_weight=.1, rollout_weight=.2)
            load_checkpoint(read_checkpoint(path), second, resume=True)
            second.step(make_batch(), 2)
            for key, value in restored.state_dict().items():
                torch.testing.assert_close(value, expected[key], rtol=0, atol=0)

    def test_eval_rng_is_repeatable_and_restores_all_cpu_streams(self):
        random.seed(7)
        np.random.seed(7)
        torch.manual_seed(7)
        before = (random.getstate(), np.random.get_state(), torch.get_rng_state())
        with evaluation_randomness("cpu", 123):
            first = torch.randn(10)
            random.random()
            np.random.rand()
        with evaluation_randomness("cpu", 123):
            torch.testing.assert_close(first, torch.randn(10), rtol=0, atol=0)
        self.assertEqual(random.getstate(), before[0])
        np.testing.assert_array_equal(np.random.get_state()[1], before[1][1])
        torch.testing.assert_close(torch.get_rng_state(), before[2], rtol=0, atol=0)

    def test_masked_labels_do_not_contribute(self):
        model, _ = fixture(rf_weight=0.)
        batch = make_batch()
        batch["labels"][0, 2:] = 5
        loss, _ = model.joint_objectives(batch)
        batch["labels"][0, 2:] = -100
        same, _ = model.joint_objectives(batch)
        torch.testing.assert_close(loss, same)

    def test_teacher_artifact_checks_source_and_restores_from_checkpoint_without_file(self):
        model, trainer = fixture(teacher_weight=.1)
        source = {"embeddings_sha256": "raw", "item_order_sha256": "order", "train_mask_sha256": "train"}
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "teacher.pt"
            torch.save({"values": torch.randn(5, 3), "source": source}, path)
            model.config.ranking.teacher_path = str(path)
            trainer.source = initialize_teacher(model, model.config, source)
            bad = {**source, "train_mask_sha256": "validation"}
            with self.assertRaisesRegex(ValueError, "train_mask"):
                initialize_teacher(model, model.config, bad)
            checkpoint_path = Path(temp) / "model.pt"
            save_checkpoint(checkpoint_path, trainer, epoch=1, best_epoch=1, best_score=.1, stale_epochs=0)
            saved = read_checkpoint(checkpoint_path)
            path.unlink()
            restored, second = fixture(teacher_weight=.1)
            second.source = initialize_teacher(restored, restored.config, source, saved)
            load_checkpoint(saved, second)
            torch.testing.assert_close(restored.teacher, model.teacher, rtol=0, atol=0)

    def test_device_feature_cache_is_exact_and_not_serialized(self):
        model, _ = fixture()
        before = model.encode_ids(torch.tensor([1, 3, 2]))
        model.cache_features()
        after = model.encode_ids(torch.tensor([1, 3, 2]))
        torch.testing.assert_close(before, after, rtol=0, atol=0)
        self.assertNotIn("_feature_cache", model.state_dict())

    def test_history_prior_is_target_independent_and_matches_inference_initial_state(self):
        model, _ = fixture(prior_mode="history", prior_noise=.3)
        model.eval()
        batch = make_batch()
        _, condition = model.encode_history(batch)
        noise = torch.randn(2, 3)
        expected = model.flow_prior(condition, noise)
        batch["labels"].fill_(4)
        _, unchanged = model.encode_history(batch)
        torch.testing.assert_close(expected, model.flow_prior(unchanged, noise), rtol=0, atol=0)
        from unittest.mock import patch
        with patch.object(model.diffloss, "sample", return_value=expected) as sample:
            model._sample_condition(condition)
        self.assertIn("initial_noise", sample.call_args.kwargs)
        self.assertFalse(expected.requires_grad)

    def test_custom_flow_start_with_zero_velocity_remains_the_same(self):
        model, _ = fixture()
        start = torch.randn(2, 3)
        condition = torch.randn(2, 8)
        result = model.diffloss.sample(condition, initial_noise=start)
        torch.testing.assert_close(result, start, rtol=0, atol=0)

    def test_endpoint_input_does_not_change_when_training_targets_change(self):
        model, _ = fixture(endpoint_weight=1., weight=0., rf_weight=0.)
        calls = []
        handle = model.diffloss.net.register_forward_pre_hook(
            lambda module, args: calls.append((args[0].detach().clone(), args[1].clone())))
        batch = make_batch()
        torch.manual_seed(123)
        objective, _ = model.joint_objectives(batch)
        first = calls[-1]
        batch["labels"][batch["labels"] > 0] = 4
        torch.manual_seed(123)
        model.joint_objectives(batch)
        handle.remove()
        torch.testing.assert_close(first[0], calls[-1][0], rtol=0, atol=0)
        self.assertTrue((first[1] == 9).all())
        objective.backward()
        self.assertTrue(all(p.grad is None for p in model.ae.parameters()))

    def test_zero_velocity_one_step_matches_history_prior(self):
        model, _ = fixture(prior_mode="history", prior_noise=0.)
        model.eval()
        model.config.evaluation.prediction_source = "one_step"
        _, condition = model.encode_history(make_batch())
        expected = model.flow_prior(condition)
        actual = model._sample_condition(condition)
        torch.testing.assert_close(expected, actual, rtol=0, atol=0)

    def test_differentiable_rollout_matches_inference_and_reaches_denoiser(self):
        model, _ = fixture()
        with torch.no_grad():
            model.diffloss.net.final_layer.linear.weight.normal_(std=.1)
        state = torch.randn(2, 3)
        condition = torch.randn(2, 8)
        trained = model.transport(state, condition, 4)
        sampled = model.diffloss.gen_diffusion.sample_ode(model.diffloss.net, state.shape,
                    noise=state, model_kwargs={"c": condition}, num_steps=5, solver="euler")
        torch.testing.assert_close(trained, sampled, rtol=0, atol=0)
        trained.square().mean().backward()
        self.assertGreater(model.diffloss.net.final_layer.linear.weight.grad.norm().item(), 0.)

    def test_batched_l2_retrieval_preserves_ties_and_aggregation(self):
        from models.evaluation_sampling import aggregate_diffusion_candidates
        generator = torch.Generator().manual_seed(17)
        items = torch.randn(96, 8, generator=generator)
        items[1] = items[0]
        samples = torch.randn(4, 8, 8, generator=generator)
        samples[0] = items[0]
        for ranking in ("support", "min_distance", "first_seen"):
            cfg = {"eval_candidate_metric": "l2", "eval_candidate_ranking": ranking,
                   "eval_candidates_per_sample": 1}
            slow, _ = aggregate_diffusion_candidates(samples, items, 10, cfg)
            fast, _ = aggregate_diffusion_candidates(samples, items, 10, {**cfg, "eval_batched_retrieval": True})
            torch.testing.assert_close(slow, fast, rtol=0, atol=0)

    def test_rollout_ranking_does_not_update_ae(self):
        model, _ = fixture(weight=0., rf_weight=0., rollout_weight=1.)
        objective, _ = model.joint_objectives(make_batch())
        objective.backward()
        self.assertTrue(all(p.grad is None for p in model.ae.parameters()))
        self.assertGreater(model.diffloss.net.final_layer.linear.weight.grad.norm().item(), 0.)

    def test_all_recommendation_ae_routes_can_be_disabled_without_removing_teacher(self):
        model, _ = fixture(teacher_weight=.1, prediction_weight=.2, rollout_weight=.2,
                           history_ae_grad=False, catalog_ae_grad=False, teacher_ae_grad=False)
        objective, terms = model.joint_objectives(make_batch())
        objective.backward()
        self.assertIn("teacher", terms)
        self.assertTrue(all(p.grad is None for p in model.ae.parameters()))
        self.assertGreater(model.teacher_projection.weight.grad.norm().item(), 0.)

    def test_marginal_catalog_loss_matches_mixture_probability_and_jensen_bound(self):
        logits = torch.tensor([[3., -2., 0.], [-2., 3., 0.], [0., 0., 4.], [1., -1., 2.]], requires_grad=True)
        targets = torch.tensor([1, 3])
        marginal = JointModel.rollout_catalog_loss(logits, targets, 2, "marginal_ce")
        probabilities = logits.softmax(-1).reshape(2, 2, 3).mean(1)
        expected = -probabilities[torch.arange(2), targets - 1].log().mean()
        torch.testing.assert_close(marginal, expected)
        independent = JointModel.rollout_catalog_loss(logits, targets, 2, "sample_ce")
        self.assertLess(marginal.item(), independent.item())
        marginal.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        torch.testing.assert_close(JointModel.rollout_catalog_loss(logits, torch.tensor([1, 2, 3, 3]), 1, "marginal_ce"),
                                   JointModel.rollout_catalog_loss(logits, torch.tensor([1, 2, 3, 3]), 1, "sample_ce"))

    def test_marginal_rollout_routes_gradients_to_generator_and_not_ae(self):
        model, _ = fixture(weight=0., rf_weight=0., rollout_weight=1.,
                           rollout_samples=3, rollout_objective="marginal_ce")
        loss, terms = model.joint_objectives(make_batch())
        loss.backward()
        self.assertTrue(torch.isfinite(terms["rollout"]))
        self.assertTrue(all(p.grad is None for p in model.ae.parameters()))
        self.assertGreater(model.diffloss.net.final_layer.linear.weight.grad.norm().item(), 0.)


if __name__ == "__main__":
    unittest.main()
