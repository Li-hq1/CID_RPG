import unittest
from unittest.mock import Mock, patch

import torch
from torch.nn import functional as F

from joint_training.config import JointConfig, parse_config
from joint_training.flowrec import bridge_terms
from joint_training.model import JointModel
from joint_training.training import JointTrainer
from tests.test_final_scheme import ROOT
from tests.test_joint_model import make_batch


def fixture(**overrides):
    settings = {"enabled": True, "path": "flowrec", "source": "history", "noise_scale": 0.,
                "draws": 1, "steps": 1, "catalog": "exact", "sampling": "uniform_corrected",
                "max_queries": 4, "negatives": 3, **overrides}
    cfg = JointConfig.from_dict({
        "data": {"max_item_seq_len": 4},
        "ae": {"architecture": "linear", "latent_dim": 3, "batch_size": 5, "warmup_ratio": 0.},
        "flow_ranking": settings,
        "recommender": {"encoder_type": "hstu", "n_embd": 8, "n_layer": 1, "n_head": 2,
                        "hstu_hidden_dim": 4, "hstu_attention_dim": 4, "hstu_output_dropout": 0.,
                        "diffloss_w": 16, "diffloss_d": 1, "num_sampling_steps": settings["steps"] + 1,
                        "rectified_flow_steps": 11, "tiger_uncond_prob": 0., "tiger_guidance_weight": 0.,
                        "warmup_ratio": 0.},
        "evaluation": {"eval_sampling_mode": "knn", "eval_num_samples": 1,
                       "tiger_knn_metric": "cosine"}})
    torch.manual_seed(19)
    return JointModel(cfg, torch.randn(5, 6))


class FlowRecTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_true_bridge_velocity_recovers_target_at_every_time(self):
        base, target = torch.randn(4, 3), torch.randn(4, 3)
        fraction = torch.tensor([0., .2, .7, 1.])[:, None]
        state, cfm, endpoint = bridge_terms(base, target, fraction, base - target)
        torch.testing.assert_close(state, (1-fraction)*target + fraction*base)
        torch.testing.assert_close(endpoint, target)
        self.assertEqual(cfm.item(), 0.)

    def test_losses_match_independent_ce_and_mse_and_share_forward(self):
        model = fixture()
        batch = make_batch()
        rows, positions = torch.tensor([0, 0, 1, 1]), torch.tensor([0, 1, 0, 1])
        encoded, _ = model.encode_history(batch)
        condition = model.condition_proj(encoded[rows, positions])
        base = model.flow_prior(condition)
        keys = model.encode_ids(torch.arange(1, model.n_items))
        labels = torch.tensor([1, 2, 0, 2])
        times = torch.tensor([0, 2, 7, 10])
        s = times[:, None]/10
        state = (1-s)*keys[labels] + s*base
        velocity = torch.full_like(state, .2)
        unit_keys = F.normalize(keys, dim=-1)
        prior = F.cross_entropy(F.normalize(base, dim=-1) @ unit_keys.T/.07, labels)
        cfm = (velocity-(base-keys[labels])).square().mean()
        align = F.cross_entropy(F.normalize(state-s*velocity, dim=-1) @ unit_keys.T/.07, labels)
        model.diffloss.net.forward = Mock(return_value=velocity)
        with patch("joint_training.flowrec.torch.randint", return_value=times):
            loss, terms = model.joint_objectives(batch)
        torch.testing.assert_close(terms["prior"], prior)
        torch.testing.assert_close(terms["cfm"], cfm)
        torch.testing.assert_close(terms["align"], align)
        torch.testing.assert_close(loss, prior+5*cfm+align)
        model.diffloss.net.forward.assert_called_once()
        torch.testing.assert_close(model.diffloss.net.forward.call_args.args[0], state)

    def test_prior_is_deterministic_and_does_not_consume_rng(self):
        model = fixture()
        _, c = model.encode_history(make_batch())
        state = torch.get_rng_state().clone()
        first, second = model.flow_prior(c), model.flow_prior(c)
        torch.testing.assert_close(first, second)
        self.assertTrue(torch.equal(torch.get_rng_state(), state))
        torch.testing.assert_close(first.norm(dim=-1), torch.full((2,), 3**.5))

    def test_training_gradients_reach_shared_ae_encoder_anchor_and_velocity(self):
        model = fixture()
        loss, _ = model.joint_objectives(make_batch())
        loss.backward()
        for p in [model.ae.encoder[0].weight, model.input_proj.linear.weight,
                  model.ranking_head.weight, model.diffloss.net.final_layer.linear.weight]:
            self.assertIsNotNone(p.grad)
            self.assertGreater(float(p.grad.norm()), 0.)

    def test_cfm_and_alignment_separately_train_velocity(self):
        for cfm, align in [(5., 0.), (0., 1.)]:
            model = fixture(cfm_weight=cfm, align_weight=align)
            loss, _ = model.joint_objectives(make_batch())
            loss.backward()
            self.assertGreater(float(model.diffloss.net.final_layer.linear.weight.grad.norm()), 0.)

    def test_prior_only_skips_velocity_and_joint_update_stays_finite(self):
        model = fixture(cfm_weight=0., align_weight=0.)
        model.diffloss.net.forward = Mock(side_effect=AssertionError("prior only"))
        trainer = JointTrainer(model, model.config, torch.arange(1, 6), {"fixture": True}, 4)
        metrics = trainer.step(make_batch(), 1)
        self.assertTrue(torch.isfinite(torch.tensor(metrics["loss"])))
        self.assertEqual((trainer.ae_steps, trainer.rec_steps), (1, 1))
        self.assertEqual(metrics["loss/cfm"], 0.)

    def test_sampled_fallback_backpropagates(self):
        model = fixture(catalog="sampled")
        loss, _ = model.joint_objectives(make_batch())
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(float(model.ae.encoder[0].weight.grad.norm()), 0.)

    def test_euler_steps_follow_reverse_clock_and_generation_never_reads_labels(self):
        model = fixture(steps=4)
        model.eval()
        model.refresh_catalog()
        batch = make_batch()
        _, c = model.encode_history(batch)
        base = model.flow_prior(c)
        model.diffloss.net.forward = Mock(side_effect=lambda x, t, c: torch.ones_like(x))
        torch.testing.assert_close(model._sample_condition(c), base-1)
        times = [int(call.args[1][0]) for call in model.diffloss.net.forward.call_args_list]
        self.assertEqual(times, [10, 8, 5, 2])
        del batch["labels"]
        first = model.generate(batch, 3)
        torch.manual_seed(109)
        second = model.generate(batch, 3)
        self.assertTrue(torch.equal(first, second))

    def test_invalid_flowrec_settings_are_rejected(self):
        for patch_settings in [dict(noise_scale=1.), dict(draws=2), dict(source="noise"),
                               dict(cfm_weight=-1.), dict(align_weight=-1.)]:
            with self.assertRaises(ValueError):
                fixture(**patch_settings)

    def test_smoke_and_explicit_control_keep_distinct_decoders(self):
        config, _ = parse_config(["--config", str(ROOT/"configs/final.json"), "--smoke-test"])
        self.assertEqual(config.evaluation.eval_num_samples, 1)
        control, _ = parse_config(["--config", str(ROOT/"configs/control.json")])
        self.assertEqual(control.flow_ranking.path, "mixture")


if __name__ == "__main__":
    unittest.main()
