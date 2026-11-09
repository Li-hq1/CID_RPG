import unittest
from unittest.mock import Mock

import torch
from torch.nn import functional as F

from joint_training.config import JointConfig
from joint_training.contrastive import select_positions
from joint_training.model import JointModel
from joint_training.training import JointTrainer
from tests.test_joint_model import make_batch


def fixture(**overrides):
    flow = dict(enabled=True, path="clean_endpoint", source="history", draws=2, steps=1,
                max_queries=4, negatives=3, catalog="exact", likelihood="per_draw",
                sampling="uniform_corrected", fm_weight=1., ce_weight=1.)
    flow.update(overrides)
    config = JointConfig.from_dict({
        "data": {"max_item_seq_len": 4},
        "ae": {"architecture": "linear", "latent_dim": 3, "batch_size": 5, "warmup_ratio": 0.},
        "flow_ranking": flow,
        "recommender": {"encoder_type": "hstu", "n_embd": 8, "n_layer": 1, "n_head": 2,
            "hstu_hidden_dim": 4, "hstu_attention_dim": 4, "hstu_output_dropout": 0.,
            "diffloss_w": 16, "diffloss_d": 1, "num_sampling_steps": flow.get("steps", 1)+1,
            "rectified_flow_steps": 11, "tiger_uncond_prob": 0., "tiger_guidance_weight": 0.,
            "warmup_ratio": 0.},
        "evaluation": {"eval_sampling_mode": "mixture", "eval_num_samples": 4}})
    torch.manual_seed(19)
    model = JointModel(config, torch.randn(5, 6))
    # A nonzero output makes the independent CE/MSE comparison informative.
    torch.nn.init.normal_(model.diffloss.net.final_layer.linear.weight, std=.05)
    return model


class CleanEndpointTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_ce_and_fm_match_independent_formula_and_share_one_forward(self):
        model, batch = fixture(), make_batch()
        captured = []
        original = model.diffloss.net.forward
        def forward(x, t, c):
            result = original(x, t, c)
            captured.append((x, t, c, result))
            return result
        model.diffloss.net.forward = forward
        torch.manual_seed(11)
        loss, terms = model.joint_objectives(batch)
        self.assertEqual(len(captured), 1)
        selected = select_positions(batch, 4)
        keys = model.encode_ids(torch.arange(1, model.n_items))
        labels = (selected.target_ids-1).repeat_interleave(2)
        prediction = captured[0][3]
        expected_ce = F.cross_entropy(F.normalize(prediction, dim=-1) @ F.normalize(keys, dim=-1).T/.07, labels)
        torch.testing.assert_close(terms["ce"], expected_ce)
        torch.testing.assert_close(terms["fm"], F.mse_loss(prediction, keys[labels]))
        torch.testing.assert_close(loss, terms["ce"] + terms["fm"])
        loss.backward()
        for p in (model.ae.encoder[0].weight, model.input_proj.linear.weight,
                  model.ranking_head.weight, model.diffloss.net.final_layer.linear.weight):
            self.assertIsNotNone(p.grad)
            self.assertGreater(float(p.grad.norm()), 0.)

    def test_one_step_returns_clean_prediction_and_multi_step_uses_endpoint_clock(self):
        model = fixture(steps=4)
        c, base = torch.randn(4, 8), torch.randn(4, 3)
        model.diffloss.net.forward = Mock(side_effect=lambda x, t, c: torch.ones_like(x)*2)
        torch.testing.assert_close(model.transport(base, c, 1), torch.ones_like(base)*2)
        model.diffloss.net.forward.reset_mock()
        torch.testing.assert_close(model.transport(base, c, 4), torch.ones_like(base)*2)
        times = [int(call.args[1][0]) for call in model.diffloss.net.forward.call_args_list]
        self.assertEqual(times, [10, 8, 5, 2])
        first_state = model.diffloss.net.forward.call_args_list[1].args[0]
        torch.testing.assert_close(first_state, .75*base+.25*2)

    def test_mixture_generation_uses_history_without_target(self):
        model = fixture()
        model.eval(); model.refresh_catalog()
        batch = make_batch(); del batch["labels"]
        torch.manual_seed(3)
        first = model.mixture_log_probabilities(batch)
        batch["labels"] = torch.full((2, 4), 999999)
        torch.manual_seed(3)
        torch.testing.assert_close(first, model.mixture_log_probabilities(batch))
        torch.testing.assert_close(first.exp().sum(-1), torch.ones(len(first)))

    def test_fm_only_and_sampled_joint_both_update_finitely(self):
        for patch in ({"ce_weight": 0.}, {"catalog": "sampled"}):
            model = fixture(**patch)
            trainer = JointTrainer(model, model.config, torch.arange(1, 6), {"fixture": True}, 4)
            metrics = trainer.step(make_batch(), 1)
            self.assertTrue(torch.isfinite(torch.tensor(metrics["loss"])))
            self.assertEqual((trainer.ae_steps, trainer.rec_steps), (1, 1))

    def test_source_ce_state_contains_no_target_and_keeps_same_rng_draws(self):
        model = fixture(endpoint_training="source", fm_weight=0.)
        calls=[]
        original = model.diffloss.net.forward
        def forward(x, t, c):
            calls.append((x.detach(), t.detach()))
            return original(x,t,c)
        model.diffloss.net.forward=forward
        model.joint_objectives(make_batch())
        self.assertTrue((calls[0][1] == 10).all())

    def test_unsupported_settings_are_rejected(self):
        for patch in ({"fm_weight": -1.}, {"fm_weight": 0., "ce_weight": 0.},
                      {"likelihood": "marginal"}, {"source": "noise"}):
            with self.assertRaises(ValueError):
                fixture(**patch)


if __name__ == "__main__":
    unittest.main()
