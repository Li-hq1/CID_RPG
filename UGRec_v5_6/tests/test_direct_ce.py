import unittest
from unittest.mock import Mock

import torch
from torch.nn import functional as F

from joint_training.checkpoint import save_checkpoint, read_checkpoint
from joint_training.config import JointConfig, parse_config
from joint_training.direct_ce import direct_log_probabilities
from joint_training.model import JointModel
from joint_training.training import JointTrainer
from tests.test_final_scheme import ROOT
from tests.test_joint_model import make_batch


def fixture(direct=True, **changes):
    flow = dict(enabled=True, path="direct_ce" if direct else "mixture", source="history",
        noise_scale=0. if direct else 1., draws=1 if direct else 8,
        catalog="exact", max_queries=4, negatives=3, sampling="uniform_corrected", steps=1)
    flow.update(changes)
    config = JointConfig.from_dict({
        "data": {"max_item_seq_len": 4},
        "ae": {"architecture": "linear", "latent_dim": 3, "batch_size": 5, "warmup_ratio": 0.},
        "flow_ranking": flow,
        "recommender": {"encoder_type": "hstu", "n_embd": 8, "n_layer": 1, "n_head": 2,
            "hstu_hidden_dim": 4, "hstu_attention_dim": 4, "hstu_output_dropout": 0.,
            "diffloss_w": 16, "diffloss_d": 1, "num_sampling_steps": 2,
            "rectified_flow_steps": 11, "tiger_uncond_prob": 0., "tiger_guidance_weight": 0.,
            "warmup_ratio": 0.},
        "evaluation": {"eval_sampling_mode": "knn" if direct else "mixture", "eval_num_samples": 1 if direct else 4}})
    torch.manual_seed(29)
    return JointModel(config, torch.randn(5, 6))


class DirectCETest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_shared_initial_weights_and_rng_match_control_but_fm_is_removed(self):
        control = fixture(direct=False)
        rng = torch.get_rng_state()
        direct = fixture()
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertFalse(hasattr(direct, "diffloss"))
        self.assertFalse(hasattr(direct, "uncond_condition"))
        for name, value in direct.state_dict().items():
            torch.testing.assert_close(value, control.state_dict()[name], rtol=0, atol=0)
        self.assertTrue(all(not name.startswith(("diffloss.", "uncond_condition"))
                            for name in direct.state_dict()))
        self.assertLess(sum(p.numel() for p in direct.parameters()), sum(p.numel() for p in control.parameters()))

    def test_exact_ce_matches_formula_and_trains_shared_paths_without_noise_or_transport(self):
        model, batch = fixture(), make_batch()
        model.flow_prior = Mock(side_effect=AssertionError("no noise source"))
        model.transport = Mock(side_effect=AssertionError("no flow transport"))
        loss, terms = model.joint_objectives(batch)
        encoded, _ = model.encode_history(batch, ae_grad=True)
        mask = batch["labels"] > 0
        prediction = model.ranking_head(model.condition_proj(encoded[mask]))
        keys = model.encode_ids(torch.arange(1, model.n_items), grad=True)
        expected = F.cross_entropy(F.normalize(prediction, dim=-1) @ F.normalize(keys, dim=-1).T/.07,
                                   batch["labels"][mask]-1)
        torch.testing.assert_close(loss, expected)
        self.assertEqual(set(terms), {"ce"})
        loss.backward()
        for p in (model.ae.encoder[0].weight, model.input_proj.linear.weight,
                  model.condition_proj[1].weight, model.ranking_head.weight):
            self.assertGreater(float(p.grad.norm()), 0.)

    def test_direct_inference_is_target_free_and_deterministic(self):
        model = fixture()
        model.eval(); model.refresh_catalog()
        batch = make_batch(); del batch["labels"]
        first = direct_log_probabilities(model, batch)
        torch.manual_seed(3939)
        batch["labels"] = torch.full((2,4), 999999)
        torch.testing.assert_close(first, direct_log_probabilities(model, batch), rtol=0, atol=0)
        generated = model.generate(batch, n_return_sequences=5).squeeze(-1)
        self.assertTrue(torch.equal(generated, first.topk(5).indices+1))

    def test_optimizers_and_saved_checkpoint_have_no_flow_parameters(self):
        import tempfile
        from pathlib import Path
        model = fixture()
        trainer = JointTrainer(model, model.config, torch.arange(1,6), {"fixture":True}, 4)
        metrics = trainer.step(make_batch(), 1)
        self.assertTrue(torch.isfinite(torch.tensor(metrics["loss"])))
        self.assertEqual((trainer.ae_steps, trainer.rec_steps), (1,1))
        active = {id(p) for group in trainer.rec_optimizer.param_groups for p in group["params"]}
        self.assertEqual(active, {id(p) for p in model.rec_parameters()})
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"direct.pt"
            save_checkpoint(path, trainer, epoch=1, best_epoch=1, best_score=0., stale_epochs=0)
            checkpoint = read_checkpoint(path)
            self.assertTrue(all(not key.startswith(("diffloss.","uncond_condition")) for key in checkpoint["model"]))

    def test_corrected_sampled_fallback_is_finite(self):
        model = fixture(catalog="sampled")
        loss, _ = model.joint_objectives(make_batch())
        loss.backward()
        self.assertTrue(torch.isfinite(loss))

    def test_smoke_uses_one_query_and_unsupported_noise_settings_fail(self):
        config,_ = parse_config(["--config",str(ROOT/"configs/direct_ce.json"),"--smoke-test"])
        self.assertEqual(config.evaluation.eval_num_samples, 1)
        for changes in ({"noise_scale":1.},{"draws":8}):
            with self.assertRaises(ValueError):
                fixture(**changes)


if __name__ == "__main__":
    unittest.main()
