import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch.nn import functional as F

from joint_training.checkpoint import load_checkpoint, read_checkpoint, save_checkpoint
from joint_training.contrastive import select_positions
from joint_training.training import JointTrainer
from models.vlad import CausalNeXtVLAD
from tests.test_final_scheme import fixture
from tests.test_joint_model import make_batch


class VLADMixtureTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_future_and_padding_do_not_change_prefix_descriptors_or_mass(self):
        torch.manual_seed(3)
        vlad = CausalNeXtVLAD(8, 4)
        x = torch.randn(2, 5, 8)
        mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]], dtype=torch.bool)
        expected, mass = vlad(x, mask)
        changed = x.clone()
        changed[:, 3:] = torch.randn_like(changed[:, 3:]) * 100
        actual, changed_mass = vlad(changed, mask)
        torch.testing.assert_close(actual[:, :3], expected[:, :3])
        torch.testing.assert_close(changed_mass[:, :3], mass[:, :3])
        torch.testing.assert_close(actual[0], expected[0])
        torch.testing.assert_close(changed_mass[0], mass[0])
        torch.testing.assert_close(mass[0, 3:], mass[0, 2:3].expand(2, -1))
        torch.testing.assert_close(expected.norm(dim=-1), torch.ones(2, 5, 4))
        empty, empty_mass = vlad(x, torch.zeros_like(mask))
        self.assertEqual(empty.abs().sum().item(), 0)
        self.assertEqual(empty_mass.sum().item(), 0)

    def test_mass_is_the_gated_soft_count_and_weights_only_use_history(self):
        vlad = CausalNeXtVLAD(8, 2)
        with torch.no_grad():
            vlad.fc_gk.weight.zero_()
            vlad.fc_gk.bias.copy_(torch.tensor([math.log(3.), 0.] * 8))
            vlad.fc_g.weight.zero_()
            vlad.fc_g.bias.zero_()
        _, mass = vlad(torch.randn(1, 3, 8), torch.ones(1, 3, dtype=torch.bool))
        torch.testing.assert_close(mass[0], torch.tensor([[3., 1.], [6., 2.], [9., 3.]]))
        model = fixture(vlad_clusters=2)
        model.vlad = vlad
        batch = make_batch()
        encoded, _ = model.encode_history(batch)
        selection = select_positions(batch, 4)
        conditions, log_pi = model.mixture_conditions(encoded, batch["attention_mask"],
                                                     selection.rows, selection.positions)
        torch.testing.assert_close(log_pi.exp(), torch.tensor([[.75, .25]]).expand(len(conditions), -1),
                                   atol=1e-6, rtol=1e-6)

    def test_cluster_and_noise_marginal_matches_explicit_probability_sum(self):
        model, batch = fixture(vlad_clusters=2, draws=2), make_batch()
        torch.manual_seed(7)
        actual, _ = model.joint_objectives(batch)
        torch.manual_seed(7)
        selection = select_positions(batch, 4)
        encoded, _ = model.encode_history(batch)
        conditions, log_pi = model.mixture_conditions(encoded, batch["attention_mask"],
                                                     selection.rows, selection.positions)
        expanded = conditions.flatten(0, 1).repeat_interleave(2, 0)
        q = model.transport(model.flow_prior(expanded), expanded, 1)
        keys = model.encode_ids(torch.arange(1, model.n_items))
        p = (F.normalize(q, dim=-1) @ F.normalize(keys, dim=-1).T / .07).softmax(-1)
        target = (selection.target_ids - 1).repeat_interleave(4)
        p_y = p.gather(1, target[:, None]).reshape(-1, 2, 2)
        expected = -((p_y.mean(-1) * log_pi.exp()).sum(-1)).log().mean()
        torch.testing.assert_close(actual, expected)

    def test_sampled_catalog_supports_vlad_gradients(self):
        model = fixture(vlad_clusters=2, draws=1, catalog="sampled")
        loss, _ = model.joint_objectives(make_batch())
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(model.vlad.fc_gk.weight.grad.norm().item(), 0)

    def test_decoder_mixes_probabilities_with_cluster_weights(self):
        model, batch = fixture(vlad_clusters=2), make_batch()
        model.eval()
        model.refresh_catalog()
        b, k, s = len(batch["input_ids"]), 2, 2
        q = torch.randn(b * k * s, model.latent_dim)
        pi = torch.tensor([.8, .2]).expand(b, -1)
        conditions = torch.zeros(b, k, model.hidden_embd)
        expected_per_query = (F.normalize(q, dim=-1)
                              @ F.normalize(model.catalog[1:], dim=-1).T / .07).softmax(-1)
        expected = (expected_per_query.reshape(b, k, s, -1).mean(2) * pi[..., None]).sum(1)
        with patch.object(model, "mixture_conditions", return_value=(conditions, pi.log())), \
                patch.object(model, "_sample_condition", return_value=q) as sample:
            actual = model.mixture_log_probabilities(batch, draws=4).exp()
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(actual.sum(-1), torch.ones(b))
        self.assertEqual(sample.call_args.args[0].shape[0], b * 4)
        with self.assertRaisesRegex(ValueError, "multiple"):
            model.mixture_log_probabilities(batch, draws=3)

    def test_k8_s1_has_eight_training_queries_and_32_evaluation_queries(self):
        model, batch = fixture(vlad_clusters=8, draws=1, eval_samples=32), make_batch()
        with patch.object(model, "transport", wraps=model.transport) as transport:
            model.joint_objectives(batch)
        self.assertEqual(transport.call_args.args[0].shape[0], model.last_training_stats["queries"] * 8)
        model.eval()
        model.refresh_catalog()
        with patch.object(model, "transport", wraps=model.transport) as transport:
            model.generate(batch, n_return_sequences=3)
        self.assertEqual(transport.call_args.args[0].shape[0], len(batch["input_ids"]) * 32)

    def test_checkpoint_resume_restores_vlad_and_next_update(self):
        model = fixture(vlad_clusters=2, draws=1)
        cfg = model.config
        trainer = JointTrainer(model, cfg, torch.arange(1, 6), {"fixture": "vlad"}, 5)
        trainer.step(make_batch(), 1)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "vlad.pt"
            save_checkpoint(path, trainer, epoch=1, best_epoch=1, best_score=.1, stale_epochs=0)
            trainer.step(make_batch(), 2)
            expected = {key: value.clone() for key, value in model.state_dict().items()}
            restored = fixture(vlad_clusters=2, draws=1)
            second = JointTrainer(restored, restored.config, torch.arange(1, 6), {"fixture": "vlad"}, 5)
            load_checkpoint(read_checkpoint(path), second, resume=True)
            second.step(make_batch(), 2)
            for key, value in restored.state_dict().items():
                torch.testing.assert_close(value, expected[key], rtol=0, atol=0)
            restored.config.flow_ranking.draws = 2
            with self.assertRaisesRegex(ValueError, "flow_ranking"):
                load_checkpoint(read_checkpoint(path), second, resume=True)

    def test_config_rejects_incompatible_sampling_and_geometry(self):
        for settings in ({"vlad_clusters": 3}, {"vlad_clusters": 2, "vlad_groups": 3},
                         {"vlad_clusters": 2, "likelihood": "per_draw"},
                         {"vlad_clusters": 2, "path": "matched"}):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                fixture(**settings)

    def test_control_has_no_vlad_or_content_code_parameters(self):
        model = fixture()
        self.assertIsNone(model.vlad)
        self.assertFalse(any(key.startswith(("vlad.", "cluster_", "codes", "code_embedding."))
                             for key in model.state_dict()))

    def test_retired_content_code_checkpoint_is_explicitly_rejected(self):
        from joint_training.checkpoint import migrate_checkpoint
        checkpoint = {"config": {"recommender": {"input_code_levels": 3}}}
        with self.assertRaisesRegex(ValueError, "Content-code checkpoints require archive"):
            migrate_checkpoint(checkpoint)

    def test_forms_use_only_the_descriptor_and_adapt_generator_width(self):
        for form in ("direct", "direct_norm", "projected", "projected_norm"):
            with self.subTest(form=form):
                model = fixture(vlad_clusters=2, vlad_condition=form)
                self.assertIsNone(model.condition_proj)
                self.assertFalse(hasattr(model, "cluster_embedding"))
                batch = make_batch()
                encoded, single = model.encode_history(batch)
                self.assertIsNone(single)
                v, mass = model.vlad(encoded, batch["attention_mask"])
                selection = select_positions(batch, 4)
                with patch.object(model.vlad, "forward", return_value=(v, mass)):
                    a, _ = model.mixture_conditions(encoded, batch["attention_mask"],
                                                    selection.rows, selection.positions)
                    b, _ = model.mixture_conditions(encoded * 100 + 100, batch["attention_mask"],
                                                    selection.rows, selection.positions)
                torch.testing.assert_close(a, b)
                expected_dim = model.vlad.S if form.startswith("direct") else model.hidden_embd
                self.assertEqual(a.shape[-1], expected_dim)
                self.assertEqual(model.ranking_head.in_features, expected_dim)
                if form == "direct":
                    torch.testing.assert_close(a, v[selection.rows, selection.positions])

    def test_assignment_temperature_preserves_counts_and_controls_sharpness(self):
        sharp = CausalNeXtVLAD(8, 4, assignment_temperature=.5)
        soft = CausalNeXtVLAD(8, 4, assignment_temperature=2.)
        soft.load_state_dict(sharp.state_dict())
        x, mask = torch.randn(2, 4, 8), torch.ones(2, 4, dtype=torch.bool)
        _, a = sharp(x, mask)
        _, b = soft(x, mask)
        torch.testing.assert_close(a.sum(-1), b.sum(-1))
        pa, pb = a / a.sum(-1, keepdim=True), b / b.sum(-1, keepdim=True)
        self.assertLess(float(-(pa * pa.log()).sum(-1).mean().detach()),
                        float(-(pb * pb.log()).sum(-1).mean().detach()))

    def test_old_three_term_vlad_checkpoint_requires_its_snapshot(self):
        from joint_training.checkpoint import migrate_checkpoint
        with self.assertRaisesRegex(ValueError, "Residual VLAD checkpoints require archive"):
            migrate_checkpoint({"config": {"flow_ranking": {"vlad_clusters": 8}}})

    def test_control_checkpoint_ignores_disabled_vlad_tuning(self):
        model = fixture()
        trainer = JointTrainer(model, model.config, torch.arange(1, 6), {"fixture": "control"}, 5)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "control.pt"
            save_checkpoint(path, trainer, epoch=0, best_epoch=0, best_score=0., stale_epochs=0)
            model.config.flow_ranking.vlad_condition = "direct_norm"
            model.config.flow_ranking.vlad_groups = 2
            load_checkpoint(read_checkpoint(path), trainer)
