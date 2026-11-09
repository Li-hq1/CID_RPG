import math
import unittest
from pathlib import Path

import torch
from torch.nn import functional as F

from joint_training.config import JointConfig, parse_config
from joint_training.model import JointModel
from tests.test_joint_model import make_batch


ROOT = Path(__file__).resolve().parents[1]


def fixture(*, eval_mode=None, eval_samples=4, **flow):
    settings = {"enabled": True, "source": "history", "max_queries": 4, "negatives": 2, "path": "mixture",
                "noise_scale": 1.0, "draws": 3, "catalog": "exact", "sampling": "uniform_corrected", **flow}
    cfg = JointConfig.from_dict({
        "data": {"max_item_seq_len": 4},
        "ae": {"architecture": "linear", "latent_dim": 3, "batch_size": 5, "warmup_ratio": 0.},
        "flow_ranking": settings,
        "recommender": {"encoder_type": "transformer", "n_embd": 8, "n_layer": 1, "n_head": 2, "n_inner": 16,
                        "diffloss_w": 16, "diffloss_d": 1, "num_sampling_steps": 2, "rectified_flow_steps": 10,
                        "embd_pdrop": 0., "attn_pdrop": 0., "tiger_uncond_prob": 0., "tiger_guidance_weight": 0.,
                        "warmup_ratio": 0.},
        "evaluation": {"eval_sampling_mode": eval_mode or ("mixture" if settings["path"] == "mixture"
                                                           else "diffusion_multi"), "eval_num_samples": eval_samples}})
    torch.manual_seed(13)
    return JointModel(cfg, torch.randn(5, 6))


class FinalSchemeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_final_config_is_the_default_scheme(self):
        final, _ = parse_config(["--config", str(ROOT / "configs/flowrec.json")])
        flow, rec = final.flow_ranking, final.recommender
        self.assertEqual((flow.path, flow.draws, flow.likelihood, flow.catalog), ("flowrec", 1, "marginal", "auto"))
        self.assertEqual((flow.noise_scale, flow.negatives, flow.max_exact_items), (0.0, 4096, 20000))
        self.assertEqual((flow.cfm_weight, flow.align_weight), (5., 1.))
        self.assertEqual(flow.vlad_clusters, 0)
        self.assertFalse(any(key.startswith("input_code_") for key in final.to_dict()["recommender"]))
        self.assertEqual(rec.num_sampling_steps, 2)
        self.assertEqual(final.evaluation.eval_sampling_mode, "knn")
        self.assertFalse(final.training.evaluate_test)
        self.assertIn('default=ROOT / "configs/flowrec.json"', (ROOT / "run.py").read_text())

    def test_control_smoke_uses_reduced_evaluation_budget(self):
        config, _ = parse_config(["--config", str(ROOT / "configs/flowrec.json"), "--smoke-test"])
        self.assertEqual(config.flow_ranking.vlad_clusters, 0)
        self.assertEqual(config.flow_ranking.draws, 1)
        self.assertEqual(config.evaluation.eval_num_samples, 1)

    def test_default_config_matches_control(self):
        final, _ = parse_config(["--config", str(ROOT / "configs/flowrec.json")])
        control, _ = parse_config(["--config", str(ROOT / "configs/control.json")])
        self.assertEqual(final.flow_ranking.path, "flowrec")
        self.assertEqual(control.flow_ranking.path, "mixture")
        self.assertEqual(final.recommender, control.recommender)
        self.assertEqual(final.ae, control.ae)

    def test_one_draw_per_draw_equals_matched_ranking(self):
        # draws=1 with the sampled catalog is the matched objective, term for term.
        matched = fixture(path="matched", draws=1, catalog="sampled")
        mixture = fixture(draws=1, catalog="sampled", likelihood="per_draw")
        mixture.load_state_dict(matched.state_dict())
        batch = make_batch()
        torch.manual_seed(5)
        expected, _ = matched.joint_objectives(batch)
        torch.manual_seed(5)
        actual, _ = mixture.joint_objectives(batch)
        torch.testing.assert_close(actual, expected)

    def test_exact_marginal_is_the_mixture_negative_log_likelihood(self):
        model = fixture(draws=3)
        batch = make_batch()
        torch.manual_seed(7)
        loss, _ = model.joint_objectives(batch)
        torch.manual_seed(7)
        from joint_training.contrastive import select_positions
        selection = select_positions(batch, 4)
        encoded, _ = model.encode_history(batch)
        condition = model.condition_proj(encoded[selection.rows, selection.positions])
        queries = model.transport(model.flow_prior(condition.repeat_interleave(3, 0)),
                                  condition.repeat_interleave(3, 0), 1)
        keys = model.encode_ids(torch.arange(1, model.n_items))
        log_p = (F.normalize(queries, dim=-1) @ F.normalize(keys, dim=-1).T / 0.07).log_softmax(-1)
        target = (selection.target_ids - 1).repeat_interleave(3)
        per = log_p.gather(1, target[:, None]).reshape(-1, 3)
        torch.testing.assert_close(loss, -(torch.logsumexp(per, 1) - math.log(3)).mean())
        self.assertLessEqual(loss.item(), -per.mean().item() + 1e-6)

    def test_mixture_gradients_reach_ae_history_and_flow(self):
        for form in ("direct", "direct_norm", "projected", "projected_norm"):
            with self.subTest(form=form):
                model = fixture(vlad_clusters=2, vlad_condition=form)
                loss, _ = model.joint_objectives(make_batch())
                loss.backward()
                parameters = [model.ae.encoder[0].weight, model.input_proj.linear.weight,
                              model.ranking_head.weight, model.diffloss.net.final_layer.linear.weight,
                              model.vlad.centers, model.vlad.fc_gk.weight, model.vlad.fc_g.weight]
                if form.startswith("projected"):
                    parameters.append(model.cluster_proj.weight)
                for parameter in parameters:
                    self.assertIsNotNone(parameter.grad)
                    self.assertGreater(parameter.grad.norm().item(), 0)

    def test_mixture_decoding_returns_ranked_catalog_items(self):
        model = fixture(vlad_clusters=2)
        model.eval()
        model.refresh_catalog()
        ids = model.generate(make_batch(), n_return_sequences=3).squeeze(-1)
        self.assertEqual(ids.shape[1], 3)
        self.assertTrue(((ids >= 1) & (ids < model.n_items)).all())
        self.assertTrue(all(len(row.unique()) == 3 for row in ids))

    def test_mixture_decoding_requires_the_mixture_path(self):
        with self.assertRaises(ValueError):
            fixture(path="matched", draws=1, catalog="sampled", eval_mode="mixture")
        with self.assertRaises(ValueError):
            fixture(eval_mode="diffusion_multi")
