import math
import unittest
from pathlib import Path

import torch
from torch.nn import functional as F

from joint_training.config import JointConfig, parse_config
from joint_training.data import fit_input_codes
from joint_training.model import JointModel
from tests.test_joint_model import make_batch


ROOT = Path(__file__).resolve().parents[1]


def fixture(codes=None, eval_mode=None, **flow):
    settings = {"enabled": True, "source": "history", "max_queries": 4, "negatives": 2, "path": "mixture",
                "noise_scale": 1.0, "draws": 3, "catalog": "exact", "sampling": "uniform_corrected", **flow}
    cfg = JointConfig.from_dict({
        "data": {"max_item_seq_len": 4},
        "ae": {"architecture": "linear", "latent_dim": 3, "batch_size": 5, "warmup_ratio": 0.},
        "flow_ranking": settings,
        "recommender": {"encoder_type": "transformer", "n_embd": 8, "n_layer": 1, "n_head": 2, "n_inner": 16,
                        "diffloss_w": 16, "diffloss_d": 1, "num_sampling_steps": 2, "rectified_flow_steps": 10,
                        "embd_pdrop": 0., "attn_pdrop": 0., "tiger_uncond_prob": 0., "tiger_guidance_weight": 0.,
                        "warmup_ratio": 0., "input_code_levels": 0 if codes is None else codes.shape[1],
                        "input_code_size": 4},
        "evaluation": {"eval_sampling_mode": eval_mode or ("mixture" if settings["path"] == "mixture"
                                                           else "diffusion_multi"), "eval_num_samples": 4}})
    torch.manual_seed(13)
    return JointModel(cfg, torch.randn(5, 6), input_codes=codes)


class FinalSchemeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_final_config_is_the_default_scheme(self):
        final, _ = parse_config(["--config", str(ROOT / "configs/final.json")])
        flow, rec = final.flow_ranking, final.recommender
        self.assertEqual((flow.path, flow.draws, flow.likelihood, flow.catalog), ("mixture", 8, "marginal", "auto"))
        self.assertEqual((flow.noise_scale, flow.negatives, flow.max_exact_items), (1.0, 4096, 20000))
        self.assertEqual((rec.input_code_levels, rec.input_code_size), (3, 64))
        self.assertEqual(rec.num_sampling_steps, 2)
        self.assertEqual(final.evaluation.eval_sampling_mode, "mixture")
        self.assertFalse(final.training.evaluate_test)
        self.assertIn('default=ROOT / "configs/final.json"', (ROOT / "run.py").read_text())

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
        model = fixture(codes=torch.randint(0, 4, (5, 2)))
        loss, _ = model.joint_objectives(make_batch())
        loss.backward()
        for parameter in (model.ae.encoder[0].weight, model.input_proj.linear.weight, model.ranking_head.weight,
                          model.diffloss.net.final_layer.linear.weight, model.code_embedding.weight):
            self.assertIsNotNone(parameter.grad)
            self.assertGreater(parameter.grad.norm().item(), 0)

    def test_zero_code_embedding_starts_as_the_code_free_model(self):
        plain = fixture()
        coded = fixture(codes=torch.randint(0, 4, (5, 2)))
        coded.load_state_dict(plain.state_dict(), strict=False)
        batch = make_batch()
        plain.eval(); coded.eval()
        torch.testing.assert_close(coded.encode_history(batch)[1], plain.encode_history(batch)[1])
        self.assertEqual(coded.codes[0].abs().sum().item(), 0)

    def test_mixture_decoding_returns_ranked_catalog_items(self):
        model = fixture(codes=torch.randint(0, 4, (5, 2)))
        model.eval()
        model.refresh_catalog()
        ids = model.generate(make_batch(), n_return_sequences=3).squeeze(-1)
        self.assertEqual(ids.shape[1], 3)
        self.assertTrue(((ids >= 1) & (ids < model.n_items)).all())
        self.assertTrue(all(len(row.unique()) == 3 for row in ids))

    def test_input_codes_are_reproducible_and_precision_independent(self):
        raw = torch.randn(40, 6)
        previous = torch.get_float32_matmul_precision()
        try:
            torch.set_float32_matmul_precision("high")
            first = fit_input_codes(raw, 2, 4, 5, 3)
            torch.set_float32_matmul_precision("highest")
            second = fit_input_codes(raw, 2, 4, 5, 3)
        finally:
            torch.set_float32_matmul_precision(previous)
        torch.testing.assert_close(first, second, rtol=0, atol=0)
        self.assertEqual(tuple(first.shape), (40, 2))
        self.assertTrue(((first >= 0) & (first < 4)).all())

    def test_mixture_decoding_requires_the_mixture_path(self):
        with self.assertRaises(ValueError):
            fixture(path="matched", draws=1, catalog="sampled", eval_mode="mixture")
        with self.assertRaises(ValueError):
            fixture(eval_mode="diffusion_multi")
