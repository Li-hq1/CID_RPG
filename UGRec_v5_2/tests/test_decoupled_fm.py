import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.nn import functional as F

from joint_training.config import JointConfig, parse_config
from joint_training.checkpoint import save_checkpoint, read_checkpoint, load_checkpoint, evaluation_randomness
from joint_training.fm import build_model, DecoupledFMModel
from joint_training.contrastive import select_positions
from joint_training.training import JointTrainer
from tests.test_joint_model import make_batch


ROOT = Path(__file__).resolve().parents[1]


def fixture(*, architecture="parallel", scope="tail", catalog="exact", draws=2):
    config = JointConfig.from_dict({
        "data": {"max_item_seq_len": 4},
        "ae": {"architecture": "linear", "latent_dim": 4, "batch_size": 5, "warmup_ratio": 0.},
        "recommender": {"encoder_type": "hstu", "n_embd": 8, "n_layer": 2, "n_head": 2,
                        "hstu_hidden_dim": 4, "hstu_attention_dim": 4, "hstu_output_dropout": 0.,
                        "diffloss_w": 16, "diffloss_d": 1, "num_sampling_steps": 5,
                        "rectified_flow_steps": 10, "tiger_uncond_prob": 0., "tiger_guidance_weight": 0.,
                        "warmup_ratio": 0.},
        "fm": {"enabled": True, "architecture": architecture, "mse_scope": scope,
               "max_queries": 4, "negatives": 3, "catalog": catalog, "train_draws": draws},
        "evaluation": {"eval_sampling_mode": "mixture", "eval_num_samples": 4},
        "training": {"evaluate_test": False, "epochs": 2, "max_steps": 4},
    })
    torch.manual_seed(2024)
    return build_model(config, torch.randn(5, 6))


def warm_field(model):
    # Test routing with a nonzero condition Jacobian. Production keeps zero init.
    with torch.no_grad():
        model.diffloss.net.final_layer.linear.weight.normal_(std=.1)
        model.diffloss.net.final_layer.adaLN_modulation[-1].weight.normal_(std=.1)


def nonzero_names(model):
    return {n for n, p in model.named_parameters() if p.grad is not None and p.grad.norm() > 0}


class DecoupledFMTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_tails_start_equal_and_have_independent_parameters(self):
        model = fixture()
        self.assertEqual(len(model.encoder.layers), 1)
        self.assertEqual(len(model.ranking_tail.layers), 1)
        self.assertEqual(len(model.fm_tail.layers), 1)
        for first, second in zip(model.ranking_tail.parameters(), model.fm_tail.parameters()):
            torch.testing.assert_close(first, second)
            self.assertNotEqual(first.data_ptr(), second.data_ptr())

    def test_public_history_condition_includes_the_ranking_tail(self):
        model = fixture().eval()
        batch = make_batch()
        _, condition = model.encode_history(batch)
        rank, _ = model.history_branches(batch)
        torch.testing.assert_close(condition, rank[torch.arange(2), batch["seq_lens"] - 1])

    def test_ce_updates_ae_and_rank_path_but_never_fm(self):
        model = fixture()
        _, terms = model.joint_objectives(make_batch())
        terms["history_ce"].backward()
        names = nonzero_names(model)
        for prefix in ("ae.encoder.", "input_proj.", "encoder.", "ranking_tail.", "condition_proj.", "ranking_head."):
            self.assertTrue(any(n.startswith(prefix) for n in names), prefix)
        for name, parameter in model.named_parameters():
            if name.startswith(("fm_tail.", "fm_condition_proj.", "diffloss.", "ae.decoder.")):
                self.assertIsNone(parameter.grad, name)

    def test_mse_routes_exclude_ae_and_prior_even_with_live_endpoints(self):
        for architecture, scope in (("parallel", "tail"), ("parallel", "shared"),
                                    ("serial", "shared"), ("serial", "flow")):
            with self.subTest(architecture=architecture, scope=scope):
                model = fixture(architecture=architecture, scope=scope)
                warm_field(model)
                _, terms = model.joint_objectives(make_batch())
                terms["rf"].backward()
                names = nonzero_names(model)
                self.assertTrue(any(n.startswith("diffloss.") for n in names))
                forbidden = ["ae.", "ranking_head."]
                if architecture == "parallel":
                    forbidden += ["ranking_tail.", "condition_proj."]
                    self.assertTrue(any(n.startswith("fm_tail.") for n in names))
                if scope != "shared":
                    forbidden += ["input_proj.", "encoder."]
                else:
                    self.assertTrue(any(n.startswith("encoder.") for n in names))
                if scope == "flow":
                    forbidden += ["condition_proj."]
                for name, parameter in model.named_parameters():
                    if name.startswith(tuple(forbidden)):
                        self.assertIsNone(parameter.grad, name)

    def test_bridge_endpoints_sign_and_detached_regression_data(self):
        clean = torch.randn(3, 4, requires_grad=True)
        base = torch.randn(3, 4, requires_grad=True)
        start, target = DecoupledFMModel.bridge(clean, base, torch.ones(3, 1))
        end, _ = DecoupledFMModel.bridge(clean, base, torch.zeros(3, 1))
        torch.testing.assert_close(start, base)
        torch.testing.assert_close(end, clean)
        torch.testing.assert_close(start - target, clean)
        self.assertFalse(start.requires_grad)
        self.assertFalse(target.requires_grad)

    def test_prior_uses_history_only_and_blocks_anchor_gradient(self):
        model = fixture()
        rank, _ = model.history_branches(make_batch())
        anchor = model.flow_prior(rank[:, 0], noise=torch.zeros(2, 4))
        torch.testing.assert_close(anchor.norm(dim=-1), torch.full((2,), math.sqrt(4)))
        self.assertFalse(anchor.requires_grad)

    def test_exact_ce_matches_full_catalog_classification(self):
        model = fixture()
        _, terms = model.joint_objectives(make_batch())
        selection = select_positions(make_batch(), 4)
        rank, _ = model.history_branches(make_batch())
        queries = F.normalize(model.ranking_head(rank[selection.rows, selection.positions]), dim=-1)
        keys = F.normalize(model.encode_ids(torch.arange(1, 6)), dim=-1)
        expected = F.cross_entropy(queries @ keys.T / model.config.fm.temperature, selection.target_ids - 1)
        torch.testing.assert_close(terms["history_ce"], expected)

    def test_sampled_ce_and_cfm_are_finite_with_gradients(self):
        model = fixture(catalog="sampled")
        objective, terms = model.joint_objectives(make_batch())
        self.assertTrue(torch.isfinite(objective))
        self.assertEqual(set(terms), {"history_ce", "rf"})
        objective.backward()
        self.assertEqual(model.last_training_stats["exact_ce"], 0)
        self.assertEqual(model.last_training_stats["fm_queries"], 8)

    def test_zero_mse_control_has_only_ce_recommendation_gradients(self):
        model = fixture(architecture="serial", scope="shared")
        model.config.fm.mse_weight = 0.
        model.config.evaluation.prediction_source = "ranking"
        model.config.validate()
        warm_field(model)
        objective, terms = model.joint_objectives(make_batch())
        torch.testing.assert_close(objective, terms["history_ce"])
        parameters = list(model.parameters())
        expected = torch.autograd.grad(terms["history_ce"], parameters, retain_graph=True, allow_unused=True)
        objective.backward()
        for parameter, gradient in zip(parameters, expected):
            if gradient is not None:
                torch.testing.assert_close(parameter.grad, gradient)
            elif parameter.grad is not None:
                self.assertEqual(parameter.grad.count_nonzero().item(), 0)

    def test_zero_mse_cannot_be_reported_as_a_trained_flow(self):
        model = fixture()
        model.config.fm.mse_weight = 0.
        with self.assertRaisesRegex(ValueError, "CE-only"):
            model.config.validate()

    def test_cfm_input_and_loss_match_continuous_straight_bridge(self):
        model = fixture()
        captured = {}
        def capture(module, args):
            captured["state"], captured["times"], captured["condition"] = args
        hook = model.diffloss.net.register_forward_pre_hook(capture)
        try:
            with patch.object(model, "bridge", wraps=model.bridge) as bridge:
                _, terms = model.joint_objectives(make_batch())
                clean, base, fraction = bridge.call_args.args
        finally:
            hook.remove()
        torch.testing.assert_close(captured["state"], (1 - fraction) * clean + fraction * base)
        torch.testing.assert_close(captured["times"], fraction[:, 0] * 9)
        self.assertTrue(((fraction > 0) & (fraction < 1)).all())
        self.assertTrue((captured["times"] != captured["times"].round()).any())
        # Zero-initialized production field has velocity zero.
        torch.testing.assert_close(terms["rf"], (base - clean).square().mean())

    def test_both_hstu_tails_are_causal_and_ignore_padding(self):
        model = fixture().eval()
        batch = make_batch()
        original = model.history_branches(batch)
        batch["input_ids"][0, 1] = 5
        batch["input_ids"][:, 2:] = 4
        changed = model.history_branches(batch)
        for first, second in zip(original, changed):
            torch.testing.assert_close(first[:, 0], second[:, 0])
            torch.testing.assert_close(first[:, 2:], second[:, 2:])

    def test_generation_is_label_independent_and_mixture_is_normalized(self):
        model = fixture().eval()
        model.refresh_catalog()
        batch = make_batch()
        for source in ("flow", "ranking", "prior", "one_step"):
            with evaluation_randomness("cpu", 20261028):
                first = model.mixture_log_probabilities(batch, source=source, draws=9)
            batch["labels"] = torch.full_like(batch["labels"], -100)
            with evaluation_randomness("cpu", 20261028):
                second = model.mixture_log_probabilities(batch, source=source, draws=9)
            torch.testing.assert_close(first, second, rtol=0, atol=0)
            torch.testing.assert_close(first.exp().sum(-1), torch.ones(2))
        prediction = model.generate(batch, n_return_sequences=3)
        self.assertEqual(prediction.shape, (2, 3, 1))
        self.assertTrue(((prediction > 0) & (prediction < model.n_items)).all())

    def test_euler_integrates_an_oracle_velocity_to_the_target(self):
        model = fixture()
        clean, base = torch.randn(2, 4), torch.randn(2, 4)
        class Oracle(nn.Module):
            def forward(self, state, time, condition):
                return base - clean
        model.diffloss.net = Oracle()
        for steps in (1, 4, 8):
            torch.testing.assert_close(model.transport(base, torch.zeros(2, 8), steps), clean)

    def test_chunked_decoder_equals_the_explicit_probability_mixture(self):
        model = fixture().eval()
        warm_field(model)
        model.refresh_catalog()
        batch = make_batch()
        with torch.no_grad(), evaluation_randomness("cpu", 20261028):
            rank_all, fm_all = model.history_branches(batch)
            rank = rank_all[torch.arange(2), batch["seq_lens"] - 1].repeat_interleave(9, 0)
            condition = fm_all[torch.arange(2), batch["seq_lens"] - 1].repeat_interleave(9, 0)
            samples = model.transport(model.flow_prior(rank), condition, 4)
            logits = F.normalize(samples, dim=-1) @ F.normalize(model.catalog[1:], dim=-1).T / model.config.fm.temperature
            expected = logits.softmax(-1).reshape(2, 9, -1).mean(1)
        with evaluation_randomness("cpu", 20261028):
            actual = model.mixture_log_probabilities(batch, draws=9).exp()
        torch.testing.assert_close(actual, expected)

    def test_production_zero_init_eventually_updates_the_fm_tail(self):
        model = fixture()
        for settings in (model.config.ae, model.config.recommender):
            settings.scheduler = "constant"
        trainer = JointTrainer(model, model.config, torch.arange(1, 6), {"fixture": True}, 4)
        before = {n: p.detach().clone() for n, p in model.fm_tail.named_parameters()}
        for step in range(4):
            trainer.step(make_batch(), 1)
        self.assertTrue(any(not torch.equal(before[n], p) for n, p in model.fm_tail.named_parameters()))
        self.assertTrue(any(n.startswith("fm_tail.") for n in nonzero_names(model)))

    def test_checkpoint_resume_restores_both_heads_optimizers_and_rng(self):
        model = fixture()
        trainer = JointTrainer(model, model.config, torch.arange(1, 6), {"fixture": True}, 4)
        trainer.step(make_batch(), 1)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "checkpoint.pt"
            save_checkpoint(path, trainer, epoch=1, best_epoch=1, best_score=.1, stale_epochs=0)
            trainer.step(make_batch(), 2)
            expected = {n: t.clone() for n, t in model.state_dict().items()}
            restored = fixture()
            second = JointTrainer(restored, restored.config, torch.arange(1, 6), {"fixture": True}, 4)
            load_checkpoint(read_checkpoint(path), second, resume=True)
            second.step(make_batch(), 2)
            for name, value in restored.state_dict().items():
                torch.testing.assert_close(value, expected[name], rtol=0, atol=0)
            restored.config.fm.mse_scope = "shared"
            with self.assertRaisesRegex(ValueError, "fm configuration mismatch"):
                load_checkpoint(read_checkpoint(path), second)

    def test_all_presets_validate_and_cannot_mix_with_output_ranking(self):
        for name in ("final", "parallel_shared", "serial_shared", "serial_flow_only"):
            config, _ = parse_config(["--config", str(ROOT / f"configs/{name}.json")])
            config.validate()
            config.flow_ranking.enabled = True
            with self.assertRaisesRegex(ValueError, "replaces"):
                config.validate()

    def test_protocol_drops_test_target_before_id_mapping(self):
        from protocol import corpus
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / "AmazonReviews2014/Beauty/processed"
            directory.mkdir(parents=True)
            (directory / "id_mapping.json").write_text(json.dumps({"item2id": {"a": 1, "b": 2},
                                                                 "id2item": [None, "a", "b"]}))
            (directory / "all_item_seqs.json").write_text(json.dumps({"u": ["a", "a", "b", "TEST_NOT_IN_MAPPING"]}))
            _, _, train, histories, targets = corpus("beauty", cache_dir=tmp)
            self.assertEqual(train, [[1, 1]])
            self.assertEqual(histories, [[1, 1]])
            self.assertEqual(targets.tolist(), [2])
