import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import torch
from torch.nn import functional as F

from joint_training.contrastive import item_infonce, sample_negative_ids, select_positions
from joint_training.model import JointModel
from joint_training.normalization import LatentNormalizer


def make_config(*, contrastive=True, encoder_type="transformer"):
    return SimpleNamespace(
        data=SimpleNamespace(max_item_seq_len=4),
        ae=SimpleNamespace(latent_dim=3, architecture="linear", hidden_dims=(8,),
                           validate=lambda: None),
        contrastive=SimpleNamespace(enabled=contrastive, max_queries=32, temperature=0.2),
        recommender=SimpleNamespace(
            encoder_type=encoder_type, n_embd=8, n_layer=1, n_head=2, n_inner=16,
            activation_function="gelu", embd_pdrop=0., attn_pdrop=0.,
            layer_norm_epsilon=1e-6, hstu_hidden_dim=4, hstu_attention_dim=4,
            hstu_use_group_norm=True, hstu_output_dropout=0., hstu_attention_dropout=0.,
            hstu_attention_scaling="max_seq_len",
            diffloss_w=16, diffloss_d=1, num_sampling_steps=2,
            grad_checkpointing=False, rectified_flow_steps=10, ode_solver="euler",
            diffusion_batch_mul=1, diff_temperature=1., tiger_uncond_prob=0.,
            tiger_guidance_weight=0.,
        ),
        evaluation=SimpleNamespace(tiger_knn_metric="l2", temperature=1.,
                                   eval_sampling_mode="knn"),
    )


def make_batch():
    return {
        "input_ids": torch.tensor([[1, 2, 0, 0], [3, 1, 0, 0]]),
        "attention_mask": torch.tensor([[1, 1, 0, 0], [1, 1, 0, 0]]),
        "labels": torch.tensor([[2, 3, -100, -100], [1, 3, -100, -100]]),
        "seq_lens": torch.tensor([2, 2]),
    }


def make_model(**kwargs):
    torch.manual_seed(17)
    return JointModel(make_config(**kwargs), torch.randn(5, 6))


class NormalizationTest(unittest.TestCase):
    def test_layernorm_matches_functional(self):
        normalizer = LatentNormalizer(3)
        z = torch.tensor([[1., 3., 5.]])
        torch.testing.assert_close(normalizer(z), F.layer_norm(z, (3,), eps=1e-6))


class BridgeTest(unittest.TestCase):
    def test_repeated_target_ids_are_one_key_and_all_keys_receive_gradients(self):
        queries = torch.tensor([[1., .2], [.3, 1.], [.8, .5]], requires_grad=True)
        keys = torch.tensor([[1., .1], [.1, 1.]], requires_grad=True)
        targets = torch.tensor([7, 8, 7])
        loss = item_infonce(queries, keys, targets, torch.tensor([7, 8]), .2)
        expected = F.cross_entropy(F.normalize(queries, dim=1) @ F.normalize(keys, dim=1).T / .2,
                                   torch.tensor([0, 1, 0]))
        torch.testing.assert_close(loss, expected)
        loss.backward()
        self.assertTrue((keys.grad.abs().sum(1) > 0).all())

    def test_single_key_is_differentiable_zero(self):
        queries = torch.randn(3, 4, requires_grad=True)
        keys = torch.randn(1, 4, requires_grad=True)
        loss = item_infonce(queries, keys, torch.ones(3, dtype=torch.long), torch.tensor([1]), .2)
        self.assertEqual(loss.item(), 0.)
        loss.backward()
        self.assertIsNotNone(queries.grad)
        self.assertIsNotNone(keys.grad)

    def test_selection_respects_masks_and_caps_before_key_deduplication(self):
        batch = make_batch()
        batch["labels"][0, 2] = 4
        selected = select_positions(batch, max_queries=2)
        self.assertEqual(len(selected), 2)
        self.assertEqual(selected.available_queries, 4)
        self.assertTrue((selected.positions < 2).all())
        final = select_positions(make_batch())
        self.assertEqual(final.target_ids.tolist(), [2, 3, 1, 3])
        self.assertEqual(final.positions.tolist(), [0, 1, 0, 1])


class JointModelTest(unittest.TestCase):
    def test_online_encoding_deduplicates_nonpadding_ids(self):
        model = make_model()
        original = model.ae.latent
        model.ae.latent = Mock(side_effect=original)
        ids = torch.tensor([[1, 1, 0], [2, 0, 1]])
        result = model.encode_ids(ids)
        self.assertEqual(len(model.ae.latent.call_args.args[0]), 2)
        torch.testing.assert_close(result[0, 0], result[1, 2])
        self.assertTrue(torch.equal(result[ids == 0], torch.zeros(2, 3)))
        result.sum().backward()
        self.assertIsNotNone(model.ae.encoder[0].weight.grad)
        self.assertEqual(model.raw_features.device.type, "cpu")
        self.assertNotIn("raw_features", model.state_dict())
        self.assertNotIn("catalog", model.state_dict())
        with self.assertRaisesRegex(ValueError, "outside the catalog"):
            model.encode_ids(torch.tensor([model.n_items]))

    def test_ae_contrastive_step_crosses_frozen_recommender(self):
        model = make_model()
        model.eval()
        for parameter in model.rec_parameters():
            parameter.requires_grad_(False)
        selection = model.select_contrastive_positions(make_batch())
        loss, metadata = model.contrastive_loss(make_batch(), selection=selection)
        loss.backward()
        self.assertEqual(metadata["queries"], 4)
        self.assertEqual(metadata["unique_keys"], 3)
        self.assertGreater(model.ae.encoder[0].weight.grad.abs().sum().item(), 0.)
        self.assertTrue(all(p.grad is None for p in model.rec_parameters()))
        self.assertTrue(all(p.grad is None for p in model.ae.decoder.parameters()))

    def test_rf_loss_does_not_backpropagate_into_ae(self):
        model = make_model()
        model.train()
        with torch.no_grad():
            model.diffloss.net.final_layer.linear.weight.normal_(std=.1)
            # DiT-style initialization also zeroes the final AdaLN modulation.
            # Give the condition path a nonzero Jacobian before asserting that
            # the recommender side receives contrastive gradients.
            model.diffloss.net.final_layer.adaLN_modulation[-1].weight.normal_(std=.1)
        loss = model.rf_loss(make_batch())
        loss.backward()
        self.assertTrue(all(p.grad is None for p in model.ae.parameters()))
        self.assertGreater(model.input_proj.linear.weight.grad.abs().sum().item(), 0.)
        self.assertGreater(model.diffloss.net.final_layer.linear.weight.grad.abs().sum().item(), 0.)
        ae_ids = {id(p) for p in model.ae.parameters()}
        self.assertTrue(all(id(p) not in ae_ids for p in model.rec_parameters()))

    def test_negative_ids_are_unique_and_exclude_full_user_sequences(self):
        batch = make_batch()
        selection = select_positions(batch)
        rows = selection.rows
        negatives = sample_negative_ids(
            batch, rows, selection.target_ids, n_items=5, count=4, device=torch.device("cpu")
        )
        self.assertEqual(negatives.shape[0], len(rows))
        for index, row in enumerate(rows.tolist()):
            values = negatives[index]
            self.assertEqual(len(values), len(values.unique()))
            forbidden = torch.cat((
                batch["input_ids"][row][batch["attention_mask"][row].bool()],
                batch["labels"][row][batch["labels"][row] > 0],
            ))
            self.assertFalse(torch.isin(values, forbidden).any())

    def test_both_history_encoders_are_causal(self):
        for encoder in ("transformer", "hstu"):
            with self.subTest(encoder=encoder):
                model = make_model(encoder_type=encoder)
                model.eval()
                batch = make_batch()
                first, condition = model.encode_history(batch)
                batch["input_ids"][0, 1] = 5
                changed, _ = model.encode_history(batch)
                torch.testing.assert_close(first[:, 0], changed[:, 0])
                self.assertEqual(condition.shape[-1], 8)
                self.assertTrue(torch.equal(first[:, 2:], torch.zeros_like(first[:, 2:])))

    def test_disabled_contrastive_has_no_projector(self):
        model = make_model(contrastive=False)
        self.assertIsNone(model.contrastive_head)
        self.assertFalse(any(name.startswith("contrastive_head") for name, _ in model.named_parameters()))
        loss, metadata = model.contrastive_loss(make_batch())
        self.assertEqual(loss.item(), 0.)
        self.assertEqual(metadata["queries"], 0)

    def test_catalog_refresh_and_generation_use_current_space(self):
        model = make_model()
        model.eval()
        batch = make_batch()
        with self.assertRaisesRegex(RuntimeError, "Refresh"):
            model.generate(batch)
        catalog = model.refresh_catalog(batch_size=2)
        torch.testing.assert_close(catalog[1:], model.encode_ids(torch.arange(1, model.n_items)))
        self.assertTrue(torch.equal(catalog[0], torch.zeros(3)))
        model.diffloss.sample = Mock(return_value=catalog[[4, 2]])
        predictions = model.generate(batch, n_return_sequences=2)
        self.assertEqual(predictions.shape, (2, 2, 1))
        self.assertEqual(predictions[:, 0, 0].tolist(), [4, 2])
        model.invalidate_catalog()
        with self.assertRaisesRegex(RuntimeError, "Refresh"):
            model.generate(batch)
        model.refresh_catalog()
        model.load_state_dict(model.state_dict())
        self.assertEqual(len(model.catalog), 0)

    def test_refresh_preserves_individual_module_modes(self):
        model = make_model()
        model.train()
        model.ae.eval()
        model.refresh_catalog()
        self.assertTrue(model.training)
        self.assertTrue(model.encoder.training)
        self.assertFalse(model.ae.training)


if __name__ == "__main__":
    unittest.main()
