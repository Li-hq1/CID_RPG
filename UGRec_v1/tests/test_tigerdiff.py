import unittest
from unittest.mock import Mock

import numpy as np
import torch

from models.tigerdiff import TigerDiff, TigerSequenceEncoder
from data.tigerdiff_data import TigerDiffTokenizer


class DummyDataset:
    n_items = 4


class DummyTokenizer:
    ignored_label = -100
    eos_token = None
    max_token_seq_len = 4

    def __init__(self):
        self.sent_embs = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        self.item2tokens = self.sent_embs


def make_config():
    return {
        "n_embd": 8,
        "n_layer": 1,
        "n_head": 2,
        "n_inner": 16,
        "activation_function": "gelu_new",
        "embd_pdrop": 0.0,
        "layer_norm_epsilon": 1e-12,
        "temperature": 1.0,
        "tiger_knn_metric": "inner_product",
        "diffusion_batch_mul": 1,
        "diff_temperature": 1.0,
        "tiger_uncond_prob": 0.0,
        "tiger_guidance_weight": 0.0,
        "diffloss_w": 16,
        "diffloss_d": 1,
        "num_sampling_steps": 2,
        "grad_checkpointing": False,
        "use_rectified_flow": False,
        "rectified_flow_steps": 10,
        "ode_solver": "euler",
    }


def make_batch():
    return {
        "input_ids": torch.tensor([[1, 2, 0, 0]]),
        "attention_mask": torch.tensor([[1, 1, 0, 0]]),
        "labels": torch.tensor([[2, 3, -100, -100]]),
        "seq_lens": torch.tensor([2]),
    }


class TigerDiffTest(unittest.TestCase):
    def test_single_process_tokenization_does_not_fork(self):
        tokenizer = TigerDiffTokenizer.__new__(TigerDiffTokenizer)
        dataset = Mock(column_names=['item_seq'])
        for num_proc, expected in [(1, None), (2, 2)]:
            tokenizer.config = {'num_proc': num_proc}
            tokenizer._tokenize_split(dataset, 'train')
            self.assertEqual(dataset.map.call_args.kwargs['num_proc'], expected)

    def test_uses_local_encoder_and_learned_position_embeddings(self):
        torch.manual_seed(0)
        model = TigerDiff(make_config(), DummyDataset(), DummyTokenizer())
        model.eval()

        self.assertIsInstance(model.encoder, TigerSequenceEncoder)
        self.assertIsInstance(model.position_emb, torch.nn.Embedding)

        batch = make_batch()
        batch["input_ids"] = torch.tensor([[1, 1, 0, 0]])
        with torch.no_grad():
            encoded, _ = model._encode_history(batch)

        self.assertFalse(torch.allclose(encoded[:, 0], encoded[:, 1]))

    def test_forward_uses_last_valid_label_for_diffusion_target(self):
        torch.manual_seed(0)
        model = TigerDiff(make_config(), DummyDataset(), DummyTokenizer())
        model.train()

        captured = {}

        def fake_forward(z, target, mask=None):
            captured["z_shape"] = z.shape
            captured["target"] = target.detach().clone()
            return z.sum() * 0.0 + target.sum() * 0.0 + 1.5

        model.diffloss.forward = fake_forward

        outputs = model(make_batch())

        self.assertAlmostEqual(outputs.loss.item(), 1.5, places=6)
        self.assertEqual(captured["z_shape"], (1, 8))
        self.assertTrue(
            torch.equal(captured["target"], model.item_id2embs[torch.tensor([3])])
        )

    def test_all_positions_uses_causal_attention_and_all_targets(self):
        config = make_config()
        config.update(supervision_mode="all_positions", attn_pdrop=0.0, n_layer=2)
        torch.manual_seed(0)
        model = TigerDiff(config, DummyDataset(), DummyTokenizer())
        model.train()

        captured = {}

        def fake_forward(z, target, mask=None):
            captured["z"] = z.detach().clone()
            captured["target"] = target.detach().clone()
            return z.sum() * 0.0 + target.sum() * 0.0

        model.diffloss.forward = fake_forward
        outputs = model(make_batch())

        self.assertEqual(captured["z"].shape, (2, 8))
        self.assertTrue(torch.equal(captured["target"], model.item_id2embs[torch.tensor([2, 3])]))
        with torch.no_grad():
            encoded, _ = model._encode_history(make_batch())
            expected_z = model.condition_proj(encoded[:, :2]).reshape(2, 8)
        torch.testing.assert_close(captured["z"], expected_z)

        future_changed = make_batch()
        future_changed["input_ids"] = torch.tensor([[1, 3, 0, 0]])
        with torch.no_grad():
            first, _ = model._encode_history(make_batch())
            changed, _ = model._encode_history(future_changed)
        torch.testing.assert_close(first[:, 0], changed[:, 0])
        self.assertFalse(torch.allclose(first[:, 1], changed[:, 1]))
        self.assertTrue(torch.equal(first[:, 2:], torch.zeros_like(first[:, 2:])))

        # A position in the full history must match encoding its prefix alone.
        prefix = {key: value[:, :1] if value.dim() == 2 else value
                  for key, value in make_batch().items()}
        prefix["seq_lens"] = torch.tensor([1])
        with torch.no_grad():
            prefix_encoded, _ = model._encode_history(prefix)
        torch.testing.assert_close(first[:, :1], prefix_encoded)

    def test_all_positions_skips_ignored_and_padded_labels(self):
        config = make_config()
        config["supervision_mode"] = "all_positions"
        model = TigerDiff(config, DummyDataset(), DummyTokenizer())
        model.diffloss.forward = Mock(return_value=torch.tensor(1.0))
        batch = make_batch()
        batch["labels"] = torch.tensor([[-100, 3, 1, -100]])
        model(batch)
        call = model.diffloss.forward.call_args.kwargs
        self.assertEqual(call["z"].shape, (1, 8))
        torch.testing.assert_close(call["target"], model.item_id2embs[[3]])

        batch["labels"] = torch.full_like(batch["labels"], -100)
        model.diffloss.forward.reset_mock()
        outputs = model(batch)
        self.assertEqual(outputs.loss.item(), 0.0)
        outputs.loss.backward()
        model.diffloss.forward.assert_not_called()

    def test_all_positions_accepts_final_only_labels_and_generates(self):
        config = make_config()
        config["supervision_mode"] = "all_positions"
        model = TigerDiff(config, DummyDataset(), DummyTokenizer())
        model.eval()
        model.diffloss.forward = Mock(return_value=torch.tensor(1.0))
        batch = make_batch()
        batch["labels"] = torch.tensor([[3]])
        model(batch)
        call = model.diffloss.forward.call_args.kwargs
        torch.testing.assert_close(call["target"], model.item_id2embs[[3]])
        with torch.no_grad():
            _, condition = model._encode_history(batch)
        torch.testing.assert_close(call["z"], condition)
        model.diffloss.sample = Mock(return_value=model.item_id2embs[[3]])
        self.assertEqual(model.generate(batch, n_return_sequences=1).tolist(), [[[3]]])
        self.assertEqual(model.diffloss.sample.call_args.args[0].shape, (1, 8))

    def test_all_positions_backpropagates_with_both_diffusion_losses(self):
        for rectified_flow in (False, True):
            with self.subTest(rectified_flow=rectified_flow):
                config = make_config()
                config.update(supervision_mode="all_positions", use_rectified_flow=rectified_flow,
                              tiger_uncond_prob=1.0, diffusion_batch_mul=2)
                model = TigerDiff(config, DummyDataset(), DummyTokenizer())
                loss = model(make_batch()).loss
                self.assertTrue(torch.isfinite(loss))
                loss.backward()
                self.assertIsNotNone(model.uncond_condition.grad)
                self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters()
                                    if p.grad is not None))

    def test_supervision_modes_keep_checkpoint_keys_compatible(self):
        old = TigerDiff(make_config(), DummyDataset(), DummyTokenizer())
        config = make_config()
        config["supervision_mode"] = "all_positions"
        new = TigerDiff(config, DummyDataset(), DummyTokenizer())
        new.load_state_dict(old.state_dict(), strict=True)
        self.assertTrue(old.encoder.layers[0].self_attn.causal)
        self.assertTrue(new.encoder.layers[0].self_attn.causal)

    def test_generate_samples_and_returns_item_ids(self):
        torch.manual_seed(0)
        model = TigerDiff(make_config(), DummyDataset(), DummyTokenizer())
        model.eval()
        model.diffloss.sample = Mock(
            return_value=torch.tensor([[0.2, 0.9, 0.1]], dtype=torch.float32)
        )

        preds = model.generate(make_batch(), n_return_sequences=2)

        self.assertEqual(preds.shape, (1, 2, 1))
        self.assertEqual(preds.squeeze(-1).tolist(), [[2, 1]])
        model.diffloss.sample.assert_called_once()

    def test_generate_uses_classifier_free_guidance_pair_when_weighted(self):
        config = make_config()
        config["tiger_guidance_weight"] = 2.0
        model = TigerDiff(config, DummyDataset(), DummyTokenizer())
        model.eval()

        def fake_sample(z, temperature=1.0, cfg=1.0):
            self.assertEqual(z.shape, (2, 8))
            self.assertEqual(cfg, 3.0)
            return torch.tensor(
                [
                    [0.0, 0.1, 0.8],
                    [0.7, 0.2, 0.1],
                ],
                dtype=torch.float32,
            )

        model.diffloss.sample = Mock(side_effect=fake_sample)

        preds = model.generate(make_batch(), n_return_sequences=1)

        self.assertEqual(preds.squeeze(-1).tolist(), [[3]])


if __name__ == "__main__":
    unittest.main()
