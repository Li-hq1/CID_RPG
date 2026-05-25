import unittest
from unittest.mock import Mock

import numpy as np
import torch

from genrec.models import TigerDiff
from genrec.models.TigerDiff.model import TigerSequenceEncoder


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
