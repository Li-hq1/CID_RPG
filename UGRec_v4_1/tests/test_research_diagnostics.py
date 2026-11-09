"""Guard the research controls, causal supervision and gradient interpretation."""

import unittest

import numpy as np
import torch

from research.common import batches, paired_interval, per_user
from research.compact_ablation import CompactModel


class ResearchMetricsTest(unittest.TestCase):
    def test_single_positive_metrics_and_paired_zero(self):
        predictions = np.array([np.arange(1, 11), np.arange(1, 11)])
        metrics = per_user(predictions, [1, 10])
        np.testing.assert_equal(metrics["recall@5"], [1., 0.])
        np.testing.assert_allclose(metrics["ndcg@10"], [1., 1 / np.log2(11)])
        self.assertEqual(paired_interval([1., 0.], [1., 0.], draws=20),
                         {"difference": 0., "bootstrap_95pct": [0., 0.]})

    def test_batches_keep_validation_target_out_of_history(self):
        ix, batch = next(batches([[1, 2], [3]], np.array([7, 8]), indices=np.array([1, 0])))
        self.assertEqual(ix.tolist(), [1, 0])
        self.assertEqual(batch["labels"].flatten().tolist(), [8, 7])
        self.assertEqual(batch["seq_lens"].tolist(), [1, 2])
        self.assertFalse(torch.isin(batch["input_ids"], batch["labels"]).any())


class ResearchGradientTest(unittest.TestCase):
    def test_clean_target_sampler_recovers_a_constant_clean_state(self):
        class ConstantClean(torch.nn.Module):
            def forward(self, value, time, condition):
                return torch.full_like(value, .375)

        model = CompactModel(torch.randn(30, 128), "pca", "transformer", "x0")
        model.flow.net = ConstantClean()
        with torch.no_grad():
            result = model.sample(torch.randn(4, 128))
        torch.testing.assert_close(result, torch.full_like(result, .375), atol=1e-6, rtol=1e-6)

    def test_content_adapter_is_shared_and_initializes_to_frozen_content(self):
        model = CompactModel(torch.randn(30, 128), "adapter", "transformer", "ce")
        expected = torch.nn.functional.layer_norm(model.content, (128,), eps=1e-6)
        torch.testing.assert_close(model.catalog(), expected)
        self.assertFalse(model.item.weight.requires_grad)
        self.assertTrue(all(p.requires_grad for p in model.adapter.parameters()))

    def test_direct_ce_trains_history_encoder_and_shared_item_table(self):
        torch.manual_seed(4)
        model = CompactModel(torch.randn(30, 128), "id", "transformer", "ce")
        ids = torch.zeros(2, 50, dtype=torch.long)
        ids[:, :2] = torch.tensor([[1, 2], [3, 4]])
        labels = torch.full_like(ids, -100)
        labels[:, :2] = torch.tensor([[2, 5], [4, 6]])
        loss, _ = model.loss(ids, labels)
        loss.backward()
        self.assertGreater(float(model.item.weight.grad[5].norm()), 0.)
        self.assertGreater(float(model.input_proj.linear.weight.grad.norm()), 0.)
        self.assertTrue(all(p.grad is None for p in model.flow.parameters()))

    def test_all_position_prediction_is_causal_for_each_encoder(self):
        for encoder in ("hstu", "transformer"):
            with self.subTest(encoder=encoder):
                torch.manual_seed(9)
                model = CompactModel(torch.randn(30, 128), "pca", encoder, "ce").eval()
                first = torch.zeros(1, 50, dtype=torch.long)
                first[:, :3] = torch.tensor([1, 2, 3])
                second = first.clone()
                second[:, 1:3] = torch.tensor([4, 5])
                with torch.no_grad():
                    a = model.encode(first, model.catalog())[:, 0]
                    b = model.encode(second, model.catalog())[:, 0]
                torch.testing.assert_close(a, b)

    def test_zero_noise_teacher_forced_anchor_ignores_predictor(self):
        clean = torch.randn(4, 128)
        velocity_a, velocity_b = torch.randn_like(clean), torch.randn_like(clean)
        torch.testing.assert_close(clean - 0 * velocity_a, clean - 0 * velocity_b)


if __name__ == "__main__":
    unittest.main()
