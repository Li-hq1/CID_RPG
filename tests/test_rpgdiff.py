import unittest
from unittest.mock import Mock

import torch

from genrec.models import RPGDiff


class DummyDataset:
    def __init__(self):
        self.n_items = 4
        self.item2id = {
            'item-1': 1,
            'item-2': 2,
            'item-3': 3,
        }


class DummyTokenizer:
    ignored_label = -100
    eos_token = 9
    max_token_seq_len = 4
    vocab_size = 10
    n_digit = 2
    codebook_size = 4

    def __init__(self):
        self.item2tokens = {
            'item-1': (1, 5),
            'item-2': (2, 6),
            'item-3': (3, 7),
        }


def make_config():
    return {
        'device': 'cpu',
        'n_embd': 8,
        'n_layer': 1,
        'n_head': 2,
        'n_inner': 16,
        'activation_function': 'gelu_new',
        'resid_pdrop': 0.0,
        'embd_pdrop': 0.0,
        'attn_pdrop': 0.0,
        'layer_norm_epsilon': 1e-12,
        'initializer_range': 0.02,
        'temperature': 0.07,
        'chunk_size': 2,
        'num_beams': 2,
        'n_edges': 2,
        'propagation_steps': 1,
        'codebook_size': 4,
        'diff_alpha': 0.1,
        'diff_warmup_steps': 2,
        'diffusion_batch_mul': 1,
        'diffloss_w': 16,
        'diffloss_d': 1,
        'num_sampling_steps': 2,
        'grad_checkpointing': False,
        'use_rectified_flow': False,
        'rectified_flow_steps': 10,
        'ode_solver': 'euler',
    }


def make_batch():
    return {
        'input_ids': torch.tensor([[1, 2, 0, 0]]),
        'attention_mask': torch.tensor([[1, 1, 0, 0]]),
        'labels': torch.tensor([[2, 3, -100, -100]]),
        'seq_lens': torch.tensor([2]),
    }


class RPGDiffTest(unittest.TestCase):
    def test_loss_uses_warmed_diffusion_regularizer(self):
        torch.manual_seed(0)
        model = RPGDiff(make_config(), DummyDataset(), DummyTokenizer())
        model.train()
        batch = make_batch()

        first = model(batch)
        self.assertTrue(torch.allclose(first.loss, first.mtp_loss))
        self.assertEqual(first.diff_alpha.item(), 0.0)

        second = model(batch)
        self.assertAlmostEqual(second.diff_alpha.item(), 0.05, places=6)
        self.assertTrue(
            torch.allclose(
                second.loss,
                second.mtp_loss + second.diff_alpha * second.diff_loss,
            )
        )

        third = model(batch)
        self.assertAlmostEqual(third.diff_alpha.item(), 0.1, places=6)

    def test_generate_keeps_rpg_discrete_path(self):
        torch.manual_seed(0)
        model = RPGDiff(make_config(), DummyDataset(), DummyTokenizer())
        model.eval()
        model.diffloss.sample = Mock(side_effect=AssertionError('DiffLoss sampling used'))

        preds = model.generate(make_batch(), n_return_sequences=2)

        self.assertEqual(preds.shape, (1, 2, 1))
        model.diffloss.sample.assert_not_called()


if __name__ == '__main__':
    unittest.main()
