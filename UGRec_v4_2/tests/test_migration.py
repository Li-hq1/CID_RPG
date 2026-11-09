"""Numerical regression against the preserved pre-refactor research code."""
import ast
from pathlib import Path
import unittest

import torch

from models.tigerdiff import TigerDiff
from tests.test_tigerdiff import DummyDataset, DummyTokenizer, make_batch, make_config


class LegacyBase(torch.nn.Module):
    def __init__(self, config, dataset, tokenizer):
        super().__init__()
        self.config, self.dataset, self.tokenizer = config, dataset, tokenizer


def legacy_model():
    source = Path(__file__).resolve().parents[1] / "tests/reference/tigerdiff.py"
    tree = ast.parse(source.read_text())
    tree.body = [
        node for node in tree.body
        if not (isinstance(node, ast.ImportFrom)
                and node.module in {"genrec.dataset", "genrec.model", "genrec.tokenizer"})
    ]
    namespace = {"AbstractModel": LegacyBase, "AbstractDataset": object, "AbstractTokenizer": object}
    exec(compile(tree, str(source), "exec"), namespace)
    return namespace["TigerDiff"]


class MigrationTest(unittest.TestCase):
    def test_checkpoint_keys_loss_gradients_and_sampling_are_unchanged(self):
        original = legacy_model()
        for rectified_flow in (False, True):
            with self.subTest(rectified_flow=rectified_flow):
                config = make_config()
                config.update(use_rectified_flow=rectified_flow, tiger_guidance_weight=2.0,
                              tiger_uncond_prob=0.5, attn_pdrop=0.0)
                torch.manual_seed(7)
                old = original(config, DummyDataset(), DummyTokenizer())
                torch.manual_seed(7)
                new = TigerDiff(config, DummyDataset(), DummyTokenizer())
                self.assertEqual(list(old.state_dict()), list(new.state_dict()))
                for name, tensor in old.state_dict().items():
                    torch.testing.assert_close(tensor, new.state_dict()[name], rtol=0, atol=0)
                losses = []
                # With one valid item, causal and legacy bidirectional masks agree.
                # This isolates checkpoint/downstream compatibility from the intentional mask change.
                batch = make_batch()
                batch["input_ids"] = torch.tensor([[1, 0, 0, 0]])
                batch["attention_mask"] = torch.tensor([[1, 0, 0, 0]])
                batch["seq_lens"] = torch.tensor([1])
                batch["labels"] = torch.tensor([[3, -100, -100, -100]])
                for model in (old, new):
                    torch.manual_seed(9)
                    loss = model(batch).loss
                    losses.append(loss.detach())
                    loss.backward()
                torch.testing.assert_close(losses[0], losses[1], rtol=0, atol=0)
                torch.testing.assert_close(old(batch, return_loss=False).final_states,
                                           new(batch, return_loss=False).final_states, rtol=0, atol=0)
                for old_param, new_param in zip(old.parameters(), new.parameters()):
                    if old_param.grad is not None:
                        torch.testing.assert_close(old_param.grad, new_param.grad, rtol=0, atol=0)
                old.eval()
                new.eval()
                with torch.no_grad():
                    torch.manual_seed(11)
                    old_predictions = old.generate(batch, n_return_sequences=2)
                    torch.manual_seed(11)
                    new_predictions = new.generate(batch, n_return_sequences=2)
                torch.testing.assert_close(old_predictions, new_predictions, rtol=0, atol=0)
