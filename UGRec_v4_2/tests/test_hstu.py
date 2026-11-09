import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import torch

from config import CONFIG
from engine.checkpoint import load_checkpoint, save_checkpoint
from misc.config import parse_config, validate_config
from models.hstu import HSTUSequenceEncoder, SimpleRotaryEmbedding, apply_rotary_pos_emb, reverse_position_ids
from models.tigerdiff import TigerDiff
from tests.test_tigerdiff import DummyDataset, DummyTokenizer, make_batch, make_config


def hstu_config(**overrides):
    config = make_config()
    config.update(encoder_type="hstu", hstu_hidden_dim=4, hstu_attention_dim=4,
                  hstu_output_dropout=0.0, hstu_attention_dropout=0.0, n_layer=2)
    config.update(overrides)
    return config


def encoder(scaling="input_length", group_norm=True):
    return HSTUSequenceEncoder(8, 2, 4, 4, 2, 4, use_group_norm=group_norm,
                               output_dropout_ratio=0.0, attention_scaling=scaling)


class HSTUTest(unittest.TestCase):
    def test_rope_preserves_norms_and_shared_position_shift_dot_products(self):
        torch.manual_seed(19)
        q, k = torch.randn(2, 2, 3, 8), torch.randn(2, 2, 3, 8)
        positions = torch.tensor([[3, 2, 1], [2, 1, 0]])
        rope = SimpleRotaryEmbedding(8)
        rotated_q, rotated_k = apply_rotary_pos_emb(q, k, *rope(q, positions))
        shifted_q, shifted_k = apply_rotary_pos_emb(q, k, *rope(q, positions + 4))
        torch.testing.assert_close(rotated_q.norm(dim=-1), q.norm(dim=-1))
        torch.testing.assert_close(rotated_k.norm(dim=-1), k.norm(dim=-1))
        torch.testing.assert_close(rotated_q @ rotated_k.transpose(-2, -1),
                                   shifted_q @ shifted_k.transpose(-2, -1), atol=2e-6, rtol=2e-5)

    def test_reverse_positions_mixed_lengths_and_fully_padded_history(self):
        mask = torch.tensor([[1, 1, 1, 0], [1, 0, 0, 0], [0, 0, 0, 0]])
        self.assertEqual(reverse_position_ids(mask).tolist(), [[3, 2, 1, 0], [1, 0, 0, 0], [0, 0, 0, 0]])
        net = encoder()
        # Masking must still hold after normalization/projection biases are learned.
        with torch.no_grad():
            for layer in net.layers:
                layer.uvqk_bias.fill_(0.25)
                layer.output_norm_bias.fill_(0.5)
        x = torch.randn(3, 4, 8)
        out, weights = net(x, mask, return_attentions=True)
        self.assertTrue(torch.isfinite(out).all())
        self.assertEqual(out.shape, (3, 4, 8))
        torch.testing.assert_close(out[~mask.bool()], torch.zeros_like(out[~mask.bool()]))
        for weight in weights:
            self.assertTrue((weight[2] == 0).all())
            self.assertTrue((weight[0, :, 3, :] == 0).all())
            self.assertTrue((weight[0, :, :, 3] == 0).all())
        single = net(x[:1], mask[:1])
        torch.testing.assert_close(single, out[:1])

    def test_causal_and_padding_isolation_for_both_encoders_and_supervision_modes(self):
        for kind in ("transformer", "hstu"):
            for mode in ("last_item", "all_positions"):
                with self.subTest(encoder=kind, mode=mode):
                    config = hstu_config(encoder_type=kind, supervision_mode=mode, attn_pdrop=0.0)
                    model = TigerDiff(config, DummyDataset(), DummyTokenizer()).eval()
                    batch = make_batch()
                    future = make_batch()
                    future["input_ids"] = torch.tensor([[1, 3, 0, 0]])
                    padded = make_batch()
                    padded["input_ids"] = torch.tensor([[1, 2, 3, 1]])
                    with torch.no_grad():
                        out = model(batch, return_loss=False)
                        future_out = model(future, return_loss=False)
                        padded_out = model(padded, return_loss=False)
                    self.assertEqual(out.last_hidden_state.shape, (1, 4, 8))
                    self.assertEqual(out.final_states.shape, (1, 8))
                    torch.testing.assert_close(out.last_hidden_state[:, 0], future_out.last_hidden_state[:, 0])
                    torch.testing.assert_close(out.last_hidden_state, padded_out.last_hidden_state)
                    self.assertTrue((out.last_hidden_state[:, 2:] == 0).all())
                    self.assertEqual(hasattr(model, "position_emb"), kind == "transformer")

    def test_max_length_scaling_preserves_causal_prefix(self):
        torch.manual_seed(9)
        for group_norm in (True, False):
            net = encoder("max_seq_len", group_norm).eval()
            x = torch.randn(2, 4, 8)
            mask = torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]])
            full = net(x, mask)
            prefix = net(x[:, :2], mask[:, :2])
            torch.testing.assert_close(full[:, :2], prefix, atol=2e-6, rtol=2e-5)

    def test_input_length_scales_weights_and_full_length_modes_agree(self):
        torch.manual_seed(8)
        dynamic = encoder("input_length").eval()
        fixed = encoder("max_seq_len").eval()
        fixed.load_state_dict(dynamic.state_dict())
        x = torch.randn(1, 4, 8)
        mask = torch.tensor([[1, 1, 0, 0]])
        full, full_weights = dynamic(x, mask, return_attentions=True)
        fixed_full, _ = fixed(x, mask, return_attentions=True)
        torch.testing.assert_close(full, fixed_full, rtol=0, atol=0)
        cropped, cropped_weights = dynamic(x[:, :2], mask[:, :2], return_attentions=True)
        torch.testing.assert_close(cropped_weights[0], full_weights[0][:, :, :2, :2] * 2)
        self.assertFalse(torch.allclose(full[:, :2], cropped))
        plain = dynamic(x, mask)
        torch.testing.assert_close(plain, full, rtol=0, atol=0)

    def test_losses_supervision_labels_and_generation(self):
        for mode in ("last_item", "all_positions"):
            for rectified in (False, True):
                with self.subTest(mode=mode, rectified=rectified):
                    model = TigerDiff(hstu_config(supervision_mode=mode, use_rectified_flow=rectified),
                                      DummyDataset(), DummyTokenizer())
                    output = model(make_batch())
                    self.assertTrue(torch.isfinite(output.loss))
                    output.loss.backward()
                    self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                                        for p in model.encoder.parameters()))
                    model.eval()
                    with torch.no_grad():
                        predictions = model.generate(make_batch(), n_return_sequences=2)
                    self.assertEqual(predictions.shape, (1, 2, 1))
                    self.assertTrue(((predictions >= 1) & (predictions <= 3)).all())
                    model.diffloss.forward = Mock(return_value=torch.tensor(1.0))
                    batch = make_batch()
                    model(batch)
                    expected = [3] if mode == "last_item" else [2, 3]
                    torch.testing.assert_close(model.diffloss.forward.call_args.kwargs["target"],
                                               model.item_id2embs[expected])
                    batch["labels"] = torch.tensor([[3]])
                    model(batch)
                    self.assertEqual(model.diffloss.forward.call_args.kwargs["z"].shape, (1, 8))
                    if mode == "all_positions":
                        batch["labels"] = torch.tensor([[-100, 3, 1, -100]])
                        model(batch)
                        torch.testing.assert_close(model.diffloss.forward.call_args.kwargs["target"],
                                                   model.item_id2embs[[3]])
                    batch["labels"] = torch.full((1, 4), -100)
                    model.diffloss.forward.reset_mock()
                    self.assertEqual(model(batch).loss.item(), 0.0)
                    model.diffloss.forward.assert_not_called()

    def test_same_architecture_checkpoint_round_trip(self):
        accelerator = SimpleNamespace(is_main_process=True, unwrap_model=lambda model: model)
        for kind in ("hstu", "transformer"):
            config = hstu_config(encoder_type=kind)
            original = TigerDiff(config, DummyDataset(), DummyTokenizer()).eval()
            restored = TigerDiff(config, DummyDataset(), DummyTokenizer()).eval()
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "latest.pth"
                save_checkpoint(original, path, accelerator, config=config)
                payload = load_checkpoint(restored, path, "cpu")
                self.assertEqual(payload["config"]["encoder_type"], kind)
                with torch.no_grad():
                    torch.testing.assert_close(original(make_batch(), return_loss=False).last_hidden_state,
                                               restored(make_batch(), return_loss=False).last_hidden_state,
                                               rtol=0, atol=0)
                other = TigerDiff(hstu_config(encoder_type="transformer" if kind == "hstu" else "hstu"),
                                  DummyDataset(), DummyTokenizer())
                with self.assertRaises(RuntimeError):
                    load_checkpoint(other, path, "cpu")

    def test_config_defaults_overrides_validation_and_legacy_fallback(self):
        config, _ = parse_config([])
        self.assertEqual(config["encoder_type"], "hstu")
        self.assertEqual((config["hstu_hidden_dim"], config["hstu_attention_dim"]), (64, 64))
        self.assertEqual(config["hstu_attention_scaling"], "input_length")
        config, _ = parse_config(["--encoder_type=transformer", "--hstu_hidden_dim=32",
                                  "--hstu_attention_dim=16", "--hstu_use_group_norm=false",
                                  "--hstu_output_dropout=0.2", "--hstu_attention_dropout=0.1",
                                  "--hstu_attention_scaling=max_seq_len"])
        self.assertEqual(config["encoder_type"], "transformer")
        self.assertEqual(config["hstu_hidden_dim"], 32)
        self.assertEqual(config["hstu_attention_dim"], 16)
        self.assertFalse(config["hstu_use_group_norm"])
        self.assertEqual(config["hstu_output_dropout"], 0.2)
        self.assertEqual(config["hstu_attention_dropout"], 0.1)
        self.assertEqual(config["hstu_attention_scaling"], "max_seq_len")
        for args in (["--hstu_hidden_dim=0"], ["--hstu_attention_dim=3"],
                     ["--hstu_output_dropout=1"], ["--hstu_attention_dropout=-0.1"],
                     ["--hstu_attention_dropout=nan"]):
            with self.subTest(args=args), self.assertRaises(ValueError):
                parse_config(args)
        legacy = deepcopy(CONFIG)
        del legacy["encoder_type"]
        validate_config(legacy)
        model = TigerDiff(make_config(), DummyDataset(), DummyTokenizer())
        self.assertEqual(model.encoder_type, "transformer")
        # HSTU widths are independent of embedding_dim / num_heads.
        legacy.update(encoder_type="hstu", n_embd=7)
        validate_config(legacy)
        net = HSTUSequenceEncoder(7, 2, 4, 4, 1, 4)
        self.assertEqual(net(torch.randn(1, 2, 7), torch.ones(1, 2)).shape, (1, 2, 7))


if __name__ == "__main__":
    unittest.main()
