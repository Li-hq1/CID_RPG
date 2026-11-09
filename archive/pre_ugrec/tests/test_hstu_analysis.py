import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from models.tigerdiff import TigerDiff
from tests.test_hstu import hstu_config
from tests.test_tigerdiff import DummyDataset, DummyTokenizer, make_batch
from vis_analysis.analysis_tigerdiff import (
    build_model_for_analysis, extract_attention_weights, plot_attention_heatmap,
)


class AttentionAnalysisTest(unittest.TestCase):
    def test_extraction_matches_forward_for_both_encoders(self):
        for kind in ("hstu", "transformer"):
            model = TigerDiff(hstu_config(encoder_type=kind), DummyDataset(), DummyTokenizer()).eval()
            with torch.no_grad():
                encoded, expected = model.encoder(model._history_inputs(make_batch()),
                                                  make_batch()["attention_mask"], return_attentions=True)
                plain, _ = model._encode_history(make_batch())
            torch.testing.assert_close(encoded, plain, rtol=0, atol=0)
            extracted = extract_attention_weights(model, make_batch(), "cpu")
            self.assertEqual(len(extracted), 2)
            for actual, reference in zip(extracted, expected):
                torch.testing.assert_close(actual, reference, rtol=0, atol=0)
                self.assertTrue((actual.triu(1) == 0).all())
                self.assertTrue((actual[:, :, :, 2:] == 0).all())
                self.assertFalse(actual.requires_grad)

    def test_hstu_plot_preserves_negative_weights(self):
        weights = [torch.tensor([[[[0.2, 0.0], [-0.1, 0.3]]]])]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "hstu.pdf"
            from matplotlib.axes import Axes
            real_imshow = Axes.imshow
            with patch.object(Axes, "imshow", autospec=True) as imshow:
                # A real plot also validates that the signed data can be exported.
                imshow.side_effect = real_imshow
                with contextlib.redirect_stdout(io.StringIO()):
                    plot_attention_heatmap(weights, 2, ["A", "B"], str(path), encoder_type="hstu")
                self.assertTrue(path.is_file())
                self.assertTrue(path.with_name("hstu_averaged.pdf").is_file())
                for call in imshow.call_args_list:
                    self.assertEqual(call.kwargs["cmap"], "RdBu_r")
                    self.assertLess(call.kwargs["vmin"], 0)
                    self.assertGreater(call.kwargs["vmax"], 0)

    def test_analysis_uses_saved_config_and_identifies_legacy_transformer(self):
        for kind in ("hstu", "transformer"):
            config = hstu_config(encoder_type=kind, hstu_attention_scaling="max_seq_len")
            original = TigerDiff(config, DummyDataset(), DummyTokenizer())
            state = original.state_dict()
            payloads = [{"model": state, "format_version": 1, "config": config}]
            if kind == "transformer":
                legacy_config = dict(config)
                del legacy_config["encoder_type"]
                payloads += [state, {"model": state, "format_version": 1, "config": legacy_config}]
            for payload in payloads:
                with self.subTest(kind=kind, keys=list(payload)[:2]):
                    with tempfile.TemporaryDirectory() as directory:
                        path = Path(directory) / "latest.pth"
                        torch.save(payload, path)
                        with patch("vis_analysis.analysis_tigerdiff.AmazonReviews2014"), \
                             patch("vis_analysis.analysis_tigerdiff.TigerDiffTokenizer", return_value=DummyTokenizer()), \
                             patch("vis_analysis.analysis_tigerdiff.init_logger"), \
                             contextlib.redirect_stdout(io.StringIO()):
                            overrides = None if "config" in payload else {
                                key: value for key, value in config.items() if key != "encoder_type"
                            }
                            model, _, _, loaded = build_model_for_analysis("Beauty", str(path), overrides)
                        self.assertEqual(model.encoder_type, kind)
                        if kind == "hstu":
                            self.assertEqual(loaded["hstu_attention_scaling"], "max_seq_len")
                        for key, value in state.items():
                            torch.testing.assert_close(value, model.state_dict()[key], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
