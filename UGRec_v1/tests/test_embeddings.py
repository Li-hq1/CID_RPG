from copy import deepcopy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch

from config import CONFIG
from data.embeddings import encode_sentences, read_sentence_embeddings, sentence_embedding_cache_path
from data.tigerdiff_data import TigerDiffTokenizer
from misc.config import parse_config


class EmbeddingTest(unittest.TestCase):
    def test_configured_qwen_directory_loads_offline_on_cpu_and_preserves_order(self):
        config = deepcopy(CONFIG)
        config.update(device="cpu", sent_emb_dim=2)
        model = Mock()
        model.get_embedding_dimension.return_value = 2
        model.encode.return_value = np.array([[1., 2.], [3., 4.]], dtype=np.float16)
        model.__getitem__ = Mock(return_value=Mock())
        with tempfile.TemporaryDirectory() as directory:
            config["sent_emb_model_paths"][config["sent_emb_model"]] = directory
            (Path(directory) / "config.json").write_text('{"model_type": "qwen3"}')
            with patch("data.embeddings.SentenceTransformer", return_value=model) as factory:
                result = encode_sentences(["first item", "second item"], config)
            self.assertEqual(factory.call_args.args[0], directory)
            self.assertTrue(factory.call_args.kwargs["local_files_only"])
            self.assertEqual(factory.call_args.kwargs["model_kwargs"]["dtype"], torch.float32)
        self.assertEqual(result.dtype, np.float32)
        np.testing.assert_allclose(np.linalg.norm(result, axis=1), 1., atol=1e-6)
        self.assertEqual(model.max_seq_length, 512)
        self.assertEqual(model.tokenizer.padding_side, "left")
        self.assertEqual(model.encode.call_args.args[0], ["first item", "second item"])
        self.assertEqual(model.encode.call_args.kwargs["prompt"], "")
        self.assertTrue(model.encode.call_args.kwargs["normalize_embeddings"])

    def test_model_dimension_mismatch_fails_before_encoding(self):
        model = Mock()
        model.get_embedding_dimension.return_value = 768
        model.__getitem__ = Mock(return_value=Mock())
        config = {**CONFIG, "device": "cpu"}
        with patch("data.embeddings.SentenceTransformer", return_value=model):
            with self.assertRaisesRegex(ValueError, "Encoder produces 768"):
                encode_sentences(["item"], config)
        model.encode.assert_not_called()

    def test_cache_changes_for_encoding_settings_but_not_batch_or_pca(self):
        original = sentence_embedding_cache_path(CONFIG, "cache/example")
        for key, value in (("sent_emb_model", "other/model"), ("sent_emb_dim", 1024),
                           ("sent_emb_max_length", 256), ("sent_emb_dtype", "float32"),
                           ("sent_emb_prompt", "Instruct: Cluster products\nQuery: "),
                           ("sent_emb_normalize", False)):
            with self.subTest(key=key):
                self.assertNotEqual(original, sentence_embedding_cache_path({**CONFIG, key: value}, "cache/example"))
        self.assertEqual(original, sentence_embedding_cache_path(
            {**CONFIG, "sent_emb_batch_size": 1, "sent_emb_pca": 64}, "cache/example"))

    def test_legacy_sentence_t5_cache_is_still_readable(self):
        config = {"sent_emb_model": "sentence-transformers/sentence-t5-base", "sent_emb_dim": 768}
        self.assertEqual(sentence_embedding_cache_path(config, "cache/example"),
                         "cache/example/processed/sentence-t5-base.sent_emb")

    def test_extraction_saves_float32_in_item_id_order_and_reuses_cache(self):
        obj = TigerDiffTokenizer.__new__(TigerDiffTokenizer)
        obj.config = {**CONFIG, "sent_emb_dim": 2}
        obj.log = Mock()
        dataset = SimpleNamespace(n_items=3, id_mapping={"id2item": ["[PAD]", "b", "a"]},
                                  item2meta={"a": "text A", "b": "text B"})
        expected = np.array([[1., 0.], [0., 1.]], dtype=np.float32)
        with tempfile.TemporaryDirectory() as directory:
            obj.cache_dir = directory
            with patch("data.tigerdiff_data.encode_sentences", return_value=expected) as encoder:
                np.testing.assert_array_equal(obj._load_sent_emb(dataset), expected)
                np.testing.assert_array_equal(obj._load_sent_emb(dataset), expected)
            encoder.assert_called_once_with(["text B", "text A"], obj.config)
            path = sentence_embedding_cache_path(obj.config, directory)
            self.assertEqual(Path(path).stat().st_size, 16)
            np.testing.assert_array_equal(read_sentence_embeddings(path, 2, 2), expected)
            with self.assertRaisesRegex(ValueError, "Invalid embedding cache"):
                read_sentence_embeddings(path, 2, 3)
            np.array([np.nan, 1., 2., 3.], dtype=np.float32).tofile(path)
            with self.assertRaisesRegex(ValueError, "Non-finite"):
                obj._load_sent_emb(dataset)

    def test_failed_encoding_does_not_create_cache(self):
        obj = TigerDiffTokenizer.__new__(TigerDiffTokenizer)
        obj.config = CONFIG
        dataset = SimpleNamespace(n_items=2, id_mapping={"id2item": ["[PAD]", "a"]}, item2meta={"a": "A"})
        with tempfile.TemporaryDirectory() as directory:
            path = f"{directory}/items.sent_emb"
            with patch("data.tigerdiff_data.encode_sentences", side_effect=RuntimeError("OOM")):
                with self.assertRaisesRegex(RuntimeError, "OOM"):
                    obj._encode_sent_emb(dataset, path)
            self.assertFalse(Path(path).exists())

    def test_embedding_cli_and_validation(self):
        config, _ = parse_config(["--sent_emb_batch_size=2", "--sent_emb_max_length=256",
                                  "--sent_emb_device=cpu", "--sent_emb_pca=64"])
        self.assertEqual((config["sent_emb_batch_size"], config["sent_emb_max_length"],
                          config["sent_emb_device"], config["sent_emb_pca"]), (2, 256, "cpu", 64))
        for argv in (["--sent_emb_dim=0"], ["--sent_emb_batch_size=0"],
                     ["--sent_emb_max_length=0"], ["--sent_emb_pca=3000"]):
            with self.subTest(argv=argv), self.assertRaises(ValueError):
                parse_config(argv)
