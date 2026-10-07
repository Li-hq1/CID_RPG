import unittest
from unittest.mock import Mock

import numpy as np

from data.amazon_reviews import AmazonReviews2014
from data.tigerdiff_data import TigerDiffTokenizer


def tokenizer():
    obj = TigerDiffTokenizer.__new__(TigerDiffTokenizer)
    obj.config = {"max_item_seq_len": 3, "sent_emb_pca": 0}
    obj.item2id = {str(i): i for i in range(1, 9)}
    obj.ignored_label = -100
    obj.cache_dir = "cache/example"
    return obj


class DataTest(unittest.TestCase):
    def test_leave_one_out_and_padding_ids_are_preserved(self):
        obj = AmazonReviews2014.__new__(AmazonReviews2014)
        obj.config = {"split": "leave_one_out"}
        obj.split_data = None
        obj.all_item_seqs = {"u": ["1", "2", "3", "4"], "short": ["1"]}
        splits = obj.split()
        self.assertEqual(splits["train"]["item_seq"], [["1", "2"]])
        self.assertEqual(splits["val"]["item_seq"], [["1", "2", "3"]])
        self.assertEqual(splits["test"]["item_seq"], [["1", "2", "3", "4"], ["1"]])
        self.assertIs(obj.split(), splits)

    def test_windows_and_validation_labels(self):
        obj = tokenizer()
        train = obj.tokenize_function({"item_seq": [["1", "2", "3", "4", "5"]]}, "train")
        self.assertEqual(train["input_ids"], [[1, 2, 3], [2, 3, 4]])
        self.assertEqual(train["labels"], [[2, 3, 4], [-100, -100, 5]])
        val = obj.tokenize_function({"item_seq": [["1", "2"]]}, "val")
        self.assertEqual(val["input_ids"], [[1, 0, 0]])
        self.assertEqual(val["attention_mask"], [[1, 0, 0]])
        self.assertEqual(val["labels"], [[2]])
        self.assertEqual(val["seq_lens"], [1])

    def test_embedding_statistics_use_training_items_only(self):
        obj = tokenizer()
        embeddings = np.array([[1., 2.], [3., 4.], [100., 200.]], dtype=np.float32)
        result = obj._fit_transform_sentence_embeddings(embeddings, np.array([True, True, False]))
        np.testing.assert_allclose(result, [[-1., -1.], [1., 1.], [98., 197.]])
        self.assertEqual(result.dtype, np.float32)

    def test_cache_key_matches_existing_sports_cache(self):
        from config import CONFIG
        obj = tokenizer()
        obj.config = {
            **CONFIG,
            "sent_emb_model": "sentence-transformers/sentence-t5-base",
            "sent_emb_dim": 768,
        }
        for key in ("sent_emb_max_length", "sent_emb_dtype", "sent_emb_prompt", "sent_emb_normalize"):
            obj.config.pop(key, None)
        obj.cache_dir = "cache/AmazonReviews2014/Sports_and_Outdoors"
        self.assertTrue(obj._tokenized_cache_path().endswith("tokenized_datasets_TigerDiff_e24633984a"))

    def test_cached_datasets_are_reused_without_retokenizing(self):
        import tempfile
        from pathlib import Path
        from datasets import Dataset
        obj = tokenizer()
        obj.REGENERATE_SAMPLE = False
        obj.log = Mock()
        obj._tokenize_split = Mock(side_effect=AssertionError("unexpected tokenization"))
        with tempfile.TemporaryDirectory() as directory:
            obj._tokenized_cache_path = Mock(return_value=directory)
            for split in ("train", "val", "test"):
                Dataset.from_dict({"input_ids": [[1, 0, 0]], "labels": [[2]]}).save_to_disk(str(Path(directory) / split))
            result = obj.tokenize(dict.fromkeys(("train", "val", "test")))
        self.assertEqual(set(result), {"train", "val", "test"})
        self.assertEqual(result["val"][0]["labels"].tolist(), [2])
        obj._tokenize_split.assert_not_called()
