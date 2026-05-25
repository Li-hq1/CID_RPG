import unittest

from genrec.models.RPG.tokenizer import RPGTokenizer


class DummyDataset:
    cache_dir = "cache/AmazonReviews2014/Beauty"


def make_tokenizer(sent_emb_pca):
    tokenizer = RPGTokenizer.__new__(RPGTokenizer)
    tokenizer.config = {
        "metadata": "sentence",
        "sent_emb_model": "sentence-transformers/sentence-t5-base",
        "sent_emb_dim": 768,
        "sent_emb_pca": sent_emb_pca,
    }
    tokenizer.index_factory = "OPQ32,IVF1,PQ32x8"
    return tokenizer


class RPGTokenizerTest(unittest.TestCase):
    def test_semantic_id_cache_path_includes_sent_emb_pca(self):
        pca128_path = make_tokenizer(128)._semantic_id_cache_path(DummyDataset())
        pca64_path = make_tokenizer(64)._semantic_id_cache_path(DummyDataset())

        self.assertNotEqual(pca128_path, pca64_path)
        self.assertTrue(pca128_path.endswith(".sem_ids"))
        self.assertIn("sentence-t5-base_OPQ32,IVF1,PQ32x8_", pca128_path)


if __name__ == "__main__":
    unittest.main()
