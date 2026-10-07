import unittest

import numpy as np

from genrec.models.DiffAR.tokenizer import DiffARTokenizer


class DiffARTokenizerTest(unittest.TestCase):
    def test_sent_emb_pca_zero_skips_pca_and_keeps_embedding_width(self):
        tokenizer = DiffARTokenizer.__new__(DiffARTokenizer)
        tokenizer.config = {"sent_emb_pca": 0}

        sent_embs = np.array(
            [
                [1.0, 2.0, 3.0],
                [3.0, 4.0, 5.0],
                [5.0, 6.0, 7.0],
            ],
            dtype=np.float32,
        )
        train_mask = np.array([True, True, False])

        transformed = tokenizer._fit_transform_sentence_embeddings(
            sent_embs,
            train_mask,
        )

        train_sent_embs = sent_embs[train_mask]
        mean = np.mean(train_sent_embs, axis=0)
        std = np.std(train_sent_embs, axis=0)
        std = np.where(std < 1e-6, 1e-6, std)
        expected = ((sent_embs - mean) / std).astype(np.float32)

        self.assertEqual(transformed.shape, sent_embs.shape)
        self.assertTrue(np.allclose(transformed, expected))


if __name__ == "__main__":
    unittest.main()
