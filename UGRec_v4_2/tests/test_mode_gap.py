import unittest

import numpy as np
import torch

from research.mode_gap import assign_tertiles, classify, nearest_ids, successor_statistics, transition_counts


class ModeGapTest(unittest.TestCase):
    def test_frequency_weighted_mean_pulls_toward_the_repeated_successor(self):
        catalog = np.array([[0., 0.], [0., 0.], [4., 0.], [0., 2.]], dtype=np.float64)
        counts = transition_counts([[1, 2], [1, 2], [1, 3]])
        record = successor_statistics(counts, catalog)[1]
        np.testing.assert_allclose(record["weighted_mean"], [8 / 3, 2 / 3])
        np.testing.assert_allclose(record["uniform_mean"], [2., 1.])
        self.assertGreater(record["entropy"], 0.)

    def test_classification_priority(self):
        labels = classify(
            n_successors=[1, 2, 2, 2, 2],
            nearest_is_target=[False, True, False, False, False],
            nearest_in_support=[False, True, False, True, False],
            d_mean=[0., 0., 0.2, 2., 2.],
            d_nearest=[1., 1., 1., 1., 1.],
        )
        self.assertEqual(labels.tolist(), ["ineligible", "correct", "gap", "wrong_mode", "off_support"])

    def test_tertiles_cover_only_the_eligible_users(self):
        entropy = np.array([0., 0.1, 0.2, 0.3, 0.9])
        eligible = np.array([False, True, True, True, True])
        labels, cuts = assign_tertiles(entropy, eligible)
        self.assertEqual(labels[0], "ineligible")
        self.assertEqual(set(labels[1:]), {"low", "mid", "high"})
        self.assertLess(cuts["low_max_nats"], cuts["high_min_exclusive_nats"])

    def test_nearest_id_keeps_the_smaller_id_on_a_tie(self):
        queries = torch.tensor([[0., 0.]])
        items = torch.tensor([[1., 0.], [1., 0.], [0., 3.]])
        ids, distance = nearest_ids(queries, items, block=2)
        self.assertEqual(int(ids[0]), 1)
        self.assertAlmostEqual(float(distance[0]), 1.0)


if __name__ == "__main__":
    unittest.main()
