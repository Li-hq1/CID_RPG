import unittest

import numpy as np

from research.candidate_budget_study import build_pool, paired_hits, rank_draws


class CandidateBudgetTests(unittest.TestCase):
    def test_duplicate_draws_padding_and_ce_overlap_do_not_inflate_budget(self):
        ce = np.arange(1, 43)[None]
        extra = np.array([[1, 1, 2, 60, 60, 61, 0]])
        pool = build_pool(ce, extra, 20)
        self.assertEqual(len(set(pool[0])), 20)
        np.testing.assert_array_equal(pool[0, :10], ce[0, :10])
        self.assertIn(60, pool[0])
        self.assertIn(61, pool[0])
        self.assertNotIn(0, pool[0])

    def test_frequency_then_similarity_then_id(self):
        ids = np.array([[2, 4, 2, 4, 9, 8]])
        cos = np.array([[.2, .3, .2, .3, .5, .5]])
        np.testing.assert_array_equal(rank_draws(ids, cos), [[4, 2, 8, 9, 0, 0]])

    def test_hits_count_users_once_and_report_losses(self):
        result = paired_hits(np.array([[1, 1], [3, 4]]), np.array([[2, 3], [2, 2]]), np.array([1, 2]))
        self.assertEqual(result['recall'], .5)
        self.assertEqual(result['gained_users'], 1)
        self.assertEqual(result['lost_users'], 1)

    def test_final10_ce_protection_survives_duplicate_candidates(self):
        ce = np.arange(1, 43)[None]
        pool = build_pool(ce, np.array([[1, 1, 50, 51, 52]]), 10, protected=8)
        np.testing.assert_array_equal(pool, [[1, 2, 3, 4, 5, 6, 7, 8, 50, 51]])
