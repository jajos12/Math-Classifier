"""Tests for the log-linear fusion module."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fusion import fuse_ranking, fuse_rankings, normalize_gnn_scores


class NormalizeGnnScoresTests(unittest.TestCase):
    def test_proportional_distribution(self):
        result = normalize_gnn_scores([0.4, 0.1, 0.5])
        self.assertEqual(result, [0.4, 0.1, 0.5])

    def test_normalizes_to_one(self):
        result = normalize_gnn_scores([2.0, 2.0, 6.0])
        self.assertAlmostEqual(sum(result), 1.0, places=12)

    def test_all_zero_becomes_uniform(self):
        result = normalize_gnn_scores([0.0, 0.0, 0.0])
        self.assertEqual(result, [1 / 3, 1 / 3, 1 / 3])

    def test_negative_scores_are_clipped(self):
        result = normalize_gnn_scores([-1.0, 1.0, 1.0])
        self.assertEqual(result[0], normalize_gnn_scores([0.0, 1.0, 1.0])[0])
        # Clipping at eps keeps every log finite, at the cost of an eps-sized
        # surplus: the sum may exceed 1 by at most one clip.
        self.assertGreaterEqual(sum(result), 1.0)
        self.assertLessEqual(sum(result), 1.0 + 1e-6)

    def test_empty_scores(self):
        self.assertEqual(normalize_gnn_scores([]), [])


class FuseRankingTests(unittest.TestCase):
    NAMES = ["rw", "simp", "exact"]

    def _probs(self, mapping=None):
        return mapping or {"rw": 0.6, "simp": 0.3, "exact": 0.1}

    def test_alpha_one_recovers_laya_order(self):
        fused = fuse_ranking(self.NAMES, [0.1, 0.1, 0.8], self._probs(), 1.0)
        self.assertEqual(fused, ["rw", "simp", "exact"])

    def test_alpha_zero_recovers_gnn_order(self):
        fused = fuse_ranking(
            self.NAMES, [0.05, 0.05, 0.9], self._probs(), 0.0
        )
        self.assertEqual(fused, ["exact", "rw", "simp"])

    def test_middle_alpha_interpolates(self):
        # Laya prefers "rw", the GNN strongly prefers "exact": the blend must
        # place the two extremes at the ends of the fused order.
        fused = fuse_ranking(self.NAMES, [0.05, 0.05, 0.9], self._probs(), 0.5)
        self.assertEqual(fused[0], "exact")
        self.assertEqual(set(fused), set(self.NAMES))

    def test_zero_probability_is_clipped_not_infinite(self):
        fused = fuse_ranking(
            self.NAMES, [1.0, 0.0, 0.0], {"rw": 1e-30, "simp": 0.5, "exact": 0.5}, 0.99
        )
        # "rw" keeps its candidate slot; clipping the log keeps every name in.
        self.assertEqual(set(fused), set(self.NAMES))

    def test_ties_break_by_name(self):
        fused = fuse_ranking(
            ["zeta", "alpha"], [0.5, 0.5], {"zeta": 0.5, "alpha": 0.5}, 0.5
        )
        self.assertEqual(fused, ["alpha", "zeta"])

    def test_empty_pool(self):
        self.assertEqual(fuse_ranking([], [], {}, 0.5), [])

    def test_invalid_alpha_raises(self):
        with self.assertRaises(ValueError):
            fuse_ranking(self.NAMES, [1, 0, 0], self._probs(), 1.5)

    def test_probability_key_mismatch_raises(self):
        with self.assertRaises(ValueError):
            fuse_ranking(self.NAMES, [1, 0, 0], {"rw": 1.0, "simp": 0.0}, 0.5)

    def test_score_length_mismatch_raises(self):
        with self.assertRaises(ValueError):
            fuse_ranking(self.NAMES, [1.0, 0.0], self._probs(), 0.5)

    def test_zip_truncates_at_shortest_input(self):
        result = fuse_rankings(
            [self.NAMES, ["a", "b"]],
            [[1.0, 0.0, 0.0]],
            [self._probs()],
            0.5,
        )
        self.assertEqual(result, [["rw", "simp", "exact"]])


class FuseRankingsRowAlignedTests(unittest.TestCase):
    def test_per_row_ranking(self):
        rankings = [["rw", "simp"], ["a", "b"]]
        scores = [[0.8, 0.2], [0.5, 0.5]]
        probs = [{"rw": 0.9, "simp": 0.1}, {"a": 0.2, "b": 0.8}]
        result = fuse_rankings(rankings, scores, probs, 1.0)
        self.assertEqual(result, [["rw", "simp"], ["b", "a"]])


if __name__ == "__main__":
    unittest.main()
