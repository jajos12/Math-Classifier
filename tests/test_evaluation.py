"""Tests for the paired ranking metrics."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation import build_report, rank_of, transition_table, evaluate_rankings, recall_keys


class RankOfTests(unittest.TestCase):
    def test_found(self):
        self.assertEqual(rank_of("b", ["a", "b", "c"]), 2)

    def test_missing(self):
        self.assertIsNone(rank_of("z", ["a", "b", "c"]))


class EvaluateRankingsTests(unittest.TestCase):
    def test_mask_excludes_unknown_targets(self):
        rankings = [["a", "b"], ["c", "a"]]
        targets = ["b", "<EMPTY_TACTIC>"]
        mask = [True, False]
        result = evaluate_rankings(rankings, targets, mask, ks=(2, 1), mrr_k=2)
        self.assertEqual(result["n"], 1)
        self.assertEqual(result["recall@1"], 0.0)  # b sits at rank 2
        self.assertEqual(result["recall@2"], 1.0)
        self.assertAlmostEqual(result["mrr@2"], 0.5, places=12)

    def test_empty_mask(self):
        result = evaluate_rankings([["a"]], ["a"], [False], ks=(1,), mrr_k=1)
        self.assertEqual(result, {"n": 0})


class TransitionTableTests(unittest.TestCase):
    def test_counts(self):
        reference = [["a", "b"], ["a", "b"], ["a", "b"], ["a", "b"]]
        candidate = [["a", "b"], ["c", "b"], ["b", "a"], ["d", "e"]]
        targets = ["a", "b", "b", "z"]
        mask = [True, True, True, False]  # index 3 has an unknown target
        result = transition_table(reference, candidate, targets, mask, k=1)
        # index 0: both right; index 1: gold outside both top-1: neither;
        # index 2: gold outside reference top-1 but first in candidate.
        self.assertEqual(result, {
            "both right": 1 / 3,
            "only GNN right": 0.0,
            "only Laya right": 1 / 3,
            "neither": 1 / 3,
            "n": 3,
        })


class BuildReportTests(unittest.TestCase):
    def test_paired_metrics_and_pool_recall(self):
        gnn = [["b", "c", "a"]]
        laya = [["a", "b", "c"]]
        targets = ["a"]
        report = build_report(gnn, laya, targets, pool_k=3, select_k=2)
        self.assertEqual(report["rows"], 1)
        self.assertEqual(report["pool_rows"], 1)
        self.assertAlmostEqual(report["pool_recall"], 1.0, places=12)
        self.assertEqual(report["metrics"]["gnn"]["recall@1"], 0.0)
        self.assertEqual(report["metrics"]["laya"]["recall@1"], 1.0)
        self.assertEqual(report["metrics"]["laya"]["conditional"]["recall@2"], 1.0)
        self.assertIn("top_1", report["transitions"])

    def test_top_out_of_pool_marks_unknown(self):
        report = build_report([["a"]], [["a"]], ["<EMPTY_TACTIC>"], pool_k=1, select_k=1)
        self.assertEqual(report["known_targets"], 0)
        self.assertEqual(report["pool_recall"], 0.0)


class RecallKeysTests(unittest.TestCase):
    def test_pool_size_first_then_five_then_one(self):
        keys = recall_keys({"recall@10": 1.0, "recall@1": 0.2, "recall@5": 0.5, "mrr@10": 0.3})
        self.assertEqual(keys, ["recall@10", "recall@5", "recall@1"])


if __name__ == "__main__":
    unittest.main()
