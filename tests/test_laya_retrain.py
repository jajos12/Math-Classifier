"""Tests for record preparation: blending, per-row pools, JSONL loading."""

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from laya_retrain import (
    InvalidRowError,
    _digest,
    _row_key,
    RowPoolCandidates,
    blend_target,
    load_row_pools,
)

STATE = "h : a = b\n⊢ a = b"


class BlendTargetTests(unittest.TestCase):
    def test_default_blend(self):
        target = blend_target("simp", ["rw", "simp"], {"rw": 0.8, "simp": 0.2}, 0.3)
        self.assertAlmostEqual(target["simp"], 0.7 * 1.0 + 0.3 * 0.2, places=12)
        self.assertAlmostEqual(target["rw"], 0.3 * 0.8, places=12)
        self.assertAlmostEqual(sum(target.values()), 1.0, places=12)

    def test_lambda_zero_is_onehot(self):
        target = blend_target("simp", ["rw", "simp"], {"rw": 0.8, "simp": 0.2}, 0.0)
        self.assertEqual(target, {"rw": 0.0, "simp": 1.0})

    def test_lambda_one_is_gnn_distribution(self):
        target = blend_target("simp", ["rw", "simp"], {"rw": 0.8, "simp": 0.2}, 1.0)
        self.assertEqual(target, {"rw": 0.8, "simp": 0.2})

    def test_uniform_when_scores_sum_to_zero(self):
        target = blend_target("simp", ["rw", "simp"], {"rw": 0.0, "simp": 0.0}, 0.3)
        self.assertAlmostEqual(sum(target.values()), 1.0, places=12)

    def test_negative_scores_fail(self):
        self.assertIsNone(blend_target("simp", ["rw", "simp"], {"rw": -1.0, "simp": 0.2}, 0.3))

    def test_gold_outside_options_fails(self):
        self.assertIsNone(blend_target("exact", ["rw", "simp"], {"rw": 0.8, "simp": 0.2}, 0.3))


class RowPoolCandidatesTests(unittest.TestCase):
    def _candidates(self, pools=None, gnn_lambda=0.3):
        pools = pools or {"1": (["rw", "simp", "exact"], (0.7, 0.2, 0.1))}
        return RowPoolCandidates(pools, gnn_lambda=gnn_lambda)

    def test_options_shuffle_deterministically_per_row(self):
        candidates = self._candidates()
        first = candidates.for_row("simp", row_key="1")
        second = candidates.for_row("simp", row_key="1")
        self.assertEqual(first[0], second[0])
        self.assertEqual(sorted(first[0]), ["exact", "rw", "simp"])

    def test_target_hits_gold(self):
        pools = {"1": (["rw", "simp", "exact"], (0.7, 0.2, 0.1))}
        candidates = RowPoolCandidates(pools, options_per_row=3)
        options, target, meta = candidates.for_row("simp", row_key="1")
        self.assertIn("simp", options)
        self.assertEqual(target["simp"], max(target.values()))
        self.assertAlmostEqual(sum(target.values()), 1.0, places=12)
        self.assertFalse(meta["short_pool"])

    def test_pool_missing(self):
        with self.assertRaises(InvalidRowError) as caught:
            self._candidates().for_row("simp", row_key="999")
        self.assertEqual(caught.exception.reason, "pool_missing")

    def test_gold_not_in_pool(self):
        pools = {"1": (["rw", "simp"], (0.7, 0.3))}
        with self.assertRaises(InvalidRowError) as caught:
            self._candidates(pools).for_row("exact", row_key="1")
        self.assertEqual(caught.exception.reason, "gold_not_in_pool")

    def test_short_pool(self):
        pools = {"1": (["simp"], (1.0,))}
        with self.assertRaises(InvalidRowError) as caught:
            self._candidates(pools).for_row("simp", row_key="1")
        self.assertEqual(caught.exception.reason, "short_pool")

    def test_no_pools_raises(self):
        with self.assertRaises(ValueError):
            RowPoolCandidates({})

    def test_too_many_options_keep_natural_length(self):
        pools = {"1": (["rw", "simp", "exact"], (0.7, 0.2, 0.1))}
        candidates = self._candidates(pools)
        self.assertEqual(len(candidates.for_row("simp", row_key="1")[0]), 3)


class RowKeyTests(unittest.TestCase):
    def test_integer_row_index(self):
        self.assertEqual(_row_key({"row_index": 123}, "state"), "123")

    def test_float_rendered_index(self):
        self.assertEqual(_row_key({"row_index": "123.0"}, "state"), "123.0")

    def test_missing_index_falls_back_to_state_hash(self):
        expected = "state:" + hashlib.sha1(STATE.encode("utf-8")).hexdigest()[:16]
        self.assertEqual(_row_key({"row_index": None}, STATE), expected)


class DigestTests(unittest.TestCase):
    def test_stable_across_calls(self):
        self.assertEqual(_digest("h : a = b"), _digest("h : a = b"))

    def test_empty_is_empty(self):
        self.assertEqual(_digest(""), "")
        self.assertEqual(_digest(None), "")


class LoadRowPoolsTests(unittest.TestCase):
    def _write(self, lines):
        file = tempfile.NamedTemporaryFile(
            "w", suffix=".jsonl", delete=False, encoding="utf-8"
        )
        file.write("\n".join(lines) + "\n")
        file.close()
        return Path(file.name)

    def _valid_line(self):
        return json.dumps(
            {"row_index": 5, "ranking": ["rw", "simp"], "scores": [0.8, 0.2]}
        )

    def test_loads_rankings_and_scores(self):
        path = self._write([self._valid_line()])
        pools = load_row_pools(path)
        self.assertEqual(pools["5"], (("rw", "simp"), (0.8, 0.2)))
        path.unlink()

    def test_negative_score_rejected(self):
        path = self._write(
            [json.dumps({"row_index": 5, "ranking": ["rw"], "scores": [-0.5]})]
        )
        with self.assertRaises(ValueError):
            load_row_pools(path)
        path.unlink()

    def test_score_length_mismatch_rejected(self):
        path = self._write(
            [json.dumps({"row_index": 5, "ranking": ["rw"], "scores": [0.5, 0.5]})]
        )
        with self.assertRaises(ValueError):
            load_row_pools(path)
        path.unlink()

    def test_unknown_tactics_dropped(self):
        path = self._write(
            [json.dumps(
                {"row_index": 5, "ranking": ["rw", "nope_tactic"], "scores": [0.8, 0.2]}
            )]
        )
        pools = load_row_pools(path)
        self.assertEqual(pools["5"], (("rw",), (0.8,)))
        path.unlink()

    def test_missing_file(self):
        with self.assertRaises(ValueError):
            load_row_pools(Path("/nonexistent/pools.jsonl"))


if __name__ == "__main__":
    unittest.main()
