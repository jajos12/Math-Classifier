"""Tests for comparison helpers: reranking, device notes, acceptance gates."""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from comparison import acceptance_block, describe_device, rerank_states


class RecordingAgent:
    """Fake Laya agent: probabilities prefer later candidates in the list."""

    def predict(self, state: str, question: dict) -> dict:
        candidates = list(question["tactic"]["criteria"])
        probabilities = {
            name: (index + 1) / len(candidates) for index, name in enumerate(candidates)
        }
        order = sorted(candidates, key=lambda name: probabilities[name])
        return {"answers": {"tactic": {"probabilities": probabilities, "choice": order[0]}}}


def base_metrics():
    return {"recall@10": 0.9, "recall@5": 0.7, "recall@1": 0.3, "mrr@10": 0.4}


class RerankStatesTests(unittest.TestCase):
    def _adapter(self):
        from laya_adapter import DEFAULT_INSTRUCTIONS, LayaAdapter

        return LayaAdapter(RecordingAgent(), DEFAULT_INSTRUCTIONS)

    def test_returns_rankings_and_probabilities(self):
        outcome = rerank_states(
            self._adapter(),
            ["h : a = b\n⊢ a = b"],
            [["rw", "simp"]],
        )
        self.assertEqual(outcome["rankings"], [["simp", "rw"]])
        self.assertEqual(outcome["probabilities"], [{"rw": 0.5, "simp": 1.0}])

    def test_balanced_mode_uses_rotations(self):
        # balanced_predict averages probabilities over candidate rotations;
        # the fake agent's per-name probabilities are rotation-invariant, so
        # both names tie (0.75 each) and the tie breaks alphabetically.
        outcome = rerank_states(
            self._adapter(),
            ["h : a = b\n⊢ a = b"],
            [["rw", "simp"]],
            mode="balanced",
        )
        self.assertEqual(outcome["rankings"], [["rw", "simp"]])
        self.assertEqual(outcome["probabilities"], [{"rw": 0.75, "simp": 0.75}])


class DescribeDeviceTests(unittest.TestCase):
    def test_records_actual_device(self):
        agent = SimpleNamespace(device="cuda:1")
        note = describe_device(agent, "cuda:1")
        self.assertEqual(note, {"requested": "cuda:1", "actual": "cuda:1", "warning": None})

    def test_silent_cpu_fallback_warns(self):
        agent = SimpleNamespace(_device="cpu:0")
        note = describe_device(agent, "cuda:1")
        self.assertIn("CPU", note["warning"])
        self.assertEqual(note["actual"], "cpu:0")


class AcceptanceBlockTests(unittest.TestCase):
    def test_fused_gates(self):
        report = {
            "n": 4392,
            "gnn": {"metrics": base_metrics()},
            "models": {
                "finetuned": {"metrics": {"recall@10": 0.9, "recall@5": 0.8, "recall@1": 0.35, "mrr@10": 0.5}},
                "base": {"metrics": {"recall@1": 0.2}},
                "fused": {"metrics": {"recall@10": 0.9, "recall@5": 0.8, "recall@1": 0.4, "mrr@10": 0.55}},
            },
        }
        block = acceptance_block(report)
        labels = [gate["gate"] for gate in block["gates"]]
        self.assertEqual(labels, [
            "fused.recall@5 > gnn.recall@5",
            "fused.recall@1 > gnn.recall@1",
            "fused.mrr@10 > gnn.mrr@10",
            "finetuned.recall@1 > base.recall@1",
        ])
        self.assertTrue(block["passed"])
        self.assertTrue(block["complete"])
        self.assertTrue(block["rows_sufficient"])

    def test_fallback_gates_without_fusion(self):
        report = {
            "n": 4392,
            "gnn": {"metrics": base_metrics()},
            "models": {
                "finetuned": {"metrics": {**base_metrics(), "recall@5": 0.8, "mrr@10": 0.5}},
                "base": {"metrics": {"recall@1": 0.2}},
            },
        }
        block = acceptance_block(report)
        labels = [gate["gate"] for gate in block["gates"]]
        self.assertEqual(labels, [
            "finetuned.recall@5 > gnn.recall@5",
            "finetuned.mrr@10 > gnn.mrr@10",
            "finetuned.recall@1 > base.recall@1",
        ])
        self.assertTrue(block["passed"])

    def test_missing_metrics_are_reported_not_passing(self):
        block = acceptance_block({"n": 10, "gnn": {"metrics": base_metrics()}, "models": {}})
        self.assertFalse(block["passed"])
        self.assertFalse(block["complete"])
        self.assertIn("finetuned/gnn metrics", block["missing"])
        self.assertIn("finetuned/base recall@1", block["missing"])

    def test_small_row_count_is_flagged(self):
        report = {
            "n": 100,
            "gnn": {"metrics": base_metrics()},
            "models": {
                "finetuned": {"metrics": {**base_metrics(), "recall@5": 0.8, "mrr@10": 0.5}},
                "base": {"metrics": {"recall@1": 0.1}},
            },
        }
        block = acceptance_block(report)
        self.assertFalse(block["rows_sufficient"])
        self.assertEqual(block["min_rows"], 4000)

    def test_worse_fused_fails(self):
        report = {
            "n": 4392,
            "gnn": {"metrics": base_metrics()},
            "models": {
                "finetuned": {"metrics": base_metrics()},
                "base": {"metrics": {"recall@1": 0.2}},
                "fused": {"metrics": {**base_metrics(), "recall@5": 0.6, "recall@1": 0.25, "mrr@10": 0.35}},
            },
        }
        block = acceptance_block(report)
        self.assertFalse(block["passed"])


if __name__ == "__main__":
    unittest.main()
