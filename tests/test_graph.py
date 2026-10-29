"""Tests for Lean state parsing and tactic normalization."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from graph import (
    normalize_tactic,
    parse_state,
    proof_state_to_dag,
    root_state_node_index,
)


class ParseStateTests(unittest.TestCase):
    def test_hypotheses_and_goal(self):
        parsed = parse_state("h : a = b\n⊢ a = b")
        self.assertEqual(len(parsed.hypotheses), 1)
        self.assertEqual(parsed.hypotheses[0].name, "h")
        self.assertEqual(parsed.hypotheses[0].type_expr, "a = b")
        self.assertEqual(parsed.goal, "a = b")

    def test_no_turnstile_is_all_goal(self):
        parsed = parse_state("a = b")
        self.assertEqual(parsed.hypotheses, [])
        self.assertEqual(parsed.goal, "a = b")

    def test_empty_goal_raises(self):
        with self.assertRaises(ValueError):
            parse_state("h : a = b\n⊢ ")


class NormalizeTacticTests(unittest.TestCase):
    def test_first_token(self):
        self.assertEqual(normalize_tactic("simp [h, g]"), "simp")

    def test_nan_becomes_sentinel(self):
        self.assertEqual(normalize_tactic(float("nan")), "<EMPTY_TACTIC>")

    def test_none_becomes_sentinel(self):
        self.assertEqual(normalize_tactic(None), "<EMPTY_TACTIC>")

    def test_plain_app(self):
        self.assertEqual(normalize_tactic("rw h"), "rw")


class ProofStateDagTests(unittest.TestCase):
    def test_single_state_root(self):
        dag = proof_state_to_dag("h : a = b\n⊢ a = b")
        self.assertGreater(dag.num_nodes, 0)
        self.assertGreater(dag.num_edges, 0)
        root = root_state_node_index(dag)
        self.assertEqual(dag.nodes[root].label, "State")

    def test_pipeline_validate_equivalent(self):
        # The same check `pipeline.py validate` runs.
        dag = proof_state_to_dag("h : α = α\n⊢ α = α")
        self.assertTrue(any(node.label == "State" for node in dag.nodes))


if __name__ == "__main__":
    unittest.main()
