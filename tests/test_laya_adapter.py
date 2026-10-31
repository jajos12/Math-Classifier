"""Tests for the Laya adapter: state translation, questions, glosses, predict."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from laya_adapter import (
    DEFAULT_INSTRUCTIONS,
    LayaAdapter,
    TACTIC_GLOSS,
    choice_question,
    laya_state,
    rotated_choice_question,
    translate_lean_statement,
    translate_lean_state,
    translate_tactic_step,
)
from laya_interface import LayaAdapter as ExportedAdapter
from tactic_choose import TACTIC_SET

STATE = "h : a = b\n⊢ a = b"


class TranslateStatementTests(unittest.TestCase):
    def test_symbols_expanded(self):
        result = translate_lean_statement("⊢ a ∧ b → ∃ x, x ≤ 5")
        self.assertIn("and", result)
        self.assertIn("implies", result)
        self.assertIn("there exists", result)
        self.assertIn("is less than or equal to", result)

    def test_identifiers_preserved(self):
        result = translate_lean_statement("my_lemma_h : a = b")
        self.assertIn("my_lemma_h", result)

    def test_turnstile_renders(self):
        result = translate_lean_statement("h : p\n⊢ p")
        self.assertIn("the goal is", result)


class TranslateTacticTests(unittest.TestCase):
    def test_known_gloss(self):
        result = translate_tactic_step("rw")
        self.assertEqual(result, "Use the Lean tactic 'rw' to rewrite with an equation.")

    def test_unknown_family_raises(self):
        with self.assertRaises(ValueError):
            translate_tactic_step("definitely_not_a_tactic")

    def test_arguments_appended(self):
        result = translate_tactic_step("rw", "h")
        self.assertIn("with arguments h", result)


class GlossTests(unittest.TestCase):
    def test_every_gloss_key_is_a_canonical_tactic(self):
        invalid = set(TACTIC_GLOSS) - TACTIC_SET
        self.assertEqual(invalid, set())

    def test_removed_duplicate_intro(self):
        # The gloss dict has no duplicate keys by construction; assert the
        # surviving value is the intended one.
        self.assertEqual(
            TACTIC_GLOSS["intro"], "introduce a binder or hypothesis"
        )

    def test_previously_circular_glosses_now_describe_the_tactic(self):
        for name in ("rintro", "obtain", "push_neg", "norm_cast", "interval_cases"):
            self.assertNotIn(f"tactic family '{name}'", TACTIC_GLOSS[name])


class ChoiceQuestionTests(unittest.TestCase):
    def test_criteria_cover_candidates(self):
        question = choice_question(["rw", "simp"], DEFAULT_INSTRUCTIONS, STATE)
        self.assertEqual(set(question["tactic"]["criteria"]), {"rw", "simp"})

    def test_state_hints_attach(self):
        question = choice_question(["rw"], DEFAULT_INSTRUCTIONS, STATE)
        description = question["tactic"]["criteria"]["rw"]
        self.assertIn("equality information", description)

    def test_empty_candidates_raise(self):
        with self.assertRaises(ValueError):
            choice_question([], DEFAULT_INSTRUCTIONS, STATE)


class RotatedQuestionTests(unittest.TestCase):
    OPTIONS = ["rw", "simp", "exact"]

    def test_rotation_is_recorded(self):
        question = rotated_choice_question(self.OPTIONS, DEFAULT_INSTRUCTIONS, 1)
        self.assertEqual(list(question["tactic"]["criteria"]), ["simp", "exact", "rw"])
        self.assertEqual(question["tactic"]["option_order"], [1, 2, 0])

    def test_rotation_wraps_around(self):
        question = rotated_choice_question(self.OPTIONS, DEFAULT_INSTRUCTIONS, 3)
        self.assertEqual(list(question["tactic"]["criteria"]), self.OPTIONS)


class LayaStateTests(unittest.TestCase):
    def test_structured_sections(self):
        state = laya_state(STATE)
        for section in ("PROOF STATE", "CURRENT GOAL", "LOCAL HYPOTHESES", "TASK"):
            self.assertIn(section, state)
        self.assertIn("h : a = b", state)


class FakeAgent:
    def __init__(self, probabilities):
        self._probabilities = probabilities

    def predict(self, state, question):
        return {"answers": {"tactic": {"probabilities": self._probabilities}}}


class AdapterPredictTests(unittest.TestCase):
    def test_order_follows_probabilities(self):
        adapter = LayaAdapter(FakeAgent({"simp": 0.2, "rw": 0.8}), DEFAULT_INSTRUCTIONS)
        result = adapter.predict(STATE, ["simp", "rw"])
        self.assertEqual(result["order"], ["rw", "simp"])
        self.assertEqual(result["probabilities"], {"simp": 0.2, "rw": 0.8})

    def test_key_mismatch_raises(self):
        adapter = LayaAdapter(FakeAgent({"simp": 0.5}), DEFAULT_INSTRUCTIONS)
        with self.assertRaises(ValueError):
            adapter.predict(STATE, ["simp", "rw"])

    def test_facade_reexports(self):
        self.assertIs(ExportedAdapter, LayaAdapter)


if __name__ == "__main__":
    unittest.main()
