"""Small adapter around the optional Laya decision model."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import re
from typing import Any

try:
    from maths_ai.gnn_inference.atp_lean_gnn.state import parse_state
except ImportError:
    from graph import parse_state

from tactic_choose import TACTIC_SET

DEFAULT_GLOSS = "a Lean 4 tactic family; choose it only when its name matches the proof step"
DEFAULT_INSTRUCTIONS = (
    "This is a multiple-choice question about a Lean 4 proof: from the candidate "
    "tactics listed with this state, pick the one that is the correct next tactic. "
    "Read the state as a whole first — the goal says what still has to be proved, "
    "and the local hypotheses say what you have to work with: a fact to rewrite "
    "with, apply, simplify, case-split, or feed to arithmetic. Every candidate "
    "comes with a short description of what that tactic does; use it to picture "
    "what would actually happen if you ran the tactic on this exact state. Cross "
    "out the options that cannot apply here, would leave the goal unchanged, or "
    "head in a different direction, and keep the ones that make concrete progress "
    "toward the goal as written — if several still fit, prefer the one that uses "
    "the hypotheses most directly. "
    "Answer with a probability for every candidate: how likely each one is to be "
    "the correct choice, with the values summing to 1 across the list. These "
    "numbers express how confident you are in the right answer, not how useful you "
    "find an option — a hesitant question spreads its probability out, a clear one "
    "concentrates it."
)
TACTIC_GLOSS: dict[str, str] = {
    ".": "move on to the next goal in the list",
    "apply": "apply a lemma or hypothesis to the goal",
    "assumption": "close the goal with a matching hypothesis",
    "cases": "destruct a value into its constructors",
    "cases'": "destruct a value into its constructors with custom names",
    "constructor": "apply the goal's first constructor",
    "exact": "close the goal with an exact term",
    "intro": "introduce a binder or hypothesis",
    "linarith": "linear arithmetic reasoning",
    "nlinarith": "nonlinear arithmetic reasoning",
    "rfl": "close the goal by reflexivity",
    "rw": "rewrite with an equation",
    "simp": "simplify with a lemma set",
    "simpa": "simplify and then close with exact",
    "ring": "commutative ring normalization",
    "refine": "refine the goal with a partial term",
    "aesop": "automatically search using safe local hypotheses and standard rules",
    "by_cases": "split the proof into cases on a proposition",
    "by_contra": "prove the goal by assuming its negation",
    "calc": "start or continue a chained equality or relation calculation",
    "change": "replace the goal with a definitionally equal form",
    "clear": "remove an unused hypothesis from the local context",
    "decide": "close a decidable proposition by computation",
    "dsimp": "perform definitional simplification",
    "exact_mod_cast": "close the goal after transporting across numeric casts",
    "exfalso": "change the goal to False and derive a contradiction",
    "exists": "provide a witness for an existential goal",
    "ext": "reduce an equality of structures or functions to extensional goals",
    "field_simp": "clear field denominators and reduce to ring-like goals",
    "funext": "prove function equality by proving equality for every input",
    "have": "introduce an intermediate proposition or local fact",
    "induction": "split the proof using an induction principle",
    "induction'": "apply an explicit induction principle with custom naming",
    "norm_num": "normalize and prove concrete numerical arithmetic",
    "omega": "solve Presburger arithmetic over natural or integer variables",
    "positivity": "prove that an expression is positive or nonnegative",
    "rcases": "destructure a hypothesis into its components",
    "ring_nf": "normalize a commutative semiring or ring expression",
    "simp_all": "simplify the goal and all local hypotheses",
    "solve_by_elim": "close the goal by applying matching local hypotheses",
    "split": "split a conjunction, structure, or equivalent goal",
    "split_ifs": "split the goal at its if-then-else condition points",
    "subst": "replace a variable using an equality hypothesis",
    "tauto": "solve a propositional-logic goal",
    "unfold": "unfold a named definition",
    "use": "supply a witness or explicit term for the goal",
    # --- tactics whose candidates previously fell back to the circular
    # --- description "use the Lean tactic family 'x'"; the glosses below are
    # --- what translate_tactic_step uses instead.
    "all_goals": "apply the given tactic to every remaining goal",
    "contradiction": "close the goal by finding a contradiction in the hypotheses",
    "convert": "change the goal to a related statement, deferring equality mismatches as side goals",
    "congr": "reduce the goal to equalities between corresponding parts",
    "done": "declare the remaining goals solved",
    "erw": "rewrite with weaker congruence matching than rw",
    "exacts": "close each remaining goal with the matching exact term in order",
    "gcongr": "prove a congruence goal by reducing it to arithmetic side conditions",
    "interval_cases": "case-split a numeric variable over the intervals it can lie in",
    "left": "reduce a disjunction goal to its left disjunct",
    "linear_combination": "close the goal as a linear combination of given equalities",
    "nth_rw": "rewrite using only the nth match of the given lemma within the goal",
    "obtain": "destruct a hypothesis into named components (rcases-style introduction)",
    "push_cast": "push casts down into expressions toward normalized forms",
    "push_neg": "push negations inward through the goal",
    "norm_cast": "normalize numeric casts between Nat, Int and Real across the goal",
    "right": "reduce a disjunction goal to its right disjunct",
    "rintro": "introduce binders from the goal, destructuring patterns as they appear",
    "rwa": "rewrite with the given equation and then close the goal by reflexivity",
    "set": "introduce a local definition for an expression, with the defining equality as a hypothesis",
    "show": "change the goal to a provably equal statement",
    "simp_rw": "repeatedly rewrite using simp lemmas",
    "specialize": "instantiate a universally quantified hypothesis with specific values",
    "suffices": "replace the goal with a sufficient new goal, keeping the original as a side goal",
    "symm": "apply symmetry to the goal, e.g. turn a = b into b = a",
    "trivial": "close an obviously true goal",
    "intros": "introduce all remaining binders of the goal as local hypotheses",
}


_LEAN_WORDS = {
    "⊢": "the goal is",
    "→": "implies",
    "->": "implies",
    "↔": "if and only if",
    "∧": "and",
    "∨": "or",
    "¬": "not",
    "∀": "for every",
    "∃": "there exists",
    "∈": "is an element of",
    "≤": "is less than or equal to",
    "≥": "is greater than or equal to",
    "≠": "is not equal to",
    "＝": "equals",
    "=": "equals",
}


def translate_lean_statement(statement: str) -> str:
    """Render common Lean symbols as short English phrases for Laya.

    This is intentionally deterministic and conservative: identifiers, theorem
    names, and type names are preserved, while only logical/connective syntax
    is expanded. The original Lean state should still be retained in datasets.
    """
    translated = statement.strip()
    for source, target in sorted(_LEAN_WORDS.items(), key=lambda item: -len(item[0])):
        translated = translated.replace(source, f" {target} ")
    translated = re.sub(r"\s+", " ", translated).strip()
    return translated


def translate_tactic_step(tactic_name: str, arguments: str = "") -> str:
    """Describe a tactic step in natural language for supervised Laya data."""
    tactic = tactic_name.strip()
    if tactic not in TACTIC_SET:
        raise ValueError(f"Unknown tactic family: {tactic_name!r}")
    gloss = TACTIC_GLOSS.get(tactic)
    if gloss is None:
        readable = tactic.replace("_", " ").replace("!", " aggressively")
        gloss = f"use the Lean tactic family '{readable}'"
    suffix = f" with arguments {arguments.strip()}" if arguments.strip() else ""
    return f"Use the Lean tactic '{tactic}' to {gloss}{suffix}."


def translate_lean_state(text_state: str) -> str:
    """Render a proof state as a clear, structured description for Laya.

    Lean identifiers and expressions remain visible so the model can use exact
    names, while headings make the goal/context relationship explicit.
    """
    parsed = parse_state(text_state)
    goal = translate_lean_statement(parsed.goal)
    if parsed.hypotheses:
        context_lines = [
            f"- {hypothesis.name}\n"
            f"  Readable: {translate_lean_statement(hypothesis.type_expr)}\n"
            f"  Lean: {hypothesis.name} : {hypothesis.type_expr}"
            for hypothesis in parsed.hypotheses
        ]
        context = "\n".join(context_lines)
    else:
        context = "- No local hypotheses are available."
    return (
        "PROOF STATE\n"
        "CURRENT GOAL\n"
        f"Readable: {goal}\n"
        f"Lean: {parsed.goal}\n"
        "LOCAL HYPOTHESES\n"
        f"{context}\n"
        "TASK\n"
        "Choose the proof action that best advances the current goal using this context."
    )


def laya_state(text_state: str) -> str:
    """Place the goal before local context so right truncation preserves it."""
    return translate_lean_state(text_state)


def choice_question(
    candidates: Sequence[str],
    instructions: str = DEFAULT_INSTRUCTIONS,
    text_state: str | None = None,
) -> dict[str, Any]:
    """Build the documented Laya choice-question payload."""
    if not candidates:
        raise ValueError("choice questions need at least one candidate")
    state_hints = _state_hints(text_state) if text_state else {}
    return {
        "tactic": {
            "type": "choice",
            "instructions": instructions,
            "criteria": {
                candidate: _candidate_description(candidate, state_hints)
                for candidate in candidates
            },
        }
    }


def _state_hints(text_state: str) -> dict[str, Any]:
    parsed = parse_state(text_state)
    goal = parsed.goal
    context = " ".join(hypothesis.type_expr for hypothesis in parsed.hypotheses)
    return {
        "goal": goal,
        "context": context,
        "hypotheses": parsed.hypotheses,
        "has_equality": "=" in goal or any("=" in h.type_expr for h in parsed.hypotheses),
        "has_arithmetic": bool(re.search(r"\b(?:Nat|Int)\b|[0-9]|[<>≤≥]", goal + context)),
        "has_implication": "→" in goal or "->" in goal,
        "has_existential": "∃" in goal,
        "has_conjunction": "∧" in goal,
    }


def _candidate_description(candidate: str, hints: Mapping[str, Any]) -> str:
    description = translate_tactic_step(candidate).removesuffix(".")
    if not hints:
        return description
    additions: list[str] = []
    if candidate in {"rw", "rwa", "nth_rewrite", "subst"} and hints["has_equality"]:
        additions.append("The state contains equality information that may be useful here.")
    if candidate in {"linarith", "nlinarith", "norm_num", "omega", "ring", "ring_nf"} and hints["has_arithmetic"]:
        additions.append("The goal or context contains arithmetic-looking expressions.")
    if candidate in {"intro", "intros"} and hints["has_implication"]:
        additions.append("The goal begins with an implication or binder.")
    if candidate in {"use", "exists"} and hints["has_existential"]:
        additions.append("The goal is existential and needs a witness.")
    if candidate in {"constructor", "split"} and hints["has_conjunction"]:
        additions.append("The goal contains a conjunction or structured target.")
    return f"{description}. {' '.join(additions)}".strip()


def rotated_choice_question(
    candidates: Sequence[str],
    instructions: str,
    rotation: int,
    text_state: str | None = None,
) -> dict[str, Any]:
    """Build a rotated question and retain the rotation metadata for auditing."""
    if not candidates:
        raise ValueError("choice questions need at least one candidate")
    offset = rotation % len(candidates)
    order = list(range(offset, len(candidates))) + list(range(offset))
    rotated = [candidates[index] for index in order]
    question = choice_question(rotated, instructions, text_state)
    question["tactic"]["option_order"] = order
    return question


class LayaUnavailableError(RuntimeError):
    """Raised when a real Laya run is requested without the optional package."""


class LayaAdapter:
    """Validated interface for real or fake Laya agents."""

    def __init__(self, agent: Any, instructions: str) -> None:
        self.agent = agent
        self.instructions = instructions

    @classmethod
    def load(
        cls,
        model_name: str,
        *,
        device: str,
        instructions: str = DEFAULT_INSTRUCTIONS,
    ) -> "LayaAdapter":
        try:
            import laya
        except ImportError as exc:
            raise LayaUnavailableError(
                "Laya is not installed. Install laya and the experiment dependencies before running reranking."
            ) from exc
        return cls(laya.load(model_name, device=device), instructions)

    def predict(self, text_state: str, candidates: Sequence[str]) -> dict[str, Any]:
        candidate_list = list(candidates)
        question = choice_question(candidate_list, self.instructions, text_state)
        result = self.agent.predict(laya_state(text_state), question)
        try:
            answer = result["answers"]["tactic"]
            probabilities = {str(key): float(value) for key, value in answer["probabilities"].items()}
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Laya response does not contain tactic probabilities") from exc
        if set(probabilities) != set(candidate_list):
            raise ValueError("Laya probabilities are not exactly a permutation of the candidates")
        order = sorted(candidate_list, key=lambda candidate: (-probabilities[candidate], candidate))
        return {
            "order": order,
            "probabilities": probabilities,
            "choice": answer.get("choice"),
            "answer_confidence": float(answer.get("answer_confidence", 0.0)),
        }

    def balanced_predict(self, text_state: str, candidates: Sequence[str]) -> dict[str, Any]:
        candidate_list = list(candidates)
        totals = {candidate: 0.0 for candidate in candidate_list}
        for rotation in range(len(candidate_list)):
            question = rotated_choice_question(candidate_list, self.instructions, rotation, text_state)
            result = self.agent.predict(laya_state(text_state), question)
            probabilities = result["answers"]["tactic"]["probabilities"]
            if set(probabilities) != set(candidate_list):
                raise ValueError("Laya probabilities are not exactly a permutation of the candidates")
            for candidate in candidate_list:
                totals[candidate] += float(probabilities[candidate]) / len(candidate_list)
        order = sorted(candidate_list, key=lambda candidate: (-totals[candidate], candidate))
        return {"order": order, "probabilities": totals, "answer_confidence": max(totals.values(), default=0.0)}
