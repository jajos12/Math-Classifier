"""Log-linear fusion of the GNN ranking signal and Laya's probabilities.

The benchmark's reranking keeps the two systems separate: Laya reorders the
GNN top-k pool and the GNN's own scores are discarded. The two signals are
partly complementary, so this module implements the cheapest way to use both:

    fused score(option) = alpha·log p_laya(option) + (1−alpha)·log p_gnn(option)

``alpha = 1`` recovers Laya's own ranking, ``alpha = 0`` the GNN's, and the
weight is meant to be chosen on validation and reported on test (see
``fuse_eval.py``). GNN scores are raw non-negative probabilities normalized
over the pool options; zero probabilities are clipped to ``eps`` so logs stay
finite. No torch, laya, or pandas dependency — importable anywhere.

Fusion is a *ranking* of the existing pool: it can reorder the pool but never
add, remove, or move a candidate out of the top-k list, so pool recall@k is
unchanged by construction.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math


def normalize_gnn_scores(
    scores: Sequence[float],
    *,
    eps: float = 1e-9,
) -> list[float]:
    """Clip and normalize non-negative scores into a distribution.

    Negative scores are clipped to ``0`` (the pool cache stores soft-max
    probabilities, so negative values only appear through rounding paths),
    an all-zero vector becomes uniform, and the result always sums to 1.
    """
    normalized = [max(float(score), 0.0) for score in scores]
    total = sum(normalized)
    if total <= 0.0:
        count = len(normalized)
        return [1.0 / count] * count if count else []
    return [max(value / total, eps) for value in normalized]


def fuse_ranking(
    ranking: Sequence[str],
    gnn_scores: Sequence[float],
    laya_probs: Mapping[str, float],
    alpha: float,
    *,
    eps: float = 1e-9,
) -> list[str]:
    """One fused ranking for one row.

    ``ranking`` is the candidate pool (the row's GNN top-k names), aligned in
    order with ``gnn_scores``; ``laya_probs`` must hold a probability for
    exactly those names (what ``LayaAdapter.predict`` guarantees). The result
    is the pool sorted by ``alpha·log p_laya + (1−alpha)·log p_gnn``
    descending, ties broken by name so the order is deterministic.
    """
    if not (math.isfinite(alpha) and 0.0 <= alpha <= 1.0):
        raise ValueError(f"alpha must be a finite weight in [0, 1], got {alpha!r}")
    names = list(ranking)
    if not names:
        return []
    if set(laya_probs) != set(names):
        raise ValueError(
            "Laya probabilities must cover exactly the pool options; got "
            f"{sorted({str(k) for k in laya_probs})} vs {sorted(names)}"
        )
    if len(gnn_scores) != len(names):
        raise ValueError(
            f"gnn_scores has {len(gnn_scores)} entries for a pool of "
            f"{len(names)} options"
        )
    gnn_dist = normalize_gnn_scores(gnn_scores, eps=eps)
    gnn_by_name = dict(zip(names, gnn_dist))

    def fused_score(name: str) -> float:
        p_laya = max(float(laya_probs[name]), eps)
        p_gnn = gnn_by_name[name]
        return alpha * math.log(p_laya) + (1.0 - alpha) * math.log(p_gnn)

    return sorted(names, key=lambda name: (-fused_score(name), name))


def fuse_rankings(
    rankings: Sequence[Sequence[str]],
    gnn_scores: Sequence[Sequence[float]],
    laya_probabilities: Sequence[Mapping[str, float]],
    alpha: float,
    *,
    eps: float = 1e-9,
) -> list[list[str]]:
    """Fused ranking per row, for a whole pool at a time."""
    return [
        fuse_ranking(ranking, scores, probs, alpha, eps=eps)
        for ranking, scores, probs in zip(rankings, gnn_scores, laya_probabilities)
    ]
