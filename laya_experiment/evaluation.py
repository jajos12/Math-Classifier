"""Paired ranking metrics for the GNN pool experiment."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any
import json
from pathlib import Path


def rank_of(target: str, ranking: Sequence[str]) -> int | None:
    try:
        return list(ranking).index(target) + 1
    except ValueError:
        return None


def evaluate_rankings(
    rankings: Sequence[Sequence[str]],
    targets: Sequence[str],
    mask: Sequence[bool],
    *,
    ks: Sequence[int] = (10, 5, 1),
    mrr_k: int = 10,
) -> dict[str, Any]:
    indices = [index for index, keep in enumerate(mask) if keep]
    result: dict[str, Any] = {"n": len(indices)}
    if not indices:
        return result
    for k in ks:
        result[f"recall@{k}"] = sum(targets[i] in rankings[i][:k] for i in indices) / len(indices)
    reciprocal_ranks = []
    for index in indices:
        rank = rank_of(targets[index], rankings[index][:mrr_k])
        reciprocal_ranks.append(0.0 if rank is None else 1.0 / rank)
    result[f"mrr@{mrr_k}"] = sum(reciprocal_ranks) / len(reciprocal_ranks)
    return result


def transition_table(
    reference: Sequence[Sequence[str]],
    candidate: Sequence[Sequence[str]],
    targets: Sequence[str],
    mask: Sequence[bool],
    *,
    k: int,
) -> dict[str, float | int]:
    indices = [index for index, keep in enumerate(mask) if keep]
    total = len(indices)
    if not total:
        return {"both right": 0.0, "only GNN right": 0.0, "only Laya right": 0.0, "neither": 0.0, "n": 0}
    reference_ok = [targets[i] in reference[i][:k] for i in indices]
    candidate_ok = [targets[i] in candidate[i][:k] for i in indices]
    return {
        "both right": sum(a and b for a, b in zip(reference_ok, candidate_ok)) / total,
        "only GNN right": sum(a and not b for a, b in zip(reference_ok, candidate_ok)) / total,
        "only Laya right": sum((not a) and b for a, b in zip(reference_ok, candidate_ok)) / total,
        "neither": sum((not a) and (not b) for a, b in zip(reference_ok, candidate_ok)) / total,
        "n": total,
    }


def build_report(
    gnn_rankings: Sequence[Sequence[str]],
    laya_rankings: Sequence[Sequence[str]],
    targets: Sequence[str],
    *,
    pool_k: int = 10,
    select_k: int = 5,
) -> dict[str, Any]:
    """Build the paired report used by the benchmark CLI."""
    known = [not target.startswith("<") for target in targets]
    in_pool = [known[i] and targets[i] in gnn_rankings[i][:pool_k] for i in range(len(targets))]
    systems = {
        "gnn": gnn_rankings,
        "laya": laya_rankings,
    }
    metrics = {
        name: {
            **evaluate_rankings(ranking, targets, known, ks=(pool_k, select_k, 1), mrr_k=pool_k),
            "conditional": evaluate_rankings(ranking, targets, in_pool, ks=(select_k, 1), mrr_k=pool_k),
        }
        for name, ranking in systems.items()
    }
    return {
        "rows": len(targets),
        "known_targets": sum(known),
        "pool_rows": sum(in_pool),
        "pool_recall": sum(in_pool) / max(sum(known), 1),
        "metrics": metrics,
        "transitions": {
            f"top_{k}": transition_table(gnn_rankings, laya_rankings, targets, known, k=k)
            for k in (1, select_k)
        },
    }


def write_report(report: dict[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return path
