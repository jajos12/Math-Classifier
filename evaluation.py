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


def write_metrics_image(report: dict[str, Any], path: Path) -> Path:
    """Write a readable visual comparison of the benchmark metrics."""
    import matplotlib.pyplot as plt

    systems = ("gnn", "laya")
    labels = ("GNN", "Laya")
    metric_keys = (
        next(key for key in report["metrics"]["gnn"] if key.startswith("recall@")),
        "recall@5",
        "recall@1",
    )
    colors = ("#4472C4", "#ED7D31")
    figure, axes = plt.subplots(2, 2, figsize=(12, 8))

    values = [
        [float(report["metrics"][system].get(key, 0.0)) for system in systems]
        for key in metric_keys
    ]
    x = list(range(len(metric_keys)))
    width = 0.36
    for index, (label, color) in enumerate(zip(labels, colors)):
        bars = axes[0, 0].bar(
            [position + (index - 0.5) * width for position in x],
            [row[index] for row in values],
            width,
            label=label,
            color=color,
        )
        axes[0, 0].bar_label(bars, fmt="%.3f", padding=2, fontsize=8)
    axes[0, 0].set_xticks(x, metric_keys)
    axes[0, 0].set_ylim(0, 1)
    axes[0, 0].set_ylabel("Fraction of rows")
    axes[0, 0].set_title("Ranking recall")
    axes[0, 0].legend(frameon=False)

    transition = report["transitions"].get("top_1", {})
    transition_labels = ("Both right", "Only GNN", "Only Laya", "Neither")
    transition_keys = ("both right", "only GNN right", "only Laya right", "neither")
    transition_values = [float(transition.get(key, 0.0)) for key in transition_keys]
    bars = axes[0, 1].bar(transition_labels, transition_values, color=("#70AD47", "#4472C4", "#ED7D31", "#A5A5A5"))
    axes[0, 1].bar_label(bars, fmt="%.3f", padding=2, fontsize=8)
    axes[0, 1].set_ylim(0, 1)
    axes[0, 1].set_ylabel("Fraction of known rows")
    axes[0, 1].set_title("Top-1 changes")
    axes[0, 1].tick_params(axis="x", rotation=25)

    conditional_values = [
        float(report["metrics"][system]["conditional"].get("recall@5", 0.0))
        for system in systems
    ]
    bars = axes[1, 0].bar(labels, conditional_values, color=colors)
    axes[1, 0].bar_label(bars, fmt="%.3f", padding=2)
    axes[1, 0].set_ylim(0, 1)
    axes[1, 0].set_ylabel("Fraction of pool rows")
    axes[1, 0].set_title("Conditional recall@5")

    axes[1, 1].axis("off")
    summary = (
        f"Rows evaluated: {report['rows']}\n"
        f"Known targets: {report['known_targets']}\n"
        f"Targets in GNN top-{report.get('pool_k', 10)}: {report['pool_rows']}\n"
        f"Pool recall: {report['pool_recall']:.3f}\n\n"
        "Interpretation:\n"
        "Laya is better when its recall is higher.\n"
        "'Only Laya' shows corrected GNN mistakes.\n"
        "'Only GNN' shows regressions after reranking."
    )
    axes[1, 1].text(0.02, 0.95, summary, va="top", fontsize=12)
    figure.suptitle("GNN versus Laya tactic-ranking evaluation", fontsize=16)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return path


def print_summary(report: dict[str, Any], image_path: Path) -> None:
    """Print a concise, human-readable conclusion after a benchmark."""
    gnn = report["metrics"]["gnn"]
    laya = report["metrics"]["laya"]
    gnn_top1 = float(gnn.get("recall@1", 0.0))
    laya_top1 = float(laya.get("recall@1", 0.0))
    gnn_top5 = float(gnn.get("recall@5", 0.0))
    laya_top5 = float(laya.get("recall@5", 0.0))
    if laya_top1 > gnn_top1:
        conclusion = "Laya performed better than the GNN on final top-1 selection."
    elif laya_top1 < gnn_top1:
        conclusion = "The GNN performed better than Laya on final top-1 selection."
    else:
        conclusion = "GNN and Laya tied on final top-1 selection."
    print("\n" + "=" * 64)
    print("BENCHMARK PERFORMANCE")
    print("=" * 64)
    print(f"Rows evaluated       : {report['rows']}")
    print(f"Known targets        : {report['known_targets']}")
    print(f"Target in GNN pool  : {report['pool_rows']} ({report['pool_recall']:.1%})")
    print(f"GNN recall@5         : {gnn_top5:.3f}")
    print(f"Laya recall@5        : {laya_top5:.3f}")
    print(f"GNN recall@1         : {gnn_top1:.3f}")
    print(f"Laya recall@1        : {laya_top1:.3f}")
    print(f"Top-1 Laya change    : {laya_top1 - gnn_top1:+.3f}")
    print(f"Conclusion           : {conclusion}")
    print(f"Visual report        : {image_path}")
    if report["rows"] < 100:
        print("Note                 : use at least 100 rows for a meaningful comparison.")
    print("=" * 64)
