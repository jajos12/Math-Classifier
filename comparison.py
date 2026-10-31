"""Paired comparison of a GNN candidate pool reranked by several Laya checkpoints.

The benchmark reranks the GNN top-10 pool with one Laya model.  After a
fine-tune the interesting question is *whether the checkpoint got better*, so
the pool is built once and reranked with every checkpoint in turn: same rows,
same candidates, same seed.  Results land in ``outputs/comparison.json`` and
every run appends one line to ``outputs/history.jsonl``.

Torch, Laya and matplotlib are only imported inside functions, so this module
can be imported (and its helpers unit-tested) without a GPU environment.
"""

from __future__ import annotations

import json
import os
import platform
import re
import subprocess
import sys
import time
import warnings
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from config import ExperimentConfig

#: Below this row count an acceptance gate is statistically weak: a ±4–5
#: point swing is ordinary sampling noise on 100 rows. ``rows_sufficient``
#: records whether the verdict clears it; the intended measurement point is
#: the full test split (~4,400 rows).
MIN_ACCEPTANCE_ROWS = 4000


# --------------------------------------------------------------------- reranking


def rerank_states(
    adapter: Any,
    states: Sequence[str],
    gnn_rankings: Sequence[Sequence[str]],
    *,
    mode: str = "standard",
) -> dict[str, Any]:
    """Rerank one candidate pool with Laya, row by row.

    Laya ranks the pool by itself: its own probability order is always the
    result. The GNN only supplies the candidate set — its order is never kept
    or blended in, so the two systems stay separate here. The probabilities
    are returned alongside the rankings so callers that *want* fusion
    (:func:`fusion.fuse_ranking`, alpha from validation) can combine them with
    the GNN's scores afterwards; plain reranking behaves exactly as before.
    """
    rankings: list[list[str]] = []
    probabilities: list[dict[str, float]] = []
    for state, candidates in zip(states, gnn_rankings):
        if mode == "balanced":
            result = adapter.balanced_predict(state, candidates)
        else:
            result = adapter.predict(state, candidates)
        rankings.append(list(result["order"]))
        probabilities.append(dict(result["probabilities"]))
    return {"rankings": rankings, "probabilities": probabilities}


def describe_device(agent: Any, requested: str) -> dict[str, Any]:
    """Record where Laya actually loaded, and warn about a silent CPU fallback."""
    actual = _agent_device(agent)
    warning = None
    if requested.startswith("cuda") and actual is not None and "cpu" in actual.lower():
        warning = f"Laya fell back to CPU (requested {requested}, using {actual})"
        warnings.warn(warning, RuntimeWarning, stacklevel=2)
    return {"requested": requested, "actual": actual, "warning": warning}


def _agent_device(agent: Any) -> str | None:
    for attribute in ("device", "_device"):
        value = getattr(agent, attribute, None)
        if value:
            return str(value)
    for attribute in ("model", "_model", "decision_model"):
        value = getattr(agent, attribute, None)
        device = getattr(value, "device", None)
        if device:
            return str(device)
    return None


# ------------------------------------------------------------------------- pool


def pool_cache_path(config: ExperimentConfig) -> Path:
    key = "__".join(
        (
            config.dataset_repo,
            f"{config.split}_{config.n_rows}_seed{config.seed}_k{config.pool_k}",
            config.model_repo,
            config.bundle_name,
        )
    )
    safe = re.sub(r"[^A-Za-z0-9._-]+", "-", key)
    return config.resolved_cache_dir / f"pool-{safe}.json"


def _pool_matches(payload: Mapping[str, Any], config: ExperimentConfig) -> bool:
    expected = {
        "dataset_repo": config.dataset_repo,
        "split": config.split,
        "rows": config.n_rows,
        "seed": config.seed,
        "pool_k": config.pool_k,
        "model_repo": config.model_repo,
        "bundle_name": config.bundle_name,
    }
    return all(payload.get(key) == value for key, value in expected.items())


def load_pool(config: ExperimentConfig, *, use_cache: bool = True) -> dict[str, Any]:
    """Build (or reuse) the GNN top-k pool for the configured test rows.

    The cache makes comparisons cheap: the GNN runs once and every checkpoint
    is reranked over the identical candidate lists.
    """
    from data import download_dataset, load_dataset, sample_dataset
    from gnn import build_graphs, build_model, download_bundle, load_bundle, predict_top_k
    from graph import normalize_tactic

    config.prepare_directories()
    cache_path = pool_cache_path(config)
    if use_cache and cache_path.is_file():
        try:
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            payload = None
        if isinstance(payload, dict) and _pool_matches(payload, config):
            payload["cached"] = True
            print(f"  pool cache hit: {cache_path}", flush=True)
            return payload

    paths = download_dataset(config.dataset_repo, config.split, config.dataset_dir)
    frame = sample_dataset(load_dataset(paths), config.n_rows, config.seed)
    states = frame["text_state"].astype(str).tolist()
    bundle = load_bundle(download_bundle(config.model_repo, config.bundle_name, config.model_dir))
    model, _ = build_model(bundle)
    graphs = build_graphs(states, bundle, edge_mode=bundle.config["edge_mode"])
    gnn_rankings, gnn_scores = predict_top_k(
        model,
        graphs,
        bundle,
        config.pool_k,
        batch_size=config.batch_size,
        device=config.resolved_device,
    )
    targets = [normalize_tactic(value) for value in frame["tactic"].tolist()]
    payload = {
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset_repo": config.dataset_repo,
        "split": config.split,
        "rows": config.n_rows,
        "seed": config.seed,
        "pool_k": config.pool_k,
        "model_repo": config.model_repo,
        "bundle_name": config.bundle_name,
        "states": states,
        "targets": targets,
        "gnn_rankings": [list(ranking) for ranking in gnn_rankings],
        "gnn_scores": [[float(score) for score in scores] for scores in gnn_scores],
        "cached": False,
    }
    cache_path.write_text(json.dumps(payload), encoding="utf-8")
    print(f"  pool written to cache: {cache_path}", flush=True)
    return payload


# -------------------------------------------------------------------- comparing


def compare_checkpoints(
    config: ExperimentConfig,
    laya_models: Mapping[str, str],
    *,
    prep_stats: Mapping[str, Any] | None = None,
    use_cache: bool = True,
) -> dict[str, Any]:
    """Rerank one cached GNN pool with every checkpoint and build one report."""
    from evaluation import build_report
    from laya_adapter import LayaAdapter

    if not laya_models:
        raise ValueError("compare_checkpoints needs at least one Laya checkpoint")
    pool = load_pool(config, use_cache=use_cache)
    baseline = build_report(
        pool["gnn_rankings"],
        pool["gnn_rankings"],
        pool["targets"],
        pool_k=config.pool_k,
        select_k=config.select_k,
    )
    entries: dict[str, Any] = {}
    probabilities_by_model: dict[str, list[dict[str, float]]] = {}
    for name, model_id in laya_models.items():
        print(f"  [{name}] loading {model_id} ...", flush=True)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            adapter = LayaAdapter.load(
                model_id,
                device=config.resolved_laya_device,
                instructions=config.laya_instructions,
            )
        device_note = describe_device(adapter, config.resolved_laya_device)
        started = time.monotonic()
        outcome = rerank_states(
            adapter,
            pool["states"],
            pool["gnn_rankings"],
            mode=config.laya_mode,
        )
        elapsed = time.monotonic() - started
        probabilities_by_model[name] = outcome["probabilities"]
        reranked = build_report(
            pool["gnn_rankings"],
            outcome["rankings"],
            pool["targets"],
            pool_k=config.pool_k,
            select_k=config.select_k,
        )
        entries[name] = {
            "model": model_id,
            "laya_device": device_note,
            "load_warnings": [str(item.message) for item in caught],
            "metrics": reranked["metrics"]["laya"],
            "conditional": reranked["metrics"]["laya"].get("conditional", {}),
            "transitions": reranked["transitions"],
            "elapsed_seconds": round(elapsed, 1),
        }
        print(
            f"  [{name}] recall@1 {reranked['metrics']['laya'].get('recall@1', 0.0):.3f} "
            f"recall@{config.pool_k} {reranked['metrics']['laya'].get(f'recall@{config.pool_k}', 0.0):.3f} "
            f"in {elapsed:.1f}s",
            flush=True,
        )
        del adapter
        _release_cuda()

    fuse_rankings = _fused_block(config, pool, entries, probabilities_by_model)
    option_count = _option_count_block(config, pool, prep_stats)
    return {
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "kind": "checkpoint-comparison",
        "rows": pool["rows"],
        "n": len(pool["states"]),
        "seed": config.seed,
        "split": config.split,
        "pool_k": config.pool_k,
        "select_k": config.select_k,
        "laya_mode": config.laya_mode,
        "pool_cached": bool(pool.get("cached")),
        "gnn": {
            "metrics": baseline["metrics"]["gnn"],
            "transitions": baseline["transitions"],
        },
        "models": entries,
        "fusion_alpha": config.fusion_alpha if fuse_rankings else None,
        "option_count": option_count,
        "target_in_pool": baseline["pool_recall"],
        "acceptance": acceptance_block(
            {
                "gnn": {"metrics": baseline["metrics"]["gnn"]},
                "models": entries,
                "n": len(pool["states"]),
            }
        ),
        "environment": environment_record(),
    }


def _fused_block(
    config: ExperimentConfig,
    pool: Mapping[str, Any],
    entries: dict[str, Any],
    probabilities_by_model: Mapping[str, list[dict[str, float]]],
) -> bool:
    """Add the fused ranking as one more system in ``entries``.

    The fused entry combines the GNN's score distribution with the *last*
    loaded Laya checkpoint's probabilities (alpha·log p_laya +
    (1−alpha)·log p_gnn); with a single checkpoint that is the fine-tune. It
    is a pure post-processing of the same pool, so it costs no extra Laya
    calls. Returns whether a fused row was added.
    """
    from evaluation import build_report
    from fusion import fuse_rankings

    alpha = config.fusion_alpha
    probabilities = probabilities_by_model.get("finetuned") or (
        probabilities_by_model[next(iter(probabilities_by_model))] if probabilities_by_model else None
    )
    if alpha is None or probabilities is None:
        return False
    fused_rankings = fuse_rankings(
        pool["gnn_rankings"],
        pool["gnn_scores"],
        probabilities,
        alpha,
    )
    fused_report = build_report(
        pool["gnn_rankings"],
        fused_rankings,
        pool["targets"],
        pool_k=config.pool_k,
        select_k=config.select_k,
    )
    entries["fused"] = {
        "model": f"log-linear fusion alpha={alpha}",
        "metrics": fused_report["metrics"]["laya"],
        "conditional": fused_report["metrics"]["laya"].get("conditional", {}),
        "transitions": fused_report["transitions"],
        "elapsed_seconds": None,
    }
    print(
        f"  [fused] alpha {alpha} "
        f"recall@1 {fused_report['metrics']['laya'].get('recall@1', 0.0):.3f} "
        f"recall@{config.pool_k} {fused_report['metrics']['laya'].get(f'recall@{config.pool_k}', 0.0):.3f}",
        flush=True,
    )
    return True


def acceptance_block(report: Mapping[str, Any]) -> dict[str, Any]:
    """The run's accept/reject verdict on one fixed pool.

    The gates measure the two questions that actually matter:

    * **fused vs GNN** — a fused ranking (GNN scores + Laya probabilities,
      alpha chosen on validation) must beat the GNN itself on ``recall@5``,
      ``recall@1`` and ``mrr@10``; when no fusion is configured the check
      falls back to the fine-tuned checkpoint alone on ``recall@5`` and
      ``mrr@10``, as before.
    * **fine-tuned vs base Laya** — the checkpoint must beat the base model
      on ``recall@1``.

    Each gate records both values so a FAIL line is self-explanatory in
    ``comparison.json`` and ``history.jsonl``. Gates on a few dozen rows are
    noise (±4–5 points on 100 rows), so ``rows_sufficient`` flags runs below
    ``MIN_ACCEPTANCE_ROWS`` instead of letting a one-row difference decide.
    """
    gnn = report.get("gnn", {}).get("metrics", {})
    models = report.get("models", {})
    finetuned = (models.get("finetuned") or {}).get("metrics", {})
    base = (models.get("base") or {}).get("metrics", {})
    fused = (models.get("fused") or {}).get("metrics", {})
    gates: list[dict[str, Any]] = []
    missing: list[str] = []

    def gate(label: str, left: Any, right: Any) -> None:
        gates.append(
            {"gate": label, "left": float(left), "right": float(right), "pass": float(left) > float(right)}
        )

    if fused:
        for key in ("recall@5", "recall@1", "mrr@10"):
            if key in fused and key in gnn:
                gate(f"fused.{key} > gnn.{key}", fused[key], gnn[key])
        if not ({"recall@5", "recall@1", "mrr@10"} <= fused.keys() and {"recall@5", "recall@1", "mrr@10"} <= gnn.keys()):
            missing.append("fused/gnn metrics")
    else:
        for key in ("recall@5", "mrr@10"):
            if key in finetuned and key in gnn:
                gate(f"finetuned.{key} > gnn.{key}", finetuned[key], gnn[key])
        if not ({"recall@5", "mrr@10"} <= finetuned.keys() and {"recall@5", "mrr@10"} <= gnn.keys()):
            missing.append("finetuned/gnn metrics")
    if "recall@1" in finetuned and "recall@1" in base:
        gate(
            "finetuned.recall@1 > base.recall@1",
            finetuned["recall@1"],
            base["recall@1"],
        )
    else:
        missing.append("finetuned/base recall@1")
    rows = report.get("n")
    rows_sufficient = rows is None or rows >= MIN_ACCEPTANCE_ROWS
    return {
        "passed": bool(gates) and not missing and all(item["pass"] for item in gates),
        "complete": not missing,
        "missing": missing,
        "rows": rows,
        "min_rows": MIN_ACCEPTANCE_ROWS,
        "rows_sufficient": rows_sufficient,
        "gates": gates,
        "note": (
            "Accept only when every gate passes and the row count is large "
            "enough: the fused ranking must beat the GNN on recall@5, "
            "recall@1 and mrr@10 (without fusion, the fine-tune itself on "
            "recall@5 and mrr@10), and the fine-tune must beat the base Laya "
            "checkpoint on recall@1 — on the same pool and rows."
        ),
    }


def _option_count_block(
    config: ExperimentConfig,
    pool: Mapping[str, Any],
    prep_stats: Mapping[str, Any] | None,
) -> dict[str, Any]:
    histogram = (prep_stats or {}).get("training_option_counts") or None
    training_counts = sorted(int(key) for key in histogram) if histogram else None
    mismatch = bool(training_counts and training_counts != [config.pool_k])
    return {
        "inference": config.pool_k,
        "training_histogram": histogram,
        "training_option_counts": training_counts,
        "mismatch": mismatch,
        "gold_in_pool": (prep_stats or {}).get("gold_in_pool"),
        "target_source": (prep_stats or {}).get("target_source"),
        "gnn_lambda": (prep_stats or {}).get("gnn_lambda"),
        "mean_gold_mass": (prep_stats or {}).get("mean_gold_mass"),
        "rows_skipped": (prep_stats or {}).get("rows_skipped"),
        "skipped_by_reason": (prep_stats or {}).get("skipped_by_reason"),
        "training_output": (prep_stats or {}).get("output"),
        "note": (
            f"Training records presented {training_counts} options per row while "
            f"inference reranks the GNN top-{config.pool_k} pool; the option counts "
            "differ and are recorded, not corrected (short GNN rankings are kept "
            "as-is, never padded)."
            if mismatch
            else None
        ),
    }


def _release_cuda() -> None:
    try:
        import torch
    except ImportError:
        return
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


# ------------------------------------------------------------------- artifacts


def write_comparison(report: Mapping[str, Any], output_dir: Path) -> tuple[Path, Path]:
    """Write ``comparison.json`` and ``comparison.png``; return both paths."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "comparison.json"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    image_path = write_comparison_image(report, output_dir / "comparison.png")
    return json_path, image_path


def write_comparison_image(report: Mapping[str, Any], path: Path) -> Path:
    """Draw recall per system and a text block with the run's caveats."""
    import matplotlib.pyplot as plt

    systems: list[tuple[str, Mapping[str, Any]]] = [("GNN", report.get("gnn", {}).get("metrics", {}))]
    systems.extend(
        (name, entry.get("metrics", {}))
        for name, entry in report.get("models", {}).items()
    )
    keys = [
        key
        for key in (f"recall@{report.get('pool_k', 10)}", "recall@5", "recall@1")
        if any(key in metrics for _, metrics in systems)
    ]
    colors = ("#4472C4", "#ED7D31", "#70AD47", "#A5A5A5", "#7030A0")
    figure, axes = plt.subplots(1, 2, figsize=(13, 5.5), width_ratios=(3, 2))

    width = 0.8 / max(len(systems), 1)
    for index, (label, metrics) in enumerate(systems):
        values = [float(metrics.get(key, 0.0)) for key in keys]
        bars = axes[0].bar(
            [position + (index - 0.5) * width for position in range(len(keys))],
            values,
            width,
            label=label,
            color=colors[index % len(colors)],
        )
        axes[0].bar_label(bars, fmt="%.3f", padding=2, fontsize=8)
    axes[0].set_xticks(range(len(keys)))
    axes[0].set_xticklabels(keys)
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("Fraction of rows")
    axes[0].set_title(f"Recall over the GNN top-{report.get('pool_k', 10)} pool")
    axes[0].legend(frameon=False)

    option_count = report.get("option_count", {})
    lines = [
        f"Rows evaluated : {report.get('n', report.get('rows', 0))}",
        f"Split / seed   : {report.get('split')} / {report.get('seed')}",
        f"Pool recall    : {report.get('target_in_pool', 0.0):.3f}",
        f"Laya mode      : {report.get('laya_mode')}",
        "",
        "Device notes:",
    ]
    for name, entry in report.get("models", {}).items():
        warning = entry.get("laya_device", {}).get("warning")
        if warning:
            lines.append(f"  {name:<10} WARNING: {warning}")
    if not any(entry.get("laya_device", {}).get("warning") for entry in report.get("models", {}).values()):
        lines.append("  (none)")
    if option_count.get("mismatch"):
        lines.extend(
            (
                "",
                f"Option mismatch: inference reranks {option_count.get('inference')} "
                f"candidates, training used {option_count.get('training_option_counts')}.",
            )
        )
    acceptance = report.get("acceptance")
    if isinstance(acceptance, dict) and acceptance.get("gates"):
        lines.extend(("", f"Acceptance: {'PASS' if acceptance.get('passed') else 'FAIL'}"))
        if acceptance.get("rows_sufficient") is False:
            lines.append(
                f"  note: only {acceptance.get('rows')} rows — below the "
                f"{acceptance.get('min_rows', 4000)}-row bar, read as a smoke test"
            )
        for item in acceptance["gates"]:
            mark = "ok  " if item.get("pass") else "FAIL"
            lines.append(
                f"  {mark} {item['gate']} ({item['left']:.3f} vs {item['right']:.3f})"
            )
    axes[1].axis("off")
    axes[1].text(0.0, 1.0, "\n".join(lines), va="top", fontsize=10, family="monospace")
    figure.suptitle("GNN pool reranked by Laya checkpoints", fontsize=15)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return path


def append_history(report: Mapping[str, Any], output_dir: Path) -> Path:
    """Append one compact line per run so results can be compared over time."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "history.jsonl"
    models = {
        name: {
            "model": entry.get("model"),
            "metrics": {
                key: entry.get("metrics", {}).get(key)
                for key in (f"recall@{report.get('pool_k', 10)}", "recall@5", "recall@1", "mrr@10")
                if entry.get("metrics", {}).get(key) is not None
            },
            "device": entry.get("laya_device", {}).get("actual"),
            "warning": entry.get("laya_device", {}).get("warning"),
            "elapsed_seconds": entry.get("elapsed_seconds"),
        }
        for name, entry in report.get("models", {}).items()
    }
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "kind": report.get("kind", "benchmark"),
        "split": report.get("split"),
        "rows": report.get("n", report.get("rows")),
        "seed": report.get("seed"),
        "pool_k": report.get("pool_k"),
        "select_k": report.get("select_k"),
        "laya_mode": report.get("laya_mode"),
        "gnn_metrics": _metrics_subset(report.get("gnn", {}).get("metrics", {}), report.get("pool_k", 10)),
        "models": models,
        "option_count": report.get("option_count"),
        "acceptance": report.get("acceptance"),
        "environment": report.get("environment"),
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return path


def _metrics_subset(metrics: Mapping[str, Any], pool_k: int) -> dict[str, Any]:
    keys = (f"recall@{pool_k}", "recall@5", "recall@1", "mrr@10", "n")
    return {key: metrics[key] for key in keys if key in metrics}


def write_environment(output_dir: Path) -> Path:
    """Write the current environment snapshot as ``environment.json``."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "environment.json"
    path.write_text(json.dumps(environment_record(), indent=2), encoding="utf-8")
    return path


def environment_record() -> dict[str, Any]:
    """Best-effort snapshot: interpreter, packages, CUDA visibility, git state."""
    record: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "platform": platform.platform(),
        "argv": list(sys.argv),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "packages": {},
    }
    for name in ("torch", "torch_geometric", "laya", "transformers", "pandas", "matplotlib", "tqdm"):
        try:
            module = __import__(name)
        except Exception as exc:  # missing or broken install
            record["packages"][name] = f"unavailable ({type(exc).__name__})"
            continue
        record["packages"][name] = getattr(module, "__version__", "unknown")
    try:
        import torch

        record["cuda_available"] = bool(torch.cuda.is_available())
        if record["cuda_available"]:
            record["cuda_device_count"] = torch.cuda.device_count()
            record["cuda_devices"] = [
                torch.cuda.get_device_name(index)
                for index in range(torch.cuda.device_count())
            ]
    except Exception:
        record["cuda_available"] = False
    record["git"] = _git_state()
    return record


def _git_state() -> dict[str, Any] | None:
    try:
        root = Path(__file__).resolve().parent
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root, capture_output=True, text=True, check=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=root, capture_output=True, text=True, check=True,
            ).stdout.strip()
        )
    except Exception:
        return None
    return {"sha": sha, "dirty": dirty}
