#!/usr/bin/env python3
"""Standalone fusion experiment: choose alpha on validation, report on test.

The post-finetune comparison already reports a fused row for one fixed alpha
(``--fusion-alpha``, default 0.5). This script is the measurement tool behind
that setting: it sweeps the log-linear fusion weight over the validation
split, picks the alpha with the best ``mrr@10``, and then reports GNN vs
Laya-only vs fused on the **test** split. Alpha is chosen on validation, so
the test numbers stay an honest estimate.

Run on the GPU server from the repository root (needs ``laya``, the GNN
bundle, and the dataset splits):

    python3 fuse_eval.py --device cuda:0 --checkpoint outputs/laya-checkpoint

Writes ``outputs/fusion.json`` with the sweep table, the chosen alpha, and the
test metrics. Whole splits are used by default: any ``--val-rows`` /
``--test-rows`` value at or above the split size is clamped to the full split
by ``sample_dataset``, so the defaults measure everything available.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from typing import Any

from config import ExperimentConfig


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="GNN/Laya score-fusion experiment")
    parser.add_argument(
        "--checkpoint",
        default="outputs/laya-checkpoint",
        metavar="PATH",
        help="fine-tuned Laya checkpoint directory (default: outputs/laya-checkpoint)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        default="auto",
        metavar="DEVICE",
        help="GNN device: auto, cpu, cuda, cuda:0, ... (default: auto)",
    )
    parser.add_argument(
        "--laya-device",
        default="auto",
        metavar="DEVICE",
        help="Laya device; defaults to the --device selection",
    )
    parser.add_argument(
        "--laya-mode",
        choices=("standard", "balanced"),
        default="standard",
        help="Laya scoring mode used to produce the probabilities",
    )
    parser.add_argument(
        "--val-rows",
        type=int,
        default=100_000,
        metavar="N",
        help="validation rows for the sweep; values above the split size clamp to the full split (default: full split)",
    )
    parser.add_argument(
        "--test-rows",
        type=int,
        default=100_000,
        metavar="N",
        help="test rows for the report; values above the split size clamp to the full split (default: full split)",
    )
    parser.add_argument(
        "--alphas",
        default="0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0",
        metavar="LIST",
        help="comma-separated alpha values to sweep (default: 0.0 to 1.0 in steps of 0.1)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        return _run(args)
    except (OSError, ValueError, RuntimeError) as exc:
        sys.stdout.flush()
        print(f"error: {exc}", file=sys.stderr)
        cause = getattr(exc, "__cause__", None)
        if cause is not None and str(cause) != str(exc):
            print(f"       caused by {type(cause).__name__}: {cause}", file=sys.stderr)
        return 1


def _run(args: argparse.Namespace) -> int:
    from comparison import load_pool, rerank_states
    from evaluation import evaluate_rankings
    from fusion import fuse_rankings
    from laya_adapter import LayaAdapter

    def config(split: str, rows: int) -> ExperimentConfig:
        return ExperimentConfig(
            split=split,
            n_rows=rows,
            seed=args.seed,
            device=args.device,
            laya_device=args.laya_device,
            laya_mode=args.laya_mode,
        )

    alphas = [float(value) for value in args.alphas.split(",") if value.strip()]
    if not alphas:
        raise ValueError("--alphas produced no values")
    for alpha in alphas:
        if not (0.0 <= alpha <= 1.0):
            raise ValueError(f"alpha {alpha!r} outside [0, 1]")

    def collect(cfg: ExperimentConfig, adapter: Any) -> dict[str, Any]:
        pool = load_pool(cfg)
        started = time.monotonic()
        outcome = rerank_states(adapter, pool["states"], pool["gnn_rankings"], mode=cfg.laya_mode)
        return {
            **pool,
            "probability_maps": outcome["probabilities"],
            "laya_rankings": outcome["rankings"],
            "rerank_seconds": round(time.monotonic() - started, 1),
        }

    def score(pool: dict[str, Any], rankings: list[list[str]]) -> dict[str, float]:
        pool_k = int(pool["pool_k"])
        known = [not target.startswith("<") for target in pool["targets"]]
        return evaluate_rankings(
            rankings, pool["targets"], known, ks=(pool_k, 5, 1), mrr_k=pool_k
        )

    val_config = config("val", args.val_rows)
    test_config = config("test", args.test_rows)
    print(f"loading checkpoint {args.checkpoint} ...", flush=True)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        adapter = LayaAdapter.load(
            args.checkpoint,
            device=val_config.resolved_laya_device,
        )
    for item in caught:
        print(f"warning: {item.message}", file=sys.stderr)

    print("\n=== validation sweep (choose alpha here) ===", flush=True)
    val = collect(val_config, adapter)
    sweep = []
    for alpha in alphas:
        fused = fuse_rankings(
            val["gnn_rankings"], val["gnn_scores"], val["probability_maps"], alpha
        )
        metrics = score(val, fused)
        sweep.append({"alpha": alpha, **metrics})
        print(
            f"  alpha {alpha:.1f}  recall@5 {metrics.get('recall@5', 0.0):.3f} "
            f"recall@1 {metrics.get('recall@1', 0.0):.3f} "
            f"mrr@10 {metrics.get('mrr@10', 0.0):.3f}"
        )
    best = max(sweep, key=lambda entry: (entry["mrr@10"], entry["recall@1"]))
    print(
        f"best alpha on validation: {best['alpha']} (mrr@10 {best['mrr@10']:.3f})"
    )

    print("\n=== test report (fixed alpha) ===", flush=True)
    test = collect(test_config, adapter)
    gnn_metrics = score(test, test["gnn_rankings"])
    laya_metrics = score(test, test["laya_rankings"])
    fused_rankings = fuse_rankings(
        test["gnn_rankings"], test["gnn_scores"], test["probability_maps"], best["alpha"]
    )
    fused_metrics = score(test, fused_rankings)
    oracle_top1 = _oracle_top1(
        test["targets"], [test["gnn_rankings"], test["laya_rankings"]]
    )
    rows = [
        ("GNN", gnn_metrics),
        ("Laya", laya_metrics),
        (f"fused alpha={best['alpha']}", fused_metrics),
    ]
    for name, metrics in rows:
        print(
            f"  {name:<24} recall@10 {metrics.get('recall@10', 0.0):.3f} "
            f"recall@5 {metrics.get('recall@5', 0.0):.3f} "
            f"recall@1 {metrics.get('recall@1', 0.0):.3f} "
            f"mrr@10 {metrics.get('mrr@10', 0.0):.3f}"
        )
    print(f"  {'oracle (per-row best)':<24} recall@1 {oracle_top1:.3f}")
    del adapter

    output = test_config.resolved_output_dir
    output.mkdir(parents=True, exist_ok=True)
    path = output / "fusion.json"
    path.write_text(
        json.dumps(
            {
                "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "checkpoint": args.checkpoint,
                "seed": args.seed,
                "laya_mode": args.laya_mode,
                "val_rows": len(val["states"]),
                "test_rows": len(test["states"]),
                "val_rerank_seconds": val["rerank_seconds"],
                "test_rerank_seconds": test["rerank_seconds"],
                "val_sweep": sweep,
                "chosen_alpha": best["alpha"],
                "test": {
                    "gnn": gnn_metrics,
                    "laya": laya_metrics,
                    "fused": fused_metrics,
                    "oracle_top1": oracle_top1,
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nfusion report: {path}")
    return 0


def _oracle_top1(
    targets: list[str], systems: list[list[list[str]]]
) -> float:
    """Top-1 accuracy of an oracle that may take one candidate from any system."""
    known = [
        index
        for index, target in enumerate(targets)
        if not target.startswith("<")
    ]
    if not known:
        return 0.0
    hits = sum(
        any(system[index][0] == targets[index] for system in systems)
        for index in known
    )
    return hits / len(known)


if __name__ == "__main__":
    raise SystemExit(main())
