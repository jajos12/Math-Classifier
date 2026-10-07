"""Small command-line entry point for validating the experiment package."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from config import ExperimentConfig
from data import download_dataset, load_dataset, sample_dataset
from evaluation import build_report, write_report
from graph import normalize_tactic, proof_state_to_dag
from gnn import build_graphs, build_model, download_bundle, load_bundle, predict_top_k
from laya_adapter import LayaAdapter


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Math Classifier GNN/Laya experiment")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("validate", help="run a dependency-free graph smoke test")
    benchmark = subparsers.add_parser("benchmark", help="run GNN and Laya ranking evaluation")
    benchmark.add_argument("--rows", type=int, default=500)
    benchmark.add_argument("--seed", type=int, default=42)
    benchmark.add_argument("--device", default="auto")
    benchmark.add_argument("--work-dir", type=Path, default=Path.cwd())
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "validate":
        dag = proof_state_to_dag("h : α = α\n⊢ α = α")
        if not any(node.label == "State" for node in dag.nodes):
            raise RuntimeError("graph smoke test did not create a State node")
        print(f"graph validation passed: {dag.num_nodes} nodes, {dag.num_edges} edges")
    elif args.command == "benchmark":
        config = ExperimentConfig(n_rows=args.rows, seed=args.seed, device=args.device, work_dir=args.work_dir)
        config.prepare_directories()
        paths = download_dataset(config.dataset_repo, config.split, config.dataset_dir)
        frame = sample_dataset(load_dataset(paths), config.n_rows, config.seed)
        bundle_dir = download_bundle(config.model_repo, config.bundle_name, config.model_dir)
        bundle = load_bundle(bundle_dir)
        model, _ = build_model(bundle)
        graphs = build_graphs(frame["text_state"].astype(str).tolist(), bundle,
                              edge_mode=bundle.config["edge_mode"])
        gnn_rankings, gnn_scores = predict_top_k(
            model, graphs, bundle, config.pool_k,
            batch_size=config.batch_size, device=config.resolved_device,
        )
        adapter = LayaAdapter.load(
            config.laya_model, device=config.resolved_laya_device,
            instructions=config.laya_instructions,
        )
        laya_rankings = []
        for state, candidates in zip(frame["text_state"], gnn_rankings):
            laya_rankings.append(adapter.predict(str(state), candidates)["order"])
        targets = frame["tactic"].map(normalize_tactic).tolist()
        report = build_report(
            gnn_rankings, laya_rankings, targets,
            pool_k=config.pool_k, select_k=config.select_k,
        )
        output = write_report(report, config.resolved_output_dir / "metrics.json")
        (config.resolved_output_dir / "predictions.json").write_text(
            json.dumps({"targets": targets, "gnn": gnn_rankings, "laya": laya_rankings, "gnn_scores": gnn_scores}, indent=2),
            encoding="utf-8",
        )
        print(f"benchmark complete: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
