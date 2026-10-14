"""Small command-line entry point for validating the experiment package."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

from comparison import (
    append_history,
    describe_device,
    rerank_states,
    write_environment,
)
from config import ExperimentConfig
from data import download_dataset, load_dataset, sample_dataset
from evaluation import build_report, print_summary, write_metrics_image, write_report
from graph import normalize_tactic, proof_state_to_dag
from gnn import build_graphs, build_model, download_bundle, load_bundle, predict_top_k
from laya_adapter import LayaAdapter


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Math Classifier GNN/Laya experiment")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("validate", help="run a dependency-free graph smoke test")
    subparsers.add_parser("devices", help="list available Torch devices")
    subparsers.add_parser(
        "runtime",
        help="show the Python environment, Torch, CUDA, and visible GPUs",
    )
    benchmark = subparsers.add_parser("benchmark", help="run GNN and Laya ranking evaluation")
    benchmark.add_argument("--rows", type=int, default=500, help="number of rows to evaluate")
    benchmark.add_argument("--seed", type=int, default=42)
    benchmark.add_argument(
        "--device",
        default="auto",
        metavar="DEVICE",
        help="GNN device: auto, cpu, cuda, cuda:0, cuda:1, ... (default: auto)",
    )
    benchmark.add_argument(
        "--laya-device",
        default="auto",
        metavar="DEVICE",
        help="Laya device; defaults to the GNN --device selection",
    )
    benchmark.add_argument(
        "--laya-mode",
        choices=("standard", "balanced"),
        default="standard",
        help="Laya scoring mode; balanced removes candidate-order bias but is slower",
    )
    benchmark.add_argument(
        "--laya-model",
        default="convaiinnovations/laya",
        help="base Laya checkpoint or fine-tuned checkpoint directory to rerank with",
    )
    benchmark.add_argument(
        "--laya-confidence-threshold",
        type=float,
        default=0.0,
        metavar="P",
        help="keep the GNN order when Laya top probability is below P (default: disabled)",
    )
    benchmark.add_argument("--work-dir", type=Path, default=Path.cwd())
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "validate":
        dag = proof_state_to_dag("h : α = α\n⊢ α = α")
        if not any(node.label == "State" for node in dag.nodes):
            raise RuntimeError("graph smoke test did not create a State node")
        print(f"graph validation passed: {dag.num_nodes} nodes, {dag.num_edges} edges")
    elif args.command == "devices":
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("Torch is required to list runtime devices") from exc
        print("cpu")
        if torch.cuda.is_available():
            for index in range(torch.cuda.device_count()):
                print(f"cuda:{index} ({torch.cuda.get_device_name(index)})")
        else:
            print("CUDA is not available")
    elif args.command == "runtime":
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError(
                "Torch is not installed in this Python environment. "
                "Activate the intended environment or select the matching Jupyter kernel."
            ) from exc
        print(f"python: {sys.executable}")
        print(f"python_version: {sys.version.split()[0]}")
        print(f"torch: {torch.__version__}")
        print(f"cuda_available: {torch.cuda.is_available()}")
        print(f"cuda_visible_devices: {os.environ.get('CUDA_VISIBLE_DEVICES', '<all>')}")
        if torch.cuda.is_available():
            print(f"cuda_device_count: {torch.cuda.device_count()}")
            for index in range(torch.cuda.device_count()):
                print(f"cuda:{index}: {torch.cuda.get_device_name(index)}")
    elif args.command == "benchmark":
        config = ExperimentConfig(
            n_rows=args.rows,
            seed=args.seed,
            device=args.device,
            laya_device=args.laya_device,
            laya_mode=args.laya_mode,
            laya_model=args.laya_model,
            laya_confidence_threshold=args.laya_confidence_threshold,
            work_dir=args.work_dir,
        )
        config.prepare_directories()
        from laya_trainer import validate_device

        validate_device(config.resolved_device, label="--device")
        validate_device(config.resolved_laya_device, label="--laya-device")
        paths = download_dataset(config.dataset_repo, config.split, config.dataset_dir)
        frame = sample_dataset(load_dataset(paths), config.n_rows, config.seed)
        states = frame["text_state"].astype(str).tolist()
        targets = [normalize_tactic(value) for value in frame["tactic"].tolist()]
        bundle_dir = download_bundle(config.model_repo, config.bundle_name, config.model_dir)
        bundle = load_bundle(bundle_dir)
        model, _ = build_model(bundle)
        graphs = build_graphs(states, bundle, edge_mode=bundle.config["edge_mode"])
        gnn_rankings, gnn_scores = predict_top_k(
            model, graphs, bundle, config.pool_k,
            batch_size=config.batch_size, device=config.resolved_device,
        )
        adapter = LayaAdapter.load(
            config.laya_model, device=config.resolved_laya_device,
            instructions=config.laya_instructions,
        )
        device_note = describe_device(adapter, config.resolved_laya_device)
        outcome = rerank_states(
            adapter,
            states,
            gnn_rankings,
            mode=config.laya_mode,
            threshold=config.laya_confidence_threshold,
        )
        laya_rankings = outcome["rankings"]
        laya_confidences = outcome["confidences"]
        laya_fallbacks = outcome["fallbacks"]
        del adapter
        report = build_report(
            gnn_rankings, laya_rankings, targets,
            pool_k=config.pool_k, select_k=config.select_k,
        )
        report["kind"] = "benchmark"
        report["split"] = config.split
        report["seed"] = config.seed
        report["n"] = len(states)
        report["pool_k"] = config.pool_k
        report["laya_mode"] = config.laya_mode
        report["laya_model"] = config.laya_model
        report["laya_confidence_threshold"] = config.laya_confidence_threshold
        report["laya_fallbacks"] = laya_fallbacks
        report["laya_device"] = device_note
        report["gnn_device"] = config.resolved_device
        output = write_report(report, config.resolved_output_dir / "metrics.json")
        image = write_metrics_image(report, config.resolved_output_dir / "metrics.png")
        (config.resolved_output_dir / "predictions.json").write_text(
            json.dumps(
                {
                    "targets": targets,
                    "gnn": gnn_rankings,
                    "laya": laya_rankings,
                    "laya_confidences": laya_confidences,
                    "gnn_scores": gnn_scores,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        history = append_history(report, config.resolved_output_dir)
        environment = write_environment(config.resolved_output_dir)
        print(f"benchmark complete: {output}")
        print_summary(report, image)
        if device_note["warning"]:
            print(f"WARNING: {device_note['warning']}")
        print(f"run history    : {history}")
        print(f"environment    : {environment}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
