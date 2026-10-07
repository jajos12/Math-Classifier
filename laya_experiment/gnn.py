"""Published GNN bundle loading and prediction helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .graph import DAGBuilder, dag_to_pyg, proof_state_to_dag


def download_bundle(repo: str, bundle_name: str, directory: Path) -> Path:
    """Download the published bundle and shared vocab files."""
    from huggingface_hub import hf_hub_download

    directory.mkdir(parents=True, exist_ok=True)
    files = [
        f"{bundle_name}/bundle.json",
        f"{bundle_name}/config.json",
        f"{bundle_name}/model.safetensors",
        f"{bundle_name}/scorer.safetensors",
        f"{bundle_name}/summary.json",
        "vocab/node_vocab.json",
        "vocab/tactic_vocab.json",
    ]
    for relative in files:
        hf_hub_download(repo, relative, local_dir=directory)
    return directory / bundle_name


def load_bundle(bundle_dir: str | Path, *, verify_hashes: bool = True):
    from .gnn_bundle import load_bundle as loader
    return loader(bundle_dir, verify_hashes=verify_hashes)


def build_model(bundle, *, strict: bool = True):
    from .gnn_bundle import build_model as builder
    return builder(bundle, strict=strict)


def build_graphs(states: list[str], bundle, *, edge_mode: str = "bidirectional"):
    return [
        dag_to_pyg(proof_state_to_dag(state), bundle.node_vocab, edge_mode=edge_mode)
        for state in states
    ]


def predict_top_k(model, graphs, bundle, k: int, *, batch_size: int = 64, device: str = "cpu"):
    from .gnn_bundle import predict_probs, topk_predictions
    probabilities = predict_probs(model, graphs, batch_size=batch_size, device=device)
    return topk_predictions(probabilities, bundle, k)
