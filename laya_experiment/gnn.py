"""Published GNN bundle loading and prediction helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .graph import DAGBuilder, dag_to_pyg, proof_state_to_dag


def load_bundle(bundle_dir: str | Path, *, verify_hashes: bool = True):
    """Load the notebook-compatible bundle implementation.

    The full checkpoint architecture is kept isolated here so the rest of the
    experiment only depends on ``load_bundle``, ``build_model`` and prediction.
    """
    from .gnn_bundle import load_bundle as _load_bundle
    return _load_bundle(bundle_dir, verify_hashes=verify_hashes)


def build_graphs(states: list[str], bundle, *, edge_mode: str = "bidirectional"):
    return [
        dag_to_pyg(proof_state_to_dag(state), bundle.node_vocab, edge_mode=edge_mode)
        for state in states
    ]


def predict_top_k(model, graphs, bundle, k: int, *, batch_size: int = 64, device: str = "cpu"):
    from .gnn_bundle import predict_probs, topk_predictions
    probabilities = predict_probs(model, graphs, batch_size=batch_size, device=device)
    return topk_predictions(probabilities, bundle, k)
