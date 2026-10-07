"""Small command-line entry point for validating the experiment package."""

from __future__ import annotations

import argparse

from .graph import proof_state_to_dag


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Math Classifier GNN/Laya experiment")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("validate", help="run a dependency-free graph smoke test")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "validate":
        dag = proof_state_to_dag("h : α = α\n⊢ α = α")
        if not any(node.label == "State" for node in dag.nodes):
            raise RuntimeError("graph smoke test did not create a State node")
        print(f"graph validation passed: {dag.num_nodes} nodes, {dag.num_edges} edges")
    return 0
