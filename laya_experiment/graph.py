"""Lean proof-state parsing and PyTorch Geometric conversion.

The published checkpoint uses the text-parser graph representation.  This
module intentionally implements that path only; S-expression graphs use a
different vocabulary and are not interchangeable with the checkpoint.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import re
from typing import Iterable

TURNSTILES = ("\u22a2", "|-")
NODE_TYPE_TO_ID = {
    "var": 0, "type": 1, "predicate": 2, "operator": 3, "app": 4,
    "meta": 5, "binder": 6, "const": 7, "sconst": 8, "sbinder": 9, "sapp": 10,
}


@dataclass(frozen=True)
class Hypothesis:
    name: str
    type_expr: str


@dataclass(frozen=True)
class ProofState:
    hypotheses: list[Hypothesis]
    goal: str


def _split_turnstile(state: str) -> tuple[str, str]:
    for turnstile in TURNSTILES:
        if turnstile in state:
            left, right = state.split(turnstile, maxsplit=1)
            return left.strip(), right.strip()
    return "", state.strip()


def parse_state(state: str) -> ProofState:
    """Parse hypotheses and goal from a pretty-printed Lean state."""
    hypothesis_text, goal = _split_turnstile(state)
    hypotheses: list[Hypothesis] = []
    for raw_line in hypothesis_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if " : " in line:
            name, _, type_expr = line.partition(" : ")
        elif ":" in line:
            name, _, type_expr = line.partition(":")
        else:
            name, type_expr = line, "Prop"
        hypotheses.append(Hypothesis(name.strip(), type_expr.strip()))
    if not goal:
        raise ValueError("proof state has an empty goal")
    return ProofState(hypotheses, goal)


_TOKEN_RE = re.compile(
    r"(?P<LPAREN>\()|(?P<RPAREN>\))|(?P<ARROW>→|->)|(?P<COMMA>,)|"
    r"(?P<AT>@)|(?P<COLON>:)|"
    r"(?P<IDENT>[^\s()→@\[\]\u27e8\u27e9,;:]+)"
)


class ExprParser:
    _BINDERS = frozenset({"∀", "∃", "λ", "let"})

    def __init__(self, dag: DAGBuilder) -> None:
        self.dag = dag
        self.tokens: list[tuple[str, str]] = []
        self.position = 0

    def parse(self, expression: str) -> int:
        self.tokens = [(m.lastgroup or "", m.group()) for m in _TOKEN_RE.finditer(expression)]
        self.position = 0
        return self._parse_arrow()

    def _peek(self) -> tuple[str, str] | None:
        return self.tokens[self.position] if self.position < len(self.tokens) else None

    def _consume(self) -> tuple[str, str]:
        token = self._peek()
        if token is None:
            raise ValueError("unexpected end of expression")
        self.position += 1
        return token

    def _parse_arrow(self) -> int:
        left = self._parse_application()
        while self._peek() and self._peek()[0] == "ARROW":
            self._consume()
            left = self.dag.get_or_create("Arrow", (left, self._parse_application()))
        return left

    def _parse_application(self) -> int:
        node = self._parse_atom()
        if node is None:
            return self.dag.get_or_create("?", ())
        label = self.dag.nodes[node].label
        if label in self._BINDERS and self._peek() and self._peek()[0] == "LPAREN":
            declaration = self._parse_atom()
            if declaration is not None:
                node = self.dag.get_or_create("App", (node, declaration))
                if self._peek() and self._peek()[0] == "COMMA":
                    self._consume()
                    node = self.dag.get_or_create("App", (node, self._parse_arrow()))
                    return node
        while True:
            argument = self._parse_atom()
            if argument is None:
                return node
            node = self.dag.get_or_create("App", (node, argument))

    def _parse_atom(self) -> int | None:
        token = self._peek()
        if token is None:
            return None
        kind, value = token
        if kind == "LPAREN":
            self._consume()
            node = self._parse_arrow()
            if self._peek() and self._peek()[0] == "RPAREN":
                self._consume()
            return node
        if kind == "AT":
            self._consume()
            inner = self._parse_atom()
            return self.dag.get_or_create("Explicit", (inner,)) if inner is not None else self.dag.get_or_create("@", ())
        if kind in {"IDENT", "COLON"}:
            self._consume()
            return self.dag.get_or_create(value, ())
        return None


def _node_type(label: str) -> str:
    if label in {"App", "Arrow", "Forall", "Explicit"}:
        return "app"
    if label in {"Hyp", "Goal", "State"}:
        return "meta"
    if label.startswith(":"):
        return "sbinder" if label in {":forall", ":lambda", ":let"} else "sconst"
    if label in {"+", "-", "*", "/", "=", "≤", "≥", "<", ">", "∧", "∨", "¬"}:
        return "operator"
    if label and label[0].isupper():
        return "type" if len(label) <= 2 else "predicate"
    return "var"


@dataclass(frozen=True)
class GraphNode:
    id: int
    label: str
    node_type: str
    children: tuple[int, ...] = ()


@dataclass
class DAGBuilder:
    nodes: list[GraphNode] = field(default_factory=list)
    edges: list[tuple[int, int]] = field(default_factory=list)
    _memo: dict[tuple[str, str, tuple[int, ...]], int] = field(default_factory=dict)

    def get_or_create(self, label: str, children: tuple[int, ...]) -> int:
        node_type = _node_type(label)
        key = (label, node_type, children)
        if key in self._memo:
            return self._memo[key]
        node_id = len(self.nodes)
        self.nodes.append(GraphNode(node_id, label, node_type, children))
        self.edges.extend((child, node_id) for child in children)
        self._memo[key] = node_id
        return node_id

    @property
    def num_nodes(self) -> int:
        return len(self.nodes)

    @property
    def num_edges(self) -> int:
        return len(self.edges)


def proof_state_to_dag(state: str) -> DAGBuilder:
    parsed = parse_state(state)
    dag = DAGBuilder()
    parser = ExprParser(dag)
    roots: list[int] = []
    for hypothesis in parsed.hypotheses:
        name = dag.get_or_create(hypothesis.name, ())
        type_node = parser.parse(hypothesis.type_expr) if hypothesis.type_expr else dag.get_or_create("?", ())
        roots.append(dag.get_or_create("Hyp", (name, type_node)))
    roots.append(dag.get_or_create("Goal", (parser.parse(parsed.goal),)))
    dag.get_or_create("State", tuple(roots))
    return dag


def root_state_node_index(dag: DAGBuilder) -> int:
    state_ids = {node.id for node in dag.nodes if node.label == "State"}
    child_ids = {child for child, _ in dag.edges}
    roots = sorted(state_ids - child_ids)
    if len(roots) != 1:
        raise ValueError(f"expected one root State node, found {len(roots)}")
    return roots[0]


def transform_edge_index(edge_index, edge_mode: str = "bidirectional"):
    import torch
    if edge_mode == "forward":
        return edge_index.to(dtype=torch.long).contiguous()
    if edge_mode != "bidirectional":
        raise ValueError(f"unsupported edge mode: {edge_mode}")
    if edge_index.numel() == 0:
        return edge_index.to(dtype=torch.long).contiguous()
    return torch.cat([edge_index, edge_index[[1, 0], :]], dim=1).contiguous()


def dag_to_pyg(dag: DAGBuilder, vocab: dict[str, int], *, edge_mode: str = "bidirectional"):
    """Convert a DAG into the feature fields expected by the GNN."""
    import torch
    from torch_geometric.data import Data

    edges = list(dict.fromkeys(dag.edges))
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous() if edges else torch.zeros((2, 0), dtype=torch.long)
    data = Data(
        x=torch.tensor([vocab.get(node.label, 0) for node in dag.nodes], dtype=torch.long),
        edge_index=transform_edge_index(edge_index, edge_mode),
        node_type=torch.tensor([NODE_TYPE_TO_ID.get(node.node_type, 5) for node in dag.nodes], dtype=torch.long),
        is_bound=torch.zeros(dag.num_nodes, dtype=torch.long),
        binder_depth=torch.zeros(dag.num_nodes, dtype=torch.long),
        binder_kind=torch.zeros(dag.num_nodes, dtype=torch.long),
        num_nodes=dag.num_nodes,
    )
    data.state_node_index = torch.tensor([root_state_node_index(dag)], dtype=torch.long)
    return data


def unknown_label_rate(dag: DAGBuilder, vocab: dict[str, int]) -> float:
    if not dag.nodes:
        return 0.0
    return sum(node.label not in vocab for node in dag.nodes) / len(dag.nodes)
