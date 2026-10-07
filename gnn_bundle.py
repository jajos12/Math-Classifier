"""Published GNN checkpoint backend.

This file is the small, self-contained backend extracted from the notebook.
It intentionally remains separate from the public graph API because checkpoint
loading requires optional Torch and PyG dependencies.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.data import Batch
from torch_geometric.nn import GATv2Conv, global_add_pool, global_max_pool, global_mean_pool
from torch_geometric.utils import softmax as graph_softmax

from graph import NODE_TYPE_TO_ID, root_state_node_index


class StateMeanAttentionReadout(nn.Module):
    def __init__(self, hidden_dim: int, *, include_mean: bool = True, include_max: bool = False) -> None:
        super().__init__()
        self.include_mean = include_mean
        self.include_max = include_max
        self.node_projection = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.state_projection = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.attention_score = nn.Linear(hidden_dim, 1, bias=False)
        count = 2 + int(include_mean) + int(include_max)
        self.fusion = nn.Linear(hidden_dim * count, hidden_dim)
        self.normalization = nn.LayerNorm(hidden_dim)

    def forward(self, node_embeddings, state_embeddings, batch_index):
        graph_count = state_embeddings.size(0)
        state_per_node = state_embeddings.index_select(0, batch_index)
        scores = self.attention_score(torch.tanh(
            self.node_projection(node_embeddings) + self.state_projection(state_per_node)
        )).squeeze(-1)
        weights = graph_softmax(scores, batch_index, num_nodes=graph_count)
        weighted = global_add_pool(weights.unsqueeze(-1) * node_embeddings, batch_index, size=graph_count)
        summaries = [state_embeddings]
        if self.include_mean:
            summaries.append(global_mean_pool(node_embeddings, batch_index, size=graph_count))
        if self.include_max:
            summaries.append(global_max_pool(node_embeddings, batch_index, size=graph_count))
        summaries.append(weighted)
        return F.gelu(self.normalization(self.fusion(torch.cat(summaries, dim=-1)))), weights


class GATv2StateClassifier(nn.Module):
    def __init__(self, *, num_node_labels: int, num_tactics: int,
                 num_node_types: int, hidden_dim: int, num_layers: int,
                 dropout: float, heads: int, use_node_type: bool,
                 readout: str) -> None:
        super().__init__()
        if hidden_dim % heads:
            raise ValueError("hidden_dim must be divisible by heads")
        self.label_embedding = nn.Embedding(num_node_labels, hidden_dim)
        self.node_type_embedding = nn.Embedding(num_node_types, hidden_dim) if use_node_type else None
        self.is_bound_embedding = nn.Embedding(2, hidden_dim)
        self.binder_depth_embedding = nn.Embedding(10, hidden_dim)
        self.binder_kind_embedding = nn.Embedding(6, hidden_dim)
        self.convs = nn.ModuleList(
            GATv2Conv(hidden_dim, hidden_dim // heads, heads=heads, dropout=dropout, concat=True)
            for _ in range(num_layers)
        )
        self.dropout = nn.Dropout(dropout)
        if readout == "state":
            self.global_readout = None
        else:
            self.global_readout = StateMeanAttentionReadout(
                hidden_dim, include_mean="mean" in readout, include_max="max" in readout
            )
        self.classifier = nn.Linear(hidden_dim, num_tactics)

    def encode_nodes(self, data):
        x = self.label_embedding(data.x)
        if self.node_type_embedding is not None:
            x = x + self.node_type_embedding(data.node_type)
        x = x + self.is_bound_embedding(data.is_bound)
        x = x + self.binder_depth_embedding(data.binder_depth.clamp(0, 9))
        x = x + self.binder_kind_embedding(data.binder_kind.clamp(0, 5))
        for conv in self.convs:
            x = self.dropout(F.relu(conv(x, data.edge_index)))
        return x

    def forward(self, data):
        nodes = self.encode_nodes(data)
        indices = data.state_node_index.view(-1).to(nodes.device)
        states = nodes.index_select(0, indices)
        if self.global_readout is not None:
            batch = getattr(data, "batch", torch.zeros(nodes.size(0), dtype=torch.long, device=nodes.device))
            states, _ = self.global_readout(nodes, states, batch)
        return self.classifier(self.dropout(states))


def stable_vocab_sha256(vocab: dict) -> str:
    payload = json.dumps(vocab, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


class Bundle:
    def __init__(self, directory, manifest, config, node_vocab, tactic_vocab):
        self.directory = directory
        self.manifest = manifest
        self.config = config
        self.node_vocab = node_vocab
        self.tactic_vocab = tactic_vocab
        self.id_to_tactic = {int(value): key for key, value in tactic_vocab.items()}


def _vocab_paths(bundle_dir: Path, manifest: dict) -> tuple[Path, Path]:
    if manifest.get("vocab_location", "local") == "shared":
        directory = bundle_dir.parent / "vocab"
    else:
        directory = bundle_dir
    node_path, tactic_path = directory / "node_vocab.json", directory / "tactic_vocab.json"
    if not node_path.is_file() or not tactic_path.is_file():
        raise FileNotFoundError(f"vocabulary files not found near {bundle_dir}")
    return node_path, tactic_path


def load_bundle(bundle_dir: str | Path, *, verify_hashes: bool = True) -> Bundle:
    directory = Path(bundle_dir)
    manifest = json.loads((directory / "bundle.json").read_text())
    config = json.loads((directory / manifest["config"]).read_text())
    node_path, tactic_path = _vocab_paths(directory, manifest)
    node_vocab = json.loads(node_path.read_text())
    tactic_vocab = json.loads(tactic_path.read_text())
    if verify_hashes:
        if sha256_file(directory / manifest["weights"]) != manifest["weights_sha256"]:
            raise ValueError("model weight SHA-256 does not match bundle manifest")
        if stable_vocab_sha256(node_vocab) != manifest["node_vocab_sha256"]:
            raise ValueError("node vocabulary SHA-256 does not match bundle manifest")
        if stable_vocab_sha256(tactic_vocab) != manifest["tactic_vocab_sha256"]:
            raise ValueError("tactic vocabulary SHA-256 does not match bundle manifest")
    return Bundle(directory, manifest, config, node_vocab, tactic_vocab)


def build_model(bundle: Bundle, *, strict: bool = True):
    from safetensors.torch import load_file
    model_cfg = bundle.config["model"]
    model = GATv2StateClassifier(
        num_node_labels=len(bundle.node_vocab), num_tactics=len(bundle.tactic_vocab),
        num_node_types=len(NODE_TYPE_TO_ID), hidden_dim=model_cfg["hidden_dim"],
        num_layers=model_cfg["num_layers"], dropout=model_cfg["dropout"],
        heads=model_cfg["heads"], use_node_type=bundle.config["use_node_type"],
        readout=model_cfg["readout"],
    )
    raw = load_file(str(bundle.directory / manifest_path(bundle, "weights")))
    backbone = {key.removeprefix("backbone."): value for key, value in raw.items() if key.startswith("backbone.")}
    result = model.load_state_dict(backbone, strict=strict)
    if strict and (result.missing_keys or result.unexpected_keys):
        raise ValueError(f"checkpoint mismatch: {result}")
    return model.eval(), sorted({key.split(".", 1)[0] for key in raw if not key.startswith("backbone.")})


def manifest_path(bundle: Bundle, name: str) -> Path:
    return bundle.directory / bundle.manifest[name]


@torch.no_grad()
def predict_probs(model, data_list, *, batch_size: int = 64, device: str = "cpu"):
    model.to(device).eval()
    outputs = []
    for start in range(0, len(data_list), batch_size):
        outputs.append(torch.softmax(model(Batch.from_data_list(data_list[start:start + batch_size]).to(device)), dim=-1).cpu())
    return torch.cat(outputs) if outputs else torch.empty((0, 0))


def topk_predictions(probs, bundle: Bundle, k: int):
    values, indices = torch.topk(probs, min(k, probs.shape[-1]), dim=-1)
    return (
        [[bundle.id_to_tactic[int(index)] for index in row] for row in indices],
        [[float(value) for value in row] for row in values],
    )
