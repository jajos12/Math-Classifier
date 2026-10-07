"""Configuration for the notebook-to-script Laya experiment."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class ExperimentConfig:
    model_repo: str = "jajostrains/Mathlib-Sexpr-GNN"
    dataset_repo: str = "jajostrains/Mathlib-Normalized-Sexpr"
    bundle_name: str = "pointer-gat-gru"
    split: str = "test"
    n_rows: int = 500
    seed: int = 42
    pool_k: int = 10
    select_k: int = 5
    batch_size: int = 64
    laya_model: str = "convaiinnovations/laya"
    laya_instructions: str = (
        "Analyze the Lean proof state and determine which proof action is most useful next. "
        "Use the current goal together with the local hypotheses: look for hypotheses that "
        "can be applied, rewritten, simplified, destructured, or used in arithmetic. "
        "Consider the candidate tactic descriptions as hints about the actions they perform, "
        "then rank the available actions by how well they fit this particular state."
    )
    checkpoint_every: int = 50
    work_dir: Path = field(default_factory=Path.cwd)
    cache_dir: Path | None = None
    output_dir: Path | None = None
    device: str = "auto"
    laya_device: str = "auto"
    laya_mode: str = "standard"
    laya_confidence_threshold: float = 0.0
    run_b1: bool = False
    run_b2: bool = False
    run_multilingual: bool = False
    export_laya: bool = False

    def __post_init__(self) -> None:
        if self.n_rows < 1:
            raise ValueError("n_rows must be positive")
        if self.pool_k < 1:
            raise ValueError("pool_k must be positive")
        if self.select_k < 1 or self.select_k > self.pool_k:
            raise ValueError("select_k must be between 1 and pool_k")
        if self.batch_size < 1 or self.checkpoint_every < 1:
            raise ValueError("batch_size and checkpoint_every must be positive")
        self._validate_device(self.device, "device")
        self._validate_device(self.laya_device, "laya-device")
        if self.laya_mode not in {"standard", "balanced"}:
            raise ValueError("laya-mode must be standard or balanced")
        if not 0.0 <= self.laya_confidence_threshold <= 1.0:
            raise ValueError("laya-confidence-threshold must be between 0 and 1")

    @staticmethod
    def _validate_device(value: str, name: str) -> None:
        if value in {"auto", "cpu"} or value == "cuda":
            return
        if value.startswith("cuda:") and value[5:].isdigit():
            return
        raise ValueError(
            f"invalid {name} {value!r}; use auto, cpu, cuda, or cuda:N "
            "(for example cuda:0 or cuda:1)"
        )

    @property
    def model_dir(self) -> Path:
        return self.work_dir / "weights" / self.model_repo.replace("/", "--")

    @property
    def bundle_dir(self) -> Path:
        return self.model_dir / self.bundle_name

    @property
    def dataset_dir(self) -> Path:
        return self.work_dir / "data" / self.dataset_repo.replace("/", "--")

    @property
    def resolved_cache_dir(self) -> Path:
        return self.cache_dir or self.work_dir / "cache"

    @property
    def resolved_output_dir(self) -> Path:
        return self.output_dir or self.work_dir / "outputs"

    @property
    def resolved_device(self) -> str:
        if self.device != "auto":
            return self.device
        try:
            import torch
        except ImportError:
            return "cpu"
        return "cuda" if torch.cuda.is_available() else "cpu"

    @property
    def resolved_laya_device(self) -> str:
        if self.laya_device != "auto":
            return self.laya_device
        return self.resolved_device

    def prepare_directories(self) -> None:
        for path in (self.model_dir, self.dataset_dir, self.resolved_cache_dir, self.resolved_output_dir):
            path.mkdir(parents=True, exist_ok=True)

    @property
    def cache_path(self) -> Path:
        return self.resolved_cache_dir / f"{self.split}_{self.n_rows}_seed{self.seed}_k{self.pool_k}.json"
