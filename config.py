"""Configuration for the notebook-to-script Laya experiment."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path

from laya_adapter import DEFAULT_INSTRUCTIONS


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
    # One prompt for training and inference: Laya is asked to choose the best
    # tactic, not to rank candidates, and its distribution is its confidence.
    laya_instructions: str = DEFAULT_INSTRUCTIONS
    work_dir: Path = field(default_factory=Path.cwd)
    cache_dir: Path | None = None
    output_dir: Path | None = None
    device: str = "auto"
    laya_device: str = "auto"
    laya_mode: str = "standard"
    # Weight of Laya's probabilities in the fused ranking
    # (alpha·log p_laya + (1−alpha)·log p_gnn) computed by compare_checkpoints;
    # None disables the fused row. Choose the value on validation
    # (fuse_eval.py), not on the measured split.
    fusion_alpha: float | None = 0.5

    def __post_init__(self) -> None:
        if self.n_rows < 1:
            raise ValueError("n_rows must be positive")
        if self.pool_k < 1:
            raise ValueError("pool_k must be positive")
        if self.select_k < 1 or self.select_k > self.pool_k:
            raise ValueError("select_k must be between 1 and pool_k")
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        self._validate_device(self.device, "device")
        self._validate_device(self.laya_device, "laya-device")
        if self.laya_mode not in {"standard", "balanced"}:
            raise ValueError("laya-mode must be standard or balanced")
        if self.fusion_alpha is not None and not (
            math.isfinite(self.fusion_alpha) and 0.0 <= self.fusion_alpha <= 1.0
        ):
            raise ValueError(
                "fusion_alpha must be None or a finite weight in [0, 1], "
                f"got {self.fusion_alpha!r}"
            )

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
