"""Dataset download, validation, and deterministic sampling."""

from __future__ import annotations

from pathlib import Path
from typing import Any

REQUIRED_COLUMNS = frozenset({"row_index", "theorem", "tactic", "text_state"})


def discover_dataset_shards(repo: str, split: str = "test") -> list[str]:
    from huggingface_hub import HfApi
    prefix = f"{split}/"
    files = sorted(HfApi().list_repo_files(repo, repo_type="dataset"))
    shards = [name for name in files if name.startswith(prefix) and name.endswith(".parquet")]
    if not shards:
        raise FileNotFoundError(f"no parquet shards found under {prefix!r} in {repo}")
    return shards


def download_dataset(repo: str, split: str, directory: Path) -> list[Path]:
    from huggingface_hub import hf_hub_download
    directory.mkdir(parents=True, exist_ok=True)
    return [
        Path(hf_hub_download(repo, shard, repo_type="dataset", local_dir=directory))
        for shard in discover_dataset_shards(repo, split)
    ]


def load_dataset(paths: list[Path]):
    import pandas as pd
    if not paths:
        raise ValueError("at least one dataset path is required")
    frame = pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"dataset is missing columns: {sorted(missing)}")
    return frame.drop_duplicates("row_index", keep="first").reset_index(drop=True)


def sample_dataset(frame, rows: int, seed: int):
    if rows < 1:
        raise ValueError("rows must be positive")
    if rows >= len(frame):
        return frame.reset_index(drop=True)
    return frame.sample(n=rows, random_state=seed).sort_index().reset_index(drop=True)
