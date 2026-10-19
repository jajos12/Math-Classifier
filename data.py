"""Dataset download, validation, deterministic sampling, split carving, and audits."""

from __future__ import annotations

import hashlib
import random
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

REQUIRED_COLUMNS = frozenset({"row_index", "theorem", "tactic", "text_state"})


def pool_row_key(value: Any) -> str:
    """Canonical string form of a dataset ``row_index``.

    The same index must produce the same key whether it arrives as an int
    (parquet/JSON), a float-rendered column value (``123.0``), or a string
    with stray whitespace — every per-row lookup (candidate pools, overlap
    checks) depends on that. Returns ``""`` for missing values.
    """
    if value is None:
        return ""
    if type(value).__name__ in {"NAType", "NaTType"}:
        return ""
    if isinstance(value, float) and value != value:
        return ""
    text = str(value).strip()
    if text.lower() in {"nan", "<na>", "none"}:
        return ""
    if text.endswith(".0"):
        head = text[:-2]
        if head.lstrip("-").isdigit():
            return head
    return text


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


def combine_shards(paths: Sequence[Path], output: Path) -> tuple[Path, int]:
    """Concatenate downloaded shards into one file; returns (path, row count)."""
    import pandas as pd

    if not paths:
        raise ValueError("no shards to combine")
    frame = pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    suffix = output.suffix.lower()
    if suffix == ".parquet":
        frame.to_parquet(output, index=False)
    elif suffix == ".csv":
        frame.to_csv(output, index=False)
    elif suffix in {".jsonl", ".json"}:
        frame.to_json(output, orient="records", lines=suffix == ".jsonl")
    else:
        raise ValueError(f"unsupported output format {suffix!r}; use .parquet, .csv or .jsonl")
    return output, len(frame)


def state_fingerprint(text: Any) -> str:
    """Stable content hash of a proof state, for cross-split duplicate detection."""
    if text is None:
        return ""
    if isinstance(text, float) and text != text:  # NaN
        return ""
    cleaned = str(text).strip()
    if not cleaned or cleaned.lower() in {"nan", "<na>", "none"}:
        return ""
    return hashlib.sha1(cleaned.encode("utf-8")).hexdigest()[:16]


def _split_keys(path: Path) -> tuple[int, set[str], set[str], int, int]:
    """Stream one parquet shard into (rows, row_index keys, state hashes, blank keys, blank states)."""
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError(
            "auditing parquet needs pyarrow; install it with `pip install pyarrow`"
        ) from exc
    try:
        file = pq.ParquetFile(path)
        names = set(file.schema_arrow.names)
        missing = {"row_index", "text_state"} - names
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        rows = blank_key = blank_state = 0
        keys: set[str] = set()
        hashes: set[str] = set()
        for batch in file.iter_batches(batch_size=2000, columns=["row_index", "text_state"]):
            indices = batch.column(0).to_pylist()
            states = batch.column(1).to_pylist()
            for raw_key, raw_state in zip(indices, states):
                rows += 1
                key = pool_row_key(raw_key)
                if key:
                    keys.add(key)
                else:
                    blank_key += 1
                digest = state_fingerprint(raw_state)
                if digest:
                    hashes.add(digest)
                else:
                    blank_state += 1
        return rows, keys, hashes, blank_key, blank_state
    except RuntimeError:
        raise
    except Exception as exc:
        raise RuntimeError(f"failed to read parquet dataset {path}") from exc


def audit_splits(splits: Mapping[str, Sequence[Path]]) -> dict[str, Any]:
    """Check splits for duplicate rows, within and across splits.

    Two identities are checked per split: the canonicalized ``row_index`` and a
    content hash of ``text_state``. Any overlap *between* splits is reported as
    a failure with example keys, because a row shared by train and test makes
    every held-out metric meaningless. Returns the report dict with
    ``passed`` set accordingly.
    """
    per_split: dict[str, dict[str, Any]] = {}
    index_sets: dict[str, set[str]] = {}
    hash_sets: dict[str, set[str]] = {}
    for name, paths in splits.items():
        if not paths:
            raise ValueError(f"no parquet shards found for split {name!r}")
        rows = blank_key = blank_state = 0
        keys: set[str] = set()
        digests: set[str] = set()
        for path in paths:
            r, k, h, bk, bs = _split_keys(path)
            rows += r
            keys |= k
            digests |= h
            blank_key += bk
            blank_state += bs
        index_sets[name] = keys
        hash_sets[name] = digests
        per_split[name] = {
            "shards": [str(path) for path in paths],
            "rows": rows,
            "unique_row_index": len(keys),
            "unique_state": len(digests),
            "duplicate_row_index": rows - blank_key - len(keys),
            "duplicate_state": rows - blank_state - len(digests),
            "blank_row_index": blank_key,
            "blank_state": blank_state,
        }

    names = list(splits)
    overlaps: dict[str, dict[str, Any]] = {}
    passed = True
    for position, left in enumerate(names):
        for right in names[position + 1:]:
            shared_keys = sorted(index_sets[left] & index_sets[right])
            shared_hashes = sorted(hash_sets[left] & hash_sets[right])
            if shared_keys or shared_hashes:
                passed = False
            overlaps[f"{left}|{right}"] = {
                "row_index": len(shared_keys),
                "state": len(shared_hashes),
                "row_index_examples": shared_keys[:5],
                "state_examples": shared_hashes[:5],
            }
    return {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "splits": per_split,
        "overlaps": overlaps,
        "passed": passed,
    }


# --------------------------------------------------------------------- splitting


def split_dataset(
    source: Path,
    train_out: Path,
    validation_out: Path,
    *,
    validation_fraction: float = 0.2,
    seed: int = 42,
) -> tuple[int, int]:
    """Carve a disjoint validation split out of ``source``.

    Returns ``(train_rows, validation_rows)``.  Row positions are assigned from
    a seeded shuffle, so the split is reproducible and the two outputs can never
    share a row — the property the ``--eval-data`` pre-flight check relies on.

    Parquet sources are streamed through pyarrow (one batch in memory at a
    time); other formats are read with pandas.
    """
    if not 0 < validation_fraction < 1:
        raise ValueError(
            f"--validation-fraction must be between 0 and 1, got {validation_fraction}"
        )
    source = Path(source)
    train_out = Path(train_out)
    validation_out = Path(validation_out)
    if not source.is_file():
        raise FileNotFoundError(f"dataset not found: {source}")
    resolved = [source.resolve(), train_out.resolve(), validation_out.resolve()]
    if len(set(resolved)) != 3:
        raise ValueError("source, train output and validation output must be different files")

    if source.suffix.lower() == ".parquet":
        total = _parquet_row_count(source)
        positions = _validation_positions(total, validation_fraction, seed)
        return _split_parquet(source, train_out, validation_out, positions)
    return _split_with_pandas(
        source, train_out, validation_out, validation_fraction, seed
    )


def _validation_positions(total: int, fraction: float, seed: int) -> set[int]:
    """Positions routed to validation; both sides always keep at least one row."""
    if total < 2:
        raise ValueError(f"cannot split {total} row(s); the dataset needs at least 2")
    count = min(total - 1, max(1, round(total * fraction)))
    return set(random.Random(seed).sample(range(total), count))


def _parquet_row_count(path: Path) -> int:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError(
            "splitting parquet needs pyarrow; install it with `pip install pyarrow`"
        ) from exc
    try:
        return int(pq.ParquetFile(path).metadata.num_rows)
    except Exception as exc:
        raise RuntimeError(f"failed to read parquet dataset {path}") from exc


def _split_parquet(
    source: Path,
    train_out: Path,
    validation_out: Path,
    validation_positions: set[int],
) -> tuple[int, int]:
    """Stream the source once, appending each row to exactly one output file."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    parquet = pq.ParquetFile(source)
    flush_rows = 32768
    writers: dict[Path, Any] = {}
    buffers: dict[Path, list[Any]] = {}
    counts: dict[Path, int] = {train_out: 0, validation_out: 0}
    position = 0

    def write(path: Path, fragment: Any) -> None:
        if path not in writers:
            path.parent.mkdir(parents=True, exist_ok=True)
            writers[path] = pq.ParquetWriter(path, fragment.schema)
            buffers[path] = []
        buffers[path].append(fragment)
        counts[path] += fragment.num_rows
        if sum(table.num_rows for table in buffers[path]) >= flush_rows:
            writers[path].write_table(pa.concat_tables(buffers[path]))
            buffers[path] = []

    try:
        for batch in parquet.iter_batches(batch_size=2000):
            table = pa.Table.from_batches([batch])
            rows = table.num_rows
            train_index: list[int] = []
            validation_index: list[int] = []
            for offset in range(rows):
                target = (
                    validation_index
                    if position + offset in validation_positions
                    else train_index
                )
                target.append(offset)
            if train_index:
                write(train_out, table.take(pa.array(train_index, type=pa.int64())))
            if validation_index:
                write(
                    validation_out,
                    table.take(pa.array(validation_index, type=pa.int64())),
                )
            position += rows
    finally:
        for path, writer in writers.items():
            if buffers.get(path):
                writer.write_table(pa.concat_tables(buffers[path]))
            writer.close()

    for path in (train_out, validation_out):
        if path not in writers:
            raise RuntimeError(f"split produced no rows for {path}")
    return counts[train_out], counts[validation_out]


def _split_with_pandas(
    source: Path,
    train_out: Path,
    validation_out: Path,
    fraction: float,
    seed: int,
) -> tuple[int, int]:
    import pandas as pd

    suffix = source.suffix.lower()
    if suffix == ".csv":
        frame = pd.read_csv(source)
    elif suffix in {".jsonl", ".json"}:
        frame = pd.read_json(source, lines=suffix == ".jsonl")
    else:
        raise RuntimeError(
            f"reading {suffix} data requires pyarrow; install it with `pip install pyarrow`"
        )
    positions = _validation_positions(len(frame), fraction, seed)
    ordered = sorted(positions)
    validation = frame.iloc[ordered]
    train = frame.drop(frame.index[ordered])
    _write_frame(train, train_out)
    _write_frame(validation, validation_out)
    return len(train), len(validation)


def _write_frame(frame: Any, path: Path) -> None:
    suffix = path.suffix.lower()
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        if suffix == ".parquet":
            frame.to_parquet(path, index=False)
        elif suffix == ".csv":
            frame.to_csv(path, index=False)
        elif suffix in {".jsonl", ".json"}:
            frame.to_json(path, orient="records", lines=suffix == ".jsonl")
        else:
            raise ValueError(
                f"unsupported split output format {suffix!r}; use .parquet, .csv or .jsonl"
            )
    except ImportError as exc:
        raise RuntimeError(
            f"writing {suffix} output needs pyarrow; install it with `pip install pyarrow`"
        ) from exc
