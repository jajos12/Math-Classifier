"""Prepare and launch text-only Laya retraining jobs.

This module deliberately does not build PyG graphs. It translates Lean proof
states into Laya-readable text before creating choice-training records. A
remote server can supply its installed Laya training callable through
``trainer=`` or the CLI's ``--trainer module:function`` option.

Preparation is streaming: parquet is read in chunks and JSONL is written row by
row, so a 200k-row dataset never has to exist in memory twice. Rows that cannot
be translated are skipped and counted per reason instead of aborting the run
(``--strict`` restores the old fail-fast behaviour).
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import sys
import time
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from laya_adapter import (
    DEFAULT_INSTRUCTIONS,
    choice_question,
    translate_lean_state,
    translate_tactic_step,
)
from graph import normalize_tactic
from progress import banner, bar, format_elapsed
from tactic_choose import TACTIC_SET, validate_tactic_names

LAYA_RETRAIN_INSTRUCTIONS = DEFAULT_INSTRUCTIONS
PARQUET_CHUNK_ROWS = 2000
SAMPLE_PREVIEW_ROWS = 200


class InvalidRowError(ValueError):
    """A row that cannot become a training record, tagged with a skip reason."""

    def __init__(self, reason: str, message: str) -> None:
        super().__init__(message)
        self.reason = reason


@dataclass(frozen=True)
class RetrainConfig:
    dataset_path: Path
    output_path: Path
    candidates: tuple[str, ...] = ()
    state_column: str = "text_state"
    tactic_column: str = "tactic"
    candidate_column: str | None = "candidates"
    instructions: str = LAYA_RETRAIN_INSTRUCTIONS
    split_name: str = "train"
    device: str = "cuda:0"
    strict: bool = False

    def __post_init__(self) -> None:
        if self.device not in {"auto", "cpu", "cuda"} and not (
            self.device.startswith("cuda:") and self.device[5:].isdigit()
        ):
            raise ValueError(
                f"invalid device {self.device!r}; use auto, cpu, cuda, or cuda:N"
            )

    @property
    def stats_path(self) -> Path:
        """Sidecar recording how many rows were usable and what they look like."""
        return self.output_path.parent / (self.output_path.stem + ".stats.json")


@dataclass
class PrepStats:
    """Counters collected while translating rows, reported after preparation."""

    rows_seen: int = 0
    written: int = 0
    target_inserted: int = 0
    candidate_counts: dict[int, int] = field(default_factory=dict)
    skipped: dict[str, int] = field(default_factory=dict)
    skip_examples: dict[str, str] = field(default_factory=dict)
    first_record: dict[str, Any] | None = None
    elapsed: float = 0.0

    def skip(self, reason: str, detail: str) -> None:
        self.skipped[reason] = self.skipped.get(reason, 0) + 1
        self.skip_examples.setdefault(reason, detail)

    def accept(self, record: Mapping[str, Any]) -> None:
        self.written += 1
        metadata = record.get("metadata", {})
        if metadata.get("target_inserted"):
            self.target_inserted += 1
        count = int(metadata.get("candidate_count", 0))
        self.candidate_counts[count] = self.candidate_counts.get(count, 0) + 1
        if self.first_record is None:
            self.first_record = dict(record)

    @property
    def total_skipped(self) -> int:
        return sum(self.skipped.values())

    @property
    def training_option_counts(self) -> dict[str, int]:
        return {str(key): value for key, value in sorted(self.candidate_counts.items())}

    def summary_lines(self) -> list[str]:
        lines = [f"  rows read      : {self.rows_seen}"]
        lines.append(f"  records written: {self.written}")
        if self.total_skipped:
            detail = ", ".join(
                f"{reason}={count}" for reason, count in
                sorted(self.skipped.items(), key=lambda item: (-item[1], item[0]))
            )
            lines.append(f"  rows skipped   : {self.total_skipped} ({detail})")
        if self.target_inserted:
            lines.append(
                f"  target inserted: {self.target_inserted} rows gained the gold tactic "
                "as an extra candidate"
            )
        if self.candidate_counts:
            counts = ", ".join(
                f"{count} options x{rows}"
                for count, rows in sorted(self.candidate_counts.items())
            )
            lines.append(f"  options per row: {counts}")
        return lines

    def to_dict(self, config: "RetrainConfig") -> dict[str, Any]:
        counts = self.training_option_counts
        return {
            "dataset": str(config.dataset_path),
            "output": str(config.output_path),
            "split": config.split_name,
            "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "rows_seen": self.rows_seen,
            "records_written": self.written,
            "rows_skipped": self.total_skipped,
            "skipped_by_reason": dict(sorted(self.skipped.items())),
            "skip_examples": dict(sorted(self.skip_examples.items())),
            "target_inserted": self.target_inserted,
            "training_option_counts": counts,
            "candidates_configured": list(config.candidates),
            "elapsed_seconds": round(self.elapsed, 2),
            "note": (
                "Training records use the fixed candidate catalog above; the benchmark "
                "reranks the GNN top-10. Compare training_option_counts with "
                "inference_pool_k in outputs/comparison.json to see the mismatch."
            ),
        }


def _is_missing(value: Any) -> bool:
    """True for None, NaN and pandas' NA, without importing pandas."""
    if value is None:
        return True
    if type(value).__name__ in {"NAType", "NaTType"}:
        return True
    if isinstance(value, float) and value != value:  # NaN, including numpy float64
        return True
    return False


def _require_text(value: Any, column: str, reason: str) -> str:
    if _is_missing(value):
        raise InvalidRowError(f"{reason}_missing", f"dataset row has no {column!r} value")
    text = str(value).strip()
    if not text or text.lower() in {"nan", "<na>", "none"}:
        raise InvalidRowError(f"{reason}_empty", f"dataset row has an empty {column!r} value")
    return text


def _parse_candidates(value: Any) -> list[str] | None:
    if _is_missing(value):
        return None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return [item.strip() for item in text.split(",") if item.strip()]
        value = parsed
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [str(item) for item in value]
    raise InvalidRowError(
        "candidates_invalid",
        "candidate options must be a list or a JSON/comma-separated string",
    )


def _row_candidates(
    row: Mapping[str, Any],
    *,
    target: str,
    configured: Sequence[str],
    candidate_column: str | None,
) -> tuple[tuple[str, ...], bool]:
    """Return the candidate pool and whether the gold tactic had to be inserted."""
    candidates = _parse_candidates(row.get(candidate_column)) if candidate_column else None
    selected = list(candidates or configured)
    if not selected:
        raise InvalidRowError(
            "candidates_missing",
            "no candidate pool was supplied; add a candidates column or pass --candidates",
        )
    try:
        selected = list(validate_tactic_names(selected))
    except ValueError as exc:
        raise InvalidRowError("candidates_invalid", str(exc)) from exc
    if target not in TACTIC_SET:
        raise InvalidRowError(
            "tactic_unknown",
            f"tactic family {target!r} is not in the canonical catalog "
            "(see tactic_choose.py)",
        )
    if target in selected:
        return tuple(selected), False
    selected.insert(0, target)
    return tuple(selected), True


def build_training_record(
    row: Mapping[str, Any],
    *,
    candidates: Sequence[str] = (),
    state_column: str = "text_state",
    tactic_column: str = "tactic",
    candidate_column: str | None = "candidates",
    instructions: str = LAYA_RETRAIN_INSTRUCTIONS,
    split_name: str = "train",
) -> dict[str, Any]:
    """Translate one dataset row into an official-style Laya choice record.

    Raises :class:`InvalidRowError` (a ``ValueError``) when the row cannot be
    translated; callers decide whether that skips the row or stops the run.
    """
    raw_state = _require_text(row.get(state_column), state_column, "state")
    raw_tactic = _require_text(row.get(tactic_column), tactic_column, "tactic")
    target = normalize_tactic(raw_tactic)
    if not target or target.startswith("<"):
        raise InvalidRowError(
            "tactic_empty",
            f"dataset row has no usable tactic in {tactic_column!r}: {raw_tactic!r}",
        )
    option_list, inserted = _row_candidates(
        row,
        target=target,
        configured=candidates,
        candidate_column=candidate_column,
    )
    try:
        translated_state = translate_lean_state(raw_state)
    except InvalidRowError:
        raise
    except (ValueError, TypeError) as exc:
        raise InvalidRowError("state_invalid", str(exc)) from exc
    arguments = raw_tactic[len(target):].strip() if raw_tactic.startswith(target) else ""
    return {
        "state": translated_state,
        "raw_state": raw_state,
        "questions": {
            "tactic": choice_question(option_list, instructions, raw_state)["tactic"]
        },
        "expected": {
            "tactic": target,
            "application": raw_tactic,
            "description": translate_tactic_step(target, arguments),
        },
        "metadata": {
            "candidate_count": len(option_list),
            "candidate_order": list(option_list),
            "target_inserted": inserted,
            "prompt_version": "lean-tactic-family-v2",
        },
        "tags": ["atp", "laya-retrain", split_name],
        "row_index": row.get("row_index"),
    }


def iter_rows(path: Path) -> Iterator[dict[str, Any]]:
    """Yield dataset rows one at a time, never materialising the whole file."""
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        try:
            import pandas as pd
        except ImportError as exc:
            raise RuntimeError("Install pandas and pyarrow to read parquet training data") from exc
        for chunk in pd.read_parquet(path, chunksize=PARQUET_CHUNK_ROWS):
            yield from chunk.to_dict("records")
        return
    if suffix == ".csv":
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                yield row
        return
    if suffix == ".jsonl":
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)
        return
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError("JSON training data must contain a list of rows")
        yield from payload
        return
    raise ValueError(f"Unsupported dataset format: {path.suffix}")


def load_rows(path: Path) -> list[dict[str, Any]]:
    """Load JSONL, JSON, CSV, or parquet rows without converting to graphs."""
    return list(iter_rows(path))


def _build_record(row: Mapping[str, Any], config: RetrainConfig) -> dict[str, Any]:
    return build_training_record(
        row,
        candidates=config.candidates,
        state_column=config.state_column,
        tactic_column=config.tactic_column,
        candidate_column=config.candidate_column,
        instructions=config.instructions,
        split_name=config.split_name,
    )


def iter_training_records(
    rows: Iterable[Mapping[str, Any]],
    *,
    config: RetrainConfig,
    stats: PrepStats | None = None,
    progress: bool = True,
    total: int | None = None,
) -> Iterator[dict[str, Any]]:
    """Translate rows, skipping (and counting) the ones that cannot be used."""
    stats = stats if stats is not None else PrepStats()
    progress_bar = bar(total=total, desc="translate", disable=not progress, file=sys.stderr)
    try:
        for row in rows:
            stats.rows_seen += 1
            progress_bar.update(1)
            try:
                record = _build_record(row, config)
            except InvalidRowError as exc:
                if config.strict:
                    raise ValueError(
                        f"invalid training row {stats.rows_seen}: {exc}"
                    ) from exc
                stats.skip(exc.reason, str(exc))
                continue
            except (ValueError, TypeError) as exc:
                if config.strict:
                    raise ValueError(
                        f"invalid training row {stats.rows_seen}: {exc}"
                    ) from exc
                stats.skip("state_invalid", str(exc))
                continue
            stats.accept(record)
            yield record
    finally:
        progress_bar.close()


def prepare_training_records(
    rows: Iterable[Mapping[str, Any]],
    *,
    config: RetrainConfig,
    stats: PrepStats | None = None,
    progress: bool = False,
) -> list[dict[str, Any]]:
    """Build every usable record into a list (kept for API compatibility)."""
    records = list(iter_training_records(rows, config=config, stats=stats, progress=progress))
    if not records:
        raise ValueError("the training dataset produced no records")
    return records


def write_training_jsonl(
    records: Iterable[Mapping[str, Any]], output_path: Path
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return output_path


def write_stats(stats: PrepStats, config: RetrainConfig) -> Path:
    path = config.stats_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(stats.to_dict(config), indent=2), encoding="utf-8")
    return path


def read_stats(output_path: Path) -> dict[str, Any] | None:
    """Load the sidecar written by :func:`write_stats`, if it exists."""
    path = output_path.parent / (output_path.stem + ".stats.json")
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


class PreparedRecords(Sequence[Mapping[str, Any]]):
    """Lazily replays the written JSONL, so trainers never need a second copy."""

    def __init__(self, path: Path, count: int) -> None:
        self.path = path
        self.count = count

    def __len__(self) -> int:
        return self.count

    def __iter__(self) -> Iterator[Mapping[str, Any]]:
        with self.path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)

    def __getitem__(self, index: int | slice) -> Any:
        items = list(self)
        return items[index]


def prepare_dataset(config: RetrainConfig, stats: PrepStats | None = None) -> Path:
    """Translate a dataset and write the text-only records consumed by Laya."""
    return retrain_laya(config, stats=stats, progress=True)


def load_callable(spec: str) -> Callable[..., Any]:
    """Load a remote-server trainer as ``module:function``."""
    module_name, separator, function_name = spec.partition(":")
    if not separator or not module_name or not function_name:
        raise ValueError("trainer must have the form module:function")
    trainer = getattr(importlib.import_module(module_name), function_name, None)
    if not callable(trainer):
        raise TypeError(f"trainer {spec!r} is not callable")
    return trainer


def retrain_laya(
    config: RetrainConfig,
    *,
    trainer: Callable[..., Any] | None = None,
    trainer_kwargs: Mapping[str, Any] | None = None,
    stats: PrepStats | None = None,
    rows: Iterable[Mapping[str, Any]] | None = None,
    total: int | None = None,
    progress: bool = True,
    on_prepared: Callable[[PrepStats, Path], None] | None = None,
) -> Any:
    """Translate the dataset to JSONL, then optionally hand it to a trainer.

    The trainer callable receives ``records`` (a lazy view of the written
    JSONL) and keyword arguments, which keeps the data preparation stable
    across Laya releases whose fine-tuning APIs differ. With no callable this
    only writes the translated JSONL dataset and returns its path.

    ``on_prepared`` runs as soon as the file is complete and before training
    starts, so a summary of the data being trained on is never interleaved
    with training output.
    """
    stats = stats if stats is not None else PrepStats()
    started = time.monotonic()
    source = rows if rows is not None else iter_rows(config.dataset_path)
    records = iter_training_records(
        source, config=config, stats=stats, progress=progress, total=total
    )
    output_path = write_training_jsonl(records, config.output_path)
    stats.elapsed = time.monotonic() - started
    if stats.written == 0:
        output_path.unlink(missing_ok=True)
        raise ValueError(
            "the training dataset produced no records"
            + (f" (skipped all {stats.rows_seen} rows: {stats.skipped})" if stats.rows_seen else "")
        )
    if on_prepared is not None:
        on_prepared(stats, output_path)
    if trainer is None:
        return output_path
    kwargs = dict(trainer_kwargs or {})
    kwargs.setdefault("device", _resolve_device(config.device))
    return trainer(
        records=PreparedRecords(output_path, stats.written),
        output_path=str(output_path),
        **kwargs,
    )


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch
    except ImportError:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


# ------------------------------------------------------------------------- pre-flight


def _estimate_rows(path: Path) -> int | None:
    """Row count for the common formats; None when it cannot be determined cheaply."""
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        try:
            import pyarrow.parquet as pq
        except ImportError:
            return None
        try:
            return int(pq.ParquetFile(path).metadata.num_rows)
        except Exception:
            return None
    if suffix == ".jsonl":
        return _count_lines(path)
    if suffix == ".csv":
        count = _count_lines(path)
        return max(0, count - 1) if count else None
    if suffix == ".json":
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None
        return len(payload) if isinstance(payload, list) else None
    return None


def _count_lines(path: Path) -> int | None:
    try:
        total = 0
        with path.open("rb") as handle:
            while True:
                chunk = handle.read(1 << 22)
                if not chunk:
                    break
                total += chunk.count(b"\n")
        return total
    except OSError:
        return None


def _check_eval_data(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(
            f"--eval-data {path} does not exist.\n"
            "Prepare it first, for example:\n"
            f"  python3 laya_retrain.py data/validation.parquet {path} "
            "--candidates <same list as the training run>"
        )
    if path.stat().st_size == 0:
        raise ValueError(f"--eval-data {path} is empty")
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"--eval-data {path} is not valid JSONL: {exc}") from exc
            if not isinstance(record, dict) or "state" not in record or "questions" not in record:
                raise ValueError(
                    f"--eval-data {path} does not look like prepared Laya records "
                    "(expected 'state' and 'questions' keys in each line)"
                )
            if "expected" not in record and "gold" not in record:
                raise ValueError(
                    f"--eval-data {path} has no 'expected' entry, so Laya cannot score it"
                )
            return


def _sample_rows(path: Path, limit: int = SAMPLE_PREVIEW_ROWS) -> tuple[list[dict[str, Any]], int | None]:
    """Take a handful of rows for sizing, and the total row count when cheap."""
    import itertools

    total = _estimate_rows(path)
    return list(itertools.islice(iter_rows(path), limit)), total


def preflight(
    args: argparse.Namespace,
    config: RetrainConfig,
    *,
    training: bool,
) -> tuple[int | None, int | None]:
    """Validate inputs before any long work; returns (row count, estimated bytes)."""
    banner("pre-flight checks")
    if not config.dataset_path.is_file():
        raise FileNotFoundError(f"dataset not found: {config.dataset_path}")
    if args.eval_data is not None:
        _check_eval_data(args.eval_data)
        print(f"  eval data     : {args.eval_data} ok")
    if training:
        from laya_trainer import prepare_device

        effective = prepare_device(args.device, label="--device")
        note = f" (visible as {effective})" if effective != args.device else ""
        print(f"  device        : {args.device} ok{note}")
    sample, total = _sample_rows(config.dataset_path)
    estimated_bytes: int | None = None
    if sample:
        prepared: list[dict[str, Any]] = []
        unusable: dict[str, str] = {}
        for row in sample:
            try:
                prepared.append(_build_record(row, config))
            except ValueError as exc:
                reason = getattr(exc, "reason", "invalid")
                unusable.setdefault(reason, str(exc))
        if not prepared:
            reasons = "; ".join(f"{key}: {value}" for key, value in unusable.items())
            raise ValueError(
                f"none of the {len(sample)} sampled rows can be translated ({reasons}). "
                "Check --state-column/--tactic-column and the candidate list."
            )
        if unusable:
            print(
                f"  sample        : {len(prepared)}/{len(sample)} sampled rows usable; "
                f"unusable ({', '.join(unusable)}) will be skipped"
                + (" — the run stops at the first one (--strict)" if config.strict else "")
            )
            if not config.strict:
                for reason, message in list(unusable.items())[:2]:
                    print(f"                  example ({reason}): {message}")
        average = sum(len(json.dumps(record, ensure_ascii=False)) for record in prepared) / len(prepared)
        counts: dict[int, int] = {}
        for record in prepared:
            count = int(record["metadata"]["candidate_count"])
            counts[count] = counts.get(count, 0) + 1
        option_note = ", ".join(f"{count} options" for count in sorted(counts))
        if not unusable:
            print(f"  sample        : {len(prepared)} records previewed ({option_note})")
        else:
            print(f"  candidates    : {option_note}")
        if total:
            estimated_bytes = int(average * total)
            print(
                f"  dataset       : {config.dataset_path.name} — {total:,} rows, "
                f"~{_human_bytes(estimated_bytes)} of JSONL"
            )
            if estimated_bytes > 2 * 1024 ** 3:
                print(
                    "  note          : fine-tuning reads that file back into memory; "
                    "consider --limit for the first run"
                )
        else:
            print(f"  dataset       : {config.dataset_path.name} (row count unknown)")
    if args.sample and total is None:
        raise ValueError(
            "--sample needs a row count; it could not be determined for this format. "
            "Use --limit, or pre-split the dataset."
        )
    return total, estimated_bytes


def _human_bytes(size: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return f"{size:.1f} {unit}" if unit != "B" else f"{int(size)} B"
        size /= 1024
    return f"{size:.1f} TB"


def _selected_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    total: int | None,
    sample: int | None,
    limit: int | None,
    seed: int,
) -> tuple[Iterable[Mapping[str, Any]], int | None]:
    """Apply --limit / --sample to a stream without loading it."""
    if sample is not None:
        import random

        if total is None:
            raise ValueError("--sample requires a known row count")
        chosen = set(random.Random(seed).sample(range(total), min(sample, total)))
        return (
            (row for index, row in enumerate(rows) if index in chosen),
            min(sample, total),
        )
    if limit is not None:
        def limited() -> Iterator[Mapping[str, Any]]:
            taken = 0
            for row in rows:
                if taken >= limit:
                    return
                yield row
                taken += 1
        return limited(), limit
    return rows, total


# ------------------------------------------------------------------------- CLI


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Translate Lean dataset rows for Laya retraining")
    parser.add_argument("dataset", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--candidates", help="comma-separated candidate tactic families")
    parser.add_argument("--state-column", default="text_state")
    parser.add_argument("--tactic-column", default="tactic")
    parser.add_argument("--candidate-column", default="candidates")
    parser.add_argument("--split", default="train")
    parser.add_argument(
        "--device",
        default="cuda:0",
        metavar="DEVICE",
        help="trainer device: auto, cpu, cuda, cuda:0, cuda:1, ... (default: cuda:0)",
    )
    parser.add_argument(
        "--instructions",
        default=LAYA_RETRAIN_INSTRUCTIONS,
        help="override the Laya choice prompt",
    )
    size = parser.add_mutually_exclusive_group()
    size.add_argument("--limit", type=int, metavar="N", help="translate only the first N rows")
    size.add_argument("--sample", type=int, metavar="N", help="translate N randomly chosen rows")
    parser.add_argument("--seed", type=int, default=42, help="sampling seed for --sample")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="stop at the first unusable row instead of skipping and counting it",
    )
    parser.add_argument(
        "--quiet", action="store_true", help="hide the preparation progress bar"
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="print the first generated record after writing it",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="prepare the records, then ask laya.train.dry_run how many become "
             "training items, without loading model weights or training",
    )
    parser.add_argument("--trainer", help="optional remote trainer as module:function")
    parser.add_argument(
        "--fine-tune",
        action="store_true",
        help="run the built-in laya.train.finetune adapter after preparing records",
    )
    parser.add_argument(
        "--model-dir",
        default="convaiinnovations/laya",
        help="base Laya checkpoint directory or Hugging Face model id",
    )
    parser.add_argument("--checkpoint-dir", help="directory for the fine-tuned Laya checkpoint")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--micro-batch", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--loss", choices=("rlcd", "soft-ce"), default="rlcd")
    parser.add_argument("--seed-training", type=int, default=42)
    parser.add_argument(
        "--log-every",
        type=int,
        default=100,
        metavar="N",
        help="Laya prints training progress every N micro-steps (0 disables it)",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="show raw trainer output instead of a progress bar",
    )
    shuffle = parser.add_mutually_exclusive_group()
    shuffle.add_argument(
        "--shuffle-options",
        dest="shuffle_options",
        action="store_true",
        default=True,
        help="reshuffle candidate order every epoch (default, fights position priors)",
    )
    shuffle.add_argument(
        "--no-shuffle-options",
        dest="shuffle_options",
        action="store_false",
        help="keep the configured candidate order fixed for every epoch",
    )
    parser.add_argument("--freeze-encoder", action="store_true")
    parser.add_argument("--eval-data", type=Path, help="optional held-out JSONL for Laya evaluation")
    parser.add_argument("--max-len", type=int)
    parser.add_argument("--head-max-len", type=int)
    compare = parser.add_mutually_exclusive_group()
    compare.add_argument(
        "--no-compare",
        action="store_true",
        help="skip the automatic post-training comparison against the base checkpoint",
    )
    compare.add_argument(
        "--compare-rows",
        type=int,
        default=100,
        metavar="N",
        help="test rows used by the automatic comparison (0 disables it; default 100)",
    )
    parser.add_argument("--compare-seed", type=int, default=42)
    parser.add_argument("--compare-split", default="test")
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=Path.cwd(),
        help="where the comparison downloads weights, data and writes outputs",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point: known failures print one actionable line, not a traceback."""
    args = _build_parser().parse_args(argv)
    try:
        return _run(args)
    except (OSError, ValueError, RuntimeError) as exc:
        sys.stdout.flush()
        print(f"error: {exc}", file=sys.stderr)
        return 1


def _run(args: argparse.Namespace) -> int:
    candidates = tuple(_parse_candidates(args.candidates) or ())
    config = RetrainConfig(
        dataset_path=args.dataset,
        output_path=args.output,
        candidates=validate_tactic_names(list(candidates)),
        state_column=args.state_column,
        tactic_column=args.tactic_column,
        candidate_column=args.candidate_column or None,
        split_name=args.split,
        instructions=args.instructions,
        device=args.device,
        strict=args.strict,
    )
    if args.fine_tune and args.trainer:
        raise ValueError("use either --fine-tune or --trainer, not both")
    if args.dry_run and args.trainer:
        raise ValueError("use either --dry-run or --trainer, not both")

    stats = PrepStats()
    training = bool(args.fine_tune or args.trainer)
    total, _ = preflight(args, config, training=training)

    banner(
        "stage 1/3 translating records" if training else "translating records"
    )
    rows, expected = _selected_rows(
        iter_rows(config.dataset_path),
        total=total,
        sample=args.sample,
        limit=args.limit,
        seed=args.seed,
    )
    trainer = None
    trainer_kwargs = None
    if args.fine_tune:
        from laya_trainer import train

        trainer = train
        trainer_kwargs = {
            "model_dir": args.model_dir,
            "checkpoint_dir": args.checkpoint_dir,
            "epochs": args.epochs,
            "micro_batch": args.micro_batch,
            "grad_accum": args.grad_accum,
            "loss": args.loss,
            "seed": args.seed_training,
            "freeze_encoder": args.freeze_encoder,
            "shuffle_options": args.shuffle_options,
            "eval_data": str(args.eval_data) if args.eval_data else None,
            "max_len": args.max_len,
            "head_max_len": args.head_max_len,
            "log_every": args.log_every,
            "progress": not args.no_progress,
        }
    elif args.trainer:
        trainer = load_callable(args.trainer)
        trainer_kwargs = None

    training_run = trainer is not None and not args.dry_run

    def on_prepared(stats: PrepStats, path: Path) -> None:
        """Report and persist the preparation summary before training begins."""
        _report_preparation(stats, config)
        stats_file = write_stats(stats, config)
        if args.preview:
            _print_preview(stats)
        print(f"preparation stats: {stats_file}")
        if training_run:
            banner("stage 2/3 fine-tuning Laya")

    if trainer is None or args.dry_run:
        result: Any = retrain_laya(
            config, stats=stats, rows=rows, total=expected,
            progress=not args.quiet, on_prepared=on_prepared,
        )
    else:
        result = retrain_laya(
            config,
            trainer=trainer,
            trainer_kwargs=trainer_kwargs,
            stats=stats,
            rows=rows,
            total=expected,
            progress=not args.quiet,
            on_prepared=on_prepared,
        )

    if args.dry_run:
        banner("dry run")
        summary = _run_dry_run(args, config)
        print(json.dumps(summary, indent=2, default=str))
        print(f"\nprepared records: {config.output_path}")
        return 0

    if trainer is None:
        print(f"\nprepared translated Laya records: {result}")
        return 0

    print(f"\nfine-tune result: {_describe_result(result)}")

    if args.fine_tune and not args.no_compare and args.compare_rows > 0:
        _run_auto_compare(args, config, stats)
    return 0


def _report_preparation(stats: PrepStats, config: RetrainConfig) -> None:
    size = config.output_path.stat().st_size if config.output_path.is_file() else 0
    print(
        f"\n  wrote {stats.written:,} records to {config.output_path} "
        f"({_human_bytes(size)} in {format_elapsed(stats.elapsed)})"
    )
    for line in stats.summary_lines():
        print(line)
    counts = sorted(stats.candidate_counts)
    if counts and set(counts) != {10}:
        print(
            "  NOTE: these records use "
            f"{', '.join(str(count) for count in counts)} candidates, but the "
            "benchmark reranks the GNN top-10. The mismatch is recorded in "
            f"{config.stats_path.name} and in outputs/comparison.json."
        )


def _print_preview(stats: PrepStats) -> None:
    if stats.first_record is None:
        print("preview: no record was generated")
        return
    print("\npreview of the first record:")
    print(json.dumps(stats.first_record, ensure_ascii=False, indent=2))


def _run_dry_run(args: argparse.Namespace, config: RetrainConfig) -> dict[str, Any]:
    """Ask Laya how many prepared rows become trainable items (no weights)."""
    try:
        import laya  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "Laya is not installed in this environment. Install it with "
            "`pip install 'laya>=0.3.28,<0.5'` before using --dry-run."
        ) from exc
    try:
        from laya.train import TrainConfig, dry_run
    except ImportError as exc:
        raise RuntimeError(
            "This Laya installation does not expose laya.train.dry_run. "
            "Install a Laya release with the fine-tuning API (laya>=0.3.28)."
        ) from exc
    train_config = TrainConfig(
        epochs=args.epochs,
        micro_batch=args.micro_batch,
        grad_accum=args.grad_accum,
        loss=args.loss,
        seed=args.seed_training,
        freeze_encoder=args.freeze_encoder,
        shuffle_options=("choice",) if args.shuffle_options else (),
        eval_data=str(args.eval_data) if args.eval_data else None,
        max_len=args.max_len,
        head_max_len=args.head_max_len,
    )
    return dry_run(str(config.output_path), args.model_dir, config=train_config)


def _describe_result(result: Any) -> str:
    if isinstance(result, dict):
        parts = [f"checkpoint_dir={result.get('checkpoint_dir', '?')}"]
        report = result.get("report")
        if isinstance(report, dict):
            for key in ("train_report", "device", "items"):
                if key in report:
                    parts.append(f"{key}={report[key]}")
        return ", ".join(str(part) for part in parts)
    return str(result)


def _run_auto_compare(args: argparse.Namespace, config: RetrainConfig, stats: PrepStats) -> None:
    """Compare the fresh checkpoint against the base one on held-out test rows."""
    from comparison import append_history, compare_checkpoints, write_comparison
    from config import ExperimentConfig
    from laya_trainer import post_remap_device

    banner("stage 3/3 comparing checkpoints")
    checkpoint = args.checkpoint_dir or str(
        config.output_path.parent / (config.output_path.stem + "-checkpoint")
    )
    # Training may have restricted CUDA visibility, which renumbers the GPUs for
    # the rest of this process; ask for the device that still exists.
    device = post_remap_device(args.device)
    if device != args.device:
        print(f"  device: {args.device} → {device} (CUDA visibility was remapped for training)")
    experiment = ExperimentConfig(
        n_rows=args.compare_rows,
        seed=args.compare_seed,
        split=args.compare_split,
        device=device,
        laya_device=device,
        laya_mode="standard",
        work_dir=args.work_dir,
    )
    print(f"  pool: {args.compare_rows} rows of the {args.compare_split!r} split "
          f"(seed {args.compare_seed}), GNN top-{experiment.pool_k}")
    print(f"  models: base {experiment.laya_model!r} vs fine-tuned {checkpoint!r}")
    report = compare_checkpoints(
        experiment,
        {"base": experiment.laya_model, "finetuned": checkpoint},
        prep_stats=stats.to_dict(config),
    )
    json_path, image_path = write_comparison(report, experiment.resolved_output_dir)
    append_history(report, experiment.resolved_output_dir)
    _print_comparison(report)
    print(f"\ncomparison json : {json_path}")
    print(f"comparison image: {image_path}")
    print(f"run history     : {Path(experiment.resolved_output_dir) / 'history.jsonl'}")


def _print_comparison(report: Mapping[str, Any]) -> None:
    models = report.get("models", {})
    gnn = report.get("gnn", {}).get("metrics", {})
    print("\n" + "=" * 68)
    print("POST-TRAINING COMPARISON (same GNN top-10 pool, same rows)")
    print("=" * 68)
    header = f"{'system':<16}{'recall@10':>11}{'recall@5':>11}{'recall@1':>11}{'mrr@10':>10}"
    print(header)
    if gnn:
        print(
            f"{'GNN':<16}{_metric(gnn, 'recall@10'):>11.3f}"
            f"{_metric(gnn, 'recall@5'):>11.3f}{_metric(gnn, 'recall@1'):>11.3f}"
            f"{_metric(gnn, 'mrr@10'):>10.3f}"
        )
    for name, entry in models.items():
        metrics = entry.get("metrics", {})
        label = f"{name}*" if entry.get("laya_device", {}).get("warning") else name
        print(
            f"{label:<16}{_metric(metrics, 'recall@10'):>11.3f}"
            f"{_metric(metrics, 'recall@5'):>11.3f}{_metric(metrics, 'recall@1'):>11.3f}"
            f"{_metric(metrics, 'mrr@10'):>10.3f}"
        )
    mismatch = report.get("option_count", {})
    if mismatch.get("mismatch"):
        print(
            f"\noption counts   : inference reranks {mismatch.get('inference')} candidates, "
            f"training used {mismatch.get('training_histogram')} — recorded as "
            "'option_count' in this report"
        )
    for name, entry in models.items():
        warning = entry.get("laya_device", {}).get("warning")
        if warning:
            print(f"warning ({name})  : {warning}")
    print("=" * 68)


def _metric(metrics: Mapping[str, Any], key: str) -> float:
    value = metrics.get(key)
    return float(value) if isinstance(value, (int, float)) else 0.0


if __name__ == "__main__":
    raise SystemExit(main())
