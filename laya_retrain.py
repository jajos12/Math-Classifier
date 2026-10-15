"""Prepare and launch text-only Laya retraining jobs.

This module deliberately does not build PyG graphs. It translates Lean proof
states into Laya-readable text before creating choice-training records.

Every record presents Laya with the same *shape* of decision it sees at
inference: exactly ``POOL_SIZE`` (10) options built from the gold tactic plus
9 unique distractors. By default the distractors are drawn at random from the
most common tactic families in the corpus (``DISTRACTOR_TOP`` = 20); when
``--row-pools`` supplies the per-row GNN rankings written by
``pipeline.py pools``, a row instead draws from *its own* GNN top-k — the same
hard negatives inference presents — and only rows without a pool fall back to
the random draw. The gold tactic is placed at a random index in the option
list, the draws are keyed per row, and training reshuffles the options every
epoch (see :mod:`laya_trainer`), so neither the position nor the ordering of a
candidate carries any signal about the answer.

Preparation is streaming: parquet is read in chunks and JSONL is written row by
row, so a 200k-row dataset never has to exist in memory twice. Rows that cannot
be translated are skipped and counted per reason instead of aborting the run
(``--strict`` restores the old fail-fast behaviour).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
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
from data import pool_row_key
from graph import normalize_tactic
from progress import banner, bar, format_elapsed
from tactic_choose import TACTIC_SET, validate_tactic_names

LAYA_RETRAIN_INSTRUCTIONS = DEFAULT_INSTRUCTIONS
PARQUET_CHUNK_ROWS = 2000
SAMPLE_PREVIEW_ROWS = 200
STATE_COLUMN = "text_state"
TACTIC_COLUMN = "tactic"
POOL_SIZE = 10  # options presented to Laya per record (gold + 9 distractors)
DISTRACTOR_TOP = 20  # most common tactics the distractors are drawn from


class InvalidRowError(ValueError):
    """A row that cannot become a training record, tagged with a skip reason."""

    def __init__(self, reason: str, message: str) -> None:
        super().__init__(message)
        self.reason = reason


@dataclass(frozen=True)
class RetrainConfig:
    dataset_path: Path
    output_path: Path
    distractor_pool: tuple[str, ...] = ()
    distractor_pool_source: str = "dataset"
    row_pools: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    row_pools_source: str = ""
    options_per_row: int = POOL_SIZE
    seed: int = 42
    device: str = "cuda:0"
    strict: bool = False

    def __post_init__(self) -> None:
        if self.options_per_row < 2:
            raise ValueError("options_per_row must be at least 2")
        if self.distractor_pool:
            # Rejects anything outside the canonical catalog before any work starts.
            object.__setattr__(
                self,
                "distractor_pool",
                validate_tactic_names(list(self.distractor_pool)),
            )
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

    @property
    def resolved_pool(self) -> tuple[str, ...]:
        """The distractor pool; computed from the dataset when not supplied."""
        return self.distractor_pool or compute_distractor_pool(self.dataset_path)


@dataclass
class PrepStats:
    """Counters collected while translating rows, reported after preparation."""

    rows_seen: int = 0
    written: int = 0
    gold_in_pool: int = 0
    row_pool_matched: int = 0
    row_pool_in_use: bool = False
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
        if metadata.get("gold_in_pool"):
            self.gold_in_pool += 1
        if "row_pool_matched" in metadata:
            self.row_pool_in_use = True
            if metadata.get("row_pool_matched"):
                self.row_pool_matched += 1
        count = int(metadata.get("candidate_count", 0))
        self.candidate_counts[count] = self.candidate_counts.get(count, 0) + 1
        if self.first_record is None:
            self.first_record = dict(record)

    @property
    def gold_inserted(self) -> int:
        """Rows whose gold tactic was outside the common pool and added anyway."""
        return self.written - self.gold_in_pool

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
        if self.candidate_counts:
            counts = ", ".join(
                f"{count} options x{rows}"
                for count, rows in sorted(self.candidate_counts.items())
            )
            lines.append(f"  options per row: {counts}")
        if self.written:
            if self.row_pool_in_use:
                lines.append(
                    f"  row pools     : {self.row_pool_matched:,} records drew "
                    "distractors from their own row pool; "
                    f"{self.written - self.row_pool_matched:,} fell back to the "
                    "global pool"
                )
                gold_source = "its row's candidate pool"
            else:
                gold_source = "the common tactics"
            lines.append(
                f"  gold in pool   : {self.gold_in_pool} rows had the gold tactic among "
                f"{gold_source}; {self.gold_inserted} rows had it inserted "
                "from outside the pool"
            )
        return lines

    def to_dict(self, config: "RetrainConfig") -> dict[str, Any]:
        counts = self.training_option_counts
        return {
            "dataset": str(config.dataset_path),
            "output": str(config.output_path),
            "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "rows_seen": self.rows_seen,
            "records_written": self.written,
            "rows_skipped": self.total_skipped,
            "skipped_by_reason": dict(sorted(self.skipped.items())),
            "skip_examples": dict(sorted(self.skip_examples.items())),
            "gold_in_pool": self.gold_in_pool,
            "gold_inserted": self.gold_inserted,
            "training_option_counts": counts,
            "distractor_pool": list(config.resolved_pool),
            "distractor_pool_source": config.distractor_pool_source,
            "row_pools_source": config.row_pools_source or None,
            "row_pool_rows": len(config.row_pools),
            "row_pool_matched": self.row_pool_matched,
            "options_per_row": config.options_per_row,
            "candidate_seed": config.seed,
            "elapsed_seconds": round(self.elapsed, 2),
            "note": (
                "Each record presents the gold tactic plus unique distractors drawn "
                "at random from the most common tactic families; the gold position "
                "is randomized per row and the options are reshuffled every epoch, "
                "so the order of the choices carries no signal."
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


class GoldRandomCandidates:
    """Build the option pools Laya trains on: gold tactic + unique distractors.

    Two distractor sources, applied per row:

    * **Row pools** (``row_pools``): when the row's key is present, the
      distractors are drawn from that row's own GNN ranking — the hard
      negatives the benchmark actually reranks at inference. If the gold
      tactic is already in the ranking it stays the only natural member; if it
      is absent, it is inserted anyway and the row counts as ``gold_in_pool
      = False``, mirroring how the benchmark treats rows whose gold is outside
      the GNN top-k.
    * **Global pool** (``pool``): the default, and the fallback for rows
      without a matching row pool — distractors drawn uniformly at random,
      without replacement, from the most common tactic families.

    In both cases the gold tactic is placed at a uniformly random index, so
    neither the set nor the order of the options leaks the answer and every
    row sees the same 10-way decision shape that inference presents.

    The draws are keyed on ``(seed, row)``, so they are identical across runs,
    unaffected by ``--limit``/``--sample`` subsets, and reproducible when a
    second file (validation) is translated with the same pool or row pools.
    """

    def __init__(
        self,
        pool: Sequence[str],
        *,
        options_per_row: int = POOL_SIZE,
        seed: int = 42,
        row_pools: Mapping[str, Sequence[str]] | None = None,
    ) -> None:
        pool = tuple(validate_tactic_names(list(pool)))
        if len(pool) < 2:
            raise ValueError(
                f"the distractor pool needs at least 2 tactics, got {len(pool)}"
            )
        self.pool = pool
        self.options_per_row = max(2, options_per_row)
        self.seed = seed
        # Keys are canonicalized once so a float-rendered row_index
        # ("123.0") still matches the key built during translation ("123").
        self.row_pools = (
            {pool_row_key(key): tuple(names) for key, names in row_pools.items()}
            if row_pools
            else {}
        )

    @property
    def distractor_count(self) -> int:
        return min(self.options_per_row - 1, len(self.pool) - 1)

    def matched(self, row_key: str) -> bool:
        """True when this row draws distractors from its own row pool."""
        return pool_row_key(row_key) in self.row_pools

    def for_row(
        self, gold: str, *, row_key: str
    ) -> tuple[list[str], bool]:
        """Return (options, gold_was_in_pool) for one row's gold tactic."""
        rng = random.Random(f"{self.seed}:{row_key}")
        ranking = self.row_pools.get(pool_row_key(row_key))
        if ranking is None:
            distractors = rng.sample(
                [name for name in self.pool if name != gold], self.distractor_count
            )
        else:
            hard = [name for name in ranking if name != gold]
            needed = self.options_per_row - 1
            if len(hard) >= needed:
                distractors = rng.sample(hard, needed)
            else:
                # A ranking shorter than k (or one the gold dominates) is
                # topped up from the global pool, never with duplicates.
                distractors = list(hard)
                taken = set(distractors) | {gold}
                fill = [name for name in self.pool if name not in taken]
                distractors.extend(
                    rng.sample(fill, min(needed - len(distractors), len(fill)))
                )
        options = [gold, *distractors]
        rng.shuffle(options)  # the gold tactic lands at a uniform random index
        return options, gold in (ranking if ranking is not None else self.pool)


def _row_key(row: Mapping[str, Any], fallback_text: str) -> str:
    """Stable per-row key for the candidate RNG.

    ``row_index`` when the dataset provides one; a hash of the raw proof state
    otherwise (the built-in ``hash()`` is salted per process and would make
    the draws differ between runs).
    """
    value = row.get("row_index")
    if value is not None:
        text = str(value).strip()
        if text and text.lower() not in {"nan", "<na>", "none"}:
            return text
    digest = hashlib.sha1(fallback_text.encode("utf-8")).hexdigest()
    return "state:" + digest[:16]


def compute_distractor_pool(
    path: Path, *, column: str = TACTIC_COLUMN, top_k: int = DISTRACTOR_TOP
) -> tuple[str, ...]:
    """The ``top_k`` most common tactic families over the whole corpus.

    One projected single-column pass (parquet reads only ``column`` when
    pyarrow can project it). Values are normalized to tactic families, and
    anything outside the canonical catalog is ignored, so the pool always
    consists of real tactic names. Ties break alphabetically, which keeps the
    pool identical across runs and machines.
    """
    counts: dict[str, int] = {}
    for value in _iter_column_values(path, column):
        if _is_missing(value):
            continue
        name = normalize_tactic(str(value))
        if name in TACTIC_SET:
            counts[name] = counts.get(name, 0) + 1
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    pool = tuple(name for name, _ in ranked[:top_k])
    if len(pool) < 2:
        raise ValueError(
            f"{path} contains fewer than 2 known tactic families in {column!r}; "
            "cannot build candidate pools from it"
        )
    return pool


def load_row_pools(path: Path) -> dict[str, tuple[str, ...]]:
    """Per-row GNN candidate pools written by ``pipeline.py pools`` (JSONL).

    Each line carries a dataset ``row_index`` and the GNN's top-k ranking for
    that row. Rankings are normalized to canonical tactic families, unknown
    names are dropped, and duplicates collapse in order, so each entry is a
    ready-made hard-negative list. Keys are canonicalized with
    :func:`data.pool_row_key`, and a bad line fails the run with its line
    number instead of silently shrinking the pool.
    """
    path = Path(path)
    if not path.is_file():
        raise ValueError(
            f"--row-pools {path} is not a file; build it with "
            f"`python3 pipeline.py pools --split train --out {path}`"
        )
    pools: dict[str, tuple[str, ...]] = {}
    with path.open(encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: not valid JSON ({exc})") from exc
            if not isinstance(payload, Mapping):
                raise ValueError(f"{path}:{lineno}: expected a JSON object")
            key = pool_row_key(payload.get("row_index"))
            if not key:
                raise ValueError(f"{path}:{lineno}: every line needs a usable row_index")
            ranking = payload.get("ranking")
            if (
                not isinstance(ranking, Sequence)
                or isinstance(ranking, (str, bytes))
                or not ranking
            ):
                raise ValueError(
                    f"{path}:{lineno}: ranking must be a non-empty list of tactics"
                )
            names = list(
                dict.fromkeys(
                    name
                    for name in (normalize_tactic(str(value)) for value in ranking)
                    if name in TACTIC_SET
                )
            )
            if not names:
                raise ValueError(
                    f"{path}:{lineno}: ranking holds no known tactics: {ranking!r}"
                )
            pools[key] = tuple(names)
    if not pools:
        raise ValueError(f"{path} contains no row pools")
    return pools


def build_training_record(
    row: Mapping[str, Any],
    *,
    candidates: GoldRandomCandidates,
    instructions: str = LAYA_RETRAIN_INSTRUCTIONS,
) -> dict[str, Any]:
    """Translate one dataset row into an official-style Laya choice record.

    The row's tactic becomes the gold option inside a ``POOL_SIZE``-option
    choice question built by ``candidates``; everything else about the record
    (state translation, descriptions) is identical for gold and distractors,
    so the only learnable signal is the proof state itself.

    Raises :class:`InvalidRowError` (a ``ValueError``) when the row cannot be
    translated; callers decide whether that skips the row or stops the run.
    """
    raw_state = _require_text(row.get(STATE_COLUMN), STATE_COLUMN, "state")
    raw_tactic = _require_text(row.get(TACTIC_COLUMN), TACTIC_COLUMN, "tactic")
    target = normalize_tactic(raw_tactic)
    if not target or target.startswith("<"):
        raise InvalidRowError(
            "tactic_empty",
            f"dataset row has no usable tactic in {TACTIC_COLUMN!r}: {raw_tactic!r}",
        )
    if target not in TACTIC_SET:
        raise InvalidRowError(
            "tactic_unknown",
            f"tactic family {target!r} is not in the canonical catalog "
            "(see tactic_choose.py)",
        )
    row_key = _row_key(row, raw_state)
    option_list, gold_in_pool = candidates.for_row(target, row_key=row_key)
    try:
        translated_state = translate_lean_state(raw_state)
    except InvalidRowError:
        raise
    except (ValueError, TypeError) as exc:
        raise InvalidRowError("state_invalid", str(exc)) from exc
    arguments = raw_tactic[len(target):].strip() if raw_tactic.startswith(target) else ""
    metadata = {
        "candidate_count": len(option_list),
        "candidate_order": option_list,
        "gold_in_pool": gold_in_pool,
        "prompt_version": "lean-tactic-family-v3",
    }
    if candidates.row_pools:
        # Only tagged when row pools are configured at all, so legacy
        # records keep their exact schema.
        metadata["row_pool_matched"] = candidates.matched(row_key)
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
        "metadata": metadata,
        "tags": ["atp", "laya-retrain"],
        "row_index": row.get("row_index"),
    }


def _iter_parquet_rows(path: Path) -> Iterator[dict[str, Any]]:
    """Stream parquet rows in batches.

    ``pandas.read_parquet(chunksize=...)`` forwards ``chunksize`` to
    ``pyarrow.parquet.read_table``, which older pyarrow releases reject, so the
    batch API is used directly. ``ParquetFile.iter_batches`` has been available
    since long before any version this project supports; a whole-file read is
    the last resort when neither path is usable.
    """
    try:
        import pyarrow.parquet as pq
    except ImportError:
        try:
            import pandas as pd
        except ImportError as exc:
            raise RuntimeError("Install pandas and pyarrow to read parquet training data") from exc
        yield from pd.read_parquet(path).to_dict("records")
        return
    try:
        parquet = pq.ParquetFile(path)
    except Exception:
        import pandas as pd
        yield from pd.read_parquet(path).to_dict("records")
        return
    for batch in parquet.iter_batches(batch_size=PARQUET_CHUNK_ROWS):
        if hasattr(batch, "to_pylist"):
            yield from batch.to_pylist()
        else:
            yield from batch.to_pandas().to_dict("records")


def iter_rows(path: Path) -> Iterator[dict[str, Any]]:
    """Yield dataset rows one at a time, never materialising the whole file."""
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        try:
            yield from _iter_parquet_rows(path)
        except Exception as exc:
            # Surface any pyarrow/pandas failure as one actionable line; the
            # handler in main() prints the chained cause underneath it.
            raise RuntimeError(f"failed to read parquet dataset {path}") from exc
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


def _sampler_for(config: RetrainConfig) -> GoldRandomCandidates:
    """The candidate builder for this run: configured pool, or computed now."""
    return GoldRandomCandidates(
        config.resolved_pool,
        options_per_row=config.options_per_row,
        seed=config.seed,
        row_pools=config.row_pools,
    )


def _build_record(
    row: Mapping[str, Any],
    sampler: GoldRandomCandidates,
) -> dict[str, Any]:
    return build_training_record(row, candidates=sampler)


def iter_training_records(
    rows: Iterable[Mapping[str, Any]],
    *,
    config: RetrainConfig,
    stats: PrepStats | None = None,
    progress: bool = True,
    total: int | None = None,
    sampler: GoldRandomCandidates | None = None,
) -> Iterator[dict[str, Any]]:
    """Translate rows, skipping (and counting) the ones that cannot be used."""
    stats = stats if stats is not None else PrepStats()
    sampler = sampler if sampler is not None else _sampler_for(config)
    progress_bar = bar(total=total, desc="translate", disable=not progress, file=sys.stderr)
    try:
        for row in rows:
            stats.rows_seen += 1
            progress_bar.update(1)
            try:
                record = _build_record(row, sampler)
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


def retrain_laya(
    config: RetrainConfig,
    *,
    trainer: Callable[..., Any] | None = None,
    trainer_kwargs: Mapping[str, Any] | None = None,
    stats: PrepStats | None = None,
    rows: Iterable[Mapping[str, Any]] | None = None,
    total: int | None = None,
    progress: bool = True,
    sampler: GoldRandomCandidates | None = None,
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
        source, config=config, stats=stats, progress=progress, total=total,
        sampler=sampler,
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
            "Carve a disjoint validation split and translate it first:\n"
            "  python3 pipeline.py split data/train.parquet\n"
            f"  python3 laya_retrain.py data/validation.parquet {path} "
            "--distractor-pool data/train_fit.parquet"
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


def _load_eval_signatures(path: Path) -> tuple[int, list[str], list[int]]:
    """Per-row keys of the eval file: (rows, row_index ids, hashes of raw_state)."""
    ids: list[str] = []
    states: list[int] = []
    rows = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            rows += 1
            if not isinstance(record, dict):
                continue
            row_index = record.get("row_index")
            if row_index is not None:
                ids.append(str(row_index))
            raw_state = record.get("raw_state")
            if isinstance(raw_state, str) and raw_state.strip():
                states.append(hash(raw_state))
    return rows, ids, states


def _dataset_columns(path: Path) -> set[str] | None:
    """Column names of a parquet file (cheap); None when a row probe is needed."""
    if path.suffix.lower() != ".parquet":
        return None
    try:
        import pyarrow.parquet as pq
    except ImportError:
        return None
    try:
        parquet = pq.ParquetFile(path)
        schema = getattr(parquet, "schema_arrow", None)
        return set(schema.names if schema is not None else parquet.schema.names)
    except Exception:
        return None


def _iter_column_values(path: Path, column: str) -> Iterator[Any]:
    """Values of one column, projected out of parquet when pyarrow can do it."""
    if path.suffix.lower() == ".parquet":
        try:
            import pyarrow.parquet as pq

            parquet = pq.ParquetFile(path)
            for batch in parquet.iter_batches(batch_size=8192, columns=[column]):
                values = batch.column(0)
                if hasattr(values, "to_pylist"):
                    yield from values.to_pylist()
                else:
                    yield from values.to_pandas().tolist()
            return
        except Exception:
            pass  # unknown column or old pyarrow: fall back to the generic stream
    for row in iter_rows(path):
        yield row.get(column)


def _check_eval_overlap(
    dataset_path: Path,
    eval_path: Path,
    *,
    id_column: str = "row_index",
) -> tuple[int, int, str] | None:
    """Count eval rows that also occur in the training dataset.

    Returns ``(shared, rows, matched on)`` — or ``None`` when the two files
    share no usable key, in which case overlap cannot be determined cheaply.
    Matching prefers ``row_index`` (cheap, exact) and falls back to the raw
    proof state, which also catches duplicated rows under different indices.
    """
    rows, ids, states = _load_eval_signatures(eval_path)
    if not rows:
        return None
    columns = _dataset_columns(dataset_path)
    if columns is None:
        first = next(iter_rows(dataset_path), None)
        columns = set(first) if isinstance(first, Mapping) else set()
    if len(ids) == rows and id_column in columns:
        known = {str(value) for value in _iter_column_values(dataset_path, id_column)}
        return sum(1 for value in ids if value in known), rows, id_column
    # The dataset column holding the raw proof state: the parquet column for
    # raw datasets, or the ``raw_state`` field when training from prepared records.
    if len(states) == rows:
        for column in (STATE_COLUMN, "raw_state"):
            if column not in columns:
                continue
            known = {
                hash(str(value)) for value in _iter_column_values(dataset_path, column)
            }
            return sum(1 for value in states if value in known), rows, column
    return None


def _report_eval_overlap(args: argparse.Namespace, config: RetrainConfig) -> None:
    """Fail pre-flight when --eval-data rows are also rows of the training set."""
    overlap = _check_eval_overlap(config.dataset_path, args.eval_data)
    if overlap is None:
        print(
            "  eval overlap  : could not be checked (no shared row_index or "
            f"{STATE_COLUMN}/raw_state key between the files)"
        )
        return
    shared, rows, key = overlap
    if not shared:
        print(
            f"  eval overlap  : none — {rows} eval rows are disjoint from "
            f"{config.dataset_path.name} (matched on {key})"
        )
        return
    if getattr(args, "allow_eval_overlap", False):
        print(
            f"  eval overlap  : WARNING {shared}/{rows} eval rows are also training "
            "rows (allowed by --allow-eval-overlap; the eval measures memorization)"
        )
        return
    suffix = config.dataset_path.suffix
    train_file = config.dataset_path.parent / f"{config.dataset_path.stem}_fit{suffix}"
    validation_file = config.dataset_path.parent / f"validation{suffix}"
    raise ValueError(
        f"--eval-data {args.eval_data} shares {shared}/{rows} rows with "
        f"{config.dataset_path} (matched on {key}); evaluating on training rows "
        "measures memorization, not generalization.\n"
        "Carve a disjoint validation split out of the training data first:\n"
        f"  python3 pipeline.py split {config.dataset_path}\n"
        f"  python3 laya_retrain.py {validation_file} {args.eval_data} "
        f"--distractor-pool {train_file}\n"
        f"then train on {train_file}, or pass --allow-eval-overlap to run anyway."
    )


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
    sampler = _sampler_for(config)
    shown = ", ".join(sampler.pool[:10]) + (" ..." if len(sampler.pool) > 10 else "")
    print(
        f"  distractor pool: {len(sampler.pool)} common tactics from "
        f"{config.distractor_pool_source}"
    )
    print(f"                 : {shown}")
    if config.row_pools:
        print(
            f"  row pools     : {len(config.row_pools):,} per-row GNN pools from "
            f"{config.row_pools_source}"
        )
        print(
            "                 : distractors come from each row's own ranking; "
            "rows without a pool fall back to the common tactics above"
        )
    if len(sampler.pool) < config.options_per_row:
        print(
            f"  warning        : the pool holds only {len(sampler.pool)} tactics, so "
            f"records get {sampler.distractor_count + 1} options instead of "
            f"{config.options_per_row}"
        )
    print(
        f"  options/record : gold + {sampler.distractor_count} unique distractors "
        f"(seed {config.seed}, random gold position, reshuffled every epoch)"
    )
    if args.eval_data is not None:
        _check_eval_data(args.eval_data)
        print(f"  eval data     : {args.eval_data} ok")
        _report_eval_overlap(args, config)
    if training:
        from laya_trainer import prepare_device

        effective = prepare_device(args.device, label="--device")
        note = f" (remapped to {effective})" if effective != args.device else ""
        print(f"  device        : {args.device} ok{note}")
    sample, total = _sample_rows(config.dataset_path)
    estimated_bytes: int | None = None
    if sample:
        prepared: list[dict[str, Any]] = []
        unusable: dict[str, str] = {}
        for row in sample:
            try:
                prepared.append(_build_record(row, sampler))
            except ValueError as exc:
                reason = getattr(exc, "reason", "invalid")
                unusable.setdefault(reason, str(exc))
        if not prepared:
            reasons = "; ".join(f"{key}: {value}" for key, value in unusable.items())
            raise ValueError(
                f"none of the {len(sample)} sampled rows can be translated ({reasons}). "
                "Check that the dataset has usable text_state/tactic columns and that "
                "the tactic families exist in tactic_choose.py."
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
        if config.row_pools and prepared:
            preview_matched = sum(
                1 for record in prepared if record["metadata"].get("row_pool_matched")
            )
            print(
                f"  row pools     : {preview_matched}/{len(prepared)} previewed rows "
                "matched a per-row pool"
            )
            if not preview_matched:
                print(
                    "                 : WARNING — no matches; every row would fall "
                    "back to the global pool (do the row_index keys line up?)"
                )
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
            if args.fine_tune:
                micro_batch = max(1, args.micro_batch)
                grad_accum = max(1, args.grad_accum)
                per_epoch = -(-total // micro_batch)
                updates = -(-per_epoch // grad_accum) * max(1, args.epochs)
                print(
                    f"  budget        : ~{per_epoch:,} micro-batches/epoch × {args.epochs} "
                    f"= ~{per_epoch * args.epochs:,} steps, effective batch "
                    f"{micro_batch * grad_accum} → ~{updates:,} optimizer updates"
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
    parser.add_argument(
        "--distractor-pool",
        metavar="POOL",
        help="tactics the distractors are drawn from: a comma-separated list, "
             "or a dataset file whose most common tactics become the pool "
             "(default: computed from the dataset being translated)",
    )
    parser.add_argument(
        "--row-pools",
        type=Path,
        metavar="PATH",
        help="per-row GNN candidate pools from `pipeline.py pools` (JSONL); each "
             "row's distractors come from its own GNN ranking (the hard negatives "
             "inference presents), falling back to --distractor-pool for rows "
             "without a pool",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        metavar="DEVICE",
        help="trainer device: auto, cpu, cuda, cuda:0, cuda:1, ... (default: cuda:0)",
    )
    size = parser.add_mutually_exclusive_group()
    size.add_argument("--limit", type=int, metavar="N", help="translate only the first N rows")
    size.add_argument("--sample", type=int, metavar="N", help="translate N randomly chosen rows")
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="seed for --sample and for the per-row distractor/gold-position draws",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="stop at the first unusable row instead of skipping and counting it",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="hide the preparation progress bar",
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
    parser.add_argument("--eval-data", type=Path, help="optional held-out JSONL for Laya evaluation")
    parser.add_argument(
        "--allow-eval-overlap",
        action="store_true",
        help="start even when --eval-data rows are also in the training dataset "
             "(the in-training evaluation then measures memorization)",
    )
    parser.add_argument(
        "--compare-rows",
        type=int,
        default=100,
        metavar="N",
        help="test rows used by the automatic comparison (0 disables it; default 100)",
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
        cause = getattr(exc, "__cause__", None)
        if cause is not None and str(cause) != str(exc):
            print(f"       caused by {type(cause).__name__}: {cause}", file=sys.stderr)
        return 1


def _resolve_distractor_pool(
    value: str | None, dataset_path: Path
) -> tuple[tuple[str, ...], str]:
    """Resolve --distractor-pool into (pool, source) before any long work.

    A comma-separated list is used as-is (validated against the canonical
    catalog); a dataset path gets its most common tactics ranked instead, so
    translating a validation split with ``--distractor-pool <train file>``
    guarantees both halves present Laya with the identical pool.
    """
    if value is None:
        return compute_distractor_pool(dataset_path), str(dataset_path)
    text = value.strip()
    if "," in text:
        names = [name.strip() for name in text.split(",") if name.strip()]
        if not names:
            raise ValueError("--distractor-pool needs tactic names or a dataset path")
        return validate_tactic_names(names), "--distractor-pool"
    source = Path(text)
    if not source.is_file():
        raise ValueError(
            f"--distractor-pool {source} is neither an existing dataset file nor a "
            "comma-separated tactic list"
        )
    return compute_distractor_pool(source), str(source)


def _run(args: argparse.Namespace) -> int:
    pool, pool_source = _resolve_distractor_pool(args.distractor_pool, args.dataset)
    row_pools = load_row_pools(args.row_pools) if args.row_pools else {}
    config = RetrainConfig(
        dataset_path=args.dataset,
        output_path=args.output,
        distractor_pool=pool,
        distractor_pool_source=pool_source,
        row_pools=row_pools,
        row_pools_source=str(args.row_pools) if args.row_pools else "",
        seed=args.seed,
        device=args.device,
        strict=args.strict,
    )

    stats = PrepStats()
    total, _ = preflight(args, config, training=args.fine_tune)

    banner(
        "stage 1/3 translating records" if args.fine_tune else "translating records"
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
            "loss": "rlcd",
            "seed": args.seed,
            "freeze_encoder": False,
            # Always reshuffle the options: no fixed order can become a prior.
            "shuffle_options": True,
            "eval_data": str(args.eval_data) if args.eval_data else None,
            "progress": not args.quiet,
        }

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

    if args.fine_tune and args.compare_rows > 0:
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
    if counts and set(counts) != {config.options_per_row}:
        print(
            "  NOTE: these records use "
            f"{', '.join(str(count) for count in counts)} options, not "
            f"{config.options_per_row}: the distractor pool holds too few distinct "
            "tactics for this dataset, and the benchmark reranks the GNN top-10. "
            f"The mismatch is recorded in {config.stats_path.name}."
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
        loss="rlcd",
        seed=args.seed,
        freeze_encoder=False,
        shuffle_options=("choice",),
        eval_data=str(args.eval_data) if args.eval_data else None,
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
        seed=42,
        split="test",
        device=device,
        laya_device=device,
        laya_mode="standard",
    )
    print(f"  pool: {args.compare_rows} rows of the 'test' split "
          f"(seed 42), GNN top-{experiment.pool_k}")
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
