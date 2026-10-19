"""Small command-line entry point for validating the experiment package."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

from comparison import (
    append_history,
    describe_device,
    rerank_states,
    write_environment,
)
from config import ExperimentConfig
from data import (
    audit_splits,
    combine_shards,
    download_dataset,
    load_dataset,
    pool_row_key,
    sample_dataset,
    split_dataset,
)
from evaluation import build_report, print_summary, write_metrics_image, write_report
from graph import normalize_tactic, proof_state_to_dag
from gnn import build_graphs, build_model, download_bundle, load_bundle, predict_top_k
from laya_adapter import LayaAdapter


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Math Classifier GNN/Laya experiment")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("validate", help="run a dependency-free graph smoke test")
    subparsers.add_parser("devices", help="list available Torch devices")
    subparsers.add_parser(
        "runtime",
        help="show the Python environment, Torch, CUDA, and visible GPUs",
    )
    audit = subparsers.add_parser(
        "audit",
        help="check the dataset splits for duplicate rows, within and across splits",
    )
    audit.add_argument(
        "--splits",
        default="train,val,test",
        metavar="A,B,C",
        help="comma-separated split names to audit (default: train,val,test)",
    )
    audit.add_argument(
        "--out",
        type=Path,
        default=None,
        metavar="PATH",
        help="audit report path (default: outputs/audit.json)",
    )
    fetch = subparsers.add_parser(
        "fetch",
        help="download one Hugging Face split and optionally combine its shards",
    )
    fetch.add_argument("--split", default="val", help="split to download (default: val)")
    fetch.add_argument(
        "--out",
        type=Path,
        default=None,
        metavar="PATH",
        help="combine the shards into this file (default: data/<split>.parquet)",
    )
    fetch.add_argument(
        "--no-combine",
        dest="combine",
        action="store_false",
        help="download the shards without combining them into one file",
    )
    benchmark = subparsers.add_parser("benchmark", help="run GNN and Laya ranking evaluation")
    benchmark.add_argument("--rows", type=int, default=500, help="number of rows to evaluate")
    benchmark.add_argument("--seed", type=int, default=42)
    benchmark.add_argument(
        "--device",
        default="auto",
        metavar="DEVICE",
        help="GNN device: auto, cpu, cuda, cuda:0, cuda:1, ... (default: auto)",
    )
    benchmark.add_argument(
        "--laya-device",
        default="auto",
        metavar="DEVICE",
        help="Laya device; defaults to the GNN --device selection",
    )
    benchmark.add_argument(
        "--laya-mode",
        choices=("standard", "balanced"),
        default="standard",
        help="Laya scoring mode; balanced removes candidate-order bias but is slower",
    )
    benchmark.add_argument(
        "--laya-model",
        default="convaiinnovations/laya",
        help="base Laya checkpoint or fine-tuned checkpoint directory to rerank with",
    )
    split = subparsers.add_parser(
        "split",
        help="carve a disjoint validation split out of a dataset (default 80/20)",
    )
    split.add_argument("dataset", type=Path)
    split.add_argument(
        "--validation-fraction",
        type=float,
        default=0.2,
        metavar="F",
        help="fraction of rows kept for validation (default: 0.2)",
    )
    split.add_argument(
        "--seed", type=int, default=42, help="seed for the row assignment (default: 42)"
    )
    pools = subparsers.add_parser(
        "pools",
        help="score a dataset split with the GNN and write per-row top-k pools "
             "(hard negatives for laya_retrain --row-pools)",
    )
    pools.add_argument(
        "--split", default="train", help="dataset split to score (default: train)"
    )
    pools.add_argument(
        "--out",
        type=Path,
        required=True,
        metavar="PATH",
        help="JSONL output; pools already in the file are resumed, not recomputed",
    )
    pools.add_argument(
        "--device",
        default="auto",
        metavar="DEVICE",
        help="GNN device: auto, cpu, cuda, cuda:0, ... (default: auto)",
    )
    pools.add_argument(
        "--limit", type=int, metavar="N", help="score only the first N rows (smoke test)"
    )
    pools.add_argument(
        "--chunk",
        type=int,
        default=4096,
        metavar="N",
        help="rows per graph-building pass (default: 4096)",
    )
    pools.add_argument(
        "--force", action="store_true", help="overwrite --out instead of resuming it"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    try:
        return _dispatch(argv)
    except (OSError, ValueError, RuntimeError) as exc:
        # Expected failures (missing files, bad arguments) get one actionable
        # line instead of a traceback; anything else still shows its traceback.
        sys.stdout.flush()
        print(f"error: {exc}", file=sys.stderr)
        cause = getattr(exc, "__cause__", None)
        if cause is not None and str(cause) != str(exc):
            print(f"       caused by {type(cause).__name__}: {cause}", file=sys.stderr)
        return 1


def _dispatch(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "validate":
        dag = proof_state_to_dag("h : α = α\n⊢ α = α")
        if not any(node.label == "State" for node in dag.nodes):
            raise RuntimeError("graph smoke test did not create a State node")
        print(f"graph validation passed: {dag.num_nodes} nodes, {dag.num_edges} edges")
    elif args.command == "devices":
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("Torch is required to list runtime devices") from exc
        print("cpu")
        if torch.cuda.is_available():
            for index in range(torch.cuda.device_count()):
                print(f"cuda:{index} ({torch.cuda.get_device_name(index)})")
        else:
            print("CUDA is not available")
    elif args.command == "runtime":
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError(
                "Torch is not installed in this Python environment. "
                "Activate the intended environment or select the matching Jupyter kernel."
            ) from exc
        print(f"python: {sys.executable}")
        print(f"python_version: {sys.version.split()[0]}")
        print(f"torch: {torch.__version__}")
        print(f"cuda_available: {torch.cuda.is_available()}")
        print(f"cuda_visible_devices: {os.environ.get('CUDA_VISIBLE_DEVICES', '<all>')}")
        if torch.cuda.is_available():
            print(f"cuda_device_count: {torch.cuda.device_count()}")
            for index in range(torch.cuda.device_count()):
                print(f"cuda:{index}: {torch.cuda.get_device_name(index)}")
    elif args.command == "audit":
        return _cmd_audit(args)
    elif args.command == "fetch":
        return _cmd_fetch(args)
    elif args.command == "benchmark":
        config = ExperimentConfig(
            n_rows=args.rows,
            seed=args.seed,
            device=args.device,
            laya_device=args.laya_device,
            laya_mode=args.laya_mode,
            laya_model=args.laya_model,
        )
        config.prepare_directories()
        from laya_trainer import validate_device

        validate_device(config.resolved_device, label="--device")
        validate_device(config.resolved_laya_device, label="--laya-device")
        paths = download_dataset(config.dataset_repo, config.split, config.dataset_dir)
        frame = sample_dataset(load_dataset(paths), config.n_rows, config.seed)
        states = frame["text_state"].astype(str).tolist()
        targets = [normalize_tactic(value) for value in frame["tactic"].tolist()]
        bundle_dir = download_bundle(config.model_repo, config.bundle_name, config.model_dir)
        bundle = load_bundle(bundle_dir)
        model, _ = build_model(bundle)
        graphs = build_graphs(states, bundle, edge_mode=bundle.config["edge_mode"])
        gnn_rankings, gnn_scores = predict_top_k(
            model, graphs, bundle, config.pool_k,
            batch_size=config.batch_size, device=config.resolved_device,
        )
        adapter = LayaAdapter.load(
            config.laya_model, device=config.resolved_laya_device,
            instructions=config.laya_instructions,
        )
        device_note = describe_device(adapter, config.resolved_laya_device)
        outcome = rerank_states(
            adapter,
            states,
            gnn_rankings,
            mode=config.laya_mode,
            threshold=config.laya_confidence_threshold,
        )
        laya_rankings = outcome["rankings"]
        laya_confidences = outcome["confidences"]
        laya_fallbacks = outcome["fallbacks"]
        del adapter
        report = build_report(
            gnn_rankings, laya_rankings, targets,
            pool_k=config.pool_k, select_k=config.select_k,
        )
        report["kind"] = "benchmark"
        report["split"] = config.split
        report["seed"] = config.seed
        report["n"] = len(states)
        report["pool_k"] = config.pool_k
        report["laya_mode"] = config.laya_mode
        report["laya_model"] = config.laya_model
        report["laya_confidence_threshold"] = config.laya_confidence_threshold
        report["laya_fallbacks"] = laya_fallbacks
        report["laya_device"] = device_note
        report["gnn_device"] = config.resolved_device
        output = write_report(report, config.resolved_output_dir / "metrics.json")
        image = write_metrics_image(report, config.resolved_output_dir / "metrics.png")
        (config.resolved_output_dir / "predictions.json").write_text(
            json.dumps(
                {
                    "targets": targets,
                    "gnn": gnn_rankings,
                    "laya": laya_rankings,
                    "laya_confidences": laya_confidences,
                    "gnn_scores": gnn_scores,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        history = append_history(report, config.resolved_output_dir)
        environment = write_environment(config.resolved_output_dir)
        print(f"benchmark complete: {output}")
        print_summary(report, image)
        if device_note["warning"]:
            print(f"WARNING: {device_note['warning']}")
        print(f"run history    : {history}")
        print(f"environment    : {environment}")
    elif args.command == "pools":
        if args.chunk < 1:
            raise ValueError("--chunk must be positive")
        if args.limit is not None and args.limit < 1:
            raise ValueError("--limit must be positive")
        from laya_trainer import validate_device

        config = ExperimentConfig(split=args.split, device=args.device)
        config.prepare_directories()
        validate_device(config.resolved_device, label="--device")
        out: Path = args.out
        done = _resume_pools(out, force=args.force)
        paths = download_dataset(config.dataset_repo, config.split, config.dataset_dir)
        frame = load_dataset(paths)
        if args.limit is not None:
            frame = frame.head(args.limit)
        pending, unusable = _pending_pool_rows(frame, done)
        print(
            f"split {config.split}: {len(frame):,} rows — {len(done):,} already "
            f"pooled, {len(pending):,} to score"
            + (f", {unusable} unusable (missing row_index/text_state)" if unusable else ""),
            flush=True,
        )
        if pending:
            bundle = load_bundle(
                download_bundle(config.model_repo, config.bundle_name, config.model_dir)
            )
            model, _ = build_model(bundle)
            out.parent.mkdir(parents=True, exist_ok=True)
            started = time.monotonic()
            before = len(done)
            written = len(done)
            hits = 0
            failed = 0
            with out.open("a", encoding="utf-8") as handle:
                for start in range(0, len(pending), args.chunk):
                    chunk = pending[start:start + args.chunk]
                    scored = _score_chunk(model, bundle, chunk, config)
                    for (key, _, target), (ranking, scores) in zip(chunk, scored):
                        if not ranking:
                            failed += 1
                            continue
                        handle.write(
                            json.dumps(
                                {
                                    "row_index": key,
                                    "target": target,
                                    "ranking": [str(name) for name in ranking],
                                    "scores": [
                                        round(float(value), 5) for value in scores
                                    ],
                                }
                            )
                            + "\n"
                        )
                        written += 1
                        if target and target in ranking:
                            hits += 1
                    handle.flush()
                    covered = min(start + args.chunk, len(pending))
                    elapsed = max(time.monotonic() - started, 1e-9)
                    rate = (written - before) / elapsed
                    eta = (len(pending) - covered) / rate if rate else 0.0
                    print(
                        f"  pools: {written:,} total, {covered:,}/{len(pending):,} "
                        f"this run ({rate:.0f} rows/s, ETA {_eta(eta)})",
                        flush=True,
                    )
            elapsed = time.monotonic() - started
            scored_count = written - before
            print(
                f"\nwrote {scored_count:,} new pools to {out} "
                f"({len(done):,} resumed, {failed} failed) in {int(elapsed)}s"
            )
            if scored_count:
                print(f"  file now holds {written:,} pools ({out.stat().st_size / 1e6:.1f} MB)")
                print(
                    f"  gold in top-{config.pool_k}: {hits}/{scored_count} "
                    f"({hits / scored_count:.1%}) of this run's rows — the "
                    "recall@k ceiling at inference"
                )
        else:
            print(f"nothing to do: {out} already covers this selection")
    elif args.command == "split":
        source = args.dataset
        train_out = source.parent / f"{source.stem}_fit{source.suffix}"
        validation_out = source.parent / f"validation{source.suffix}"
        train_rows, validation_rows = split_dataset(
            source,
            train_out,
            validation_out,
            validation_fraction=args.validation_fraction,
            seed=args.seed,
        )
        total = train_rows + validation_rows
        print(
            f"split {source} — {total:,} rows, seed {args.seed}, "
            f"validation fraction {args.validation_fraction:g}"
        )
        print(f"  train       : {train_out} — {train_rows:,} rows ({train_rows / total:.0%})")
        print(
            f"  validation  : {validation_out} — {validation_rows:,} rows "
            f"({validation_rows / total:.0%})"
        )
        print(
            "  the two files share no rows: train on the train file, and translate "
            "the validation file for --eval-data"
        )
    return 0


def _cmd_audit(args: argparse.Namespace) -> int:
    """Audit dataset splits for duplicate rows and record the result."""
    config = ExperimentConfig()
    config.prepare_directories()
    names = [name.strip() for name in args.splits.split(",") if name.strip()]
    if len(names) < 2:
        raise ValueError("--splits needs at least two split names (e.g. train,val,test)")
    splits = {
        name: download_dataset(config.dataset_repo, name, config.dataset_dir)
        for name in names
    }
    report = audit_splits(splits)
    out: Path = args.out or config.resolved_output_dir / "audit.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"audit: {config.dataset_repo} splits {', '.join(names)}")
    for name, entry in report["splits"].items():
        print(
            f"  {name:<6} {entry['rows']:>7,} rows — "
            f"{entry['unique_row_index']:,} unique row_index, "
            f"{entry['unique_state']:,} unique states, "
            f"within-split dupes {entry['duplicate_row_index']} (index) / "
            f"{entry['duplicate_state']} (state)"
        )
    for pair, entry in report["overlaps"].items():
        left, right = pair.split("|")
        status = "ok" if not entry["row_index"] and not entry["state"] else "DUPLICATES"
        print(
            f"  {left} vs {right}: {status} — "
            f"{entry['row_index']} shared row_index, {entry['state']} shared states"
        )
        if status != "ok":
            print(f"    row_index examples: {entry['row_index_examples']}")
            print(f"    state hash examples: {entry['state_examples']}")
    print(f"report: {out}")
    if not report["passed"]:
        print(
            "audit FAILED: the splits share rows, so held-out metrics would be "
            "contaminated; remove the duplicates before training",
            file=sys.stderr,
        )
        return 1
    print("audit passed: no cross-split duplicate rows")
    return 0


def _cmd_fetch(args: argparse.Namespace) -> int:
    """Download one split and optionally combine its shards into one file."""
    config = ExperimentConfig()
    config.prepare_directories()
    paths = download_dataset(config.dataset_repo, args.split, config.dataset_dir)
    for path in paths:
        print(f"shard: {path}")
    if not args.combine:
        return 0
    output: Path = args.out or config.work_dir / "data" / f"{args.split}.parquet"
    combined, rows = combine_shards(paths, output)
    print(f"combined {len(paths)} shard(s) → {combined} ({rows:,} rows)")
    return 0


def _resume_pools(path: Path, *, force: bool = False) -> set[str]:
    """Row keys already present in ``path``; repairs a file cut off mid-write.

    A crashed run can leave a partial last line behind. Unreadable rows are
    dropped (they were never completed) and the readable ones are kept, so
    rerunning the same command resumes instead of duplicating or losing
    pools. A file that holds no readable pool row at all is refused — it is
    probably not a pools file — unless ``force`` overwrites it outright.
    """
    path = Path(path)
    if force:
        path.unlink(missing_ok=True)
        return set()
    if not path.is_file():
        return set()
    done: set[str] = set()
    kept: list[str] = []
    dropped = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
            key = pool_row_key(payload.get("row_index")) if isinstance(payload, dict) else ""
            if not key or not isinstance(payload.get("ranking"), list):
                raise ValueError("incomplete pool row")
        except (json.JSONDecodeError, ValueError):
            dropped += 1
            continue
        done.add(key)
        kept.append(line)
    if dropped and not kept:
        raise ValueError(
            f"{path} holds no readable pool rows; it does not look like a "
            "pools file — pass --force to overwrite it"
        )
    if dropped:
        path.write_text("".join(f"{line}\n" for line in kept), encoding="utf-8")
        print(f"repaired {path}: dropped {dropped} unreadable line(s)")
    return done


def _pending_pool_rows(frame, done: set[str]) -> tuple[list[tuple[str, str, str]], int]:
    """(key, state, target) rows still to score, plus the unusable-row count."""
    pending: list[tuple[str, str, str]] = []
    unusable = 0
    for raw_key, raw_state, raw_target in zip(
        frame["row_index"].tolist(),
        frame["text_state"].tolist(),
        frame["tactic"].tolist(),
    ):
        key = pool_row_key(raw_key)
        state = raw_state.strip() if isinstance(raw_state, str) else ""
        if not key or not state:
            unusable += 1
            continue
        if key in done:
            continue
        try:
            target = normalize_tactic(str(raw_target))
        except Exception:
            target = ""
        pending.append((key, state, target))
    return pending, unusable


def _score_chunk(model, bundle, chunk, config) -> list[tuple[list, list]]:
    """GNN top-k for every row of one chunk: ``(ranking, scores)`` each.

    One unparseable state must not cost a whole chunk of a multi-hour build,
    so a failed chunk is retried row by row; rows that still fail come back
    with an empty ranking and are skipped by the caller.
    """
    states = [state for _, state, _ in chunk]
    edge_mode = bundle.config["edge_mode"]
    try:
        graphs = build_graphs(states, bundle, edge_mode=edge_mode)
        rankings, scores = predict_top_k(
            model,
            graphs,
            bundle,
            config.pool_k,
            batch_size=config.batch_size,
            device=config.resolved_device,
        )
        return [(list(ranking), list(score)) for ranking, score in zip(rankings, scores)]
    except Exception:  # noqa: BLE001 - fallback path, failures reported per row
        results = []
        for state in states:
            try:
                graphs = build_graphs([state], bundle, edge_mode=edge_mode)
                ranking, score = predict_top_k(
                    model,
                    graphs,
                    bundle,
                    config.pool_k,
                    batch_size=1,
                    device=config.resolved_device,
                )
                results.append((list(ranking[0]), list(score[0])))
            except Exception:
                results.append(([], []))
        return results


def _eta(seconds: float) -> str:
    total = int(seconds)
    if total >= 3600:
        return f"{total // 3600}h{total % 3600 // 60:02d}m"
    if total >= 60:
        return f"{total // 60}m{total % 60:02d}s"
    return f"{total}s"


if __name__ == "__main__":
    raise SystemExit(main())
