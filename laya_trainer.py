"""Built-in single-device fine-tuning adapter for Laya.

The adapter targets the public ``laya.train.finetune`` API.  It deliberately
does not reimplement Laya's model, loss, calibration, or checkpoint format.

``laya.train.finetune`` is silent while it reads and encodes the JSONL file and
only prints ``epoch e/E step n loss x`` during training, so this module wraps it
with a progress bar and a heartbeat: a long quiet stretch is never mistaken for
a hang.
"""

from __future__ import annotations

import contextlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from progress import LineFilter, format_elapsed, heartbeat

# Physical GPU that was remapped to logical ``cuda`` by _normalise_cuda_device.
# Any device string used *after* training has to be interpreted with this in
# mind: with CUDA_VISIBLE_DEVICES=1 only one GPU exists, and it is cuda:0.
_REMAPPED_INDEX: str | None = None


def train(
    records: Sequence[Mapping[str, Any]] | None,
    output_path: str,
    *,
    device: str = "cuda:0",
    model_dir: str = "convaiinnovations/laya",
    checkpoint_dir: str | None = None,
    epochs: int = 4,
    micro_batch: int = 8,
    grad_accum: int = 8,
    loss: str = "rlcd",
    seed: int = 42,
    freeze_encoder: bool = False,
    shuffle_options: bool = False,
    eval_data: str | None = None,
    max_len: int | None = None,
    head_max_len: int | None = None,
    log_every: int = 100,
    progress: bool = True,
) -> dict[str, Any]:
    """Fine-tune Laya and return its training report.

    ``output_path`` is the prepared JSONL written by :mod:`laya_retrain` and is
    handed to ``finetune`` as-is; it is only (re)written from ``records`` when
    it does not exist yet.  The saved checkpoint directory is loadable by
    ``laya.load``.
    """
    if records is not None and len(records) == 0:
        raise ValueError("cannot fine-tune Laya with an empty record set")
    # First thing: fix CUDA visibility before anything else can query the driver.
    effective_device = prepare_device(device, label="--device")

    prepared_path = Path(output_path)
    if not prepared_path.is_file():
        if records is None:
            raise FileNotFoundError(
                f"prepared records not found: {prepared_path}; run the preparation step first"
            )
        prepared_path.parent.mkdir(parents=True, exist_ok=True)
        with prepared_path.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    if prepared_path.stat().st_size == 0:
        raise ValueError(f"prepared records file is empty: {prepared_path}")
    if eval_data and not Path(eval_data).is_file():
        raise FileNotFoundError(
            f"--eval-data {eval_data} does not exist; prepare it before fine-tuning"
        )

    try:
        import laya  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "Laya is not installed in this environment. Install it with "
            "`pip install 'laya>=0.3.28,<0.5'` before fine-tuning."
        ) from exc
    try:
        from laya.train import TrainConfig, finetune
    except ImportError as exc:
        raise RuntimeError(
            "This Laya installation does not expose laya.train.finetune. "
            "Install a Laya release with the fine-tuning API (laya>=0.3.28)."
        ) from exc

    destination = Path(checkpoint_dir) if checkpoint_dir else prepared_path.with_name(
        prepared_path.stem + "-checkpoint"
    )
    config_kwargs: dict[str, Any] = {
        "epochs": epochs,
        "micro_batch": micro_batch,
        "grad_accum": grad_accum,
        "loss": loss,
        "seed": seed,
        "freeze_encoder": freeze_encoder,
        "log_every": log_every,
    }
    if shuffle_options:
        config_kwargs["shuffle_options"] = ("choice",)
    if eval_data:
        config_kwargs["eval_data"] = str(Path(eval_data))
    if max_len is not None:
        config_kwargs["max_len"] = max_len
    if head_max_len is not None:
        config_kwargs["head_max_len"] = head_max_len

    config = TrainConfig(**config_kwargs)

    size_mb = prepared_path.stat().st_size / (1024 ** 2)
    print(f"  training file : {prepared_path} ({size_mb:.1f} MB)")
    print(f"  checkpoint    : {destination}")
    print(f"  device        : {effective_device} (requested {device})")
    print(
        f"  progress      : Laya prints every {log_every} micro-steps; "
        "the bar below follows those lines"
    )
    print(
        "  silence       : Laya reports nothing while it reads and encodes the "
        "file — a heartbeat appears if that stage goes quiet for 60s"
    )
    sys.stdout.flush()

    started = time.monotonic()
    line_filter = LineFilter(
        sys.stdout,
        progress=progress,
        epochs=epochs,
        micro_batch=micro_batch,
    )
    gate = heartbeat("fine-tune", interval=60.0, line_filter=line_filter)
    line_filter.on_activity = gate.activity
    report: Any
    with gate:
        try:
            with contextlib.redirect_stdout(line_filter):
                report = finetune(
                    data=str(prepared_path),
                    model_dir=model_dir,
                    output_dir=str(destination),
                    config=config,
                    device=effective_device,
                )
        finally:
            line_filter.close()
    elapsed = time.monotonic() - started

    if not isinstance(report, dict):
        raise RuntimeError(
            f"Laya fine-tuning returned {type(report).__name__}, expected a report dictionary"
        )
    report.setdefault("device", effective_device)
    report["requested_device"] = device
    report["elapsed_seconds"] = round(elapsed, 1)
    report_path = destination / "training_report.json"
    report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(
        f"  trained in {format_elapsed(elapsed)} → checkpoint {destination} "
        f"(report: {report_path})"
    )
    return {
        "checkpoint_dir": str(destination),
        "training_data": str(prepared_path),
        "device": effective_device,
        "elapsed_seconds": round(elapsed, 1),
        "report": report,
    }


def validate_device(device: str, *, label: str = "device") -> None:
    """Fail early with an actionable message instead of mid-run.

    ``auto`` is resolved elsewhere and bare ``cuda`` is checked by Torch itself.
    Call :func:`prepare_device` first when a ``cuda:N`` may still be remapped:
    querying CUDA before ``CUDA_VISIBLE_DEVICES`` is set can lock the driver
    into the original device list.
    """
    if device in {"auto", "cpu"} or not device.startswith("cuda"):
        return
    try:
        import torch
    except ImportError:
        return
    index = int(device[5:]) if ":" in device else 0
    if not torch.cuda.is_available():
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        hint = f" (CUDA_VISIBLE_DEVICES={visible})" if visible else ""
        raise ValueError(
            f"{label} {device} was requested but no CUDA device is visible{hint}; "
            "use --device cpu, or check `python3 pipeline.py devices`"
        )
    count = torch.cuda.device_count()
    if index >= count:
        raise ValueError(
            f"{label} {device} was requested but only {count} CUDA device(s) are "
            "visible; run `python3 pipeline.py devices` to list them"
        )


def prepare_device(device: str, *, label: str = "--device") -> str:
    """Remap the requested GPU if needed, *then* verify it exists.

    The remap writes ``CUDA_VISIBLE_DEVICES`` and must happen before the first
    CUDA call in the process, otherwise the driver keeps the original device
    list and the request would silently land on GPU 0.
    """
    effective = _normalise_cuda_device(device)
    validate_device(effective, label=label)
    return effective


def post_remap_device(device: str) -> str:
    """Translate a pre-training device string after a visibility remap.

    Training on ``cuda:1`` sets ``CUDA_VISIBLE_DEVICES=1``, which makes that
    physical GPU appear as ``cuda:0`` for the rest of the process.  Anything
    loaded afterwards (the GNN, the comparison adapters) has to use the
    remapped name or it will not find a device.
    """
    if _REMAPPED_INDEX is None:
        return device
    if device in {"cuda", f"cuda:{_REMAPPED_INDEX}"}:
        return "cuda"
    if device.startswith("cuda:"):
        raise ValueError(
            f"device {device} is no longer visible after remapping "
            f"CUDA_VISIBLE_DEVICES={_REMAPPED_INDEX}; pass cuda or "
            f"cuda:{_REMAPPED_INDEX} instead"
        )
    return device


def _normalise_cuda_device(device: str) -> str:
    """Avoid Laya multi-index placement bugs.

    Laya's trainer reliably uses the logical ``cuda`` device.  When the caller
    requests a physical GPU and CUDA visibility is not already configured,
    remap that GPU to logical index zero before importing Laya/PyTorch.

    When ``CUDA_VISIBLE_DEVICES`` is already set by the caller the index is
    valid *relative to that selection* and must be passed through unchanged —
    rewriting it to bare ``cuda`` would silently move the job to GPU 0.
    """
    global _REMAPPED_INDEX
    if not device.startswith("cuda:"):
        return device
    index = device[5:]
    if not index.isdigit():
        raise ValueError(f"invalid CUDA device {device!r}")
    if _REMAPPED_INDEX == index and os.environ.get("CUDA_VISIBLE_DEVICES") == index:
        return "cuda"  # already remapped earlier in this process
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = index
        _REMAPPED_INDEX = index
        return "cuda"
    return device
