"""Progress reporting helpers shared by record preparation and fine-tuning.

Two problems are solved here:

* **Silence.** ``laya.train.finetune`` prints nothing while it reads and encodes a
  large JSONL file, so a run looks hung for minutes.  :func:`heartbeat` keeps the
  console alive during that window, and :class:`LineFilter` turns the training
  lines it does print (``epoch 1/4 step 100 loss 1.23``) into a progress bar.
* **Optional dependencies.**  ``tqdm`` is used when installed and a minimal
  stand-in is used otherwise, so preparation still runs in a bare environment.

Nothing in this module imports torch or laya.
"""

from __future__ import annotations

import re
import sys
import threading
import time
from typing import Any, IO, Iterable, Iterator, Optional

try:  # pragma: no cover - exercised implicitly by whichever environment runs
    from tqdm import tqdm as _tqdm
except ImportError:  # pragma: no cover
    _tqdm = None


class _FallbackBar:
    """A small tqdm-shaped counter used when tqdm is unavailable."""

    def __init__(self, *, total: Optional[int] = None, desc: str = "",
                 file: Optional[IO[str]] = None, **_ignored: Any) -> None:
        self.total = total
        self.desc = desc
        self.file = file if file is not None else sys.stderr
        self.n = 0
        self._last_print = 0.0
        self._closed = False

    def update(self, n: int = 1) -> None:
        self.n += n
        now = time.monotonic()
        if now - self._last_print >= 2.0:
            self._last_print = now
            self.refresh()

    def set_postfix(self, **kwargs: Any) -> None:
        self._postfix = ", ".join(f"{k}={v}" for k, v in kwargs.items())
        self.refresh()

    def refresh(self) -> None:
        if self._closed:
            return
        total = f"/{self.total}" if self.total else ""
        suffix = getattr(self, "_postfix", "")
        line = f"{self.desc}: {self.n}{total} {suffix}".rstrip()
        print(f"\r{line}", end="", file=self.file, flush=True)

    def write(self, message: str, file: Optional[IO[str]] = None, **_kw: Any) -> None:
        print(message, file=file if file is not None else self.file, flush=True)

    def close(self) -> None:
        if not self._closed:
            print(file=self.file)
            self._closed = True


def bar(iterable: Optional[Iterable[Any]] = None, *, total: Optional[int] = None,
        desc: str = "", disable: bool = False, **kwargs: Any) -> Any:
    """Return a tqdm bar (or the stand-in) over ``iterable``."""
    if disable:
        return _NullBar(iterable)
    if _tqdm is not None:
        return _tqdm(iterable, total=total, desc=desc, **kwargs)
    return _FallbackBar(total=total, desc=desc, **kwargs)


class _NullBar:
    """A bar that does nothing, used with ``--quiet``."""

    def __init__(self, iterable: Optional[Iterable[Any]] = None) -> None:
        self._iterable = iterable

    def __iter__(self) -> Iterator[Any]:
        return iter(self._iterable or ())

    def update(self, n: int = 1) -> None:
        return None

    def set_postfix(self, **kwargs: Any) -> None:
        return None

    def refresh(self) -> None:
        return None

    def write(self, message: str, file: Optional[IO[str]] = None, **_kw: Any) -> None:
        print(message, file=file if file is not None else sys.stdout, flush=True)

    def close(self) -> None:
        return None


EPOCH_RE = re.compile(r"^epoch (\d+)/(\d+) step (\d+) loss ([0-9.]+)\s*$")
EPOCH_MEAN_RE = re.compile(r"^epoch (\d+)/(\d+) mean loss ([0-9.]+)\s*$")
ITEMS_RE = re.compile(r"^train items (\d+), calibration items (\d+), eval items (\d+)")


class LineFilter:
    """Filter stdout coming from ``laya.train.finetune`` into a progress bar.

    ``finetune`` writes plain newline-terminated lines.  Lines that describe the
    training loop update the bar and are not repeated on stdout; every other line
    is forwarded untouched, through ``tqdm.write`` while a bar is active so the
    bar is not corrupted.

    ``on_activity`` is called whenever a line arrives, which lets the heartbeat
    know the run is still making progress.
    """

    def __init__(self, stream: IO[str], *, progress: bool = True,
                 epochs: Optional[int] = None, micro_batch: int = 8,
                 on_activity: Optional[Any] = None) -> None:
        self._stream = stream
        self._progress = progress
        #: Called on every line the wrapped process prints (wire a heartbeat to it).
        self.on_activity = on_activity
        self._buffer = ""
        self._bar = None
        self._total: Optional[int] = None
        self._epochs = epochs or 1
        self._micro_batch = max(1, micro_batch)
        self._steps_per_epoch: Optional[int] = None
        self._raw = not progress

    # -- file-like interface used by contextlib.redirect_stdout -------------
    def write(self, text: str) -> int:
        if "\r" in text and self._bar is None:
            # A tqdm bar from huggingface_hub: pass it straight through.
            self._emit_raw(text)
            return len(text)
        self._buffer += text
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            self._handle_line(line)
        return len(text)

    def flush(self) -> None:
        if self._buffer:
            line, self._buffer = self._buffer, ""
            self._handle_line(line)
        if self._bar is not None:
            self._bar.refresh()

    def isatty(self) -> bool:
        return getattr(self._stream, "isatty", lambda: False)()

    @property
    def bar(self) -> Any:
        return self._bar

    # -- internals ----------------------------------------------------------
    def _handle_line(self, line: str) -> None:
        if self.on_activity is not None:
            self.on_activity()
        match = ITEMS_RE.match(line)
        if match and self._progress and self._bar is None:
            train_items = int(match.group(1))
            self._steps_per_epoch = max(1, -(-train_items // self._micro_batch))
            self._total = self._steps_per_epoch * self._epochs
            self._bar = bar(total=self._total, desc="finetune")
            self._emit_through_bar(line)
            return
        match = EPOCH_RE.match(line)
        if match and self._progress:
            if self._bar is None:
                self._bar = bar(total=self._total, desc="finetune")
            epoch, epochs, step, loss = match.groups()
            self._set_total_from_epoch(int(epoch), int(epochs), int(step))
            self._bar.set_postfix(epoch=f"{epoch}/{epochs}", step=step, loss=loss)
            self._bar.refresh()
            return
        match = EPOCH_MEAN_RE.match(line)
        if match and self._bar is not None:
            self._emit_through_bar(line)
            return
        self._emit_through_bar(line)

    def _set_total_from_epoch(self, epoch: int, epochs: int, step: int) -> None:
        if self._bar is None:
            return
        if self._total in (None, 0):
            # ``n_steps`` restarts each epoch, so one reported step bounds the rest.
            self._total = step * epochs
        if self._bar.total != self._total:
            try:
                self._bar.total = self._total
            except Exception:
                pass
        target = (epoch - 1) * (self._total // max(1, epochs)) + step
        self._bar.n = min(target, self._total or target)

    def _emit_through_bar(self, line: str) -> None:
        if self._raw or self._bar is None:
            self._emit_raw(line + "\n")
            return
        try:
            # ``file`` must be the captured stdout: tqdm.write() defaults to
            # sys.stdout, which is this filter itself inside redirect_stdout().
            self._bar.write(line, file=self._stream)
        except TypeError:
            self._emit_raw(line + "\n")

    def _emit_raw(self, text: str) -> None:
        try:
            self._stream.write(text)
            self._stream.flush()
        except Exception:
            pass

    def close(self) -> None:
        if self._buffer:
            line, self._buffer = self._buffer, ""
            self._handle_line(line)
        if self._bar is not None:
            self._bar.close()
            self._bar = None


class heartbeat:
    """Print ``still running`` while a silent stage takes a long time.

    Only prints when nothing has been printed for ``interval`` seconds, so it
    never competes with output the run is already producing.  While a progress
    bar is active the line is routed through it to avoid corrupting the bar.
    """

    def __init__(self, label: str, *, interval: float = 60.0,
                 file: Optional[IO[str]] = None, line_filter: Optional[LineFilter] = None) -> None:
        self.label = label
        self.interval = interval
        self.file = file if file is not None else sys.stderr
        self._line_filter = line_filter
        self._last_activity = time.monotonic()
        self._started = time.monotonic()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def activity(self) -> None:
        self._last_activity = time.monotonic()

    def __enter__(self) -> "heartbeat":
        self._thread = threading.Thread(target=self._run, daemon=True, name="heartbeat")
        self._thread.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def _run(self) -> None:
        while not self._stop.wait(self.interval):
            idle = time.monotonic() - self._last_activity
            if idle < self.interval:
                continue
            elapsed = int(time.monotonic() - self._started)
            message = (f"  [{self.label}] still running — elapsed {format_elapsed(elapsed)}, "
                       f"no output for {max(1, int(idle))}s")
            bar_obj = getattr(self._line_filter, "bar", None) if self._line_filter else None
            if bar_obj is not None:
                try:
                    bar_obj.write(message, file=self.file)
                except TypeError:
                    print(message, file=self.file, flush=True)
            else:
                print(message, file=self.file, flush=True)


def format_elapsed(seconds: float) -> str:
    """Human readable elapsed time, e.g. ``6m12s``."""
    total = int(seconds)
    hours, rest = divmod(total, 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return f"{hours}h{minutes:02d}m"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


#: Process start, so every banner can say how long the run has been going.
_STARTED_AT = time.monotonic()


def banner(title: str, *, file: Optional[IO[str]] = None) -> None:
    """Print a stage marker with the elapsed time, so a long run never lies about progress."""
    target = file if file is not None else sys.stdout
    elapsed = format_elapsed(time.monotonic() - _STARTED_AT)
    print(f"\n=== {title} [+{elapsed}] ===", file=target, flush=True)
