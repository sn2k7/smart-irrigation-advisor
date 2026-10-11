"""Run blocking API calls off the GUI thread.

``TaskRunner.run(fn, *args, on_done=..., on_error=...)`` executes ``fn`` in
Qt's thread pool and delivers the result back on the GUI thread (queued
signal into this QObject, which lives in the GUI thread). With
``synchronous=True`` (used by tests) the call runs inline.
"""

from __future__ import annotations

import itertools
import logging
from typing import Any, Callable

from PySide6.QtCore import QObject, QRunnable, Qt, QThreadPool, Signal, Slot

from .api_client import ApiError

logger = logging.getLogger(__name__)
Callback = Callable[[Any], None]


class _Signals(QObject):
    done = Signal(int, object)
    failed = Signal(int, str)


class _Job(QRunnable):
    def __init__(self, job_id: int, fn: Callable[..., Any], args: tuple, signals: _Signals) -> None:
        super().__init__()
        self.job_id, self.fn, self.args, self.signals = job_id, fn, args, signals

    def run(self) -> None:  # executes in a pool thread
        try:
            result = self.fn(*self.args)
        except ApiError as exc:
            self.signals.failed.emit(self.job_id, str(exc))
        except Exception as exc:  # noqa: BLE001 - surface every error in the UI
            logger.exception("Background task failed")
            self.signals.failed.emit(self.job_id, f"{type(exc).__name__}: {exc}")
        else:
            self.signals.done.emit(self.job_id, result)


def _message(exc: Exception) -> str:
    return str(exc) if isinstance(exc, ApiError) else f"{type(exc).__name__}: {exc}"


class TaskRunner(QObject):
    """Dispatch callables to a thread pool; callbacks run on the GUI thread."""

    busy_changed = Signal(bool)
    error = Signal(str)  # emitted for failures without an on_error handler

    def __init__(self, synchronous: bool = False, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.synchronous = synchronous
        self._pool = QThreadPool.globalInstance()
        self._ids = itertools.count(1)
        self._pending: dict[int, tuple[Callback | None, Callable[[str], None] | None, _Signals]] = {}

    @property
    def busy(self) -> bool:
        return bool(self._pending)

    def run(self, fn: Callable[..., Any], *args: Any, on_done: Callback | None = None,
            on_error: Callable[[str], None] | None = None) -> None:
        if self.synchronous:
            try:
                result = fn(*args)
            except Exception as exc:  # noqa: BLE001
                self._fail(on_error, _message(exc))
            else:
                if on_done:
                    on_done(result)
            return
        job_id = next(self._ids)
        signals = _Signals()
        signals.done.connect(self._on_done, Qt.ConnectionType.QueuedConnection)
        signals.failed.connect(self._on_failed, Qt.ConnectionType.QueuedConnection)
        self._pending[job_id] = (on_done, on_error, signals)
        if len(self._pending) == 1:
            self.busy_changed.emit(True)
        self._pool.start(_Job(job_id, fn, args, signals))

    def _finish(self, job_id: int):
        entry = self._pending.pop(job_id, None)
        if not self._pending:
            self.busy_changed.emit(False)
        return entry

    @Slot(int, object)
    def _on_done(self, job_id: int, result: object) -> None:
        entry = self._finish(job_id)
        if entry and entry[0]:
            entry[0](result)

    @Slot(int, str)
    def _on_failed(self, job_id: int, message: str) -> None:
        entry = self._finish(job_id)
        self._fail(entry[1] if entry else None, message)

    def _fail(self, on_error: Callable[[str], None] | None, message: str) -> None:
        if on_error:
            on_error(message)
        else:
            self.error.emit(message)

    def wait(self, msecs: int = 30000) -> None:
        """Block until all pool jobs finished (tests / shutdown)."""
        self._pool.waitForDone(msecs)
