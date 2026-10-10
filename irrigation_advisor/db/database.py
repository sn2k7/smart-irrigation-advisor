"""SQLite connection management.

A new connection is opened per unit of work (per API request). SQLite
connections are cheap, and this avoids sharing a connection between the
threads FastAPI uses for sync endpoints. WAL mode lets the IoT ingest and
dashboard reads proceed concurrently.
"""

from __future__ import annotations

import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Iterator

logger = logging.getLogger(__name__)

SCHEMA_PATH = Path(__file__).with_name("schema.sql")


class Database:
    """Owns the database file path and creates connections."""

    def __init__(self, path: Path | str) -> None:
        # A file path is required: ":memory:" would give every connection its own empty DB.
        if str(path) == ":memory:":
            raise ValueError("Use a file path for the database (tests use a temporary file)")
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.path), timeout=15, detect_types=0, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA synchronous = NORMAL")
        return conn

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        """Connection that commits on success and rolls back on error."""
        conn = self.connect()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def initialise(self) -> None:
        """Create tables and indexes (idempotent)."""
        with self.connection() as conn:
            conn.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
        logger.info("Database ready at %s", self.path)

    def ping(self) -> bool:
        try:
            with self.connection() as conn:
                conn.execute("SELECT 1").fetchone()
            return True
        except sqlite3.Error:
            return False


def now_iso() -> str:
    """Current local time as ISO text with seconds precision."""
    return datetime.now().isoformat(timespec="seconds")


def to_iso(ts: datetime) -> str:
    """Naive datetime -> ISO text (seconds precision) used as the storage format."""
    return ts.replace(microsecond=0, tzinfo=None).isoformat(timespec="seconds")
