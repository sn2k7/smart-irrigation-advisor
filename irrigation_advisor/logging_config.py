"""Central logging configuration (console + rotating file)."""

from __future__ import annotations

import logging
import logging.config
from pathlib import Path

_CONFIGURED = False


def configure_logging(level: str = "INFO", log_dir: Path | None = None) -> None:
    """Configure root logging once.

    Args:
        level: Log level name such as ``"INFO"`` or ``"DEBUG"``.
        log_dir: If given, a rotating ``app.log`` file is written there.
    """
    global _CONFIGURED
    if _CONFIGURED:
        logging.getLogger().setLevel(level)
        return

    handlers: dict[str, dict] = {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "standard",
            "level": level,
        }
    }
    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        handlers["file"] = {
            "class": "logging.handlers.RotatingFileHandler",
            "formatter": "standard",
            "level": level,
            "filename": str(log_dir / "app.log"),
            "maxBytes": 2_000_000,
            "backupCount": 3,
            "encoding": "utf-8",
        }

    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "standard": {"format": "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"}
            },
            "handlers": handlers,
            "root": {"handlers": list(handlers), "level": level},
        }
    )
    _CONFIGURED = True
