"""Application bootstrap: ``python -m irrigation_gui [--api-url URL]``."""

from __future__ import annotations

import argparse
import logging
import os
import sys

from PySide6.QtWidgets import QApplication

from .api_client import DEFAULT_URL, ApiClient
from .main_window import MainWindow
from .theme import STYLE_SHEET


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="irrigation_gui", description="Smart Irrigation Advisor desktop app")
    parser.add_argument("--api-url", default=os.getenv("SIA_API_URL", DEFAULT_URL),
                        help=f"backend URL (default {DEFAULT_URL} or $SIA_API_URL)")
    parser.add_argument("--timeout", type=float, default=20.0, help="HTTP timeout in seconds")
    parser.add_argument("--screenshot", metavar="PNG", help=argparse.SUPPRESS)  # used for docs/tests
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    app = QApplication.instance() or QApplication(sys.argv[:1])
    app.setApplicationName("Smart Irrigation Advisor")
    app.setStyleSheet(STYLE_SHEET)
    window = MainWindow(ApiClient(args.api_url, args.timeout))
    window.show()
    window.start()
    return app.exec()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
