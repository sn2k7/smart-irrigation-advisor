"""Check and train on YOUR REAL dataset (new, standalone script).

Steps (PowerShell, from the project root):
    1. Put your file in data\\real\\            (PLACEHOLDER(real-dataset))
    2. Fill in config\\real_dataset.json        (every value marked PLACEHOLDER)
    3. python scripts/train_real_dataset.py --check-only
    4. python scripts/train_real_dataset.py              # full training
       python scripts/train_real_dataset.py --fast       # quick look first
    5. If the API is running: POST /api/v1/model/reload (or the button in the app)
"""

from __future__ import annotations

import argparse
import json
import logging

import _bootstrap  # noqa: F401

from irrigation_advisor.config import get_settings
from irrigation_advisor.exceptions import DataValidationError
from irrigation_advisor.logging_config import configure_logging
from irrigation_advisor.real_dataset_workflow import (
    DEFAULT_REAL_CONFIG,
    DEFAULT_REAL_MODEL,
    DEFAULT_REAL_REPORTS,
    check_real_dataset,
    train_real_dataset,
)

logger = logging.getLogger("train_real_dataset")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # PLACEHOLDER(real-dataset): change the defaults below if you keep files elsewhere.
    parser.add_argument("--config", default=str(DEFAULT_REAL_CONFIG), help="dataset config JSON")
    parser.add_argument("--model-out", default=str(DEFAULT_REAL_MODEL), help="where to save the trained model")
    parser.add_argument("--reports-dir", default=str(DEFAULT_REAL_REPORTS))
    parser.add_argument("--check-only", action="store_true", help="validate the data, do not train")
    parser.add_argument("--fast", action="store_true", help="fewer trees for a quick first run")
    parser.add_argument("--stride", type=int, default=1, help="use every n-th hour for fitting")
    args = parser.parse_args()
    configure_logging(get_settings().log_level)

    try:
        check = check_real_dataset(args.config)
    except DataValidationError as exc:
        logger.error("Fix the dataset config/file first: %s", exc)
        return 1
    print(json.dumps({"conversion": check.conversion, "quality": check.quality}, indent=2, default=str))
    for warning in check.warnings:
        logger.warning(warning)
    for problem in check.problems:
        logger.error(problem)
    print("\nPreview (internal schema):")
    print(check.preview.to_string(index=False))
    if args.check_only or not check.ok:
        return 0 if check.ok else 2

    summary = train_real_dataset(args.config, fast=args.fast, model_path=args.model_out,
                                 reports_dir=args.reports_dir, stride_hours=args.stride)
    logger.info("Selection: %s", summary["selected"]["notes"])
    for name, m in summary["classification"].items():
        if m:
            logger.info("  %-20s test F1 %.3f precision %.3f recall %.3f", name, m["f1"], m["precision"], m["recall"])
    logger.info("Artifacts: %s", json.dumps(summary["artifacts"], indent=2))
    logger.info(summary["disclaimer"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
