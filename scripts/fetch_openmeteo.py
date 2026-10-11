"""Download the Open-Meteo dataset and (optionally) train on it.

Parameters: soil moisture, temperature, relative humidity, rain in the last 24 h,
24 h rain forecast, ET0 (FAO-56 Penman-Monteith) and wind speed - for the location
in config/openmeteo_dataset.json (default 11.93975 N, 79.49244 E).

Usage (PowerShell, needs internet):
    python scripts/fetch_openmeteo.py                 # download -> data/real/openmeteo_dataset.csv
    python scripts/fetch_openmeteo.py --train --fast  # ... and train/evaluate on it
    python real_dataset_gui.py --config config/openmeteo_generated_dataset.json   # check/train in the GUI
"""

from __future__ import annotations

import argparse
import json
import logging

import _bootstrap  # noqa: F401

from irrigation_advisor.config import PROJECT_ROOT, get_settings
from irrigation_advisor.data.openmeteo import (
    OpenMeteoClient,
    OpenMeteoError,
    build_dataset,
    load_openmeteo_config,
    write_dataset_config,
)
from irrigation_advisor.exceptions import DataValidationError
from irrigation_advisor.logging_config import configure_logging

logger = logging.getLogger("fetch_openmeteo")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config" / "openmeteo_dataset.json"))
    parser.add_argument("--no-cache", action="store_true", help="ignore previously downloaded responses")
    parser.add_argument("--train", action="store_true", help="train and evaluate after downloading")
    parser.add_argument("--fast", action="store_true", help="quick training (fewer trees)")
    args = parser.parse_args()
    configure_logging(get_settings().log_level)

    try:
        cfg = load_openmeteo_config(args.config)
        client = OpenMeteoClient(cache_dir=None if args.no_cache else cfg.cache_dir)
        df, report = build_dataset(cfg, client)
    except (OpenMeteoError, DataValidationError) as exc:
        logger.error("%s", exc)
        return 1
    cfg.output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cfg.output_csv, index=False)
    dataset_cfg = write_dataset_config(cfg, cfg.output_csv)
    report_path = cfg.output_csv.with_suffix(".report.json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    logger.info("Dataset  -> %s (%d rows)", cfg.output_csv, len(df))
    logger.info("Config   -> %s", dataset_cfg)
    logger.info("Report   -> %s", report_path)

    if args.train:
        from irrigation_advisor.real_dataset_workflow import DEFAULT_REAL_REPORTS, train_real_dataset

        summary = train_real_dataset(dataset_cfg, fast=args.fast, reports_dir=DEFAULT_REAL_REPORTS.parent / "openmeteo")
        logger.info("Selection: %s", summary["selected"]["notes"])
        for name, m in summary["classification"].items():
            if m:
                logger.info("  %-20s test F1 %.3f precision %.3f recall %.3f", name, m["f1"], m["precision"], m["recall"])
        logger.info(summary["disclaimer"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
