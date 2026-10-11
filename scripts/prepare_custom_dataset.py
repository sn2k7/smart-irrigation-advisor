"""Check and convert YOUR dataset using a JSON config (no training).

Usage (PowerShell):
    python scripts/prepare_custom_dataset.py --config config\\my_dataset.json
    python scripts/prepare_custom_dataset.py --config config\\dataset_config.sample.json --out data\\prepared.csv

Prints a conversion report (rows, fields, time range, missing values,
value ranges) and a data-quality summary after the shared cleaning step,
then optionally writes the converted readings in the internal schema.
"""

from __future__ import annotations

import argparse
import json
import logging

import _bootstrap  # noqa: F401

from irrigation_advisor.config import get_settings
from irrigation_advisor.data.custom_dataset import load_custom_dataset, load_dataset_config
from irrigation_advisor.data.loader import save_dataset
from irrigation_advisor.data.preprocessing import data_quality_report, prepare_sensor_frame
from irrigation_advisor.exceptions import DataValidationError
from irrigation_advisor.logging_config import configure_logging

logger = logging.getLogger("prepare_custom_dataset")


def main() -> int:
    settings = get_settings()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="JSON dataset config")
    parser.add_argument("--out", help="write converted readings (internal schema) to this CSV")
    args = parser.parse_args()
    configure_logging(settings.log_level)
    try:
        df, report = load_custom_dataset(load_dataset_config(args.config))
        quality = data_quality_report(prepare_sensor_frame(df))
    except DataValidationError as exc:
        logger.error("Dataset problem: %s", exc)
        return 1
    print(json.dumps({"conversion": report, "quality_after_cleaning": quality}, indent=2, default=str))
    print("\nFirst rows in the internal schema:")
    print(df.head(5).to_string(index=False))
    if args.out:
        logger.info("Converted readings -> %s", save_dataset(df, args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
