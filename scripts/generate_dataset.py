"""Generate the SYNTHETIC training dataset (and a small external-format sample CSV).

Usage (PowerShell):
    python scripts/generate_dataset.py --fields 30 --seed 42
"""

from __future__ import annotations

import argparse
import logging

import _bootstrap  # noqa: F401

from irrigation_advisor.config import get_settings
from irrigation_advisor.data.loader import save_dataset
from irrigation_advisor.data.simulator import generate_dataset, specs_to_frame
from irrigation_advisor.logging_config import configure_logging

logger = logging.getLogger("generate_dataset")


def main() -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fields", type=int, default=30, help="number of simulated fields (default 30)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--year", type=int, default=2025)
    parser.add_argument("--climate", default=settings.climate_profile)
    parser.add_argument("--fault-rate", type=float, default=1.0, help="0 disables sensor faults")
    parser.add_argument("--out", default=str(settings.data_dir / "synthetic_irrigation_dataset.csv"))
    args = parser.parse_args()
    configure_logging(settings.log_level)

    data, specs = generate_dataset(args.fields, args.seed, args.climate, args.year, args.fault_rate)
    path = save_dataset(data, args.out)
    specs_path = save_dataset(specs_to_frame(specs), str(path).replace(".csv", "_fields.csv"))
    logger.info("SYNTHETIC dataset: %d rows, %d fields -> %s", len(data), data["field_id"].nunique(), path)
    logger.info("Field specifications -> %s", specs_path)
    logger.info("Irrigation-needed rate: %.1f%%", 100 * data["target_irrigation_needed"].mean())
    logger.info("Injected sensor faults: %s", data["sensor_fault"].value_counts().to_dict())


if __name__ == "__main__":
    main()
