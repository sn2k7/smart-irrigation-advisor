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
    parser.add_argument("--export-foreign", metavar="CSV",
                        help="also write the readings in a 'foreign' format to exercise the custom-dataset slot")
    args = parser.parse_args()
    configure_logging(settings.log_level)

    data, specs = generate_dataset(args.fields, args.seed, args.climate, args.year, args.fault_rate)
    path = save_dataset(data, args.out)
    specs_path = save_dataset(specs_to_frame(specs), str(path).replace(".csv", "_fields.csv"))
    logger.info("SYNTHETIC dataset: %d rows, %d fields -> %s", len(data), data["field_id"].nunique(), path)
    logger.info("Field specifications -> %s", specs_path)
    logger.info("Irrigation-needed rate: %.1f%%", 100 * data["target_irrigation_needed"].mean())
    logger.info("Injected sensor faults: %s", data["sensor_fault"].value_counts().to_dict())
    if args.export_foreign:
        foreign = export_foreign_format(data)
        save_dataset(foreign, args.export_foreign)
        logger.info("Foreign-format copy (no labels) -> %s  (see config/dataset_config.sample.json)", args.export_foreign)


def export_foreign_format(data):
    """Same readings with other names/units: what a third-party logger export might look like."""
    import pandas as pd

    out = pd.DataFrame({
        "DateTime": pd.to_datetime(data["timestamp"]).dt.strftime("%d/%m/%Y %H:%M"),
        "NodeID": data["field_id"],
        "VWC_frac": (data["soil_moisture"] / 100.0).round(4),
        "AirTemp_F": (data["temperature_c"] * 9 / 5 + 32).round(1),
        "RH": data["humidity_pct"],
        "Crop": data["crop_type"].str.upper(),
        "SoilTexture": data["soil_type"],
        "Sown": data["planting_date"],
    })
    rain_in = data["rainfall_mm"].fillna(0) / 25.4
    out["RainTotal_in"] = rain_in.groupby(data["field_id"]).cumsum().round(4)
    return out


if __name__ == "__main__":
    main()
