"""Closed-loop comparison: fixed schedules vs. the advisor (rules / ML hybrid).

Usage (PowerShell):
    python scripts/run_simulation.py --fields 10
    python scripts/run_simulation.py --fields 10 --no-model     # rule-based advisor only

All outputs are SIMULATED and labelled as such.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import _bootstrap  # noqa: F401

from irrigation_advisor.config import get_settings
from irrigation_advisor.logging_config import configure_logging
from irrigation_advisor.ml.model_store import try_load_model
from irrigation_advisor.simulation.comparison import SIMULATION_DISCLAIMER, run_comparison, summarise, write_simulation_reports

logger = logging.getLogger("run_simulation")


def main() -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fields", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2024, help="different from the training seed (42)")
    parser.add_argument("--interval-days", type=int, default=3, help="fixed schedule interval")
    parser.add_argument("--fault-rate", type=float, default=1.0)
    parser.add_argument("--no-model", action="store_true", help="skip the ML hybrid strategy")
    parser.add_argument("--reports-dir", default=str(settings.reports_dir))
    args = parser.parse_args()
    configure_logging(settings.log_level)

    model = None if args.no_model else try_load_model(settings.model_path)
    if model is None and not args.no_model:
        logger.warning("No trained model found - only rule-based advisor will be simulated")
    table, results = run_comparison(
        n_fields=args.fields, seed=args.seed, model=model, climate_name=settings.climate_profile,
        fault_rate=args.fault_rate, fixed_interval_days=args.interval_days,
    )
    config = {
        "seed": args.seed, "fields": args.fields, "fixed_interval_days": args.interval_days,
        "fault_rate": args.fault_rate, "model_version": model.version if model else None,
        "climate": settings.climate_profile,
    }
    paths = write_simulation_reports(table, results, Path(args.reports_dir), config)
    summary, comparison = summarise(table)
    logger.info("Mean per field:\n%s", summary.to_string())
    logger.info("Versus fixed schedule:\n%s", json.dumps(comparison["vs_baseline"], indent=2))
    logger.info("Reports: %s", json.dumps(paths, indent=2))
    logger.info(SIMULATION_DISCLAIMER)


if __name__ == "__main__":
    main()
