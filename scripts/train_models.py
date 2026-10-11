"""Train, evaluate and save the irrigation models.

Usage (PowerShell):
    python scripts/train_models.py                          # uses data/synthetic_irrigation_dataset.csv
    python scripts/train_models.py --generate --fields 30    # generate data first
    python scripts/train_models.py --external-csv my.csv --crop maize --soil loam
    python scripts/train_models.py --fast                   # quick run (fewer trees)
    python scripts/train_models.py --dataset-config config/my_dataset.json   # YOUR dataset (see config/)
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import _bootstrap  # noqa: F401

from irrigation_advisor.config import get_settings
from irrigation_advisor.data.custom_dataset import load_custom_dataset, load_dataset_config
from irrigation_advisor.data.loader import load_dataset, load_external_csv, save_dataset
from irrigation_advisor.data.simulator import generate_dataset
from irrigation_advisor.logging_config import configure_logging
from irrigation_advisor.ml.training import TrainingConfig, train_models

logger = logging.getLogger("train_models")


def main() -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default=str(settings.data_dir / "synthetic_irrigation_dataset.csv"))
    parser.add_argument("--generate", action="store_true", help="generate the synthetic dataset first")
    parser.add_argument("--fields", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset-config", help="JSON config describing YOUR dataset (config/dataset_config.example.json)")
    parser.add_argument("--external-csv", help="train on an external CSV with standard/alias column names (weak labels)")
    parser.add_argument("--crop", help="crop for external CSV rows without crop_type")
    parser.add_argument("--soil", help="soil for external CSV rows without soil_type")
    parser.add_argument("--fast", action="store_true", help="fewer trees (quick demo/CI)")
    parser.add_argument("--stride", type=int, default=2, help="use every n-th hour of the train split")
    parser.add_argument("--model-out", default=str(settings.model_path))
    parser.add_argument("--reports-dir", default=str(settings.reports_dir))
    args = parser.parse_args()
    configure_logging(settings.log_level)

    source = "synthetic"
    if args.dataset_config:
        raw, conversion = load_custom_dataset(load_dataset_config(args.dataset_config))
        logger.info("Custom dataset: %s", json.dumps(conversion, indent=2, default=str))
        source = "custom"
    elif args.external_csv:
        defaults = {k: v for k, v in {"crop_type": args.crop, "soil_type": args.soil}.items() if v}
        raw = load_external_csv(args.external_csv, defaults)
        source = "external"
    elif args.generate or not Path(args.data).exists():
        logger.info("Generating synthetic dataset (%d fields, seed %d)", args.fields, args.seed)
        raw, _ = generate_dataset(args.fields, args.seed)
        save_dataset(raw, args.data)
    else:
        raw = load_dataset(args.data)

    config = TrainingConfig(random_state=args.seed, fast=args.fast, train_stride_hours=args.stride, data_source=source)
    result = train_models(raw, config, Path(args.model_out), Path(args.reports_dir))

    sel = result.report["selected"]
    logger.info("Selected classifier %s (threshold %.3f) and regressor %s", sel["classifier"], sel["threshold"], sel["regressor"])
    for name, res in result.report["classification"].items():
        t = res.get("test_temporal", {})
        logger.info("  %-20s test F1 %.3f  precision %.3f  recall %.3f", name, t.get("f1", 0), t.get("precision", 0), t.get("recall", 0))
    for name, res in result.report["regression"].items():
        t = res.get("test_temporal", {})
        logger.info("  %-20s test MAE %s mm  RMSE %s mm", name, t.get("mae_mm"), t.get("rmse_mm"))
    logger.info("Artifacts:\n%s", json.dumps(result.artifacts, indent=2))
    logger.info(result.report["disclaimer"])


if __name__ == "__main__":
    main()
