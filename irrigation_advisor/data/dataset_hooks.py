"""YOUR CODE GOES HERE - hooks for a custom dataset.

``data/custom_dataset.py`` calls these two functions while loading the
dataset described by your JSON config. By default they do nothing, so a
plain CSV only needs the JSON config. Edit them when your data needs logic
that a column mapping cannot express.

Contract
--------
* ``before_mapping`` receives the file exactly as read (your column names).
* ``custom_transform`` receives the converted frame in the INTERNAL schema:
  ``timestamp`` (datetime, field-local), ``field_id`` (str),
  ``soil_moisture`` (% VWC), ``temperature_c``, ``humidity_pct``,
  ``rainfall_mm`` (rain since previous row), ``crop_type``, ``soil_type``,
  ``irrigation_method``, ``planting_date`` (YYYY-MM-DD), ``latitude`` and,
  if ``labels.mode == "columns"``, the target columns.
* Both must return a DataFrame. Don't create look-ahead features here -
  features are built later by ``data/features.py`` (shared with inference).
* ``config.extra`` holds anything you put under ``"extra"`` in the JSON.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:  # pragma: no cover
    from .custom_dataset import DatasetConfig


def before_mapping(df: pd.DataFrame, config: "DatasetConfig") -> pd.DataFrame:
    """Raw-file hook, runs before the column mapping.

    Examples of what to put here:
      * merge a second file (e.g. a separate weather-station CSV) on time;
      * pivot "long" data (one row per sensor/variable) into one column per variable:
            df = df.pivot_table(index=["time", "node"], columns="sensor", values="value").reset_index()
      * drop test/calibration rows:  df = df[df["status"] != "TEST"]
    """
    # TODO(dataset): add file-level preparation for your dataset here.
    return df


def custom_transform(df: pd.DataFrame, config: "DatasetConfig") -> pd.DataFrame:
    """Internal-schema hook, runs after units/metadata/labels are applied.

    Examples:
      * per-probe calibration:  df["soil_moisture"] = 1.08 * df["soil_moisture"] - 1.5
      * remove a known bad period:
            df = df[~df["timestamp"].between("2024-03-01", "2024-03-04")]
      * custom labels from an irrigation log (when you have one):
            df["target_irrigation_needed"] = ...
    """
    # TODO(dataset): add dataset-specific corrections here.
    return df
