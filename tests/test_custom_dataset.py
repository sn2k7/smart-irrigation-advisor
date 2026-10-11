"""Custom-dataset slot: JSON config, column mapping, unit conversion, hooks."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from irrigation_advisor.config import PROJECT_ROOT
from irrigation_advisor.data import dataset_hooks
from irrigation_advisor.data.custom_dataset import DatasetConfig, load_custom_dataset, load_dataset_config
from irrigation_advisor.data.labels import TARGET_CLASS, TARGET_REG
from irrigation_advisor.exceptions import DataValidationError


def write_csv(tmp_path: Path, hours: int = 48) -> Path:
    ts = pd.date_range("2026-07-01", periods=hours, freq="h")
    df = pd.DataFrame({
        "When": ts.strftime("%d/%m/%Y %H:%M"),
        "Node": ["A"] * (hours // 2) + ["B"] * (hours - hours // 2),
        "Moist": np.linspace(0.25, 0.20, hours),             # fraction
        "TempF": np.full(hours, 86.0),                        # 30 °C
        "Hum": np.full(hours, 0.55),                          # 0-1 humidity
        "RainCum": np.r_[np.zeros(10), np.full(hours - 10, 0.5)],  # inches, cumulative
        "Needed": [0, 1] * (hours // 2),
        "Junk": "x",
    })
    path = tmp_path / "foreign.csv"
    df.to_csv(path, index=False)
    return path


def make_config(path: Path, **overrides) -> DatasetConfig:
    base = dict(
        name="t", path=path,
        column_map={"timestamp": "When", "field_id": "Node", "soil_moisture": "Moist", "temperature_c": "TempF",
                    "humidity_pct": "Hum", "rainfall_mm": "RainCum"},
        timestamp_format="%d/%m/%Y %H:%M", soil_moisture_unit="fraction", temperature_unit="fahrenheit",
        rainfall_unit="inch", rainfall_mode="cumulative",
        field_defaults={"crop_type": "Maize", "soil_type": "loam", "irrigation_method": "drip",
                        "planting_date": "2026-06-01", "latitude": 18.5},
        per_field_metadata={"B": {"crop_type": "tomato"}},
    )
    base.update(overrides)
    return DatasetConfig(**base)


def test_mapping_and_unit_conversion(tmp_path):
    df, report = load_custom_dataset(make_config(write_csv(tmp_path)))
    assert list(df["field_id"].unique()) == ["A", "B"]
    assert df["soil_moisture"].iloc[0] == pytest.approx(25.0)
    np.testing.assert_allclose(df["temperature_c"], 30.0)
    np.testing.assert_allclose(df["humidity_pct"], 55.0)
    # cumulative 0.5 inch jump -> 12.7 mm once, then zero
    a = df[df["field_id"] == "A"]
    assert a["rainfall_mm"].sum() == pytest.approx(12.7)
    assert set(df.loc[df["field_id"] == "B", "crop_type"]) == {"tomato"}
    assert set(df.loc[df["field_id"] == "A", "crop_type"]) == {"maize"}
    assert "Junk" not in df.columns and TARGET_CLASS not in df.columns
    assert report["rows_after_conversion"] == 48 and report["fields"] == 2


def test_label_columns_mode(tmp_path):
    cfg = make_config(write_csv(tmp_path), labels_mode="columns",
                      column_map={**make_config(tmp_path / "x").column_map, TARGET_CLASS: "Needed"})
    df, report = load_custom_dataset(cfg)
    assert set(df[TARGET_CLASS].unique()) == {0, 1}
    assert df.loc[df[TARGET_CLASS] == 0, TARGET_REG].eq(0).all()
    assert df.loc[df[TARGET_CLASS] == 1, TARGET_REG].isna().all()  # unknown depth -> regression skipped
    assert report["positive_rate"] == 0.5


def test_raw_adc_calibration(tmp_path):
    path = tmp_path / "raw.csv"
    pd.DataFrame({"t": ["2026-07-01 00:00", "2026-07-01 01:00"], "adc": [3300, 1350]}).to_csv(path, index=False)
    cfg = DatasetConfig(name="raw", path=path, column_map={"timestamp": "t", "soil_moisture": "adc"},
                        soil_moisture_unit="raw",
                        soil_raw_calibration={"dry_raw": 3300, "wet_raw": 1350, "dry_pct": 3, "wet_pct": 45},
                        field_defaults={"crop_type": "onion", "soil_type": "clay", "irrigation_method": "furrow",
                                        "latitude": 20})
    df, _ = load_custom_dataset(cfg)
    assert df["soil_moisture"].tolist() == pytest.approx([3.0, 45.0])
    assert (df["rainfall_mm"] == 0).all() and df["field_id"].eq("field-1").all()
    assert df["planting_date"].iloc[0] == "2026-06-01"  # 30 days before first reading


def test_config_errors_are_explicit(tmp_path):
    path = write_csv(tmp_path)
    with pytest.raises(DataValidationError, match="placeholder"):
        make_config(path, column_map={"timestamp": "<your timestamp column>", "soil_moisture": "Moist"})
    with pytest.raises(DataValidationError, match="not in"):
        make_config(path, temperature_unit="rankine")
    with pytest.raises(DataValidationError, match="not in"):
        load_custom_dataset(make_config(path, column_map={"timestamp": "When", "soil_moisture": "NOPE"}))
    with pytest.raises(DataValidationError, match="Unknown crop_type"):
        load_custom_dataset(make_config(path, field_defaults={"crop_type": "rice", "soil_type": "loam",
                                                              "irrigation_method": "drip", "latitude": 1}))
    with pytest.raises(DataValidationError, match="metadata missing"):
        load_custom_dataset(make_config(path, field_defaults={}, per_field_metadata={}))


def test_template_is_rejected_until_filled_and_sample_parses(tmp_path):
    with pytest.raises(DataValidationError):
        load_dataset_config(PROJECT_ROOT / "config" / "dataset_config.example.json")
    sample = load_dataset_config(PROJECT_ROOT / "config" / "dataset_config.sample.json")
    assert sample.rainfall_mode == "cumulative" and sample.soil_moisture_unit == "fraction"
    # a filled-in copy of the template loads
    template = json.loads((PROJECT_ROOT / "config" / "dataset_config.example.json").read_text())
    template["path"] = str(write_csv(tmp_path))
    template["column_map"] = {"timestamp": "When", "field_id": "Node", "soil_moisture": "Moist"}
    template["units"] = {"soil_moisture": "fraction"}
    template["fields"] = {"defaults": {"crop_type": "maize", "soil_type": "loam", "irrigation_method": "drip",
                                       "latitude": 18.5}}
    cfg_path = tmp_path / "mine.json"
    cfg_path.write_text(json.dumps(template))
    df, _ = load_custom_dataset(load_dataset_config(cfg_path))
    assert len(df) == 48


def test_hooks_are_called(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(dataset_hooks, "before_mapping", lambda df, cfg: calls.append("before") or df[df["Node"] == "A"])
    monkeypatch.setattr(dataset_hooks, "custom_transform",
                        lambda df, cfg: calls.append("after") or df.assign(soil_moisture=df["soil_moisture"] + 1))
    df, _ = load_custom_dataset(make_config(write_csv(tmp_path)))
    assert calls == ["before", "after"]
    assert df["field_id"].unique().tolist() == ["A"]
    assert df["soil_moisture"].iloc[0] == pytest.approx(26.0)


def test_training_with_own_labels_but_no_depth(small_dataset, tmp_path):
    """Yes/no labels only: classifier trains, depth falls back to the FAO-56 formula."""
    from irrigation_advisor.ml.training import RULE, TrainingConfig, train_models

    src = small_dataset[["timestamp", "field_id", "soil_moisture", "temperature_c", "humidity_pct", "rainfall_mm",
                         "crop_type", "soil_type", "irrigation_method", "planting_date", "latitude",
                         TARGET_CLASS]].rename(columns={"soil_moisture": "sm", TARGET_CLASS: "needs_water"})
    path = tmp_path / "own_labels.csv"
    src.to_csv(path, index=False)
    cols = {c: c for c in ("timestamp", "field_id", "temperature_c", "humidity_pct", "rainfall_mm", "crop_type",
                           "soil_type", "irrigation_method", "planting_date", "latitude")}
    cfg = DatasetConfig(name="own", path=path, labels_mode="columns",
                        column_map={**cols, "soil_moisture": "sm", TARGET_CLASS: "needs_water"})
    df, _ = load_custom_dataset(cfg)
    result = train_models(df, TrainingConfig(fast=True, train_stride_hours=3, importance_sample=300,
                                             data_source="custom"))
    assert result.report["selected"]["regressor"] == RULE
    assert result.bundle["regressor"] is None
    assert "weak labels" in result.report["disclaimer"] or "user-supplied" in result.report["disclaimer"]
