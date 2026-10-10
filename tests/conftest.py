"""Shared pytest fixtures.

Expensive objects (synthetic dataset, a quickly trained model) are created
once per test session. Every API test gets its own temporary database.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from irrigation_advisor.config import Settings
from irrigation_advisor.data.simulator import generate_dataset
from irrigation_advisor.engine.recommender import FieldConfig
from irrigation_advisor.ml.model_store import ModelPrediction, load_model
from irrigation_advisor.ml.training import TrainingConfig, train_models


@pytest.fixture(scope="session")
def small_dataset() -> pd.DataFrame:
    data, _ = generate_dataset(n_fields=8, seed=7, year=2025)
    return data


@pytest.fixture(scope="session")
def trained(tmp_path_factory: pytest.TempPathFactory, small_dataset: pd.DataFrame):
    """A fast model trained on the small dataset (saved to a temp file)."""
    out = tmp_path_factory.mktemp("model")
    result = train_models(
        small_dataset,
        TrainingConfig(fast=True, train_stride_hours=2, importance_sample=500),
        model_path=out / "model.joblib",
        reports_dir=out / "reports",
    )
    return result, out / "model.joblib", out / "reports"


@pytest.fixture(scope="session")
def trained_model(trained):
    return load_model(trained[1])


@pytest.fixture()
def settings(tmp_path: Path) -> Settings:
    """Isolated settings: temp DB, no model, temp logs."""
    return Settings(
        database_path=tmp_path / "test.db",
        model_path=tmp_path / "missing_model.joblib",
        data_dir=tmp_path,
        reports_dir=tmp_path / "reports",
        log_dir=tmp_path / "logs",
        log_level="WARNING",
    )


def _client(settings: Settings):
    from irrigation_advisor.api.main import create_app

    return TestClient(create_app(settings))


@pytest.fixture()
def client(settings: Settings):
    with _client(settings) as c:
        yield c


@pytest.fixture()
def client_with_model(settings: Settings, trained):
    with _client(replace(settings, model_path=trained[1])) as c:
        yield c


@pytest.fixture()
def field_cfg() -> FieldConfig:
    return FieldConfig(
        field_id="1",
        crop_type="maize",
        soil_type="loam",
        irrigation_method="drip",
        planting_date=date(2025, 6, 1),
        area_m2=2000,
        flow_rate_lpm=120,
        latitude=18.5,
        max_irrigation_minutes=600,
    )


def make_readings(
    end: datetime, hours: int, moisture_start: float, moisture_end: float, rain: float = 0.0
) -> pd.DataFrame:
    """Hourly readings with a linear moisture trend and a diurnal temperature cycle."""
    ts = [end - timedelta(hours=h) for h in range(hours - 1, -1, -1)]
    hour = np.array([t.hour for t in ts])
    return pd.DataFrame(
        {
            "timestamp": ts,
            "soil_moisture": np.round(np.linspace(moisture_start, moisture_end, hours), 2),
            "temperature_c": 26 + 7 * np.sin((hour - 9) / 24 * 2 * np.pi),
            "humidity_pct": 55.0,
            "rainfall_mm": rain,
        }
    )


class StubModel:
    """Deterministic stand-in for IrrigationModel used to test engine logic."""

    def __init__(self, probability: float, depth: float | None = 10.0, threshold: float = 0.5) -> None:
        self.probability = probability
        self.depth = depth
        self.threshold = threshold
        self.version = "stub-1"

    def predict(self, features: pd.DataFrame) -> list[ModelPrediction]:
        return [ModelPrediction(self.probability, self.probability >= self.threshold, self.depth) for _ in range(len(features))]
