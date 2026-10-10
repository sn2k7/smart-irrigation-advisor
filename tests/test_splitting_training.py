"""Leakage-safe splitting, training workflow and the saved model bundle."""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
import pytest

from irrigation_advisor.data.features import FEATURE_COLUMNS, build_features
from irrigation_advisor.data.labels import HORIZON_HOURS, TARGET_CLASS, derive_labels_from_observations
from irrigation_advisor.data.preprocessing import prepare_sensor_frame
from irrigation_advisor.exceptions import ModelNotAvailableError, ModelSchemaMismatchError
from irrigation_advisor.ml.evaluation import best_threshold, classification_metrics, regression_metrics
from irrigation_advisor.ml.model_store import IrrigationModel, load_model, try_load_model
from irrigation_advisor.ml.splitting import assert_no_leakage, rolling_origin_folds, split_dataset
from irrigation_advisor.ml.training import RULE


def _frame(n_fields=5, hours=500):
    rows = []
    for f in range(n_fields):
        ts = pd.date_range("2025-01-01", periods=hours, freq="h") + pd.Timedelta(days=10 * f)
        rows.append(pd.DataFrame({"field_id": f"F{f}", "timestamp": ts, TARGET_CLASS: np.arange(hours) % 2}))
    return pd.concat(rows, ignore_index=True)


def test_split_has_embargo_and_disjoint_holdout_fields():
    df = _frame()
    split = split_dataset(df, n_folds=3, holdout_field_fraction=0.2)
    gap = pd.Timedelta(hours=HORIZON_HOURS)
    assert split.development["timestamp"].max() + gap <= split.test_temporal["timestamp"].min()
    assert set(split.test_unseen_fields["field_id"]).isdisjoint(split.development["field_id"])
    assert len(split.folds) == 3
    for train_idx, val_idx in split.folds:
        assert df.loc[train_idx, "timestamp"].max() + gap <= df.loc[val_idx, "timestamp"].min()
    # expanding window: each fold trains on more data
    sizes = [len(t) for t, _ in split.folds]
    assert sizes == sorted(sizes)


def test_leakage_check_detects_overlap():
    df = _frame()
    split = split_dataset(df)
    split.test_temporal = pd.concat([split.test_temporal, split.development.tail(5)])
    with pytest.raises(AssertionError):
        assert_no_leakage(split)


def test_split_rejects_short_embargo_and_bad_fraction():
    with pytest.raises(ValueError):
        split_dataset(_frame(), embargo_hours=HORIZON_HOURS - 1)
    with pytest.raises(ValueError):
        split_dataset(_frame(), test_fraction=1.5)
    with pytest.raises(ValueError):
        rolling_origin_folds(_frame(), n_folds=0)


def test_metric_helpers():
    y = np.array([0, 0, 1, 1])
    m = classification_metrics(y, np.array([0.1, 0.6, 0.7, 0.9]), 0.5)
    assert m["confusion_matrix"] == {"tn": 1, "fp": 1, "fn": 0, "tp": 2}
    assert m["precision"] == pytest.approx(2 / 3, abs=1e-4) and m["recall"] == 1.0
    assert 0.5 < best_threshold(y, np.array([0.1, 0.6, 0.7, 0.9])) <= 0.7
    r = regression_metrics(np.array([1.0, 3.0]), np.array([2.0, 3.0]))
    assert r["mae_mm"] == 0.5 and r["rmse_mm"] == pytest.approx(np.sqrt(0.5), abs=1e-3)


def test_training_report_contains_all_models_and_valid_metrics(trained):
    result, model_path, reports_dir = trained
    report = result.report
    assert set(report["classification"]) == {"logistic_regression", "random_forest", "gradient_boosting", RULE}
    for name, sets in report["classification"].items():
        for set_name in ("cross_validation", "test_temporal"):
            m = sets[set_name]
            for key in ("accuracy", "precision", "recall", "f1"):
                assert 0.0 <= m[key] <= 1.0
            cm = m["confusion_matrix"]
            assert sum(cm.values()) == m["n"]
    assert set(report["regression"]) == {"ridge", "random_forest", "gradient_boosting", RULE}
    assert report["regression"][RULE]["test_temporal"]["mae_mm"] >= 0
    assert "SYNTHETIC" in report["disclaimer"]
    assert (reports_dir / "evaluation_report.md").exists()
    assert (reports_dir / "confusion_matrices_test_temporal.png").exists()


def test_selection_is_consistent_with_cross_validation(trained):
    report = trained[0].report
    sel = report["selected"]
    cv = {n: v["cross_validation"]["f1"] for n, v in report["classification"].items()}
    if sel["classifier"] == RULE:
        assert cv[RULE] >= max(v for n, v in cv.items() if n != RULE)
    else:
        assert cv[sel["classifier"]] == max(cv.values())


def test_saved_bundle_round_trip_and_predictions(trained, trained_model, small_dataset):
    result = trained[0]
    if result.bundle["classifier"] is None and result.bundle["regressor"] is None:
        pytest.skip("rule won both tasks on the tiny dataset")
    feats = build_features(prepare_sensor_frame(small_dataset.head(200)))
    preds = trained_model.predict(feats)
    assert len(preds) == len(feats)
    for p in preds:
        if trained_model.has_classifier:
            assert 0.0 <= p.probability <= 1.0
        if p.net_depth_mm is not None:
            assert p.net_depth_mm >= 0


def test_schema_mismatch_is_rejected(trained, tmp_path):
    bundle = dict(joblib.load(trained[1]))
    bundle["feature_schema_version"] = "0.0"
    with pytest.raises(ModelSchemaMismatchError):
        IrrigationModel(bundle)
    bad = tmp_path / "bad.joblib"
    joblib.dump(bundle, bad)
    assert try_load_model(bad) is None


def test_missing_model_file(tmp_path):
    with pytest.raises(ModelNotAvailableError):
        load_model(tmp_path / "nope.joblib")
    assert try_load_model(tmp_path / "nope.joblib") is None


def test_weak_labels_for_external_data(small_dataset):
    feats = build_features(prepare_sensor_frame(small_dataset[small_dataset["field_id"] == "F001"].head(300)))
    labelled = derive_labels_from_observations(feats)
    assert labelled[TARGET_CLASS].iloc[-24:].isna().all()  # no full future window
    valid = labelled.dropna(subset=[TARGET_CLASS])
    assert set(valid[TARGET_CLASS].unique()) <= {0.0, 1.0}
    stressed = valid["depletion_mm"] >= valid["raw_mm"]
    assert (valid.loc[stressed, TARGET_CLASS] == 1).all()
    assert not [c for c in FEATURE_COLUMNS if c in (TARGET_CLASS,)]
