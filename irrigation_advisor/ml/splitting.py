"""Leakage-safe splitting for sensor time series.

Random row-wise splits are wrong for this problem: consecutive hours of the
same field are almost identical, and labels look 24 h ahead, so a random
split would put near-duplicates of test rows (and their futures) into the
training set and inflate every metric.

Instead we use:

* a **held-out set of fields** that never appear in development, giving a
  test of generalisation to unseen farms;
* a **chronological test period** at the end of the timeline for the
  remaining fields;
* **rolling-origin (expanding-window) cross-validation** inside the
  development period for model selection and threshold tuning;
* an **embargo gap** (>= the 24 h label horizon) between every training and
  evaluation block, so no training label window overlaps evaluation data.

The test sets are used exactly once, after selection.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..data.labels import HORIZON_HOURS


@dataclass
class DataSplit:
    """Result of :func:`split_dataset` (frames keep their original index)."""

    development: pd.DataFrame
    test_temporal: pd.DataFrame
    test_unseen_fields: pd.DataFrame
    #: (train_index, validation_index) label arrays into ``development``.
    folds: list[tuple[np.ndarray, np.ndarray]]
    info: dict = field(default_factory=dict)


def rolling_origin_folds(
    df: pd.DataFrame, n_folds: int = 3, embargo_hours: int = HORIZON_HOURS
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Expanding-window folds: train on blocks 0..k, validate on block k+1.

    The time range is cut into ``n_folds + 1`` equal blocks; an embargo gap
    is removed from the end of every training window.
    """
    if n_folds < 1:
        raise ValueError("n_folds must be >= 1")
    t_min, t_max = df["timestamp"].min(), df["timestamp"].max()
    edges = [t_min + (t_max - t_min) * i / (n_folds + 1) for i in range(n_folds + 2)]
    gap = pd.Timedelta(hours=embargo_hours)
    folds = []
    for k in range(1, n_folds + 1):
        train_idx = df.index[df["timestamp"] < edges[k] - gap].to_numpy()
        upper = df["timestamp"] <= edges[k + 1] if k == n_folds else df["timestamp"] < edges[k + 1]
        val_idx = df.index[(df["timestamp"] >= edges[k]) & upper].to_numpy()
        if len(train_idx) and len(val_idx):
            folds.append((train_idx, val_idx))
    return folds


def split_dataset(
    df: pd.DataFrame,
    test_fraction: float = 0.25,
    n_folds: int = 3,
    embargo_hours: int = HORIZON_HOURS,
    holdout_field_fraction: float = 0.2,
    random_state: int = 42,
) -> DataSplit:
    """Held-out fields + chronological test period + rolling-origin CV folds."""
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be in (0, 1)")
    if embargo_hours < HORIZON_HOURS:
        raise ValueError(f"embargo_hours must be >= label horizon ({HORIZON_HOURS} h)")

    fields = np.array(sorted(df["field_id"].unique()))
    rng = np.random.default_rng(random_state)
    n_holdout = int(round(len(fields) * holdout_field_fraction))
    n_holdout = min(n_holdout, max(0, len(fields) - 1))
    holdout = set(rng.choice(fields, size=n_holdout, replace=False)) if n_holdout else set()

    seen = df[~df["field_id"].isin(holdout)]
    unseen = df[df["field_id"].isin(holdout)]
    t_min, t_max = seen["timestamp"].min(), seen["timestamp"].max()
    t_test = t_min + (t_max - t_min) * (1 - test_fraction)
    gap = pd.Timedelta(hours=embargo_hours)

    development = seen[seen["timestamp"] < t_test - gap]
    test_temporal = seen[seen["timestamp"] >= t_test]
    folds = rolling_origin_folds(development, n_folds, embargo_hours)

    info = {
        "method": "held-out fields + chronological test period + rolling-origin CV with embargo",
        "embargo_hours": embargo_hours,
        "cv_folds": len(folds),
        "development_end": str(development["timestamp"].max()),
        "test_start": str(test_temporal["timestamp"].min()),
        "holdout_fields": sorted(str(f) for f in holdout),
        "rows": {
            "development": len(development),
            "test_temporal": len(test_temporal),
            "test_unseen_fields": len(unseen),
            "cv_validation_rows": int(sum(len(v) for _, v in folds)),
        },
    }
    split = DataSplit(development, test_temporal, unseen, folds, info)
    assert_no_leakage(split, embargo_hours)
    return split


def assert_no_leakage(split: DataSplit, embargo_hours: int = HORIZON_HOURS) -> None:
    """Raise ``AssertionError`` if any evaluation block overlaps its training data."""
    gap = pd.Timedelta(hours=embargo_hours)
    dev = split.development
    if len(dev) and len(split.test_temporal):
        assert dev["timestamp"].max() + gap <= split.test_temporal["timestamp"].min(), "development/test overlap"
    for train_idx, val_idx in split.folds:
        assert dev.loc[train_idx, "timestamp"].max() + gap <= dev.loc[val_idx, "timestamp"].min(), "CV fold overlap"
    unseen = set(split.test_unseen_fields["field_id"].unique())
    assert not (unseen & set(dev["field_id"].unique())), "held-out fields used in development"
    assert not (unseen & set(split.test_temporal["field_id"].unique())), "held-out fields in temporal test"
