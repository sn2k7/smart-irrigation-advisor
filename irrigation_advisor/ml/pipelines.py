"""scikit-learn pipelines.

Each model is a single ``Pipeline(preprocess -> estimator)`` so that
imputation, scaling and one-hot encoding are fitted on the training split
only and are serialised together with the estimator. Inference therefore
applies exactly the same transformations as training.
"""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ..data.features import CATEGORICAL_FEATURES, NUMERIC_FEATURES


def build_preprocessor() -> ColumnTransformer:
    """Median-impute + scale numeric features, impute + one-hot categoricals."""
    numeric = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median", add_indicator=False)),
            ("scale", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        [("num", numeric, NUMERIC_FEATURES), ("cat", categorical, CATEGORICAL_FEATURES)],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_classifiers(random_state: int = 42, fast: bool = False) -> dict[str, Pipeline]:
    """The three required classifiers, each wrapped in its own pipeline."""
    n_trees = 60 if fast else 250
    return {
        "logistic_regression": Pipeline(
            [
                ("preprocess", build_preprocessor()),
                ("model", LogisticRegression(max_iter=3000, C=1.0, class_weight="balanced")),
            ]
        ),
        "random_forest": Pipeline(
            [
                ("preprocess", build_preprocessor()),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=n_trees,
                        max_depth=14,
                        min_samples_leaf=5,
                        class_weight="balanced_subsample",
                        n_jobs=-1,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "gradient_boosting": Pipeline(
            [
                ("preprocess", build_preprocessor()),
                (
                    "model",
                    GradientBoostingClassifier(
                        n_estimators=60 if fast else 200,
                        learning_rate=0.08,
                        max_depth=3,
                        subsample=0.8,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
    }


def build_regressors(random_state: int = 42, fast: bool = False) -> dict[str, Pipeline]:
    """Regressors for the net irrigation depth (mm)."""
    return {
        "ridge": Pipeline([("preprocess", build_preprocessor()), ("model", Ridge(alpha=1.0))]),
        "random_forest": Pipeline(
            [
                ("preprocess", build_preprocessor()),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=60 if fast else 200,
                        max_depth=14,
                        min_samples_leaf=5,
                        n_jobs=-1,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "gradient_boosting": Pipeline(
            [
                ("preprocess", build_preprocessor()),
                (
                    "model",
                    GradientBoostingRegressor(
                        n_estimators=60 if fast else 250,
                        learning_rate=0.06,
                        max_depth=3,
                        subsample=0.8,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
    }
