from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from defi_security.config import ExperimentProfile
from defi_security.data import (
    CATEGORICAL_FEATURES,
    GRAPH_FEATURES,
    TEMPORAL_FEATURES,
    DataValidationError,
    eligible_point_in_time_features,
)


@dataclass
class ExperimentOutput:
    metrics: dict[str, Any]
    y_test: np.ndarray
    baseline_probability: np.ndarray
    semantic_probability: np.ndarray


def add_temporal_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    result = dataframe.copy()
    result["year"] = result["incident_date"].dt.year
    result["month"] = result["incident_date"].dt.month
    result["day_of_week"] = result["incident_date"].dt.dayofweek
    for column in CATEGORICAL_FEATURES:
        result[column] = result[column].fillna("unknown").astype(str)
    return result


def chronological_split(
    dataframe: pd.DataFrame, train_fraction: float = 0.75
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not 0.5 <= train_fraction < 1:
        raise ValueError("train_fraction must be in [0.5, 1)")
    boundary = int(len(dataframe) * train_fraction)
    if boundary < 20 or len(dataframe) - boundary < 5:
        raise DataValidationError("At least 25 chronologically ordered rows are required")
    train = dataframe.iloc[:boundary].copy()
    test = dataframe.iloc[boundary:].copy()
    if train["incident_date"].max() > test["incident_date"].min():
        raise DataValidationError("Chronological split overlaps in time")
    return train, test


def _estimator(features: list[str], seed: int) -> Pipeline:
    categorical = [feature for feature in features if feature in CATEGORICAL_FEATURES]
    numeric = [feature for feature in features if feature not in categorical]
    transforms: list[tuple[str, Any, list[str]]] = []
    if numeric:
        transforms.append(("numeric", SimpleImputer(strategy="median"), numeric))
    if categorical:
        categorical_pipeline = Pipeline(
            [
                ("impute", SimpleImputer(strategy="most_frequent")),
                (
                    "encode",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                ),
            ]
        )
        transforms.append(("categorical", categorical_pipeline, categorical))
    preprocessing = ColumnTransformer(transforms, remainder="drop")
    model = LGBMClassifier(
        objective="binary",
        n_estimators=250,
        learning_rate=0.04,
        num_leaves=15,
        min_child_samples=8,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=seed,
        n_jobs=1,
        verbosity=-1,
    )
    return Pipeline([("preprocess", preprocessing), ("model", model)])


def _scores(y_true: pd.Series | np.ndarray, probability: np.ndarray) -> dict[str, Any]:
    prediction = (probability >= 0.5).astype(int)
    return {
        "auc": round(float(roc_auc_score(y_true, probability)), 6),
        "f1": round(float(f1_score(y_true, prediction, zero_division=0)), 6),
        "precision": round(float(precision_score(y_true, prediction, zero_division=0)), 6),
        "recall": round(float(recall_score(y_true, prediction, zero_division=0)), 6),
        "confusion_matrix": confusion_matrix(y_true, prediction, labels=[0, 1]).tolist(),
    }


def _fit_and_score(
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
    threshold: float,
    seed: int,
) -> tuple[dict[str, Any], np.ndarray]:
    y_train = (train["loss_usd"] >= threshold).astype(int)
    y_test = (test["loss_usd"] >= threshold).astype(int)
    if y_train.nunique() < 2 or y_test.nunique() < 2:
        raise DataValidationError("Both train and test periods must contain both target classes")
    estimator = _estimator(features, seed)
    estimator.fit(train[features], y_train)
    probability = estimator.predict_proba(test[features])[:, 1]
    return _scores(y_test, probability), probability


def _temporal_cross_validation(
    dataframe: pd.DataFrame,
    features: list[str],
    threshold_scope: str,
    severity_quantile: float,
    seed: int,
) -> dict[str, Any]:
    splitter = TimeSeriesSplit(n_splits=5)
    auc_values: list[float] = []
    fixed_threshold = float(dataframe["loss_usd"].quantile(severity_quantile))
    for train_index, validation_index in splitter.split(dataframe):
        train = dataframe.iloc[train_index]
        validation = dataframe.iloc[validation_index]
        threshold = (
            float(train["loss_usd"].quantile(severity_quantile))
            if threshold_scope == "training"
            else fixed_threshold
        )
        y_train = (train["loss_usd"] >= threshold).astype(int)
        y_validation = (validation["loss_usd"] >= threshold).astype(int)
        if y_train.nunique() < 2 or y_validation.nunique() < 2:
            continue
        estimator = _estimator(features, seed)
        estimator.fit(train[features], y_train)
        probability = estimator.predict_proba(validation[features])[:, 1]
        auc_values.append(float(roc_auc_score(y_validation, probability)))
    if len(auc_values) < 2:
        raise DataValidationError("Temporal CV produced fewer than two valid folds")
    return {
        "auc_mean": round(float(np.mean(auc_values)), 6),
        "auc_std": round(float(np.std(auc_values)), 6),
        "valid_folds": len(auc_values),
        "fold_auc": [round(value, 6) for value in auc_values],
    }


def run_experiment(
    dataframe: pd.DataFrame,
    profile: ExperimentProfile,
    *,
    seed: int = 42,
    severity_quantile: float = 0.75,
) -> ExperimentOutput:
    if not 0.5 < severity_quantile < 1:
        raise ValueError("severity_quantile must be between 0.5 and 1")
    prepared = add_temporal_features(dataframe)
    eligible, excluded = eligible_point_in_time_features(prepared)
    if profile.require_point_in_time:
        graph_features = eligible
        task_label = "point-in-time incident severity prediction"
        if not graph_features:
            raise DataValidationError(
                "No graph feature has complete point-in-time availability; "
                "refusing predictive claims"
            )
    else:
        graph_features = [feature for feature in GRAPH_FEATURES if prepared[feature].notna().any()]
        task_label = "archival retrospective severity classification"

    train, test = chronological_split(prepared)
    threshold_source = train if profile.threshold_scope == "training" else prepared
    threshold = float(threshold_source["loss_usd"].quantile(severity_quantile))
    baseline_features = TEMPORAL_FEATURES + CATEGORICAL_FEATURES
    semantic_features = baseline_features + graph_features

    baseline_metrics, baseline_probability = _fit_and_score(
        train, test, baseline_features, threshold, seed
    )
    semantic_metrics, semantic_probability = _fit_and_score(
        train, test, semantic_features, threshold, seed
    )
    y_test = (test["loss_usd"] >= threshold).astype(int).to_numpy()
    relative_auc = (
        100 * (semantic_metrics["auc"] - baseline_metrics["auc"]) / baseline_metrics["auc"]
        if baseline_metrics["auc"]
        else 0.0
    )
    metrics = {
        "schema_version": 1,
        "profile": profile.name,
        "profile_description": profile.description,
        "task_label": task_label,
        "dataset": {
            "rows": len(prepared),
            "date_start": prepared["incident_date"].min().date().isoformat(),
            "date_end": prepared["incident_date"].max().date().isoformat(),
            "train_rows": len(train),
            "test_rows": len(test),
            "split_date": test["incident_date"].min().date().isoformat(),
        },
        "target": {
            "severity_quantile": severity_quantile,
            "threshold_usd": round(threshold, 2),
            "threshold_scope": profile.threshold_scope,
            "test_positive_rate": round(float(y_test.mean()), 6),
        },
        "features": {
            "baseline": baseline_features,
            "graph_included": graph_features,
            "graph_excluded": excluded,
        },
        "holdout": {
            "baseline": baseline_metrics,
            "semantic": semantic_metrics,
            "relative_auc_change_pct": round(relative_auc, 3),
        },
        "temporal_cv": {
            "baseline": _temporal_cross_validation(
                prepared, baseline_features, profile.threshold_scope, severity_quantile, seed
            ),
            "semantic": _temporal_cross_validation(
                prepared, semantic_features, profile.threshold_scope, severity_quantile, seed
            ),
        },
        "random_seed": seed,
    }
    return ExperimentOutput(
        metrics=metrics,
        y_test=y_test,
        baseline_probability=baseline_probability,
        semantic_probability=semantic_probability,
    )


def roc_coordinates(y_true: np.ndarray, probability: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    false_positive_rate, true_positive_rate, _ = roc_curve(y_true, probability)
    return false_positive_rate, true_positive_rate
