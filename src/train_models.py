"""Model training utilities."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

from .paths import FEATURE_STATS_PATH, METRICS_PATH, MODEL_METADATA_PATH, MODEL_PATH

LOGGER = logging.getLogger(__name__)


@dataclass
class ModelResult:
    name: str
    pipeline: Pipeline
    metrics: Dict[str, float]


NUMERIC_FEATURES = [
    "surface_total_m2",
    "surface_covered_m2",
    "rooms",
    "bedrooms",
    "bathrooms",
    "latitude",
    "longitude",
    "covered_ratio",
    "rooms_per_surface",
    "property_age",
]

CATEGORICAL_FEATURES = ["property_type", "state_name", "neighborhood"]
TARGET_COLUMN = "target_price"


def _build_preprocessor() -> ColumnTransformer:
    numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])
    categorical_transformer = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )


def _evaluate_model(pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
    predictions = pipeline.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    rmse = mean_squared_error(y_test, predictions, squared=False)
    r2 = r2_score(y_test, predictions)
    return {"mae": float(mae), "rmse": float(rmse), "r2": float(r2)}


def _train_single_model(name: str, estimator, X_train, y_train) -> Pipeline:
    preprocessor = _build_preprocessor()
    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("estimator", estimator)])
    pipeline.fit(X_train, y_train)
    return pipeline


def _prepare_dataset(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    df = df.copy()
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=[TARGET_COLUMN, "surface_total_m2"])
    for column in NUMERIC_FEATURES:
        if column not in df:
            df[column] = 0.0
    for column in CATEGORICAL_FEATURES:
        if column not in df:
            df[column] = "Desconocido"
    df[NUMERIC_FEATURES] = df[NUMERIC_FEATURES].fillna(0)
    df[CATEGORICAL_FEATURES] = df[CATEGORICAL_FEATURES].fillna("Desconocido")
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET_COLUMN]
    return X, y


def _collect_feature_stats(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    stats = {}
    for column in NUMERIC_FEATURES:
        if column in df:
            stats[column] = {
                "min": float(df[column].min()),
                "max": float(df[column].max()),
                "median": float(df[column].median()),
            }
    for column in CATEGORICAL_FEATURES:
        if column in df:
            top_values = (
                df[column].value_counts().head(10).index.astype(str).tolist()
            )
            stats[column] = {"top_values": top_values}
    return stats


def train_models(parquet_path: str) -> ModelResult:
    LOGGER.info("Loading dataset from %s", parquet_path)
    df = pd.read_parquet(parquet_path)

    X, y = _prepare_dataset(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models = {
        "linear_regression": LinearRegression(),
        "decision_tree": DecisionTreeRegressor(random_state=42, max_depth=12),
        "random_forest": RandomForestRegressor(random_state=42, n_estimators=200, max_depth=16),
        "gradient_boosting": GradientBoostingRegressor(random_state=42),
    }

    results: List[ModelResult] = []

    for name, estimator in models.items():
        LOGGER.info("Training %s", name)
        pipeline = _train_single_model(name, estimator, X_train, y_train)
        metrics = _evaluate_model(pipeline, X_test, y_test)
        results.append(ModelResult(name=name, pipeline=pipeline, metrics=metrics))
        LOGGER.info("%s metrics: %s", name, metrics)

    best_model = min(results, key=lambda item: item.metrics["rmse"])

    LOGGER.info("Saving best model (%s) to %s", best_model.name, MODEL_PATH)
    joblib.dump(best_model.pipeline, MODEL_PATH)

    metrics_payload = {
        result.name: result.metrics for result in results
    }
    metrics_payload["best_model"] = best_model.name
    metrics_payload["best_model_metrics"] = best_model.metrics

    METRICS_PATH.write_text(json.dumps(metrics_payload, indent=2))
    feature_stats = _collect_feature_stats(pd.concat([X, y], axis=1))
    FEATURE_STATS_PATH.write_text(json.dumps(feature_stats, indent=2))

    metadata = {
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "target": TARGET_COLUMN,
        "band_percentage": 0.10,
    }
    MODEL_METADATA_PATH.write_text(json.dumps(metadata, indent=2))

    return best_model

