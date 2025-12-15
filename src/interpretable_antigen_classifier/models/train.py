"""Model training utilities."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, NamedTuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from interpretable_antigen_classifier import config
from interpretable_antigen_classifier.evaluation.metrics import compute_classification_metrics
from interpretable_antigen_classifier.utils.logging import get_logger

logger = get_logger(__name__)


class ModelArtifacts(NamedTuple):
    """Container for train/test splits."""

    X_train: pd.DataFrame
    y_train: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series


def train_baseline_models(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = config.DEFAULT_TEST_SIZE,
    random_state: int = config.DEFAULT_RANDOM_STATE,
    use_xgboost: bool = False,
) -> tuple[Dict[str, Any], Dict[str, Dict[str, float]], ModelArtifacts]:
    """Train baseline classifiers and return fitted models and metrics."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    candidate_models: Dict[str, Pipeline] = {
        "log_reg": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=1000, class_weight="balanced")),
            ]
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300, max_depth=None, n_jobs=-1, class_weight="balanced_subsample", random_state=random_state
        ),
    }

    if use_xgboost:
        try:
            from xgboost import XGBClassifier  # type: ignore

            candidate_models["xgboost"] = XGBClassifier(
                n_estimators=400,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                objective="binary:logistic",
                eval_metric="logloss",
                random_state=random_state,
            )
        except ImportError:
            logger.warning("XGBoost not installed; skipping xgboost model.")

    fitted: Dict[str, Any] = {}
    metrics: Dict[str, Dict[str, float]] = {}

    for name, model in candidate_models.items():
        logger.info("Training model: %s", name)
        model.fit(X_train, y_train)
        y_score = model.predict_proba(X_test)[:, 1]
        metrics[name] = compute_classification_metrics(y_test.values, y_score)
        metrics[name]["n_train"] = int(len(X_train))
        metrics[name]["n_test"] = int(len(X_test))
        fitted[name] = model

    return fitted, metrics, ModelArtifacts(X_train, y_train, X_test, y_test)


def save_model(model: Any, path: Path) -> Path:
    """Persist a trained model using joblib."""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    logger.info("Saved model to %s", path)
    return path
