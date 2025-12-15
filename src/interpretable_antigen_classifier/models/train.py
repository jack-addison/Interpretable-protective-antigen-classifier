"""Model training utilities."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, NamedTuple, Optional

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    GroupKFold,
    GroupShuffleSplit,
    StratifiedKFold,
    cross_val_predict,
    train_test_split,
)
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
    groups_train: Optional[pd.Series]
    groups_test: Optional[pd.Series]


def train_baseline_models(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = config.DEFAULT_TEST_SIZE,
    random_state: int = config.DEFAULT_RANDOM_STATE,
    use_xgboost: bool = True,
    split_strategy: str = "stratified",
    groups: Optional[pd.Series] = None,
    cv_folds: int = 0,
) -> tuple[Dict[str, Any], Dict[str, Dict[str, Any]], ModelArtifacts]:
    """Train baseline classifiers and return fitted models and metrics."""
    split = _train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        strategy=split_strategy,
        groups=groups,
    )
    X_train, X_test, y_train, y_test, groups_train, groups_test = split

    if y_train.nunique() < 2 or y_test.nunique() < 2:
        logger.warning("Train/test split resulted in a single class; falling back to stratified split.")
        X_train, X_test, y_train, y_test, groups_train, groups_test = _train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            strategy="stratified",
            groups=None,
        )
        split_strategy = "stratified"
        groups = None

    candidate_models: Dict[str, Any] = {
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
    metrics: Dict[str, Dict[str, Any]] = {}

    for name, model in candidate_models.items():
        logger.info("Training model: %s", name)
        model.fit(X_train, y_train)
        y_score = model.predict_proba(X_test)[:, 1]
        metrics[name] = compute_classification_metrics(y_test.to_numpy(), y_score)
        metrics[name]["n_train"] = int(len(X_train))
        metrics[name]["n_test"] = int(len(X_test))
        metrics[name]["split_strategy"] = split_strategy

        if cv_folds and cv_folds > 1 and y.nunique() >= 2:
            cv_groups = groups if split_strategy == "group" else None
            cv_obj = _make_cv(cv_folds, random_state, split_strategy, groups)
            try:
                cv_scores = cross_val_predict(
                    model, X, y, cv=cv_obj, method="predict_proba", groups=cv_groups, n_jobs=-1
                )[:, 1]
                cv_metrics = compute_classification_metrics(y.to_numpy(), cv_scores)
                metrics[name]["cv_roc_auc"] = cv_metrics["roc_auc"]
                metrics[name]["cv_pr_auc"] = cv_metrics["pr_auc"]
                metrics[name]["cv_folds"] = cv_folds
            except Exception as exc:
                logger.warning("Cross-validation failed for %s: %s", name, exc)
        elif cv_folds and cv_folds > 1:
            logger.warning("Skipping cross-validation because the dataset has <2 classes.")

        fitted[name] = model

    return fitted, metrics, ModelArtifacts(X_train, y_train, X_test, y_test, groups_train, groups_test)


def save_model(model: Any, path: Path) -> Path:
    """Persist a trained model using joblib."""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    logger.info("Saved model to %s", path)
    return path


def _train_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float,
    random_state: int,
    strategy: str = "stratified",
    groups: Optional[pd.Series] = None,
):
    """Perform train/test split with optional group awareness."""
    if strategy == "group" and groups is not None:
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        train_idx, test_idx = next(splitter.split(X, y, groups))
        return (
            X.iloc[train_idx],
            X.iloc[test_idx],
            y.iloc[train_idx],
            y.iloc[test_idx],
            groups.iloc[train_idx],
            groups.iloc[test_idx],
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    return X_train, X_test, y_train, y_test, None, None


def _make_cv(cv_folds: int, random_state: int, strategy: str, groups: Optional[pd.Series]):
    """Return a CV splitter respecting strategy."""
    if strategy == "group" and groups is not None:
        return GroupKFold(n_splits=cv_folds)
    return StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
