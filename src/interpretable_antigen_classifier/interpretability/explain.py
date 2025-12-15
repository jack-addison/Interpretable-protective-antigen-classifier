"""Interpretability helpers: feature importances and SHAP (optional)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.inspection import permutation_importance
from sklearn.utils import Bunch

from interpretable_antigen_classifier.utils.logging import get_logger

logger = get_logger(__name__)


def extract_feature_importances(model: Any, feature_names: list[str]) -> pd.DataFrame:
    """Return feature importances if available from the estimator."""
    estimator = _unwrap_estimator(model)
    importances: Optional[np.ndarray] = None

    if hasattr(estimator, "feature_importances_"):
        importances = getattr(estimator, "feature_importances_")
    elif hasattr(estimator, "coef_"):
        coef = getattr(estimator, "coef_")
        if coef.ndim > 1:
            coef = coef[0]
        importances = np.abs(coef)

    if importances is None:
        logger.warning("Model %s does not expose feature importances.", estimator.__class__.__name__)
        return pd.DataFrame(columns=["feature", "importance"])

    df = pd.DataFrame({"feature": feature_names, "importance": importances})
    df = df.sort_values(by="importance", ascending=False).reset_index(drop=True)
    return df


def compute_shap_summary(
    model: Any,
    X: pd.DataFrame,
    output_path: Path,
    max_samples: int = 500,
    top_n: int = 25,
) -> bool:
    """Compute and save a SHAP summary plot if shap is installed.

    Returns True on success, False otherwise.
    """
    try:
        import shap  # type: ignore
    except ImportError:
        logger.warning("shap is not installed; skipping SHAP summary plot.")
        return False

    estimator = _unwrap_estimator(model)
    X_numeric = _to_numeric_frame(X)
    sample_df = X_numeric.sample(min(len(X_numeric), max_samples), random_state=42)
    sample_array = sample_df.to_numpy()
    try:
        explainer = _make_shap_explainer(estimator, sample_array)
        shap_values = explainer(sample_array)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        shap.summary_plot(shap_values, feature_names=sample_df.columns, max_display=top_n, show=False)
        plt.tight_layout()
        plt.savefig(output_path, bbox_inches="tight")
        plt.close()
        logger.info("Saved SHAP summary plot to %s", output_path)
        return True
    except Exception as exc:  # pragma: no cover - visualization heavy
        logger.warning("Failed to compute SHAP values: %s", exc)
        return False


def permutation_importance_report(
    model: Any, X: pd.DataFrame, y: pd.Series, n_repeats: int = 10, random_state: int = 42
) -> pd.DataFrame:
    """Compute permutation importance as a SHAP fallback."""
    result = cast(
        Bunch,
        permutation_importance(model, X, y, n_repeats=n_repeats, random_state=random_state, n_jobs=-1),
    )
    df = pd.DataFrame(
        {
            "feature": X.columns,
            "mean_importance": result.importances_mean,
            "std_importance": result.importances_std,
        }
    ).sort_values(by="mean_importance", ascending=False)
    return df


def save_feature_importances(importances: pd.DataFrame, path: Path) -> None:
    """Save feature importances to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    importances.to_csv(path, index=False)
    logger.info("Saved feature importances to %s", path)


def plot_feature_importances(importances: pd.DataFrame, path: Path, top_n: int = 20) -> None:
    """Save a bar plot of top feature importances."""
    if importances.empty:
        logger.info("No feature importances to plot.")
        return
    subset = importances.head(top_n)
    plt.figure(figsize=(8, max(4, top_n * 0.3)))
    sns.barplot(data=subset, x="importance", y="feature", color="steelblue", legend=False)
    plt.title("Top feature importances")
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    logger.info("Saved feature importance plot to %s", path)


def plot_permutation_importance(perm_df: pd.DataFrame, path: Path, top_n: int = 20) -> None:
    """Save a bar plot of permutation importances."""
    if perm_df.empty:
        logger.info("No permutation importances to plot.")
        return
    subset = perm_df.head(top_n)
    plt.figure(figsize=(8, max(4, top_n * 0.3)))
    sns.barplot(data=subset, x="mean_importance", y="feature", color="indianred", legend=False)
    plt.title("Top permutation importances")
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    logger.info("Saved permutation importance plot to %s", path)


def _unwrap_estimator(model: Any) -> Any:
    """Return the final estimator if a Pipeline was provided."""
    if hasattr(model, "steps"):
        return model.steps[-1][1]
    return model


def _make_shap_explainer(estimator: Any, background: np.ndarray):
    """Choose an appropriate SHAP explainer based on estimator type."""
    try:
        import shap  # type: ignore
    except ImportError as exc:  # pragma: no cover - guarded earlier
        raise exc

    if _is_tree_model(estimator):
        return shap.TreeExplainer(estimator, background, feature_perturbation="interventional")
    if _is_linear_model(estimator):
        return shap.LinearExplainer(estimator, background)
    return shap.Explainer(estimator, background)


def _is_tree_model(estimator: Any) -> bool:
    """Heuristic to detect tree/ensemble models."""
    tree_types = ("XGBClassifier", "XGBRegressor", "RandomForest", "GradientBoosting", "HistGradientBoosting")
    name = estimator.__class__.__name__
    return any(t in name for t in tree_types)


def _is_linear_model(estimator: Any) -> bool:
    """Heuristic to detect linear models with coef_."""
    return hasattr(estimator, "coef_")


def _to_numeric_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure dataframe is numeric for SHAP computation."""
    numeric = df.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    numeric = numeric.astype(float)
    return numeric
