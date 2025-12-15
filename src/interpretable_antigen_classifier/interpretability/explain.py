"""Interpretability helpers: feature importances and SHAP (optional)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance

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
    sample = X.sample(min(len(X), max_samples), random_state=42)
    try:
        explainer = shap.Explainer(estimator, sample)
        shap_values = explainer(sample)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        shap.plots.beeswarm(shap_values, show=False)
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
    result = permutation_importance(model, X, y, n_repeats=n_repeats, random_state=random_state, n_jobs=-1)
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


def _unwrap_estimator(model: Any) -> Any:
    """Return the final estimator if a Pipeline was provided."""
    if hasattr(model, "steps"):
        return model.steps[-1][1]
    return model
