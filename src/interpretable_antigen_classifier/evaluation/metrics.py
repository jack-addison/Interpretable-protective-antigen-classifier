"""Evaluation utilities for classification models."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from interpretable_antigen_classifier.utils.logging import get_logger

logger = get_logger(__name__)


def compute_classification_metrics(y_true: np.ndarray, y_score: np.ndarray) -> Dict[str, float]:
    """Compute ROC-AUC and PR-AUC, handling edge cases gracefully."""
    metrics: Dict[str, float] = {}
    try:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_score))
    except ValueError as exc:
        logger.warning("Could not compute ROC-AUC: %s", exc)
        metrics["roc_auc"] = float("nan")

    try:
        metrics["pr_auc"] = float(average_precision_score(y_true, y_score))
    except ValueError as exc:
        logger.warning("Could not compute PR-AUC: %s", exc)
        metrics["pr_auc"] = float("nan")

    return metrics


def save_metrics(metrics: Dict[str, Any], path: Path) -> None:
    """Persist metrics as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        json.dump(metrics, fp, indent=2)
    logger.info("Saved metrics to %s", path)
