"""Dataset validation and summary helpers."""
from __future__ import annotations

from typing import Dict

import pandas as pd

from interpretable_antigen_classifier import config
from interpretable_antigen_classifier.utils.logging import get_logger

logger = get_logger(__name__)


def summarize_dataset(df: pd.DataFrame) -> Dict[str, object]:
    """Return basic dataset summary stats for logging."""
    summary: Dict[str, object] = {}
    summary["n_rows"] = len(df)
    summary["class_counts"] = df[config.DEFAULT_TARGET_COLUMN].value_counts(dropna=False).to_dict()
    lengths = df[config.DEFAULT_TEXT_COLUMN].astype(str).str.len()
    summary["length_stats"] = lengths.describe().to_dict()
    summary["empty_sequences"] = int((lengths == 0).sum())
    summary["duplicate_ids"] = int(df[config.DEFAULT_ID_COLUMN].duplicated().sum())
    summary["duplicate_sequences"] = int(df[config.DEFAULT_TEXT_COLUMN].duplicated().sum())
    if config.DEFAULT_ORGANISM_COLUMN in df.columns:
        summary["top_organisms"] = df[config.DEFAULT_ORGANISM_COLUMN].value_counts().head(10).to_dict()
    return summary


def log_dataset_summary(df: pd.DataFrame) -> None:
    """Log dataset summary in a readable form."""
    summary = summarize_dataset(df)
    logger.info(
        "Dataset: %d rows | class counts: %s | length mean=%.1f std=%.1f min=%d max=%d | empty seqs=%d | dup ids=%d | dup seqs=%d",
        summary["n_rows"],
        summary["class_counts"],
        summary["length_stats"]["mean"],
        summary["length_stats"]["std"],
        summary["length_stats"]["min"],
        summary["length_stats"]["max"],
        summary["empty_sequences"],
        summary["duplicate_ids"],
        summary["duplicate_sequences"],
    )
    if "top_organisms" in summary:
        logger.info("Top organisms (first 10): %s", summary["top_organisms"])


def filter_organisms_with_both_labels(
    df: pd.DataFrame, min_per_label: int = 1
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Keep only organisms that have at least min_per_label samples in each class."""
    if config.DEFAULT_ORGANISM_COLUMN not in df.columns:
        return df, {"kept": len(df), "dropped": 0, "organisms_kept": 0, "organisms_dropped": 0}

    counts = df.groupby([config.DEFAULT_ORGANISM_COLUMN, config.DEFAULT_TARGET_COLUMN]).size().unstack(fill_value=0)
    valid_orgs = counts[(counts >= min_per_label).all(axis=1)].index
    kept = df[df[config.DEFAULT_ORGANISM_COLUMN].isin(valid_orgs)]
    report = {
        "kept": len(kept),
        "dropped": len(df) - len(kept),
        "organisms_kept": len(valid_orgs),
        "organisms_dropped": counts.shape[0] - len(valid_orgs),
    }
    return kept, report
