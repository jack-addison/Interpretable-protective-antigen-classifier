"""Deduplication utilities for labeled sequence datasets."""
from __future__ import annotations

from typing import Dict, Tuple

import pandas as pd

from interpretable_antigen_classifier import config
from interpretable_antigen_classifier.utils.logging import get_logger

logger = get_logger(__name__)


def deduplicate_sequences(df: pd.DataFrame, mode: str = "strict") -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Remove duplicate/ambiguous sequences and ids.

    Parameters
    ----------
    df:
        Input labeled dataframe with at least id, sequence, and label columns.
    mode:
        "strict" drops conflicting ids entirely; "lenient" keeps the first occurrence of a conflicting id.
    """
    if mode not in {"strict", "lenient"}:
        raise ValueError(f"Unsupported deduplication mode: {mode}")

    work = df.copy()
    work["__seq_norm"] = work[config.DEFAULT_TEXT_COLUMN].astype(str).str.upper().str.strip()

    report: Dict[str, int] = {
        "dropped_conflicting_labels": 0,
        "dropped_conflicting_ids": 0,
        "dropped_duplicate_rows": 0,
        "dropped_duplicate_sequences": 0,
    }

    # Drop sequences that appear with multiple labels
    seq_label_counts = work.groupby("__seq_norm")[config.DEFAULT_TARGET_COLUMN].nunique()
    conflict_seqs = seq_label_counts[seq_label_counts > 1].index
    if len(conflict_seqs):
        before = len(work)
        work = work[~work["__seq_norm"].isin(conflict_seqs)]
        report["dropped_conflicting_labels"] = before - len(work)
        logger.info("Dropped %d rows with conflicting labels for identical sequences.", report["dropped_conflicting_labels"])

    # Handle IDs that map to multiple sequences
    id_seq_counts = work.groupby(config.DEFAULT_ID_COLUMN)["__seq_norm"].nunique()
    conflict_ids = id_seq_counts[id_seq_counts > 1].index
    if len(conflict_ids):
        if mode == "strict":
            before = len(work)
            work = work[~work[config.DEFAULT_ID_COLUMN].isin(conflict_ids)]
            report["dropped_conflicting_ids"] = before - len(work)
            logger.info("Strict mode: dropped %d rows from IDs mapping to multiple sequences.", report["dropped_conflicting_ids"])
        else:
            before = len(work)
            work = work.sort_values(config.DEFAULT_ID_COLUMN).drop_duplicates(subset=config.DEFAULT_ID_COLUMN, keep="first")
            report["dropped_conflicting_ids"] = before - len(work)
            logger.info("Lenient mode: kept first occurrence for conflicting IDs, dropped %d rows.", report["dropped_conflicting_ids"])

    # Drop exact duplicate rows (id + sequence)
    before = len(work)
    work = work.drop_duplicates(subset=[config.DEFAULT_ID_COLUMN, "__seq_norm"])
    report["dropped_duplicate_rows"] = before - len(work)

    # Drop duplicate sequences keeping first occurrence to avoid leakage
    before = len(work)
    work = work.drop_duplicates(subset="__seq_norm", keep="first")
    report["dropped_duplicate_sequences"] = before - len(work)

    work = work.drop(columns="__seq_norm")
    return work, report


def rebalance_negatives_length_matched(
    df: pd.DataFrame, negative_multiplier: int = 3, bins: int = 10, random_state: int = 42
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Downsample negatives to a target ratio after deduplication, matching length distribution."""
    report: Dict[str, int] = {"negatives_before": 0, "negatives_after": 0, "positives": 0}
    if negative_multiplier <= 0:
        return df, report

    if config.DEFAULT_TARGET_COLUMN not in df.columns or config.DEFAULT_TEXT_COLUMN not in df.columns:
        logger.warning("Dataframe missing required columns for rebalancing; skipping.")
        return df, report

    pos = df[df[config.DEFAULT_TARGET_COLUMN] == 1].copy()
    neg = df[df[config.DEFAULT_TARGET_COLUMN] == 0].copy()
    report["positives"] = len(pos)
    report["negatives_before"] = len(neg)

    if pos.empty or neg.empty:
        logger.warning("Cannot rebalance: positives or negatives are empty.")
        return df, report

    pos["__len"] = pos[config.DEFAULT_TEXT_COLUMN].str.len()
    neg["__len"] = neg[config.DEFAULT_TEXT_COLUMN].str.len()

    pos["__len_bin"] = pd.qcut(pos["__len"], q=bins, duplicates="drop")
    bins_used = pos["__len_bin"].cat.categories
    edges = [bins_used[0].left] + [cat.right for cat in bins_used]
    neg["__len_bin"] = pd.cut(neg["__len"], bins=edges, include_lowest=True)

    target_neg_total = min(len(neg), negative_multiplier * len(pos))
    sampled_idxs = []

    for bin_cat, pos_group in pos.groupby("__len_bin"):
        target_bin = min(len(neg[neg["__len_bin"] == bin_cat]), negative_multiplier * len(pos_group))
        if target_bin <= 0:
            continue
        chosen = neg[neg["__len_bin"] == bin_cat].sample(n=target_bin, replace=False, random_state=random_state)
        sampled_idxs.extend(chosen.index.tolist())

    remaining_needed = target_neg_total - len(sampled_idxs)
    if remaining_needed > 0:
        remaining_pool = neg[~neg.index.isin(sampled_idxs)]
        if not remaining_pool.empty:
            extra = remaining_pool.sample(n=min(remaining_needed, len(remaining_pool)), random_state=random_state)
            sampled_idxs.extend(extra.index.tolist())

    neg_sampled = neg.loc[sampled_idxs].drop(columns=["__len", "__len_bin"])
    pos = pos.drop(columns=["__len", "__len_bin"])

    combined = pd.concat([pos, neg_sampled], ignore_index=True)
    report["negatives_after"] = len(neg_sampled)
    logger.info(
        "Rebalanced negatives: %d -> %d (multiplier=%d)",
        report["negatives_before"],
        report["negatives_after"],
        negative_multiplier,
    )
    return combined, report
