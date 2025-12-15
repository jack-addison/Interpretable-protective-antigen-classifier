"""Feature extraction utilities for protein sequences."""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from interpretable_antigen_classifier import config
from interpretable_antigen_classifier.utils.logging import get_logger

logger = get_logger(__name__)

AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")
HYDROPHOBIC = set("AILMFWYV")
POSITIVE = set("KRH")
NEGATIVE = set("DE")


def build_feature_table(
    data: pd.DataFrame, skip_psortb: bool = False
) -> tuple[pd.DataFrame, pd.Series]:
    """Compute feature matrix and labels.

    Parameters
    ----------
    data:
        DataFrame containing sequences and labels.
    skip_psortb:
        If True, do not attempt PSORTb features even if available.
    """
    basic = compute_basic_features(data)
    psortb_df = None
    if not skip_psortb and psortb_available():
        try:
            psortb_df = run_psortb_stub()
        except Exception as exc:  # pragma: no cover - placeholder for real call
            logger.warning("PSORTb feature stage failed: %s", exc)
    elif not skip_psortb:
        logger.info("PSORTb binary not detected; skipping localisation features.")

    features = basic
    if psortb_df is not None:
        features = pd.concat([basic, psortb_df], axis=1)

    labels = data[config.DEFAULT_TARGET_COLUMN]
    return features, labels


def compute_basic_features(data: pd.DataFrame) -> pd.DataFrame:
    """Compute simple sequence-derived features.

    Features include sequence length, amino-acid composition, and coarse
    physicochemical proxies. Expand this with k-mer counts and other descriptors
    as needed.
    """
    feature_rows = []
    for row in data.itertuples(index=False):
        seq = getattr(row, config.DEFAULT_TEXT_COLUMN, None)
        if not isinstance(seq, str) or not seq:
            logger.warning("Encountered empty sequence for record %s", getattr(row, config.DEFAULT_ID_COLUMN, "unknown"))
            seq = ""

        seq = seq.upper()
        seq_len = len(seq)
        aa_features: Dict[str, float] = {f"aa_{aa}_freq": (seq.count(aa) / seq_len) if seq_len else 0.0 for aa in AMINO_ACIDS}
        hydrophobic_frac = sum(seq.count(aa) for aa in HYDROPHOBIC) / seq_len if seq_len else 0.0
        positive_charge_frac = sum(seq.count(aa) for aa in POSITIVE) / seq_len if seq_len else 0.0
        negative_charge_frac = sum(seq.count(aa) for aa in NEGATIVE) / seq_len if seq_len else 0.0

        feature_rows.append(
            {
                config.DEFAULT_ID_COLUMN: getattr(row, config.DEFAULT_ID_COLUMN, "unknown"),
                "seq_length": seq_len,
                "hydrophobic_fraction": hydrophobic_frac,
                "positive_charge_fraction": positive_charge_frac,
                "negative_charge_fraction": negative_charge_frac,
                **aa_features,
            }
        )

    feature_df = pd.DataFrame(feature_rows).set_index(config.DEFAULT_ID_COLUMN)
    logger.info("Computed basic features for %d sequences", len(feature_df))
    return feature_df


def psortb_available() -> bool:
    """Return True if PSORTb binary is on PATH."""
    return shutil.which(config.PSORTB_BINARY) is not None


def run_psortb_stub() -> Optional[pd.DataFrame]:
    """Placeholder for PSORTb integration.

    Replace this with a call that writes sequences to FASTA, executes PSORTb,
    and parses the output. The function should return a DataFrame indexed by
    protein_id containing localisation probability features.
    """
    logger.warning("PSORTb integration not yet implemented; skipping localisation features.")
    return None
