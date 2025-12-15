"""Feature extraction utilities for protein sequences."""
from __future__ import annotations

import shutil
from collections import Counter
from typing import Dict, Iterable, Optional, Sequence

import pandas as pd

from interpretable_antigen_classifier import config
from interpretable_antigen_classifier.utils.logging import get_logger

logger = get_logger(__name__)

AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")
HYDROPHOBIC = set("AILMFWYV")
POSITIVE = set("KRH")
NEGATIVE = set("DE")
AROMATIC = set("FYW")
ALIPHATIC = set("VILA")

# Approximate monoisotopic masses (Daltons) for 20 amino acids
AMINO_MASS = {
    "A": 71.03711,
    "R": 156.10111,
    "N": 114.04293,
    "D": 115.02694,
    "C": 103.00919,
    "E": 129.04259,
    "Q": 128.05858,
    "G": 57.02146,
    "H": 137.05891,
    "I": 113.08406,
    "L": 113.08406,
    "K": 128.09496,
    "M": 131.04049,
    "F": 147.06841,
    "P": 97.05276,
    "S": 87.03203,
    "T": 101.04768,
    "W": 186.07931,
    "Y": 163.06333,
    "V": 99.06841,
}


def build_feature_table(
    data: pd.DataFrame,
    skip_psortb: bool = False,
    use_kmers: bool = True,
    kmer_sizes: Sequence[int] | None = (2, 3),
    kmer_top_n: int = 256,
) -> tuple[pd.DataFrame, pd.Series]:
    """Compute feature matrix and labels."""
    basic = compute_basic_features(data)
    psortb_df = None
    if not skip_psortb and psortb_available():
        try:
            psortb_df = run_psortb_stub()
        except Exception as exc:  # pragma: no cover - placeholder for real call
            logger.warning("PSORTb feature stage failed: %s", exc)
    elif not skip_psortb:
        logger.info("PSORTb binary not detected; skipping localisation features.")

    kmers_df = None
    if use_kmers and kmer_sizes:
        kmers_df = compute_kmer_features(data, kmer_sizes=kmer_sizes, top_n=kmer_top_n)

    features = basic
    for extra in [kmers_df, psortb_df]:
        if extra is not None and not extra.empty:
            features = pd.concat([features, extra], axis=1)

    labels = data[config.DEFAULT_TARGET_COLUMN]
    logger.info("Assembled feature matrix with shape %s", features.shape)
    return features, labels


def compute_basic_features(data: pd.DataFrame) -> pd.DataFrame:
    """Compute simple sequence-derived features."""
    feature_rows = []
    for row in data.itertuples(index=False):
        seq = getattr(row, config.DEFAULT_TEXT_COLUMN, None)
        if not isinstance(seq, str) or not seq:
            logger.warning("Encountered empty sequence for record %s", getattr(row, config.DEFAULT_ID_COLUMN, "unknown"))
            seq = ""

        seq = seq.upper()
        seq_len = len(seq)
        aa_features: Dict[str, float] = {
            f"aa_{aa}_freq": (seq.count(aa) / seq_len) if seq_len else 0.0 for aa in AMINO_ACIDS
        }
        hydrophobic_frac = sum(seq.count(aa) for aa in HYDROPHOBIC) / seq_len if seq_len else 0.0
        positive_charge_frac = sum(seq.count(aa) for aa in POSITIVE) / seq_len if seq_len else 0.0
        negative_charge_frac = sum(seq.count(aa) for aa in NEGATIVE) / seq_len if seq_len else 0.0
        aromatic_frac = sum(seq.count(aa) for aa in AROMATIC) / seq_len if seq_len else 0.0
        aliphatic_frac = sum(seq.count(aa) for aa in ALIPHATIC) / seq_len if seq_len else 0.0
        net_charge = estimate_net_charge(seq, seq_len)
        mol_weight = estimate_molecular_weight(seq)

        feature_rows.append(
            {
                config.DEFAULT_ID_COLUMN: getattr(row, config.DEFAULT_ID_COLUMN, "unknown"),
                "seq_length": seq_len,
                "hydrophobic_fraction": hydrophobic_frac,
                "positive_charge_fraction": positive_charge_frac,
                "negative_charge_fraction": negative_charge_frac,
                "aromatic_fraction": aromatic_frac,
                "aliphatic_fraction": aliphatic_frac,
                "net_charge_estimate": net_charge,
                "molecular_weight": mol_weight,
                **aa_features,
            }
        )

    feature_df = pd.DataFrame(feature_rows).set_index(config.DEFAULT_ID_COLUMN)
    logger.info("Computed basic features for %d sequences", len(feature_df))
    return feature_df


def compute_kmer_features(
    data: pd.DataFrame,
    kmer_sizes: Sequence[int] = (3,),
    top_n: int = 256,
) -> pd.DataFrame:
    """Compute k-mer frequency features for provided k values."""
    kmer_sizes = tuple(sorted(set(k for k in kmer_sizes if k > 0)))
    if not kmer_sizes:
        return pd.DataFrame()

    sequences = data[[config.DEFAULT_ID_COLUMN, config.DEFAULT_TEXT_COLUMN]].copy()
    sequences[config.DEFAULT_TEXT_COLUMN] = sequences[config.DEFAULT_TEXT_COLUMN].astype(str).str.upper()

    top_kmers: dict[int, list[str]] = {}
    for k in kmer_sizes:
        counter: Counter[str] = Counter()
        for seq in sequences[config.DEFAULT_TEXT_COLUMN]:
            if len(seq) < k:
                continue
            counter.update(iter_kmers(seq, k))
        top_kmers[k] = [kmer for kmer, _ in counter.most_common(top_n)]
        if not top_kmers[k]:
            logger.warning("No k-mers of size %d found; skipping", k)

    feature_rows = []
    for row in sequences.itertuples(index=False):
        seq = getattr(row, config.DEFAULT_TEXT_COLUMN, "") or ""
        seq_len = len(seq)
        row_feats: Dict[str, float] = {}
        for k, vocab in top_kmers.items():
            if not vocab:
                continue
            denom = max(seq_len - k + 1, 1)
            counts = Counter(iter_kmers(seq, k)) if seq_len >= k else Counter()
            for kmer in vocab:
                row_feats[f"kmer{k}_{kmer}"] = counts.get(kmer, 0) / denom
        feature_rows.append({config.DEFAULT_ID_COLUMN: getattr(row, config.DEFAULT_ID_COLUMN), **row_feats})

    if not feature_rows:
        logger.warning("No k-mer features were generated.")
        return pd.DataFrame()

    feature_df = pd.DataFrame(feature_rows).set_index(config.DEFAULT_ID_COLUMN)
    logger.info("Computed k-mer features with shape %s", feature_df.shape)
    return feature_df


def iter_kmers(seq: str, k: int) -> Iterable[str]:
    """Yield valid k-mers from a sequence, skipping non-standard residues."""
    upper = seq.upper()
    for i in range(len(upper) - k + 1):
        kmer = upper[i : i + k]
        if any(c not in AMINO_ACIDS for c in kmer):
            continue
        yield kmer


def estimate_net_charge(seq: str, seq_len: int) -> float:
    """Rough net charge estimate at neutral pH."""
    if not seq_len:
        return 0.0
    seq = seq.upper()
    positive = seq.count("K") + seq.count("R") + 0.1 * seq.count("H")
    negative = seq.count("D") + seq.count("E") + 0.05 * seq.count("C") + 0.05 * seq.count("Y")
    return (positive - negative) / seq_len


def estimate_molecular_weight(seq: str) -> float:
    """Approximate molecular weight (Daltons) ignoring post-translational changes."""
    seq = seq.upper()
    return float(sum(AMINO_MASS.get(aa, 0.0) for aa in seq))


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
