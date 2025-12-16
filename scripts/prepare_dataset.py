"""Utility script to build the processed labeled dataset from raw FASTA files.

Steps:
1) Parse Protegen positives FASTA -> DataFrame
2) Parse proteome FASTA(s) for negatives -> DataFrame
3) Remove overlap with positives (by protein_id and sequence)
4) Length-matched sampling of negatives to mirror positive length distribution
5) Assign labels (1 positives, 0 negatives) and write processed CSV

Run from repo root:
    python scripts/prepare_dataset.py \\
        --protegen-path data/raw/protegen-all-4.0-2019-01-09.faa \\
        --proteome-paths data/raw/proteomes/staphylococcus_aureus.faa
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Tuple
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from interpretable_antigen_classifier import config
from interpretable_antigen_classifier.utils.logging import get_logger

logger = get_logger(__name__)


def read_fasta(path: Path) -> Iterable[Tuple[str, str]]:
    """Yield (header, sequence) from a FASTA file."""
    header: str | None = None
    seq_parts: List[str] = []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(seq_parts)
                header = line[1:].strip()
                seq_parts = []
            else:
                seq_parts.append(line)
        if header is not None:
            yield header, "".join(seq_parts)


def parse_protegen_header(header: str) -> Tuple[str, str]:
    """Extract protein_id and organism from a Protegen header."""
    parts = header.split("|")
    protein_id = header.split()[0]
    if len(parts) >= 2:
        protein_id = parts[-1].strip().split()[0]
    organism = "unknown"
    if "[" in header and "]" in header:
        organism = header.rsplit("[", 1)[-1].rstrip("]")
    return protein_id, organism


def parse_proteome_header(header: str) -> Tuple[str, str]:
    """Extract protein_id and organism from a proteome (UniProt-style) header."""
    protein_id = header.split()[0]
    if "|" in header:
        segments = header.split("|")
        if len(segments) >= 2 and segments[1]:
            protein_id = segments[1]
    organism = "unknown"
    if "OS=" in header and " OX=" in header:
        try:
            organism = header.split("OS=")[1].split(" OX=")[0].strip()
        except Exception:
            pass
    return protein_id, organism


def load_protegen_df(path: Path) -> pd.DataFrame:
    rows = []
    for header, seq in read_fasta(path):
        protein_id, organism = parse_protegen_header(header)
        rows.append(
            {
                config.DEFAULT_ID_COLUMN: protein_id,
                config.DEFAULT_TEXT_COLUMN: seq,
                config.DEFAULT_ORGANISM_COLUMN: organism,
                "source": "protegen",
                config.DEFAULT_TARGET_COLUMN: 1,
            }
        )
    df = pd.DataFrame(rows)
    logger.info("Loaded %d Protegen positives from %s", len(df), path)
    return df


def load_proteome_df(paths: Iterable[Path]) -> pd.DataFrame:
    rows = []
    paths = list(paths)
    for path in paths:
        for header, seq in read_fasta(path):
            protein_id, organism = parse_proteome_header(header)
            rows.append(
                {
                    config.DEFAULT_ID_COLUMN: protein_id,
                    config.DEFAULT_TEXT_COLUMN: seq,
                    config.DEFAULT_ORGANISM_COLUMN: organism,
                    "source": path.stem,
                    config.DEFAULT_TARGET_COLUMN: 0,
                }
            )
    df = pd.DataFrame(rows)
    logger.info("Loaded %d candidate negatives from %d proteome file(s)", len(df), len(paths))
    return df


def remove_overlaps(pos_df: pd.DataFrame, neg_df: pd.DataFrame) -> pd.DataFrame:
    """Remove negatives overlapping positives by id or sequence."""
    pos_ids = set(pos_df[config.DEFAULT_ID_COLUMN])
    pos_seqs = set(pos_df[config.DEFAULT_TEXT_COLUMN])
    mask = (~neg_df[config.DEFAULT_ID_COLUMN].isin(pos_ids)) & (~neg_df[config.DEFAULT_TEXT_COLUMN].isin(pos_seqs))
    removed = len(neg_df) - mask.sum()
    if removed:
        logger.info("Removed %d overlapping negatives", removed)
    return neg_df[mask]


def sample_length_matched(
    pos_df: pd.DataFrame, neg_df: pd.DataFrame, bins: int = 10, random_state: int = 42
) -> pd.DataFrame:
    """Length-match negatives to positives using quantile bins with fallback to nearest lengths."""
    pos_df = pos_df.copy()
    neg_df = neg_df.copy()
    pos_df["seq_len"] = pos_df[config.DEFAULT_TEXT_COLUMN].str.len()
    neg_df["seq_len"] = neg_df[config.DEFAULT_TEXT_COLUMN].str.len()

    if pos_df.empty or neg_df.empty:
        logger.warning("Empty positives or negatives; returning empty sample.")
        return neg_df.iloc[0:0].copy()

    pos_df["len_bin"] = pd.qcut(pos_df["seq_len"], q=bins, duplicates="drop")
    categories = pos_df["len_bin"].cat.categories
    edges = [categories[0].left] + [cat.right for cat in categories]
    neg_df["len_bin"] = pd.cut(neg_df["seq_len"], bins=edges, include_lowest=True)

    used_indices: set = set()
    samples: list[pd.DataFrame] = []

    for bin_cat, pos_group in pos_df.groupby("len_bin", observed=False):
        target = len(pos_group)
        candidates = neg_df[(neg_df["len_bin"] == bin_cat) & (~neg_df.index.isin(used_indices))]
        take = min(target, len(candidates))
        if take > 0:
            chosen = candidates.sample(take, random_state=random_state)
            used_indices.update(chosen.index)
            samples.append(chosen)

        remaining = target - take
        if remaining > 0:
            available = neg_df[~neg_df.index.isin(used_indices)].copy()
            if available.empty:
                logger.warning("Ran out of negatives while matching lengths.")
                break
            midpoint = (bin_cat.left + bin_cat.right) / 2 if hasattr(bin_cat, "left") else pos_group["seq_len"].median()
            available["len_diff"] = (available["seq_len"] - midpoint).abs()
            extra = available.nsmallest(remaining, "len_diff")
            used_indices.update(extra.index)
            samples.append(extra)

    sampled = pd.concat(samples) if samples else neg_df.iloc[0:0].copy()
    sampled = sampled.drop(columns=["seq_len", "len_bin"], errors="ignore")
    logger.info("Selected %d length-matched negatives", len(sampled))
    return sampled


def build_dataset(protegen_path: Path, proteome_paths: list[Path], output_path: Path) -> Path:
    config.ensure_directories()
    pos_df = load_protegen_df(protegen_path)
    neg_df = load_proteome_df(proteome_paths)
    neg_df = remove_overlaps(pos_df, neg_df)
    neg_sampled = sample_length_matched(pos_df, neg_df)

    combined = pd.concat([pos_df, neg_sampled], ignore_index=True)
    combined = combined.drop_duplicates(subset=[config.DEFAULT_ID_COLUMN, config.DEFAULT_TEXT_COLUMN])
    combined = combined[[config.DEFAULT_ID_COLUMN, config.DEFAULT_TEXT_COLUMN, config.DEFAULT_ORGANISM_COLUMN, "source", config.DEFAULT_TARGET_COLUMN]]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_path, index=False)
    logger.info("Wrote %d labeled sequences to %s", len(combined), output_path)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare labeled sequences CSV from Protegen and proteome FASTAs.")
    parser.add_argument("--protegen-path", type=Path, default=Path("data/raw/protegen-all-4.0-2019-01-09.faa"), help="Path to Protegen FASTA")
    parser.add_argument(
        "--proteome-paths",
        type=Path,
        nargs="+",
        default=[Path("data/raw/proteomes/staphylococcus_aureus.faa")],
        help="Paths to proteome FASTA files for negatives",
    )
    parser.add_argument("--output-path", type=Path, default=config.PROCESSED_DATA_PATH, help="Output CSV path")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    build_dataset(args.protegen_path, args.proteome_paths, args.output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
