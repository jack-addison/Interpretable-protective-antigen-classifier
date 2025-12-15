"""Data ingestion and curation utilities.

These functions are intentionally lightweight and include placeholders/TODOs for
manual downloads. The main pipeline expects a processed CSV at
``data/processed/labeled_sequences.csv`` with columns:
`protein_id`, `label` (1 protective, 0 non-protective), `organism`, `source`, `sequence`.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import pandas as pd

try:
    from Bio import SeqIO

    HAS_BIO = True
except ImportError:  # pragma: no cover - optional dependency
    HAS_BIO = False

from interpretable_antigen_classifier import config
from interpretable_antigen_classifier.utils.logging import get_logger

logger = get_logger(__name__)

EXPECTED_COLUMNS: list[str] = [
    config.DEFAULT_ID_COLUMN,
    config.DEFAULT_TEXT_COLUMN,
    config.DEFAULT_ORGANISM_COLUMN,
    "source",
    config.DEFAULT_TARGET_COLUMN,
]


def download_protegen_dataset(output_path: Path | None = None) -> Optional[Path]:
    """Stub for downloading protective antigen sequences from Protegen.

    Parameters
    ----------
    output_path:
        Where to store the downloaded Protegen FASTA/CSV export.

    Notes
    -----
    - Protegen may require manual export. Place the file at the desired path and
      rerun the pipeline.
    - Update this function with authenticated requests or scraping logic if
      permissible by the data provider.
    """
    output = output_path or (config.RAW_DATA_DIR / "protegen.fasta")
    if output.exists():
        logger.info("Found existing Protegen dataset at %s", output)
        return output

    logger.warning(
        "Protegen dataset not found at %s. TODO: implement download or place file manually.",
        output,
    )
    return None


def sample_negative_proteins(
    proteome_paths: Sequence[Path],
    target_count: int = 1000,
    output_path: Path | None = None,
) -> Optional[Path]:
    """Stub for creating a matched negative protein set from bacterial proteomes.

    Parameters
    ----------
    proteome_paths:
        Paths to proteome FASTA files to sample negatives from.
    target_count:
        Approximate number of negatives to sample.
    output_path:
        Where to store the sampled negatives FASTA.

    Notes
    -----
    - Implement organism-aware sampling to avoid leakage.
    - Consider matching sequence length distribution to positives.
    - If PSORTb is available, you may pre-compute localisation predictions here.
    """
    output = output_path or (config.RAW_DATA_DIR / "negatives.fasta")
    if output.exists():
        logger.info("Found existing negative set at %s", output)
        return output

    if not proteome_paths:
        logger.warning("No proteome paths provided; cannot sample negatives.")
        return None

    logger.warning(
        "Negative set not generated. TODO: sample ~%d proteins from: %s -> %s",
        target_count,
        ", ".join(str(p) for p in proteome_paths),
        output,
    )
    return None


def parse_fasta_to_frame(fasta_path: Path, label: int, source: str) -> pd.DataFrame:
    """Convert a FASTA file into a labeled DataFrame.

    This helper is useful once the Protegen and negative FASTA files are
    available; it is not invoked by default until data are in place.
    """
    if not HAS_BIO:
        raise ImportError("Biopython is required for FASTA parsing. Install with `pip install biopython`.")

    if not fasta_path.exists():
        raise FileNotFoundError(f"FASTA file not found: {fasta_path}")

    records = []
    for record in SeqIO.parse(str(fasta_path), "fasta"):
        organism = record.annotations.get("organism", "unknown") if hasattr(record, "annotations") else "unknown"
        records.append(
            {
                config.DEFAULT_ID_COLUMN: record.id,
                config.DEFAULT_TEXT_COLUMN: str(record.seq),
                config.DEFAULT_ORGANISM_COLUMN: organism,
                "source": source,
                config.DEFAULT_TARGET_COLUMN: label,
            }
        )
    return pd.DataFrame(records)


def load_labeled_sequences(csv_path: Path | None = None) -> Optional[pd.DataFrame]:
    """Load the processed labeled sequences table if available.

    Returns None and logs a hint if the file does not exist or is malformed.
    """
    path = csv_path or config.PROCESSED_DATA_PATH
    if not path.exists():
        logger.warning(
            "Processed labeled dataset not found at %s. Please create it with columns: %s",
            path,
            ", ".join(EXPECTED_COLUMNS),
        )
        return None

    df = pd.read_csv(path)
    missing_cols = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing_cols:
        logger.error("Dataset at %s is missing columns: %s", path, ", ".join(missing_cols))
        return None

    try:
        df[config.DEFAULT_TARGET_COLUMN] = df[config.DEFAULT_TARGET_COLUMN].astype(int)
    except Exception as exc:
        logger.warning("Could not cast %s to int: %s", config.DEFAULT_TARGET_COLUMN, exc)

    logger.info("Loaded labeled dataset with %d rows from %s", len(df), path)
    return df
