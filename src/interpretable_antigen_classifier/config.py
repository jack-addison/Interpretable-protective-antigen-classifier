"""Shared configuration for paths and defaults."""
from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
PROCESSED_DATA_PATH: Path = PROCESSED_DATA_DIR / "labeled_sequences.csv"
ARTIFACT_DIR: Path = PROJECT_ROOT / "artifacts"
RESULTS_DIR: Path = PROJECT_ROOT / "results"
DEFAULT_TARGET_COLUMN: str = "label"
DEFAULT_TEXT_COLUMN: str = "sequence"
DEFAULT_ID_COLUMN: str = "protein_id"
DEFAULT_ORGANISM_COLUMN: str = "organism"
DEFAULT_TEST_SIZE: float = 0.2
DEFAULT_RANDOM_STATE: int = 42

# Optional external tools
PSORTB_BINARY: str = "psortb"


def ensure_directories() -> None:
    """Create expected output directories if they do not already exist."""
    for path in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, ARTIFACT_DIR, RESULTS_DIR]:
        path.mkdir(parents=True, exist_ok=True)
