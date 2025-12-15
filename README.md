# Interpretable Protective Antigen Classifier

A lightweight, reproducible pipeline for training interpretable classifiers that distinguish protective antigens (positives) from non-protective proteins (negatives) using protein sequences and biologically motivated features.

## Goals
- Curate positives from Protegen and construct a matched negative set from bacterial proteomes.
- Extract sequence-derived features (length, amino-acid composition, k-mers, simple physicochemical proxies) with optional PSORTb localisation features.
- Train baseline models (logistic regression, random forest; optional XGBoost) with stratified or group-aware splits.
- Produce evaluation metrics (ROC-AUC, PR-AUC) and interpretable outputs (feature importances, SHAP summary if available).
- One-command pipeline (`antigen-pipeline`) that prepares data, trains, evaluates, and writes results to `results/`.

## Repository layout
- `src/interpretable_antigen_classifier/` – package code.
  - `data/ingest.py` – stubs for downloading/assembling curated datasets.
  - `features/extract.py` – sequence feature computation and optional PSORTb integration.
  - `models/train.py` – model training utilities and pipelines.
  - `evaluation/metrics.py` – metrics calculation and persistence.
  - `interpretability/explain.py` – feature importance, SHAP or permutation-based fallbacks.
  - `utils/` – configuration and logging helpers.
- `data/` – place raw and processed data (`raw/`, `processed/`).
- `results/` – model outputs, metrics, and plots.

## Setup
1. Create an environment (Python 3.10+ recommended) and install dependencies:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -e .
   # Optional extras
   pip install .[shap]
   pip install .[xgboost]
   ```

## Data preparation
- **Quick start with provided FASTAs:** If you have a Protegen FASTA at `data/raw/protegen-*.faa` and one or more proteome FASTAs under `data/raw/proteomes/`, run:
  ```bash
  python scripts/prepare_dataset.py \
    --protegen-path data/raw/protegen-all-4.0-2019-01-09.faa \
    --proteome-paths data/raw/proteomes/staphylococcus_aureus.faa
  ```
  This writes `data/processed/labeled_sequences.csv` with length-matched negatives.
- **Positives (Protegen):** Use `download_protegen_dataset` stub in `data/ingest.py` as guidance. You will likely need to manually download the Protegen export (FASTA/CSV) and place it under `data/raw/protegen.fasta` (or update `config.py`).
- **Negatives (non-protective proteins):** Sample bacterial proteomes matched by length and organism where possible. See `sample_negative_proteins` in `data/ingest.py` for the workflow; provide your own proteome FASTA files under `data/raw/`.
- **Unified labeled table:** After curation, produce a CSV at `data/processed/labeled_sequences.csv` with columns: `protein_id`, `label` (`1` protective, `0` non-protective), `organism`, `source`, `sequence`.

The CLI will check for this processed CSV and exit with guidance if it is missing.

## Running the pipeline
Once `data/processed/labeled_sequences.csv` exists:
```bash
antigen-pipeline --results-dir results/
# Or explicitly run
python -m interpretable_antigen_classifier.cli --results-dir results/
```

CLI flags:
- `--skip-psortb` to bypass PSORTb feature attempts (default auto-detects if PSORTb is installed).
- `--skip-shap` to skip SHAP; permutation importance is used otherwise.
- `--test-size` and `--random-state` to control the split.
- `--split-strategy stratified|group` and `--group-column organism` for organism-aware splits.
- `--cv-folds N` to add a light cross-validation evaluation (0 to disable).
- `--disable-kmers`, `--kmer-sizes 2 3`, `--kmer-top-n 256` to control k-mer feature generation.
- `--disable-xgboost` to skip XGBoost even if installed (it is attempted by default).
- `--shap-top-n`, `--perm-top-n` to control interpretability plots.
- `--tune-hyperparameters` to run a small sweep for RF/XGBoost/logreg before final training.
- `--dedup-mode strict|lenient` and `--negative-multiplier 3` to control deduplication and post-dedup negative rebalance.
- `--rebalance-seeds 1 2 3` to repeat rebalance/train across seeds and save aggregated metrics.

Outputs (written to `results/`):
- `metrics.json` – ROC-AUC, PR-AUC, and split info (plus dataset summary stats).
- `feature_importances.csv` – model-derived importances.
- `shap_summary.png` (if SHAP available) or `permutation_importance.csv` + `permutation_importance.png`.
- Trained model pickle under `artifacts/` (placeholder path configurable in `config.py`).
- `feature_importances.png` – quick bar plot of top importances.

## Features (current)
- Sequence length, amino-acid composition, hydrophobic/charge fractions.
- Aromatic/aliphatic fractions, approximate molecular weight, simple net charge estimate.
- k-mer frequencies (default 2-mer + 3-mer, top-N per size, length-normalized).
- Optional PSORTb localisation (stubbed; auto-skipped if unavailable).

## Graceful degradation
- If PSORTb or SHAP are unavailable, the pipeline logs a warning and continues with available features/interpretability methods.
- If data are missing, the CLI exits after printing the required files and locations.
