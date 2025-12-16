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
- **Quick start:** With Protegen FASTA at `data/raw/protegen-*.faa` and proteome FASTAs under `data/raw/proteomes/` (`.faa` or `.fasta`), rebuild the processed dataset:
  ```bash
  python scripts/prepare_dataset.py \
    --protegen-path data/raw/protegen-all-4.0-2019-01-09.faa \
    --proteome-paths data/raw/proteomes/*.faa data/raw/proteomes/*.fasta
  ```
  Overlaps between Protegen and proteomes are removed by ID or exact sequence; IDs are parsed from the last pipe-delimited header token.
- **Outputs:** `data/processed/labeled_sequences.csv` with columns `protein_id`, `sequence`, `organism`, `source`, `label` (1 protective, 0 negative).

## Running the pipeline
Once `data/processed/labeled_sequences.csv` exists:
```bash
antigen-pipeline --results-dir results/
# Or explicitly run
PYTHONPATH=src python -m interpretable_antigen_classifier.cli --results-dir results/
```

Key flags:
- `--skip-psortb` / `--skip-shap` to disable optional stages (defaults auto-detect).
- `--split-strategy stratified|group|matched` (`matched` keeps organisms with both labels and uses group split; `--min-per-label` controls the threshold).
- `--dedup-mode strict|lenient` and `--negative-multiplier N` to control deduplication and post-dedup negative downsampling (set to 0 to disable).
- `--tune-hyperparameters` to run a small sweep for RF/XGBoost/logreg before final training.
- `--rebalance-seeds 1 2 3` to repeat rebalance/train across seeds and save aggregated metrics/plots.
- `--disable-kmers`, `--kmer-sizes`, `--kmer-top-n` to control k-mer features; `--disable-xgboost` to skip XGBoost.
- `--shap-top-n`, `--perm-top-n` to control interpretability plots.

Outputs (written to `results/`):
- `metrics.json` – ROC-AUC, PR-AUC, split info, dataset/dedup/rebalance summaries.
- `feature_importances.csv` / `feature_importances.png`.
- `shap_summary.png` (if SHAP available) or `permutation_importance.csv` + `permutation_importance.png`.
- Trained model pickle under `artifacts/`.
- `aggregated_metrics.json` + `aggregated_metrics.png` when multiple seeds are provided (`--rebalance-seeds`).
- `aggregated_metrics.json` + `aggregated_metrics.png` when multiple seeds are provided (`--rebalance-seeds`).

## Features (current)
- Sequence length, amino-acid composition, hydrophobic/charge fractions.
- Aromatic/aliphatic fractions, approximate molecular weight, simple net charge estimate.
- k-mer frequencies (default 2-mer + 3-mer, top-N per size, length-normalized).
- Optional PSORTb localisation (stubbed; auto-skipped if unavailable).

## Graceful degradation
- If PSORTb or SHAP are unavailable, the pipeline logs a warning and continues with available features/interpretability methods.
- If data are missing, the CLI exits after printing the required files and locations.
