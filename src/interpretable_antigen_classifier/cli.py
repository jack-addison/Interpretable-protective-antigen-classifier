"""Command-line entrypoint for running the full pipeline."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

from interpretable_antigen_classifier import config
from interpretable_antigen_classifier.data.ingest import load_labeled_sequences
from interpretable_antigen_classifier.data.validation import log_dataset_summary, summarize_dataset
from interpretable_antigen_classifier.evaluation.metrics import save_metrics
from interpretable_antigen_classifier.features.extract import build_feature_table
from interpretable_antigen_classifier.interpretability.explain import (
    compute_shap_summary,
    extract_feature_importances,
    plot_feature_importances,
    plot_permutation_importance,
    permutation_importance_report,
    save_feature_importances,
)
from interpretable_antigen_classifier.models.train import save_model, train_baseline_models
from interpretable_antigen_classifier.utils.logging import get_logger

logger = get_logger(__name__)


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the interpretable antigen classification pipeline")
    parser.add_argument("--results-dir", type=Path, default=config.RESULTS_DIR, help="Directory to write outputs")
    parser.add_argument("--test-size", type=float, default=config.DEFAULT_TEST_SIZE, help="Test size fraction")
    parser.add_argument("--random-state", type=int, default=config.DEFAULT_RANDOM_STATE, help="Random seed")
    parser.add_argument("--skip-psortb", action="store_true", help="Skip PSORTb feature extraction")
    parser.add_argument("--skip-shap", action="store_true", help="Skip SHAP computation")
    parser.add_argument("--disable-xgboost", action="store_true", help="Disable XGBoost even if installed")
    parser.add_argument("--split-strategy", choices=["stratified", "group"], default="stratified", help="Train/test split strategy")
    parser.add_argument("--group-column", type=str, default=config.DEFAULT_ORGANISM_COLUMN, help="Column to use for group splits")
    parser.add_argument("--cv-folds", type=int, default=0, help="Optional cross-validation folds (0 to disable)")
    parser.add_argument("--disable-kmers", action="store_true", help="Disable k-mer feature generation")
    parser.add_argument("--kmer-sizes", type=int, nargs="+", default=[2, 3], help="k-mer sizes to include")
    parser.add_argument("--kmer-top-n", type=int, default=256, help="Top N k-mers per size to keep")
    parser.add_argument("--shap-top-n", type=int, default=25, help="Top features to display in SHAP beeswarm")
    parser.add_argument("--perm-top-n", type=int, default=25, help="Top features to plot for permutation importance")
    parser.add_argument("--tune-hyperparameters", action="store_true", help="Run a small hyperparameter sweep for models")
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    config.ensure_directories()
    results_dir: Path = args.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)

    data = load_labeled_sequences()
    if data is None:
        logger.error("Labeled dataset missing. See README.md for preparation instructions.")
        return 1

    log_dataset_summary(data)
    before = len(data)
    data = data.drop_duplicates(subset=[config.DEFAULT_ID_COLUMN, config.DEFAULT_TEXT_COLUMN])
    dropped = before - len(data)
    if dropped:
        logger.info("Dropped %d duplicate sequences/IDs", dropped)

    features, labels = build_feature_table(
        data,
        skip_psortb=args.skip_psortb,
        use_kmers=not args.disable_kmers,
        kmer_sizes=tuple(args.kmer_sizes) if args.kmer_sizes else (),
        kmer_top_n=args.kmer_top_n,
    )
    if features.empty:
        logger.error("No features were generated; ensure sequences are present in the dataset.")
        return 1

    groups = None
    if args.split_strategy == "group":
        if args.group_column in data.columns:
            groups = data[args.group_column]
        else:
            logger.warning("Group column %s not found; falling back to stratified split.", args.group_column)
            args.split_strategy = "stratified"

    models, metrics, splits = train_baseline_models(
        features,
        labels,
        test_size=args.test_size,
        random_state=args.random_state,
        use_xgboost=not args.disable_xgboost,
        split_strategy=args.split_strategy,
        groups=groups,
        cv_folds=args.cv_folds,
        tune_hyperparameters=args.tune_hyperparameters,
    )

    best_model_name = max(metrics.keys(), key=lambda k: metrics[k].get("roc_auc", float("-inf")))
    best_model = models[best_model_name]

    metrics_output = {
        "best_model": best_model_name,
        "models": metrics,
        "data_summary": summarize_dataset(data),
    }
    save_metrics(metrics_output, results_dir / "metrics.json")

    model_path = config.ARTIFACT_DIR / f"{best_model_name}_model.joblib"
    save_model(best_model, model_path)

    feature_importances = extract_feature_importances(best_model, list(features.columns))
    if not feature_importances.empty:
        save_feature_importances(feature_importances, results_dir / "feature_importances.csv")
        plot_feature_importances(feature_importances, results_dir / "feature_importances.png", top_n=25)
    else:
        logger.info("No feature importances available for model %s", best_model_name)

    shap_ok = False
    if not args.skip_shap:
        shap_ok = compute_shap_summary(
            best_model,
            splits.X_test,
            results_dir / "shap_summary.png",
            top_n=args.shap_top_n,
        )

    if not shap_ok:
        perm_df = permutation_importance_report(best_model, splits.X_test, splits.y_test)
        perm_path = results_dir / "permutation_importance.csv"
        perm_df.to_csv(perm_path, index=False)
        logger.info("Saved permutation importance to %s", perm_path)
        plot_permutation_importance(perm_df, results_dir / "permutation_importance.png", top_n=args.perm_top_n)

    logger.info("Pipeline complete. Best model: %s", best_model_name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
