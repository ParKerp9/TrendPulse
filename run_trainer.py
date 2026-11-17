"""CLI Runner to Train, Evaluate, and Persist the 5 TrendPulse ML Models."""

import argparse
from pathlib import Path
import pandas as pd

from src.models.trainer import TrendPulseMLPipeline
from src.utils.logger import setup_logger

logger = setup_logger("RunTrainer")


def main():
    parser = argparse.ArgumentParser(description="TrendPulse ML Model Training Pipeline")
    parser.add_argument(
        "--input",
        type=str,
        default="data/processed/posts_enriched.csv",
        help="Input enriched dataset (defaults to data/processed/posts_enriched.csv)",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=str,
        default="models/artifacts",
        help="Directory to save fitted model artifacts",
    )

    args = parser.parse_args()
    input_path = Path(args.input)

    if not input_path.exists():
        # Fallback to raw if processed not yet created
        raw_path = Path("data/raw/reddit_posts.csv")
        sample_path = Path("data/raw/sample_reddit_data.csv")
        if raw_path.exists():
            input_path = raw_path
        elif sample_path.exists():
            input_path = sample_path
        else:
            logger.error(f"Input data not found at {args.input}. Run run_eda.py or run_collector.py first.")
            return

    logger.info(f"Loading training data from: {input_path}...")
    df = pd.read_csv(input_path)
    logger.info(f"Loaded {len(df)} records for ML pipeline.")

    # Execute training
    pipeline = TrendPulseMLPipeline(artifacts_dir=args.artifacts_dir)
    metrics = pipeline.train_and_evaluate(df)

    print("\n" + "=" * 65)
    print("🎯 TRENDPULSE MACHINE LEARNING TRAINING COMPLETE")
    print("=" * 65)
    print("Model 1: 24-Hour Virality Classifier (Binary)")
    v_m = metrics.get("virality_classifier", {})
    print(f"  • Baseline Accuracy: {v_m.get('baseline_accuracy', 0):.2%}")
    print(f"  • Advanced Accuracy: {v_m.get('advanced_accuracy', 0):.2%}")
    print(f"  • ROC-AUC Score:     {v_m.get('roc_auc', 0):.4f}")
    print(f"  • F1-Score:          {v_m.get('f1_score', 0):.4f}")

    print("\nModel 2: Trend Lifecycle Classifier (Multiclass: 4 Stages)")
    lc_m = metrics.get("lifecycle_classifier", {})
    print(f"  • Advanced Accuracy: {lc_m.get('advanced_accuracy', 0):.2%}")
    print(f"  • Macro F1-Score:    {lc_m.get('macro_f1', 0):.4f}")

    print("\nModel 3: Popularity Regressor (Continuous 24h Engagement)")
    pop_m = metrics.get("popularity_forecaster", {})
    print(f"  • Baseline MAE:      {pop_m.get('baseline_mae', 0):.4f}")
    print(f"  • Advanced MAE:      {pop_m.get('advanced_mae', 0):.4f}")
    print(f"  • Advanced R² Score: {pop_m.get('advanced_r2', 0):.4f}")

    print("\nModel 4: Peak Popularity Time Regressor (Hours to Peak)")
    pk_m = metrics.get("peak_time_estimator", {})
    print(f"  • Advanced MAE:      {pk_m.get('advanced_mae', 0):.2f} hours")
    print(f"  • Advanced R² Score: {pk_m.get('advanced_r2', 0):.4f}")

    print("\nModel 5: Sentiment Classification Model")
    s_m = metrics.get("sentiment_model", {})
    print(f"  • Accuracy:          {s_m.get('accuracy', 0):.2%}")
    print(f"  • Weighted F1:       {s_m.get('f1_weighted', 0):.4f}")

    print(f"\nArtifacts successfully persisted to: {args.artifacts_dir}/")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
