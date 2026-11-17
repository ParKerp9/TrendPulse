"""CLI Runner for Preprocessing, Feature Engineering, and Exploratory Data Analysis (EDA)."""

import argparse
from pathlib import Path
import pandas as pd

from src.eda.analysis import TrendPulseEDA
from src.eda.visualizer import EDAVisualizer
from src.preprocessing.aggregator import TimeAggregator
from src.preprocessing.feature_engineering import TimeSeriesFeatureEngineer, TopicIdentifier
from src.preprocessing.text_cleaner import TextCleaner
from src.utils.logger import setup_logger

logger = setup_logger("RunEDA")


def main():
    parser = argparse.ArgumentParser(description="TrendPulse: Preprocessing & EDA Pipeline")
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Input CSV file path (defaults to data/raw/reddit_posts.csv or data/raw/sample_reddit_data.csv)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/eda_reports",
        help="Directory to output EDA visualizations and reports",
    )
    parser.add_argument(
        "--processed-dir",
        type=str,
        default="data/processed",
        help="Directory to save preprocessed feature datasets",
    )

    args = parser.parse_args()

    # 1. Determine input file
    input_file = None
    if args.input:
        input_file = Path(args.input)
    else:
        primary = Path("data/raw/reddit_posts.csv")
        sample = Path("data/raw/sample_reddit_data.csv")
        if primary.exists():
            input_file = primary
        elif sample.exists():
            input_file = sample
        else:
            logger.error("No input data found. Please run run_collector.py first or provide an input path.")
            return

    logger.info(f"Loading raw social media data from: {input_file}...")
    raw_df = pd.read_csv(input_file)
    logger.info(f"Loaded {len(raw_df)} records.")

    # 2. Text Preprocessing & Linguistic Analysis
    logger.info("Executing text cleaning and linguistic feature extraction...")
    posts_df = TimeAggregator.prepare_post_dataframe(raw_df)

    # 3. Topic Identification (fit on clean corpus)
    logger.info("Fitting TF-IDF and Latent Topic Identifier...")
    topic_model = TopicIdentifier(n_topics=5, max_features=300)
    topic_model.fit(posts_df["cleaned_title"].tolist())
    _, dominant_keywords = topic_model.transform(posts_df["cleaned_title"].tolist())
    posts_df["dominant_topic_keywords"] = dominant_keywords

    # 4. Hourly Aggregation & Engagement Metrics
    logger.info("Aggregating posts into uniform hourly intervals...")
    hourly_df = TimeAggregator.aggregate_hourly(posts_df, by_subreddit=True, fill_gaps=True)
    logger.info(f"Generated {len(hourly_df)} hourly bins across subreddits.")

    # 5. Time-Series Feature Engineering (Strictly Backward-Looking)
    logger.info("Engineering time-series kinetic, lag, and rolling features...")
    fe = TimeSeriesFeatureEngineer(target_col="total_interactions")
    features_df = fe.engineer_time_series_features(hourly_df, is_grouped_by_subreddit=True)

    # Attach context to post-level records
    enriched_posts_df = fe.attach_contextual_features_to_posts(posts_df, features_df)

    # 6. Save Processed Datasets
    processed_dir = Path(args.processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    enriched_posts_path = processed_dir / "posts_enriched.csv"
    hourly_path = processed_dir / "hourly_features.csv"

    enriched_posts_df.to_csv(enriched_posts_path, index=False)
    features_df.to_csv(hourly_path, index=False)
    logger.info(f"Saved processed datasets to '{processed_dir}':")
    logger.info(f" - Enriched Posts: {enriched_posts_path} ({len(enriched_posts_df)} rows)")
    logger.info(f" - Hourly Features: {hourly_path} ({len(features_df)} rows)")

    # 7. Visualization Generation (Plotly)
    logger.info(f"Generating interactive Plotly visualizations in '{args.output_dir}'...")
    visualizer = EDAVisualizer(output_dir=args.output_dir)
    visualizer.generate_all_visualizations(
        posts_df=enriched_posts_df,
        hourly_df=hourly_df,
        features_df=features_df,
    )

    # 8. Analytical Summary Report
    logger.info("Compiling comprehensive statistical EDA report...")
    analyzer = TrendPulseEDA(output_dir=args.output_dir)
    report_summary = analyzer.run_comprehensive_analysis(
        posts_df=enriched_posts_df,
        hourly_df=hourly_df,
        features_df=features_df,
        topic_identifier=topic_model,
    )

    # Output Terminal Summary
    print("\n" + "=" * 65)
    print("🚀 TRENDPULSE PREPROCESSING & EDA PIPELINE COMPLETE")
    print("=" * 65)
    print(f"Total Posts Processed:     {report_summary.get('total_posts')}")
    print(f"Hourly Intervals Analyzed: {len(features_df)}")
    print(f"Subreddits Covered:        {', '.join(report_summary.get('subreddits', []))}")
    print("\nScore Statistics (Engagement Distribution):")
    score_s = report_summary.get("score_stats", {})
    print(f"  • Mean Score:            {score_s.get('mean', 0):,.1f}")
    print(f"  • Median Score:          {score_s.get('median', 0):,.1f}")
    print(f"  • 90th % (Viral Target): {score_s.get('p90', 0):,.1f}")
    print(f"  • Max Score:             {score_s.get('max', 0):,}")
    print("\nGenerated Visualizations & Reports:")
    print(f"  • Master Dashboard Index: {args.output_dir}/index.html")
    print(f"  • Trend Growth Trajectory: {args.output_dir}/trend_growth.html")
    print(f"  • Engagement Velocity:     {args.output_dir}/engagement_velocity.html")
    print(f"  • Sentiment Breakdown:     {args.output_dir}/sentiment_distribution.html")
    print(f"  • Historical Popularity:   {args.output_dir}/historical_popularity.html")
    print(f"  • Analytical Report:       {args.output_dir}/eda_summary.md")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
