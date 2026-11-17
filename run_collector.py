"""CLI Runner for Reddit Data Collection."""

import argparse
import sys
from pathlib import Path
import pandas as pd
import yaml

from src.ingestion.reddit_collector import RedditCollector
from src.ingestion.storage import StorageHandler
from src.utils.logger import setup_logger

logger = setup_logger("RunCollector")


def load_config() -> dict:
    config_path = Path("configs/collector_config.yaml")
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {}


def main():
    parser = argparse.ArgumentParser(description="TrendPulse: Reddit Data Ingestion Module")
    parser.add_argument(
        "--subreddits",
        nargs="+",
        help="List of subreddits to collect from (e.g. technology artificial)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=50,
        help="Max posts per listing to collect",
    )
    parser.add_argument(
        "--storage",
        type=str,
        choices=["csv", "sqlite", "postgresql"],
        default="csv",
        help="Storage backend",
    )
    parser.add_argument(
        "--use-sample",
        action="store_true",
        help="Load benchmark test dataset without hitting live API",
    )

    args = parser.parse_args()
    config = load_config()

    collector_cfg = config.get("collector", {})
    subreddits = args.subreddits or collector_cfg.get("default_subreddits", ["technology", "artificial", "science"])
    listings = collector_cfg.get("listings", ["hot", "rising"])
    storage_type = args.storage or config.get("storage", {}).get("type", "csv")

    storage = StorageHandler(storage_type=storage_type)
    collector = RedditCollector(storage_handler=storage)

    sample_path = Path("data/raw/sample_reddit_data.csv")
    posts_path = Path("data/raw/reddit_posts.csv")

    if args.use_sample or not collector.is_configured():
        if not collector.is_configured():
            logger.warning("==================================================================")
            logger.warning("NOTICE: Reddit API credentials are not detected in .env.")
            logger.warning("Per policy: Never fabricating fake live data.")
            logger.warning("Switching automatically to the high-fidelity offline test dataset.")
            logger.warning("==================================================================")

        if sample_path.exists():
            df_sample = pd.read_csv(sample_path)
            sample_records = df_sample.to_dict(orient="records")
            saved = storage.save_posts(sample_records)
            logger.info(f"Loaded {len(sample_records)} benchmark test records into storage ({saved} new).")
            logger.info(f"Storage path ready: {posts_path if posts_path.exists() else sample_path}")
            return
        else:
            logger.error(f"Sample dataset not found at {sample_path}.")
            sys.exit(1)

    # If configured, proceed with official API live collection
    logger.info(f"Initiating live Reddit collection across subreddits: {subreddits}...")
    total_saved = collector.collect_and_store(
        subreddits=subreddits,
        limit_per_sub=args.limit,
        listings=listings,
    )
    logger.info(f"Collection complete. Total new items stored: {total_saved}")


if __name__ == "__main__":
    main()
