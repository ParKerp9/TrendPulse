"""Data loader and PostgreSQL / CSV integration layer for TrendPulse Dashboard."""

from dataclasses import dataclass
from enum import Enum
import os
from pathlib import Path
from typing import Optional, Tuple
import pandas as pd

from src.ingestion.storage import StorageHandler
from src.models.predictor import TrendPulsePredictor
from src.utils.env_loader import get_env_var, load_env
from src.utils.logger import setup_logger

logger = setup_logger("DashboardDataLoader")


class DataSourceType(Enum):
    POSTGRESQL = "PostgreSQL Database"
    CSV_LIVE = "Live Ingested CSV"
    BENCHMARK_OFFLINE = "Benchmark Test Dataset"


@dataclass
class DataSourceInfo:
    source_type: DataSourceType
    description: str
    is_live: bool
    record_count: int
    storage_path_or_host: str


class DashboardDataLoader:
    """Central repository connector with automatic PostgreSQL check and CSV fallback."""

    def __init__(self):
        load_env()
        self.predictor = None
        self._init_predictor()

    def _init_predictor(self):
        """Initializes ML predictor."""
        self.predictor = TrendPulsePredictor()

    def get_predictor(self) -> TrendPulsePredictor:
        return self.predictor

    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, DataSourceInfo]:
        """
        Attempts to load data from:
        1. PostgreSQL (if running and credentials provided)
        2. data/processed/posts_enriched.csv
        3. data/raw/reddit_posts.csv
        4. data/raw/sample_reddit_data.csv (Benchmark)
        
        Returns:
            (posts_df, hourly_df, data_source_info)
        """
        # 1. Attempt PostgreSQL
        storage_type = get_env_var("STORAGE_TYPE", "csv").lower()
        if storage_type == "postgresql":
            try:
                import psycopg2

                conn = psycopg2.connect(
                    host=get_env_var("POSTGRES_HOST", "localhost"),
                    port=int(get_env_var("POSTGRES_PORT", 5432)),
                    dbname=get_env_var("POSTGRES_DB", "trendpulse_db"),
                    user=get_env_var("POSTGRES_USER", "postgres"),
                    password=get_env_var("POSTGRES_PASSWORD", ""),
                )
                posts_df = pd.read_sql_query("SELECT * FROM reddit_posts", conn)
                conn.close()

                if not posts_df.empty:
                    info = DataSourceInfo(
                        source_type=DataSourceType.POSTGRESQL,
                        description="Live telemetry stream connected to PostgreSQL instance",
                        is_live=True,
                        record_count=len(posts_df),
                        storage_path_or_host=get_env_var("POSTGRES_HOST", "localhost"),
                    )
                    hourly_df = self._load_hourly_features()
                    return posts_df, hourly_df, info
            except Exception as e:
                logger.debug(f"PostgreSQL connection unavailable: {e}. Falling back to CSV.")

        # 2. Check processed posts
        processed_posts = Path("data/processed/posts_enriched.csv")
        raw_posts = Path("data/raw/reddit_posts.csv")
        benchmark_posts = Path("data/raw/sample_reddit_data.csv")

        if processed_posts.exists() and os.path.getsize(processed_posts) > 100:
            posts_df = pd.read_csv(processed_posts)
            hourly_df = self._load_hourly_features()
            info = DataSourceInfo(
                source_type=DataSourceType.CSV_LIVE,
                description="Persisted CSV Data Lake (Preprocessed & Enriched)",
                is_live=True,
                record_count=len(posts_df),
                storage_path_or_host=str(processed_posts),
            )
            return posts_df, hourly_df, info

        # 3. Check raw collected posts
        if raw_posts.exists() and os.path.getsize(raw_posts) > 100:
            posts_df = pd.read_csv(raw_posts)
            hourly_df = self._load_hourly_features()
            info = DataSourceInfo(
                source_type=DataSourceType.CSV_LIVE,
                description="Raw Reddit Posts Lake (Local Ingestion)",
                is_live=True,
                record_count=len(posts_df),
                storage_path_or_host=str(raw_posts),
            )
            return posts_df, hourly_df, info

        # 4. Fallback to Benchmark Test Dataset
        if benchmark_posts.exists():
            posts_df = pd.read_csv(benchmark_posts)
            hourly_df = self._load_hourly_features()
            info = DataSourceInfo(
                source_type=DataSourceType.BENCHMARK_OFFLINE,
                description="Benchmark Reference Dataset (Offline Development Mode)",
                is_live=False,
                record_count=len(posts_df),
                storage_path_or_host=str(benchmark_posts),
            )
            return posts_df, hourly_df, info

        # Empty fallback
        empty_info = DataSourceInfo(
            source_type=DataSourceType.BENCHMARK_OFFLINE,
            description="No data found",
            is_live=False,
            record_count=0,
            storage_path_or_host="None",
        )
        return pd.DataFrame(), pd.DataFrame(), empty_info

    def _load_hourly_features(self) -> pd.DataFrame:
        """Loads hourly features CSV if available."""
        hourly_path = Path("data/processed/hourly_features.csv")
        if hourly_path.exists():
            try:
                df = pd.read_csv(hourly_path)
                return df
            except Exception:
                pass
        return pd.DataFrame()
