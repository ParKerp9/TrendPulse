"""Hourly interval aggregation and engagement metric computations."""

from typing import Optional
import numpy as np
import pandas as pd

from src.preprocessing.text_cleaner import TextCleaner
from src.utils.logger import setup_logger

logger = setup_logger("TimeAggregator")


class TimeAggregator:
    """Aggregates post data into uniform hourly intervals and calculates engagement density metrics."""

    @staticmethod
    def prepare_post_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        """
        Cleans and enriches raw post records with datetime indices and linguistic/sentiment metrics.
        """
        if df.empty:
            return df

        df = df.copy()

        # Parse timestamps into datetime with UTC
        if "timestamp_utc" in df.columns:
            df["datetime"] = pd.to_datetime(df["timestamp_utc"], unit="s", utc=True)
        elif "timestamp_iso" in df.columns:
            df["datetime"] = pd.to_datetime(df["timestamp_iso"], utc=True)
        else:
            raise ValueError("DataFrame must contain 'timestamp_utc' or 'timestamp_iso' column.")

        # Floor datetime to hourly interval bucket
        df["hourly_bucket"] = df["datetime"].dt.floor("h")

        # Linguistic & Sentiment features if not already calculated
        if "sentiment_polarity" not in df.columns:
            sentiment_records = []
            for _, row in df.iterrows():
                feats = TextCleaner.extract_linguistic_features(
                    title=str(row.get("title", "")),
                    body=str(row.get("text", "")),
                )
                sentiment_records.append(feats)

            sentiment_df = pd.DataFrame(sentiment_records)
            for col in sentiment_df.columns:
                df[col] = sentiment_df[col].values

        # Ensure numeric score and comment_count
        df["score"] = pd.to_numeric(df["score"], errors="coerce").fillna(0).astype(int)
        df["comment_count"] = pd.to_numeric(df["comment_count"], errors="coerce").fillna(0).astype(int)
        df["interactions"] = df["score"] + df["comment_count"]
        df["discussion_ratio"] = df["comment_count"] / (df["score"] + 1)

        return df

    @classmethod
    def aggregate_hourly(
        cls,
        df: pd.DataFrame,
        by_subreddit: bool = True,
        fill_gaps: bool = True,
    ) -> pd.DataFrame:
        """
        Aggregates posts into hourly bins.
        If by_subreddit is True, aggregates per (subreddit, hourly_bucket).
        If fill_gaps is True, inserts empty hourly bins with 0 volume across the observed range.
        """
        prepared_df = cls.prepare_post_dataframe(df)
        if prepared_df.empty:
            return pd.DataFrame()

        group_cols = ["subreddit", "hourly_bucket"] if by_subreddit else ["hourly_bucket"]

        # Aggregate functions
        agg_funcs = {
            "post_id": "count",
            "score": ["sum", "mean", "median", "max"],
            "comment_count": ["sum", "mean", "median", "max"],
            "interactions": ["sum", "mean", "max"],
            "discussion_ratio": "mean",
            "sentiment_polarity": "mean",
            "sentiment_subjectivity": "mean",
            "title_word_count": "mean",
            "uppercase_ratio": "mean",
        }

        if "upvote_ratio" in prepared_df.columns:
            agg_funcs["upvote_ratio"] = "mean"

        agg_df = prepared_df.groupby(group_cols).agg(agg_funcs)

        # Flatten multi-level columns
        flat_columns = []
        for col, stat in agg_df.columns:
            if stat:
                flat_columns.append(f"{col}_{stat}")
            else:
                flat_columns.append(col)
        agg_df.columns = flat_columns

        # Rename count column to post_volume
        agg_df = agg_df.rename(
            columns={
                "post_id_count": "post_volume",
                "score_sum": "total_score",
                "comment_count_sum": "total_comments",
                "interactions_sum": "total_interactions",
                "discussion_ratio_mean": "avg_discussion_ratio",
                "sentiment_polarity_mean": "avg_sentiment_polarity",
                "sentiment_subjectivity_mean": "avg_sentiment_subjectivity",
                "title_word_count_mean": "avg_title_words",
                "uppercase_ratio_mean": "avg_uppercase_ratio",
            }
        )

        agg_df = agg_df.reset_index()

        if fill_gaps and not agg_df.empty:
            agg_df = cls._fill_hourly_gaps(agg_df, by_subreddit=by_subreddit)

        # Engagement density: comments per total score
        agg_df["engagement_density"] = agg_df["total_comments"] / (agg_df["total_score"] + 1)

        return agg_df

    @classmethod
    def _fill_hourly_gaps(cls, df: pd.DataFrame, by_subreddit: bool) -> pd.DataFrame:
        """Fills missing hourly timestamps with zero-counts so lag/rolling operations stay regular."""
        if by_subreddit:
            subreddits = df["subreddit"].unique()
            filled_list = []
            min_time = df["hourly_bucket"].min()
            max_time = df["hourly_bucket"].max()
            full_range = pd.date_range(start=min_time, end=max_time, freq="1h", tz="UTC")

            for sub in subreddits:
                sub_df = df[df["subreddit"] == sub].set_index("hourly_bucket")
                sub_reindexed = sub_df.reindex(full_range)
                sub_reindexed["subreddit"] = sub
                # Fill metrics with zero or reasonable defaults
                numeric_cols = sub_reindexed.select_dtypes(include=[np.number]).columns
                sub_reindexed[numeric_cols] = sub_reindexed[numeric_cols].fillna(0.0)
                sub_reindexed = sub_reindexed.reset_index().rename(columns={"index": "hourly_bucket"})
                filled_list.append(sub_reindexed)

            return pd.concat(filled_list, ignore_index=True).sort_values(
                by=["subreddit", "hourly_bucket"]
            )
        else:
            min_time = df["hourly_bucket"].min()
            max_time = df["hourly_bucket"].max()
            full_range = pd.date_range(start=min_time, end=max_time, freq="1h", tz="UTC")
            df_reindexed = df.set_index("hourly_bucket").reindex(full_range)
            numeric_cols = df_reindexed.select_dtypes(include=[np.number]).columns
            df_reindexed[numeric_cols] = df_reindexed[numeric_cols].fillna(0.0)
            return df_reindexed.reset_index().rename(columns={"index": "hourly_bucket"}).sort_values(by="hourly_bucket")
