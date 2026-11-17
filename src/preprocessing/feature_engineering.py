"""Time-Series and NLP Topic Feature Engineering with Strict Leakage Prevention."""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

from src.utils.logger import setup_logger

logger = setup_logger("FeatureEngineering")


class TopicIdentifier:
    """Extracts thematic topics and semantic keyword representations using TF-IDF and SVD."""

    def __init__(self, n_topics: int = 5, max_features: int = 500):
        self.n_topics = n_topics
        self.max_features = max_features
        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=self.max_features,
            ngram_range=(1, 2),
            min_df=1,
        )
        self.svd = TruncatedSVD(n_components=self.n_topics, random_state=42)
        self.is_fitted = False
        self.topic_keywords: Dict[int, List[str]] = {}

    def fit(self, texts: List[str]):
        """Fits topic model strictly on training corpus."""
        clean_texts = [str(t).strip() for t in texts if str(t).strip()]
        if not clean_texts:
            logger.warning("Empty text list provided for topic fitting.")
            return self

        tfidf_matrix = self.vectorizer.fit_transform(clean_texts)
        actual_components = min(self.n_topics, tfidf_matrix.shape[1] - 1, tfidf_matrix.shape[0])
        if actual_components < 1:
            actual_components = 1
        self.svd = TruncatedSVD(n_components=actual_components, random_state=42)
        self.svd.fit(tfidf_matrix)
        self.is_fitted = True

        # Extract top keywords per topic component
        terms = self.vectorizer.get_feature_names_out()
        for idx, comp in enumerate(self.svd.components_):
            top_term_indices = comp.argsort()[:-6:-1]
            top_terms = [terms[i] for i in top_term_indices]
            self.topic_keywords[idx] = top_terms

        logger.info(f"Fitted TopicIdentifier with {actual_components} latent topics.")
        return self

    def transform(self, texts: List[str]) -> Tuple[np.ndarray, List[str]]:
        """
        Transforms texts into topic weights and extracts top dominant keyword for each text.
        Guaranteed to use only vocabulary learned during training (zero test-leakage).
        """
        if not self.is_fitted:
            raise RuntimeError("TopicIdentifier must be fitted before calling transform().")

        clean_texts = [str(t).strip() for t in texts]
        tfidf_matrix = self.vectorizer.transform(clean_texts)
        topic_weights = self.svd.transform(tfidf_matrix)

        dominant_keywords = []
        for i, row in enumerate(topic_weights):
            dominant_topic = int(np.argmax(row)) if len(row) > 0 else 0
            keywords = self.topic_keywords.get(dominant_topic, ["general"])
            dominant_keywords.append(", ".join(keywords[:2]))

        return topic_weights, dominant_keywords


class TimeSeriesFeatureEngineer:
    """
    Constructs backward-looking lag, rolling window, kinetic, and cyclical calendar features.
    
    STRICT ML ORDERING & LEAKAGE PREVENTION:
    All features for timestamp t are computed strictly using data from (t - 1, t - 2, ...).
    No contemporaneous or future values are ever included in the input feature matrix.
    """

    def __init__(self, target_col: str = "total_interactions"):
        self.target_col = target_col

    def engineer_time_series_features(
        self,
        hourly_df: pd.DataFrame,
        is_grouped_by_subreddit: bool = True,
        lags: Tuple[int, ...] = (1, 2, 3, 6, 12, 24),
        rolling_windows: Tuple[int, ...] = (3, 6, 12, 24),
    ) -> pd.DataFrame:
        """
        Generates strictly causal features on an hourly time-series DataFrame.
        """
        if hourly_df.empty:
            return hourly_df

        df = hourly_df.copy()

        # Ensure correct chronological ordering
        sort_cols = ["subreddit", "hourly_bucket"] if is_grouped_by_subreddit and "subreddit" in df.columns else ["hourly_bucket"]
        df = df.sort_values(by=sort_cols).reset_index(drop=True)

        # 1. Cyclical Calendar Features (Known a priori at prediction timestamp t)
        dt_series = pd.to_datetime(df["hourly_bucket"], utc=True)
        hour = dt_series.dt.hour
        day_of_week = dt_series.dt.dayofweek

        df["hour_of_day"] = hour
        df["day_of_week"] = day_of_week
        df["is_weekend"] = day_of_week.isin([5, 6]).astype(int)

        # Sinusoidal / Cosine transformations
        df["sin_hour"] = np.sin(2 * np.pi * hour / 24.0)
        df["cos_hour"] = np.cos(2 * np.pi * hour / 24.0)
        df["sin_day"] = np.sin(2 * np.pi * day_of_week / 7.0)
        df["cos_day"] = np.cos(2 * np.pi * day_of_week / 7.0)

        # 2. Backward-Looking Lags (Shift by 1 ensures no value from current timestamp t is observed)
        group_obj = df.groupby("subreddit") if is_grouped_by_subreddit and "subreddit" in df.columns else df

        for lag in lags:
            if is_grouped_by_subreddit and "subreddit" in df.columns:
                df[f"lag_{lag}_{self.target_col}"] = group_obj[self.target_col].shift(lag)
                df[f"lag_{lag}_volume"] = group_obj["post_volume"].shift(lag)
            else:
                df[f"lag_{lag}_{self.target_col}"] = df[self.target_col].shift(lag)
                df[f"lag_{lag}_volume"] = df["post_volume"].shift(lag)

        # 3. Strictly Causal Rolling Window Aggregations
        # Shifting the series by 1 prior to rolling ensures window [t-W, ..., t-1]
        for w in rolling_windows:
            if is_grouped_by_subreddit and "subreddit" in df.columns:
                shifted_target = group_obj[self.target_col].shift(1)
                df[f"rolling_mean_{w}h_{self.target_col}"] = (
                    shifted_target.groupby(df["subreddit"]).rolling(window=w, min_periods=1).mean().values
                )
                df[f"rolling_std_{w}h_{self.target_col}"] = (
                    shifted_target.groupby(df["subreddit"]).rolling(window=w, min_periods=1).std().fillna(0).values
                )
                df[f"rolling_max_{w}h_{self.target_col}"] = (
                    shifted_target.groupby(df["subreddit"]).rolling(window=w, min_periods=1).max().values
                )
            else:
                shifted_target = df[self.target_col].shift(1)
                df[f"rolling_mean_{w}h_{self.target_col}"] = (
                    shifted_target.rolling(window=w, min_periods=1).mean()
                )
                df[f"rolling_std_{w}h_{self.target_col}"] = (
                    shifted_target.rolling(window=w, min_periods=1).std().fillna(0)
                )
                df[f"rolling_max_{w}h_{self.target_col}"] = (
                    shifted_target.rolling(window=w, min_periods=1).max()
                )

        # 4. Kinetic Features: Velocity, Acceleration, and Momentum
        # Velocity at t = (Interactions_{t-1} - Interactions_{t-2}) / 1 hour
        lag1 = df[f"lag_1_{self.target_col}"]
        lag2 = df[f"lag_2_{self.target_col}"] if f"lag_2_{self.target_col}" in df.columns else df[self.target_col].shift(2)
        lag3 = df[f"lag_3_{self.target_col}"] if f"lag_3_{self.target_col}" in df.columns else df[self.target_col].shift(3)

        df["velocity_interactions"] = (lag1 - lag2).fillna(0.0)
        df["velocity_prev_interactions"] = (lag2 - lag3).fillna(0.0)
        df["acceleration_interactions"] = (df["velocity_interactions"] - df["velocity_prev_interactions"]).fillna(0.0)

        # Relative Growth Rate: (lag1 - rolling_mean_6h) / (rolling_std_6h + 1)
        rolling_mean_6h = df.get(f"rolling_mean_6h_{self.target_col}", lag1)
        rolling_std_6h = df.get(f"rolling_std_6h_{self.target_col}", 1.0)
        df["growth_zscore_surge"] = ((lag1 - rolling_mean_6h) / (rolling_std_6h + 1e-5)).fillna(0.0)

        # 5. Future Target for Supervised Modeling (e.g. Predict whether next-hour interactions surge)
        # target_virality_next_hour: Is interactions at t in the top 20% (> 80th percentile)
        p80 = df[self.target_col].quantile(0.80) if not df.empty else 1000
        df["is_viral_event"] = (df[self.target_col] >= p80).astype(int)

        logger.info(
            f"Engineered {df.shape[1]} features across {df.shape[0]} hourly time points with strict backward causality."
        )
        return df

    @classmethod
    def attach_contextual_features_to_posts(
        cls,
        posts_df: pd.DataFrame,
        hourly_features_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Enriches individual post records with the prevailing community momentum and kinetic state
        available immediately prior to the post creation timestamp.
        """
        if posts_df.empty or hourly_features_df.empty:
            return posts_df

        merged = posts_df.copy()
        # Merge on (subreddit, hourly_bucket)
        if "subreddit" in merged.columns and "subreddit" in hourly_features_df.columns:
            context_cols = [
                c for c in hourly_features_df.columns
                if c.startswith("lag_") or c.startswith("rolling_") or c.startswith("velocity_") or c.startswith("sin_") or c.startswith("cos_")
            ]
            lookup = hourly_features_df[["subreddit", "hourly_bucket"] + context_cols].copy()
            merged = pd.merge(merged, lookup, on=["subreddit", "hourly_bucket"], how="left")
            # Fill missing lags with 0
            for col in context_cols:
                if col in merged.columns:
                    merged[col] = merged[col].fillna(0.0)

        return merged
