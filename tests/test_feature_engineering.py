"""Unit tests ensuring backward-looking causality and strict lookahead-leakage prevention."""

import unittest
import numpy as np
import pandas as pd

from src.preprocessing.feature_engineering import TimeSeriesFeatureEngineer, TopicIdentifier


class TestFeatureEngineering(unittest.TestCase):
    def test_topic_identifier(self):
        texts = [
            "Quantum computing breakthrough in qubit coherence",
            "Large language model achieves new reasoning benchmark",
            "Semiconductor foundry reports 2nm transistor yields",
            "Autonomous robotics foundation model demonstrated",
        ]
        topic_id = TopicIdentifier(n_topics=2, max_features=50)
        topic_id.fit(texts)
        self.assertTrue(topic_id.is_fitted)

        weights, dominant_keywords = topic_id.transform(["Quantum processor test"])
        self.assertEqual(weights.shape[0], 1)
        self.assertEqual(len(dominant_keywords), 1)
        self.assertIsInstance(dominant_keywords[0], str)

    def test_strict_backward_causality_and_no_lookahead_leakage(self):
        """
        CRITICAL TEST:
        Verifies that lag and rolling window features for timestamp t depend strictly
        on timestamps <= (t - 1) and NEVER on current or future timestamps.
        """
        # Create 10 consecutive hours with known increasing values
        hours = pd.date_range("2023-11-14 00:00:00", periods=10, freq="1h", tz="UTC")
        interactions = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

        hourly_df = pd.DataFrame({
            "hourly_bucket": hours,
            "post_volume": [1] * 10,
            "total_interactions": interactions,
        })

        fe = TimeSeriesFeatureEngineer(target_col="total_interactions")
        feat_df = fe.engineer_time_series_features(
            hourly_df,
            is_grouped_by_subreddit=False,
            lags=(1, 2),
            rolling_windows=(3,),
        )

        # 1. Row 0 has NO prior history -> lag_1 MUST be NaN or None
        self.assertTrue(pd.isna(feat_df["lag_1_total_interactions"].iloc[0]))

        # 2. Row 1 (t=1, value=20) -> lag_1 MUST be 10 (value at t=0)
        self.assertEqual(feat_df["lag_1_total_interactions"].iloc[1], 10)
        # Current value at t=1 is 20, which must NOT appear in lag_1
        self.assertNotEqual(feat_df["lag_1_total_interactions"].iloc[1], 20)

        # 3. For any index i >= 1, lag_1 must equal total_interactions at i - 1
        for i in range(1, len(feat_df)):
            self.assertEqual(feat_df["lag_1_total_interactions"].iloc[i], hourly_df["total_interactions"].iloc[i - 1])

        # 4. For any index i >= 2, lag_2 must equal total_interactions at i - 2
        for i in range(2, len(feat_df)):
            self.assertEqual(feat_df["lag_2_total_interactions"].iloc[i], hourly_df["total_interactions"].iloc[i - 2])

        # 5. Rolling Mean: At row 3 (where values so far are 10, 20, 30, and current is 40):
        # Backward rolling mean of window 3 over shifted series must be mean(10, 20, 30) = 20.0
        # It must NOT include current value 40!
        self.assertEqual(feat_df["rolling_mean_3h_total_interactions"].iloc[3], 20.0)

        # 6. Velocity at row 2: (interactions at t-1 minus interactions at t-2) = (20 - 10) = 10.0
        self.assertEqual(feat_df["velocity_interactions"].iloc[2], 10.0)


if __name__ == "__main__":
    unittest.main()
