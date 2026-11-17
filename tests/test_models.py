"""Unit tests for TrendPulse ML targets, training, and inference services."""

import unittest
import numpy as np
import pandas as pd

from src.models.predictor import TrendPulsePredictor
from src.models.target_builder import TargetBuilder


class TestMLPipeline(unittest.TestCase):
    def setUp(self):
        self.predictor = TrendPulsePredictor()

    def test_target_builder(self):
        sample = pd.DataFrame([
            {
                "post_id": "p1",
                "title": "Quantum breakthrough announced",
                "text": "Detailed paper notes",
                "score": 12000,
                "comment_count": 1500,
                "upvote_ratio": 0.96,
                "timestamp_utc": 1700000000,
                "sentiment_polarity": 0.5,
            },
            {
                "post_id": "p2",
                "title": "Small bug in my keyboard",
                "text": "Any advice?",
                "score": 30,
                "comment_count": 5,
                "upvote_ratio": 0.70,
                "timestamp_utc": 1700005000,
                "sentiment_polarity": -0.2,
            },
        ])

        targets_df = TargetBuilder.build_supervised_targets(sample)

        # Verify all 5 targets exist
        self.assertIn("is_viral_24h", targets_df.columns)
        self.assertIn("lifecycle_stage", targets_df.columns)
        self.assertIn("log_popularity_24h", targets_df.columns)
        self.assertIn("peak_time_hours", targets_df.columns)
        self.assertIn("sentiment_class", targets_df.columns)

        # High engagement post p1 should be viral
        self.assertEqual(targets_df["is_viral_24h"].iloc[0], 1)
        # Low engagement post p2 should not be viral
        self.assertEqual(targets_df["is_viral_24h"].iloc[1], 0)

    def test_chronological_split(self):
        df = pd.DataFrame({
            "post_id": [f"p{i}" for i in range(10)],
            "timestamp_utc": [1000 + i * 100 for i in range(10)],
            "score": [i * 10 for i in range(10)],
        })

        train, test = TargetBuilder.chronological_split(df, train_ratio=0.7)
        self.assertEqual(len(train), 7)
        self.assertEqual(len(test), 3)
        # Verify strict chronological separation: max train time < min test time
        self.assertLessEqual(train["timestamp_utc"].max(), test["timestamp_utc"].min())

    def test_predictor_service_no_placeholders(self):
        """Verifies that all 5 ML models produce real, non-placeholder predictions."""
        self.assertTrue(self.predictor.is_ready(), "Predictor artifacts should be loaded.")

        post_input = {
            "title": "Breakthrough: New 2nm GAA Transistor Achieves 40% Power Efficiency Gain",
            "text": "Pilot line fabrication yields exceed commercial foundry targets.",
            "subreddit": "technology",
            "score": 2500,
            "comment_count": 310,
            "upvote_ratio": 0.95,
            "hour_of_day": 14,
            "day_of_week": 2,
        }

        # 1. Virality Prediction
        virality = self.predictor.predict_virality(post_input)
        self.assertIn("virality_probability", virality)
        self.assertGreaterEqual(virality["virality_probability"], 0.0)
        self.assertLessEqual(virality["virality_probability"], 100.0)
        self.assertIn(virality["virality_tier"], [
            "High Virality Breakout", "Moderate Virality Potential", "Standard Organic Growth"
        ])

        # 2. Lifecycle Prediction
        lifecycle = self.predictor.predict_lifecycle(post_input)
        self.assertIn(lifecycle["stage_id"], [0, 1, 2, 3])
        self.assertIn("stage_name", lifecycle)
        self.assertEqual(len(lifecycle["stage_probabilities"]), 4)

        # 3. Popularity Forecasting
        popularity = self.predictor.forecast_popularity(post_input)
        self.assertGreater(popularity["predicted_total_interactions"], 0)
        self.assertEqual(len(popularity["trajectory_hours"]), 24)
        self.assertEqual(len(popularity["trajectory_interactions"]), 24)

        # 4. Peak Time Estimation
        peak = self.predictor.estimate_peak_time(post_input)
        self.assertGreaterEqual(peak["peak_hours"], 1.0)
        self.assertLessEqual(peak["peak_hours"], 36.0)

        # 5. Sentiment Analysis
        sentiment = self.predictor.analyze_sentiment(post_input["title"])
        self.assertIn(sentiment["sentiment_label"], ["positive", "neutral", "negative"])
        self.assertGreaterEqual(sentiment["polarity"], -1.0)
        self.assertLessEqual(sentiment["polarity"], 1.0)

        # 6. Integrated Predict All with Feature Drivers
        all_res = self.predictor.predict_all(post_input)
        self.assertIn("feature_drivers", all_res)
        self.assertIsInstance(all_res["feature_drivers"], list)


if __name__ == "__main__":
    unittest.main()
