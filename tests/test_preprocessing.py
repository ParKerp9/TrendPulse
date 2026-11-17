"""Unit tests for text preprocessing, linguistic extraction, and hourly interval aggregation."""

import unittest
import pandas as pd

from src.preprocessing.aggregator import TimeAggregator
from src.preprocessing.text_cleaner import TextCleaner


class TestPreprocessing(unittest.TestCase):
    def test_text_cleaner(self):
        raw_markdown = "Check out this [link](https://example.com/test) &amp; see **bold** and *italic* text >!spoiler!<."
        cleaned = TextCleaner.clean_text(raw_markdown)
        self.assertNotIn("https://example.com/test", cleaned)
        self.assertNotIn("&amp;", cleaned)
        self.assertIn("&", cleaned)
        self.assertIn("bold", cleaned)
        self.assertIn("spoiler", cleaned)
        self.assertNotIn("*", cleaned)

    def test_linguistic_and_sentiment_features(self):
        title = "Huge breakthrough: Revolutionary AI model achieves record benchmark scores!"
        features = TextCleaner.extract_linguistic_features(title)

        self.assertEqual(features["has_exclamation"], 1)
        self.assertGreater(features["sentiment_polarity"], 0.0)
        self.assertEqual(features["sentiment_label"], "positive")
        self.assertGreater(features["title_word_count"], 5)

    def test_negative_sentiment(self):
        title = "System collapsed after catastrophic security exploit and massive financial loss."
        features = TextCleaner.extract_linguistic_features(title)

        self.assertLess(features["sentiment_polarity"], 0.0)
        self.assertEqual(features["sentiment_label"], "negative")

    def test_hourly_aggregation(self):
        sample_df = pd.DataFrame([
            {
                "post_id": "p1",
                "title": "First post",
                "text": "Body",
                "timestamp_utc": 1700000000,
                "score": 100,
                "comment_count": 20,
                "subreddit": "technology",
                "upvote_ratio": 0.9,
            },
            {
                "post_id": "p2",
                "title": "Second post same hour",
                "text": "Body",
                "timestamp_utc": 1700001800,  # +30 minutes, same hour
                "score": 250,
                "comment_count": 50,
                "subreddit": "technology",
                "upvote_ratio": 0.95,
            },
            {
                "post_id": "p3",
                "title": "Third post different subreddit",
                "text": "Body",
                "timestamp_utc": 1700001000,
                "score": 300,
                "comment_count": 30,
                "subreddit": "science",
                "upvote_ratio": 0.92,
            },
        ])

        agg = TimeAggregator.aggregate_hourly(sample_df, by_subreddit=True, fill_gaps=False)

        tech_agg = agg[agg["subreddit"] == "technology"]
        self.assertEqual(len(tech_agg), 1)
        self.assertEqual(tech_agg["post_volume"].iloc[0], 2)
        self.assertEqual(tech_agg["total_score"].iloc[0], 350)
        self.assertEqual(tech_agg["total_comments"].iloc[0], 70)
        self.assertEqual(tech_agg["total_interactions"].iloc[0], 420)


if __name__ == "__main__":
    unittest.main()
