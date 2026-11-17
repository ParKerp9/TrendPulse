"""Unit tests for Reddit data collection, validation, and storage modules."""

import os
import tempfile
import unittest
from pathlib import Path
import pandas as pd

from src.ingestion.base_collector import BaseCollector
from src.ingestion.reddit_collector import RedditCollector
from src.ingestion.storage import StorageHandler


class TestCollector(unittest.TestCase):
    def test_timestamp_normalization(self):
        # 1. Epoch integer
        epoch_ts, iso_ts = BaseCollector.normalize_timestamp(1700000000)
        self.assertEqual(epoch_ts, 1700000000)
        self.assertEqual(iso_ts, "2023-11-14T22:13:20Z")

        # 2. ISO 8601 String
        epoch_ts2, iso_ts2 = BaseCollector.normalize_timestamp("2023-11-14T22:13:20Z")
        self.assertEqual(epoch_ts2, 1700000000)
        self.assertEqual(iso_ts2, "2023-11-14T22:13:20Z")

    def test_post_validation(self):
        valid_post = {
            "post_id": "t3_test123",
            "title": "Quantum Computing Breakthrough Observed",
            "text": "Detailed laboratory description here.",
            "timestamp": 1700000000,
            "score": 450,
            "comment_count": 32,
            "subreddit": "technology",
        }
        is_valid, reason = BaseCollector.validate_post(valid_post)
        self.assertTrue(is_valid)
        self.assertIsNone(reason)

        # Missing required field
        invalid_post = valid_post.copy()
        del invalid_post["score"]
        is_valid, reason = BaseCollector.validate_post(invalid_post)
        self.assertFalse(is_valid)
        self.assertIn("score", reason)

        # Negative comment count
        invalid_comments = valid_post.copy()
        invalid_comments["comment_count"] = -5
        is_valid, reason = BaseCollector.validate_post(invalid_comments)
        self.assertFalse(is_valid)

    def test_deduplication(self):
        collector = RedditCollector()
        posts = [
            {"post_id": "p1", "title": "A"},
            {"post_id": "p2", "title": "B"},
            {"post_id": "p1", "title": "A duplicate"},
            {"post_id": "p3", "title": "C"},
        ]
        unique = collector.deduplicate_posts(posts)
        self.assertEqual(len(unique), 3)
        self.assertEqual([p["post_id"] for p in unique], ["p1", "p2", "p3"])

    def test_storage_csv(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            test_csv = Path(tmp_dir) / "test_posts.csv"
            storage = StorageHandler(storage_type="csv", csv_path=str(test_csv))

            posts = [
                {
                    "post_id": "p100",
                    "title": "Test Title 1",
                    "text": "Body 1",
                    "timestamp_utc": 1700000000,
                    "timestamp_iso": "2023-11-14T22:13:20Z",
                    "score": 100,
                    "comment_count": 10,
                    "subreddit": "technology",
                    "author": "author1",
                    "url": "http://example.com/1",
                    "upvote_ratio": 0.95,
                    "is_self": True,
                },
                {
                    "post_id": "p101",
                    "title": "Test Title 2",
                    "text": "Body 2",
                    "timestamp_utc": 1700001000,
                    "timestamp_iso": "2023-11-14T22:30:00Z",
                    "score": 200,
                    "comment_count": 20,
                    "subreddit": "artificial",
                    "author": "author2",
                    "url": "http://example.com/2",
                    "upvote_ratio": 0.92,
                    "is_self": False,
                },
            ]

            saved_count = storage.save_posts(posts)
            self.assertEqual(saved_count, 2)
            self.assertTrue(test_csv.exists())

            # Attempting to save same posts again should skip duplicates
            second_save = storage.save_posts(posts)
            self.assertEqual(second_save, 0)

            # Load back and verify
            df = storage.load_posts(str(test_csv))
            self.assertEqual(len(df), 2)
            self.assertEqual(set(df["post_id"]), {"p100", "p101"})


if __name__ == "__main__":
    unittest.main()
