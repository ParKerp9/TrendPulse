"""Storage abstractions supporting CSV, SQLite, and PostgreSQL for TrendPulse."""

import csv
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd

from src.utils.logger import setup_logger

logger = setup_logger("StorageHandler")


class StorageHandler:
    """Manages persistence of social media post records across CSV, SQLite, and PostgreSQL."""

    def __init__(
        self,
        storage_type: str = "csv",
        csv_path: str = "data/raw/reddit_posts.csv",
        sqlite_path: str = "data/trendpulse.db",
        pg_config: Optional[Dict[str, Any]] = None,
    ):
        self.storage_type = storage_type.lower()
        self.csv_path = Path(csv_path)
        self.sqlite_path = Path(sqlite_path)
        self.pg_config = pg_config or {}

        # Ensure parent directory exists for file-based outputs
        if self.storage_type in ("csv", "all"):
            self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        if self.storage_type in ("sqlite", "all"):
            self.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
            self._init_sqlite()

        if self.storage_type == "postgresql":
            self._init_postgres()

    def _init_sqlite(self):
        """Initializes SQLite schema if not existing."""
        import sqlite3

        with sqlite3.connect(self.sqlite_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS reddit_posts (
                    post_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    text TEXT,
                    timestamp_utc INTEGER NOT NULL,
                    timestamp_iso TEXT NOT NULL,
                    score INTEGER NOT NULL,
                    comment_count INTEGER NOT NULL,
                    subreddit TEXT NOT NULL,
                    author TEXT,
                    url TEXT,
                    upvote_ratio REAL,
                    is_self BOOLEAN,
                    ingested_at TEXT DEFAULT CURRENT_TIMESTAMP
                );
                """
            )
            conn.commit()

    def _init_postgres(self):
        """Attempts connection and schema initialization on PostgreSQL."""
        try:
            import psycopg2

            conn = psycopg2.connect(
                host=self.pg_config.get("host", "localhost"),
                port=self.pg_config.get("port", 5432),
                dbname=self.pg_config.get("dbname", "trendpulse_db"),
                user=self.pg_config.get("user", "postgres"),
                password=self.pg_config.get("password", ""),
            )
            with conn.cursor() as cur:
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS reddit_posts (
                        post_id VARCHAR(50) PRIMARY KEY,
                        title TEXT NOT NULL,
                        text TEXT,
                        timestamp_utc BIGINT NOT NULL,
                        timestamp_iso VARCHAR(50) NOT NULL,
                        score INT NOT NULL,
                        comment_count INT NOT NULL,
                        subreddit VARCHAR(100) NOT NULL,
                        author VARCHAR(150),
                        url TEXT,
                        upvote_ratio FLOAT,
                        is_self BOOLEAN,
                        ingested_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                    );
                    """
                )
            conn.commit()
            conn.close()
            logger.info("Successfully initialized PostgreSQL table 'reddit_posts'.")
        except Exception as e:
            logger.warning(
                f"Could not connect to PostgreSQL ({e}). Falling back to CSV storage at '{self.csv_path}'."
            )
            self.storage_type = "csv"
            self.csv_path.parent.mkdir(parents=True, exist_ok=True)

    def save_posts(self, posts: List[Dict[str, Any]]) -> int:
        """
        Saves a list of validated post dictionaries to the configured storage.
        De-duplicates against existing records. Returns count of new records saved.
        """
        if not posts:
            logger.info("No posts provided to save.")
            return 0

        if self.storage_type == "csv":
            return self._save_to_csv(posts)
        elif self.storage_type == "sqlite":
            return self._save_to_sqlite(posts)
        elif self.storage_type == "postgresql":
            return self._save_to_postgres(posts)
        else:
            return self._save_to_csv(posts)

    def _save_to_csv(self, posts: List[Dict[str, Any]]) -> int:
        """Append to CSV file while skipping existing post_ids."""
        existing_ids = set()
        file_exists = self.csv_path.exists()

        if file_exists:
            try:
                df_existing = pd.read_csv(self.csv_path, usecols=["post_id"])
                existing_ids = set(df_existing["post_id"].astype(str).tolist())
            except Exception:
                pass

        new_posts = [p for p in posts if str(p["post_id"]) not in existing_ids]

        if not new_posts:
            logger.info(f"All {len(posts)} posts are already in CSV storage.")
            return 0

        fieldnames = [
            "post_id",
            "title",
            "text",
            "timestamp_utc",
            "timestamp_iso",
            "score",
            "comment_count",
            "subreddit",
            "author",
            "url",
            "upvote_ratio",
            "is_self",
        ]

        with open(self.csv_path, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            if not file_exists or os.path.getsize(self.csv_path) == 0:
                writer.writeheader()
            for p in new_posts:
                writer.writerow(p)

        logger.info(f"Saved {len(new_posts)} new posts to CSV at '{self.csv_path}'.")
        return len(new_posts)

    def _save_to_sqlite(self, posts: List[Dict[str, Any]]) -> int:
        """Upsert posts into SQLite database."""
        import sqlite3

        inserted_count = 0
        with sqlite3.connect(self.sqlite_path) as conn:
            cursor = conn.cursor()
            for p in posts:
                cursor.execute(
                    """
                    INSERT INTO reddit_posts (
                        post_id, title, text, timestamp_utc, timestamp_iso,
                        score, comment_count, subreddit, author, url, upvote_ratio, is_self
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(post_id) DO UPDATE SET
                        score=excluded.score,
                        comment_count=excluded.comment_count,
                        upvote_ratio=excluded.upvote_ratio;
                    """,
                    (
                        str(p["post_id"]),
                        p.get("title", ""),
                        p.get("text", ""),
                        int(p.get("timestamp_utc", 0)),
                        str(p.get("timestamp_iso", "")),
                        int(p.get("score", 0)),
                        int(p.get("comment_count", 0)),
                        str(p.get("subreddit", "")),
                        str(p.get("author", "")),
                        str(p.get("url", "")),
                        float(p.get("upvote_ratio", 0.0) or 0.0),
                        bool(p.get("is_self", False)),
                    ),
                )
                if cursor.rowcount > 0:
                    inserted_count += 1
            conn.commit()

        logger.info(f"Saved/Updated {len(posts)} posts in SQLite at '{self.sqlite_path}'.")
        return inserted_count

    def _save_to_postgres(self, posts: List[Dict[str, Any]]) -> int:
        """Upsert posts into PostgreSQL database."""
        import psycopg2
        from psycopg2.extras import execute_batch

        query = """
            INSERT INTO reddit_posts (
                post_id, title, text, timestamp_utc, timestamp_iso,
                score, comment_count, subreddit, author, url, upvote_ratio, is_self
            ) VALUES (
                %(post_id)s, %(title)s, %(text)s, %(timestamp_utc)s, %(timestamp_iso)s,
                %(score)s, %(comment_count)s, %(subreddit)s, %(author)s, %(url)s, %(upvote_ratio)s, %(is_self)s
            )
            ON CONFLICT (post_id) DO UPDATE SET
                score = EXCLUDED.score,
                comment_count = EXCLUDED.comment_count,
                upvote_ratio = EXCLUDED.upvote_ratio;
        """
        try:
            conn = psycopg2.connect(
                host=self.pg_config.get("host", "localhost"),
                port=self.pg_config.get("port", 5432),
                dbname=self.pg_config.get("dbname", "trendpulse_db"),
                user=self.pg_config.get("user", "postgres"),
                password=self.pg_config.get("password", ""),
            )
            with conn.cursor() as cur:
                execute_batch(cur, query, posts)
            conn.commit()
            conn.close()
            logger.info(f"Saved/Updated {len(posts)} posts in PostgreSQL.")
            return len(posts)
        except Exception as e:
            logger.error(f"PostgreSQL batch insert failed: {e}. Fallback to CSV.")
            return self._save_to_csv(posts)

    def load_posts(self, source_path: Optional[str] = None) -> pd.DataFrame:
        """Loads posts into a pandas DataFrame from CSV or SQLite."""
        target = Path(source_path) if source_path else self.csv_path
        if target.exists():
            df = pd.read_csv(target)
            return df

        if self.sqlite_path.exists():
            import sqlite3

            with sqlite3.connect(self.sqlite_path) as conn:
                return pd.read_sql_query("SELECT * FROM reddit_posts", conn)

        return pd.DataFrame()
