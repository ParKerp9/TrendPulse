"""Abstract Base Collector interface and data validation schemas for TrendPulse."""

from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set
import logging

logger = logging.getLogger("TrendPulse")


class BaseCollector(ABC):
    """Abstract interface defining standards for social media data collection."""

    def __init__(self, platform_name: str):
        self.platform_name = platform_name
        self.seen_post_ids: Set[str] = set()

    @abstractmethod
    def authenticate(self) -> bool:
        """Authenticate with the target platform's API."""
        pass

    @abstractmethod
    def fetch_posts(self, source: str, limit: int = 50, listing: str = "hot") -> List[Dict[str, Any]]:
        """Fetch posts from a specific channel/subreddit/source."""
        pass

    @staticmethod
    def normalize_timestamp(raw_timestamp: Any) -> tuple[int, str]:
        """
        Normalizes any given timestamp (epoch seconds, datetime, or ISO string)
        into a uniform (epoch_seconds: int, iso_utc_string: str) tuple.
        """
        if raw_timestamp is None:
            now = datetime.now(timezone.utc)
            return int(now.timestamp()), now.strftime("%Y-%m-%dT%H:%M:%SZ")

        if isinstance(raw_timestamp, (int, float)):
            dt = datetime.fromtimestamp(raw_timestamp, tz=timezone.utc)
            return int(raw_timestamp), dt.strftime("%Y-%m-%dT%H:%M:%SZ")

        if isinstance(raw_timestamp, datetime):
            if raw_timestamp.tzinfo is None:
                raw_timestamp = raw_timestamp.replace(tzinfo=timezone.utc)
            dt = raw_timestamp.astimezone(timezone.utc)
            return int(dt.timestamp()), dt.strftime("%Y-%m-%dT%H:%M:%SZ")

        if isinstance(raw_timestamp, str):
            try:
                # Try parsing ISO format
                clean_str = raw_timestamp.replace("Z", "+00:00")
                dt = datetime.fromisoformat(clean_str)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                dt = dt.astimezone(timezone.utc)
                return int(dt.timestamp()), dt.strftime("%Y-%m-%dT%H:%M:%SZ")
            except Exception:
                # Try parsing as float string
                try:
                    epoch = float(raw_timestamp)
                    dt = datetime.fromtimestamp(epoch, tz=timezone.utc)
                    return int(epoch), dt.strftime("%Y-%m-%dT%H:%M:%SZ")
                except Exception:
                    now = datetime.now(timezone.utc)
                    return int(now.timestamp()), now.strftime("%Y-%m-%dT%H:%M:%SZ")

        now = datetime.now(timezone.utc)
        return int(now.timestamp()), now.strftime("%Y-%m-%dT%H:%M:%SZ")

    @classmethod
    def validate_post(cls, post: Dict[str, Any]) -> tuple[bool, Optional[str]]:
        """
        Validates the structure and field values of an ingested post.
        Returns (is_valid: bool, error_reason: Optional[str]).
        """
        required_fields = ["post_id", "title", "timestamp", "score", "comment_count", "subreddit"]
        for field in required_fields:
            if field not in post:
                return False, f"Missing required field: '{field}'"
            if post[field] is None:
                return False, f"Field '{field}' cannot be None"

        if not str(post["post_id"]).strip():
            return False, "post_id cannot be empty"

        if not str(post["title"]).strip():
            return False, "title cannot be empty"

        if not isinstance(post["score"], (int, float)):
            return False, f"score must be numeric, got {type(post['score'])}"

        if not isinstance(post["comment_count"], (int, float)) or post["comment_count"] < 0:
            return False, f"comment_count must be a non-negative number, got {post['comment_count']}"

        return True, None

    def deduplicate_posts(self, posts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Removes duplicate posts based on post_id, updating internal tracker."""
        unique_posts = []
        duplicates_count = 0
        for p in posts:
            pid = str(p.get("post_id", "")).strip()
            if pid and pid not in self.seen_post_ids:
                self.seen_post_ids.add(pid)
                unique_posts.append(p)
            else:
                duplicates_count += 1

        if duplicates_count > 0:
            logger.debug(f"Removed {duplicates_count} duplicate posts during collection.")
        return unique_posts
