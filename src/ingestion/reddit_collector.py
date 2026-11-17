"""Official Reddit API Data Collector complying with current OAuth2 policies."""

import time
from typing import Any, Dict, List, Optional
import requests
from requests.auth import HTTPBasicAuth

from src.ingestion.base_collector import BaseCollector
from src.ingestion.storage import StorageHandler
from src.utils.env_loader import get_env_var, load_env
from src.utils.logger import setup_logger

logger = setup_logger("RedditCollector")


class RedditCollector(BaseCollector):
    """
    Data collector utilizing Reddit's official OAuth2 REST API.
    Adheres to Reddit API Rules:
    - Unique and descriptive User-Agent header
    - Token-based OAuth2 authentication (Application-Only / Script)
    - Paced requests honoring 'x-ratelimit-*' headers
    - Dynamic pagination with 'after' fullnames
    """

    OAUTH_TOKEN_URL = "https://www.reddit.com/api/v1/access_token"
    OAUTH_API_BASE = "https://oauth.reddit.com"

    def __init__(
        self,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        user_agent: Optional[str] = None,
        storage_handler: Optional[StorageHandler] = None,
        min_request_interval: float = 1.1,
    ):
        super().__init__(platform_name="reddit")
        load_env()

        self.client_id = client_id or get_env_var("REDDIT_CLIENT_ID")
        self.client_secret = client_secret or get_env_var("REDDIT_CLIENT_SECRET")
        self.user_agent = user_agent or get_env_var(
            "REDDIT_USER_AGENT", default="TrendPulse:v1.0.0 (by /u/TrendPulseApp)"
        )
        self.username = get_env_var("REDDIT_USERNAME")
        self.password = get_env_var("REDDIT_PASSWORD")

        self.storage = storage_handler or StorageHandler()
        self.min_request_interval = min_request_interval

        self.access_token: Optional[str] = None
        self.token_expiry: float = 0.0
        self.last_request_time: float = 0.0

        # Rate limit tracking from official Reddit headers
        self.rate_limit_remaining: float = 60.0
        self.rate_limit_reset: float = 60.0

    def is_configured(self) -> bool:
        """Returns True if minimum required Reddit API credentials are provided."""
        return bool(
            self.client_id
            and self.client_secret
            and self.client_id != "your_client_id_here"
            and self.client_secret != "your_client_secret_here"
        )

    def authenticate(self) -> bool:
        """
        Obtains OAuth2 bearer token from Reddit official OAuth endpoint.
        Supports both Script grant_type=password and Application-Only grant_type=client_credentials.
        """
        if not self.is_configured():
            logger.warning(
                "Reddit API credentials are not configured. "
                "Set REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET in .env or environment. "
                "You can continue development using the offline test dataset."
            )
            return False

        # Check if current token is still valid
        if self.access_token and time.time() < (self.token_expiry - 60):
            return True

        auth = HTTPBasicAuth(self.client_id, self.client_secret)
        headers = {"User-Agent": self.user_agent}

        if self.username and self.password:
            data = {
                "grant_type": "password",
                "username": self.username,
                "password": self.password,
            }
        else:
            data = {"grant_type": "client_credentials"}

        try:
            logger.info("Requesting OAuth2 access token from Reddit API...")
            response = requests.post(
                self.OAUTH_TOKEN_URL,
                auth=auth,
                data=data,
                headers=headers,
                timeout=15,
            )

            if response.status_code == 200:
                payload = response.json()
                self.access_token = payload.get("access_token")
                expires_in = payload.get("expires_in", 3600)
                self.token_expiry = time.time() + expires_in
                logger.info(f"Authentication successful. Token expires in {expires_in} seconds.")
                return True
            else:
                logger.error(
                    f"Authentication failed [Status {response.status_code}]: {response.text}"
                )
                return False

        except requests.exceptions.RequestException as e:
            logger.error(f"Network error during Reddit authentication: {e}")
            return False

    def _respect_rate_limits(self, response: Optional[requests.Response] = None):
        """Enforces request pacing and pauses when rate limits approach exhaustion."""
        # 1. Update limits from headers if available
        if response is not None:
            rem = response.headers.get("x-ratelimit-remaining")
            rst = response.headers.get("x-ratelimit-reset")
            if rem is not None:
                try:
                    self.rate_limit_remaining = float(rem)
                except ValueError:
                    pass
            if rst is not None:
                try:
                    self.rate_limit_reset = float(rst)
                except ValueError:
                    pass

        # 2. If remaining quota is critically low, sleep until reset
        if self.rate_limit_remaining < 5.0 and self.rate_limit_reset > 0:
            sleep_duration = self.rate_limit_reset + 1.0
            logger.warning(
                f"Rate limit nearing limit ({self.rate_limit_remaining:.1f} remaining). "
                f"Sleeping for {sleep_duration:.1f} seconds until window reset."
            )
            time.sleep(sleep_duration)
            self.rate_limit_remaining = 60.0

        # 3. Minimum pacing between requests
        elapsed = time.time() - self.last_request_time
        if elapsed < self.min_request_interval:
            time.sleep(self.min_request_interval - elapsed)
        self.last_request_time = time.time()

    def fetch_posts(
        self,
        source: str,
        limit: int = 50,
        listing: str = "hot",
        max_pages: int = 3,
    ) -> List[Dict[str, Any]]:
        """
        Fetches posts from a subreddit using official OAuth endpoint with pagination.
        Never fabricates live data: returns empty list if unauthenticated or network fails.
        """
        if not self.authenticate():
            logger.warning("Skipping live fetch: Reddit API is not authenticated.")
            return []

        subreddit = source.strip().lower().replace("r/", "")
        endpoint = f"{self.OAUTH_API_BASE}/r/{subreddit}/{listing}"
        headers = {
            "Authorization": f"Bearer {self.access_token}",
            "User-Agent": self.user_agent,
        }

        collected_posts: List[Dict[str, Any]] = []
        after_anchor: Optional[str] = None
        posts_to_fetch_per_page = min(limit, 100)

        for page in range(max_pages):
            params: Dict[str, Any] = {"limit": posts_to_fetch_per_page}
            if after_anchor:
                params["after"] = after_anchor

            self._respect_rate_limits()

            try:
                logger.info(
                    f"Fetching r/{subreddit} [{listing}] page {page + 1} (after={after_anchor})..."
                )
                response = requests.get(
                    endpoint,
                    headers=headers,
                    params=params,
                    timeout=15,
                )

                self._respect_rate_limits(response)

                # Handle HTTP 429 (Too Many Requests)
                if response.status_code == 429:
                    retry_after = float(response.headers.get("Retry-After", self.rate_limit_reset or 60.0))
                    logger.warning(f"Received HTTP 429. Backing off for {retry_after} seconds.")
                    time.sleep(retry_after)
                    continue

                if response.status_code != 200:
                    logger.error(
                        f"Failed fetching r/{subreddit}: Status {response.status_code} - {response.text[:200]}"
                    )
                    break

                data = response.json()
                children = data.get("data", {}).get("children", [])
                if not children:
                    logger.info(f"No further posts returned for r/{subreddit}.")
                    break

                for item in children:
                    post_data = item.get("data", {})
                    parsed = self._parse_reddit_post(post_data, subreddit)
                    is_valid, reason = self.validate_post(parsed)
                    if is_valid:
                        collected_posts.append(parsed)
                    else:
                        logger.debug(f"Invalid post rejected: {reason}")

                after_anchor = data.get("data", {}).get("after")
                if not after_anchor:
                    break

                if len(collected_posts) >= limit:
                    break

            except requests.exceptions.RequestException as e:
                logger.error(f"Network error querying r/{subreddit}: {e}")
                break

        # Deduplicate and return
        unique_posts = self.deduplicate_posts(collected_posts)
        logger.info(
            f"Successfully collected {len(unique_posts)} valid, unique posts from r/{subreddit} [{listing}]."
        )
        return unique_posts

    def _parse_reddit_post(self, raw: Dict[str, Any], fallback_subreddit: str) -> Dict[str, Any]:
        """Maps raw Reddit JSON object into uniform TrendPulse schema."""
        raw_utc = raw.get("created_utc", time.time())
        epoch_ts, iso_ts = self.normalize_timestamp(raw_utc)

        # Reddit score can be 0 or positive; upvote ratio is 0.0 - 1.0
        score = int(raw.get("score", 0))
        num_comments = int(raw.get("num_comments", 0))
        upvote_ratio = float(raw.get("upvote_ratio", 0.0) or 0.0)

        # Combine title and text if present
        title = str(raw.get("title", "")).strip()
        selftext = str(raw.get("selftext", "")).strip()

        # Reddit full post ID
        post_id = str(raw.get("id", "")).strip()
        if not post_id.startswith("t3_") and post_id:
            # Standardize identifier
            post_id = f"t3_{post_id}"

        return {
            "post_id": post_id,
            "title": title,
            "text": selftext,
            "timestamp_utc": epoch_ts,
            "timestamp_iso": iso_ts,
            "score": score,
            "comment_count": num_comments,
            "subreddit": str(raw.get("subreddit", fallback_subreddit)).lower(),
            "author": str(raw.get("author", "[deleted]")),
            "url": str(raw.get("url", "")),
            "upvote_ratio": upvote_ratio,
            "is_self": bool(raw.get("is_self", False)),
        }

    def collect_and_store(
        self,
        subreddits: List[str],
        limit_per_sub: int = 50,
        listings: Optional[List[str]] = None,
    ) -> int:
        """
        Orchestrates collection across multiple subreddits and listings,
        persisting all unique, validated items into storage.
        """
        if listings is None:
            listings = ["hot", "rising"]

        total_saved = 0
        for sub in subreddits:
            sub_posts = []
            for listing in listings:
                posts = self.fetch_posts(sub, limit=limit_per_sub, listing=listing)
                sub_posts.extend(posts)

            unique_sub_posts = self.deduplicate_posts(sub_posts)
            saved = self.storage.save_posts(unique_sub_posts)
            total_saved += saved

        logger.info(f"Total new posts stored in this batch: {total_saved}")
        return total_saved
