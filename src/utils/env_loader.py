"""Environment variable loader for TrendPulse."""

import os
from pathlib import Path
from typing import Optional


def load_env(env_path: Optional[str] = None) -> dict:
    """
    Load environment variables from a .env file into os.environ.
    Does not overwrite existing environment variables.
    """
    if env_path is None:
        # Search current working directory and parent directories
        current = Path.cwd()
        potential_env = current / ".env"
        if not potential_env.exists():
            # Check project root relative to this file
            potential_env = Path(__file__).resolve().parent.parent.parent / ".env"
        target_path = potential_env
    else:
        target_path = Path(env_path)

    loaded = {}
    if target_path.exists() and target_path.is_file():
        with open(target_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                # Skip comments and empty lines
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    key, val = line.split("=", 1)
                    key = key.strip()
                    val = val.strip().strip("'\"")
                    # Do not overwrite if already set in OS environment
                    if key not in os.environ:
                        os.environ[key] = val
                    loaded[key] = os.environ[key]

    return loaded


def get_env_var(key: str, default: Optional[str] = None, required: bool = False) -> str:
    """Retrieve an environment variable with optional validation."""
    val = os.getenv(key, default)
    if required and (val is None or val.strip() == ""):
        raise ValueError(
            f"Missing required environment variable: '{key}'. "
            f"Please set it in your environment or .env file."
        )
    return val if val is not None else ""
