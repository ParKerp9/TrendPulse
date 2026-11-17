"""Custom structured logger for TrendPulse."""

import logging
import sys
from pathlib import Path


def setup_logger(name: str = "TrendPulse", log_file: str = "trendpulse.log", level: int = logging.INFO) -> logging.Logger:
    """Configures and returns a structured logger for stdout and file output."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(level)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-7s | %(name)s | %(filename)s:%(lineno)d | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console Handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File Handler
    try:
        log_dir = Path("logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_dir / log_file, encoding="utf-8")
        fh.setLevel(level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    except Exception:
        pass  # Fall back to console only if file creation is not permitted

    return logger
