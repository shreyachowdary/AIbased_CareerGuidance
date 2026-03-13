"""
Logging configuration for the application.
"""

import logging
import sys
from pathlib import Path

from config.settings import PROJECT_ROOT

LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / "career_guidance.log"


def setup_logging(
    level: int = logging.INFO,
    log_to_file: bool = True,
) -> logging.Logger:
    """
    Configure application logging.

    Args:
        level: Logging level (default INFO).
        log_to_file: Whether to write logs to file.

    Returns:
        Root logger instance.
    """
    root = logging.getLogger("career_guidance")
    root.setLevel(level)

    if root.handlers:
        return root

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    root.addHandler(sh)

    if log_to_file:
        fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
        fh.setFormatter(fmt)
        root.addHandler(fh)

    return root


def get_logger(name: str) -> logging.Logger:
    """Get a child logger for a module."""
    return logging.getLogger(f"career_guidance.{name}")
