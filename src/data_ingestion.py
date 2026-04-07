"""
Data ingestion module: load and discover Kaggle LinkedIn job postings dataset.
Supports LinkedIn Job 2023 (kagglehub) and simple CSV formats.
"""

from pathlib import Path
from typing import Optional

import pandas as pd

from config.settings import (
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    DEFAULT_JOB_CSV,
    CLEANED_CSV,
    CLEANED_PARQUET,
    DATASET_COLUMN_MAPPING,
)
from utils.logging_config import get_logger

logger = get_logger("data_ingestion")

LINKEDIN_2023_RAW = RAW_DATA_DIR / "linkedin_jobs_2023.csv"


def linkedin_loader_sample_frac() -> Optional[float]:
    """None = full LinkedIn 2023 merge; set LINKEDIN_SAMPLE_FRAC=0.1 for a quick dev sample."""
    import os

    env_s = os.environ.get("LINKEDIN_SAMPLE_FRAC", "").strip()
    return float(env_s) if env_s else None


def _try_linkedin_2023() -> Optional[pd.DataFrame]:
    """Load LinkedIn 2023 if available (from download script)."""
    if LINKEDIN_2023_RAW.exists():
        logger.info("Loading LinkedIn 2023 from %s", LINKEDIN_2023_RAW)
        return pd.read_csv(LINKEDIN_2023_RAW, low_memory=False, on_bad_lines="skip")
    return None


def _find_data_file() -> Optional[Path]:
    """Locate job postings CSV in raw or processed dir, or project root."""
    candidates = [
        RAW_DATA_DIR / DEFAULT_JOB_CSV,
        RAW_DATA_DIR / "linkedin_jobs_sample.csv",
        RAW_DATA_DIR / "sample_jobs_expanded.csv",
        RAW_DATA_DIR / "sample_jobs.csv",
        RAW_DATA_DIR / "jobs.csv",
        RAW_DATA_DIR / "linkedin_jobs.csv",
        PROCESSED_DATA_DIR / CLEANED_CSV,
        Path.cwd() / DEFAULT_JOB_CSV,
        Path.cwd() / "data" / "raw" / DEFAULT_JOB_CSV,
    ]
    for p in candidates:
        if p.exists():
            return p
    # Fallback: any CSV in raw
    for f in sorted(RAW_DATA_DIR.glob("*.csv")):
        return f
    return None


def _map_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Map dataset columns to canonical names."""
    canonical = {}
    for canon_name, aliases in DATASET_COLUMN_MAPPING.items():
        for alias in aliases:
            if alias in df.columns:
                canonical[alias] = canon_name
                break
    if not canonical:
        return df
    return df.rename(columns=canonical)


def load_raw_dataset(path=None) -> pd.DataFrame:
    """
    Load raw job postings dataset.
    Tries: 1) LinkedIn 2023 (if prepared), 2) LINKEDIN_2023_PATH env, 3) path arg, 4) auto-discover CSV.

    Args:
        path: Explicit path to CSV or LinkedIn 2023 folder (from kagglehub.dataset_download).
              If None, checks env LINKEDIN_2023_PATH and auto-discover.

    Returns:
        DataFrame with job_id, title, company, location, description, skills, etc.
    """
    import os

    # 1) LinkedIn 2023 pre-merged CSV (from download script)
    df = _try_linkedin_2023()
    if df is not None:
        return df

    # 2) LINKEDIN_2023_PATH env (path from kagglehub.dataset_download)
    env_path = os.environ.get("LINKEDIN_2023_PATH")
    if env_path:
        p = Path(env_path)
        if p.is_dir() and (p / "job_postings.csv").exists():
            from src.linkedin_2023_loader import load_linkedin_2023
            return load_linkedin_2023(p, sample_frac=linkedin_loader_sample_frac())

    # 3) Explicit path - could be LinkedIn 2023 folder from kagglehub
    if path is not None:
        p = Path(path)
        if p.is_dir() and (p / "job_postings.csv").exists():
            from src.linkedin_2023_loader import load_linkedin_2023
            return load_linkedin_2023(p, sample_frac=linkedin_loader_sample_frac())
        if str(p).endswith(".csv"):
            df = pd.read_csv(p, low_memory=False, on_bad_lines="skip")
            df = _map_columns(df)
            return df

    # 4) Auto-discover CSV
    p = _find_data_file()
    if p is None:
        raise FileNotFoundError(
            f"No job data found. Run: python scripts/download_linkedin_2023.py\n"
            f"Or place a CSV in {RAW_DATA_DIR}"
        )
    logger.info("Loading dataset from %s", p)
    df = pd.read_csv(p, low_memory=False, on_bad_lines="skip")
    df = _map_columns(df)
    logger.info("Loaded %d rows, %d columns", len(df), len(df.columns))
    return df


def load_cleaned_dataset(use_parquet: bool = True) -> Optional[pd.DataFrame]:
    """
    Load pre-cleaned dataset from cache (Parquet or CSV).

    Args:
        use_parquet: Prefer Parquet if available.

    Returns:
        DataFrame or None if cache missing.
    """
    pq_path = PROCESSED_DATA_DIR / CLEANED_PARQUET
    csv_path = PROCESSED_DATA_DIR / CLEANED_CSV
    if use_parquet and pq_path.exists():
        logger.info("Loading cached Parquet from %s", pq_path)
        return pd.read_parquet(pq_path)
    if csv_path.exists():
        logger.info("Loading cached CSV from %s", csv_path)
        return pd.read_csv(csv_path, low_memory=False)
    return None


def save_cleaned_dataset(df: pd.DataFrame) -> None:
    """Persist cleaned dataset to Parquet and CSV."""
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    pq_path = PROCESSED_DATA_DIR / CLEANED_PARQUET
    csv_path = PROCESSED_DATA_DIR / CLEANED_CSV
    df.to_parquet(pq_path, index=False)
    df.to_csv(csv_path, index=False)
    logger.info("Saved cleaned dataset to %s and %s", pq_path, csv_path)
