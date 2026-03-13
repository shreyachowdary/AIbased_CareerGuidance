"""
Data cleaning and normalization: dedupe, null handling, text standardization, date parsing.
"""

import re
from typing import Optional

import pandas as pd

from utils.logging_config import get_logger

logger = get_logger("data_cleaning")


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and normalize the job postings dataset.

    Args:
        df: Raw DataFrame.

    Returns:
        Cleaned DataFrame.
    """
    df = df.copy()
    n_orig = len(df)

    # 1. Deduplicate
    df = _deduplicate(df)

    # 2. Handle nulls
    df = _handle_nulls(df)

    # 3. Standardize text fields
    df = _standardize_text(df)

    # 4. Parse dates
    df = _parse_dates(df)

    logger.info("Cleaning: %d -> %d rows", n_orig, len(df))
    return df


def _deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate rows, preferring job_id if available."""
    if "job_id" in df.columns and df["job_id"].notna().any():
        before = len(df)
        df = df.drop_duplicates(subset=["job_id"], keep="first")
        logger.info("Deduped by job_id: %d -> %d", before, len(df))
    else:
        key_cols = [c for c in ["title", "company", "description"] if c in df.columns]
        if key_cols:
            before = len(df)
            df = df.drop_duplicates(subset=key_cols, keep="first")
            logger.info("Deduped by %s: %d -> %d", key_cols, before, len(df))
        else:
            df = df.drop_duplicates(keep="first")
    return df


def _handle_nulls(df: pd.DataFrame) -> pd.DataFrame:
    """Fill or drop nulls appropriately."""
    # Critical text columns: fill with empty string for downstream processing
    text_cols = ["title", "company", "location", "description", "skills", "job_type"]
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str)

    # Drop rows with no usable content (no title and no description)
    if "title" in df.columns and "description" in df.columns:
        mask = (df["title"].str.strip() == "") & (df["description"].str.strip() == "")
        df = df[~mask]
        logger.info("Dropped %d rows with empty title and description", mask.sum())

    # Ensure job_id exists for matching; generate if missing
    if "job_id" not in df.columns or df["job_id"].isna().all():
        df["job_id"] = range(len(df))
    else:
        df["job_id"] = df["job_id"].fillna(-1).astype(int)

    return df


def _standardize_text(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize text: lowercase, strip, collapse whitespace."""
    text_cols = [c for c in df.columns if df[c].dtype == object or df[c].dtype.name == "string"]
    for col in text_cols:
        df[col] = (
            df[col]
            .astype(str)
            .str.strip()
            .str.replace(r"\s+", " ", regex=True)
        )
    return df


def _parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Parse date columns if present."""
    date_cols = ["posted_date", "posted_at", "date_posted"]
    for col in date_cols:
        if col not in df.columns:
            continue
        try:
            df[col] = pd.to_datetime(df[col], errors="coerce")
        except Exception as e:
            logger.warning("Could not parse %s: %s", col, e)
    return df


def build_job_text(df: pd.DataFrame) -> pd.Series:
    """
    Build a single text representation per job for TF-IDF / matching.

    Combines title, company, location, description, skills into one string.
    """
    parts = []
    for col in ["title", "company", "location", "job_type", "description", "skills"]:
        if col in df.columns:
            parts.append(df[col].astype(str))
        else:
            parts.append(pd.Series([""] * len(df), index=df.index))
    return (parts[0] + " " + parts[1] + " " + parts[2] + " " +
            parts[3] + " " + parts[4] + " " + parts[5]).str.strip()
