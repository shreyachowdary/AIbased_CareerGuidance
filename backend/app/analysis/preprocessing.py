"""Consistent job/resume text preprocessing for vectorization (delegates to `src`)."""

from __future__ import annotations

import pandas as pd

from src.data_cleaning import build_job_text


def job_text_series(df: pd.DataFrame) -> pd.Series:
    """Single string per job row (title, company, location, type, description, skills)."""
    return build_job_text(df)


def normalize_whitespace(text: str) -> str:
    import re

    return re.sub(r"\s+", " ", (text or "").strip())
