"""Test preprocessing produces standardized jobs.csv."""
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_preprocess_creates_standardized_output():
    """If jobs.csv exists after preprocess, it has required columns."""
    out_path = PROJECT_ROOT / "data" / "processed" / "jobs.csv"
    if not out_path.exists():
        pytest.skip("jobs.csv not found - run: python scripts/preprocess.py")
    df = pd.read_csv(out_path)
    required = ["job_id", "title", "description", "company", "location"]
    for col in required:
        assert col in df.columns, f"Missing column: {col}"
    assert len(df) > 0
