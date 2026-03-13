"""Tests for data cleaning module."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from src.data_cleaning import clean_dataset, build_job_text


def test_clean_dataset_dedupe():
    df = pd.DataFrame({
        "job_id": [1, 1, 2],
        "title": ["A", "A", "B"],
        "company": ["X", "X", "Y"],
        "description": ["d1", "d1", "d2"],
    })
    out = clean_dataset(df)
    assert len(out) == 2


def test_build_job_text():
    df = pd.DataFrame({
        "title": ["Data Scientist"],
        "company": ["Acme"],
        "description": ["Python SQL"],
    })
    text = build_job_text(df)
    assert "Data Scientist" in text.iloc[0]
    assert "Python" in text.iloc[0]
