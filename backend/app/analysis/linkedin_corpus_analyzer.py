"""Batch analysis on the local LinkedIn / job postings corpus."""

from __future__ import annotations

from typing import Any, Dict, Optional

import pandas as pd

from src.analysis_pipelines import build_corpus_artifacts, corpus_aggregate_skill_signals, resolve_corpus_dataframe, run_corpus_pipeline


def load_corpus_dataframe() -> Optional[pd.DataFrame]:
    """Load cleaned postings, or build from raw (`resolve_corpus_dataframe`)."""
    return resolve_corpus_dataframe()


def analyze_corpus(
    resume_text: str,
    skills: list,
    raw_hint: str,
    df: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """Run full corpus pipeline; `df` optional override for tests."""
    frame = df if df is not None else load_corpus_dataframe()
    return run_corpus_pipeline(resume_text, skills, frame, raw_hint)


def corpus_artifacts_only(df: pd.DataFrame):
    """Expose TF-IDF + skill series for external tools."""
    return build_corpus_artifacts(df)


def aggregate_role_skills(role_fit_df: pd.DataFrame, top_k: int = 5) -> Dict[str, Any]:
    return corpus_aggregate_skill_signals(role_fit_df, top_k=top_k)
