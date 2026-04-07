"""Live / API job matching — delegates to JSearch pipeline in `src`."""

from __future__ import annotations

from typing import Any, Dict, List

from src.analysis_pipelines import resolve_corpus_dataframe, run_live_pipeline


def analyze_live(
    resume_text: str,
    skills: List[str],
    raw_hint: str,
    profile: Dict[str, Any],
) -> Dict[str, Any]:
    df = resolve_corpus_dataframe()
    return run_live_pipeline(resume_text, skills, raw_hint, profile, corpus_df=df)
