"""Course / roadmap generation — same module boundaries as Streamlit."""

from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd

from src.recommendations import (
    augment_recommendations_with_market_role,
    empty_recommendations,
    generate_all_recommendations,
    get_course_recommendations,
)


def from_matched_jobs(
    user_skills: List[str],
    matched_jobs: pd.DataFrame,
    job_skills_series: pd.Series,
) -> Dict[str, Any]:
    return generate_all_recommendations(user_skills, matched_jobs, job_skills_series)


def from_corpus_top_role(role_fit_row: pd.Series) -> Dict[str, Any]:
    base = empty_recommendations()
    return augment_recommendations_with_market_role(base, role_fit_row)


def courses_for_skills(skills: List[str], max_skills: int = 25) -> List[dict]:
    return get_course_recommendations(skills[:max_skills])
