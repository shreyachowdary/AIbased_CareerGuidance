"""
Skill gap analysis: identify missing skills vs target roles.
"""

from typing import Dict, List, Set, Tuple

import pandas as pd

from src.feature_engineering import build_skill_set_per_job, is_valid_skill
from src.skill_curation import is_recommendable_skill
from utils.logging_config import get_logger

logger = get_logger("skill_gap")


def _normalize_skill(s: str) -> str:
    """Normalize skill string for comparison."""
    return s.strip().lower()


def compute_skill_gaps(
    user_skills: List[str],
    target_job_skills: List[str],
) -> Tuple[List[str], List[str], float]:
    """
    Compute skill gaps between user and target role.

    Args:
        user_skills: User's skill list.
        target_job_skills: Required skills for the role.

    Returns:
        (matched_skills, missing_skills, match_ratio)
    """
    user_set = {_normalize_skill(s) for s in user_skills if s and is_valid_skill(s)}
    target_set = {_normalize_skill(s) for s in target_job_skills if s and is_valid_skill(s)}
    if not target_set:
        return list(user_set), [], 1.0
    matched = user_set & target_set
    missing = target_set - user_set
    ratio = len(matched) / len(target_set) if target_set else 1.0
    return list(matched), list(missing), ratio


def skill_gap_per_match(
    user_skills: List[str],
    matched_jobs: pd.DataFrame,
    job_skills_series: pd.Series,
    job_id_col: str = "job_id",
) -> pd.DataFrame:
    """
    Compute skill gaps for each matched job.

    Args:
        user_skills: User's skills.
        matched_jobs: DataFrame of matched jobs (index aligns with job_skills_series).
        job_skills_series: Series of skill lists, index-aligned with job metadata.

    Returns:
        DataFrame with match_score, matched_skills, missing_skills, gap_ratio per job.
    """
    rows = []
    for _, row in matched_jobs.iterrows():
        # Use job_index (original df index) to look up skills
        jidx = row.get("job_index", row.get("job_id", row.name))
        skills = job_skills_series.loc[jidx] if jidx in job_skills_series.index else (job_skills_series.iloc[jidx] if isinstance(jidx, int) and 0 <= jidx < len(job_skills_series) else [])
        if not isinstance(skills, list):
            skills = list(skills) if hasattr(skills, "__iter__") and not isinstance(skills, str) else [str(skills)]
        matched, missing, ratio = compute_skill_gaps(user_skills, skills)
        rows.append({
            "job_id": row.get(job_id_col, jidx),
            "job_index": jidx,
            "job_title": row.get("title", ""),
            "company": row.get("company", ""),
            "match_score": row.get("match_score", 0),
            "matched_skills": matched,
            "missing_skills": missing,
            "gap_ratio": 1 - ratio,
            "match_ratio": ratio,
        })
    return pd.DataFrame(rows)


def aggregate_missing_skills(gap_df: pd.DataFrame) -> List[Tuple[str, int]]:
    """
    Aggregate missing skills across all target jobs with frequency.

    Returns:
        List of (skill, frequency) sorted by frequency descending.
    """
    from collections import Counter
    counter: Counter = Counter()
    for missing in gap_df["missing_skills"]:
        for s in (missing or []):
            if s and is_valid_skill(s) and is_recommendable_skill(s):
                counter[s] += 1
    return counter.most_common()
