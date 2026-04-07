"""
Recommendation engine: missing skills, courses/certifications (Google links), learning plan.
"""

from typing import List, Optional, Tuple
from urllib.parse import quote_plus

import pandas as pd

from src.feature_engineering import is_valid_skill
from src.skill_gap import aggregate_missing_skills, skill_gap_per_match
from src.skill_curation import is_recommendable_skill
from utils.logging_config import get_logger

logger = get_logger("recommendations")


def build_google_search_url(query: str) -> str:
    """Build Google search URL for courses/certifications."""
    encoded = quote_plus(query)
    return f"https://www.google.com/search?q={encoded}"


def _platform_urls(skill: str) -> dict:
    """Generate direct search URLs for major course platforms."""
    q = quote_plus(skill)
    return {
        "coursera": f"https://www.coursera.org/courses?query={q}",
        "udemy": f"https://www.udemy.com/courses/search/?q={q}",
        "linkedin": f"https://www.linkedin.com/learning/search?keywords={q}",
        "edx": f"https://www.edx.org/search?q={q}",
        "google": build_google_search_url(f"{skill} certification course"),
    }


def get_course_recommendations(
    missing_skills: List[str],
    max_per_skill: int = 2,
) -> List[dict]:
    """
    Generate course/certification recommendations with platform links.
    """
    recommendations = []
    seen: set = set()
    for skill in missing_skills[:25]:
        if not skill or not str(skill).strip() or not is_valid_skill(str(skill)) or not is_recommendable_skill(str(skill)):
            continue
        sk = skill.lower()
        if sk in seen:
            continue
        seen.add(sk)
        recommendations.append({
            "skill": skill,
            "query": f"{skill} certification & course",
            "google_url": build_google_search_url(f"{skill} certification course"),
            "platform_urls": _platform_urls(skill),
        })
    return recommendations


def build_learning_roadmap(
    missing_skills: List[Tuple[str, int]],
    course_recs: List[dict],
    top_n: int = 15,
) -> List[dict]:
    """
    Build prioritized learning plan based on skill frequency and course availability.

    Args:
        missing_skills: List of (skill, frequency) from aggregate_missing_skills.
        course_recs: Course recommendations from get_course_recommendations.
        top_n: Number of items in roadmap.

    Returns:
        Prioritized list of learning items with priority, skill, action, link.
    """
    skill_to_courses = {}
    for rec in course_recs:
        s = rec["skill"]
        if s not in skill_to_courses:
            skill_to_courses[s] = []
        skill_to_courses[s].append(rec)

    roadmap = []
    for i, (skill, freq) in enumerate(missing_skills[:top_n]):
        if not is_valid_skill(str(skill)) or not is_recommendable_skill(str(skill)):
            continue
        priority = i + 1
        courses = skill_to_courses.get(skill, [])
        if courses:
            rec = courses[0]
            roadmap.append({
                "priority": priority,
                "skill": skill,
                "action": f"Take course/certification: {rec['query']}",
                "google_url": rec["google_url"],
                "platform_urls": rec.get("platform_urls", {}),
                "frequency": freq,
            })
        else:
            roadmap.append({
                "priority": priority,
                "skill": skill,
                "action": f"Learn {skill} - search for resources",
                "google_url": build_google_search_url(f"{skill} course tutorial"),
                "platform_urls": _platform_urls(skill),
                "frequency": freq,
            })
    return roadmap


def empty_recommendations() -> dict:
    """Baseline recommendation payload with no per-job gaps (for corpus-only enrichment)."""
    return {
        "missing_skills": [],
        "missing_with_freq": [],
        "skill_gaps_df": pd.DataFrame(),
        "course_recommendations": [],
        "learning_roadmap": [],
        "market_role_focus": "",
        "gaps_source": "matched_postings",
        "market_prioritized_skills": set(),
    }


def generate_all_recommendations(
    user_skills: List[str],
    matched_jobs: pd.DataFrame,
    job_skills_series: pd.Series,
) -> dict:
    """
    Generate full recommendation suite: gaps, courses, roadmap.

    Args:
        user_skills: User's skills.
        matched_jobs: Top matched jobs.
        job_skills_series: Skills per job.

    Returns:
        Dict with missing_skills, skill_gaps_df, course_recommendations, learning_roadmap.
    """
    gap_df = skill_gap_per_match(user_skills, matched_jobs, job_skills_series)
    missing_agg = aggregate_missing_skills(gap_df)
    missing_list = [s for s, _ in missing_agg]
    course_recs = get_course_recommendations(missing_list)
    roadmap = build_learning_roadmap(missing_agg, course_recs)
    return {
        "missing_skills": missing_list,
        "missing_with_freq": missing_agg,
        "skill_gaps_df": gap_df,
        "course_recommendations": course_recs,
        "learning_roadmap": roadmap,
        "market_role_focus": "",
        "gaps_source": "matched_postings",
        "market_prioritized_skills": set(),
    }


def augment_recommendations_with_market_role(recs: dict, top_role_row: pd.Series) -> dict:
    """
    Re-rank gaps, courses, and roadmap using skills that are **often required** for the
    corpus role that best matches the resume (first row of role-market fit).
    """
    if top_role_row is None or top_role_row.empty:
        return recs
    gaps = top_role_row.get("market_skills_gap")
    if gaps is None or (isinstance(gaps, float) and pd.isna(gaps)):
        gaps = []
    if not isinstance(gaps, list):
        gaps = list(gaps) if hasattr(gaps, "__iter__") and not isinstance(gaps, str) else []

    role_name = str(top_role_row.get("role_display") or top_role_row.get("role_key") or "").strip()
    n_post = int(top_role_row.get("postings_in_corpus") or 0)

    market_pairs: List[Tuple[str, int]] = []
    base = 250
    for i, s in enumerate(gaps):
        if not s or not is_valid_skill(str(s)) or not is_recommendable_skill(str(s)):
            continue
        market_pairs.append((str(s).strip(), base - i))

    if not market_pairs:
        return recs

    old_pairs = list(recs.get("missing_with_freq") or [])
    seen = {str(s).strip().lower() for s, _ in market_pairs}
    rest = [(s, f) for s, f in old_pairs if str(s).strip().lower() not in seen]

    merged = market_pairs + rest
    missing_list = [s for s, _ in merged[:40]]

    course_recs = get_course_recommendations(missing_list)
    roadmap = build_learning_roadmap(merged, course_recs)

    out = dict(recs)
    out["missing_skills"] = missing_list
    out["missing_with_freq"] = merged[:45]
    out["course_recommendations"] = course_recs
    out["learning_roadmap"] = roadmap
    out["market_role_focus"] = role_name
    out["market_role_postings"] = n_post
    out["market_role_gap_count"] = len(market_pairs)
    out["gaps_source"] = "corpus_best_role" if market_pairs else "matched_postings"
    out["market_prioritized_skills"] = {str(s).strip().lower() for s, _ in market_pairs}
    return out
