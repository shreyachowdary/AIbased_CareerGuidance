"""Skill-gap analysis: matched vs missing skills per job."""
from backend.app.services.skill_extraction import extract_skills


def analyze(
    query_skills: list[str],
    job_text: str,
    top_missing: int = 10,
) -> tuple[list[str], list[str]]:
    """
    Returns (matched_skills, missing_skills).
    missing_skills limited to top_missing.
    """
    job_skills = set(extract_skills(job_text))
    query_set = set(query_skills)
    matched = sorted(query_set & job_skills)
    missing = sorted(job_skills - query_set)[:top_missing]
    return matched, missing
