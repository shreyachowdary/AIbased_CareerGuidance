"""
ATS (Applicant Tracking System) keywords: extract and match resume against job market.
"""

from typing import Dict, List, Optional, Set, Tuple

from utils.logging_config import get_logger

logger = get_logger("ats_keywords")

# Common ATS keywords for tech/data roles (used when job dataset not yet loaded)
COMMON_ATS_KEYWORDS = {
    "python", "sql", "java", "javascript", "react", "aws", "docker", "kubernetes",
    "machine learning", "data science", "analytics", "excel", "tableau", "power bi",
    "tensorflow", "pytorch", "scikit-learn", "pandas", "numpy", "git", "agile",
    "rest api", "api", "etl", "data visualization", "statistics", "nlp",
    "cloud", "azure", "gcp", "linux", "sql", "nosql", "mongodb", "postgresql",
    "ci/cd", "jenkins", "terraform", "spark", "hadoop", "kafka", "redis",
    "communication", "leadership", "problem solving", "teamwork", "project management",
    "scrum", "jira", "looker", "dbt", "airflow", "mlops", "computer vision",
    "r", "typescript", "node", "node.js", "html", "css", "sas", "matlab",
}


def _normalize(s: str) -> str:
    return s.strip().lower().replace("-", " ")


def _keyword_in_resume(kw: str, resume_set: Set[str], text_lower: str) -> bool:
    """Check if keyword appears in resume (skills or text)."""
    if kw in resume_set:
        return True
    if kw in text_lower:
        return True
    for rs in resume_set:
        if kw == rs or kw in rs or rs in kw:
            return True
    return False


def extract_ats_keywords_from_resume(
    resume_skills: List[str],
    resume_text: str,
    job_keywords: Optional[Set[str]] = None,
) -> Tuple[List[str], List[str]]:
    """
    Compare resume against ATS keywords; return found vs missing.

    Args:
        resume_skills: Skills extracted from resume.
        resume_text: Raw resume text (for additional keyword matching).
        job_keywords: Optional set from job dataset; if None, use COMMON_ATS_KEYWORDS.

    Returns:
        (ats_found, ats_missing)
    """
    keywords = job_keywords or COMMON_ATS_KEYWORDS
    resume_set: Set[str] = {_normalize(s) for s in resume_skills if s}
    text_lower = _normalize(resume_text)

    found = []
    missing = []
    for kw in sorted(keywords):
        if _keyword_in_resume(kw, resume_set, text_lower):
            found.append(kw)
        else:
            missing.append(kw)
    return found, missing


def get_ats_keywords_from_jobs(job_skills_series) -> Set[str]:
    """Build ATS keyword set from job dataset skills."""
    keywords: Set[str] = set()
    for skills in job_skills_series:
        for s in (skills or []):
            if isinstance(s, str) and len(s) > 1:
                keywords.add(_normalize(s))
    return keywords


def _resume_evidence_for_keyword(kw: str, resume_skills: List[str], resume_text: str) -> str:
    """How the resume shows this job-market keyword (skill line vs body)."""
    kn = _normalize(kw)
    for s in resume_skills:
        if not s:
            continue
        sn = _normalize(str(s))
        if sn == kn or kn in sn or sn in kn or kn.replace(" ", "") in sn.replace(" ", ""):
            return f"Listed as skill: **{s}**"
    return "Mentioned in resume (matches job-market keyword)"


def build_ats_resume_job_alignment(
    resume_skills: List[str],
    resume_text: str,
    job_keywords: Set[str],
) -> Tuple[List[Dict[str, str]], List[str]]:
    """
    Compare resume to vocabulary from **matched target jobs** only (no generic ATS list on upload).

    Returns:
        matched_rows: [{ "keyword": normalized term from jobs, "evidence": str }, ...]
        missing: job-market terms not found in the resume
    """
    resume_set = {_normalize(s) for s in resume_skills if s}
    text_lower = _normalize(resume_text or "")
    matched_rows: List[Dict[str, str]] = []
    missing: List[str] = []
    for kw in sorted(job_keywords):
        if _keyword_in_resume(kw, resume_set, text_lower):
            matched_rows.append({
                "keyword": kw,
                "evidence": _resume_evidence_for_keyword(kw, resume_skills, resume_text or ""),
            })
        else:
            missing.append(kw)
    return matched_rows, missing
