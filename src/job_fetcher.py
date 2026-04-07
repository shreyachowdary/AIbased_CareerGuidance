"""
Fetch real job listings from JSearch API (aggregates LinkedIn, Indeed, Glassdoor, etc.).
Key resolution order: session override → JSEARCH_API_KEY env / .env → Streamlit secrets.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import requests

from utils.logging_config import get_logger

logger = get_logger("job_fetcher")

# Load .env if present
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass

JSEARCH_HOST = "jsearch.p.rapidapi.com"
JSEARCH_URL = "https://jsearch.p.rapidapi.com/search"

_key_override: Optional[str] = None


def set_jsearch_api_key(key: Optional[str]) -> None:
    """Streamlit session can set this so users don't have to rely only on .env."""
    global _key_override
    if key is None or not str(key).strip():
        _key_override = None
    else:
        _key_override = str(key).strip()


def get_jsearch_api_key() -> str:
    """Effective RapidAPI key for JSearch."""
    if _key_override:
        return _key_override
    env_key = (os.environ.get("JSEARCH_API_KEY", "") or "").strip()
    if env_key:
        return env_key
    try:
        import streamlit as st

        if hasattr(st, "secrets") and "JSEARCH_API_KEY" in st.secrets:
            sec = st.secrets["JSEARCH_API_KEY"]
            if sec is not None and str(sec).strip():
                return str(sec).strip()
    except Exception:
        pass
    return ""


def jsearch_configured() -> bool:
    return bool(get_jsearch_api_key())


def _apply_link_from_option(opt: Any) -> str:
    if not isinstance(opt, dict):
        return ""
    for key in ("apply_link", "url", "link", "application_url"):
        v = opt.get(key)
        if isinstance(v, str) and v.strip().lower().startswith("http"):
            return v.strip()
    return ""


def _best_apply_url(job: Dict[str, Any]) -> str:
    """Prefer direct apply link, then any publisher option, then Google Jobs / employer site."""
    link = (job.get("job_apply_link") or "").strip()
    if link.lower().startswith("http"):
        return link
    for opt in job.get("apply_options") or []:
        al = _apply_link_from_option(opt)
        if al:
            return al
    g = (job.get("job_google_link") or "").strip()
    if g.lower().startswith("http"):
        return g
    for key in ("employer_website", "employer_url", "employer_website_url"):
        u = (job.get(key) or "").strip()
        if u.lower().startswith("http"):
            return u
    return ""


def fetch_jobs_from_api(
    query: str,
    num_pages: int = 2,
    date_posted: str = "month",  # all, today, 3days, week, month
) -> Optional[pd.DataFrame]:
    """
    Fetch jobs from JSearch API (real listings from LinkedIn, Indeed, etc.).

    Args:
        query: Job search query (e.g. "Data Scientist", "Python Developer").
        num_pages: Number of result pages (10 jobs per page).
        date_posted: Filter by posting date.

    Returns:
        DataFrame with job_id, title, company, location, job_type, posted_date, description, skills.
    """
    api_key = get_jsearch_api_key()
    if not api_key:
        return None

    all_jobs: List[Dict[str, Any]] = []
    for page in range(num_pages):
        try:
            resp = requests.get(
                JSEARCH_URL,
                headers={
                    "X-RapidAPI-Key": api_key,
                    "X-RapidAPI-Host": JSEARCH_HOST,
                },
                params={
                    "query": query,
                    "page": str(page + 1),
                    "date_posted": date_posted,
                },
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()
            jobs = data.get("data", [])
            if not jobs:
                break
            for j in jobs:
                emp_type = j.get("job_employment_type", "Full-time")
                if isinstance(emp_type, list):
                    emp_type = emp_type[0] if emp_type else "Full-time"
                skills = j.get("job_required_skills") or j.get("job_highlights", {}).get("Qualifications", [])
                if isinstance(skills, list):
                    skills = "|".join(str(s) for s in skills) if skills else ""
                elif isinstance(skills, str):
                    pass
                else:
                    skills = ""
                apply_url = _best_apply_url(j)
                all_jobs.append({
                    "job_id": j.get("job_id", f"api_{len(all_jobs)}"),
                    "title": j.get("job_title", ""),
                    "company": j.get("employer_name", ""),
                    "location": j.get("job_city") or j.get("job_country", ""),
                    "job_type": emp_type,
                    "posted_date": j.get("job_posted_at_datetime_utc", ""),
                    "description": j.get("job_description", ""),
                    "skills": skills,
                    "apply_link": apply_url,
                    "job_google_link": (j.get("job_google_link") or "").strip(),
                    "job_publisher": (j.get("job_publisher") or "").strip(),
                })
        except Exception as e:
            logger.warning("JSearch API error (page %d): %s", page + 1, e)
            break

    if not all_jobs:
        return None
    df = pd.DataFrame(all_jobs)
    logger.info("Fetched %d jobs from JSearch API", len(df))
    return df


def fetch_jobs_for_skills(
    skills: List[str],
    queries: Optional[List[str]] = None,
    date_posted: str = "month",
    num_pages_per_query: int = 2,
) -> pd.DataFrame:
    """
    Fetch jobs matching user skills.
    date_posted: "all", "today", "3days", "week", "month"
    """
    if not get_jsearch_api_key():
        return pd.DataFrame()

    if queries:
        search_queries = queries[:8]
    else:
        top = skills[:5] if skills else ["Data Scientist", "Software Engineer"]
        search_queries = [f"{s} jobs" for s in top]

    all_dfs: List[pd.DataFrame] = []
    for q in search_queries:
        df = fetch_jobs_from_api(q, num_pages=num_pages_per_query, date_posted=date_posted)
        if df is not None and not df.empty:
            all_dfs.append(df)

    if not all_dfs:
        return pd.DataFrame()
    combined = pd.concat(all_dfs, ignore_index=True)
    combined = combined.drop_duplicates(subset=["job_id"], keep="first")
    return combined


def fetch_recent_jobs_for_roles(
    role_titles: List[str],
    date_posted: str = "today",
    num_pages: int = 5,
) -> pd.DataFrame:
    """
    Fetch last 24h (today) jobs for top roles. Load as many as possible.
    """
    if not get_jsearch_api_key():
        return pd.DataFrame()

    all_dfs: List[pd.DataFrame] = []
    for role in role_titles[:3]:
        df = fetch_jobs_from_api(role, num_pages=num_pages, date_posted=date_posted)
        if df is not None and not df.empty:
            all_dfs.append(df)

    if not all_dfs:
        return pd.DataFrame()
    combined = pd.concat(all_dfs, ignore_index=True)
    combined = combined.drop_duplicates(subset=["job_id"], keep="first")
    return combined
