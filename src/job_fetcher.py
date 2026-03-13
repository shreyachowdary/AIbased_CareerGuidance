"""
Fetch real job listings from JSearch API (aggregates LinkedIn, Indeed, Glassdoor, etc.).
Requires JSEARCH_API_KEY in environment. Free tier: 500 requests/month.
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

JSEARCH_API_KEY = os.environ.get("JSEARCH_API_KEY", "")
JSEARCH_HOST = "jsearch.p.rapidapi.com"
JSEARCH_URL = "https://jsearch.p.rapidapi.com/search"


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
    if not JSEARCH_API_KEY:
        return None

    all_jobs: List[Dict[str, Any]] = []
    for page in range(num_pages):
        try:
            resp = requests.get(
                JSEARCH_URL,
                headers={
                    "X-RapidAPI-Key": JSEARCH_API_KEY,
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
                all_jobs.append({
                    "job_id": j.get("job_id", f"api_{len(all_jobs)}"),
                    "title": j.get("job_title", ""),
                    "company": j.get("employer_name", ""),
                    "location": j.get("job_city") or j.get("job_country", ""),
                    "job_type": emp_type,
                    "posted_date": j.get("job_posted_at_datetime_utc", ""),
                    "description": j.get("job_description", ""),
                    "skills": skills,
                    "apply_link": j.get("job_apply_link", ""),
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
    if not JSEARCH_API_KEY:
        return pd.DataFrame()

    if queries:
        search_queries = queries[:5]
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
    if not JSEARCH_API_KEY:
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
