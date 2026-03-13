"""
Load LinkedIn Job 2023 dataset (Kaggle: rajatraj0502/linkedin-job-2023).
Use: path = kagglehub.dataset_download("rajatraj0502/linkedin-job-2023")
"""

from pathlib import Path
from typing import Optional

import pandas as pd

from utils.logging_config import get_logger

logger = get_logger("linkedin_2023_loader")


def load_linkedin_2023(dataset_path: str | Path, sample_frac: Optional[float] = None) -> pd.DataFrame:
    """
    Load and merge LinkedIn 2023 dataset into unified job format.

    Args:
        dataset_path: Path from kagglehub.dataset_download("rajatraj0502/linkedin-job-2023")
        sample_frac: If set (e.g. 0.1), use a random sample for faster loading. None = full dataset.

    Returns:
        DataFrame with job_id, title, company, location, job_type, description, skills, posted_date
    """
    base = Path(dataset_path)
    if not base.exists():
        raise FileNotFoundError(f"LinkedIn 2023 dataset path not found: {base}")

    # Load core tables
    job_postings = pd.read_csv(base / "job_postings.csv", low_memory=False, on_bad_lines="skip")
    companies = pd.read_csv(base / "companies.csv", low_memory=False, on_bad_lines="skip")
    job_skills = pd.read_csv(base / "job_skills.csv", low_memory=False, on_bad_lines="skip")

    if sample_frac and sample_frac < 1.0:
        job_postings = job_postings.sample(frac=sample_frac, random_state=42)
        job_ids = set(job_postings["job_id"])
        job_skills = job_skills[job_skills["job_id"].isin(job_ids)]

    # Merge company name (column may be "name" or "company_name")
    company_name_col = "name" if "name" in companies.columns else "company_name"
    job_postings = job_postings.merge(
        companies[["company_id", company_name_col]].rename(columns={company_name_col: "company_name"}),
        on="company_id",
        how="left",
    )

    # Aggregate skills per job (skill_abr)
    skills_agg = (
        job_skills.groupby("job_id")["skill_abr"]
        .apply(lambda x: "|".join(str(s) for s in x.dropna().unique()))
        .reset_index()
    )
    skills_agg.columns = ["job_id", "skills"]

    job_postings = job_postings.merge(skills_agg, on="job_id", how="left")
    job_postings["skills"] = job_postings["skills"].fillna("")

    # Map to canonical columns
    df = pd.DataFrame()
    df["job_id"] = job_postings["job_id"]
    df["title"] = job_postings["title"].fillna("").astype(str)
    df["company"] = job_postings["company_name"].fillna("").astype(str)
    df["location"] = job_postings["location"].fillna("").astype(str)
    work_col = "formatted_work_type" if "formatted_work_type" in job_postings.columns else "work_type" if "work_type" in job_postings.columns else None
    df["job_type"] = job_postings[work_col].fillna("").astype(str) if work_col else pd.Series([""] * len(job_postings), index=job_postings.index)
    df["description"] = job_postings["description"].fillna("").astype(str)
    df["skills"] = job_postings["skills"]
    date_col = "listed_time" if "listed_time" in job_postings.columns else "posted_at" if "posted_at" in job_postings.columns else None
    df["posted_date"] = job_postings[date_col] if date_col else None

    logger.info("Loaded LinkedIn 2023: %d jobs from %s", len(df), base)
    return df
