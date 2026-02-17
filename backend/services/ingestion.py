"""
Data ingestion: load job postings from CSV with expected schema.
Handles missing optional columns (company, skills).
"""
import csv
from pathlib import Path
from typing import Any, Dict, List, Optional

from backend.services.skill_extraction import extract_skills_from_text, parse_skills_column


def load_jobs_from_csv(csv_path: str | Path) -> List[Dict[str, Any]]:
    """
    Load jobs from CSV. Expects: job_id, title, description; optional: company, skills.
    If 'skills' column is missing or empty for a row, skills are extracted from description.
    Returns list of dicts with keys: job_id, title, company, description, skills (list).
    """
    path = Path(csv_path)
    if not path.exists():
        return []

    jobs: List[Dict[str, Any]] = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"job_id", "title", "description"}
        for row in reader:
            if not all(row.get(k) for k in required):
                continue
            job_id = row["job_id"].strip()
            title = row["title"].strip()
            description = (row.get("description") or "").strip()
            company = (row.get("company") or "").strip() or None

            skills_raw = row.get("skills") or ""
            if skills_raw.strip():
                skills = parse_skills_column(skills_raw)
            else:
                skills = extract_skills_from_text(description)

            jobs.append({
                "job_id": job_id,
                "title": title,
                "company": company,
                "description": description,
                "skills": skills,
            })
    return jobs
