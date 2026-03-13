#!/usr/bin/env python3
"""
Preprocess raw LinkedIn job data.
Auto-detect columns, normalize text, output standardized jobs.csv.
"""
import os
import re
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = Path(os.environ.get("RAW_DIR", PROJECT_ROOT / "data" / "raw"))
PROCESSED_DIR = Path(os.environ.get("PROCESSED_DIR", PROJECT_ROOT / "data" / "processed"))
OUTPUT_FILE = PROCESSED_DIR / "jobs.csv"

# Column mapping: our standard -> possible raw column names
COLUMN_MAPPING = {
    "title": ["title", "job_title", "position", "job", "Job Title", "job_titles"],
    "description": ["description", "job_description", "job_desc", "Job Description", "descriptions"],
    "company": ["company", "company_name", "Company", "company_names", "Company Name"],
    "location": ["location", "city", "state", "Location", "locations", "job_location", "Job Location"],
}

# Config
MAX_JOBS_SAMPLE = int(os.environ.get("MAX_JOBS_SAMPLE", "20000"))
MIN_DESCRIPTION_LEN = int(os.environ.get("MIN_DESCRIPTION_LEN", "50"))


def find_best_csv() -> Path | None:
    """Find the best CSV in data/raw (largest with job-like content)."""
    if not RAW_DIR.exists():
        return None
    csv_files = list(RAW_DIR.glob("*.csv"))
    if not csv_files:
        return None
    # Prefer largest
    return max(csv_files, key=lambda p: p.stat().st_size)


def map_columns(df: pd.DataFrame) -> dict[str, str]:
    """Map raw columns to standard names. Returns {standard: raw}."""
    result = {}
    cols_lower = {c.lower(): c for c in df.columns}
    cols_exact = {c: c for c in df.columns}

    for standard, candidates in COLUMN_MAPPING.items():
        for cand in candidates:
            cand_lower = cand.lower()
            if cand_lower in cols_lower:
                result[standard] = cols_lower[cand_lower]
                break
            if cand in cols_exact:
                result[standard] = cand
                break
    return result


def clean_text(text: str) -> str:
    """Remove HTML, lowercase, normalize whitespace."""
    if pd.isna(text) or not isinstance(text, str):
        return ""
    # Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)
    # Lowercase
    text = text.lower()
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def main():
    csv_path = find_best_csv()
    if not csv_path:
        print(f"ERROR: No CSV found in {RAW_DIR}")
        print("Run scripts/download_kaggle.py first.")
        sys.exit(1)

    print(f"Reading: {csv_path}")
    df = pd.read_csv(csv_path, low_memory=False, nrows=None)
    before_rows = len(df)
    print(f"Before: {before_rows:,} rows, columns: {list(df.columns)}")

    mapping = map_columns(df)
    if "title" not in mapping or "description" not in mapping:
        print("ERROR: Could not find required columns (title, description).")
        print("Available columns:", list(df.columns))
        print("Mapping found:", mapping)
        sys.exit(1)

    print(f"Column mapping: {mapping}")

    # Build standardized dataframe
    out = pd.DataFrame()
    out["title"] = df[mapping["title"]].fillna("").astype(str).apply(clean_text)
    out["description"] = df[mapping["description"]].fillna("").astype(str).apply(clean_text)
    out["company"] = df[mapping.get("company", mapping["title"])].fillna("").astype(str).apply(clean_text)
    out["location"] = df[mapping.get("location", mapping["title"])].fillna("").astype(str).apply(clean_text)

    # Drop null/very short descriptions
    out = out[out["description"].str.len() >= MIN_DESCRIPTION_LEN]
    out = out[out["title"].str.len() > 0]

    # Remove duplicates (title + company + first 200 chars of description)
    out["_dup_key"] = out["title"] + "|" + out["company"] + "|" + out["description"].str[:200]
    out = out.drop_duplicates(subset=["_dup_key"]).drop(columns=["_dup_key"])

    # Sample if huge
    if len(out) >= 100_000:
        out = out.sample(n=MAX_JOBS_SAMPLE, random_state=42).reset_index(drop=True)
        print(f"Sampled to {MAX_JOBS_SAMPLE:,} rows (dataset was >= 100k)")

    # Add job_id
    out.insert(0, "job_id", [f"job_{i}" for i in range(len(out))])

    after_rows = len(out)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_FILE, index=False)

    print(f"After: {after_rows:,} rows")
    print(f"Output: {OUTPUT_FILE}")
    print("\nSample titles:")
    for t in out["title"].head(5).tolist():
        print(f"  - {t[:60]}..." if len(t) > 60 else f"  - {t}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
