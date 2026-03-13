"""
Configuration settings for the AI Career Guidance System.
"""

from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
UPLOADS_DIR = PROJECT_ROOT / "uploads"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

# Ensure directories exist
for d in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, ARTIFACTS_DIR, UPLOADS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Dataset configuration - flexible column mapping for various Kaggle LinkedIn datasets
# Supports: 1.3M Linkedin Jobs & Skills (2024), LinkedIn Job Postings (2023-2024), etc.
DATASET_COLUMN_MAPPING = {
    "job_id": ["job_id", "id", "job-id", "Job ID"],
    "title": ["title", "job_title", "Job Title", "position"],
    "company": ["company", "company_name", "Company Name", "company_name"],
    "location": ["location", "job_location", "Location", "job_location"],
    "job_type": ["job_type", "employment_type", "Job Type", "work_type"],
    "posted_date": ["posted_date", "posted_at", "Posted Date", "posted_on", "date_posted"],
    "description": ["description", "job_description", "Description", "job_desc"],
    "skills": ["skills", "job_skills", "Skills", "required_skills", "skills_list"],
}

# Fallback: if dataset has different column names, we detect common patterns
DEFAULT_JOB_CSV = "linkedin_job_postings.csv"
CLEANED_CSV = "cleaned_jobs.csv"
CLEANED_PARQUET = "cleaned_jobs.parquet"

# TF-IDF / matching
TFIDF_VECTORIZER_PKL = "tfidf_vectorizer.pkl"
JOB_EMBEDDINGS_PKL = "job_embeddings.pkl"
JOB_METADATA_PKL = "job_metadata.pkl"

# Resume parsing
SUPPORTED_RESUME_EXTENSIONS = {".pdf", ".docx", ".txt"}
MAX_RESUME_SIZE_MB = 5

# Matching
TOP_N_MATCHES = 25
MIN_SKILL_OVERLAP = 0.1

# EDA / viz
PLOT_STYLE = "seaborn-v0_8-whitegrid"
FIGURE_DPI = 120
