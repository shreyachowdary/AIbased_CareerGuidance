# Services: ingestion, skill_extraction, recommend, gap_analysis
from backend.services.ingestion import load_jobs_from_csv
from backend.services.skill_extraction import extract_skills_from_text, normalize_skills
from backend.services.gap_analysis import compute_skill_gaps
from backend.services.recommend import RecommendationService

__all__ = [
    "load_jobs_from_csv",
    "extract_skills_from_text",
    "normalize_skills",
    "compute_skill_gaps",
    "RecommendationService",
]
