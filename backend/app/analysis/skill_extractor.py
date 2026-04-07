"""Resume and job skill extraction using the same curated pipeline as Streamlit."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from src.feature_engineering import extract_skills_from_text
from src.resume_parser import parse_resume


def skills_from_resume_file(path: Path) -> Dict[str, Any]:
    """Parse resume file and return skills + raw payload."""
    data = parse_resume(path)
    return {
        "skills": list(data.get("skills") or []),
        "raw_text": data.get("raw_text") or "",
        "resume_payload": data,
    }


def skills_from_free_text(text: str) -> List[str]:
    """Lightweight token extraction (same helper as resume fallback)."""
    return extract_skills_from_text(text or "")
