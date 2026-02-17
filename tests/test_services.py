"""
Basic unit tests: skill extraction, gap analysis, determinism.
Run from project root: pytest tests/ -v
"""
import sys
from pathlib import Path

# Add project root for imports
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.services.skill_extraction import (
    extract_skills_from_text,
    normalize_skills,
    parse_skills_column,
)
from backend.services.gap_analysis import compute_skill_gaps, matched_skills


def test_normalize_skills() -> None:
    assert normalize_skills(["Python", "python", "  SQL  "]) == ["python", "sql"]
    assert normalize_skills([]) == []
    assert normalize_skills(["A", "B", "A"]) == ["a", "b"]


def test_parse_skills_column() -> None:
    assert parse_skills_column("Python, SQL, Git") == ["git", "python", "sql"]
    assert parse_skills_column("") == []
    assert parse_skills_column("  one  ,  two  ") == ["one", "two"]


def test_extract_skills_from_text() -> None:
    text = "We use Python, JavaScript and REST APIs. Experience with Git required."
    skills = extract_skills_from_text(text)
    assert "python" in skills
    assert "javascript" in skills
    assert "rest" in skills or "api" in skills
    assert "git" in skills
    # Deterministic: sorted
    assert skills == sorted(skills)


def test_compute_skill_gaps() -> None:
    req = ["python", "sql", "docker"]
    user = ["python", "sql"]
    assert compute_skill_gaps(req, user) == ["docker"]
    assert compute_skill_gaps([], user) == []
    assert compute_skill_gaps(req, ["python", "sql", "docker"]) == []


def test_matched_skills() -> None:
    req = ["python", "sql", "docker"]
    user = ["python", "sql", "java"]
    assert matched_skills(req, user) == ["python", "sql"]
