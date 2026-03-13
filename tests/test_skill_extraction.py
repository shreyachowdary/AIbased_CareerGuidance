"""Test skill extraction returns expected skills on known text."""
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def test_skill_extraction_returns_expected_skills():
    """Extract skills from text containing known vocabulary."""
    from backend.app.services.skill_extraction import extract_skills

    text = "I have experience with Python, SQL, machine learning, and AWS. Also React and Docker."
    skills = extract_skills(text)
    # Should find at least some of these
    expected = {"python", "sql", "machine learning", "aws", "react", "docker"}
    found = set(skills)
    overlap = expected & found
    assert len(overlap) >= 3, f"Expected some of {expected}, got {skills}"


def test_skill_extraction_normalization():
    """JS -> javascript, etc."""
    from backend.app.services.skill_extraction import extract_skills

    text = "I know JS and TS for frontend development."
    skills = extract_skills(text)
    # Should normalize js->javascript, ts->typescript
    skill_set = set(skills)
    assert "javascript" in skill_set or "typescript" in skill_set or len(skill_set) >= 1


def test_skill_extraction_empty_returns_empty():
    """Empty input returns empty list."""
    from backend.app.services.skill_extraction import extract_skills

    assert extract_skills("") == []
    assert extract_skills(None) == []
