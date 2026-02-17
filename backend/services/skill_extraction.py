"""
Skill extraction from text: keyword-based + optional skills dictionary.
Normalizes and deduplicates skills for matching.
"""
import re
from typing import List, Set

# Common tech/soft skills dictionary for keyword-based extraction (subset for MVP)
SKILLS_DICTIONARY: Set[str] = {
    "python", "javascript", "java", "sql", "react", "node.js", "git", "rest", "api", "apis",
    "aws", "docker", "kubernetes", "linux", "mongodb", "postgresql", "mysql", "typescript",
    "html", "css", "scikit-learn", "pandas", "numpy", "tensorflow", "pytorch", "nlp", "ml",
    "machine learning", "data science", "etl", "spark", "airflow", "dbt", "terraform",
    "ci/cd", "selenium", "figma", "jira", "agile", "swift", "kotlin", "react native",
    "fastapi", "django", "flask", "microservices", "testing", "security", "spacy",
    "looker", "tableau", "power bi", "data modeling", "statistics", "communication",
}


def normalize_skills(skill_list: List[str]) -> List[str]:
    """
    Normalize skill strings: lowercase, strip, collapse spaces.
    Returns sorted unique list for deterministic output.
    """
    if not skill_list:
        return []
    seen: Set[str] = set()
    result: List[str] = []
    for s in skill_list:
        t = " ".join(s.lower().strip().split())
        if t and t not in seen:
            seen.add(t)
            result.append(t)
    return sorted(result)


def _tokenize_description(text: str) -> List[str]:
    """Tokenize by non-alphanumeric, keep words and hyphenated terms."""
    if not text or not isinstance(text, str):
        return []
    # Split on non-alphanumeric but keep sequences like "Node.js" / "CI/CD"
    tokens = re.findall(r"[A-Za-z0-9]+(?:\.?[A-Za-z0-9/]+)*", text)
    return [t.lower() for t in tokens if len(t) > 1]


def extract_skills_from_text(text: str) -> List[str]:
    """
    Extract skills from free text using skills dictionary matching.
    Returns normalized, unique list of matched skills.
    """
    if not text or not isinstance(text, str):
        return []
    tokens = _tokenize_description(text)
    token_set = set(tokens)
    # Also check multi-word phrases (e.g. "machine learning")
    text_lower = text.lower()
    found: Set[str] = set()
    for skill in SKILLS_DICTIONARY:
        if skill in text_lower or skill in token_set:
            found.add(skill)
    # Add single tokens that are in our dictionary
    for t in token_set:
        if t in SKILLS_DICTIONARY:
            found.add(t)
    return normalize_skills(list(found))


def parse_skills_column(skills_str: str) -> List[str]:
    """
    Parse comma-separated skills string from CSV.
    Returns normalized list.
    """
    if not skills_str or not isinstance(skills_str, str):
        return []
    parts = [p.strip() for p in skills_str.split(",") if p.strip()]
    return normalize_skills(parts)
