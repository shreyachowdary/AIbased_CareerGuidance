"""
Feature engineering: build text representations and extract skills for matching.
Uses curated skills column (mapped) + prominent-only extraction from descriptions.
"""

import re
from typing import List, Set

import pandas as pd

from src.skill_curation import extract_prominent_skills, is_prominent_skill, map_abbrev
from utils.logging_config import get_logger

logger = get_logger("feature_engineering")

# For resume parsing only (extract_skills_from_text)
STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "as", "is", "was", "are", "were", "been",
    "be", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "must", "can", "need",
    "this", "that", "these", "those", "it", "its", "they", "them", "their",
    "we", "our", "you", "your", "he", "she", "him", "her", "his", "i", "me", "my",
    "all", "each", "every", "some", "many", "most", "such", "both", "either", "neither",
    "other", "others", "another", "any", "same", "own", "team", "work", "experience",
    "required", "preferred", "skills", "ability", "knowledge", "etc",
}


def is_valid_skill(s: str) -> bool:
    """True if string is a real skill (not common word, not empty, not numeric)."""
    if not s or not isinstance(s, str):
        return False
    t = s.strip().lower()
    if len(t) < 2:
        return False
    if t in STOPWORDS:
        return False
    if sum(c.isdigit() for c in t) > len(t) // 2:
        return False
    return True


def extract_skills_from_text(text: str, min_len: int = 4, max_len: int = 40) -> List[str]:
    """
    Extract skills from free text (used for resume parsing only).
    """
    if not text or not isinstance(text, str):
        return []
    tokens = re.findall(r"[a-zA-Z0-9]+(?:[-][a-zA-Z0-9]+)*", text)
    seen: Set[str] = set()
    skills = []
    for t in tokens:
        t_lower = t.lower()
        if t_lower in STOPWORDS:
            continue
        if min_len <= len(t) <= max_len and t_lower not in seen:
            seen.add(t_lower)
            skills.append(t)
    return skills


def parse_skills_column(series: pd.Series) -> pd.Series:
    """Parse skills column (pipe/comma/separated). Returns Series of skill lists."""
    def _parse(s):
        if pd.isna(s) or str(s).strip() == "":
            return []
        s = str(s)
        for sep in ["|", ";", ",", "\n", "•"]:
            if sep in s:
                return [x.strip() for x in s.split(sep) if x.strip()]
        return [s.strip()] if s.strip() else []

    return series.apply(_parse)


def build_skill_set_per_job(df: pd.DataFrame) -> pd.Series:
    """
    Build skills per job: mapped abbreviations + prominent-only extraction from description.
    Only AI, AWS, Python, etc. — no rubbish words.
    """
    skills_from_col = parse_skills_column(df["skills"]) if "skills" in df.columns else pd.Series([[]] * len(df))
    desc_col = df["description"] if "description" in df.columns else pd.Series([""] * len(df))

    def _build(raw, desc):
        combined: Set[str] = set()
        for ab in (raw or []):
            if isinstance(ab, str) and ab.strip():
                mapped = map_abbrev(ab)
                # Only add if we have a mapping (avoid unknown abbrevs like PRJM if not in map)
                if mapped and mapped != ab:
                    combined.add(mapped)
        for s in extract_prominent_skills(str(desc or "")):
            combined.add(s)
        return list(combined)

    return pd.Series(
        [_build(raw, desc) for raw, desc in zip(skills_from_col, desc_col)],
        index=df.index,
    )


def get_global_skill_vocabulary(df: pd.DataFrame) -> Set[str]:
    """Build global vocabulary of skills across all jobs."""
    skill_col = build_skill_set_per_job(df)
    return set().union(*(set(s) for s in skill_col))
