"""
Skill-gap analysis: set difference between required (job) skills and user skills.
"""
from typing import List, Set


def compute_skill_gaps(required_skills: List[str], user_skills: List[str]) -> List[str]:
    """
    Returns sorted list of skills that are required but not in user profile.
    Uses normalized set difference for deterministic output.
    """
    req = set(s.lower().strip() for s in required_skills if s and isinstance(s, str))
    user = set(s.lower().strip() for s in user_skills if s and isinstance(s, str))
    missing: Set[str] = req - user
    return sorted(missing)


def matched_skills(required_skills: List[str], user_skills: List[str]) -> List[str]:
    """
    Returns sorted list of required skills that the user has.
    """
    req = set(s.lower().strip() for s in required_skills if s and isinstance(s, str))
    user = set(s.lower().strip() for s in user_skills if s and isinstance(s, str))
    return sorted(req & user)
