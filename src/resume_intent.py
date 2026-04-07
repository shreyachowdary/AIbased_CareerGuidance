"""
Infer job-search intent from resume text so matching favors the right track (e.g. SWE vs QA).
"""

import re
from typing import Any, Dict, List, Optional, Set

# Strong signals someone is a software engineer / developer track (not primary QA)
_SWE_LINE = re.compile(
    r"(?i)(software\s*engineer|software\s*developer|full[\s\-]*stack|front[\s\-]*end|back[\s\-]*end|"
    r"web\s*developer|application\s*developer|sde\b|java\s*developer|python\s*developer|"
    r"\.net\s*developer|mobile\s*developer|devops\s*engineer|platform\s*engineer|"
    r"microservices|rest\s*api|spring\s+boot|react\.?js|node\.?js)"
)

_QA_PRIMARY_LINE = re.compile(
    r"(?i)(^\s*qa\s+engineer|quality\s+assurance\s+engineer|^test\s+engineer\b|sdet\b|"
    r"only\s+qa|manual\s+testing\s+only)"
)

# Skills that imply SWE over generic QA queries
_TECH_SKILL_PRIOR = frozenset(
    {
        "java", "python", "javascript", "typescript", "c++", "c#", "go", "rust", "kotlin", "swift",
        "react", "angular", "vue", "node", "django", "flask", "spring", "kubernetes", "docker",
        "aws", "azure", "gcp", "sql", "postgresql", "mongodb", "redis", "graphql", "kafka",
        "microservices", "ci/cd", "terraform",
    }
)

# Skills often first in list but poor as sole search query for role fit
_WEAK_QUERY_SKILLS = frozenset(
    {
        "jira", "confluence", "agile", "scrum", "excel", "word", "communication", "leadership",
        "teamwork", "english",
    }
)


def resume_signals_swe_primary(raw_text: str, skills: List[str]) -> bool:
    """True if resume reads as software engineering / development as primary path."""
    if not raw_text:
        raw_text = ""
    if _QA_PRIMARY_LINE.search(raw_text) and not _SWE_LINE.search(raw_text):
        return False
    if _SWE_LINE.search(raw_text):
        return True
    sk = {s.lower().strip() for s in skills if s}
    if len(sk & _TECH_SKILL_PRIOR) >= 2:
        return True
    if sk & _TECH_SKILL_PRIOR and re.search(r"(?i)\b(developer|engineer|programming|deployment)\b", raw_text):
        return True
    return False


def infer_job_search_queries(
    raw_text: str,
    skills: List[str],
    profile: Optional[Dict[str, Any]] = None,
) -> List[str]:
    """
    Build search queries for job APIs — prioritize role + tech stack, not brittle single skills.
    """
    profile = profile or {}
    career = (profile.get("career_goal") or "").strip()
    queries: List[str] = []
    seen: Set[str] = set()

    def add(q: str) -> None:
        q = " ".join(q.split())
        if len(q) < 3:
            return
        key = q.lower()
        if key in seen:
            return
        seen.add(key)
        queries.append(q)

    if career:
        add(career if career.lower().endswith("jobs") else f"{career} jobs")

    swe = resume_signals_swe_primary(raw_text or "", skills or [])
    if swe:
        for q in (
            "Software Engineer",
            "Software Developer",
            "Backend Developer",
            "Full Stack Developer",
            "Java Developer",
            "Python Developer",
        ):
            add(q)

    ranked_skills = sorted(
        skills or [],
        key=lambda s: (0 if str(s).lower().strip() in _TECH_SKILL_PRIOR else 1, len(str(s))),
    )
    for s in ranked_skills:
        if len(queries) >= 8:
            break
        sl = str(s).lower().strip()
        if not sl or sl in _WEAK_QUERY_SKILLS:
            continue
        if sl in _TECH_SKILL_PRIOR:
            add(f"{str(s).strip()} developer")
        else:
            add(f"{str(s).strip()} jobs")

    if not queries:
        add("Data Scientist")
        add("Software Engineer")
    return queries[:8]


def build_matching_resume_text(data: Dict[str, Any], profile: Optional[Dict[str, Any]] = None) -> str:
    """Combine resume fields for TF‑IDF / hybrid matching (no forced role list)."""
    profile = profile or {}
    raw = data.get("raw_text") or ""
    skills_list = data.get("skills") or []
    skills = " ".join(str(s) for s in skills_list)
    cg = (profile.get("career_goal") or "").strip()
    parts = [raw, skills]
    if cg:
        parts.extend([cg, cg])
    return " \n ".join(p for p in parts if p)


def title_role_alignment_score(job_title: str, raw_text_lower: str, swe_primary: bool = False) -> float:
    """0..1: title fit vs resume intent (uses swe_primary to avoid SWE↔QA mismatches)."""
    t = (job_title or "").strip()
    if not t:
        return 0.0
    raw = raw_text_lower or ""

    qa_heavy = re.search(
        r"(?i)(^\s*qa\s|quality\s+assurance|^test\s+engineer\b|sdet\b|manual\s+tester)",
        t,
    )
    swe_in_title = re.search(
        r"(?i)(software\s+engineer|software\s+developer|full[\s\-]*stack|backend|frontend|"
        r"\bdeveloper\b|devops|platform\s+engineer|sde\b)",
        t,
    )

    if swe_primary and qa_heavy and not swe_in_title:
        return 0.2

    ds_de_ml = re.search(
        r"(?i)(data\s+scientist|data\s+engineer|machine\s+learning|ml\s+engineer|"
        r"machine\s+learning\s+engineer|ai\s+engineer|research\s+scientist)",
        t,
    )
    if ds_de_ml:
        if re.search(r"(?i)(data\s+scientist|data\s+engineer|machine\s+learning|research|ml\b)", raw):
            return 1.0
        return 0.85

    if swe_in_title:
        return 1.0 if swe_primary else 0.75

    if re.search(r"(?i)(business\s+analyst|product\s+manager|program\s+manager)", t):
        return 0.75

    if re.search(r"(?i)(analyst|coordinator|specialist|consultant|\bengineer\b)", t):
        return 0.55

    return 0.35
