"""
Resume parsing: extract text from PDF/DOCX/TXT and parse skills, education, experience.
"""

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from utils.logging_config import get_logger

logger = get_logger("resume_parser")

try:
    import PyPDF2
    HAS_PDF = True
except ImportError:
    HAS_PDF = False

try:
    import docx
    HAS_DOCX = True
except ImportError:
    HAS_DOCX = False


def extract_text_from_file(file_path: Path) -> str:
    """Extract raw text from PDF, DOCX, or TXT file."""
    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix == ".txt":
        return path.read_text(encoding="utf-8", errors="replace")

    if suffix == ".pdf":
        if not HAS_PDF:
            raise ValueError("PyPDF2 required for PDF. Install: pip install PyPDF2")
        with open(path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            parts = [p.extract_text() or "" for p in reader.pages]
            return "\n".join(parts)

    if suffix == ".docx":
        if not HAS_DOCX:
            raise ValueError("python-docx required for DOCX. Install: pip install python-docx")
        doc = docx.Document(path)
        return "\n".join(p.text for p in doc.paragraphs)

    raise ValueError(f"Unsupported format: {suffix}. Use .pdf, .docx, or .txt")


# Degree patterns - comprehensive (handles B.S., MS, MBA, B.Tech, etc.)
DEGREE_PATTERN = re.compile(
    r"(?i)\b((?:bachelor|b\.?s\.?|b\.?a\.?|b\.?sc\.?|b\.?tech|b\.?e\.?|b\.?com|"
    r"master|m\.?s\.?|m\.?a\.?|m\.?sc\.?|m\.?tech|m\.?e\.?|m\.?com|"
    r"phd|ph\.?d\.?|doctorate|mba|m\.?b\.?a\.?|"
    r"associate|a\.?s\.?|a\.?a\.?|diploma|certificate|"
    r"b\.?arch|m\.?arch|llb|llm|b\.?pharm|m\.?pharm)"
    r"(?:\s+(?:of\s+)?(?:science|arts|engineering|technology|business|computer\s+science|mathematics|statistics|commerce|architecture))?"
    r"(?:\s+in\s+[\w\s\-]+)?)\b",
    re.IGNORECASE
)
YEAR_PATTERN = re.compile(r"\b(19|20)\d{2}(?:\s*[-–—to]+\s*(?:present|current|(?:19|20)\d{2}))?\b", re.IGNORECASE)
INSTITUTION_PATTERN = re.compile(
    r"(?i)([A-Za-z][A-Za-z\s&\.\-]*(?:University|College|Institute|School|Academy|Polytechnic|IIT|IIM|NIT)[A-Za-z\s&\.\-]*)"
)


def _clean(text: str) -> str:
    """Clean extracted text."""
    if not text:
        return ""
    return re.sub(r"\s+", " ", text.strip()).strip()


def _extract_year(text: str) -> Optional[str]:
    """Extract graduation year from text."""
    m = YEAR_PATTERN.search(text)
    return m.group(0) if m else None


def _extract_degree(text: str) -> str:
    """Extract degree from text."""
    m = DEGREE_PATTERN.search(text)
    if m:
        return _clean(m.group(0))
    return ""


def _extract_institution(text: str) -> str:
    """Extract institution name from text."""
    m = INSTITUTION_PATTERN.search(text)
    if m:
        return _clean(m.group(0))
    return ""


def parse_education(text: str) -> List[Dict[str, Any]]:
    """
    Extract education entries with crisp degree, institution, year.
    """
    entries: List[Dict[str, Any]] = []
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]

    in_education = False
    buffer: List[str] = []

    for i, line in enumerate(lines):
        if re.search(r"(?i)^(education|academic|qualifications)\s*[:\-]?\s*$", line):
            in_education = True
            continue
        if in_education:
            if re.search(r"(?i)^(experience|work|skills|projects|certifications|summary|employment)\s", line):
                break
            if line:
                buffer.append(line)

    if not buffer:
        for i, line in enumerate(lines):
            if DEGREE_PATTERN.search(line):
                buffer.append(line)
                if i + 1 < len(lines) and not DEGREE_PATTERN.search(lines[i + 1]):
                    buffer.append(lines[i + 1])

    for line in buffer:
        if not DEGREE_PATTERN.search(line):
            continue

        degree = _extract_degree(line)
        year = _extract_year(line)
        institution = _extract_institution(line)

        if not institution:
            parts = re.split(r"[|\-–—,;]", line)
            for p in parts:
                p = _clean(p)
                if not p or DEGREE_PATTERN.search(p) or YEAR_PATTERN.search(p):
                    continue
                if len(p) > 4 and "university" not in p.lower() and "college" not in p.lower():
                    institution = p
                    break
                elif len(p) > 4:
                    institution = p
                    break

        entries.append({
            "degree": degree or _clean(line[:50]),
            "institution": institution,
            "year": year or "",
            "raw": line,
        })

    return entries[:8]


def parse_experience(text: str) -> List[Dict[str, Any]]:
    """
    Extract experience entries with job_title, company, dates, bullets.
    """
    entries: List[Dict[str, Any]] = []
    lines = text.split("\n")

    in_section = False
    current_block: List[str] = []
    date_range = re.compile(r"(19|20)\d{2}\s*[-–—]\s*(?:present|current|(?:19|20)\d{2})", re.IGNORECASE)
    bullet = re.compile(r"^[\-\*•·]\s*")

    for i, line in enumerate(lines):
        raw = line.strip()
        if not raw:
            continue

        if re.search(r"(?i)^(experience|work\s+history|employment|professional\s+experience)\s*[:\-]?\s*$", raw):
            in_section = True
            current_block = []
            continue

        if in_section:
            if re.search(r"(?i)^(education|academic|skills|projects|certifications|summary|qualifications)\b", raw) and len(current_block) > 1:
                break
            current_block.append(raw)

    if not current_block:
        return entries

    i = 0
    while i < len(current_block):
        line = current_block[i]
        if "|" in line:
            parts = [p.strip() for p in line.split("|")]
            if len(parts) >= 2:
                title = parts[0]
                company = parts[1] if len(parts) > 1 else ""
                dates = parts[2] if len(parts) > 2 else ""
                bullets: List[str] = []
                j = i + 1
                while j < len(current_block) and (bullet.match(current_block[j]) or current_block[j].startswith("-")):
                    bullets.append(bullet.sub("", current_block[j]).strip())
                    j += 1
                entries.append({
                    "job_title": _clean(title),
                    "company": _clean(company),
                    "dates": _clean(dates),
                    "bullets": bullets[:5],
                    "text": " ".join(bullets)[:400] if bullets else line,
                })
                i = j
                continue

        if date_range.search(line) or re.search(r"(19|20)\d{2}", line):
            parts = re.split(r"\s+[-–—|]\s+", line, 2)
            if len(parts) >= 2:
                title = parts[0]
                rest = parts[1] if len(parts) > 1 else ""
                company = ""
                dates = ""
                if date_range.search(rest) or re.search(r"(19|20)\d{2}", rest):
                    sub = re.split(r"\s+[-–—|]\s+", rest, 1)
                    company = sub[0]
                    dates = sub[1] if len(sub) > 1 else rest
                else:
                    company = rest
                bullets = []
                j = i + 1
                while j < len(current_block) and (bullet.match(current_block[j]) or current_block[j].startswith("-")):
                    bullets.append(bullet.sub("", current_block[j]).strip())
                    j += 1
                entries.append({
                    "job_title": _clean(title),
                    "company": _clean(company),
                    "dates": _clean(dates),
                    "bullets": bullets[:5],
                    "text": " ".join(bullets)[:400] if bullets else line,
                })
                i = j
                continue

        i += 1

    if not entries:
        entries.append({
            "job_title": "",
            "company": "",
            "dates": "",
            "bullets": [],
            "text": " ".join(current_block[:8])[:500],
        })

    # Filter out entries that look like education or projects
    filtered = []
    for e in entries:
        title = (e.get("job_title") or "").lower()
        company = (e.get("company") or "").lower()
        # Exclude education
        if DEGREE_PATTERN.search(e.get("job_title", "")) and not e.get("bullets"):
            continue
        if e.get("job_title") and DEGREE_PATTERN.search(e["job_title"]) and "university" in company:
            continue
        # Exclude projects - do not include project entries in experience
        if re.search(r"\bproject\b", title) or re.search(r"\bproject\b", company):
            continue
        if re.search(r"^(personal|academic|capstone|side)\s", title):
            continue
        filtered.append(e)
    return filtered[:6] if filtered else entries[:6]


def parse_skills_from_resume(text: str) -> List[str]:
    """Extract skills from resume text."""
    from src.feature_engineering import extract_skills_from_text

    skills = extract_skills_from_text(text)
    skills_section = re.search(
        r"(?i)skills?\s*[:\-]?\s*([\s\S]*?)(?=\n\n|\n[A-Z][a-z]*\s*[:\-]|\Z)",
        text,
    )
    if skills_section:
        section_text = skills_section.group(1)
        for s in extract_skills_from_text(section_text):
            if s not in skills:
                skills.append(s)
    return skills[:100]


def parse_resume(file_path: Path) -> Dict[str, Any]:
    """Full resume parsing pipeline."""
    raw = extract_text_from_file(file_path)
    skills = parse_skills_from_resume(raw)
    education = parse_education(raw)
    experience = parse_experience(raw)

    from src.ats_keywords import extract_ats_keywords_from_resume
    ats_found, ats_missing = extract_ats_keywords_from_resume(skills, raw)

    return {
        "raw_text": raw,
        "skills": skills,
        "education": education,
        "experience": experience,
        "ats_keywords_found": ats_found,
        "ats_keywords_missing": ats_missing,
        "profile_summary": {
            "skills_count": len(skills),
            "education_count": len(education),
            "experience_count": len(experience),
            "text_length": len(raw),
            "ats_found_count": len(ats_found),
            "ats_missing_count": len(ats_missing),
        },
    }
