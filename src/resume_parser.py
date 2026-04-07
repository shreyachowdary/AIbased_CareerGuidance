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


def _normalize_education_line_for_gpa(line: str) -> Tuple[str, str]:
    """
    Fix PDF glues like 'SystemsGPA' / 'EngineeringGPA', extract GPA into its own field.
    Returns (line_without_gpa, gpa_display or "").
    """
    s = line.strip()
    if not s:
        return "", ""
    # Insert missing space before GPA / CGPA when glued to a word
    s = re.sub(r"(?i)([a-z219])(gpa)\b", r"\1 \2", s)
    s = re.sub(r"(?i)(systems|engineering|science|analytics|intelligence|technology|statistics)\s*(gpa)\b", r"\1 \2", s)

    gpa_display = ""
    # GPA: 3.7/4.0, GPA 3.7, CGPA: 8.5/10, etc.
    gpa_pat = re.compile(
        r"(?i)(?:cgpa|gpa)\s*[:.]?\s*(\d+(?:\.\d+)?)(?:\s*/\s*\d+(?:\.\d+)?)?"
    )
    m = gpa_pat.search(s)
    if m:
        gpa_display = m.group(0).strip()
        s = (s[: m.start()] + " " + s[m.end() :]).strip(" ,;·|-")
    alt = re.search(r"(?i)(?<!\w)(\d+(?:\.\d+)?)\s*/\s*4\.0\b", s)
    if alt and not gpa_display:
        gpa_display = alt.group(0).strip()
        s = (s[: alt.start()] + " " + s[alt.end() :]).strip(" ,;·|-")

    s = re.sub(r"\s+", " ", s).strip()
    return s, gpa_display


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

    # Attach "GPA …" line that PDF put on the next line under the degree
    merged_buf: List[str] = []
    j = 0
    while j < len(buffer):
        ln = buffer[j]
        if j + 1 < len(buffer):
            nxt = buffer[j + 1].strip()
            if DEGREE_PATTERN.search(ln) and (
                re.match(r"(?i)^(gpa|cgpa)\b[\s:.]*[\d./ ]+$", nxt)
                or re.match(r"^[\d.]+\s*/\s*[\d.]+\s*$", nxt)
            ):
                merged_buf.append(ln + " " + nxt)
                j += 2
                continue
        merged_buf.append(ln)
        j += 1
    buffer = merged_buf

    for line in buffer:
        if not DEGREE_PATTERN.search(line):
            continue

        line_n, gpa_val = _normalize_education_line_for_gpa(line)
        degree = _extract_degree(line_n) or _extract_degree(line)
        year = _extract_year(line_n) or _extract_year(line)
        institution = _extract_institution(line_n)

        if not institution:
            parts = re.split(r"[|\-–—,;]", line_n)
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

        # Degree text without glued GPA for display
        deg_display = degree or _clean(line_n[:120]) or _clean(line[:120])

        entries.append({
            "degree": deg_display,
            "institution": institution,
            "year": year or "",
            "gpa": gpa_val,
            "raw": line,
        })

    return entries[:8]


_RE_EXPERIENCE_HEAD = re.compile(
    r"(?i)^(experience|work\s*experience|work\s+history|employment(?:\s*history)?|"
    r"professional\s+experience|relevant\s+experience|career\s+history|employment\s+experience)\s*[:\-]?\s*$"
)
_RE_PROJECTS_HEAD = re.compile(
    r"(?i)^(projects?|academic\s+projects?|personal\s+projects?|key\s+projects?|"
    r"notable\s+projects?|selected\s+projects?|technical\s+projects?|relevant\s+projects?|"
    r"engineering\s+projects?|project\s+experience|capstone(?:\s+projects?)?|"
    r"research\s+projects?)\s*[:\-]?\s*$"
)
_RE_STOPS_EXP = re.compile(
    r"(?i)^(education|academic|qualifications|skills?|projects?|certifications?|summary|objective|profile|"
    r"leadership|achievements?|awards?|honors?|volunteer|publications?|references?)\b"
)


def _collect_section_body(lines: List[str], start_index: int, stop_at_projects_too: bool) -> Tuple[List[str], int]:
    """
    Collect non-empty stripped lines until a line that looks like a section header.
    Returns (body_lines, next_line_index).
    """
    body: List[str] = []
    i = start_index
    while i < len(lines):
        raw = lines[i]
        stripped = raw.strip()
        i += 1
        if not stripped:
            continue
        if _RE_EXPERIENCE_HEAD.match(stripped):
            break
        if stop_at_projects_too and _RE_PROJECTS_HEAD.match(stripped):
            break
        if _RE_STOPS_EXP.match(stripped):
            break
        body.append(stripped)
    return body, i


def _slice_work_experience_text(text: str) -> List[str]:
    """Lines belonging only to work experience headers (not projects section)."""
    lines = text.split("\n")
    collected: List[str] = []
    i = 0
    while i < len(lines):
        raw = lines[i].strip()
        if _RE_EXPERIENCE_HEAD.match(raw):
            i += 1
            chunk, i = _collect_section_body(lines, i, stop_at_projects_too=True)
            collected.extend(chunk)
            continue
        i += 1
    return collected


def _slice_projects_text(text: str) -> List[str]:
    """Lines belonging only to project section(s)."""
    lines = text.split("\n")
    collected: List[str] = []
    i = 0
    while i < len(lines):
        raw = lines[i].strip()
        if _RE_PROJECTS_HEAD.match(raw):
            i += 1
            chunk, i = _collect_section_body(lines, i, stop_at_projects_too=False)
            collected.extend(chunk)
            continue
        i += 1
    return collected


def _unstick_dates_and_words(line: str) -> str:
    """Insert spaces when PDF extraction jams words against dates (e.g. Florida09/2024)."""
    s = line.strip()
    if not s:
        return s
    s = re.sub(r"([a-zA-Z\.\)])(\d{2}/\d{4})", r"\1 \2", s)
    s = re.sub(r"(\d{2}/\d{4})([A-Za-z])", r"\1 \2", s)
    s = re.sub(r"(\d{4})([A-Za-z][a-z]+)", r"\1 \2", s)
    # Avoid variable-width lookbehind (Python requires fixed width): match date + "at Present" explicitly.
    s = re.sub(r"(?i)(\d{2}/\d{4})\s+at\s+(Present|Current)\b", r"\1 – \2", s)
    s = re.sub(r"(?i)(\d{4})\s+at\s+(Present|Current)\b", r"\1 – \2", s)
    return re.sub(r"\s+", " ", s).strip()


def _is_education_bleed_in_experience(header_joined: str) -> bool:
    """Skip rows that look like education (school + dates) under Experience."""
    h = header_joined.lower()
    if any(
        k in h
        for k in (
            " intern",
            "intern,",
            "internship",
            "research assistant",
            "teaching assistant",
            "graduate assistant",
            "postdoctoral",
        )
    ):
        return False
    inst_hit = any(
        k in h
        for k in (
            "university",
            "college",
            "institute of",
            "school of",
            "memorial institute",
            "iit ",
            "iim ",
            "nit ",
        )
    )
    if not inst_hit:
        return False
    role_kw = (
        "engineer",
        "developer",
        "analyst",
        "scientist",
        "consultant",
        "intern",
        "manager",
        "director",
        "specialist",
        "architect",
        "researcher",
        "fellow",
    )
    if any(k in h for k in role_kw):
        return False
    if re.search(r"\d{2}/\d{4}", h) or re.search(r"\d{4}\s*[-–—]", h):
        return True
    return False


def _line_looks_like_project_stack_line(line: str) -> bool:
    """Tech stack lines: comma-separated tools (1+ commas) or known stack keywords with one comma."""
    s = line.strip()
    if not s or len(s) >= 180:
        return False
    if s.count(",") >= 2:
        return True
    if s.count(",") == 1 and re.search(
        r"(?i)\b(python|java|javascript|typescript|react|angular|vue|node|go|rust|spark|kafka|django|flask|fastapi|"
        r"tensorflow|keras|pytorch|pandas|numpy|scikit|sklearn|opencv|docker|kubernetes|aws|gcp|azure|sql|mongodb|"
        r"postgres|redis|grpc|git|c\+\+|c\#|\.net)\b",
        s,
    ):
        return True
    return False


def _line_looks_like_project_skills_summary(line: str) -> bool:
    """Resume row like 'Computer Vision — Deep Learning' (not a long product title)."""
    s = line.strip()
    if not s:
        return False
    m = re.match(r"^(.+?)\s*(?:[—–]|\s-\s)\s*(.+)$", s)
    if not m:
        return False
    left, right = m.group(1).strip(), m.group(2).strip()
    if len(right) > 38 or len(left) + len(right) > 72:
        return False
    return bool(
        re.search(
            r"(?i)learning|tensorflow|pytorch|vision|nlp|kubernetes|\bml\b|deep\s+learning|"
            r"opencv|cnn|gpu|neural",
            s,
        )
    )


def _group_lines_into_experience_roles(
    lines: List[str],
    *,
    is_work_section: bool = True,
) -> List[List[str]]:
    """
    Preserve resume structure: header lines as written, then bullets, until the next header.
    Projects often use title + tech-stack lines with no bullets; split when a new title follows stack/dates.
    """
    groups: List[List[str]] = []
    cur: List[str] = []
    bullet = re.compile(r"^[\-\*•·]\s*")

    for line in lines:
        ln = _unstick_dates_and_words(line)
        if not ln:
            continue
        if bullet.match(ln):
            if cur:
                cur.append(ln)
            elif groups:
                groups[-1].append(ln)
            else:
                cur = [ln]
            continue

        if not is_work_section and cur:
            has_bullets = any(bullet.match(x) for x in cur)
            if not has_bullets:
                last = cur[-1]
                last_is_aux = _line_looks_like_project_stack_line(last) or bool(
                    _WORK_DATE_RANGE.search(last) or _MONTH_YEAR_RANGE.search(last)
                )
                if last_is_aux and len(cur) >= 2:
                    modest = 8 <= len(ln) <= 120
                    few_commas = ln.count(",") <= 1
                    starts_title = not ln[:1].islower() if ln else False
                    not_date_ln = not (
                        _WORK_DATE_RANGE.search(ln) or _MONTH_YEAR_RANGE.search(ln)
                    )
                    if modest and few_commas and starts_title and not_date_ln:
                        groups.append(cur)
                        cur = [ln]
                        continue
                if len(cur) == 1:
                    prev = cur[0]
                    if _line_looks_like_project_stack_line(ln):
                        cur.append(ln)
                        continue
                    if (
                        8 <= len(prev) <= 100
                        and prev.count(",") <= 1
                        and 8 <= len(ln) <= 100
                        and ln.count(",") <= 1
                        and not (
                            _WORK_DATE_RANGE.search(prev)
                            or _MONTH_YEAR_RANGE.search(prev)
                        )
                        and not (
                            _WORK_DATE_RANGE.search(ln) or _MONTH_YEAR_RANGE.search(ln)
                        )
                        and prev[:1].isupper()
                        and ln[:1].isupper()
                    ):
                        groups.append(cur)
                        cur = [ln]
                        continue

        if cur and any(bullet.match(x) for x in cur):
            groups.append(cur)
            cur = [ln]
        elif not cur:
            cur = [ln]
        else:
            cur.append(ln)
    if cur:
        groups.append(cur)
    return groups


def _parse_experience_block_to_entries(
    current_block: List[str],
    *,
    is_work_section: bool,
) -> List[Dict[str, Any]]:
    """Turn experience section lines into entries; keep header text as on the resume."""
    entries: List[Dict[str, Any]] = []
    if not current_block:
        return entries

    bullet_re = re.compile(r"^[\-\*•·]\s*")
    groups = _group_lines_into_experience_roles(current_block, is_work_section=is_work_section)

    for group in groups:
        header_lines = [ln for ln in group if not bullet_re.match(ln)]
        bullets = [bullet_re.sub("", ln).strip() for ln in group if bullet_re.match(ln)]

        if not header_lines and not bullets:
            continue
        if not header_lines and bullets:
            header_lines = ["—"]

        header_joined = " ".join(header_lines)
        if is_work_section and _is_education_bleed_in_experience(header_joined):
            continue

        # Optional: pipe-separated single line still supported
        job_title, company, dates = "", "", ""
        if len(header_lines) == 1 and "|" in header_lines[0]:
            parts = [p.strip() for p in header_lines[0].split("|")]
            job_title = parts[0] if len(parts) > 0 else ""
            company = parts[1] if len(parts) > 1 else ""
            dates = parts[2] if len(parts) > 2 else ""
        else:
            job_title = header_lines[0] if header_lines else ""
            if len(header_lines) >= 2:
                # Second line often dates or location — keep both in headers for display
                dates = header_lines[1]
                if len(header_lines) > 2:
                    company = " · ".join(header_lines[2:])

        hj_low = header_joined.lower()
        if is_work_section and re.search(r"^(personal|academic|capstone|side)\s+project", hj_low):
            continue

        display_headers = list(header_lines)
        entries.append({
            "header_lines": display_headers,
            "job_title": _clean(job_title or (header_lines[0] if header_lines else "")),
            "company": _clean(company),
            "dates": _clean(dates),
            "bullets": bullets[:12],
            "text": " \n ".join(display_headers)[:600],
        })

    def _keep_parsed_experience_entry(e: Dict[str, Any]) -> bool:
        """Drop PDF wrap artifacts for jobs; keep project blocks (titles + stack lines, not employers)."""
        headers = [
            str(h).strip()
            for h in (e.get("header_lines") or [])
            if h and str(h).strip() not in ("—", "-")
        ]
        if not headers:
            return False
        hjoin = " ".join(headers)

        # Projects: do not require employer/date heuristics — resumes often lead with a tech stack line.
        if not is_work_section:
            ptitle = _first_substantive_project_header(headers)
            if ptitle is not None and _is_junk_project_heading_line(ptitle):
                return False
            blob = _collect_entry_text_chunks(e)
            dt = _extract_date_range(blob)
            co_ext = _extract_employer_name(headers)
            h0 = headers[0]
            tech_intro = h0.count(",") >= 2 and len(h0) < 160
            if tech_intro and len(headers) >= 2:
                return True
            if tech_intro and (e.get("bullets") or []):
                return True
            if dt or co_ext:
                return True
            if len(headers) >= 2:
                return True
            if e.get("bullets") and not _is_sentence_or_bullet_fragment(h0):
                return True
            if e.get("bullets") and tech_intro:
                return True
            frag0 = _is_sentence_or_bullet_fragment(h0)
            frag_join = _is_sentence_or_bullet_fragment(hjoin)
            if not dt and not co_ext and (frag0 or frag_join):
                return False
            if not dt and not co_ext and len(hjoin.strip()) < 3:
                return False
            return True

        if is_work_section and _is_education_bleed_in_experience(hjoin):
            return False
        blob = _collect_entry_text_chunks(e)
        dt = _extract_date_range(blob)
        co_ext = _extract_employer_name(headers)
        has_signal = bool(dt or co_ext)
        frag0 = _is_sentence_or_bullet_fragment(headers[0])
        frag_join = _is_sentence_or_bullet_fragment(hjoin)
        if not has_signal and (frag0 or frag_join):
            return False
        if not has_signal and len(headers) == 1 and headers[0].endswith("."):
            return False
        if not has_signal and len(hjoin) < 12:
            return False
        return True

    entries = [e for e in entries if _keep_parsed_experience_entry(e)]

    # Don’t inject a raw-text blob if we only threw away noise; that re-inflates bogus “entries”.
    if not entries and not groups:
        entries.append({
            "header_lines": [_unstick_dates_and_words(ln) for ln in current_block[:6] if ln.strip()],
            "job_title": "",
            "company": "",
            "dates": "",
            "bullets": [],
            "text": " ".join(current_block[:8])[:500],
        })

    filtered: List[Dict[str, Any]] = []
    for e in entries:
        jt = e.get("job_title") or (e.get("header_lines") or [""])[0]
        company = (e.get("company") or "").lower()
        if DEGREE_PATTERN.search(str(jt)) and not e.get("bullets"):
            continue
        if jt and DEGREE_PATTERN.search(str(jt)) and "university" in company:
            continue
        if is_work_section:
            tl = str(jt).lower()
            if re.search(r"^(personal|academic|capstone|side)\s", tl):
                continue
        filtered.append(e)
    return filtered[:8] if filtered else entries[:8]


def _entries_have_real_roles(entries: List[Dict[str, Any]]) -> bool:
    """True if at least one entry looks like a job (not only a fallback text blob)."""
    if not entries:
        return False
    for e in entries:
        if _clean(e.get("job_title", "")) or _clean(e.get("company", "")):
            return True
        if e.get("bullets"):
            return True
    return False


_WORK_DATE_RANGE = re.compile(
    r"(?i)(?:"
    r"\d{1,2}/\d{4}\s*[-–—]\s*(?:\d{1,2}/\d{4}|present|current)|"
    r"(?:19|20)\d{2}\s*[-–—]\s*(?:(?:19|20)\d{2}|present|current)|"
    r"\d{1,2}/\d{4}\s*[-–—]\s*present"
    r")",
)

_MONTH_YEAR_RANGE = re.compile(
    r"(?i)(?:"
    r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+\d{4}\s*[-–—]\s*"
    r"(?:(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+\d{4}|present|current)"
    r")",
)


def _header_line_is_date_range_only(s: str) -> bool:
    t = s.strip()
    if not t or len(t) > 90:
        return False
    return bool(_WORK_DATE_RANGE.search(t) or _MONTH_YEAR_RANGE.search(t))


def _is_junk_project_heading_line(s: str) -> bool:
    """Wrapped bullets / PDF orphans mis-parsed as headings (e.g. 'asynchronous processing.', 'tuning.')."""
    t = (s or "").strip()
    if not t:
        return True
    if _header_line_is_date_range_only(t):
        return False
    if _line_looks_like_project_stack_line(t) or _line_looks_like_project_skills_summary(t):
        return False
    low = t.lower()
    if t.endswith("."):
        if t[0].islower():
            return True
        w = t.split()
        if len(w) <= 3 and len(t) <= 52:
            return True
    stem = low.rstrip(".")
    if len(stem) <= 14 and " " not in stem and t[0].islower():
        return True
    if (
        t[0].islower()
        and len(t) <= 48
        and " " in t
        and not re.search(
            r"(?i)\b(platform|system|dashboard|application|service|engine|interface|suite|portal|"
            r"framework|toolkit|workflow|predictor|detector|recognition|classifier|monitor)\b",
            low,
        )
    ):
        return True
    return False


def _first_substantive_project_header(headers: List[str]) -> Optional[str]:
    for h in headers:
        hh = str(h).strip()
        if not hh:
            continue
        if _header_line_is_date_range_only(hh):
            continue
        if _line_looks_like_project_stack_line(hh) or _line_looks_like_project_skills_summary(hh):
            continue
        return hh
    return None


def _collect_entry_text_chunks(entry: Dict[str, Any]) -> str:
    parts: List[str] = []
    for key in ("company", "dates", "job_title"):
        v = entry.get(key)
        if v:
            parts.append(str(v))
    for h in entry.get("header_lines") or []:
        if h and str(h).strip() not in ("—",):
            parts.append(_unstick_dates_and_words(str(h)))
    for b in entry.get("bullets") or []:
        if b:
            parts.append(str(b))
    if entry.get("text"):
        parts.append(str(entry["text"]))
    return "\n".join(parts)


def _extract_date_range(blob: str) -> str:
    if not blob:
        return ""
    found: List[str] = []
    for rx in (_MONTH_YEAR_RANGE, _WORK_DATE_RANGE):
        for m in rx.finditer(blob):
            found.append(m.group(0).strip())
    return max(found, key=len) if found else ""


_EMPLOYER_LINE_PATTERNS = [
    r"(?i)(University of\s+[A-Za-z][A-Za-z\s&]+?)(?=\s*(?:\n|$|Software|Graduate|Senior|Junior|Lead|Principal|Student|Research|Intern|\d{1,2}/|\d{4}))",
    r"(?i)\b([\w\s&,\.'-]{2,}?Consulting)\b",
    r"(?i)([\w\s&,'-]+?\s+Institute of Technology)",
    r"(?i)([\w\s&,'-]+?Memorial Institute[\w\s&-]*)",
    r"(?i)([\w\s&,'-]+?\s+(?:Inc\.?|LLC|Corp\.?|Corporation|Ltd\.?))\b",
]


def _extract_employer_name(headers: List[str]) -> str:
    """Pick the employer line — scan each header line before joining (handles role / company / dates layout)."""
    if not headers:
        return ""
    for h in headers:
        hh = h.strip()
        if not hh or _WORK_DATE_RANGE.fullmatch(hh.strip()) or _MONTH_YEAR_RANGE.fullmatch(hh.strip()):
            continue
        for pat in _EMPLOYER_LINE_PATTERNS:
            m = re.search(pat, hh)
            if m:
                return _clean(m.group(1))

    joined = "\n".join(headers)
    for pat in _EMPLOYER_LINE_PATTERNS:
        m = re.search(pat, joined)
        if m:
            return _clean(m.group(1))

    for h in headers:
        mm = INSTITUTION_PATTERN.search(h)
        if mm:
            return _clean(mm.group(0))

    h0 = headers[0]
    h0 = _WORK_DATE_RANGE.sub("", h0).strip()
    h0 = re.sub(
        r"(?i)\s*(Software|Graduate|Senior|Junior|Lead|Principal|Student)?\s*"
        r"(Developer|Engineer|Intern|Researcher|Analyst|Scientist|Architect)\b.*$",
        "",
        h0,
    ).strip()
    h0 = re.sub(r",?\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?\s*,\s*[A-Z]{2}\s*$", "", h0).strip()
    return _clean(h0)


def project_upload_display_line(entry: Dict[str, Any]) -> str:
    """Upload tab: project name and dates only (no tech stack or skills lines)."""
    headers = [
        _unstick_dates_and_words(str(h)).strip()
        for h in (entry.get("header_lines") or [])
        if h and str(h).strip() not in ("—", "-")
    ]
    blob = _collect_entry_text_chunks(entry)
    dt = (_extract_date_range(blob) or (entry.get("dates") or "").strip()).strip()

    title = ""
    for h in headers:
        if _header_line_is_date_range_only(h):
            if not dt:
                dt = (_extract_date_range(h) or h).strip()
            continue
        if _line_looks_like_project_stack_line(h) or _line_looks_like_project_skills_summary(h):
            continue
        title = _clean(h)
        break

    if not title:
        jt = _clean(entry.get("job_title") or "")
        if jt and not _line_looks_like_project_stack_line(
            jt
        ) and not _line_looks_like_project_skills_summary(jt):
            title = _WORK_DATE_RANGE.sub("", jt).strip(" ·-|–—/")
    if not title and headers:
        h0 = headers[0]
        if not _line_looks_like_project_stack_line(h0) and not _line_looks_like_project_skills_summary(h0):
            title = _clean(_WORK_DATE_RANGE.sub("", h0).strip())
    if not title and headers:
        title = _clean(_WORK_DATE_RANGE.sub("", headers[0]).strip())

    title = re.sub(r"\s+", " ", title).strip()
    if _is_junk_project_heading_line(title):
        return ""
    parts = [p for p in (title, dt) if p]
    return " · ".join(parts)


def compact_company_and_dates(entry: Dict[str, Any]) -> Tuple[str, str]:
    """
    Employer name + date range only. Scans headers and bullets for dates.
    """
    co = (entry.get("company") or "").strip()
    blob = _collect_entry_text_chunks(entry)
    dt = _extract_date_range(blob) or (entry.get("dates") or "").strip()

    headers = [
        _unstick_dates_and_words(str(h))
        for h in (entry.get("header_lines") or [])
        if h and str(h).strip() and str(h).strip() != "—"
    ]

    if headers:
        extracted = _extract_employer_name(headers)
        if extracted:
            co = extracted
        elif not co:
            line0 = headers[0]
            co = _WORK_DATE_RANGE.sub("", line0).strip(" ·-|–—/")
            co = _normalize_company_display(co)

    co = _WORK_DATE_RANGE.sub("", co).strip(" ·-|–—/")
    co = _normalize_company_display(co)
    co = re.sub(r"\s+", " ", co).strip()

    if not headers and (entry.get("job_title") or "").strip():
        jt = (entry.get("job_title") or "").strip()
        if not co:
            co = _WORK_DATE_RANGE.sub("", jt).strip(" ·-|–—/")
            co = _normalize_company_display(co)
        if not dt:
            dt = _extract_date_range(jt)

    return _clean(co), _clean(dt)


_BULLET_VERB_STARTS = (
    "built ", "designed ", "developed ", "implemented ", "created ", "improved ", "led ",
    "managed ", "refactored ", "optimized ", "enhanced ", "delivered ", "collaborated ",
    "utilized ", "used ", "worked ", "maintained ", "integrated ", "automated ", "deployed ",
    "achieved ", "streamlined ", "conducted ", "analyzed ", "established ", "reduced ",
)


def _is_sentence_or_bullet_fragment(text: str) -> bool:
    """True if this is obviously a duty bullet / sentence, not an employer name."""
    if not text or len(text.strip()) < 3:
        return True
    t = text.strip()
    low = t.lower()
    if t.endswith("."):
        return True
    if any(low.startswith(v) for v in _BULLET_VERB_STARTS):
        return True
    if re.match(r"^[a-z]", low) and (len(t) > 40 or " and " in low or " the " in low):
        return True
    if " using" in low or "usingdjango" in low or "usingfastapi" in low or "usinggo" in low:
        return True
    if re.search(r"\b(datasets?|throughput|latency|pipeline|microservices|scalable)\b", low):
        return True
    # Stack-only line (many commas, no org keywords)
    if t.count(",") >= 2 and len(t) < 180:
        if not re.search(
            r"(?i)(university|consulting|institute|inc\.?|llc|corp|technologies|labs|solutions)\b",
            t,
        ):
            return True
    return False


def is_plausible_work_upload_line(co: str, dt: str, entry: Dict[str, Any]) -> bool:
    """Upload tab: only show rows that look like employer + employment dates (not bullets)."""
    if not co or not dt:
        return False
    if _is_sentence_or_bullet_fragment(co):
        return False
    return True


def count_plausible_role_rows(entries: List[Dict[str, Any]]) -> int:
    """How many entries are real Company · dates rows (matches Upload work list)."""
    n = 0
    for ex in entries:
        co, dt = compact_company_and_dates(ex)
        if is_plausible_work_upload_line(co, dt, ex):
            n += 1
    return n


def _normalize_company_display(company: str) -> str:
    """Trim role/location noise; keep employer fragment."""
    if not company:
        return ""
    co = re.sub(r"\s+", " ", company).strip()
    for sep in (" — ", " – ", " | ", " / "):
        if sep in co:
            parts = co.split(sep, 1)
            if len(parts) == 2 and len(parts[1].strip()) > 1:
                co = parts[1].strip()
                break
    low = co.lower()
    if " at " in low:
        tail = co[low.rfind(" at ") + 4 :].strip()
        if len(tail) > 1 and not _WORK_DATE_RANGE.match(tail):
            co = tail
    return co.strip(" ·-|–—/")


def parse_work_experience(text: str) -> List[Dict[str, Any]]:
    """Only lines under work experience headers (stops before a Projects header)."""
    body = _slice_work_experience_text(text)
    return _parse_experience_block_to_entries(body, is_work_section=True)


def parse_projects_experience(text: str) -> List[Dict[str, Any]]:
    """Lines under Projects / Academic projects / etc."""
    body = _slice_projects_text(text)
    return _parse_experience_block_to_entries(body, is_work_section=False)


def _split_skill_phrases(line: str) -> List[str]:
    """Split one line from a Skills section into separate items."""
    line = re.sub(r"^[\-\*•·]\s*", "", line.strip())
    if not line:
        return []
    parts = re.split(r"[,;|•·/]|(?:\s{2,})", line)
    out = []
    for p in parts:
        p = p.strip().strip("·")
        if len(p) >= 2:
            out.append(p)
    return out


def parse_skills_from_resume(text: str) -> List[str]:
    """Extract skills as written on the resume (Skills section first, then light fallback)."""
    from src.feature_engineering import extract_skills_from_text, is_valid_skill

    seen: set = set()
    skills: List[str] = []

    skills_section = re.search(
        r"(?i)skills?\s*(?:&|and)?\s*(?:competencies|technical\s*skills)?\s*[:\-]?\s*"
        r"([\s\S]*?)(?=\n\s*(?:experience|work\s+history|education|employment|projects?|"
        r"certifications?|summary|objective|references?)\b|\Z)",
        text,
    )
    if skills_section:
        block = skills_section.group(1)
        for raw_line in block.split("\n"):
            line = raw_line.strip()
            if not line:
                continue
            for phrase in _split_skill_phrases(line):
                key = phrase.lower()
                if key in seen:
                    continue
                if not is_valid_skill(phrase):
                    continue
                seen.add(key)
                skills.append(phrase)

    if len(skills) < 5:
        for s in extract_skills_from_text(text):
            key = s.lower()
            if key in seen or not is_valid_skill(s):
                continue
            seen.add(key)
            skills.append(s)

    return skills[:100]


def parse_resume(file_path: Path) -> Dict[str, Any]:
    """Full resume parsing pipeline."""
    raw = extract_text_from_file(file_path)
    skills = parse_skills_from_resume(raw)
    education = parse_education(raw)
    work_entries = parse_work_experience(raw)
    project_entries = parse_projects_experience(raw)
    # Real jobs only; if no parsed roles under Experience, show Projects section here
    has_work = _entries_have_real_roles(work_entries)
    experience = work_entries if has_work else project_entries

    # ATS vs job corpus is filled only after **Run Career Analysis** (see app.run_analysis).
    if has_work:
        exp_metric = count_plausible_role_rows(work_entries)
        # Projects are titled blocks, not jobs — count parsed project groups (after project-friendly keep).
        proj_metric = len(project_entries)
    else:
        exp_metric = count_plausible_role_rows(project_entries)
        proj_metric = 0

    return {
        "raw_text": raw,
        "skills": skills,
        "education": education,
        "experience": experience,
        "projects": project_entries,
        "experience_work": work_entries,
        "ats_keywords_found": [],
        "ats_keywords_missing": [],
        "ats_matched_detail": [],
        "profile_summary": {
            "skills_count": len(skills),
            "education_count": len(education),
            "experience_count": exp_metric,
            "has_work_experience": has_work,
            "projects_count": proj_metric,
            "text_length": len(raw),
            "ats_found_count": 0,
            "ats_missing_count": 0,
        },
    }
