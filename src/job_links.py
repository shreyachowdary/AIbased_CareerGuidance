"""Job-board search URLs when feeds omit apply links (offline samples, sparse API rows)."""

from __future__ import annotations

import re
from typing import List, Tuple
from urllib.parse import quote_plus


def simplify_job_title_for_search(title: str, max_words: int = 8) -> str:
    """
    ATS titles are often long: ``Role - Dept - Level - Institution``.
    Job boards match poorly on the full string; the first segment is usually the role.
    """
    t = (title or "").strip()
    if not t:
        return "jobs"
    # Prefer the segment before the first multi-part separator (role is almost always first).
    for sep in (" — ", " – ", " - ", " | ", " • "):
        if sep in t:
            t = t.split(sep, 1)[0].strip()
            break
    t = re.sub(r"\s+", " ", t).strip()
    words = t.split()
    if len(words) > max_words:
        t = " ".join(words[:max_words])
    if len(t) < 2:
        return (title or "jobs").strip()[:80]
    return t


def job_board_search_links(title: str, company: str, location: str = "") -> List[Tuple[str, str]]:
    """
    Stable search URLs for finding similar roles. Uses a shortened title for boards
    (not title + company verbatim) so LinkedIn/Indeed return results instead of
    zero matches on one-off posting titles.
    """
    t = (title or "").strip()
    c = (company or "").strip()
    loc = (location or "").strip()

    role_q = simplify_job_title_for_search(t)
    core = " ".join(p for p in (role_q, c) if p).strip() or role_q or "jobs"

    # Google: still include company to help locate the original posting when possible.
    web_q = quote_plus(f"{core} jobs" + (f" {loc}" if loc else ""))
    keywords = quote_plus(role_q)
    indeed_q = quote_plus(role_q)
    indeed_l = quote_plus(loc)

    linkedin = f"https://www.linkedin.com/jobs/search/?keywords={keywords}"
    if loc:
        linkedin += f"&location={quote_plus(loc)}"

    return [
        ("Google", f"https://www.google.com/search?q={web_q}"),
        ("LinkedIn Jobs", linkedin),
        ("Indeed", f"https://www.indeed.com/jobs?q={indeed_q}&l={indeed_l}"),
        ("Glassdoor", f"https://www.glassdoor.com/Job/jobs.htm?sc.keyword={keywords}"),
    ]
