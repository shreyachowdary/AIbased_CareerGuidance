"""Job-board search URLs when feeds omit apply links (offline samples, sparse API rows)."""

from __future__ import annotations

from typing import List, Tuple
from urllib.parse import quote_plus


def job_board_search_links(title: str, company: str, location: str = "") -> List[Tuple[str, str]]:
    """
    Stable search URLs for the title + company (+ location).
    Use when ``apply_link`` / ``job_google_link`` are empty.
    """
    t = (title or "").strip()
    c = (company or "").strip()
    loc = (location or "").strip()

    core = " ".join(p for p in (t, c) if p).strip()
    if not core:
        core = t or "jobs"

    # Query used in a normal Google web search (reliable across regions)
    web_q = quote_plus(f"{core} jobs" + (f" {loc}" if loc else ""))
    # Board-specific parameters
    keywords = quote_plus(f"{t} {c}".strip() or t or "jobs")
    indeed_q = quote_plus(t or "jobs")
    indeed_l = quote_plus(loc)

    return [
        ("Google", f"https://www.google.com/search?q={web_q}"),
        ("LinkedIn Jobs", f"https://www.linkedin.com/jobs/search/?keywords={keywords}"),
        ("Indeed", f"https://www.indeed.com/jobs?q={indeed_q}&l={indeed_l}"),
        ("Glassdoor", f"https://www.glassdoor.com/Job/jobs.htm?sc.keyword={keywords}"),
    ]
