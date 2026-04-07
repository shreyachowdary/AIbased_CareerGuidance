"""Backend entry for resume parsing — thin wrapper."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from src.resume_parser import parse_resume


def parse_resume_path(path: Path) -> Dict[str, Any]:
    return parse_resume(path)
