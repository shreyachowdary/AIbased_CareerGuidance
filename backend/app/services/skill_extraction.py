"""
Vocabulary-based skill extraction with phrase matching and normalization.
"""
import re
from pathlib import Path

from backend.app.core.config import SKILLS_VOCAB

# Normalization: alias -> canonical
NORMALIZE_MAP = {
    "js": "javascript",
    "ts": "typescript",
    "nodejs": "node",
    "golang": "go",
    "sklearn": "scikit-learn",
    "ml": "machine learning",
    "dl": "deep learning",
    "nlp": "natural language processing",
    "cv": "computer vision",
    "aws": "amazon web services",
    "gcp": "google cloud",
    "k8s": "kubernetes",
    "postgres": "postgresql",
    "py": "python",
}

_skills_vocab: list[tuple[str, str]] | None = None  # (pattern, canonical)


def _load_skills_vocab() -> list[tuple[str, str]]:
    """Load skills vocabulary: list of (regex_pattern, canonical_name)."""
    global _skills_vocab
    if _skills_vocab is not None:
        return _skills_vocab

    if not SKILLS_VOCAB.exists():
        _skills_vocab = []
        return _skills_vocab

    skills = []
    with open(SKILLS_VOCAB, encoding="utf-8") as f:
        for line in f:
            line = line.strip().lower()
            if not line or line.startswith("#"):
                continue
            # Take first token as canonical (before any comment)
            canonical = line.split("#")[0].strip()
            if not canonical:
                continue
            # Multi-word: use phrase matching with word boundaries
            if " " in canonical or len(canonical) >= 4:
                pattern = r"\b" + re.escape(canonical) + r"\b"
            else:
                pattern = r"\b" + re.escape(canonical) + r"\b"
            skills.append((pattern, canonical))

    # Sort by length descending so longer phrases match first
    skills.sort(key=lambda x: -len(x[1]))
    _skills_vocab = skills
    return _skills_vocab


def extract_skills(text: str) -> list[str]:
    """
    Extract skills from text using vocabulary and normalization.
    Returns sorted unique skills (canonical names).
    """
    if not text or not isinstance(text, str):
        return []
    text = text.lower()
    vocab = _load_skills_vocab()
    found = set()

    for pattern, canonical in vocab:
        if re.search(pattern, text, re.IGNORECASE):
            found.add(canonical)

    # Apply normalization for common aliases in text
    for alias, canonical in NORMALIZE_MAP.items():
        if re.search(r"\b" + re.escape(alias) + r"\b", text, re.IGNORECASE):
            found.add(canonical)

    return sorted(found)
