"""Text cleaning utilities."""
import re


def clean_text(text: str) -> str:
    """Remove HTML, lowercase, normalize whitespace."""
    if not text or not isinstance(text, str):
        return ""
    text = re.sub(r"<[^>]+>", " ", text)
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text
