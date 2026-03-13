"""Generate short human-readable explanations for recommendations."""
from backend.app.services.gap_analysis import analyze


def explain(
    query_skills: list[str],
    job_text: str,
    matched_skills: list[str],
    missing_skills: list[str],
) -> str:
    """
    Produce a short explanation: "Recommended because... Missing skills..."
    """
    parts = []
    if matched_skills:
        parts.append(f"Recommended because you have: {', '.join(matched_skills[:5])}")
        if len(matched_skills) > 5:
            parts[0] += f" (+{len(matched_skills) - 5} more)"
    else:
        parts.append("Recommended based on semantic/text similarity.")

    if missing_skills:
        parts.append(f"Missing skills: {', '.join(missing_skills[:5])}")
        if len(missing_skills) > 5:
            parts[-1] += f" (+{len(missing_skills) - 5} more)"
    else:
        parts.append("No significant skill gaps identified.")

    return " ".join(parts)
