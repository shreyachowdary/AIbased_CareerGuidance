"""
Role-level fit vs real job corpus: aggregate postings by title, score resume vs market profile.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


def normalize_role_title(raw: str) -> str:
    """Stable key for grouping similar posting titles in the corpus."""
    t = (raw or "").strip()
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"\s*[-–—]\s*(remote|hybrid|on[- ]?site|contract|freelance|part[- ]?time|full[- ]?time)\b.*$", "", t, flags=re.I)
    t = t.lower()
    # Fold common level prefixes so market stats aren’t split across “Senior X” vs “X”
    t = re.sub(
        r"^(?:senior|sr\.?|junior|jr\.?|entry[- ]level|mid[- ]?level|intern|graduate)\s+",
        "",
        t,
    )
    t = re.sub(r"\s+", " ", t).strip()
    return t[:120]


def _embedding_row_mean(embeddings, row_indices: np.ndarray) -> np.ndarray:
    sub = embeddings[row_indices]
    if hasattr(sub, "toarray"):
        sub = sub.toarray()
    return np.asarray(sub.mean(axis=0), dtype=np.float64).ravel()


def rank_resume_against_market_roles(
    resume_matching_text: str,
    resume_skills: Sequence[str],
    vectorizer,
    job_embeddings,
    job_metadata: pd.DataFrame,
    job_skills_series: pd.Series,
    *,
    min_postings_per_role: int = 12,
    max_roles: int = 45,
    top_market_skills_per_role: int = 30,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Rank normalized job titles by how well the resume matches **aggregated real postings**
    in the corpus (mean TF-IDF embedding per role + skill overlap with frequent jobs skills).

    Returns:
        (DataFrame sorted by combined score, stats dict with corpus_job_count)
    """
    n_meta = len(job_metadata)
    n_emb = job_embeddings.shape[0] if hasattr(job_embeddings, "shape") else len(job_embeddings)
    if n_meta == 0 or n_emb == 0 or n_meta != n_emb:
        return pd.DataFrame(), {"corpus_job_count": 0, "error": "corpus_alignment"}

    norm = job_metadata["title"].fillna("").map(normalize_role_title)
    raw_titles = job_metadata["title"].fillna("")
    counts = norm.value_counts()
    eligible = counts[counts >= min_postings_per_role].head(max_roles).index.tolist()
    used_min = min_postings_per_role
    # Small / diverse posting files often have no title bucket ≥ min_postings; fall back so local role view still renders.
    if not eligible and min_postings_per_role > 1:
        eligible = counts[counts >= 1].head(max_roles).index.tolist()
        used_min = 1
    if not eligible:
        return pd.DataFrame(), {"corpus_job_count": int(n_meta), "error": "no_eligible_roles"}

    resume_vec = vectorizer.transform([resume_matching_text or " "])
    rs = {str(s).strip().lower() for s in resume_skills if s and str(s).strip()}

    rows: List[dict] = []
    for role_key in eligible:
        idxs = np.flatnonzero((norm == role_key).values)
        if len(idxs) < used_min:
            continue
        centroid = _embedding_row_mean(job_embeddings, idxs)
        text_sim = float(cosine_similarity(resume_vec, centroid.reshape(1, -1))[0, 0])

        # Market skill frequencies for this role (real postings only)
        sk_counter: Counter = Counter()
        for pos in idxs:
            idx = job_metadata.index[pos]
            try:
                skills_cell = job_skills_series.loc[idx]
            except Exception:
                skills_cell = job_skills_series.iloc[pos]
            if not isinstance(skills_cell, list):
                skills_cell = list(skills_cell) if hasattr(skills_cell, "__iter__") and not isinstance(skills_cell, str) else []
            for sk in skills_cell:
                if sk and str(sk).strip():
                    sk_counter[str(sk).strip().lower()] += 1

        top_market = [s for s, _ in sk_counter.most_common(top_market_skills_per_role)]
        top_set = set(top_market)
        overlap = rs & top_set
        skill_density = len(overlap) / max(len(top_set), 1)

        # Blend: text similarity to role centroid (market language) + structured skill coverage
        combined = 0.72 * max(0.0, text_sim) + 0.28 * min(1.0, skill_density * 1.4)

        # Display title: most common raw title string in this bucket
        bucket_raw = raw_titles.iloc[idxs]
        display = bucket_raw.mode().iloc[0] if len(bucket_raw.mode()) else role_key.title()

        missing_freq: List[Tuple[str, int]] = [(s, sk_counter[s]) for s in top_market[:18] if s not in rs]

        rows.append({
            "role_key": role_key,
            "role_display": str(display).strip(),
            "postings_in_corpus": int(len(idxs)),
            "text_similarity": round(text_sim, 4),
            "market_skill_overlap": round(skill_density, 4),
            "fit_score": round(combined, 4),
            "your_skills_matching_market": sorted(overlap)[:24],
            "market_skills_gap": [s for s, _ in missing_freq[:14]],
        })

    out = pd.DataFrame(rows)
    if out.empty:
        return out, {"corpus_job_count": int(n_meta), "error": "empty_result"}
    out = out.sort_values("fit_score", ascending=False).reset_index(drop=True)
    stats = {
        "corpus_job_count": int(n_meta),
        "roles_ranked": int(len(out)),
        "min_postings_per_role": used_min,
    }
    return out, stats
