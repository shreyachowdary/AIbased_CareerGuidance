"""
Job matching: TF-IDF-based similarity between resume and job descriptions.
"""

from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from config.settings import (
    ARTIFACTS_DIR,
    TFIDF_VECTORIZER_PKL,
    JOB_EMBEDDINGS_PKL,
    JOB_METADATA_PKL,
    TOP_N_MATCHES,
)
from src.data_cleaning import build_job_text
from utils.logging_config import get_logger

logger = get_logger("matching")


def fit_tfidf_and_transform(
    job_texts: pd.Series,
    max_features: int = 10000,
    ngram_range: Tuple[int, int] = (1, 2),
) -> Tuple[TfidfVectorizer, np.ndarray]:
    """
    Fit TF-IDF vectorizer on job texts and transform.

    Returns:
        (vectorizer, job_embeddings matrix)
    """
    n_docs = max(len(job_texts), 1)
    # Small API result sets need min_df=1 or the vocabulary collapses
    min_df = 1 if n_docs < 50 else 2
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        stop_words="english",
        min_df=min_df,
        max_df=0.95,
        sublinear_tf=True,
    )
    embeddings = vectorizer.fit_transform(job_texts.astype(str))
    return vectorizer, embeddings


def _normalize_scores(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64)
    lo, hi = float(scores.min()), float(scores.max())
    if hi - lo < 1e-9:
        return np.ones_like(scores) * 0.5
    return (scores - lo) / (hi - lo)


def match_resume_to_jobs_hybrid(
    resume_text: str,
    resume_skills: Sequence[str],
    vectorizer: TfidfVectorizer,
    job_embeddings: np.ndarray,
    job_metadata: pd.DataFrame,
    job_skills_series: Optional[pd.Series] = None,
    raw_text_for_titles: str = "",
    top_n: int = TOP_N_MATCHES,
    tfidf_weight: float = 0.52,
    skill_weight: float = 0.33,
    title_weight: float = 0.15,
) -> pd.DataFrame:
    """
    Combine TF-IDF cosine similarity with skill overlap and job-title / career alignment.
    """
    from src.resume_intent import resume_signals_swe_primary, title_role_alignment_score

    resume_vec = vectorizer.transform([resume_text])
    tfidf = cosine_similarity(resume_vec, job_embeddings).flatten()
    n = len(tfidf)
    skill_part = np.zeros(n, dtype=np.float64)
    title_part = np.zeros(n, dtype=np.float64)

    rs = {str(s).lower().strip() for s in resume_skills if s and str(s).strip()}
    rl = (raw_text_for_titles or resume_text or "").lower()
    swe = resume_signals_swe_primary(raw_text_for_titles or resume_text, list(resume_skills))

    for pos in range(n):
        idx = job_metadata.index[pos]
        title = str(job_metadata.iloc[pos].get("title", "") or "")
        title_part[pos] = title_role_alignment_score(title, rl, swe)

        if job_skills_series is not None and idx in job_skills_series.index:
            js = job_skills_series.loc[idx]
            if not isinstance(js, list):
                js = list(js) if hasattr(js, "__iter__") and not isinstance(js, str) else [str(js)]
            jset = {str(x).lower().strip() for x in js if x}
            if jset:
                overlap = len(rs & jset) / max(len(jset), 1)
                skill_part[pos] = min(1.0, overlap * 1.35)
            else:
                skill_part[pos] = 0.2 * title_part[pos]

    combined = (
        tfidf_weight * _normalize_scores(tfidf)
        + skill_weight * _normalize_scores(skill_part)
        + title_weight * title_part
    )
    top_indices = np.argsort(combined)[::-1][:top_n]
    results = job_metadata.iloc[top_indices].copy()
    results["match_score"] = combined[top_indices]
    results["job_index"] = results.index
    results = results.reset_index(drop=True)
    return results


def match_resume_to_jobs_dynamic(
    resume_text: str,
    job_df: pd.DataFrame,
    top_n: int = TOP_N_MATCHES,
    resume_skills: Optional[Sequence[str]] = None,
    raw_text_hint: str = "",
) -> Tuple[pd.DataFrame, TfidfVectorizer, np.ndarray, pd.Series]:
    """
    Match resume to a dynamic job DataFrame (e.g. from API).
    Fits vectorizer on the fly. Returns (matches, vectorizer, embeddings, job_skills).
    """
    from src.feature_engineering import build_skill_set_per_job

    job_texts = build_job_text(job_df)
    vectorizer, embeddings = fit_tfidf_and_transform(job_texts)
    want = [
        "job_id", "title", "company", "location", "job_type", "description",
        "posted_date", "apply_link", "job_google_link", "job_publisher",
    ]
    cols = [c for c in want if c in job_df.columns]
    metadata = job_df[cols].copy() if cols else job_df.iloc[:, :0].copy()
    for c in want:
        if c not in metadata.columns:
            metadata[c] = ""
    job_skills = build_skill_set_per_job(job_df)
    matches = match_resume_to_jobs_hybrid(
        resume_text,
        resume_skills or [],
        vectorizer,
        embeddings,
        metadata,
        job_skills_series=job_skills,
        raw_text_for_titles=raw_text_hint or "",
        top_n=top_n,
    )
    return matches, vectorizer, embeddings, job_skills


def match_resume_to_jobs(
    resume_text: str,
    vectorizer: TfidfVectorizer,
    job_embeddings: np.ndarray,
    job_metadata: pd.DataFrame,
    top_n: int = TOP_N_MATCHES,
) -> pd.DataFrame:
    """
    Match resume text to jobs using cosine similarity.

    Args:
        resume_text: Combined resume text (skills, experience, etc.).
        vectorizer: Fitted TF-IDF vectorizer.
        job_embeddings: Job TF-IDF matrix.
        job_metadata: DataFrame with job_id, title, company, etc.
        top_n: Number of top matches to return.

    Returns:
        DataFrame of top matches with match_score column.
    """
    resume_vec = vectorizer.transform([resume_text])
    scores = cosine_similarity(resume_vec, job_embeddings).flatten()
    top_indices = np.argsort(scores)[::-1][:top_n]
    results = job_metadata.iloc[top_indices].copy()
    results["match_score"] = scores[top_indices]
    results["job_index"] = results.index  # preserve for skill gap lookup
    results = results.reset_index(drop=True)
    return results


def save_matching_artifacts(
    vectorizer: TfidfVectorizer,
    job_embeddings: np.ndarray,
    job_metadata: pd.DataFrame,
) -> None:
    """Persist vectorizer, embeddings, and metadata as pickle files."""
    import pickle

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(ARTIFACTS_DIR / TFIDF_VECTORIZER_PKL, "wb") as f:
        pickle.dump(vectorizer, f)
    with open(ARTIFACTS_DIR / JOB_EMBEDDINGS_PKL, "wb") as f:
        pickle.dump(job_embeddings, f)
    with open(ARTIFACTS_DIR / JOB_METADATA_PKL, "wb") as f:
        pickle.dump(job_metadata, f)
    logger.info("Saved matching artifacts to %s", ARTIFACTS_DIR)


def load_matching_artifacts() -> Optional[Tuple[TfidfVectorizer, np.ndarray, pd.DataFrame]]:
    """Load vectorizer, embeddings, and metadata from pickle files."""
    import pickle

    v_path = ARTIFACTS_DIR / TFIDF_VECTORIZER_PKL
    e_path = ARTIFACTS_DIR / JOB_EMBEDDINGS_PKL
    m_path = ARTIFACTS_DIR / JOB_METADATA_PKL
    if not all(p.exists() for p in [v_path, e_path, m_path]):
        return None
    with open(v_path, "rb") as f:
        vectorizer = pickle.load(f)
    with open(e_path, "rb") as f:
        embeddings = pickle.load(f)
    with open(m_path, "rb") as f:
        metadata = pickle.load(f)
    return vectorizer, embeddings, metadata
