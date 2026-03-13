"""
Job matching: TF-IDF-based similarity between resume and job descriptions.
"""

from pathlib import Path
from typing import List, Optional, Tuple

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
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        stop_words="english",
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
    )
    embeddings = vectorizer.fit_transform(job_texts.astype(str))
    return vectorizer, embeddings


def match_resume_to_jobs_dynamic(
    resume_text: str,
    job_df: pd.DataFrame,
    top_n: int = TOP_N_MATCHES,
) -> Tuple[pd.DataFrame, TfidfVectorizer, np.ndarray, pd.Series]:
    """
    Match resume to a dynamic job DataFrame (e.g. from API).
    Fits vectorizer on the fly. Returns (matches, vectorizer, embeddings, job_skills).
    """
    from src.feature_engineering import build_skill_set_per_job

    job_texts = build_job_text(job_df)
    vectorizer, embeddings = fit_tfidf_and_transform(job_texts)
    base_cols = ["job_id", "title", "company", "location", "job_type", "description"]
    extra = ["apply_link"] if "apply_link" in job_df.columns else []
    metadata = job_df[[c for c in base_cols + extra if c in job_df.columns]].copy()
    if "apply_link" not in metadata.columns:
        metadata["apply_link"] = ""
    matches = match_resume_to_jobs(resume_text, vectorizer, embeddings, metadata, top_n)
    job_skills = build_skill_set_per_job(job_df)
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
