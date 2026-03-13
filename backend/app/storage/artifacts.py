"""Load TF-IDF and embedding artifacts."""
import joblib
from pathlib import Path

import numpy as np
import pandas as pd

from backend.app.core.config import TFIDF_DIR, EMBED_DIR

_vectorizer = None
_tfidf_matrix = None
_tfidf_jobs = None
_embeddings = None
_embed_metadata = None
_embed_jobs = None


def load_tfidf():
    """Lazy load TF-IDF artifacts."""
    global _vectorizer, _tfidf_matrix, _tfidf_jobs
    if _vectorizer is None:
        _vectorizer = joblib.load(TFIDF_DIR / "vectorizer.joblib")
        _tfidf_matrix = joblib.load(TFIDF_DIR / "matrix.joblib")
        _tfidf_jobs = pd.read_csv(TFIDF_DIR / "jobs.csv")
    return _vectorizer, _tfidf_matrix, _tfidf_jobs


def load_embeddings():
    """Lazy load embedding artifacts."""
    global _embeddings, _embed_metadata, _embed_jobs
    if _embeddings is None:
        _embeddings = np.load(EMBED_DIR / "embeddings.npy")
        _embed_metadata = joblib.load(EMBED_DIR / "metadata.joblib")
        _embed_jobs = pd.read_csv(EMBED_DIR / "jobs.csv")
    return _embeddings, _embed_metadata, _embed_jobs
