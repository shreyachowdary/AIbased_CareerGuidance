"""TF-IDF fit/transform and cosine helpers — shared numerical core."""

from __future__ import annotations

from typing import Any, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


def fit_tfidf_on_jobs(job_texts: pd.Series) -> Tuple[Any, Any]:
    """Delegates to project vectorizer (min_df scales for small sets)."""
    from src.matching import fit_tfidf_and_transform

    return fit_tfidf_and_transform(job_texts.astype(str))


def cosine_sparse_dense(vec_a, vec_b_2d: np.ndarray) -> float:
    """Cosine similarity query row vs dense centroid."""
    return float(cosine_similarity(vec_a, vec_b_2d)[0, 0])
