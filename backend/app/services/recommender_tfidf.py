"""TF-IDF recommender using cosine similarity."""
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from backend.app.services.dataset_loader import load_jobs
from backend.app.services.skill_extraction import extract_skills
from backend.app.storage.artifacts import load_tfidf


def recommend(
    query_text: str,
    top_k: int = 10,
    job_ids: list[str] | None = None,
) -> list[tuple[str, float]]:
    """
    Return list of (job_id, score) sorted by relevance.
    job_ids: optional filter to restrict to these job IDs.
    """
    vectorizer, matrix, jobs_df = load_tfidf()
    q_vec = vectorizer.transform([query_text])
    scores = cosine_similarity(q_vec, matrix).flatten()

    idx_scores = list(enumerate(scores))
    idx_scores.sort(key=lambda x: -x[1])

    result = []
    for idx, score in idx_scores:
        if len(result) >= top_k:
            break
        jid = jobs_df.iloc[idx]["job_id"]
        if job_ids is not None and jid not in job_ids:
            continue
        result.append((jid, float(score)))

    return result
