"""Embedding-based recommender using cosine similarity."""
import joblib
import numpy as np

from backend.app.core.config import EMBED_DIR
from backend.app.storage.artifacts import load_embeddings

_embed_model = None


def _get_model():
    global _embed_model
    if _embed_model is None:
        _embed_model = joblib.load(EMBED_DIR / "model.joblib")
    return _embed_model


def recommend(
    query_text: str,
    top_k: int = 10,
    job_ids: list[str] | None = None,
) -> list[tuple[str, float]]:
    """
    Return list of (job_id, score) sorted by relevance.
    """
    embeddings, metadata, _ = load_embeddings()
    job_id_list = metadata["job_ids"]
    model = _get_model()

    q_emb = model.encode([query_text])
    # Sentence-transformers outputs are L2-normalized, so dot product = cosine similarity
    scores = np.dot(embeddings, q_emb.T).flatten()

    idx_scores = list(enumerate(scores))
    idx_scores.sort(key=lambda x: -x[1])

    result = []
    for idx, score in idx_scores:
        if len(result) >= top_k:
            break
        jid = job_id_list[idx]
        if job_ids is not None and jid not in job_ids:
            continue
        result.append((jid, float(score)))

    return result
