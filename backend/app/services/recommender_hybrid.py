"""Hybrid recommender: alpha * tfidf + (1-alpha) * embed."""
from backend.app.services.recommender_tfidf import recommend as tfidf_recommend
from backend.app.services.recommender_embeddings import recommend as embed_recommend


def recommend(
    query_text: str,
    top_k: int = 10,
    alpha: float = 0.5,
    job_ids: list[str] | None = None,
) -> list[tuple[str, float]]:
    """
    Combine TF-IDF and embedding scores: score = alpha * tfidf + (1-alpha) * embed.
    """
    tfidf_results = tfidf_recommend(query_text, top_k=top_k * 3, job_ids=job_ids)
    embed_results = embed_recommend(query_text, top_k=top_k * 3, job_ids=job_ids)

    tfidf_scores = {jid: s for jid, s in tfidf_results}
    embed_scores = {jid: s for jid, s in embed_results}

    # Normalize to [0,1] for fair combination
    def _norm(d):
        if not d:
            return d
        vals = list(d.values())
        mn, mx = min(vals), max(vals)
        if mx == mn:
            return {k: 1.0 for k in d}
        return {k: (v - mn) / (mx - mn) for k, v in d.items()}

    tfidf_n = _norm(tfidf_scores)
    embed_n = _norm(embed_scores)

    all_jobs = set(tfidf_n) | set(embed_n)
    combined = {}
    for jid in all_jobs:
        t = tfidf_n.get(jid, 0.0)
        e = embed_n.get(jid, 0.0)
        combined[jid] = alpha * t + (1 - alpha) * e

    sorted_jobs = sorted(combined.items(), key=lambda x: -x[1])[:top_k]
    return sorted_jobs
