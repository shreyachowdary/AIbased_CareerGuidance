"""FastAPI application for AI Career Guidance."""
import pandas as pd
from fastapi import FastAPI, HTTPException

from backend.app.core.config import PROCESSED_CSV
from backend.app.core.logger import logger
from backend.app.schemas import (
    ActionPlan,
    RecommendationItem,
    RecommendRequest,
    RecommendResponse,
)
from backend.app.services.dataset_loader import load_jobs
from backend.app.services.explanations import explain
from backend.app.services.gap_analysis import analyze
from backend.app.services.recommender_embeddings import recommend as embed_recommend
from backend.app.services.recommender_hybrid import recommend as hybrid_recommend
from backend.app.services.recommender_tfidf import recommend as tfidf_recommend
from backend.app.services.skill_extraction import extract_skills

app = FastAPI(title="AI Career Guidance API", version="1.0.0")

_jobs_df: pd.DataFrame | None = None


def get_jobs_df() -> pd.DataFrame:
    global _jobs_df
    if _jobs_df is None:
        if not PROCESSED_CSV.exists():
            raise HTTPException(
                status_code=503,
                detail="Jobs data not loaded. Run preprocess and build scripts first.",
            )
        _jobs_df = load_jobs()
    return _jobs_df


def build_query(req: RecommendRequest) -> str:
    """Build search query from user profile."""
    parts = [req.skills_text]
    if req.interests:
        parts.append(req.interests)
    if req.desired_role:
        parts.append(req.desired_role)
    if req.education:
        parts.append(req.education)
    return " ".join(parts)


def filter_job_ids(df: pd.DataFrame, filters) -> list[str] | None:
    """Apply filters, return list of job_ids to consider, or None for no filter."""
    if not filters:
        return None
    mask = pd.Series([True] * len(df))
    if filters.location_contains:
        loc = str(filters.location_contains).lower()
        mask &= df["location"].fillna("").str.lower().str.contains(loc, na=False)
    if filters.company_contains:
        comp = str(filters.company_contains).lower()
        mask &= df["company"].fillna("").str.lower().str.contains(comp, na=False)
    if filters.title_contains:
        tit = str(filters.title_contains).lower()
        mask &= df["title"].fillna("").str.lower().str.contains(tit, na=False)
    return df.loc[mask, "job_id"].tolist()


def build_action_plan(recommendations: list[RecommendationItem]) -> ActionPlan:
    """Aggregate top missing skills and suggest next steps."""
    all_missing = []
    for r in recommendations:
        all_missing.extend(r.missing_skills)

    from collections import Counter
    counts = Counter(all_missing)
    top_missing = [s for s, _ in counts.most_common(10)]

    if top_missing:
        steps = (
            f"Focus on learning these skills next: {', '.join(top_missing[:5])}. "
            "They appear frequently in your top job matches. Consider online courses, "
            "projects, or certifications to build these competencies."
        )
    else:
        steps = "Your skills align well with the recommended roles. Consider applying to these positions."

    return ActionPlan(top_missing_skills=top_missing, suggested_next_steps=steps)


@app.get("/health")
def health():
    """Health check."""
    return {"status": "ok"}


@app.post("/recommend", response_model=RecommendResponse)
def recommend(req: RecommendRequest):
    """Get job recommendations with skill-gap analysis and action plan."""
    try:
        df = get_jobs_df()
    except HTTPException:
        raise

    query = build_query(req)
    query_skills = extract_skills(query)
    job_ids = filter_job_ids(df, req.filters)

    if req.mode == "tfidf":
        results = tfidf_recommend(query, top_k=req.top_k, job_ids=job_ids)
    elif req.mode == "embed":
        results = embed_recommend(query, top_k=req.top_k, job_ids=job_ids)
    else:
        results = hybrid_recommend(
            query, top_k=req.top_k, alpha=req.alpha, job_ids=job_ids
        )

    job_lookup = df.set_index("job_id").to_dict("index")
    recommendations = []

    for jid, score in results:
        if jid not in job_lookup:
            continue
        row = job_lookup[jid]
        title = row.get("title", "")
        company = row.get("company", "")
        location = row.get("location", "")
        desc = row.get("description", "")
        job_text = f"{title} {desc}"

        matched, missing = analyze(query_skills, job_text, top_missing=10)
        expl = explain(query_skills, job_text, matched, missing)

        recommendations.append(
            RecommendationItem(
                job_id=jid,
                title=title,
                company=company,
                location=location,
                score=round(score, 4),
                matched_skills=matched,
                missing_skills=missing,
                explanation=expl,
            )
        )

    action_plan = build_action_plan(recommendations)

    return RecommendResponse(
        query_skills=query_skills,
        mode_used=req.mode,
        recommendations=recommendations,
        action_plan=action_plan,
    )
