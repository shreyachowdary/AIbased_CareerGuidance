"""
FastAPI application: /health and POST /recommend.
Lazy-loads RecommendationService (and sklearn/numpy) on first use to avoid slow startup.
"""
from pathlib import Path
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Project root: assume we run from project root (uvicorn backend.main:app)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"

app = FastAPI(
    title="AI Career Guidance System",
    description="MVP: recommend jobs and skill-gap analysis",
    version="0.1.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Lazy-loaded to avoid slow import of sklearn/numpy at server start
_recommendation_service: Optional[object] = None


def _get_recommendation_service() -> object:
    """Load recommendation service on first use (sklearn/numpy load here)."""
    global _recommendation_service
    if _recommendation_service is None:
        from backend.services.recommend import RecommendationService
        _recommendation_service = RecommendationService(MODELS_DIR)
        _recommendation_service.load()
    return _recommendation_service


class UserProfile(BaseModel):
    """User profile for recommendations."""

    skills: str | list[str] | None = Field(default=None, description="Comma-separated or list of skills")
    education: str | None = Field(default=None, description="Education background")
    interests: str | None = Field(default=None, description="Interests")
    desired_role: str | None = Field(default=None, description="Preferred job role")


class RecommendResponse(BaseModel):
    """Response for /recommend."""

    recommendations: list[dict]
    message: str | None = None


@app.get("/health")
def health() -> dict:
    """Health check. Fast; models_loaded is True if model files exist (no sklearn import)."""
    loaded = False
    if _recommendation_service is not None:
        loaded = _recommendation_service.is_loaded()
    else:
        # Fast check: do model files exist? (avoids importing sklearn)
        loaded = (
            (MODELS_DIR / "tfidf_vectorizer.pkl").exists()
            and (MODELS_DIR / "job_metadata.pkl").exists()
            and (MODELS_DIR / "job_vectors.npy").exists()
        )
    return {
        "status": "ok",
        "models_loaded": loaded,
    }


@app.post("/recommend", response_model=RecommendResponse)
def recommend(profile: UserProfile, top_n: int = 10) -> RecommendResponse:
    """
    Get top-N job recommendations and skill-gap analysis for the given user profile.
    """
    if top_n < 1 or top_n > 50:
        top_n = 10
    recommendation_service = _get_recommendation_service()
    profile_dict = profile.model_dump()
    recs = recommendation_service.recommend(profile_dict, top_n=top_n)

    # Learning roadmap: aggregate missing skills across top jobs + placeholder for resources
    skills_to_learn: list[str] = []
    if recs:
        seen: set[str] = set()
        for r in recs:
            for s in r.get("missing_skills") or []:
                if s not in seen:
                    seen.add(s)
                    skills_to_learn.append(s)
        # Limit to 15 for readability
        skills_to_learn = skills_to_learn[:15]

    message = None
    if not recs:
        message = "No recommendations. Run scripts/build_vectors.py to build models."
    elif skills_to_learn:
        message = f"Skills to learn next (priority): {', '.join(skills_to_learn)}. Resource links: [placeholder]"

    return RecommendResponse(
        recommendations=recs,
        message=message,
    )
