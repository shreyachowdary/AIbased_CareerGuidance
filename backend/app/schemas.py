"""API request/response schemas."""
from typing import Literal

from pydantic import BaseModel, Field


class RecommendFilters(BaseModel):
    """Optional filters for recommendations."""
    location_contains: str | None = None
    company_contains: str | None = None
    title_contains: str | None = None


class RecommendRequest(BaseModel):
    """POST /recommend request body."""
    skills_text: str = Field(..., description="User skills as free text")
    education: str | None = None
    interests: str | None = None
    desired_role: str | None = None
    top_k: int = Field(default=10, ge=1, le=50)
    mode: Literal["tfidf", "embed", "hybrid"] = "hybrid"
    alpha: float = Field(default=0.5, ge=0.0, le=1.0)
    filters: RecommendFilters | None = None


class RecommendationItem(BaseModel):
    """Single job recommendation."""
    job_id: str
    title: str
    company: str
    location: str
    score: float
    matched_skills: list[str]
    missing_skills: list[str]
    explanation: str


class ActionPlan(BaseModel):
    """Aggregated action plan."""
    top_missing_skills: list[str]
    suggested_next_steps: str


class RecommendResponse(BaseModel):
    """POST /recommend response."""
    query_skills: list[str]
    mode_used: str
    recommendations: list[RecommendationItem]
    action_plan: ActionPlan
