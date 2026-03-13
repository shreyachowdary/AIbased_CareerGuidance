"""Test API contract: /recommend returns required keys and structure."""
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def client():
    from backend.app.main import app
    return TestClient(app)


def test_health_endpoint(client):
    """GET /health returns 200."""
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json().get("status") == "ok"


def test_recommend_returns_required_keys(client):
    """POST /recommend returns query_skills, mode_used, recommendations, action_plan."""
    # May fail with 503 if data not loaded - that's expected in CI without data
    r = client.post(
        "/recommend",
        json={
            "skills_text": "Python SQL",
            "top_k": 3,
            "mode": "tfidf",
        },
    )
    if r.status_code == 503:
        pytest.skip("Backend has no data - run preprocess and build first")

    assert r.status_code == 200
    data = r.json()
    assert "query_skills" in data
    assert "mode_used" in data
    assert "recommendations" in data
    assert "action_plan" in data
    assert data["mode_used"] == "tfidf"
    assert isinstance(data["query_skills"], list)
    assert isinstance(data["recommendations"], list)
    assert len(data["recommendations"]) <= 3


def test_recommend_action_plan_structure(client):
    """Action plan has top_missing_skills and suggested_next_steps."""
    r = client.post(
        "/recommend",
        json={"skills_text": "Python", "top_k": 2, "mode": "tfidf"},
    )
    if r.status_code == 503:
        pytest.skip("Backend has no data")
    assert r.status_code == 200
    ap = r.json()["action_plan"]
    assert "top_missing_skills" in ap
    assert "suggested_next_steps" in ap
    assert isinstance(ap["top_missing_skills"], list)
    assert isinstance(ap["suggested_next_steps"], str)
