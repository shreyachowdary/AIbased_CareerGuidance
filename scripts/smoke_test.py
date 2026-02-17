"""
Smoke test: GET /health and POST /recommend.
Run from project root. Expect backend at http://127.0.0.1:8000.
"""
import sys
from pathlib import Path

import requests

BACKEND_URL = "http://127.0.0.1:8000"


def main() -> int:
    print("Smoke test — backend at", BACKEND_URL)
    ok = True

    # GET /health
    try:
        r = requests.get(f"{BACKEND_URL}/health", timeout=5)
        r.raise_for_status()
        data = r.json()
        print("  GET /health:", data)
        if not data.get("models_loaded"):
            print("  Warning: models not loaded. Run scripts/build_vectors.py")
    except Exception as e:
        print("  GET /health FAILED:", e)
        ok = False

    # POST /recommend
    try:
        r = requests.post(
            f"{BACKEND_URL}/recommend",
            json={
                "skills": "Python, SQL, Git",
                "education": "B.S. CS",
                "interests": "backend",
                "desired_role": "Software Engineer",
            },
            params={"top_n": 3},
            timeout=10,
        )
        r.raise_for_status()
        data = r.json()
        recs = data.get("recommendations") or []
        print("  POST /recommend: received", len(recs), "recommendations")
        if recs:
            print("  First:", recs[0].get("title"), "— score:", recs[0].get("similarity_score"))
        if data.get("message"):
            print("  Message:", data.get("message")[:80] + "..." if len(data.get("message", "")) > 80 else data.get("message"))
    except Exception as e:
        print("  POST /recommend FAILED:", e)
        ok = False

    if ok:
        print("Smoke test passed.")
        return 0
    print("Smoke test failed.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
