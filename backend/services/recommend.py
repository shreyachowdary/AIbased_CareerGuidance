"""
Recommendation service: TF-IDF + cosine similarity ranking.
Uses precomputed vectorizer and job vectors from preprocessing.
"""
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from backend.services.gap_analysis import compute_skill_gaps, matched_skills
from backend.services.skill_extraction import extract_skills_from_text, normalize_skills


def _ensure_list(x: Any) -> List[str]:
    """Convert input to list of strings (skills)."""
    if x is None:
        return []
    if isinstance(x, str):
        return normalize_skills([s.strip() for s in x.split(",") if s.strip()])
    if isinstance(x, list):
        return normalize_skills([str(s).strip() for s in x if str(s).strip()])
    return []


class RecommendationService:
    """
    Loads saved TF-IDF vectorizer and job matrix; recommends jobs by cosine similarity
    and provides gap analysis.
    """

    def __init__(self, models_dir: str | Path) -> None:
        self.models_dir = Path(models_dir)
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.job_vectors: Optional[np.ndarray] = None
        self.job_metadata: List[Dict[str, Any]] = []
        self._loaded = False

    def load(self) -> bool:
        """Load vectorizer and job vectors from disk. Returns True if successful."""
        import pickle

        vec_path = self.models_dir / "tfidf_vectorizer.pkl"
        meta_path = self.models_dir / "job_metadata.pkl"
        matrix_path = self.models_dir / "job_vectors.npy"

        if not vec_path.exists() or not meta_path.exists() or not matrix_path.exists():
            return False

        with open(vec_path, "rb") as f:
            self.vectorizer = pickle.load(f)
        with open(meta_path, "rb") as f:
            self.job_metadata = pickle.load(f)
        self.job_vectors = np.load(matrix_path)
        self._loaded = True
        return True

    def is_loaded(self) -> bool:
        return self._loaded and self.vectorizer is not None and self.job_vectors is not None

    def recommend(
        self,
        user_profile: Dict[str, Any],
        top_n: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Given user profile (skills, education, interests, desired_role), return
        top_n recommendations with similarity score, matched skills, and missing skills.
        """
        if not self.is_loaded():
            return []

        # Build query text from profile for TF-IDF
        skills = _ensure_list(user_profile.get("skills"))
        education = user_profile.get("education") or ""
        interests = user_profile.get("interests") or ""
        desired_role = user_profile.get("desired_role") or ""
        parts = [
            " ".join(skills),
            str(education),
            str(interests),
            str(desired_role),
        ]
        query_text = " ".join(p for p in parts if p).strip() or "general"
        # Also add any extracted skills from free text
        if not skills and query_text:
            skills = extract_skills_from_text(query_text)
        else:
            # Normalize so we use same format as job skills
            skills = normalize_skills(skills) if skills else extract_skills_from_text(query_text)

        query_vec = self.vectorizer.transform([query_text])
        sims = cosine_similarity(query_vec, self.job_vectors).ravel()

        # Deterministic: sort by score desc, then by job_id asc for ties
        indices = np.argsort(-sims)
        if top_n < len(indices):
            indices = indices[:top_n]

        results: List[Dict[str, Any]] = []
        for idx in indices:
            score = float(sims[idx])
            meta = self.job_metadata[idx]
            job_skills = meta.get("skills") or []
            missing = compute_skill_gaps(job_skills, skills)
            matched = matched_skills(job_skills, skills)
            results.append({
                "job_id": meta.get("job_id"),
                "title": meta.get("title"),
                "company": meta.get("company"),
                "similarity_score": round(score, 4),
                "matched_skills": matched,
                "missing_skills": missing,
            })
        return results
