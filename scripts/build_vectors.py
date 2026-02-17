"""
Offline preprocessing: build TF-IDF vectorizer on job descriptions (and title/skills),
save vectorizer + job vectors + metadata to models/.
Run from project root: python scripts/build_vectors.py
"""
import pickle
import sys
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# Ensure project root is on path when running as script
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.services.ingestion import load_jobs_from_csv


def main() -> None:
    data_path = PROJECT_ROOT / "data" / "jobs.csv"
    models_dir = PROJECT_ROOT / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    jobs = load_jobs_from_csv(data_path)
    if not jobs:
        print("No jobs loaded. Ensure data/jobs.csv exists and has job_id, title, description.")
        sys.exit(1)

    # One document per job: title + description + skills as text (for TF-IDF)
    documents = []
    metadata = []
    for j in jobs:
        title = j.get("title") or ""
        desc = j.get("description") or ""
        skills = j.get("skills") or []
        skills_text = " ".join(skills)
        doc = f"{title} {desc} {skills_text}".strip()
        documents.append(doc)
        metadata.append({
            "job_id": j.get("job_id"),
            "title": j.get("title"),
            "company": j.get("company"),
            "skills": list(skills),
        })

    vectorizer = TfidfVectorizer(
        max_features=10_000,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.95,
        sublinear_tf=True,
    )
    job_vectors = vectorizer.fit_transform(documents)

    # Save as dense numpy for simpler loading (MVP size is small)
    job_vectors_np = job_vectors.toarray().astype(np.float32)

    with open(models_dir / "tfidf_vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)
    with open(models_dir / "job_metadata.pkl", "wb") as f:
        pickle.dump(metadata, f)
    np.save(models_dir / "job_vectors.npy", job_vectors_np)

    print(f"Saved vectorizer and {len(metadata)} job vectors to {models_dir}")


if __name__ == "__main__":
    main()
