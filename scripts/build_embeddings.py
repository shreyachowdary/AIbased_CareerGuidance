#!/usr/bin/env python3
"""
Build embeddings for job titles + descriptions using sentence-transformers.
Saves embeddings and metadata to models/embeddings/.
"""
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_CSV = PROJECT_ROOT / "data" / "processed" / "jobs.csv"
EMBED_DIR = PROJECT_ROOT / "models" / "embeddings"

MODEL_NAME = os.environ.get("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
BATCH_SIZE = 64


def main():
    if not PROCESSED_CSV.exists():
        print(f"ERROR: {PROCESSED_CSV} not found. Run scripts/preprocess.py first.")
        sys.exit(1)

    EMBED_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(PROCESSED_CSV)
    df["text"] = df["title"].fillna("") + " " + df["description"].fillna("")
    texts = df["text"].tolist()

    print(f"Loading model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    print(f"Encoding {len(texts)} documents...")
    embeddings = model.encode(texts, batch_size=BATCH_SIZE, show_progress_bar=True)

    np.save(EMBED_DIR / "embeddings.npy", embeddings.astype(np.float32))
    joblib.dump(model, EMBED_DIR / "model.joblib")
    metadata = {
        "job_ids": df["job_id"].tolist(),
        "model_name": MODEL_NAME,
        "dim": embeddings.shape[1],
    }
    joblib.dump(metadata, EMBED_DIR / "metadata.joblib")
    df.drop(columns=["text"], errors="ignore").to_csv(EMBED_DIR / "jobs.csv", index=False)

    print(f"Embeddings: {embeddings.shape}")
    print(f"Saved to {EMBED_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
