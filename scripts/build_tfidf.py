#!/usr/bin/env python3
"""
Build TF-IDF model on title + description.
Saves vectorizer, matrix, and jobs dataframe to models/tfidf/.
"""
import os
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_CSV = PROJECT_ROOT / "data" / "processed" / "jobs.csv"
TFIDF_DIR = PROJECT_ROOT / "models" / "tfidf"

MIN_DF = int(os.environ.get("TFIDF_MIN_DF", "2"))
NGRAM_RANGE = tuple(int(x) for x in os.environ.get("TFIDF_NGRAM_RANGE", "1,2").split(","))


def main():
    if not PROCESSED_CSV.exists():
        print(f"ERROR: {PROCESSED_CSV} not found. Run scripts/preprocess.py first.")
        sys.exit(1)

    TFIDF_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(PROCESSED_CSV)
    df["text"] = df["title"].fillna("") + " " + df["description"].fillna("")
    texts = df["text"].tolist()

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=NGRAM_RANGE,
        min_df=MIN_DF,
        max_features=50000,
        sublinear_tf=True,
    )
    matrix = vectorizer.fit_transform(texts)

    joblib.dump(vectorizer, TFIDF_DIR / "vectorizer.joblib")
    joblib.dump(matrix, TFIDF_DIR / "matrix.joblib")
    df.drop(columns=["text"], errors="ignore").to_csv(TFIDF_DIR / "jobs.csv", index=False)

    print(f"TF-IDF built: {matrix.shape[0]} docs, {matrix.shape[1]} features")
    print(f"Saved to {TFIDF_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
