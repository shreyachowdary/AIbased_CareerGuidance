"""Application configuration."""
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_CSV = DATA_DIR / "processed" / "jobs.csv"
SKILLS_VOCAB = DATA_DIR / "skills_vocab.txt"
TFIDF_DIR = PROJECT_ROOT / "models" / "tfidf"
EMBED_DIR = PROJECT_ROOT / "models" / "embeddings"

DEFAULT_TOP_K = 10
DEFAULT_ALPHA = 0.5
DEFAULT_MODE = "hybrid"
