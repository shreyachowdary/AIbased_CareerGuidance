"""
End-to-end pipeline: data load -> clean -> EDA -> train matching -> persist.
Run this once to prepare artifacts before using the app.
"""

import sys
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import RAW_DATA_DIR, DEFAULT_JOB_CSV
from src.data_ingestion import load_raw_dataset, load_cleaned_dataset, save_cleaned_dataset
from src.data_cleaning import clean_dataset, build_job_text
from src.data_assessment import assess_dataset
from src.eda_visualization import run_full_eda
from src.feature_engineering import build_skill_set_per_job
from src.matching import fit_tfidf_and_transform, save_matching_artifacts
from utils.logging_config import setup_logging, get_logger

setup_logging()
logger = get_logger("pipeline")


def run_pipeline(data_path: Optional[Path] = None) -> None:
    """
    Run full pipeline: load, assess, clean, EDA, train TF-IDF, save artifacts.
    """
    # 1. Load
    df = load_cleaned_dataset()
    if df is None:
        df = load_raw_dataset(path=data_path)
        # 2. Assess
        assessment = assess_dataset(df)
        logger.info("Assessment: %d rows, %d cols", assessment["basic_stats"]["row_count"], assessment["basic_stats"]["column_count"])
        # 3. Clean
        df = clean_dataset(df)
        save_cleaned_dataset(df)
    else:
        logger.info("Using cached cleaned dataset")

    # 4. EDA
    run_full_eda(df)

    # 5. Train matching
    job_texts = build_job_text(df)
    vectorizer, embeddings = fit_tfidf_and_transform(job_texts)
    metadata_cols = [c for c in ["job_id", "title", "company", "location", "job_type", "description"] if c in df.columns]
    metadata = df[metadata_cols].copy() if metadata_cols else df.copy()
    save_matching_artifacts(vectorizer, embeddings, metadata)

    logger.info("Pipeline complete. Artifacts saved.")


if __name__ == "__main__":
    run_pipeline()
