"""
Download LinkedIn Job 2023 dataset via kagglehub and prepare for CareerPath AI.
Run once: python scripts/download_linkedin_2023.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_cleaning import clean_dataset
from src.data_ingestion import linkedin_loader_sample_frac, save_cleaned_dataset
from src.linkedin_2023_loader import load_linkedin_2023


def main():
    print("Downloading LinkedIn Job 2023 dataset via kagglehub...")
    try:
        import kagglehub
        path = kagglehub.dataset_download("rajatraj0502/linkedin-job-2023")
        print(f"Dataset path: {path}")
    except ImportError:
        print("Install kagglehub: pip install kagglehub")
        sys.exit(1)

    # Full corpus by default (100% of postings). Dev sample: LINKEDIN_SAMPLE_FRAC=0.1
    df_raw = load_linkedin_2023(path, sample_frac=linkedin_loader_sample_frac())
    frac = linkedin_loader_sample_frac()
    print(f"Loaded {len(df_raw):,} jobs (sample_frac={frac!r} — full corpus when None)")

    # Save raw merged CSV first (for load_raw_dataset fallback)
    raw_path = PROJECT_ROOT / "data" / "raw" / "linkedin_jobs_2023.csv"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    df_raw.to_csv(raw_path, index=False)
    print(f"Saved raw to {raw_path}")

    # Clean and save to processed (app uses this automatically)
    df = clean_dataset(df_raw)
    save_cleaned_dataset(df)
    print("Saved cleaned to data/processed/ - app will use this automatically.")


if __name__ == "__main__":
    main()
