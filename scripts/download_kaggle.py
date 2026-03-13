#!/usr/bin/env python3
"""
Download LinkedIn Job 2023 dataset from Kaggle.
Verifies Kaggle auth, downloads to data/raw, idempotent.
"""
import os
import subprocess
import sys
from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
KAGGLE_DATASET = "rajatraj0502/linkedin-job-2023"


def get_kaggle_json_path() -> Path:
    """Return platform-appropriate path to kaggle.json."""
    home = Path.home()
    if sys.platform == "win32":
        return home / ".kaggle" / "kaggle.json"
    return home / ".kaggle" / "kaggle.json"


def verify_kaggle_auth() -> bool:
    """Check if kaggle.json exists and is valid."""
    kaggle_path = get_kaggle_json_path()
    if not kaggle_path.exists():
        return False
    if kaggle_path.stat().st_size == 0:
        return False
    return True


def print_kaggle_auth_error():
    """Print clear instructions for Kaggle auth setup."""
    kaggle_path = get_kaggle_json_path()
    print("=" * 60)
    print("ERROR: Kaggle authentication not found.")
    print("=" * 60)
    print()
    print("Kaggle requires API credentials to download datasets.")
    print()
    print("Steps to set up:")
    print("  1. Go to https://www.kaggle.com/")
    print("  2. Log in to your account")
    print("  3. Click your profile icon → Account")
    print("  4. Scroll to 'API' section → 'Create New API Token'")
    print("  5. This downloads kaggle.json")
    print()
    print("  6. Create the directory:")
    print(f"     mkdir -p {kaggle_path.parent}")
    print()
    print("  7. Move kaggle.json to:")
    print(f"     {kaggle_path}")
    print()
    print("  8. Set permissions (Linux/macOS):")
    print("     chmod 600 ~/.kaggle/kaggle.json")
    print()
    print("  On Windows: place kaggle.json in:")
    print("     C:\\Users\\<YourUsername>\\.kaggle\\kaggle.json")
    print()
    print("Do NOT proceed without valid kaggle.json.")
    print("=" * 60)


def detect_csv_files() -> list[Path]:
    """Find CSV files in data/raw."""
    if not RAW_DIR.exists():
        return []
    return sorted(RAW_DIR.glob("*.csv"))


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    if not verify_kaggle_auth():
        print_kaggle_auth_error()
        sys.exit(1)

    # Check if already downloaded (idempotent)
    csv_files = detect_csv_files()
    if csv_files:
        print(f"Dataset already present in {RAW_DIR}")
        print("CSV file(s) found:")
        for f in csv_files:
            print(f"  - {f.name} ({f.stat().st_size:,} bytes)")
        print()
        print("Will use for preprocessing:")
        # Prefer largest CSV or first one
        best = max(csv_files, key=lambda p: p.stat().st_size)
        print(f"  {best.name}")
        return 0

    print(f"Downloading dataset: {KAGGLE_DATASET}")
    print(f"Destination: {RAW_DIR}")
    print()

    cmd = [
        "kaggle", "datasets", "download",
        "-d", KAGGLE_DATASET,
        "-p", str(RAW_DIR),
        "--unzip"
    ]

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Kaggle download failed: {e}")
        if e.stderr:
            print(e.stderr)
        sys.exit(1)
    except FileNotFoundError:
        print("ERROR: 'kaggle' command not found.")
        print("Install with: pip install kaggle")
        sys.exit(1)

    csv_files = detect_csv_files()
    if csv_files:
        best = max(csv_files, key=lambda p: p.stat().st_size)
        print(f"\nDownload complete. CSV for preprocessing: {best.name}")
    else:
        print("\nDownload completed but no CSV files found. Check data/raw/")

    return 0


if __name__ == "__main__":
    sys.exit(main())
