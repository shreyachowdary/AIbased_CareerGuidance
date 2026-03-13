"""Load processed jobs dataset."""
import pandas as pd

from backend.app.core.config import PROCESSED_CSV


def load_jobs() -> pd.DataFrame:
    """Load processed jobs CSV."""
    return pd.read_csv(PROCESSED_CSV)
