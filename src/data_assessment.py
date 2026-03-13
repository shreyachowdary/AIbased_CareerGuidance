"""
Data assessment module: schema summary, missing values, duplicates, basic stats.
"""

from typing import Any

import pandas as pd

from utils.logging_config import get_logger

logger = get_logger("data_assessment")


def assess_dataset(df: pd.DataFrame) -> dict[str, Any]:
    """
    Perform full dataset assessment.

    Args:
        df: Raw or cleaned DataFrame.

    Returns:
        Dictionary with schema_summary, missing_report, duplicates, basic_stats.
    """
    schema = _schema_summary(df)
    missing = _missing_report(df)
    dup_info = _duplicate_report(df)
    stats = _basic_stats(df)
    return {
        "schema_summary": schema,
        "missing_report": missing,
        "duplicates": dup_info,
        "basic_stats": stats,
    }


def _schema_summary(df: pd.DataFrame) -> dict[str, Any]:
    """Produce schema summary: columns, dtypes, sample values."""
    summary = []
    for col in df.columns:
        dtype = str(df[col].dtype)
        non_null = df[col].notna().sum()
        sample = df[col].dropna().head(1).iloc[0] if non_null > 0 else None
        summary.append({
            "column": col,
            "dtype": dtype,
            "non_null_count": int(non_null),
            "sample_value": str(sample)[:80] if sample is not None else None,
        })
    return {"columns": summary, "total_columns": len(df.columns)}


def _missing_report(df: pd.DataFrame) -> dict[str, Any]:
    """Report missing values per column and overall."""
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    report = []
    for col in df.columns:
        cnt = int(missing[col])
        pct = float(missing_pct[col])
        if cnt > 0:
            report.append({"column": col, "missing_count": cnt, "missing_pct": pct})
    return {
        "per_column": report,
        "total_rows": len(df),
        "columns_with_missing": len(report),
    }


def _duplicate_report(df: pd.DataFrame) -> dict[str, Any]:
    """Report duplicate rows and key-based duplicates if applicable."""
    dup_rows = df.duplicated(keep=False)
    n_dup = int(dup_rows.sum())
    n_unique_dup = int(df.duplicated(keep="first").sum()) if n_dup > 0 else 0

    # If we have job_id, check duplicates by job_id
    job_id_dup = {}
    if "job_id" in df.columns and df["job_id"].notna().any():
        id_dup = df["job_id"].duplicated(keep=False)
        job_id_dup = {
            "duplicate_count": int(id_dup.sum()),
            "unique_duplicate_ids": int(df.loc[id_dup, "job_id"].nunique()),
        }
    elif "title" in df.columns:
        # Fallback: duplicates by title+company
        key_cols = [c for c in ["title", "company"] if c in df.columns]
        if key_cols:
            key_dup = df.duplicated(subset=key_cols, keep=False)
            job_id_dup = {"duplicate_by_key_count": int(key_dup.sum())}

    return {
        "total_duplicate_rows": n_dup,
        "unique_duplicate_groups": n_unique_dup,
        "job_id_duplicates": job_id_dup,
    }


def _basic_stats(df: pd.DataFrame) -> dict[str, Any]:
    """Basic statistics: row count, memory, numeric summaries."""
    stats: dict[str, Any] = {
        "row_count": len(df),
        "column_count": len(df.columns),
        "memory_mb": round(df.memory_usage(deep=True).sum() / 1024 / 1024, 2),
    }
    numeric = df.select_dtypes(include=["number"])
    if not numeric.empty:
        stats["numeric_columns"] = list(numeric.columns)
        stats["numeric_summary"] = numeric.describe().to_dict()
    return stats
