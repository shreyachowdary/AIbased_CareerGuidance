"""Dual analysis API: live listings + LinkedIn corpus."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from backend.app.analysis.linkedin_corpus_analyzer import analyze_corpus, load_corpus_dataframe
from backend.app.analysis.live_job_analyzer import analyze_live
from backend.app.analysis.resume_parser import parse_resume_path
from src.resume_intent import build_matching_resume_text

router = APIRouter(prefix="/analysis", tags=["dual-analysis"])


def _df_to_jsonable(df: Optional[pd.DataFrame]) -> Any:
    if df is None:
        return None
    if df.empty:
        return []
    return json.loads(df.replace({np.nan: None}).to_json(orient="records", date_format="iso"))


def _sanitize_live_payload(d: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(d)
    if out.get("matches") is not None:
        out["matches"] = _df_to_jsonable(out["matches"])
    rec = out.get("recommendations")
    if isinstance(rec, dict) and rec.get("skill_gaps_df") is not None:
        rec = dict(rec)
        rec["skill_gaps_df"] = _df_to_jsonable(rec["skill_gaps_df"])
        out["recommendations"] = rec
    if "job_skills" in out:
        del out["job_skills"]
    return out


def _sanitize_corpus_payload(d: Dict[str, Any]) -> Dict[str, Any]:
    out = {k: v for k, v in d.items() if k not in ("vectorizer", "embeddings", "metadata", "fallback_job_skills")}
    rdf = out.get("role_market_fit_df")
    out["role_market_fit_df"] = _df_to_jsonable(rdf) if rdf is not None else None
    rec = out.get("recommendations")
    if isinstance(rec, dict):
        rec = dict(rec)
        rec["skill_gaps_df"] = _df_to_jsonable(rec.get("skill_gaps_df"))
        mps = rec.get("market_prioritized_skills")
        if isinstance(mps, set):
            rec["market_prioritized_skills"] = list(mps)
        out["recommendations"] = rec
    return out


@router.get("/corpus-status")
def corpus_status():
    from config.settings import CLEANED_CSV, CLEANED_PARQUET, PROCESSED_DATA_DIR

    df = load_corpus_dataframe()
    if df is None:
        return {
            "loaded": False,
            "rows": 0,
            "message": "No cleaned dataset in data/processed/. Run scripts/download_linkedin_2023.py",
        }
    return {
        "loaded": True,
        "rows": len(df),
        "parquet_exists": (PROCESSED_DATA_DIR / CLEANED_PARQUET).exists(),
        "csv_exists": (PROCESSED_DATA_DIR / CLEANED_CSV).exists(),
        "linkedin_sample_frac_env": os.environ.get("LINKEDIN_SAMPLE_FRAC", ""),
    }


@router.post("/dual")
async def dual_resume_analysis(
    resume: UploadFile = File(..., description="Resume PDF, DOCX, or TXT"),
    career_goal: str = Form(""),
    name: str = Form(""),
):
    """
    Run **both** pipelines on one upload: live (JSearch) + full local corpus.
    """
    suf = Path(resume.filename or "resume.pdf").suffix.lower()
    if suf not in (".pdf", ".docx", ".doc", ".txt"):
        raise HTTPException(400, "Unsupported file type — use PDF, DOCX, or TXT.")

    raw = await resume.read()
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suf)
    tmp.write(raw)
    tmp.flush()
    tmp.close()
    path = Path(tmp.name)

    try:
        data = parse_resume_path(path)
        skills = list(data.get("skills") or [])
        raw_text = data.get("raw_text") or ""
        profile = {"name": name or "Candidate", "career_goal": career_goal, "email": ""}
        resume_text = build_matching_resume_text(data, profile)

        live = analyze_live(resume_text, skills, raw_text, profile)
        df = load_corpus_dataframe()
        corpus = (
            analyze_corpus(resume_text, skills, raw_text, df=df)
            if df is not None and len(df) >= 10
            else {
                "ok": False,
                "source": "linkedin_corpus",
                "error": "Corpus unavailable or too small — run download_linkedin_2023.py",
            }
        )

        return {
            "resume_skills_sample": skills[:40],
            "live": _sanitize_live_payload(live),
            "corpus": _sanitize_corpus_payload(corpus),
        }
    finally:
        path.unlink(missing_ok=True)
