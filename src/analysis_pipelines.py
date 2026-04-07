"""
Dual analysis pipelines: live (API) job rows vs LinkedIn / local job corpus.
Shared by Streamlit and FastAPI — single TF-IDF + skill logic.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from config.settings import ROLE_MARKET_MAX_ROLES, ROLE_MARKET_MIN_POSTINGS, ROLE_MARKET_TOP_SKILLS_PER_ROLE, TOP_N_MATCHES
from src.data_cleaning import clean_dataset, build_job_text
from src.data_ingestion import load_cleaned_dataset, load_raw_dataset, save_cleaned_dataset
from src.feature_engineering import build_skill_set_per_job
from src.job_fetcher import fetch_jobs_for_skills, get_jsearch_api_key
from src.matching import fit_tfidf_and_transform, match_resume_to_jobs_dynamic, match_resume_to_jobs_hybrid
from src.recommendations import augment_recommendations_with_market_role, empty_recommendations, generate_all_recommendations
from src.resume_intent import infer_job_search_queries
from src.role_market_fit import rank_resume_against_market_roles


def build_corpus_artifacts(df: pd.DataFrame) -> Tuple[Any, Any, pd.DataFrame, pd.Series]:
    """Fit TF-IDF on full cleaned corpus; align metadata and per-job skills."""
    texts = build_job_text(df)
    vectorizer, embeddings = fit_tfidf_and_transform(texts)
    want = ["job_id", "title", "company", "location", "job_type", "description"]
    cols = [c for c in want if c in df.columns]
    meta = df[cols].copy() if cols else df.iloc[:, :0].copy()
    for c in want:
        if c not in meta.columns:
            meta[c] = ""
    job_skills = build_skill_set_per_job(df)
    return vectorizer, embeddings, meta, job_skills


LISTING_PREVIEW_MAX_JOBS = 3000


def resolve_corpus_dataframe() -> Optional[pd.DataFrame]:
    """
    Prefer cleaned Parquet/CSV; otherwise load any raw job CSV / LinkedIn export and persist cleaned data.
    """
    df = load_cleaned_dataset()
    if df is not None and len(df) >= 10:
        return df
    try:
        raw = load_raw_dataset()
        df = clean_dataset(raw)
        if len(df) >= 10:
            save_cleaned_dataset(df)
        return df if len(df) >= 10 else None
    except FileNotFoundError:
        return None


def corpus_aggregate_skill_signals(role_fit_df: pd.DataFrame, top_k: int = 5) -> Dict[str, Any]:
    """Roll up matched / missing skills across top ranked corpus roles."""
    matched: set = set()
    missing_counter: Counter = Counter()
    if role_fit_df is None or role_fit_df.empty:
        return {"aggregate_matched_skills": [], "aggregate_missing_skills": [], "roles_used": 0}
    for _, r in role_fit_df.head(top_k).iterrows():
        for s in r.get("your_skills_matching_market") or []:
            matched.add(str(s).strip().lower())
        for s in r.get("market_skills_gap") or []:
            missing_counter[str(s).strip().lower()] += 1
    return {
        "aggregate_matched_skills": sorted(matched),
        "aggregate_missing_skills": [s for s, _ in missing_counter.most_common(35)],
        "roles_used": min(top_k, len(role_fit_df)),
    }


def run_live_listing_preview(
    resume_text: str,
    skills: List[str],
    raw_hint: str,
    corpus_df: pd.DataFrame,
) -> Dict[str, Any]:
    """
    When JSearch is unavailable: per-posting match chart from a **random sample** of local jobs
    (same math as live, different data source — clearly labeled in UI).
    """
    out: Dict[str, Any] = {
        "ok": False,
        "source": "local_snapshot",
        "matches": None,
        "job_skills": None,
        "recommendations": None,
        "error": None,
        "preview_note": "",
    }
    if corpus_df is None or len(corpus_df) < 10:
        out["error"] = "No local job data for a listing preview."
        return out
    try:
        if len(corpus_df) > LISTING_PREVIEW_MAX_JOBS:
            df_p = corpus_df.sample(n=LISTING_PREVIEW_MAX_JOBS, random_state=42)
            note = (
                f"Sample of **{LISTING_PREVIEW_MAX_JOBS:,}** rows from your **{len(corpus_df):,}** local postings "
                "(listing-style match from your file)."
            )
        else:
            df_p = corpus_df
            note = (
                f"**{len(corpus_df):,}** postings from your local file used for this listing column "
                "(same matching pipeline as web listings)."
            )
        matches, _, _, job_skills = match_resume_to_jobs_dynamic(
            resume_text,
            df_p,
            top_n=TOP_N_MATCHES,
            resume_skills=skills,
            raw_text_hint=raw_hint,
        )
        recs = generate_all_recommendations(skills, matches, job_skills)
        out.update({
            "ok": True,
            "matches": matches,
            "job_skills": job_skills,
            "recommendations": recs,
            "preview_note": note,
            "preview_sample_size": len(df_p),
        })
    except Exception as e:
        out["error"] = str(e)
    return out


def run_live_pipeline(
    resume_text: str,
    skills: List[str],
    raw_hint: str,
    profile: Dict[str, Any],
    corpus_df: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """Match resume to **live / API** job postings (JSearch), else optional **local snapshot**."""
    out: Dict[str, Any] = {
        "ok": False,
        "source": "live_api",
        "matches": None,
        "job_skills": None,
        "recommendations": None,
        "error": None,
    }
    if not get_jsearch_api_key():
        out["error"] = "Web listings unavailable — using your local posting file for this column."
        if corpus_df is not None and len(corpus_df) >= 10:
            snap = run_live_listing_preview(resume_text, skills, raw_hint, corpus_df)
            if snap.get("ok"):
                return snap
        return out
    try:
        search_queries = infer_job_search_queries(raw_hint, skills, profile)
        api_jobs = fetch_jobs_for_skills(
            skills,
            queries=search_queries,
            date_posted="month",
            num_pages_per_query=2,
        )
        if api_jobs is None or len(api_jobs) == 0:
            out["error"] = "No live listings returned — try again or broaden your Skills section."
            return out
        matches, _, _, job_skills = match_resume_to_jobs_dynamic(
            resume_text,
            api_jobs,
            top_n=TOP_N_MATCHES,
            resume_skills=skills,
            raw_text_hint=raw_hint,
        )
        recs = generate_all_recommendations(skills, matches, job_skills)
        out.update({
            "ok": True,
            "matches": matches,
            "job_skills": job_skills,
            "recommendations": recs,
            "error": None,
        })
    except Exception as e:
        out["error"] = str(e)
    if not out.get("ok") and corpus_df is not None and len(corpus_df) >= 10:
        snap = run_live_listing_preview(resume_text, skills, raw_hint, corpus_df)
        if snap.get("ok"):
            return snap
    return out


def run_corpus_pipeline(
    resume_text: str,
    skills: List[str],
    df: pd.DataFrame,
    raw_hint: str,
) -> Dict[str, Any]:
    """Batch analysis on **local LinkedIn / postings corpus**: role clustering + fit scores."""
    out: Dict[str, Any] = {
        "ok": False,
        "source": "linkedin_corpus",
        "corpus_rows": len(df),
        "role_market_fit_df": None,
        "role_market_stats": None,
        "recommendations": None,
        "aggregates": None,
        "fallback_matches": None,
        "fallback_job_skills": None,
        "vectorizer": None,
        "embeddings": None,
        "metadata": None,
        "error": None,
    }
    if df is None or len(df) < 10:
        out["error"] = "Corpus too small or missing — run scripts/create_sample_data.py or download_linkedin_2023.py"
        return out
    try:
        vectorizer, embeddings, meta, job_skills = build_corpus_artifacts(df)
        if embeddings.shape[0] != len(meta):
            out["error"] = "Corpus alignment error (text rows vs metadata)."
            return out
        min_per_role = 2 if len(df) < 200 else ROLE_MARKET_MIN_POSTINGS
        role_df, role_stats = rank_resume_against_market_roles(
            resume_text,
            skills,
            vectorizer,
            embeddings,
            meta,
            job_skills,
            min_postings_per_role=min_per_role,
            max_roles=ROLE_MARKET_MAX_ROLES,
            top_market_skills_per_role=ROLE_MARKET_TOP_SKILLS_PER_ROLE,
        )
        corpus_recs = empty_recommendations()
        if role_df is not None and not role_df.empty:
            corpus_recs = augment_recommendations_with_market_role(corpus_recs, role_df.iloc[0])
        aggregates = corpus_aggregate_skill_signals(role_df if role_df is not None else pd.DataFrame())
        fb_matches = match_resume_to_jobs_hybrid(
            resume_text,
            skills,
            vectorizer,
            embeddings,
            meta,
            job_skills_series=job_skills,
            raw_text_for_titles=raw_hint,
            top_n=TOP_N_MATCHES,
        )
        out.update({
            "ok": True,
            "role_market_fit_df": role_df,
            "role_market_stats": role_stats,
            "recommendations": corpus_recs,
            "aggregates": aggregates,
            "fallback_matches": fb_matches,
            "fallback_job_skills": job_skills,
            "vectorizer": vectorizer,
            "embeddings": embeddings,
            "metadata": meta,
        })
    except Exception as e:
        out["error"] = str(e)
    return out
