"""
CareerPath AI - Production-quality career guidance app.
"""

import sys
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import pandas as pd
from config.settings import (
    UPLOADS_DIR,
    TOP_N_MATCHES,
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    CLEANED_CSV,
    CLEANED_PARQUET,
)
from src.analysis_pipelines import resolve_corpus_dataframe, run_corpus_pipeline, run_live_pipeline
from src.data_ingestion import load_raw_dataset, load_cleaned_dataset, save_cleaned_dataset
from src.data_assessment import assess_dataset
from src.data_cleaning import clean_dataset, build_job_text
from src.eda_plotly import (
    plot_role_frequency_plotly,
    plot_top_skills_plotly,
    plot_location_plotly,
    plot_trend_plotly,
    plot_skill_match_gaps_plotly,
    plot_role_market_fit_plotly,
)
from src.feature_engineering import build_skill_set_per_job, is_valid_skill
from src.skill_curation import is_recommendable_skill
from src.resume_parser import (
    compact_company_and_dates,
    is_plausible_work_upload_line,
    parse_resume,
    project_upload_display_line,
)
from src.resume_intent import build_matching_resume_text
from src.matching import (
    fit_tfidf_and_transform,
    match_resume_to_jobs_hybrid,
    match_resume_to_jobs_dynamic,
    save_matching_artifacts,
    load_matching_artifacts,
)
from src.recommendations import empty_recommendations
from src.ats_keywords import build_ats_resume_job_alignment, get_ats_keywords_from_jobs
from src.job_fetcher import fetch_recent_jobs_for_roles, get_jsearch_api_key, set_jsearch_api_key
from src.job_links import job_board_search_links
from src.gap_descriptions import get_gap_description
from src.course_recommendations import get_course_options
from src.roadmap_flowchart_html import build_flowchart_html
from src.graph import Graph
from src.skill_graph import build_skill_cooccurrence_graph, build_learning_dependency_graph
from src.graph_visualization import plot_skill_graph_plotly
from src.password_validation import validate_password
from src.email_service import send_registration_email
from utils.logging_config import setup_logging

setup_logging()


def _skill_gap_frequency_tables(gap_df: pd.DataFrame):
    """Turn per-job gap rows into two sorted tables: skill × how many postings."""
    from collections import Counter

    if gap_df is None or gap_df.empty:
        return pd.DataFrame(), pd.DataFrame()
    mc: Counter = Counter()
    gc: Counter = Counter()
    for _, r in gap_df.iterrows():
        for s in r.get("matched_skills") or []:
            if s and is_valid_skill(str(s)):
                mc[str(s).strip()] += 1
        for s in r.get("missing_skills") or []:
            if s and is_valid_skill(str(s)) and is_recommendable_skill(str(s)):
                gc[str(s).strip()] += 1
    t_m = pd.DataFrame([{"Skill": k, "Postings matched": v} for k, v in mc.most_common(500)])
    t_g = pd.DataFrame([{"Skill": k, "Postings with gap": v} for k, v in gc.most_common(500)])
    return t_m, t_g


st.set_page_config(
    page_title="CareerPath AI | Smart Career Guidance",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed",
)

for key in ["resume_data", "matched_jobs", "recommendations", "job_skills", "job_metadata",
            "vectorizer", "job_embeddings", "jobs_from_api", "profile", "current_tab",
            "recent_24h_jobs", "jobs_display_count", "roadmap_selected",
            "role_market_fit_df", "role_market_stats",
            "live_analysis", "corpus_analysis", "gaps_mode", "roadmap_mode"]:
    if key not in st.session_state:
        st.session_state[key] = None

if st.session_state.jobs_display_count is None:
    st.session_state.jobs_display_count = 25


def ensure_data_loaded():
    artifacts = load_matching_artifacts()
    if artifacts:
        st.session_state.vectorizer, st.session_state.job_embeddings, st.session_state.job_metadata = artifacts
        df = load_cleaned_dataset()
        if df is not None:
            st.session_state.job_skills = build_skill_set_per_job(df)
        return True
    df = load_cleaned_dataset()
    if df is None:
        try:
            df = load_raw_dataset()
            df = clean_dataset(df)
            save_cleaned_dataset(df)
        except FileNotFoundError:
            st.error(
                f"No job data. Run: **python scripts/download_linkedin_2023.py** "
                f"(downloads via kagglehub) — or place a CSV in `{RAW_DATA_DIR}`."
            )
            return False
    job_texts = build_job_text(df)
    vectorizer, embeddings = fit_tfidf_and_transform(job_texts)
    metadata = df[["job_id", "title", "company", "location", "job_type", "description"]].copy() if "job_id" in df.columns else df.copy()
    save_matching_artifacts(vectorizer, embeddings, metadata)
    st.session_state.vectorizer = vectorizer
    st.session_state.job_embeddings = embeddings
    st.session_state.job_metadata = metadata
    st.session_state.job_skills = build_skill_set_per_job(df)
    return True


def render_profile_gate():
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div style="text-align:center; padding:2.5rem; background:linear-gradient(135deg,#667eea 0%,#764ba2 100%); 
        border-radius:16px; color:white; margin-bottom:2rem; box-shadow:0 8px 32px rgba(102,126,234,0.4);">
        <h2 style="margin:0; font-weight:800;">CareerPath AI</h2>
        <p style="margin:0.5rem 0 0 0; opacity:0.95;">Create your profile to get started</p>
        </div>
        """, unsafe_allow_html=True)
        with st.form("profile_form"):
            name = st.text_input("Full Name", placeholder="John Doe")
            email = st.text_input("Email", placeholder="john@example.com")
            password = st.text_input("Password", type="password", placeholder="Min 8 chars, 1 upper, 1 lower, 1 number, 1 special")
            career_goal = st.text_input("Career Goal (optional)", placeholder="e.g. Data Scientist")
            submitted = st.form_submit_button("Create Profile & Continue")
            if submitted:
                if not name or not email:
                    st.error("Please enter your name and email.")
                elif not password:
                    st.error("Please enter a password.")
                else:
                    valid, errors = validate_password(password)
                    if not valid:
                        st.error("Password must have: " + ", ".join(errors))
                    else:
                        st.session_state.profile = {
                            "name": name.strip(),
                            "email": email.strip(),
                            "career_goal": career_goal.strip() or "",
                        }
                        ok, msg = send_registration_email(email.strip(), name.strip())
                        if ok:
                            st.success(f"Profile created! Welcome, {name}. Check your email for a welcome message.")
                        else:
                            st.success(f"Profile created! Welcome, {name}.")
                            st.caption(f"Email: {msg}")
                        st.rerun()


def run_analysis(data):
    profile = st.session_state.profile or {}
    skills = data["skills"] or []
    resume_text = build_matching_resume_text(data, profile)
    raw_hint = data.get("raw_text") or ""

    try:
        df_corpus = resolve_corpus_dataframe()
        with st.spinner("Running **left column**: live API or local listing preview…"):
            live = run_live_pipeline(resume_text, skills, raw_hint, profile, corpus_df=df_corpus)
        st.session_state.live_analysis = live

        with st.spinner("Running **local job postings file** (full dataset, by role)…"):
            corpus = (
                run_corpus_pipeline(resume_text, skills, df_corpus, raw_hint)
                if df_corpus is not None and len(df_corpus) >= 10
                else {
                    "ok": False,
                    "source": "linkedin_corpus",
                    "error": "No job data found. Place a postings CSV under `data/raw/` or run `python scripts/download_linkedin_2023.py` / create sample data.",
                }
            )
        st.session_state.corpus_analysis = corpus

        if corpus.get("ok") and corpus.get("vectorizer") is not None:
            save_matching_artifacts(corpus["vectorizer"], corpus["embeddings"], corpus["metadata"])
            st.session_state.vectorizer = corpus["vectorizer"]
            st.session_state.job_embeddings = corpus["embeddings"]
            st.session_state.job_metadata = corpus["metadata"]
            # Keep per-job skills aligned with the listing column (live/preview), not the full-file series.
            if live.get("ok") and live.get("job_skills") is not None:
                st.session_state.job_skills = live["job_skills"]
            else:
                st.session_state.job_skills = corpus["fallback_job_skills"]

        st.session_state.role_market_fit_df = corpus.get("role_market_fit_df") if corpus.get("ok") else None
        st.session_state.role_market_stats = corpus.get("role_market_stats") if corpus.get("ok") else None

        # Jobs tab: live postings first; else corpus hybrid matches
        if live.get("ok") and live.get("matches") is not None and len(live["matches"]) > 0:
            st.session_state.matched_jobs = live["matches"]
            st.session_state.job_skills = live["job_skills"]
            real_api = live.get("source") == "live_api"
            st.session_state.jobs_from_api = real_api
            if real_api and get_jsearch_api_key():
                top_roles = live["matches"]["title"].unique().tolist()[:3]
                with st.spinner("Fetching today's live listings…"):
                    recent = fetch_recent_jobs_for_roles(top_roles, date_posted="today", num_pages=5)
                if recent is not None and len(recent) > 0:
                    recent_ranked, _, _, _ = match_resume_to_jobs_dynamic(
                        resume_text,
                        recent,
                        top_n=min(120, len(recent)),
                        resume_skills=skills,
                        raw_text_hint=raw_hint,
                    )
                    st.session_state.recent_24h_jobs = recent_ranked
                else:
                    st.session_state.recent_24h_jobs = live["matches"]
            else:
                st.session_state.recent_24h_jobs = live["matches"]
        elif corpus.get("ok") and corpus.get("fallback_matches") is not None:
            st.session_state.matched_jobs = corpus["fallback_matches"]
            st.session_state.job_skills = corpus["fallback_job_skills"]
            st.session_state.jobs_from_api = False
            st.session_state.recent_24h_jobs = None
        elif ensure_data_loaded():
            matches = match_resume_to_jobs_hybrid(
                resume_text,
                skills,
                st.session_state.vectorizer,
                st.session_state.job_embeddings,
                st.session_state.job_metadata,
                job_skills_series=st.session_state.job_skills,
                raw_text_for_titles=raw_hint,
                top_n=TOP_N_MATCHES,
            )
            st.session_state.matched_jobs = matches
            st.session_state.jobs_from_api = False
            st.session_state.recent_24h_jobs = None
        else:
            st.session_state.matched_jobs = pd.DataFrame()
            st.session_state.recent_24h_jobs = None
            st.session_state.jobs_from_api = False

        # Default recommendations / Gaps tab: prefer live, else corpus-only
        if live.get("ok") and live.get("recommendations"):
            st.session_state.recommendations = live["recommendations"]
            st.session_state.gaps_mode = "live"
        elif corpus.get("ok") and corpus.get("recommendations"):
            st.session_state.recommendations = corpus["recommendations"]
            st.session_state.gaps_mode = "corpus"
        else:
            st.session_state.recommendations = empty_recommendations()
            st.session_state.gaps_mode = "none"
        st.session_state.roadmap_mode = st.session_state.gaps_mode

        # ATS vocabulary = skills from **JSearch live API** rows only (not local preview or file).
        if (
            live.get("ok")
            and live.get("source") == "live_api"
            and live.get("job_skills") is not None
        ):
            job_kw = get_ats_keywords_from_jobs(live["job_skills"])
            ats_detail, ats_m = build_ats_resume_job_alignment(
                skills, data.get("raw_text") or "", job_kw
            )
        else:
            ats_detail, ats_m = [], []
        data["ats_matched_detail"] = ats_detail
        data["ats_keywords_found"] = [r["keyword"] for r in ats_detail]
        data["ats_keywords_missing"] = ats_m
        data["profile_summary"]["ats_found_count"] = len(ats_detail)
        data["profile_summary"]["ats_missing_count"] = len(ats_m)

        st.session_state.resume_data = data
        st.session_state.jobs_display_count = 25
        st.session_state.current_tab = "Analysis"
        st.rerun()
    except Exception as e:
        st.error(f"Career analysis failed: {e}")
        import traceback
        st.code(traceback.format_exc())


def inject_styles():
    """Inject CSS and animated background - must run every rerun."""
    CSS_PATH = PROJECT_ROOT / "assets" / "styles.css"
    if CSS_PATH.exists():
        with open(CSS_PATH) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    # Floating orbs - always visible, cozy moving effect
    st.markdown("""
    <div id="floating-orbs">
        <div class="orb orb1"></div>
        <div class="orb orb2"></div>
        <div class="orb orb3"></div>
    </div>
    <style>
    #floating-orbs { position: fixed; top: 0; left: 0; width: 100%; height: 100%; 
        pointer-events: none; z-index: 0; overflow: hidden; }
    .orb { position: absolute; border-radius: 50%; filter: blur(60px); opacity: 0.4; }
    .orb1 { width: 300px; height: 300px; background: #6366f1; top: 10%; left: 10%; 
        animation: float1 12s ease-in-out infinite; }
    .orb2 { width: 250px; height: 250px; background: #0ea5e9; top: 60%; right: 15%; 
        animation: float2 10s ease-in-out infinite; }
    .orb3 { width: 200px; height: 200px; background: #8b5cf6; bottom: 20%; left: 40%; 
        animation: float3 14s ease-in-out infinite; }
    @keyframes float1 { 0%,100%{ transform: translate(0,0) scale(1); } 
        33%{ transform: translate(40px,-30px) scale(1.1); } 
        66%{ transform: translate(-20px,20px) scale(0.95); } }
    @keyframes float2 { 0%,100%{ transform: translate(0,0); } 
        50%{ transform: translate(-50px,-20px); } }
    @keyframes float3 { 0%,100%{ transform: translate(0,0); } 
        33%{ transform: translate(30px,40px); } 
        66%{ transform: translate(-40px,-10px); } }
    </style>
    """, unsafe_allow_html=True)


def main():
    inject_styles()
    if st.session_state.profile is None:
        render_profile_gate()
        return

    profile = st.session_state.profile
    st.sidebar.markdown(f"**{profile['name']}**")
    st.sidebar.caption(profile["email"])

    if st.session_state.get("jsearch_rapidapi_key"):
        set_jsearch_api_key(st.session_state["jsearch_rapidapi_key"])
    else:
        set_jsearch_api_key(None)

    with st.sidebar.expander("Web job listings (JSearch)", expanded=False):
        st.caption("Optional RapidAPI key for live postings. Otherwise listing match uses your local file only.")
        nk = st.text_input(
            "RapidAPI key",
            type="password",
            key="jsearch_rapidapi_input",
            autocomplete="off",
            label_visibility="collapsed",
            placeholder="Paste key → Apply",
        )
        b1, b2 = st.columns(2)
        with b1:
            if st.button("Apply", key="jsearch_apply_btn", use_container_width=True):
                st.session_state.jsearch_rapidapi_key = (nk or "").strip()
                st.rerun()
        with b2:
            if st.session_state.get("jsearch_rapidapi_key") and st.button(
                "Forget key", key="jsearch_forget_btn", use_container_width=True
            ):
                st.session_state.jsearch_rapidapi_key = ""
                st.rerun()
        if get_jsearch_api_key():
            st.caption("Status: web listings **enabled**.")
        else:
            st.caption("Status: **local posting file** only.")

    # Show "Download LinkedIn 2023" when no job data exists
    has_data = (
        (PROCESSED_DATA_DIR / CLEANED_PARQUET).exists()
        or (PROCESSED_DATA_DIR / CLEANED_CSV).exists()
        or (RAW_DATA_DIR / "linkedin_jobs_2023.csv").exists()
    )
    if not has_data:
        st.sidebar.markdown("---")
        if st.sidebar.button("📥 Download LinkedIn 2023 dataset", help="Fetch job data via kagglehub (requires pip install kagglehub)"):
            with st.spinner("Downloading dataset via kagglehub…"):
                try:
                    result = subprocess.run(
                        [sys.executable, str(PROJECT_ROOT / "scripts" / "download_linkedin_2023.py")],
                        cwd=str(PROJECT_ROOT),
                        capture_output=True,
                        text=True,
                        timeout=300,
                    )
                    if result.returncode == 0:
                        st.sidebar.success("Downloaded! Refreshing…")
                        st.rerun()
                    else:
                        err = result.stderr or result.stdout or "Unknown error"
                        st.sidebar.error(f"Download failed: {err[:500]}")
                except subprocess.TimeoutExpired:
                    st.sidebar.error("Download timed out (5 min). Try running: python scripts/download_linkedin_2023.py")
                except FileNotFoundError:
                    st.sidebar.error("kagglehub not found. Run: pip install kagglehub")
                except Exception as e:
                    st.sidebar.error(str(e))
        if st.sidebar.button("Create small demo dataset (offline)", help="~10 sample jobs in data/raw — no Kaggle"):
            with st.spinner("Creating sample CSV…"):
                try:
                    r = subprocess.run(
                        [sys.executable, str(PROJECT_ROOT / "scripts" / "create_sample_data.py")],
                        cwd=str(PROJECT_ROOT),
                        capture_output=True,
                        text=True,
                        timeout=60,
                    )
                    if r.returncode == 0:
                        st.sidebar.success("Demo CSV created. Run **Career Analysis** again.")
                        st.rerun()
                    else:
                        st.sidebar.error((r.stderr or r.stdout or "Error")[:400])
                except Exception as e:
                    st.sidebar.error(str(e))

    if st.sidebar.button("Sign Out"):
        st.session_state.profile = None
        st.session_state.current_tab = "Home"
        st.rerun()

    tab_options = ["Home", "Upload", "Analysis", "Jobs", "Gaps & Courses", "Roadmap", "Graph", "Insights"]
    current = st.session_state.current_tab or "Home"
    if current in ("Gaps", "Courses"):
        current = "Gaps & Courses"
        st.session_state.current_tab = current
    if current not in tab_options:
        current = "Home"

    cols = st.columns(8)
    for col, opt in zip(cols, tab_options):
        with col:
            if st.button(opt, key=f"nav_{opt}", use_container_width=True, type="primary" if opt == current else "secondary"):
                st.session_state.current_tab = opt
                st.rerun()

    st.markdown("---")

    if current == "Home":
        render_home()
    elif current == "Upload":
        render_upload()
    elif current == "Analysis":
        render_analysis()
    elif current == "Jobs":
        render_matches()
    elif current == "Gaps & Courses":
        render_gaps_and_courses()
    elif current == "Roadmap":
        render_roadmap()
    elif current == "Graph":
        render_graph()
    else:
        render_insights()


def render_home():
    name = st.session_state.profile.get("name", "there")
    st.markdown("""
    <div class="home-hero">
        <h1 class="hero-title">CareerPath AI</h1>
        <p class="hero-sub">Your intelligent career companion — match, learn, grow.</p>
        <p class="hero-greet">Hello, <strong>{}</strong>! Ready to level up your career?</p>
    </div>
    """.format(name), unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("### What CareerPath AI Does")
    st.markdown("""
    **CareerPath AI** analyzes your resume and connects you with your next opportunity. Upload your resume, 
    and we'll match you to real jobs, identify skill gaps, recommend courses, and give you a day-by-day 
    learning roadmap for every skill you need.
    """)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("📄 Upload", "Resume", "PDF, DOCX, TXT")
    with c2:
        st.metric("🎯 Match", "Real Jobs", "24h postings")
    with c3:
        st.metric("📚 Learn", "Courses", "Curated links")
    with c4:
        st.metric("🗺️ Roadmap", "All Skills", "Day-by-day flow")
    st.markdown("---")
    st.markdown("### How It Works")
    st.markdown("""
    1. **Upload** — Add your resume (PDF, DOCX, or TXT)
    2. **Analyze** — Click *Run Career Analysis* to match your skills to jobs
    3. **Explore** — Jobs, **Gaps & Courses** (per-skill learning picks), Roadmaps, and market insights
    """)
    if st.button("Get Started → Upload Resume", type="primary", use_container_width=True):
        st.session_state.current_tab = "Upload"
        st.rerun()


def render_upload():
    st.markdown("## Upload Resume")
    uploaded = st.file_uploader("Choose a resume file", type=["pdf", "docx", "txt"], label_visibility="collapsed")

    # Show Run Career Analysis when resume is parsed (from current upload or previous session)
    if st.session_state.resume_data:
        data = st.session_state.resume_data
        if st.button("Run Career Analysis", type="primary", use_container_width=True, key="run_analysis_btn"):
            run_analysis(data)

    if uploaded:
        UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
        path = UPLOADS_DIR / uploaded.name
        with open(path, "wb") as f:
            f.write(uploaded.getvalue())

        with st.spinner("Parsing resume..."):
            try:
                data = parse_resume(path)
                st.session_state.resume_data = data
                st.success("Resume parsed successfully!")
            except Exception as e:
                st.error(f"Failed: {e}")
                return

    if st.session_state.resume_data:
        data = st.session_state.resume_data

        st.markdown("#### Skills (from your resume)")
        skills_list = data.get("skills") or []
        if skills_list:
            st.markdown(", ".join(f"`{s}`" for s in skills_list[:80]) + (" …" if len(skills_list) > 80 else ""))
        else:
            st.info("No skills block detected — use a **Skills** heading with comma-, pipe-, or bullet-separated items.")

        st.markdown("#### Education")
        for e in data["education"]:
            d = e.get("degree", "") or ""
            inst = e.get("institution", "") or ""
            y = e.get("year", "") or ""
            gpa = e.get("gpa", "") or ""
            bits = []
            if d:
                bits.append(f"**{d}**")
            if inst:
                bits.append(f"*{inst}*")
            if y:
                bits.append(y)
            if gpa:
                bits.append(gpa)
            if bits:
                st.markdown(" · ".join(bits))

        st.markdown("#### Work experience")
        if not data.get("profile_summary", {}).get("has_work_experience", True):
            st.caption("No **Experience** section header was detected — only clear **Company · dates** rows are listed above; use **Raw extracted text** for full detail.")
        for ex in data["experience"]:
            co, dt = compact_company_and_dates(ex)
            if is_plausible_work_upload_line(co, dt, ex):
                st.markdown(f"{co} · {dt}")

        st.markdown("#### Projects")
        projects = data.get("projects") or []
        if projects:
            for p in projects:
                line = project_upload_display_line(p)
                if line:
                    st.markdown(line)
        else:
            st.caption("No **Projects** (or similar) section heading was found — see **Raw extracted text** if you list projects elsewhere.")

        with st.expander("Raw extracted text (for your reference)"):
            st.text_area("Resume text", value=(data.get("raw_text") or "")[:120_000], height=240, disabled=True, label_visibility="collapsed")

        st.divider()


def render_analysis():
    if st.session_state.resume_data is None:
        st.info("Upload a resume and run analysis first.")
        return

    live = st.session_state.live_analysis or {}
    corpus = st.session_state.corpus_analysis or {}
    if not live.get("ok") and not corpus.get("ok"):
        st.info("Click **Run Career Analysis** on the Upload tab.")
        return

    data = st.session_state.resume_data
    recs = st.session_state.recommendations
    role_fit = st.session_state.role_market_fit_df
    role_stats = st.session_state.role_market_stats or {}

    st.markdown("## Compare: **Live listings** vs **Local job postings**")

    col_live, col_corpus = st.columns(2)

    with col_live:
        st.markdown("### Live listings")
        if live.get("ok"):
            n_live = len(live["matches"]) if live.get("matches") is not None else 0
            if live.get("source") == "live_api":
                st.success(f"**{n_live}** matched **JSearch** postings")
            else:
                st.info(
                    live.get("preview_note")
                    or f"**{n_live}** matches from your **local posting file** (same matching pipeline as web listings)."
                )
            live_recs = live.get("recommendations") or {}
            gap_live = live_recs.get("skill_gaps_df")
            if gap_live is not None and not gap_live.empty:
                st.plotly_chart(plot_skill_match_gaps_plotly(gap_live), use_container_width=True)
                tbl_m, tbl_g = _skill_gap_frequency_tables(gap_live)
                ctab1, ctab2 = st.columns(2)
                with ctab1:
                    st.markdown("**Matched skills (by posting)**")
                    if not tbl_m.empty:
                        st.dataframe(tbl_m, use_container_width=True, hide_index=True)
                    else:
                        st.caption("No matched skills in this set.")
                with ctab2:
                    st.markdown("**Skill gaps (by posting)**")
                    if not tbl_g.empty:
                        st.dataframe(tbl_g, use_container_width=True, hide_index=True)
                    else:
                        st.caption("No gaps in this set.")
        else:
            st.warning(
                live.get("error")
                or "No listing run — add a postings CSV under `data/raw/` or use **Web job listings** in the sidebar."
            )

    with col_corpus:
        st.markdown("### Local job postings (by role)")
        n_corpus = role_stats.get("corpus_job_count", 0) or corpus.get("corpus_rows", 0)
        if corpus.get("ok") and role_fit is not None and not role_fit.empty:
            st.success(
                f"**{corpus.get('corpus_rows', 0):,}** rows from your file · **{n_corpus:,}** in the TF‑IDF index"
            )
            st.plotly_chart(plot_role_market_fit_plotly(role_fit, top_n=12), use_container_width=True)
            top = role_fit.iloc[0]
            st.info(
                f"**Strongest role vs your resume:** **{top.get('role_display', '—')}** — "
                f"{int(top.get('postings_in_corpus', 0))} postings · score **{float(top.get('fit_score', 0)):.3f}**"
            )
            agg = corpus.get("aggregates") or {}
            am = agg.get("aggregate_matched_skills") or []
            ax = agg.get("aggregate_missing_skills") or []
            g1, g2 = st.columns(2)
            with g1:
                st.markdown("**Your skills that appear in top roles**")
                if am:
                    st.dataframe(pd.DataFrame({"Skill": am}), use_container_width=True, hide_index=True)
                else:
                    st.caption("—")
            with g2:
                st.markdown("**Market gaps in those roles**")
                if ax:
                    st.dataframe(pd.DataFrame({"Skill": ax}), use_container_width=True, hide_index=True)
                else:
                    st.caption("—")
            with st.expander("Role ranking (tabular)", expanded=False):
                show_cols = [
                    c
                    for c in (
                        "role_display",
                        "postings_in_corpus",
                        "fit_score",
                        "text_similarity",
                        "market_skill_overlap",
                        "your_skills_matching_market",
                        "market_skills_gap",
                    )
                    if c in role_fit.columns
                ]
                st.dataframe(role_fit.head(25)[show_cols], use_container_width=True, hide_index=True)
        else:
            st.warning(
                corpus.get("error")
                or "Load a postings CSV (or run `python scripts/download_linkedin_2023.py`), then **Run Career Analysis**."
            )

    st.markdown("## ATS keywords (JSearch live postings only)")
    ats_detail = data.get("ats_matched_detail") or []
    missing_kw = data.get("ats_keywords_missing") or []
    live_src = (live or {}).get("source")
    if live_src != "live_api":
        st.info(
            "ATS vocabulary comes from **web** job postings only. Run analysis after enabling "
            "**Web job listings** in the sidebar, or ignore this section when using the file-only path."
        )
    elif not ats_detail and not missing_kw:
        st.info(
            "No skill vocabulary was extracted from live matches (or none returned). "
            "Try broader **Skills** on your resume or rerun."
        )
    else:
        if ats_detail:
            st.markdown("### In your resume (also common on live postings)")
            st.dataframe(
                pd.DataFrame(ats_detail).rename(
                    columns={"keyword": "Keyword", "evidence": "How it shows up"}
                ),
                use_container_width=True,
                hide_index=True,
            )
        if missing_kw:
            st.markdown("### Not in your resume (often on those live postings)")
            st.caption("Add only skills you honestly have.")
            st.dataframe(
                pd.DataFrame({"Keyword not in resume": missing_kw}),
                use_container_width=True,
                hide_index=True,
            )


def render_matches():
    if st.session_state.matched_jobs is None:
        st.info("Run Career Analysis first.")
        return

    live = st.session_state.live_analysis or {}
    if st.session_state.jobs_from_api:
        st.caption("**Source:** live listings (JSearch). Per-row gaps from **live** analysis.")
    elif live.get("source") == "local_snapshot":
        st.caption("**Source:** local **listing preview** (sample of your job file). Gaps reflect that sample, not the web.")
    else:
        st.caption("**Source:** LinkedIn / local corpus hybrid. Matches are TF‑IDF picks from your downloaded dataset.")

    jobs_to_show = st.session_state.recent_24h_jobs if st.session_state.recent_24h_jobs is not None else st.session_state.matched_jobs

    recs = st.session_state.recommendations
    live = st.session_state.live_analysis or {}
    if live.get("ok") and live.get("recommendations"):
        _raw_gaps = live["recommendations"].get("skill_gaps_df")
    else:
        _raw_gaps = (recs or {}).get("skill_gaps_df")
    # Never use `df or pd.DataFrame()` — empty DataFrame is falsy and triggers ambiguous truth-value errors.
    gap_df = _raw_gaps if isinstance(_raw_gaps, pd.DataFrame) else pd.DataFrame()
    display_count = st.session_state.jobs_display_count
    jobs_slice = jobs_to_show.head(display_count)

    for i, row in jobs_slice.iterrows():
        jidx = row.get("job_index", i)
        gap_row = gap_df[gap_df["job_index"] == jidx] if not gap_df.empty else pd.DataFrame()
        g = gap_row.iloc[0] if not gap_row.empty else {}
        st.markdown(f"### {row.get('title', 'N/A')}")
        meta_bits = []
        jid = row.get("job_id")
        if jid is not None and str(jid).strip():
            meta_bits.append(f"ID: `{jid}`")
        pub = row.get("job_publisher") or ""
        if pub:
            meta_bits.append(str(pub))
        posted = row.get("posted_date") or ""
        if posted:
            meta_bits.append(str(posted)[:19])
        score = row.get('match_score')
        cap = f"**{row.get('company', 'N/A')}** · {row.get('location', 'N/A')}"
        if pd.notna(score):
            cap += f" · Match: **{score:.0%}**"
        if meta_bits:
            cap += " · " + " · ".join(meta_bits)
        st.caption(cap)
        if not gap_row.empty:
            matched = [s for s in (g.get('matched_skills', []) or []) if is_valid_skill(str(s))]
            missing = [s for s in (g.get('missing_skills', []) or []) if is_valid_skill(str(s))]
            st.markdown(f"**Matched:** {', '.join(matched[:8]) or '—'}")
            st.markdown(f"**Gaps:** {', '.join(missing[:8]) or '—'}")
        apply_url = (row.get("apply_link") or "").strip()
        if not apply_url:
            apply_url = (row.get("job_google_link") or "").strip()
        if apply_url.lower().startswith("http"):
            st.link_button("Apply / view listing →", apply_url, type="primary")
        else:
            st.caption(
                "No direct apply URL in this row (typical for offline sample data or sparse API fields). "
                "Use the job-board searches below — queries use **title**, **company**, and **location**."
            )
        st.markdown("**Find this job on:**")
        q_title = str(row.get("title") or "").strip()
        q_company = str(row.get("company") or "").strip()
        q_loc = str(row.get("location") or "").strip()
        brd = job_board_search_links(q_title, q_company, q_loc)
        bc1, bc2, bc3, bc4 = st.columns(4)
        for (label, url), col in zip(brd, (bc1, bc2, bc3, bc4)):
            with col:
                st.link_button(label, url, use_container_width=True)
        with st.expander("Description"):
            st.write(row.get("description", "N/A"))
        st.divider()

    if len(jobs_to_show) > display_count:
        if st.button("Load More Jobs"):
            st.session_state.jobs_display_count = min(display_count + 25, len(jobs_to_show))
            st.rerun()


def render_gaps_and_courses():
    live = st.session_state.live_analysis or {}
    corpus = st.session_state.corpus_analysis or {}
    if not live.get("ok") and not corpus.get("ok"):
        st.info("Run Career Analysis first.")
        return

    st.markdown("## Skill gaps and courses")
    view = st.radio(
        "Base recommendations on",
        ["Live listings (JSearch or local preview)", "Local job file — by role"],
        horizontal=True,
        key="gaps_view_mode",
    )
    use_corpus = view.startswith("Local")
    if use_corpus:
        recs_hdr = corpus.get("recommendations") if corpus.get("ok") else None
    else:
        recs_hdr = live.get("recommendations") if live.get("ok") else None
    if not recs_hdr:
        st.warning("No recommendation data for this mode — run analysis with job data or enable web listings.")
        return

    focus = (recs_hdr.get("market_role_focus") or "").strip()
    src = recs_hdr.get("gaps_source") or ""
    npost = int(recs_hdr.get("market_role_postings") or 0)
    if use_corpus and focus and src == "corpus_best_role":
        st.info(
            f"**Local file mode:** gaps prioritize **{focus}** ({npost} postings in your dataset), then other gaps."
        )

    mwf = recs_hdr.get("missing_with_freq") or []
    if mwf:
        freq_df = pd.DataFrame(mwf, columns=["Skill", "Weight (postings)"])
        st.markdown("### Prioritized gaps (tabular)")
        st.dataframe(freq_df, use_container_width=True, hide_index=True)

    missing = [
        (s, f)
        for s, f in recs_hdr.get("missing_with_freq", [])[:25]
        if is_valid_skill(str(s)) and is_recommendable_skill(str(s))
    ]
    mkt_sk = recs_hdr.get("market_prioritized_skills") or set()
    for skill, freq in missing:
        desc = get_gap_description(skill)
        from_market = str(skill).strip().lower() in mkt_sk
        tag = (
            "local file — top role"
            if use_corpus and from_market
            else ("listing frequency" if not use_corpus else "local file / blended")
        )
        with st.expander(f"**{skill}** — weight **{freq}** ({tag})"):
            st.markdown(f"**What it is:** {desc['what']}")
            st.markdown(f"**Why it matters:** {desc['why']}")
            st.markdown(f"**Impact:** {desc['impact']}")
            st.markdown("**Courses to close this gap:**")
            options = get_course_options(skill)
            if not options:
                st.caption("No curated links — try a web search for this skill plus “course” or “certification”.")
                continue
            best = next((o for o in options if o.get("is_best")), options[0] if options else None)
            if best:
                st.markdown(f"- **⭐ Best pick:** [{best['name']}]({best['url']}) — _{best['platform']}_")
            for o in options:
                if o != best:
                    st.markdown(f"- [{o['name']}]({o['url']}) — _{o['platform']}_")


def _safe_key(s: str) -> str:
    """Sanitize string for Streamlit widget key - alphanumeric only."""
    import re
    return re.sub(r"[^a-zA-Z0-9_]", "_", str(s))[:50]


def render_roadmap():
    live = st.session_state.live_analysis or {}
    corpus = st.session_state.corpus_analysis or {}
    if not live.get("ok") and not corpus.get("ok"):
        st.info("Run Career Analysis first.")
        return

    try:
        import streamlit.components.v1 as components
        view = st.radio(
            "Roadmap based on",
            ["Live listings (JSearch or local preview)", "Local job file — by role"],
            horizontal=True,
            key="roadmap_view_mode",
        )
        use_corpus = view.startswith("Local")
        recs_src = (
            (corpus.get("recommendations") if corpus.get("ok") else None)
            if use_corpus
            else (live.get("recommendations") if live.get("ok") else None)
        )
        if not recs_src:
            st.warning("No roadmap for this mode.")
            return
        roadmap = recs_src.get("learning_roadmap") or []
        if not roadmap:
            st.info("No learning roadmap for this mode.")
            return

        st.markdown("## Your Learning Roadmap — All Skills")
        st.markdown("Day-by-day flowcharts for every skill you need to build. Focus on the main topics in order.")
        for i, item in enumerate(roadmap[:12]):
            skill = item.get("skill", "Skill")
            if not is_valid_skill(str(skill)) or not is_recommendable_skill(str(skill)):
                continue
            priority = item.get("priority", i + 1)
            freq = item.get("frequency", 0)
            st.markdown(f"### {priority}. {skill} — in {freq} target roles")
            components.html(build_flowchart_html(str(skill)), height=320, scrolling=False)
            st.markdown("**Courses:**")
            options = get_course_options(str(skill))
            for o in options[:3]:
                badge = " ⭐" if o.get("is_best") else ""
                st.markdown(f"- [{o['name']}{badge}]({o['url']}) — {o['platform']}")
            st.divider()
    except Exception as e:
        st.error(f"Roadmap error: {e}")
        import traceback
        st.code(traceback.format_exc())


def render_insights():
    try:
        if not ensure_data_loaded():
            return
        df = load_cleaned_dataset()
        if df is None:
            df = load_raw_dataset()
            df = clean_dataset(df)
        if df is None or len(df) == 0:
            st.warning("No job data available for insights.")
            return
        st.markdown("### Job Market Analytics")
        st.plotly_chart(plot_role_frequency_plotly(df), use_container_width=True)
        st.plotly_chart(plot_top_skills_plotly(df), use_container_width=True)
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(plot_location_plotly(df), use_container_width=True)
        with c2:
            st.plotly_chart(plot_trend_plotly(df), use_container_width=True)
        with st.expander("Data Summary"):
            st.json(assess_dataset(df))
    except Exception as e:
        st.error(f"Could not load insights: {e}")
        st.info("Ensure job data exists in data/raw/ (or enable web listings in the sidebar).")


def render_graph():
    """Graph implementations: skill co-occurrence network, BFS, DFS, shortest path."""
    try:
        if not ensure_data_loaded():
            return
        df = load_cleaned_dataset()
        if df is None:
            df = load_raw_dataset()
            df = clean_dataset(df)
        if df is None or len(df) == 0:
            st.warning("No job data for graph. Add data in data/raw/.")
            return

        st.markdown("## Graph Implementation")
        st.markdown("""
        **Internal graph data structure** (adjacency list) with **BFS**, **DFS**, and **shortest path** algorithms.
        Skills are nodes; edges = skills that co-occur in the same job.
        """)

        g, freq = build_skill_cooccurrence_graph(df, min_cooccurrence=2, top_skills=40)
        nodes = g.nodes()

        if not nodes:
            st.info("Not enough skill co-occurrence data for a graph.")
            return

        # Metrics
        c1, c2, c3 = st.columns(3)
        c1.metric("Nodes (Skills)", g.num_nodes())
        c2.metric("Edges (Co-occurrences)", g.num_edges())
        start_node = nodes[0] if nodes else ""
        c3.metric("Sample BFS from", start_node[:20] + "..." if len(start_node) > 20 else start_node)

        # Highlight user skills if available
        highlight = []
        if st.session_state.resume_data:
            highlight = st.session_state.resume_data.get("skills", [])[:15]

        st.markdown("### Skill Network Graph")
        fig = plot_skill_graph_plotly(g, highlight_skills=highlight, title="Skills as Nodes, Co-occurrence as Edges")
        st.plotly_chart(fig, use_container_width=True)

        # Algorithm demos
        st.markdown("### Graph Algorithms")
        with st.expander("BFS (Breadth-First Search)"):
            if start_node:
                bfs_result = g.bfs(start_node)
                st.markdown(f"**BFS from '{start_node}'** visits {len(bfs_result)} nodes in level order:")
                st.code(", ".join(bfs_result[:15]) + (" ..." if len(bfs_result) > 15 else ""))

        with st.expander("DFS (Depth-First Search)"):
            if start_node:
                dfs_result = g.dfs(start_node)
                st.markdown(f"**DFS from '{start_node}'** visits {len(dfs_result)} nodes:")
                st.code(", ".join(dfs_result[:15]) + (" ..." if len(dfs_result) > 15 else ""))

        with st.expander("Shortest Path (BFS-based)"):
            if len(nodes) >= 2:
                end_node = nodes[min(1, len(nodes) - 1)]
                path = g.shortest_path_bfs(start_node, end_node)
                if path:
                    st.markdown(f"**Shortest path** from '{start_node}' to '{end_node}':")
                    st.code(" → ".join(path))
                else:
                    st.markdown("No path found (disconnected components).")
    except Exception as e:
        st.error(f"Graph error: {e}")
        import traceback
        st.code(traceback.format_exc())


if __name__ == "__main__":
    main()
