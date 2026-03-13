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
from urllib.parse import quote_plus

from config.settings import UPLOADS_DIR, TOP_N_MATCHES, RAW_DATA_DIR, PROCESSED_DATA_DIR, CLEANED_CSV, CLEANED_PARQUET
from src.data_ingestion import load_raw_dataset, load_cleaned_dataset, save_cleaned_dataset
from src.data_assessment import assess_dataset
from src.data_cleaning import clean_dataset, build_job_text
from src.eda_plotly import (
    plot_role_frequency_plotly,
    plot_top_skills_plotly,
    plot_location_plotly,
    plot_trend_plotly,
    plot_skill_match_gaps_plotly,
)
from src.feature_engineering import build_skill_set_per_job, is_valid_skill
from src.skill_curation import is_recommendable_skill
from src.resume_parser import parse_resume
from src.matching import (
    fit_tfidf_and_transform,
    match_resume_to_jobs,
    match_resume_to_jobs_dynamic,
    save_matching_artifacts,
    load_matching_artifacts,
)
from src.recommendations import generate_all_recommendations
from src.ats_keywords import extract_ats_keywords_from_resume, get_ats_keywords_from_jobs
from src.job_fetcher import fetch_jobs_for_skills, fetch_recent_jobs_for_roles, JSEARCH_API_KEY
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

st.set_page_config(
    page_title="CareerPath AI | Smart Career Guidance",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed",
)

for key in ["resume_data", "matched_jobs", "recommendations", "job_skills", "job_metadata",
            "vectorizer", "job_embeddings", "jobs_from_api", "profile", "current_tab",
            "recent_24h_jobs", "jobs_display_count", "roadmap_selected"]:
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
                f"(downloads via kagglehub) — or place a CSV in `{RAW_DATA_DIR}` or add JSEARCH_API_KEY."
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
    resume_text = data["raw_text"] + " " + " ".join(data["skills"])
    skills = data["skills"] or []

    try:
        if JSEARCH_API_KEY:
            with st.spinner("Fetching jobs and matching..."):
                api_jobs = fetch_jobs_for_skills(skills, date_posted="month", num_pages_per_query=2)
            if api_jobs is not None and len(api_jobs) > 0:
                matches, _, _, job_skills = match_resume_to_jobs_dynamic(resume_text, api_jobs, top_n=TOP_N_MATCHES)
                st.session_state.matched_jobs = matches
                st.session_state.job_skills = job_skills
                st.session_state.jobs_from_api = True
                top_roles = matches["title"].unique().tolist()[:3]
                with st.spinner("Fetching last 24h jobs for your top roles..."):
                    recent = fetch_recent_jobs_for_roles(top_roles, date_posted="today", num_pages=5)
                if recent is not None and len(recent) > 0:
                    st.session_state.recent_24h_jobs = recent
                else:
                    st.session_state.recent_24h_jobs = matches
            else:
                if not ensure_data_loaded():
                    return
                matches = match_resume_to_jobs(resume_text, st.session_state.vectorizer,
                    st.session_state.job_embeddings, st.session_state.job_metadata, top_n=TOP_N_MATCHES)
                st.session_state.matched_jobs = matches
                st.session_state.jobs_from_api = False
                st.session_state.recent_24h_jobs = None
        else:
            if not ensure_data_loaded():
                return
            matches = match_resume_to_jobs(resume_text, st.session_state.vectorizer,
                st.session_state.job_embeddings, st.session_state.job_metadata, top_n=TOP_N_MATCHES)
            st.session_state.matched_jobs = matches
            st.session_state.jobs_from_api = False
            st.session_state.recent_24h_jobs = None

        recs = generate_all_recommendations(skills, matches, st.session_state.job_skills)
        st.session_state.recommendations = recs
        job_kw = get_ats_keywords_from_jobs(st.session_state.job_skills)
        ats_f, ats_m = extract_ats_keywords_from_resume(skills, data["raw_text"], job_kw)
        data["ats_keywords_found"] = ats_f
        data["ats_keywords_missing"] = ats_m
        data["profile_summary"]["ats_found_count"] = len(ats_f)
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

    if st.sidebar.button("Sign Out"):
        st.session_state.profile = None
        st.session_state.current_tab = "Home"
        st.rerun()

    tab_options = ["Home", "Upload", "Analysis", "Jobs", "Gaps", "Courses", "Roadmap", "Graph", "Insights"]
    current = st.session_state.current_tab or "Home"
    if current not in tab_options:
        current = "Home"

    cols = st.columns(9)
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
    elif current == "Gaps":
        render_gaps()
    elif current == "Courses":
        render_courses()
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
    3. **Explore** — See Jobs, Skill Gaps, Courses, and Roadmaps for every skill
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

        ps = data["profile_summary"]
        cols = st.columns(5)
        cols[0].metric("Skills", ps["skills_count"])
        cols[1].metric("Education", ps["education_count"])
        cols[2].metric("Experience", ps["experience_count"])
        cols[3].metric("ATS Found", ps.get("ats_found_count", 0))
        cols[4].metric("ATS to Add", ps.get("ats_missing_count", 0))

        st.markdown("#### ATS Keywords")
        c1, c2 = st.columns(2)
        c1.markdown("**✓ In resume:** " + ", ".join(data.get("ats_keywords_found", [])[:25]) or "—")
        c2.markdown("**+ Add:** " + ", ".join(data.get("ats_keywords_missing", [])[:25]) or "—")

        st.markdown("#### Education")
        for e in data["education"]:
            d, i, y = e.get("degree", ""), e.get("institution", ""), e.get("year", "")
            if d or i or y:
                st.markdown(f"**{d or '—'}** · *{i or '—'}* · {y or '—'}")

        st.markdown("#### Experience")
        for ex in data["experience"]:
            title = ex.get("job_title", "")
            company = ex.get("company", "")
            dates = ex.get("dates", "")
            bullets = ex.get("bullets", [])
            if title or company:
                st.markdown(f"**{title or '—'}** at *{company or '—'}* · {dates or '—'}")
                for b in bullets[:3]:
                    st.markdown(f"- {b}")
            else:
                st.write(ex.get("text", ""))

        st.divider()


def render_analysis():
    if st.session_state.resume_data is None:
        st.info("Upload a resume and run analysis first.")
        return
    if st.session_state.recommendations is None:
        st.info("Click **Run Career Analysis** on the Upload tab.")
        return

    recs = st.session_state.recommendations
    gap_df = recs["skill_gaps_df"]
    fig = plot_skill_match_gaps_plotly(gap_df)
    st.plotly_chart(fig, use_container_width=True)
    m1, m2 = st.columns(2)
    m1.metric("Matched Skills", int(gap_df["matched_skills"].apply(len).sum()))
    m2.metric("Missing Skills", int(sum(len(m) for m in gap_df["missing_skills"])))


def render_matches():
    if st.session_state.matched_jobs is None:
        st.info("Run Career Analysis first.")
        return

    jobs_to_show = st.session_state.recent_24h_jobs if st.session_state.recent_24h_jobs is not None else st.session_state.matched_jobs

    recs = st.session_state.recommendations
    gap_df = recs["skill_gaps_df"] if recs else pd.DataFrame()
    display_count = st.session_state.jobs_display_count
    jobs_slice = jobs_to_show.head(display_count)

    for i, row in jobs_slice.iterrows():
        jidx = row.get("job_index", i)
        gap_row = gap_df[gap_df["job_index"] == jidx] if not gap_df.empty else pd.DataFrame()
        g = gap_row.iloc[0] if not gap_row.empty else {}
        st.markdown(f"### {row.get('title', 'N/A')}")
        score = row.get('match_score')
        if pd.notna(score):
            st.caption(f"**{row.get('company', 'N/A')}** · {row.get('location', 'N/A')} · Match: **{score:.0%}**")
        else:
            st.caption(f"**{row.get('company', 'N/A')}** · {row.get('location', 'N/A')}")
        if not gap_row.empty:
            matched = [s for s in (g.get('matched_skills', []) or []) if is_valid_skill(str(s))]
            missing = [s for s in (g.get('missing_skills', []) or []) if is_valid_skill(str(s))]
            st.markdown(f"**Matched:** {', '.join(matched[:8]) or '—'}")
            st.markdown(f"**Gaps:** {', '.join(missing[:8]) or '—'}")
        apply_url = row.get("apply_link") or ""
        if not apply_url:
            q = quote_plus(f"{row.get('company', '')} {row.get('title', '')} jobs")
            apply_url = f"https://www.google.com/search?q={q}"
        st.link_button("View job & apply →", apply_url, type="primary")
        with st.expander("Description"):
            st.write(row.get("description", "N/A"))
        st.divider()

    if len(jobs_to_show) > display_count:
        if st.button("Load More Jobs"):
            st.session_state.jobs_display_count = min(display_count + 25, len(jobs_to_show))
            st.rerun()


def render_gaps():
    if st.session_state.recommendations is None:
        st.info("Run Career Analysis first.")
        return

    missing = [(s, f) for s, f in st.session_state.recommendations["missing_with_freq"][:25] if is_valid_skill(str(s)) and is_recommendable_skill(str(s))]
    for skill, freq in missing:
        desc = get_gap_description(skill)
        with st.expander(f"**{skill}** — appears in {freq} target roles"):
            st.markdown(f"**What it is:** {desc['what']}")
            st.markdown(f"**Why it matters:** {desc['why']}")
            st.markdown(f"**Impact:** {desc['impact']}")


def render_courses():
    if st.session_state.recommendations is None:
        st.info("Run Career Analysis first.")
        return

    recs = st.session_state.recommendations["course_recommendations"]
    if not recs:
        recs = [{"skill": s} for s in st.session_state.recommendations.get("missing_skills", [])[:10] if is_valid_skill(str(s)) and is_recommendable_skill(str(s))]

    for r in recs[:15]:
        skill = r.get("skill", "")
        if not is_valid_skill(str(skill)) or not is_recommendable_skill(str(skill)):
            continue
        options = get_course_options(skill)
        st.markdown(f"### {skill}")
        best = next((o for o in options if o.get("is_best")), options[0] if options else None)
        if best:
            st.markdown(f"**⭐ Best:** [{best['name']}]({best['url']}) on {best['platform']}")
        for o in options:
            if o != best:
                st.markdown(f"- [{o['name']}]({o['url']}) — {o['platform']}")
        st.divider()


def _safe_key(s: str) -> str:
    """Sanitize string for Streamlit widget key - alphanumeric only."""
    import re
    return re.sub(r"[^a-zA-Z0-9_]", "_", str(s))[:50]


def render_roadmap():
    if st.session_state.recommendations is None:
        st.info("Run Career Analysis first.")
        return

    try:
        import streamlit.components.v1 as components
        roadmap = st.session_state.recommendations.get("learning_roadmap", [])
        if not roadmap:
            st.info("No learning roadmap yet. Run Career Analysis first.")
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
        st.info("Ensure job data exists in data/raw/ or add JSEARCH_API_KEY.")


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
            st.warning("No job data for graph. Add data in data/raw/ or use JSEARCH_API_KEY.")
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
