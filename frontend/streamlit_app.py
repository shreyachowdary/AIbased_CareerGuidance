"""Streamlit UI for AI Career Guidance with data visualizations."""
import io
import os
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

API_URL = os.environ.get("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="AI Career Guidance",
    page_icon="🎯",
    layout="wide",
)

st.title("🎯 AI Career Guidance System")
st.markdown("Get personalized job recommendations and skill-gap analysis with **data visualizations**.")

with st.sidebar:
    st.header("Your Profile")
    skills_text = st.text_area(
        "Skills (comma-separated or free text)",
        placeholder="Python, SQL, machine learning, pandas, AWS...",
        height=100,
    )
    interests = st.text_input("Interests", placeholder="Data science, AI, cloud")
    desired_role = st.text_input("Desired Role", placeholder="Data Scientist")
    education = st.text_input("Education", placeholder="BS Computer Science")

    st.header("Recommendation Settings")
    mode = st.selectbox(
        "Retrieval Mode",
        ["tfidf", "embed", "hybrid"],
        index=2,
        help="TF-IDF: fast, keyword-based. Embed: semantic. Hybrid: combined.",
    )
    alpha = st.slider(
        "Hybrid alpha (TF-IDF weight)",
        0.0,
        1.0,
        0.5,
        0.1,
        help="Only used when mode=hybrid. alpha=1 is pure TF-IDF, 0 is pure embeddings.",
    )
    top_k = st.slider("Top K recommendations", 3, 25, 10)

    st.header("Filters")
    loc_filter = st.text_input("Location contains", placeholder="Remote, NYC...")
    company_filter = st.text_input("Company contains", placeholder="Google...")
    title_filter = st.text_input("Title contains", placeholder="Data, Engineer...")

if st.button("Get Recommendations", type="primary"):
    if not skills_text.strip():
        st.error("Please enter at least your skills.")
    else:
        filters = {}
        if loc_filter.strip():
            filters["location_contains"] = loc_filter.strip()
        if company_filter.strip():
            filters["company_contains"] = company_filter.strip()
        if title_filter.strip():
            filters["title_contains"] = title_filter.strip()

        payload = {
            "skills_text": skills_text,
            "interests": interests or None,
            "desired_role": desired_role or None,
            "education": education or None,
            "top_k": top_k,
            "mode": mode,
            "alpha": alpha,
            "filters": filters if filters else None,
        }

        with st.spinner("Fetching recommendations..."):
            try:
                r = requests.post(f"{API_URL}/recommend", json=payload, timeout=60)
                r.raise_for_status()
                data = r.json()
            except requests.exceptions.ConnectionError:
                st.error(
                    f"Cannot connect to API at {API_URL}. Is the backend running? "
                    "Start with: uvicorn backend.app.main:app --port 8000"
                )
                st.stop()
            except requests.exceptions.RequestException as e:
                st.error(f"API error: {e}")
                if hasattr(e, "response") and e.response is not None:
                    st.code(e.response.text)
                st.stop()

        st.success("Recommendations ready!")

        st.subheader("Extracted Skills")
        st.write(", ".join(data["query_skills"]) if data["query_skills"] else "None detected")

        st.subheader("Ranked Recommendations")
        recs = data["recommendations"]
        if not recs:
            st.info("No recommendations found. Try relaxing filters or adding more skills.")
        else:
            table_data = []
            for i, r in enumerate(recs, 1):
                table_data.append({
                    "#": i,
                    "Title": r["title"][:50] + "..." if len(r["title"]) > 50 else r["title"],
                    "Company": r["company"][:30] + "..." if len(r["company"]) > 30 else r["company"],
                    "Location": r["location"][:20] + "..." if len(r["location"]) > 20 else r["location"],
                    "Score": f"{r['score']:.3f}",
                    "Matched": ", ".join(r["matched_skills"][:3]) or "-",
                    "Missing": ", ".join(r["missing_skills"][:3]) or "-",
                })
            st.dataframe(pd.DataFrame(table_data), use_container_width=True, hide_index=True)

            # ========== DATA VISUALIZATIONS ==========
            st.divider()
            st.subheader("📊 Data Visualizations")

            # 1. Cosine Similarity Bar Chart
            st.markdown("#### Cosine Similarity Scores")
            st.caption("Similarity between your profile and each job (higher = better match)")
            sim_df = pd.DataFrame({
                "Job": [f"{r['title'][:35]}... @ {r['company']}" if len(r["title"]) > 35 else f"{r['title']} @ {r['company']}" for r in recs],
                "Similarity": [r["score"] for r in recs],
            })
            fig_sim = px.bar(
                sim_df, x="Similarity", y="Job", orientation="h",
                color="Similarity", color_continuous_scale="Blues",
                labels={"Similarity": "Cosine Similarity Score"},
            )
            fig_sim.update_layout(
                height=300 + len(recs) * 25,
                margin=dict(l=20, r=20, t=30, b=20),
                xaxis_range=[0, max(1.0, max(r["score"] for r in recs) * 1.1)],
                showlegend=False,
            )
            st.plotly_chart(fig_sim, use_container_width=True)

            # 2. Matched vs Missing Skills per Job
            st.markdown("#### Skill Overlap: Matched vs Missing per Job")
            st.caption("Green = skills you have; Orange = skills to learn")
            job_labels = [f"#{i+1} {r['title'][:25]}..." if len(r["title"]) > 25 else f"#{i+1} {r['title']}" for i, r in enumerate(recs)]
            fig_skills = go.Figure(data=[
                go.Bar(name="Matched Skills", x=job_labels, y=[len(r["matched_skills"]) for r in recs], marker_color="#2ecc71"),
                go.Bar(name="Missing Skills", x=job_labels, y=[len(r["missing_skills"]) for r in recs], marker_color="#e67e22"),
            ])
            fig_skills.update_layout(
                barmode="group",
                height=400,
                margin=dict(l=20, r=20, t=30, b=80),
                xaxis_tickangle=-45,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            st.plotly_chart(fig_skills, use_container_width=True)

            # 3. Skill-Job Heatmap (which skills appear in which jobs)
            all_skills = set()
            for r in recs:
                all_skills.update(r["matched_skills"], r["missing_skills"])
            all_skills = sorted(all_skills)[:20]  # Top 20 skills
            if all_skills:
                st.markdown("#### Skill Presence Heatmap")
                st.caption("Green = matched (you have it), Orange = missing (job needs it), Gray = not in job")
                heatmap_data = []
                for r in recs:
                    row = []
                    for skill in all_skills:
                        if skill in r["matched_skills"]:
                            row.append(1)
                        elif skill in r["missing_skills"]:
                            row.append(-1)
                        else:
                            row.append(0)
                    heatmap_data.append(row)
                fig_heat = go.Figure(data=go.Heatmap(
                    z=heatmap_data,
                    x=all_skills,
                    y=[f"#{i+1} {r['title'][:20]}..." for i, r in enumerate(recs)],
                    colorscale=[[0, "#e67e22"], [0.5, "#ecf0f1"], [1, "#2ecc71"]],  # orange=missing, gray=none, green=matched
                    zmin=-1, zmax=1,
                    hovertemplate="Job: %{y}<br>Skill: %{x}<br>Value: %{z}<extra></extra>",
                ))
                fig_heat.update_layout(
                    height=250 + len(recs) * 30,
                    margin=dict(l=20, r=20, t=30, b=100),
                    xaxis_tickangle=-45,
                )
                st.plotly_chart(fig_heat, use_container_width=True)

            # 4. Action Plan - Top Missing Skills (frequency)
            ap = data["action_plan"]
            if ap["top_missing_skills"]:
                st.markdown("#### Action Plan: Top Missing Skills to Learn")
                st.caption("Skills most frequently required across your top recommendations")
                from collections import Counter
                all_missing = []
                for r in recs:
                    all_missing.extend(r["missing_skills"])
                counts = Counter(all_missing)
                top_skills = counts.most_common(10)
                plan_df = pd.DataFrame({"Skill": [s for s, _ in top_skills], "Frequency": [c for _, c in top_skills]})
                fig_plan = px.bar(
                    plan_df, x="Skill", y="Frequency",
                    color="Frequency", color_continuous_scale="Reds",
                )
                fig_plan.update_layout(
                    height=350,
                    margin=dict(l=20, r=20, t=30, b=80),
                    xaxis_tickangle=-45,
                    showlegend=False,
                )
                st.plotly_chart(fig_plan, use_container_width=True)

            st.divider()
            st.subheader("Explanations")
            for i, r in enumerate(recs, 1):
                with st.expander(f"#{i} {r['title'][:60]}... — {r['company']}"):
                    st.write(r["explanation"])
                    st.caption(f"Matched: {', '.join(r['matched_skills']) or 'None'}")
                    st.caption(f"Missing: {', '.join(r['missing_skills']) or 'None'}")

            st.subheader("Action Plan")
            ap = data["action_plan"]
            st.write("**Top missing skills to learn:**")
            st.write(", ".join(ap["top_missing_skills"]) or "None")
            st.write("**Suggested next steps:**")
            st.info(ap["suggested_next_steps"])

            st.subheader("Download Results")
            download_df = pd.DataFrame([
                {
                    "job_id": r["job_id"],
                    "title": r["title"],
                    "company": r["company"],
                    "location": r["location"],
                    "score": r["score"],
                    "matched_skills": ", ".join(r["matched_skills"]),
                    "missing_skills": ", ".join(r["missing_skills"]),
                    "explanation": r["explanation"],
                }
                for r in recs
            ])
            buf = io.BytesIO()
            download_df.to_csv(buf, index=False)
            buf.seek(0)
            st.download_button(
                "Download results CSV",
                buf,
                file_name="career_recommendations.csv",
                mime="text/csv",
            )
