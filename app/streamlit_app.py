"""
Streamlit UI for AI Career Guidance System.
Inputs: education, skills, interests, desired role -> Get Recommendations.
Output: table of top jobs, similarity score, matched skills, missing skills.
"""
import streamlit as st
import requests

# Backend URL (default when running locally)
BACKEND_URL = "http://127.0.0.1:8000"


def main() -> None:
    st.set_page_config(
        page_title="AI Career Guidance",
        page_icon="🎯",
        layout="centered",
    )
    st.title("🎯 AI Career Guidance System")
    st.caption("MVP: Get job recommendations and skill-gap analysis from your profile.")

    education = st.text_input("Education", placeholder="e.g. B.S. Computer Science")
    skills = st.text_area(
        "Skills (comma-separated)",
        placeholder="e.g. Python, SQL, JavaScript, Git",
        height=80,
    )
    interests = st.text_input("Interests", placeholder="e.g. backend development, data")
    desired_role = st.text_input("Desired role", placeholder="e.g. Software Engineer, Data Scientist")
    top_n = st.slider("Number of recommendations", min_value=1, max_value=20, value=10)

    if st.button("Get Recommendations"):
        with st.spinner("Fetching recommendations..."):
            payload = {
                "skills": skills.strip() or None,
                "education": education.strip() or None,
                "interests": interests.strip() or None,
                "desired_role": desired_role.strip() or None,
            }
            try:
                r = requests.post(
                    f"{BACKEND_URL}/recommend",
                    json=payload,
                    params={"top_n": top_n},
                    timeout=5,
                )
                r.raise_for_status()
                data = r.json()
            except requests.exceptions.ConnectionError:
                st.error(
                    "Could not connect to the backend. Start it with: "
                    "`uvicorn backend.main:app --reload` from the project root."
                )
                return
            except requests.exceptions.RequestException as e:
                st.error(f"Request failed: {e}")
                return

        recs = data.get("recommendations") or []
        message = data.get("message")

        if message:
            st.info(message)

        if not recs:
            st.warning("No recommendations returned. Ensure you have run `scripts/build_vectors.py` and the backend has loaded the models.")
            return

        st.subheader("Top job recommendations")
        for i, job in enumerate(recs, 1):
            with st.expander(
                f"**{i}. {job.get('title', 'N/A')}** — Score: {job.get('similarity_score', 0):.2f}"
                + (f" @ {job.get('company')}" if job.get("company") else ""),
                expanded=(i == 1),
            ):
                st.write(f"**Job ID:** {job.get('job_id')}")
                matched = job.get("matched_skills") or []
                missing = job.get("missing_skills") or []
                st.write("**Matched skills:** " + (", ".join(matched) if matched else "—"))
                st.write("**Missing skills:** " + (", ".join(missing) if missing else "—"))

        # Table view
        st.subheader("Summary table")
        table_data = []
        for j in recs:
            table_data.append({
                "Title": j.get("title"),
                "Company": j.get("company") or "—",
                "Score": j.get("similarity_score"),
                "Matched skills": ", ".join(j.get("matched_skills") or []),
                "Missing skills": ", ".join(j.get("missing_skills") or []),
            })
        st.dataframe(table_data, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
