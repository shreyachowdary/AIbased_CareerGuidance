# CareerPath AI — Smart Career Guidance

A production-quality app that matches your resume to **real jobs** (LinkedIn, Indeed, Glassdoor), performs skill-gap analysis, and generates actionable learning recommendations with direct links to Coursera, Udemy, LinkedIn Learning, and more.

## Features

- **Real job data**: JSearch API fetches recent jobs from LinkedIn, Indeed, Glassdoor (optional API key)
- **Resume parsing**: PDF, DOCX, TXT — extracts skills, education, experience, ATS keywords
- **Job matching**: TF-IDF cosine similarity; Apply links for real jobs
- **Skill-gap analysis**: Matched vs missing skills per role
- **Courses**: Direct links to Coursera, Udemy, LinkedIn Learning, edX
- **Market Insights**: Interactive Plotly charts — roles, skills, locations, trends

## Project Structure

```
ProjectWorking/
├── config/           # Settings, paths, column mappings
├── src/              # Core logic
│   ├── data_ingestion.py
│   ├── data_assessment.py
│   ├── data_cleaning.py
│   ├── feature_engineering.py
│   ├── eda_visualization.py
│   ├── resume_parser.py
│   ├── matching.py
│   ├── skill_gap.py
│   ├── recommendations.py
│   └── pipeline.py
├── utils/            # Logging, helpers
├── data/
│   ├── raw/          # Place CSV here (sample included)
│   └── processed/    # Cached cleaned data
├── artifacts/        # Pickle: vectorizer, embeddings, metadata
├── uploads/          # Uploaded resumes
├── notebooks/        # Optional EDA notebooks
├── app.py            # Streamlit multi-page app
├── requirements.txt
└── README.md
```

## Setup

```bash
cd ProjectWorking
pip install -r requirements.txt
```

### Real jobs (optional)

1. Sign up at [RapidAPI JSearch](https://rapidapi.com/letscrape-6bRBa3QguO5/api/jsearch)
2. Copy your API key
3. Create `.env` with: `JSEARCH_API_KEY=your_key`

Without the API key, the app uses the sample dataset in `data/raw/sample_jobs.csv`.

## Run

**Option 1: Run app directly** (loads and processes data on first use)

```bash
streamlit run app.py
```

**Option 2: Pre-run pipeline** (recommended for large datasets)

```bash
python -m src.pipeline
streamlit run app.py
```

## App Pages

1. **Upload & Profile** — Upload resume (PDF/DOCX/TXT), view parsed profile
2. **Resume Analysis** — Bar chart of skill match vs gaps per role
3. **Top Role Matches** — Best-fit roles with per-role detail (why matched, required skills, gaps)
4. **Skill Gaps** — Aggregated missing skills with frequency
5. **Courses & Certifications** — Google search links for each skill (no hardcoded results)
6. **Learning Roadmap** — Prioritized learning plan
7. **EDA & Data Overview** — Schema, missing values, duplicates, plots

## Persistence

- **Cleaned data**: `data/processed/cleaned_jobs.csv` and `.parquet`
- **Artifacts**: `artifacts/tfidf_vectorizer.pkl`, `job_embeddings.pkl`, `job_metadata.pkl`
- **EDA plots**: `artifacts/eda_*.png`

## License
The project belongs to Shreya Chennupati
Any ochanges or contributions are welcomed. 
