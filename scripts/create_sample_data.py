#!/usr/bin/env python3
"""
Create minimal sample data for testing without Kaggle download.
Run this to test the pipeline without downloading the full dataset.
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

SAMPLE_JOBS = [
    {
        "title": "Software Engineer",
        "description": "Build scalable systems. Python, Java, React, AWS, Docker. "
        "We use microservices and cloud infrastructure." * 3,
        "company": "TechCorp",
        "location": "Remote",
    },
    {
        "title": "Data Scientist",
        "description": "ML models, Python, SQL, pandas, machine learning, PyTorch. "
        "Experience with NLP and deep learning required." * 3,
        "company": "DataCo",
        "location": "NYC",
    },
    {
        "title": "Backend Engineer",
        "description": "Node.js, Python, PostgreSQL, Kubernetes, microservices. "
        "Strong experience with REST APIs and databases." * 3,
        "company": "StartupXYZ",
        "location": "San Francisco",
    },
    # Additional jobs for better demo coverage (USA, Google, Engineer, etc.)
    {
        "title": "Data Engineer",
        "description": "Data pipelines, ETL, Python, SQL, Spark, data engineering. "
        "Build and maintain data infrastructure." * 3,
        "company": "Google",
        "location": "Mountain View, USA",
    },
    {
        "title": "Software Engineer",
        "description": "Python, Java, distributed systems. Google Cloud, Kubernetes. "
        "Large-scale systems engineering." * 3,
        "company": "Google",
        "location": "New York, USA",
    },
    {
        "title": "Machine Learning Engineer",
        "description": "Python, TensorFlow, PyTorch, ML models, data science. "
        "NLP and computer vision experience." * 3,
        "company": "Google",
        "location": "Seattle, USA",
    },
    {
        "title": "Data Engineer",
        "description": "SQL, Python, ETL, data warehousing, data engineering. "
        "Airflow, dbt, cloud platforms." * 3,
        "company": "Amazon",
        "location": "Seattle, USA",
    },
    {
        "title": "Full Stack Engineer",
        "description": "React, Node.js, Python, JavaScript, REST APIs. "
        "Frontend and backend development." * 3,
        "company": "Meta",
        "location": "Menlo Park, USA",
    },
    {
        "title": "DevOps Engineer",
        "description": "AWS, Docker, Kubernetes, CI/CD, Terraform. "
        "Infrastructure and automation." * 3,
        "company": "Microsoft",
        "location": "Redmond, USA",
    },
    {
        "title": "Data Analyst",
        "description": "SQL, Python, pandas, Tableau, analytics. "
        "Data visualization and reporting." * 3,
        "company": "Netflix",
        "location": "Los Gatos, USA",
    },
]


def main():
    import pandas as pd

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = RAW_DIR / "linkedin_jobs_sample.csv"
    df = pd.DataFrame(SAMPLE_JOBS)
    df2 = df.copy()
    df2["title"] = df2["title"].astype(str) + " (alt posting)"
    df = pd.concat([df, df2], ignore_index=True)
    df.to_csv(csv_path, index=False)
    print(f"Created {csv_path} with {len(df)} rows")
    print("Run: python scripts/preprocess.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
