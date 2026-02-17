# Data Directory

Place your job postings CSV in this folder as `jobs.csv`.

## Expected CSV format

| Column     | Required | Description                          |
|-----------|----------|--------------------------------------|
| job_id    | Yes      | Unique identifier                    |
| title     | Yes      | Job title                            |
| company   | No       | Company name                         |
| description | Yes    | Full job description                 |
| skills    | No       | Comma-separated skills (extracted from description if missing) |

If `skills` is missing, skills are extracted from the `description` during preprocessing.
