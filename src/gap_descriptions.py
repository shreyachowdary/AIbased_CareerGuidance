"""
Detailed skill gap descriptions - what each gap means and why it matters.
"""

from typing import Dict, Optional

# Curated descriptions for common skills - what the gap means, why it matters
GAP_CONTEXT: Dict[str, dict] = {
    "artificial intelligence": {
        "what": "AI and machine learning fundamentals",
        "why": "Critical for modern tech roles. AI is transforming every industry.",
        "impact": "High — key differentiator across roles",
    },
    "ai": {
        "what": "Artificial Intelligence",
        "why": "Essential for data science, product, and engineering. High demand skill.",
        "impact": "High — increasingly required",
    },
    "python": {
        "what": "Python programming language",
        "why": "Core skill for data science, ML, automation, and backend development. Most job postings require it.",
        "impact": "High — blocks many technical roles",
    },
    "sql": {
        "what": "Structured Query Language for databases",
        "why": "Essential for data retrieval, analysis, and reporting. Used across analytics, engineering, and product roles.",
        "impact": "High — required in most data roles",
    },
    "machine learning": {
        "what": "ML algorithms and model building",
        "why": "Core competency for data scientists and ML engineers. Enables predictive modeling and AI applications.",
        "impact": "High — key differentiator for DS/ML roles",
    },
    "aws": {
        "what": "Amazon Web Services cloud platform",
        "why": "Leading cloud provider. Used for deployment, scaling, and infrastructure. Required for many cloud/DevOps roles.",
        "impact": "Medium-High — increasingly expected",
    },
    "react": {
        "what": "React.js frontend framework",
        "why": "Most popular frontend framework. Used for building web UIs. Essential for frontend/full-stack roles.",
        "impact": "High — required for frontend roles",
    },
    "docker": {
        "what": "Containerization platform",
        "why": "Standard for packaging and deploying applications. Used in DevOps and modern software development.",
        "impact": "Medium-High — common in modern stacks",
    },
    "tensorflow": {
        "what": "Deep learning framework",
        "why": "Industry-standard for ML model deployment. Used in production ML systems.",
        "impact": "Medium — important for ML/DS roles",
    },
    "pytorch": {
        "what": "Deep learning framework",
        "why": "Preferred for research and many production ML systems. Growing in industry adoption.",
        "impact": "Medium — important for ML/DS roles",
    },
    "tableau": {
        "what": "Data visualization tool",
        "why": "Widely used for dashboards and business intelligence. Common in analytics and reporting roles.",
        "impact": "Medium — common in analytics roles",
    },
    "spark": {
        "what": "Apache Spark for big data processing",
        "why": "Standard for large-scale data processing. Used in data engineering pipelines.",
        "impact": "Medium — important for data engineering",
    },
    "kubernetes": {
        "what": "Container orchestration platform",
        "why": "Industry standard for container management. Essential for DevOps and cloud-native roles.",
        "impact": "Medium-High — increasingly required",
    },
    "excel": {
        "what": "Spreadsheet and data analysis",
        "why": "Ubiquitous for basic analytics, reporting, and modeling. Expected in many business roles.",
        "impact": "Medium — common baseline",
    },
}


def get_gap_description(skill: str) -> dict:
    """Get detailed description for a skill gap. Returns default if not found."""
    key = skill.lower().strip()
    if key in GAP_CONTEXT:
        return GAP_CONTEXT[key]
    return {
        "what": skill,
        "why": "This skill appears in your target roles. Building it will improve your match.",
        "impact": "Medium — recommended to add",
    }
