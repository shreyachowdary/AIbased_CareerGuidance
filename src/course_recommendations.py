"""
Multiple course options per skill with best recommendation.
Uses platform search URLs to avoid 404s - users land on search results.
"""

from typing import List
from urllib.parse import quote_plus


def _platform_search_url(skill: str, platform: str) -> str:
    """Build reliable search URL - never 404s."""
    q = quote_plus(skill)
    urls = {
        "coursera": f"https://www.coursera.org/search?query={q}",
        "udemy": f"https://www.udemy.com/courses/search/?q={q}",
        "linkedin": f"https://www.linkedin.com/learning/search?keywords={q}",
        "edx": f"https://www.edx.org/search?q={q}",
        "datacamp": f"https://www.datacamp.com/search?q={q}",
        "google": f"https://www.google.com/search?q={quote_plus(skill + ' certification course')}",
    }
    return urls.get(platform.lower(), urls["google"])


# Curated: skill -> list of (platform, course_name, search_query, is_best)
# Prominent skills: AI, AWS, Python, etc. — no rubbish
COURSE_OPTIONS: dict = {
    "artificial intelligence": [
        ("Coursera", "AI for Everyone", "artificial intelligence", True),
        ("Coursera", "Machine Learning", "machine learning", False),
        ("edX", "AI Fundamentals", "artificial intelligence", False),
    ],
    "ai": [
        ("Coursera", "AI for Everyone", "artificial intelligence", True),
        ("Google", "Google AI Essentials", "google ai", False),
        ("LinkedIn", "AI courses", "artificial intelligence", False),
    ],
    "aws": [
        ("Coursera", "AWS Fundamentals", "aws", True),
        ("Udemy", "AWS Certified Solutions Architect", "aws certification", False),
        ("AWS", "AWS Training", "aws training", False),
    ],
    "python": [
        ("Coursera", "Python courses", "python", True),
        ("Udemy", "Python Bootcamp", "python", False),
        ("edX", "Python programs", "python", False),
        ("LinkedIn", "Python Essential Training", "python", False),
    ],
    "sql": [
        ("Coursera", "SQL for Data Science", "sql", True),
        ("Udemy", "SQL Bootcamp", "sql", False),
        ("DataCamp", "Introduction to SQL", "sql", False),
    ],
    "machine learning": [
        ("Coursera", "Machine Learning", "machine learning", True),
        ("Coursera", "Deep Learning", "deep learning", False),
        ("Fast.ai", "Practical Deep Learning", "deep learning fast.ai", False),
    ],
    "aws": [
        ("Coursera", "AWS Fundamentals", "aws", True),
        ("Udemy", "AWS Certified Solutions Architect", "aws certification", False),
        ("AWS", "AWS Training", "aws training", False),
    ],
    "react": [
        ("React", "Official React Tutorial", "react", True),
        ("Udemy", "React Complete Guide", "react", False),
        ("Scrimba", "Learn React", "react", False),
    ],
    "docker": [
        ("Docker", "Docker Getting Started", "docker", True),
        ("Udemy", "Docker Mastery", "docker", False),
        ("Pluralsight", "Docker", "docker", False),
    ],
    "tensorflow": [
        ("Coursera", "TensorFlow", "tensorflow", True),
        ("TensorFlow", "TensorFlow Tutorials", "tensorflow", False),
    ],
    "pytorch": [
        ("Udemy", "PyTorch Deep Learning", "pytorch", True),
        ("PyTorch", "PyTorch Tutorials", "pytorch", False),
    ],
    "tableau": [
        ("Coursera", "Data Visualization", "tableau", True),
        ("Tableau", "Tableau Training", "tableau", False),
        ("Udemy", "Tableau", "tableau", False),
    ],
    "spark": [
        ("Databricks", "Spark", "apache spark", True),
        ("Udemy", "Apache Spark", "spark", False),
    ],
    "kubernetes": [
        ("Coursera", "Kubernetes", "kubernetes", True),
        ("Udemy", "Kubernetes", "kubernetes", False),
    ],
    "excel": [
        ("Coursera", "Excel Skills", "excel", True),
        ("LinkedIn", "Excel Essential Training", "excel", False),
    ],
    "engineering": [
        ("Coursera", "Engineering courses", "engineering", True),
        ("edX", "Engineering programs", "engineering", False),
    ],
    "information technology": [
        ("Coursera", "IT Fundamentals", "information technology", True),
        ("Udemy", "IT courses", "it certification", False),
    ],
    "management": [
        ("Coursera", "Management courses", "management", True),
        ("LinkedIn", "Leadership & Management", "management", False),
    ],
    "data science": [
        ("Coursera", "Data Science", "data science", True),
        ("DataCamp", "Data Scientist track", "data science", False),
    ],
}


def _url_for_platform(platform: str, query: str) -> str:
    """Map platform to search URL."""
    p = platform.lower()
    if p in ("coursera", "udemy", "linkedin", "edx", "datacamp"):
        return _platform_search_url(query, p)
    return _platform_search_url(query, "google")


def get_course_options(skill: str) -> List[dict]:
    """
    Get multiple course options for a skill with best recommendation.
    All URLs point to platform search - reliable, no 404s.
    """
    key = skill.lower().strip()
    if key in COURSE_OPTIONS:
        return [
            {
                "platform": p,
                "name": n,
                "url": _url_for_platform(p, q),
                "is_best": b,
            }
            for p, n, q, b in COURSE_OPTIONS[key]
        ]
    # Fallback: search URLs for unknown skills
    return [
        {"platform": "Coursera", "name": f"{skill} courses", "url": _platform_search_url(skill, "coursera"), "is_best": True},
        {"platform": "Udemy", "name": f"{skill} courses", "url": _platform_search_url(skill, "udemy"), "is_best": False},
        {"platform": "LinkedIn", "name": f"{skill} courses", "url": _platform_search_url(skill, "linkedin"), "is_best": False},
        {"platform": "Google", "name": f"Search {skill} certification", "url": _platform_search_url(skill, "google"), "is_best": False},
    ]
