"""
Curated skill definitions: prominent tech skills, abbreviation mapping, course-worthy skills.
Ensures recommendations show AI, AWS, Python, etc. — not rubbish words.
"""

import re
from typing import List, Set

# LinkedIn 2023 skill_abr -> human-readable (domain codes from dataset)
ABBREV_TO_SKILL = {
    "ACCT": "Accounting",
    "ADM": "Administration",
    "ART": "Design",
    "BD": "Business Development",
    "DSGN": "Design",
    "EDU": "Education",
    "ENG": "Engineering",
    "FIN": "Finance",
    "HCPR": "Healthcare",
    "HR": "Human Resources",
    "IT": "Information Technology",
    "LGL": "Legal",
    "MGMT": "Management",
    "MNFC": "Manufacturing",
    "MRKT": "Marketing",
    "OTHR": "Other",
    "PR": "Public Relations",
    "PRJM": "Project Management",
    "ADVR": "Advertising",
    "SALE": "Sales",
    "TRNG": "Training",
}

# Prominent tech/professional skills — course-worthy, no rubbish
# Used to: 1) extract from descriptions, 2) filter recommendations
PROMINENT_SKILLS: Set[str] = {
    # Languages
    "python", "java", "javascript", "typescript", "sql", "r", "c++", "c#", "go", "scala", "kotlin", "swift",
    # Cloud & DevOps
    "aws", "azure", "gcp", "docker", "kubernetes", "terraform", "jenkins", "ci/cd",
    # AI/ML
    "machine learning", "deep learning", "artificial intelligence", "ai", "nlp", "computer vision",
    "tensorflow", "pytorch", "scikit-learn", "keras",
    # Data
    "data science", "data analysis", "spark", "hadoop", "tableau", "power bi", "excel",
    # Web & Frameworks
    "react", "angular", "vue", "node.js", "django", "flask", "spring", ".net",
    # Other
    "project management", "agile", "scrum", "git", "linux", "rest api", "graphql",
}

# Normalized variants for extraction (description search)
PROMINENT_PATTERNS = [
    (r"\bpython\b", "Python"),
    (r"\bjava\b", "Java"),
    (r"\bjavascript\b", "JavaScript"),
    (r"\btypescript\b", "TypeScript"),
    (r"\bsql\b", "SQL"),
    (r"\baws\b", "AWS"),
    (r"\bazure\b", "Azure"),
    (r"\bgcp\b", "GCP"),
    (r"\bdocker\b", "Docker"),
    (r"\bkubernetes\b", "Kubernetes"),
    (r"\bmachine learning\b", "Machine Learning"),
    (r"\bdeep learning\b", "Deep Learning"),
    (r"\bartificial intelligence\b", "Artificial Intelligence"),
    (r"\bai\b", "AI"),
    (r"\btensorflow\b", "TensorFlow"),
    (r"\bpytorch\b", "PyTorch"),
    (r"\breact\b", "React"),
    (r"\bangular\b", "Angular"),
    (r"\bvue\b", "Vue"),
    (r"\bnode\.?js\b", "Node.js"),
    (r"\bdjango\b", "Django"),
    (r"\bflask\b", "Flask"),
    (r"\bspark\b", "Apache Spark"),
    (r"\btableau\b", "Tableau"),
    (r"\bexcel\b", "Excel"),
    (r"\bterraform\b", "Terraform"),
    (r"\bjenkins\b", "Jenkins"),
    (r"\bproject management\b", "Project Management"),
    (r"\bagile\b", "Agile"),
    (r"\bscrum\b", "Scrum"),
    (r"\bgit\b", "Git"),
    (r"\blinux\b", "Linux"),
    (r"\brest api\b", "REST API"),
    (r"\bgraphql\b", "GraphQL"),
    (r"\bpower bi\b", "Power BI"),
    (r"\bscikit-learn\b", "Scikit-learn"),
    (r"\bkeras\b", "Keras"),
]


def extract_prominent_skills(text: str) -> List[str]:
    """
    Extract only known prominent skills from text (no rubbish).
    Searches for curated terms only.
    """
    if not text or not isinstance(text, str):
        return []
    text_lower = text.lower()
    found: Set[str] = set()
    for pattern, name in PROMINENT_PATTERNS:
        if re.search(pattern, text_lower, re.IGNORECASE):
            found.add(name)
    return list(found)


def is_prominent_skill(s: str) -> bool:
    """True if skill is course-worthy (AI, AWS, Python, etc.)."""
    if not s:
        return False
    t = s.strip().lower()
    return t in PROMINENT_SKILLS


def map_abbrev(abbrev: str) -> str:
    """Map skill abbreviation to human-readable name."""
    if not abbrev:
        return ""
    key = str(abbrev).strip().upper()
    return ABBREV_TO_SKILL.get(key, abbrev)


# Skills we recommend courses for (prominent + mapped domains)
RECOMMENDABLE_SKILLS: Set[str] = PROMINENT_SKILLS | set(v.lower() for v in ABBREV_TO_SKILL.values())


def is_recommendable_skill(s: str) -> bool:
    """True if we should recommend courses for this skill."""
    if not s:
        return False
    t = s.strip().lower()
    return t in RECOMMENDABLE_SKILLS
