"""
Day-by-day learning flowchart for each skill - subtopics per day.
"""

from typing import Dict, List

# Curated day-by-day learning plans: skill -> list of (day, subtopics)
SKILL_DAY_PLANS: Dict[str, List[tuple]] = {
    "python": [
        (1, ["Install Python & IDE", "Variables & Data Types", "Basic I/O"]),
        (2, ["Conditionals & Loops", "Functions", "Lists & Tuples"]),
        (3, ["Dictionaries & Sets", "File Handling", "Error Handling"]),
        (4, ["OOP Basics", "Classes & Objects", "Inheritance"]),
        (5, ["Modules & Packages", "Pip & Virtual Env", "Mini Project"]),
    ],
    "sql": [
        (1, ["What is SQL", "SELECT & FROM", "WHERE clause"]),
        (2, ["JOINs (INNER, LEFT)", "GROUP BY", "Aggregations"]),
        (3, ["Subqueries", "Window Functions", "CTEs"]),
        (4, ["Indexes", "Query Optimization", "Practice Queries"]),
        (5, ["Real Dataset Practice", "Build a Report", "Review"]),
    ],
    "aws": [
        (1, ["AWS Account Setup", "IAM Basics", "EC2 Overview"]),
        (2, ["Launch EC2 Instance", "S3 Basics", "Security Groups"]),
        (3, ["RDS & Databases", "Lambda Intro", "CloudWatch"]),
        (4, ["VPC & Networking", "Load Balancer", "Auto Scaling"]),
        (5, ["Hands-on Project", "Cost Management", "Best Practices"]),
    ],
    "machine learning": [
        (1, ["ML Overview", "Supervised vs Unsupervised", "Data Prep"]),
        (2, ["Linear Regression", "Logistic Regression", "Evaluation Metrics"]),
        (3, ["Decision Trees", "Random Forest", "Cross-Validation"]),
        (4, ["Clustering (K-Means)", "Dimensionality Reduction", "Feature Engineering"]),
        (5, ["End-to-end Project", "Model Deployment", "Review"]),
    ],
    "react": [
        (1, ["React Setup", "JSX Basics", "Components"]),
        (2, ["Props & State", "Event Handling", "Conditional Render"]),
        (3, ["Hooks (useState, useEffect)", "Lists & Keys", "Forms"]),
        (4, ["React Router", "API Calls", "Context API"]),
        (5, ["Build a Mini App", "Deploy", "Review"]),
    ],
    "docker": [
        (1, ["What is Docker", "Install Docker", "Images vs Containers"]),
        (2, ["Dockerfile Basics", "Build & Run", "Docker Hub"]),
        (3, ["Volumes", "Networking", "Docker Compose"]),
        (4, ["Multi-stage Builds", "Best Practices", "Debugging"]),
        (5, ["Deploy an App", "CI/CD Intro", "Review"]),
    ],
    "tensorflow": [
        (1, ["TF Install", "Tensors Basics", "Operations"]),
        (2, ["Keras API", "Sequential Model", "Training Loop"]),
        (3, ["CNNs", "Image Classification", "Transfer Learning"]),
        (4, ["RNNs/LSTMs", "Text Data", "Saving Models"]),
        (5, ["Deploy Model", "TF Serving", "Project"]),
    ],
    "pytorch": [
        (1, ["PyTorch Install", "Tensors", "Autograd"]),
        (2, ["nn.Module", "Training Loop", "Datasets"]),
        (3, ["CNNs", "Transfer Learning", "Visualization"]),
        (4, ["RNNs", "NLP Basics", "Saving Models"]),
        (5, ["Project", "Deployment", "Review"]),
    ],
    "tableau": [
        (1, ["Tableau Install", "Connect Data", "Basic Charts"]),
        (2, ["Filters", "Parameters", "Calculated Fields"]),
        (3, ["Dashboards", "Story Points", "Formatting"]),
        (4, ["Advanced Charts", "Maps", "Best Practices"]),
        (5, ["Build Report", "Share", "Review"]),
    ],
    "spark": [
        (1, ["Spark Overview", "PySpark Setup", "RDD Basics"]),
        (2, ["DataFrames", "Transformations", "Actions"]),
        (3, ["SQL in Spark", "Joins", "Aggregations"]),
        (4, ["Optimization", "Partitioning", "Caching"]),
        (5, ["ETL Pipeline", "Cluster Mode", "Project"]),
    ],
    "kubernetes": [
        (1, ["K8s Concepts", "Minikube/kind", "Pods"]),
        (2, ["Deployments", "Services", "ConfigMaps"]),
        (3, ["Ingress", "Secrets", "Namespaces"]),
        (4, ["Helm", "Monitoring", "Scaling"]),
        (5, ["Deploy App", "CI/CD", "Review"]),
    ],
    "excel": [
        (1, ["Interface", "Formulas Basics", "Cell Ref"]),
        (2, ["VLOOKUP/XLOOKUP", "Pivot Tables", "Charts"]),
        (3, ["Conditional Formatting", "Data Validation", "Filters"]),
        (4, ["Macros Intro", "Power Query", "Dashboards"]),
        (5, ["Build Report", "Automation", "Review"]),
    ],
}


def get_flowchart_plan(skill: str) -> List[tuple]:
    """Get day-by-day plan for a skill. Returns default if not found."""
    key = skill.lower().strip()
    if key in SKILL_DAY_PLANS:
        return SKILL_DAY_PLANS[key]
    return [
        (1, ["Introduction", "Core Concepts", "Setup"]),
        (2, ["Fundamentals", "Practice", "Examples"]),
        (3, ["Advanced Topics", "Hands-on", "Projects"]),
        (4, ["Real-world Use", "Best Practices", "Review"]),
        (5, ["Build Something", "Share", "Next Steps"]),
    ]


def build_mermaid_flowchart(skill: str) -> str:
    """Build Mermaid flowchart code for the skill's day-by-day plan."""
    plan = get_flowchart_plan(skill)
    lines = ["flowchart LR", "    Start((Start))"]
    prev = "Start"
    for day, subtopics in plan:
        node_id = f"D{day}"
        label = f"Day {day}\\n" + "\\n".join(subtopics[:3])
        lines.append(f'    {node_id}["{label}"]')
        lines.append(f"    {prev} --> {node_id}")
        prev = node_id
    lines.append(f"    {prev} --> End((Complete))")
    return "\n".join(lines)
