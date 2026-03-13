"""
Build skill graph from job data - skills as nodes, co-occurrence as edges.
Uses the Graph class for internal representation.
"""

from collections import Counter
from typing import List, Tuple

import pandas as pd

from src.feature_engineering import build_skill_set_per_job
from src.graph import Graph


def build_skill_cooccurrence_graph(
    df: pd.DataFrame,
    min_cooccurrence: int = 2,
    top_skills: int = 50,
) -> Tuple[Graph, Counter]:
    """
    Build a graph where:
    - Nodes = skills
    - Edge (A, B) = skills A and B appear together in at least min_cooccurrence jobs

    Returns (graph, skill_frequency_counter).
    """
    skill_series = build_skill_set_per_job(df)
    freq: Counter = Counter()
    cooccur: Counter = Counter()

    for skills in skill_series:
        skills = [s for s in skills if s and isinstance(s, str) and len(s.strip()) > 1]
        for s in skills:
            freq[s] += 1
        # Count co-occurrence pairs
        skills_list = list(set(skills))
        for i, a in enumerate(skills_list):
            for b in skills_list[i + 1 :]:
                if a != b:
                    pair = tuple(sorted([a, b]))
                    cooccur[pair] += 1

    # Top skills by frequency
    top = [s for s, _ in freq.most_common(top_skills)]
    g = Graph(directed=False)

    for skill in top:
        g.add_node(skill)

    for (a, b), count in cooccur.items():
        if count >= min_cooccurrence and a in top and b in top:
            g.add_edge(a, b)

    return g, freq


def build_learning_dependency_graph(
    user_skills: List[str],
    missing_skills: List[str],
    skill_prerequisites: dict = None,
) -> Graph:
    """
    Build a directed graph for learning order: prerequisite skills -> dependent skills.
    Uses curated prerequisites when available; otherwise assumes no dependencies.
    """
    if skill_prerequisites is None:
        skill_prerequisites = {
            "machine learning": ["python", "sql", "statistics"],
            "deep learning": ["machine learning", "python"],
            "tensorflow": ["python", "machine learning"],
            "pytorch": ["python", "machine learning"],
            "aws": ["python", "linux"],
            "docker": ["linux", "python"],
            "kubernetes": ["docker", "linux"],
            "react": ["javascript", "html", "css"],
            "spark": ["python", "sql"],
        }

    user_set = {s.lower().strip() for s in user_skills if s}
    g = Graph(directed=True)

    all_skills = set(user_set) | {s.lower().strip() for s in missing_skills if s}
    for s in all_skills:
        g.add_node(s)

    for skill, prereqs in skill_prerequisites.items():
        sk = skill.lower().strip()
        if sk not in all_skills:
            continue
        for pre in prereqs:
            pre = pre.lower().strip()
            if pre in all_skills and pre != sk:
                g.add_edge(pre, sk)

    return g
