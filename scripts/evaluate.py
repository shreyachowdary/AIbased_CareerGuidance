#!/usr/bin/env python3
"""
Generate evaluation report: runtime, memory, qualitative checks.
"""
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
REPORTS_DIR = PROJECT_ROOT / "reports"


def main():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_DIR / "eval_report.md"

    try:
        import psutil
        process = psutil.Process(os.getpid())
    except ImportError:
        psutil = None
        process = None

    lines = []

    # Runtime + memory
    lines.append("# AI Career Guidance - Evaluation Report\n")
    lines.append("## Runtime & Memory Summary\n")

    start = time.perf_counter()
    try:
        from backend.app.services.skill_extraction import extract_skills
        from backend.app.services.recommender_tfidf import recommend as tfidf_rec
        from backend.app.services.recommender_embeddings import recommend as embed_rec
        from backend.app.services.recommender_hybrid import recommend as hybrid_rec
    except Exception as e:
        lines.append(f"**Error loading modules:** {e}\n")
        lines.append("Run preprocess and build scripts first.\n")
        with open(report_path, "w", encoding="utf-8") as f:
            f.writelines(lines)
        print(f"Report written to {report_path} (with errors)")
        return 1

    load_time = time.perf_counter() - start
    mem_mb = process.memory_info().rss / 1024 / 1024 if process else 0
    lines.append(f"- Module load time: {load_time:.2f}s\n")
    lines.append(f"- Memory after load: {mem_mb:.1f} MB\n")

    query = "Python SQL machine learning pandas AWS"
    top_k = 5

    # TF-IDF
    t0 = time.perf_counter()
    tfidf_res = tfidf_rec(query, top_k=top_k)
    tfidf_time = time.perf_counter() - t0
    lines.append(f"- TF-IDF recommend ({top_k}): {tfidf_time:.3f}s\n")

    # Embed
    t0 = time.perf_counter()
    embed_res = embed_rec(query, top_k=top_k)
    embed_time = time.perf_counter() - t0
    lines.append(f"- Embed recommend ({top_k}): {embed_time:.3f}s\n")

    # Hybrid
    t0 = time.perf_counter()
    hybrid_res = hybrid_rec(query, top_k=top_k, alpha=0.5)
    hybrid_time = time.perf_counter() - t0
    lines.append(f"- Hybrid recommend ({top_k}): {hybrid_time:.3f}s\n")

    mem_after = process.memory_info().rss / 1024 / 1024 if process else 0
    lines.append(f"- Memory after inference: {mem_after:.1f} MB\n")

    # Skill extraction
    skills = extract_skills(query)
    lines.append("\n## Skill Extraction\n")
    lines.append(f"Query: `{query}`\n")
    lines.append(f"Extracted skills: {skills}\n")

    # Qualitative: skill overlap
    lines.append("\n## Qualitative Sanity Checks\n")
    lines.append("### Skill Overlap (sample)\n")
    from backend.app.services.dataset_loader import load_jobs
    from backend.app.services.gap_analysis import analyze

    try:
        df = load_jobs()
        job_lookup = df.set_index("job_id").to_dict("index")
        for jid, score in tfidf_res[:3]:
            if jid in job_lookup:
                row = job_lookup[jid]
                job_text = str(row.get("title", "")) + " " + str(row.get("description", ""))
                matched, missing = analyze(skills, job_text, top_missing=10)
                lines.append(f"- Job `{jid}`: matched={len(matched)}, missing={len(missing)}\n")
    except Exception as e:
        lines.append(f"Could not compute overlap: {e}\n")

    lines.append("\n### Mode Comparison\n")
    lines.append("| Mode | Top-1 Job ID | Score |\n")
    lines.append("|------|--------------|-------|\n")
    if tfidf_res:
        lines.append(f"| tfidf | {tfidf_res[0][0]} | {tfidf_res[0][1]:.4f} |\n")
    if embed_res:
        lines.append(f"| embed | {embed_res[0][0]} | {embed_res[0][1]:.4f} |\n")
    if hybrid_res:
        lines.append(f"| hybrid | {hybrid_res[0][0]} | {hybrid_res[0][1]:.4f} |\n")

    lines.append("\n## Sample Run Output\n")
    lines.append("```\n")
    lines.append(f"Query: {query}\n")
    lines.append(f"TF-IDF top-3: {[(j, round(s, 4)) for j, s in tfidf_res[:3]]}\n")
    lines.append(f"Embed top-3: {[(j, round(s, 4)) for j, s in embed_res[:3]]}\n")
    lines.append(f"Hybrid top-3: {[(j, round(s, 4)) for j, s in hybrid_res[:3]]}\n")
    lines.append("```\n")

    with open(report_path, "w", encoding="utf-8") as f:
        f.writelines(lines)

    print(f"Report written to {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
