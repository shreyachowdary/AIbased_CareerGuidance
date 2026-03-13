"""
Generate visual flowchart for roadmap - main topics as flow.
"""

from src.skill_roadmap_flowchart import get_flowchart_plan


def build_flowchart_html(skill: str) -> str:
    """Build a visual flowchart showing main topics per day (no raw code)."""
    plan = get_flowchart_plan(skill)
    # Main topic = first subtopic of each day (the key focus)
    nodes = []
    for day, subtopics in plan:
        main_topic = subtopics[0] if subtopics else f"Day {day}"
        nodes.append((day, main_topic, subtopics))

    # Build flow: Start -> D1 -> D2 -> D3 -> D4 -> D5 -> Done
    boxes_html = []
    for i, (day, main, subs) in enumerate(nodes):
        subs_list = "".join(f"<li>{s}</li>" for s in subs[:4])
        boxes_html.append(f'''
        <div class="flow-node" style="animation-delay: {i * 0.1}s">
            <div class="flow-day-badge">Day {day}</div>
            <div class="flow-main">{main}</div>
            <ul class="flow-list">{subs_list}</ul>
        </div>
        ''')
        if i < len(nodes) - 1:
            boxes_html.append('<div class="flow-connector"><span>→</span></div>')

    html = f'''
<!DOCTYPE html>
<html>
<head>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ font-family: 'Plus Jakarta Sans', sans-serif; padding: 16px; }}
.flowchart {{ display: flex; align-items: stretch; flex-wrap: wrap; gap: 0; 
    padding: 28px; background: linear-gradient(135deg, #f0f4ff 0%, #e8f4fd 50%, #fdf4ff 100%);
    border-radius: 20px; border: 2px solid rgba(99,102,241,0.25); overflow-x: auto; }}
.flow-node {{ min-width: 180px; padding: 20px; background: white; border-radius: 14px;
    box-shadow: 0 4px 24px rgba(99,102,241,0.15); border: 2px solid rgba(99,102,241,0.2);
    transition: all 0.4s cubic-bezier(0.4,0,0.2,1); animation: slideIn 0.5s ease-out forwards;
    opacity: 0; }}
.flow-node:hover {{ transform: translateY(-6px) scale(1.02); box-shadow: 0 12px 40px rgba(99,102,241,0.25);
    border-color: #6366f1; }}
@keyframes slideIn {{ from {{ opacity: 0; transform: translateX(-20px); }} to {{ opacity: 1; transform: translateX(0); }} }}
.flow-day-badge {{ font-weight: 800; color: #6366f1; font-size: 0.9rem; margin-bottom: 8px;
    background: linear-gradient(135deg,#6366f1,#0ea5e9); -webkit-background-clip: text;
    -webkit-text-fill-color: transparent; }}
.flow-main {{ font-weight: 700; color: #0f172a; font-size: 1rem; margin-bottom: 12px; line-height: 1.3; }}
.flow-list {{ font-size: 0.8rem; color: #64748b; list-style: none; line-height: 1.8; }}
.flow-list li::before {{ content: "• "; color: #6366f1; font-weight: bold; }}
.flow-connector {{ display: flex; align-items: center; padding: 0 8px; font-size: 28px; color: #6366f1;
    font-weight: bold; }}
</style>
</head>
<body>
<div class="flowchart">
{"".join(boxes_html)}
</div>
</body>
</html>
'''
    return html
