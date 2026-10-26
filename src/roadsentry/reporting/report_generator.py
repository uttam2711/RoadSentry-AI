"""JSON and self-contained HTML reporting for inspection results."""

import html
import json
from pathlib import Path


def write_json_report(report, output_path):
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    return path


def write_html_report(report, output_path):
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    records = report.get("inspections", [])
    rows = []
    for item in records:
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(item.get('inspection_id', '')))}</td>"
            f"<td>{html.escape(str(item.get('class_name', 'UNKNOWN')))}</td>"
            f"<td>{html.escape(str(item.get('confidence', '')))}</td>"
            f"<td>{html.escape(str(item.get('priority', '')))}</td>"
            f"<td>{html.escape(str(item.get('triage_score', '')))}</td>"
            f"<td>{html.escape(str(item.get('source', '')))}</td>"
            f"<td>{html.escape(str(item.get('review_status', '')))}</td>"
            "</tr>"
        )

    body = "\n".join(rows) or "<tr><td colspan='7'>No detections</td></tr>"
    document = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>RoadSentry AI Inspection Report</title>
<style>
body{{font-family:system-ui,sans-serif;max-width:1200px;margin:2rem auto;padding:0 1rem;color:#222}}
table{{border-collapse:collapse;width:100%;font-size:.92rem}}
th,td{{border:1px solid #ccc;padding:.55rem;text-align:left;overflow-wrap:anywhere}}
th{{background:#eee}} .notice{{padding:1rem;background:#fff4d6}}
</style>
</head>
<body>
<h1>RoadSentry AI — Inspection Triage Report</h1>
<p>Source: {html.escape(str(report.get('source', 'unknown')))}</p>
<p>Detections: {len(records)} | Candidate groups: {len(report.get('tracks', []))}</p>
<div class="notice"><strong>Prototype:</strong> automated detections and triage scores
are unverified candidates, not confirmed damage severity or safety decisions.
Every finding requires human review.</div>
<table><thead><tr><th>ID</th><th>Class</th><th>Confidence</th>
<th>Priority</th><th>Score</th><th>Source</th><th>Status</th></tr></thead>
<tbody>{body}</tbody></table>
<h2>Workflow metadata</h2>
<pre>{html.escape(json.dumps(report.get('metadata', {}), indent=2))}</pre>
</body></html>
"""
    path.write_text(document, encoding="utf-8")
    return path
