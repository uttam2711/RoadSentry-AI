from pathlib import Path
import argparse
import csv
import html
import json


def esc(value):
    return html.escape(str(value if value is not None else ""))


def main():
    root = Path(__file__).resolve().parents[1]
    default_dir = root / "outputs/demo/video_inspection_targets_960"

    parser = argparse.ArgumentParser(
        description="Build a reviewer-friendly RoadSentry HTML report."
    )
    parser.add_argument("--input-dir", type=Path, default=default_dir)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    folder = args.input_dir.expanduser().resolve()
    report_path = folder / "Cars_Moving_into_pothole_report.json"
    tracks_path = folder / "Cars_Moving_into_pothole_report_tracks.json"
    original_video_path = folder / "Cars_Moving_into_pothole_roadsentry.mp4"
    h264_video_path = folder / "Cars_Moving_into_pothole_roadsentry_h264.mp4"
    video_path = (
        h264_video_path if h264_video_path.is_file() else original_video_path
    )

    for path in (report_path, tracks_path, video_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required file missing: {path}")

    report = json.loads(report_path.read_text(encoding="utf-8"))
    tracks_data = json.loads(tracks_path.read_text(encoding="utf-8"))
    tracks = tracks_data.get("tracks", [])
    events = report.get("events", [])

    validation_summary_path = (
        root / "outputs/demo/validation_examples/"
        "validation_example_summary.json"
    )
    validation_data = (
        json.loads(validation_summary_path.read_text(encoding="utf-8"))
        if validation_summary_path.is_file()
        else {"examples": []}
    )
    validation_examples = validation_data.get("examples", [])

    metrics_path = (
        root / "runs/roadsentry/full_validation_baseline_640_20261015/"
        "full_validation_metrics.json"
    )
    evaluation_data = (
        json.loads(metrics_path.read_text(encoding="utf-8"))
        if metrics_path.is_file()
        else None
    )

    output = (
        args.output.expanduser().resolve()
        if args.output
        else folder / "RoadSentry_inspection_report.html"
    )
    output.parent.mkdir(parents=True, exist_ok=True)

    # Relative paths let the HTML report and video folder be moved together.
    video_href = Path(
        __import__("os").path.relpath(video_path, output.parent)
    ).as_posix()
    events_href = Path(
        __import__("os").path.relpath(
            folder / "Cars_Moving_into_pothole_events.csv", output.parent
        )
    ).as_posix()
    tracks_csv = folder / "Cars_Moving_into_pothole_report_tracks.csv"
    tracks_href = Path(
        __import__("os").path.relpath(tracks_csv, output.parent)
    ).as_posix()

    track_rows = []
    for track in tracks:
        track_rows.append(
            "<tr>"
            f"<td>{esc(track.get('track_id'))}</td>"
            f"<td>{esc(track.get('class_name'))}</td>"
            f"<td>{esc(track.get('detection_count'))}</td>"
            f"<td>{esc(track.get('first_seen_seconds'))} s</td>"
            f"<td>{esc(track.get('last_seen_seconds'))} s</td>"
            f"<td>{esc(track.get('peak_confidence'))}</td>"
            f"<td>{esc(track.get('peak_frame'))}</td>"
            f"<td><span class='review'>Human review required</span></td>"
            "</tr>"
        )

    if not track_rows:
        track_rows.append("<tr><td colspan='8'>No estimated tracks found.</td></tr>")

    event_rows = []
    for event in events:
        event_rows.append(
            "<tr>"
            f"<td>{esc(event.get('frame'))}</td>"
            f"<td>{esc(event.get('timestamp_seconds'))} s</td>"
            f"<td>{esc(event.get('class_name'))}</td>"
            f"<td>{esc(event.get('confidence'))}</td>"
            f"<td>{esc(event.get('x1'))}, {esc(event.get('y1'))}, "
            f"{esc(event.get('x2'))}, {esc(event.get('y2'))}</td>"
            "</tr>"
        )
    if not event_rows:
        event_rows.append("<tr><td colspan='5'>No detections recorded.</td></tr>")

    validation_rows = []
    for example in validation_examples:
        image_path = Path(example.get("output_image", "")).resolve()
        if not image_path.is_file():
            continue

        image_href = Path(
            __import__("os").path.relpath(image_path, output.parent)
        ).as_posix()
        class_name = esc(example.get("class_name", "Unknown"))
        source_name = esc(Path(example.get("image", "")).name)
        confidence = example.get("confidence", 0)
        overlap = example.get("iou", 0)

        validation_rows.append(
            "<article class='evidence-card'>"
            f"<h3>{class_name}</h3>"
            f"<a href='{esc(image_href)}'>"
            f"<img src='{esc(image_href)}' loading='lazy' "
            f"alt='Ground truth and prediction comparison for {class_name}'></a>"
            f"<div class='evidence-stats'>"
            f"<span>IoU: <strong>{float(overlap):.3f}</strong></span>"
            f"<span>Confidence: <strong>{float(confidence):.3f}</strong></span>"
            "</div>"
            f"<p>Source: {source_name}</p>"
            "</article>"
        )

    evaluation_section = ""
    if evaluation_data:
        overall = evaluation_data.get("overall", {})
        metric_labels = [
            ("precision", "Precision"),
            ("recall", "Recall"),
            ("mAP50", "mAP@0.5"),
            ("mAP50_95", "mAP@0.5:0.95"),
        ]
        metric_cards = "".join(
            "<div class='metric'>"
            f"<strong>{float(overall[key]):.4f}</strong>"
            f"<span>{label}</span></div>"
            for key, label in metric_labels
            if key in overall
        )

        class_rows = []
        for item in evaluation_data.get("class_metrics", []):
            class_rows.append(
                "<tr>"
                f"<td>{esc(item.get('class_name'))}</td>"
                f"<td>{float(item.get('precision', 0)):.4f}</td>"
                f"<td>{float(item.get('recall', 0)):.4f}</td>"
                f"<td>{float(item.get('mAP50', 0)):.4f}</td>"
                f"<td>{float(item.get('mAP50_95', 0)):.4f}</td>"
                "</tr>"
            )

        evaluation_section = (
            "<section class='panel'>"
            "<h2>Full validation metrics</h2>"
            "<p>Results from the full validation split with the saved "
            "baseline checkpoint at image size 640. These full-split "
            "metrics are separate from the curated examples.</p>"
            f"<div class='metrics'>{metric_cards}</div>"
            "<div class='table-wrap'><table>"
            "<thead><tr><th>Class</th><th>Precision</th><th>Recall</th>"
            "<th>mAP@0.5</th><th>mAP@0.5:0.95</th></tr></thead>"
            f"<tbody>{''.join(class_rows)}</tbody>"
            "</table></div>"
            "<p>Results are specific to this checkpoint, dataset, and "
            "evaluation configuration.</p>"
            "</section>"
        )

    validation_content = "".join(validation_rows) or (
        "No validation example images are available."
    )
    validation_section = (
        "<section class='panel'>"
        "<h2>Curated validation evidence</h2>"
        "<p>These examples were selected from a reproducible sample of the "
        "labeled validation split. Green boxes indicate ground truth; "
        "magenta boxes indicate predictions. These curated examples are not "
        "full-dataset performance metrics.</p>"
        f"<div class='evidence-grid'>{validation_content}</div>"
        "</section>"
    )

    html_doc = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>RoadSentry AI | Inspection Report</title>
<style>
:root {{
  color-scheme: dark;
  --bg: #0b1220; --panel: #121d30; --line: #263750;
  --text: #edf4ff; --muted: #a6b6cc; --accent: #55d6be;
}}
* {{ box-sizing: border-box; }}
body {{ margin: 0; background: var(--bg); color: var(--text);
  font: 15px/1.6 system-ui, sans-serif; }}
main {{ max-width: 1180px; margin: auto; padding: 30px 20px 60px; }}
header {{ display: flex; flex-wrap: wrap; align-items: center;
  justify-content: space-between; gap: 16px; margin-bottom: 26px; }}
.eyebrow {{ color: var(--accent); text-transform: uppercase;
  letter-spacing: .14em; font-size: 12px; font-weight: 700; }}
h1 {{ margin: 4px 0; font-size: clamp(28px, 5vw, 42px); }}
h2 {{ font-size: 21px; margin-top: 0; }}
p, .muted {{ color: var(--muted); }}
.panel {{ background: var(--panel); border: 1px solid var(--line);
  border-radius: 16px; padding: 22px; margin: 18px 0; }}
.metrics {{ display: grid; grid-template-columns: repeat(auto-fit,minmax(160px,1fr));
  gap: 12px; }}
.metric {{ background: #0c1627; border: 1px solid var(--line);
  padding: 18px; border-radius: 12px; }}
.metric strong {{ display: block; font-size: 30px; color: var(--accent); }}
.metric span {{ color: var(--muted); font-size: 13px; }}
video {{ width: 100%; max-height: 620px; background: #000; border-radius: 10px; }}
.table-wrap {{ overflow-x: auto; }}
table {{ width: 100%; border-collapse: collapse; min-width: 700px; }}
th, td {{ text-align: left; padding: 11px 12px; border-bottom: 1px solid var(--line); }}
th {{ color: var(--muted); font-size: 12px; text-transform: uppercase; }}
.review {{ display: inline-block; border: 1px solid #9d7935;
  color: #ffd889; padding: 3px 8px; border-radius: 99px; font-size: 12px; }}
.evidence-grid {{ display: grid;
  grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 14px; }}
.evidence-card {{ background: #0c1627; border: 1px solid var(--line);
  border-radius: 12px; padding: 14px; overflow: hidden; }}
.evidence-card img {{ width: 100%; height: auto; display: block;
  border-radius: 8px; background: #000; }}
.evidence-card h3 {{ margin: 0 0 10px; font-size: 16px; }}
.evidence-stats {{ display: flex; flex-wrap: wrap; gap: 14px;
  color: var(--muted); margin-top: 10px; font-size: 13px; }}
a {{ color: var(--accent); }}
.notice {{ border-left: 4px solid #e5ae48; padding: 12px 16px;
  background: #241e13; border-radius: 6px; }}
footer {{ margin-top: 30px; color: var(--muted); font-size: 12px; }}
@media print {{
  body {{ color: #111; background: white; }}
  .panel, .metric {{ background: white; color: #111; border-color: #ccc; }}
  p, .muted, th {{ color: #444; }}
  video {{ max-height: 400px; }}
}}
</style>
</head>
<body>
<main>
<header>
  <div>
    <div class="eyebrow">Computer vision · Road inspection</div>
    <h1>RoadSentry AI</h1>
    <div class="muted">Video inspection report</div>
  </div>
  <div class="muted">Model: {esc(Path(report.get("model_weights", "")).name)}</div>
</header>

<section class="panel">
  <h2>Inspection overview</h2>
  <div class="metrics">
    <div class="metric"><strong>{len(events)}</strong><span>Frame-level detections</span></div>
    <div class="metric"><strong>{len(tracks)}</strong><span>Estimated tracks</span></div>
    <div class="metric"><strong>{esc(report.get("analyzed_frames", 0))}</strong><span>Frames analyzed</span></div>
    <div class="metric"><strong>{esc(report.get("image_size", "N/A"))}</strong><span>Inference image size</span></div>
  </div>
  <p>Source video: {esc(Path(report.get("source_video", "")).name)}</p>
  <p>Resolution: {esc(report.get("video_width"))} × {esc(report.get("video_height"))}
  · Input FPS: {esc(report.get("input_fps"))}
  · Frame step: {esc(report.get("frame_step"))}
  · Confidence threshold: {esc(report.get("confidence_threshold"))}</p>
</section>

<section class="panel">
  <h2>Annotated inspection video</h2>
  <video controls preload="metadata">
    <source src="{esc(video_href)}" type="video/mp4">
    Your browser does not support embedded video.
  </video>
  <p><a href="{esc(video_href)}">Open or download annotated video</a></p>
</section>

<section class="panel">
  <h2>Estimated damage tracks</h2>
  <p>Tracks group nearby same-class detections using temporal proximity and bounding-box overlap.</p>
  <div class="table-wrap"><table>
    <thead><tr><th>Track</th><th>Class</th><th>Detections</th>
    <th>First seen</th><th>Last seen</th><th>Peak confidence</th>
    <th>Peak frame</th><th>Review status</th></tr></thead>
    <tbody>{''.join(track_rows)}</tbody>
  </table></div>
  <p><a href="{esc(tracks_href)}">Download track CSV</a></p>
</section>

<section class="panel">
  <h2>Frame-level detection evidence</h2>
  <div class="table-wrap"><table>
    <thead><tr><th>Frame</th><th>Timestamp</th><th>Class</th>
    <th>Confidence</th><th>Bounding box (x1, y1, x2, y2)</th></tr></thead>
    <tbody>{''.join(event_rows)}</tbody>
  </table></div>
  <p><a href="{esc(events_href)}">Download frame-level events CSV</a></p>
</section>

{evaluation_section}

{validation_section}

<section class="panel">
  <h2>Interpretation and limitations</h2>
  <div class="notice"><strong>Human review required.</strong>
  Detections and tracks are model-generated candidates, not verified road defects.
  Multiple detections may refer to the same physical area, and false positives are possible.</div>
  <p>Track count is a heuristic grouping of frame-level detections. This report does not establish
  defect severity, geographic location, physical dimensions, or ground-truth accuracy.</p>
  <p>Processing time: {esc(report.get("processing_seconds"))} seconds.
  Reported analyzed-frame throughput: {esc(report.get("analyzed_frames_per_second"))} frames/second.
  Treat throughput as specific to this run and environment.</p>
</section>
<footer>RoadSentry AI · Generated from the existing inspection JSON and track JSON.
No new model inference was performed to build this report.</footer>
</main>
</body>
</html>
"""

    output.write_text(html_doc, encoding="utf-8")
    print("HTML report created:", output)
    print("Video link:", video_href)
    print("Frame detections:", len(events))
    print("Estimated tracks:", len(tracks))
    print("Report size:", output.stat().st_size, "bytes")


if __name__ == "__main__":
    main()
