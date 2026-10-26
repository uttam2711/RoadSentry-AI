from pathlib import Path
import argparse
import csv
import json


def iou(a, b):
    """Intersection over Union for two xyxy bounding boxes."""
    x1 = max(a["x1"], b["x1"])
    y1 = max(a["y1"], b["y1"])
    x2 = min(a["x2"], b["x2"])
    y2 = min(a["y2"], b["y2"])

    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)

    area_a = max(0.0, a["x2"] - a["x1"]) * max(
        0.0, a["y2"] - a["y1"]
    )
    area_b = max(0.0, b["x2"] - b["x1"]) * max(
        0.0, b["y2"] - b["y1"]
    )

    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def main():
    parser = argparse.ArgumentParser(
        description="Group nearby video detections into estimated tracks."
    )
    parser.add_argument("--report", required=True)
    parser.add_argument("--max-gap", type=float, default=1.0)
    parser.add_argument("--min-iou", type=float, default=0.10)
    args = parser.parse_args()

    report_path = Path(args.report).expanduser().resolve()
    if not report_path.is_file():
        raise FileNotFoundError(report_path)

    if args.max_gap <= 0:
        parser.error("--max-gap must be positive")
    if not 0 <= args.min_iou <= 1:
        parser.error("--min-iou must be between 0 and 1")

    report = json.loads(report_path.read_text(encoding="utf-8"))
    events = sorted(
        report.get("events", []),
        key=lambda e: (e["timestamp_seconds"], e["frame"]),
    )

    tracks = []

    for event in events:
        best_track = None
        best_iou = -1.0

        for track in tracks:
            if track["class_id"] != event["class_id"]:
                continue

            last = track["events"][-1]
            gap = event["timestamp_seconds"] - last["timestamp_seconds"]

            if gap < 0 or gap > args.max_gap:
                continue

            overlap = iou(last, event)
            if overlap >= args.min_iou and overlap > best_iou:
                best_iou = overlap
                best_track = track

        if best_track is None:
            tracks.append({
                "class_id": event["class_id"],
                "class_name": event["class_name"],
                "events": [event],
            })
        else:
            best_track["events"].append(event)

    rows = []
    for index, track in enumerate(tracks, start=1):
        detections = track["events"]
        best = max(detections, key=lambda e: e["confidence"])

        rows.append({
            "track_id": f"T{index:03d}",
            "class_id": track["class_id"],
            "class_name": track["class_name"],
            "detection_count": len(detections),
            "first_seen_seconds": round(
                detections[0]["timestamp_seconds"], 3
            ),
            "last_seen_seconds": round(
                detections[-1]["timestamp_seconds"], 3
            ),
            "peak_confidence": round(best["confidence"], 4),
            "peak_frame": best["frame"],
            "x1": best["x1"],
            "y1": best["y1"],
            "x2": best["x2"],
            "y2": best["y2"],
            "review_status": "UNVERIFIED - HUMAN REVIEW REQUIRED",
        })

    output_dir = report_path.parent
    csv_path = output_dir / f"{report_path.stem}_tracks.csv"
    json_path = output_dir / f"{report_path.stem}_tracks.json"

    fields = list(rows[0].keys()) if rows else [
        "track_id", "class_id", "class_name", "detection_count",
        "first_seen_seconds", "last_seen_seconds", "peak_confidence",
        "peak_frame", "x1", "y1", "x2", "y2", "review_status",
    ]

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    result = {
        "source_report": str(report_path),
        "method": "Greedy class-matched temporal IoU grouping",
        "max_gap_seconds": args.max_gap,
        "min_iou": args.min_iou,
        "frame_detection_count": len(events),
        "estimated_track_count": len(rows),
        "warning": (
            "Tracks are heuristic groups of detections, not verified unique "
            "physical defects. Human review is required."
        ),
        "tracks": rows,
    }
    json_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    print("=== TRACK SUMMARY ===")
    print("Frame detections:", len(events))
    print("Estimated tracks:", len(rows))
    for row in rows:
        print(
            f'{row["track_id"]} | {row["class_name"]} | '
            f'{row["first_seen_seconds"]:.2f}-{row["last_seen_seconds"]:.2f}s | '
            f'detections={row["detection_count"]} | '
            f'peak_conf={row["peak_confidence"]:.3f} | '
            f'{row["review_status"]}'
        )
    print("\nCSV:", csv_path)
    print("JSON:", json_path)


if __name__ == "__main__":
    main()
