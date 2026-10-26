from pathlib import Path
from collections import Counter
import argparse
import csv
import json
import time

import cv2
import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "runs/roadsentry/baseline_yolo26n_9class/weights/best.pt"
CAND = ROOT / "runs/roadsentry/yolo11s_rdd_warmstart_9class_640_20261021/weights/best.pt"

CANDIDATE_CLASSES = [0, 1, 2, 3]
BASELINE_CLASSES = [4, 5, 6, 7, 8]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument(
        "--output",
        default=str(ROOT / "outputs/demo/video_inspection_rainwater_routed_20261021"),
    )
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()

    source = Path(args.source).resolve()
    output = Path(args.output).resolve()

    if not source.is_file():
        raise FileNotFoundError(source)
    if not BASE.is_file() or not CAND.is_file():
        raise FileNotFoundError("A model checkpoint is missing.")
    if not 0 <= args.conf <= 1 or args.imgsz < 32:
        raise ValueError("Invalid confidence or image size.")

    output.mkdir(parents=True, exist_ok=True)
    device = 0 if torch.cuda.is_available() else "cpu"

    print("Loading both models...")
    baseline = YOLO(str(BASE))
    candidate = YOLO(str(CAND))

    def ordered_names(names):
        return [
            names[i] if i in names else names[str(i)]
            for i in range(len(names))
        ] if isinstance(names, dict) else list(names)

    names = ordered_names(baseline.names)
    if ordered_names(candidate.names) != names or len(names) != 9:
        raise RuntimeError("Model class mappings differ; stopping.")

    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {source}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    video_path = output / f"{source.stem}_routed.mp4"
    csv_path = output / f"{source.stem}_routed_events.csv"
    json_path = output / f"{source.stem}_routed_report.json"

    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Cannot create video: {video_path}")

    colors = {
        0: (0, 200, 255), 1: (255, 180, 0),
        2: (0, 220, 0), 3: (255, 0, 220),
        4: (255, 100, 100), 5: (100, 255, 255),
        6: (150, 100, 255), 7: (255, 200, 100),
        8: (180, 255, 100),
    }
    events = []
    frames_read = 0
    frames_with_detections = 0
    frames_with_d40 = 0
    started = time.perf_counter()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            # YOLO11s handles the four common damage categories.
            r1 = candidate.predict(
                source=frame, imgsz=args.imgsz, conf=0.001, iou=0.7,
                max_det=300, classes=CANDIDATE_CLASSES, device=device,
                verbose=False, save=False,
            )[0]

            # YOLO26n retains the other five categories.
            r2 = baseline.predict(
                source=frame, imgsz=args.imgsz, conf=0.001, iou=0.7,
                max_det=300, classes=BASELINE_CLASSES, device=device,
                verbose=False, save=False,
            )[0]

            frame_detections = []
            for result, model_label, allowed in (
                (r1, "YOLO11s", set(CANDIDATE_CLASSES)),
                (r2, "YOLO26n", set(BASELINE_CLASSES)),
            ):
                boxes = result.boxes
                if boxes is None or not len(boxes):
                    continue

                coords = boxes.xyxy.cpu().tolist()
                class_ids = boxes.cls.cpu().tolist()
                scores = boxes.conf.cpu().tolist()

                for xyxy, cls, score in zip(coords, class_ids, scores):
                    cls = int(cls)
                    score = float(score)
                    if cls not in allowed or score < args.conf:
                        continue

                    # Exploratory size filter: reject boxes covering >10% of frame.
                    box_w = max(0.0, float(xyxy[2]) - float(xyxy[0]))
                    box_h = max(0.0, float(xyxy[3]) - float(xyxy[1]))
                    area_pct = 100.0 * box_w * box_h / (width * height)
                    if area_pct > 10.0:
                        continue

                    x1, y1, x2, y2 = map(int, xyxy)
                    color = colors.get(cls, (255, 255, 255))
                    label = f"{names[cls]} {score:.2f}"

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    text_y = max(20, y1 - 7)
                    cv2.putText(
                        frame, label, (x1, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.85, color, 2,
                        cv2.LINE_AA,
                    )

                    event = {
                        "frame": frames_read,
                        "timestamp_seconds": round(frames_read / fps, 3),
                        "class_id": cls,
                        "class_name": names[cls],
                        "confidence": round(score, 6),
                        "model": model_label,
                        "x1": round(float(xyxy[0]), 2),
                        "y1": round(float(xyxy[1]), 2),
                        "x2": round(float(xyxy[2]), 2),
                        "y2": round(float(xyxy[3]), 2),
                    }
                    events.append(event)
                    frame_detections.append(event)

            if frame_detections:
                frames_with_detections += 1
            if any(d["class_name"] == "D40" for d in frame_detections):
                frames_with_d40 += 1

            cv2.putText(
                frame,
                f"RoadSentry routed | frame {frames_read} | "
                f"detections {len(frame_detections)}",
                (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255, 255, 255), 2, cv2.LINE_AA,
            )
            writer.write(frame)
            frames_read += 1

            if frames_read % 150 == 0:
                print(f"Processed {frames_read}/{total or '?'} frames")

    finally:
        cap.release()
        writer.release()

    counts = Counter(e["class_name"] for e in events)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        columns = [
            "frame", "timestamp_seconds", "class_id", "class_name",
            "confidence", "model", "x1", "y1", "x2", "y2",
        ]
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        w.writerows(events)

    elapsed = time.perf_counter() - started
    report = {
        "source": str(source),
        "baseline_model": str(BASE),
        "candidate_model": str(CAND),
        "routing": {
            "YOLO11s": [names[i] for i in CANDIDATE_CLASSES],
            "YOLO26n": [names[i] for i in BASELINE_CLASSES],
        },
        "confidence_threshold": args.conf,
        "image_size": args.imgsz,
        "frames_read": frames_read,
        "frames_with_detections": frames_with_detections,
        "frames_with_D40": frames_with_d40,
        "detection_events": len(events),
        "events_by_class": dict(counts),
        "processing_seconds": round(elapsed, 2),
        "output_video": str(video_path),
        "events_csv": str(csv_path),
    }
    json_path.write_text(json.dumps(report, indent=2) + "\n")

    print("\n=== ROUTED VIDEO COMPLETE ===")
    print(json.dumps(report, indent=2))
    print("\nReview the rendered video for false positives before using it.")


if __name__ == "__main__":
    main()
