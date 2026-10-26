from pathlib import Path
from collections import Counter
import argparse
import csv
import json
import time

import cv2
from ultralytics import YOLO

ROOT = Path.cwd()
YOLO26_WEIGHTS = ROOT / "runs/roadsentry/baseline_yolo26n_9class/weights/best.pt"
YOLO11_WEIGHTS = ROOT / "runs/roadsentry/yolo11s_rdd_warmstart_9class_640_20261021/weights/best.pt"

CORE_CLASSES = {"D00", "D10", "D20"}
OTHER_CLASSES = {"D01", "D11", "D43", "D44", "D50"}
POTHOLE_CLASS = "D40"


def class_map(model):
    names = model.names
    return {int(i): str(n) for i, n in
            (names.items() if isinstance(names, dict) else enumerate(names))}


def box_iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2-x1) * max(0, y2-y1)
    aa = max(0, a[2]-a[0]) * max(0, a[3]-a[1])
    ab = max(0, b[2]-b[0]) * max(0, b[3]-b[1])
    union = aa + ab - inter
    return inter / union if union else 0.0


def tile_starts(length, size, stride):
    if length <= size:
        return [0]
    starts = list(range(0, length-size+1, stride))
    last = length-size
    if starts[-1] != last:
        starts.append(last)
    return starts


def make_tiles(frame, size, overlap):
    h, w = frame.shape[:2]
    stride = max(1, int(size * (1-overlap)))
    tiles = []
    for y in tile_starts(h, size, stride):
        for x in tile_starts(w, size, stride):
            crop = frame[y:min(y+size, h), x:min(x+size, w)]
            tiles.append((crop, x, y))
    return tiles


def merge_duplicates(detections, threshold=0.55):
    kept = []
    for det in sorted(detections, key=lambda d: d["confidence"], reverse=True):
        duplicate = any(
            old["class"] == det["class"]
            and box_iou(old["box"], det["box"]) >= threshold
            for old in kept
        )
        if not duplicate:
            kept.append(det)
    return kept


def add_result(detections, result, names, wanted, offset=(0, 0), source="full"):
    if result.boxes is None:
        return

    ox, oy = offset
    for box in result.boxes:
        cid = int(box.cls.item())
        name = names.get(cid, str(cid))
        if name not in wanted:
            continue
        x1, y1, x2, y2 = map(float, box.xyxy[0].tolist())
        detections.append({
            "class": name,
            "confidence": float(box.conf.item()),
            "box": [x1+ox, y1+oy, x2+ox, y2+oy],
            "source": source,
        })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--conf", type=float, default=0.15)
    parser.add_argument("--tile", type=int, default=512)
    parser.add_argument("--tile-imgsz", type=int, default=640)
    parser.add_argument("--overlap", type=float, default=0.30)
    args = parser.parse_args()

    if not 0 < args.conf < 1:
        raise SystemExit("--conf must be between 0 and 1.")
    if args.tile < 320:
        raise SystemExit("--tile must be at least 320.")
    if not 0 <= args.overlap < 0.75:
        raise SystemExit("--overlap must be between 0 and 0.75.")

    source = Path(args.source).resolve()
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    if not source.is_file():
        raise SystemExit(f"Video not found: {source}")
    if not YOLO26_WEIGHTS.is_file() or not YOLO11_WEIGHTS.is_file():
        raise SystemExit("One or both required checkpoints are missing.")

    print("Loading existing checkpoints...")
    model26 = YOLO(str(YOLO26_WEIGHTS))
    model11 = YOLO(str(YOLO11_WEIGHTS))
    names26 = class_map(model26)
    names11 = class_map(model11)

    ids26 = [i for i, n in names26.items() if n in OTHER_CLASSES]
    ids11_core = [i for i, n in names11.items() if n in CORE_CLASSES]
    ids11_d40 = [i for i, n in names11.items() if n == POTHOLE_CLASS]

    if not ids26 or not ids11_core or not ids11_d40:
        raise SystemExit(
            f"Unexpected class mappings. YOLO26n={names26}; YOLO11s={names11}"
        )

    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {source}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 25.0
    expected = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = cv2.VideoWriter(
        str(output), cv2.VideoWriter_fourcc(*"mp4v"),
        fps, (width, height)
    )
    if not writer.isOpened():
        cap.release()
        raise SystemExit(f"Could not create output video: {output}")

    csv_path = output.with_name(output.stem + "_events.csv")
    report_path = output.with_name(output.stem + "_report.json")
    counts = Counter()
    total_events = 0
    frames_with_d40 = 0
    frame_index = 0
    start = time.time()

    with csv_path.open("w", newline="") as f:
        fields = [
            "frame", "time_seconds", "class", "confidence",
            "x1", "y1", "x2", "y2", "inference_source"
        ]
        csv_writer = csv.DictWriter(f, fieldnames=fields)
        csv_writer.writeheader()

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            detections = []

            # Preserve full-frame inference for the non-pothole classes.
            full11 = model11.predict(
                source=frame, imgsz=640, conf=args.conf,
                classes=ids11_core, max_det=150, device=0,
                verbose=False
            )[0]
            add_result(
                detections, full11, names11, CORE_CLASSES,
                source="YOLO11s-full"
            )

            full26 = model26.predict(
                source=frame, imgsz=640, conf=args.conf,
                classes=ids26, max_det=150, device=0,
                verbose=False
            )[0]
            add_result(
                detections, full26, names26, OTHER_CLASSES,
                source="YOLO26n-full"
            )

            # Detect D40 on overlapping original-resolution crops.
            tiles = make_tiles(frame, args.tile, args.overlap)
            tile_results = model11.predict(
                source=[item[0] for item in tiles],
                imgsz=args.tile_imgsz, conf=args.conf,

                classes=ids11_d40, max_det=150, batch=4,
                device=0, verbose=False
            )

            for result, (_, ox, oy) in zip(tile_results, tiles):
                add_result(
                    detections, result, names11, {POTHOLE_CLASS},
                    offset=(ox, oy), source="YOLO11s-tiled"
                )

            detections = merge_duplicates(detections)

            if any(d["class"] == POTHOLE_CLASS for d in detections):
                frames_with_d40 += 1

            for det in detections:
                name = det["class"]
                conf = det["confidence"]
                x1, y1, x2, y2 = det["box"]
                color = (0, 0, 255) if name == POTHOLE_CLASS else (0, 200, 255)

                x1i, y1i = max(0, int(x1)), max(0, int(y1))
                x2i, y2i = min(width-1, int(x2)), min(height-1, int(y2))

                cv2.rectangle(frame, (x1i, y1i), (x2i, y2i), color, 2)
                label = f"{name} {conf:.2f}"
                if det["source"] == "YOLO11s-tiled":
                    label += " [tile]"
                cv2.putText(
                    frame, label, (x1i, max(18, y1i-5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, color, 2
                )

                csv_writer.writerow({
                    "frame": frame_index,
                    "time_seconds": round(frame_index / fps, 3),
                    "class": name,
                    "confidence": round(conf, 5),
                    "x1": round(x1, 2), "y1": round(y1, 2),
                    "x2": round(x2, 2), "y2": round(y2, 2),
                    "inference_source": det["source"],
                })
                counts[name] += 1
                total_events += 1

            cv2.putText(
                frame, f"RoadSentry tiled D40 | Frame {frame_index}",
                (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (255, 255, 255), 2
            )
            writer.write(frame)
            frame_index += 1

            if frame_index % 100 == 0:
                print(f"Processed {frame_index}/{expected or '?'} frames")

    cap.release()
    writer.release()

    report = {
        "source": str(source),
        "output_video": str(output),
        "events_csv": str(csv_path),
        "frames_read": frame_index,
        "frames_with_D40": frames_with_d40,
        "detection_events": total_events,
        "events_by_class": dict(counts),
        "confidence_threshold": args.conf,
        "tile_size": args.tile,
        "tile_overlap": args.overlap,
        "processing_seconds": round(time.time()-start, 2),
        "note": "Events are frame-level detections, not counts of unique potholes.",
    }
    report_path.write_text(json.dumps(report, indent=2))
    print("\nTILED VIDEO COMPLETE")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
