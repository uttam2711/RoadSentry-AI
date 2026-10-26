from pathlib import Path
import argparse
import csv
import json
import time

import cv2
import torch
from ultralytics import YOLO


def main():
    root = Path(__file__).resolve().parents[1]
    default_weights = (
        root / "runs/roadsentry/baseline_yolo26n_9class/weights/best.pt"
    )

    parser = argparse.ArgumentParser(
        description="Inspect a road video and export annotated video and events."
    )
    parser.add_argument("--source", required=True, help="Input video path")
    parser.add_argument("--weights", default=str(default_weights))
    parser.add_argument(
        "--output", default=str(root / "outputs/demo/video_inspection")
    )
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument(
        "--frame-step", type=int, default=1,
        help="Analyze every Nth frame; 1 means every frame"
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--d40-conf", type=float, default=0.20,
                        help="Experimental confidence threshold for D40 potholes")
    args = parser.parse_args()
    class_thresholds = {
        0: min(0.20, args.conf),
        1: min(0.20, args.conf),
        2: args.conf,
        3: args.d40_conf,
    }

    source = Path(args.source).expanduser().resolve()
    weights = Path(args.weights).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()

    if not source.is_file():
        raise FileNotFoundError(f"Video not found: {source}")
    if not weights.is_file():
        raise FileNotFoundError(f"Model weights not found: {weights}")
    if not 0 <= args.conf <= 1:
        parser.error("--conf must be between 0 and 1")
    if not 0 <= args.d40_conf <= 1:
        parser.error("--d40-conf must be between 0 and 1")
    if args.imgsz < 32:
        parser.error("--imgsz must be at least 32")
    if args.frame_step < 1:
        parser.error("--frame-step must be at least 1")

    output.mkdir(parents=True, exist_ok=True)
    device = args.device
    if device is None:
        device = 0 if torch.cuda.is_available() else "cpu"

    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {source}")

    fps = capture.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 25.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    reported_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    output_video = output / f"{source.stem}_roadsentry.mp4"
    writer = cv2.VideoWriter(
        str(output_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Could not create output video: {output_video}")

    model = YOLO(str(weights))
    events = []
    frame_index = 0
    analyzed_frames = 0
    detected_frames = 0
    start_time = time.perf_counter()

    print("Input:", source)
    print("Model:", weights)
    print("Video dimensions:", f"{width}x{height}")
    print("Input FPS:", round(fps, 3))
    print("Reported frames:", reported_frames)
    print("Frame step:", args.frame_step)
    print("Device:", device)

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            if frame_index % args.frame_step == 0:
                prediction = model.predict(
                    source=frame,
                    imgsz=args.imgsz,
                    conf=min(class_thresholds.values()),
                    classes=[0, 1, 2, 3],
                    device=device,
                    verbose=False,
                    save=False,
                )[0]
                if prediction.boxes is not None and len(prediction.boxes):
                    boxes = prediction.boxes
                    class_ids = boxes.cls.to(dtype=torch.int64).tolist()
                    limits = torch.tensor(
                        [class_thresholds[int(cid)] for cid in class_ids],
                        device=boxes.conf.device, dtype=boxes.conf.dtype,
                    )
                    prediction.boxes = boxes[boxes.conf >= limits]
                analyzed_frames += 1
                frame_ms = frame_index * 1000.0 / fps

                if prediction.boxes is not None and len(prediction.boxes):
                    detected_frames += 1
                    boxes = prediction.boxes
                    coords = boxes.xyxy.cpu().tolist()
                    classes = boxes.cls.cpu().tolist()
                    scores = boxes.conf.cpu().tolist()

                    for xyxy, class_id, score in zip(coords, classes, scores):
                        class_id = int(class_id)
                        events.append({
                            "frame": frame_index,
                            "timestamp_seconds": round(frame_index / fps, 3),
                            "timestamp_ms": round(frame_ms, 1),
                            "class_id": class_id,
                            "class_name": model.names[class_id],
                            "confidence": round(float(score), 6),
                            "x1": round(float(xyxy[0]), 2),
                            "y1": round(float(xyxy[1]), 2),
                            "x2": round(float(xyxy[2]), 2),
                            "y2": round(float(xyxy[3]), 2),
                        })

                frame = prediction.plot()

                # Keep output dimensions consistent with the input video.
                if frame.shape[1] != width or frame.shape[0] != height:
                    frame = cv2.resize(frame, (width, height))

                cv2.putText(
                    frame,
                    f"RoadSentry | t={frame_index / fps:.1f}s "
                    f"| detections={len(events)}",
                    (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

            writer.write(frame)
            frame_index += 1
    finally:
        capture.release()
        writer.release()

    elapsed = time.perf_counter() - start_time
    event_csv = output / f"{source.stem}_events.csv"
    event_json = output / f"{source.stem}_report.json"

    fieldnames = [
        "frame", "timestamp_seconds", "timestamp_ms",
        "class_id", "class_name", "confidence",
        "x1", "y1", "x2", "y2",
    ]
    with event_csv.open("w", newline="", encoding="utf-8") as f:
        writer_csv = csv.DictWriter(f, fieldnames=fieldnames)
        writer_csv.writeheader()
        writer_csv.writerows(events)

    report = {
        "source_video": str(source),
        "model_weights": str(weights),
        "confidence_threshold_floor": float(min(class_thresholds.values())),
        "class_confidence_thresholds": {
            str(model.names[cid]): float(value)
            for cid, value in class_thresholds.items()
        },
        "image_size": args.imgsz,
        "frame_step": args.frame_step,
        "device": str(device),
        "input_fps": round(fps, 3),
        "video_width": width,
        "video_height": height,
        "total_frames_read": frame_index,
        "analyzed_frames": analyzed_frames,
        "frames_with_detections": detected_frames,
        "detection_event_count": len(events),
        "processing_seconds": round(elapsed, 3),
        "analyzed_frames_per_second": round(
            analyzed_frames / elapsed, 3
        ) if elapsed else None,
        "output_video": str(output_video),
        "events_csv": str(event_csv),
        "events": events,
        "note": (
            "Events are per-frame detections, not unique physical defects. "
            "Repeated detections across frames may refer to the same damage."
        ),
    }
    event_json.write_text(json.dumps(report, indent=2) + "\n")

    print("\n=== INSPECTION COMPLETE ===")
    print("Frames read:", frame_index)
    print("Frames analyzed:", analyzed_frames)
    print("Frames with detections:", detected_frames)
    print("Detection events:", len(events))
    print("Processing seconds:", round(elapsed, 2))
    print("Annotated video:", output_video)
    print("Events CSV:", event_csv)
    print("JSON report:", event_json)


if __name__ == "__main__":
    main()
