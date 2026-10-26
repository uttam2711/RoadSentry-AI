from pathlib import Path
from ultralytics import YOLOWorld
import json

img_dir = Path("data/processed/roadsentry_real_eval_v1/images")
images = sorted(
    p for p in img_dir.iterdir()
    if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
)

if not images:
    raise SystemExit(f"No images found: {img_dir}")

prompts = [
    "pothole",
    "road pothole",
    "water-filled pothole",
    "hole in the road",
    "road surface damage",
    "damaged road surface",
]

print(f"Images: {len(images)}")
print("Loading the already-downloaded YOLO-World model...")
model = YOLOWorld("yolov8s-worldv2.pt")
reports = []

for prompt in prompts:
    print(f"\nPrompt: {prompt}")
    model.set_classes([prompt])

    results = model.predict(
        source=[str(p) for p in images],
        imgsz=640,
        conf=0.001,
        batch=4,
        device=0,
        max_det=100,
        verbose=False,
    )

    scores = []
    frames_with_detections = 0

    for result in results:
        frame_scores = []
        if result.boxes is not None:
            frame_scores = [
                float(box.conf.item()) for box in result.boxes
                if int(box.cls.item()) == 0
            ]
        scores.extend(frame_scores)
        if frame_scores:
            frames_with_detections += 1

    counts = {
        str(threshold): sum(score >= threshold for score in scores)
        for threshold in (0.001, 0.01, 0.03, 0.05, 0.10, 0.20)
    }

    row = {
        "prompt": prompt,
        "total_detections_at_0.001": len(scores),
        "frames_with_detections": frames_with_detections,
        "max_confidence": round(max(scores), 5) if scores else 0,
        "detections_by_threshold": counts,
    }
    reports.append(row)

    print(f"Total detections: {len(scores)}")
    print(f"Frames with detections: {frames_with_detections}/{len(images)}")
    print(f"Maximum confidence: {row['max_confidence']}")
    print(f"Counts by threshold: {counts}")

out = Path("outputs/final_delivery/model_comparison_realroad")
out.mkdir(parents=True, exist_ok=True)
path = out / "yoloworld_prompt_diagnostic.json"
path.write_text(json.dumps(reports, indent=2))
print(f"\nSaved: {path}")
