from pathlib import Path
import csv
import cv2

from ultralytics import YOLOWorld

ROOT = Path("data/raw/external_eval/independent_sample_v1")
FRAME_DIR = ROOT / "frames"
OUT_DIR = ROOT / "yoloworld_proposals"
OVERLAY_DIR = OUT_DIR / "overlays"

PROMPTS = [
    "pothole",
    "road crack",
    "alligator cracking on asphalt",
    "damaged asphalt road surface",
]

if not FRAME_DIR.is_dir():
    raise SystemExit(f"Frame folder missing: {FRAME_DIR}")

if OUT_DIR.exists():
    raise SystemExit(
        f"Output already exists: {OUT_DIR}\n"
        "Refusing to overwrite previous proposals."
    )

frames = sorted(FRAME_DIR.glob("frame_*s.jpg"))
if not frames:
    raise SystemExit(f"No sampled frames found in {FRAME_DIR}")

OUT_DIR.mkdir(parents=True)
OVERLAY_DIR.mkdir()

print("Loading YOLO-World. First run may download its weights.")
model = YOLOWorld("yolov8s-world.pt")
model.set_classes(PROMPTS)

csv_path = OUT_DIR / "candidate_proposals.csv"
total = 0

with csv_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow([
        "frame", "prompt", "confidence",
        "x1", "y1", "x2", "y2",
    ])

    for frame_path in frames:
        image = cv2.imread(str(frame_path))
        if image is None:
            print("Skipping unreadable frame:", frame_path.name)
            continue

        result = model.predict(
            source=image,
            imgsz=960,
            conf=0.05,
            iou=0.50,
            max_det=100,
            device=0,
            verbose=False,
        )[0]

        canvas = image.copy()
        count = 0
        boxes = result.boxes

        if boxes is not None:
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            classes = boxes.cls.cpu().numpy().astype(int)

            for box, score, cls in zip(xyxy, confs, classes):
                x1, y1, x2, y2 = map(int, box)
                prompt = PROMPTS[cls] if 0 <= cls < len(PROMPTS) else str(cls)

                cv2.rectangle(
                    canvas, (x1, y1), (x2, y2), (0, 165, 255), 3
                )
                cv2.putText(
                    canvas,
                    f"{prompt} {score:.2f}",
                    (x1, max(22, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 165, 255),
                    2,
                )

                writer.writerow([
                    frame_path.name, prompt, f"{float(score):.5f}",
                    x1, y1, x2, y2,
                ])
                count += 1
                total += 1

        cv2.imwrite(
            str(OVERLAY_DIR / frame_path.name), canvas
        )
        print(f"{frame_path.name}: {count} proposals", flush=True)

print("\nProposal generation complete.")
print("Frames:", len(frames))
print("Total proposals:", total)
print("Overlay folder:", OVERLAY_DIR.resolve())
print("Candidate list:", csv_path.resolve())
print("RoadSentry checkpoint and ground-truth labels were not changed.")
