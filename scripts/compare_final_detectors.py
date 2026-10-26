from pathlib import Path
import csv
import json
import cv2

from ultralytics import YOLO

ROOT = Path.cwd()
IMG_DIR = ROOT / "data/processed/roadsentry_real_eval_v1/images"
LBL_DIR = ROOT / "data/processed/roadsentry_real_eval_v1/labels"
OUT = ROOT / "outputs/final_delivery/model_comparison_realroad"
OUT.mkdir(parents=True, exist_ok=True)

MODELS = {
    "YOLO26n": (
        "runs/roadsentry/baseline_yolo26n_9class/weights/best.pt",
        "D40",
    ),
    "YOLO11s": (
        "runs/roadsentry/yolo11s_rdd_warmstart_9class_640_20261021/weights/best.pt",
        "D40",
    ),
    "YOLO-World": ("yolov8s-worldv2.pt", "pothole"),
}

CONFIDENCES = [0.03, 0.05, 0.10, 0.15, 0.20, 0.25]
IOUS = [0.25, 0.50]

images = sorted(
    p for p in IMG_DIR.iterdir()
    if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
)

if not images:
    raise SystemExit(f"No images found: {IMG_DIR}")

ground_truth = {}
image_shapes = {}

for path in images:
    image = cv2.imread(str(path))
    if image is None:
        raise SystemExit(f"Cannot read image: {path}")

    h, w = image.shape[:2]
    image_shapes[path.stem] = (w, h)
    label_path = LBL_DIR / f"{path.stem}.txt"

    if not label_path.exists():
        raise SystemExit(f"Missing label file: {label_path}")

    boxes = []
    for line in label_path.read_text().splitlines():
        parts = line.split()
        if not parts:
            continue
        if len(parts) != 5:
            raise SystemExit(f"Invalid YOLO label in {label_path}: {line}")

        _, xc, yc, bw, bh = map(float, parts)
        boxes.append([
            (xc - bw / 2) * w,
            (yc - bh / 2) * h,
            (xc + bw / 2) * w,
            (yc + bh / 2) * h,
        ])

    ground_truth[path.stem] = boxes

total_gt = sum(len(v) for v in ground_truth.values())
if total_gt == 0:
    raise SystemExit("No ground-truth boxes found.")

def box_iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0

def evaluate(predictions, confidence, match_iou):
    tp = fp = fn = 0

    for stem, gt_boxes in ground_truth.items():
        candidates = sorted(
            [p for p in predictions.get(stem, [])
             if p["confidence"] >= confidence],
            key=lambda p: p["confidence"],
            reverse=True,
        )

        matched = set()
        for prediction in candidates:
            options = [
                (box_iou(prediction["box"], gt), index)
                for index, gt in enumerate(gt_boxes)
                if index not in matched
            ]
            score, index = max(options, default=(0.0, -1))

            if index >= 0 and score >= match_iou:
                tp += 1
                matched.add(index)
            else:
                fp += 1

        fn += len(gt_boxes) - len(matched)

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0.0
    )

    return {
        "TP": tp, "FP": fp, "FN": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "F1": round(f1, 4),
    }

all_rows = []
successful_models = []

print(f"Images: {len(images)} | Annotated boxes: {total_gt}")
print("Running inference on GPU; no training will occur.")

for model_name, (weights, prompt) in MODELS.items():
    print(f"\n--- {model_name} ---")

    try:
        if model_name == "YOLO-World":
            from ultralytics import YOLOWorld
            model = YOLOWorld(weights)
            model.set_classes([prompt])
            target_ids = {0}
        else:
            weight_path = ROOT / weights
            if not weight_path.exists():
                print(f"SKIPPED: checkpoint missing: {weight_path}")
                continue

            model = YOLO(str(weight_path))
            names = model.names
            items = names.items() if isinstance(names, dict) else enumerate(names)
            target_ids = {
                int(i) for i, name in items
                if str(name).strip().lower() == prompt.lower()
            }

            if not target_ids:
                print(f"SKIPPED: class {prompt!r} not found: {names}")
                continue

        results = model.predict(
            source=[str(p) for p in images],
            imgsz=640,
            conf=0.001,
            batch=4,
            device=0,
            max_det=300,
            verbose=False,
        )

        predictions = {p.stem: [] for p in images}
        for result in results:
            stem = Path(result.path).stem
            if result.boxes is None:
                continue

            for box in result.boxes:
                if int(box.cls.item()) not in target_ids:
                    continue
                predictions[stem].append({
                    "confidence": float(box.conf.item()),
                    "box": [float(v) for v in box.xyxy[0].tolist()],
                })

        successful_models.append(model_name)

        for threshold in CONFIDENCES:
            for match_iou in IOUS:
                metrics = evaluate(predictions, threshold, match_iou)
                row = {
                    "model": model_name,
                    "confidence": threshold,
                    "match_IoU": match_iou,
                    **metrics,
                }
                all_rows.append(row)
                print(
                    f"conf={threshold:.2f} IoU={match_iou:.2f} "
                    f"TP={metrics['TP']} FP={metrics['FP']} "
                    f"FN={metrics['FN']} P={metrics['precision']:.3f} "
                    f"R={metrics['recall']:.3f} F1={metrics['F1']:.3f}"
                )

        # Save one visual comparison contact sheet per model.
        tiles = []
        overlay_dir = OUT / f"overlays_{model_name.lower().replace('-', '_')}"
        overlay_dir.mkdir(exist_ok=True)

        for path in images:
            image = cv2.imread(str(path))
            h, w = image.shape[:2]
            canvas = image.copy()

            # Ground truth in green.
            for gt in ground_truth[path.stem]:
                cv2.rectangle(
                    canvas,
                    (int(gt[0]), int(gt[1])),
                    (int(gt[2]), int(gt[3])),
                    (0, 255, 0), 2,
                )

            # Predictions in red.
            shown = 0
            for pred in predictions[path.stem]:
                if pred["confidence"] < 0.10:
                    continue
                b = pred["box"]
                cv2.rectangle(
                    canvas,
                    (int(b[0]), int(b[1])),
                    (int(b[2]), int(b[3])),
                    (0, 0, 255), 2,
                )
                shown += 1

            cv2.putText(
                canvas,
                f"GT={len(ground_truth[path.stem])} Pred@0.10={shown}",
                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255, 255, 255), 2,
            )
            cv2.imwrite(
                str(overlay_dir / f"{path.stem}_overlay.jpg"), canvas
            )

            thumb = cv2.resize(canvas, (384, 216))
            tile = cv2.copyMakeBorder(
                thumb, 26, 0, 0, 0, cv2.BORDER_CONSTANT,
                value=(245, 245, 245),
            )
            cv2.putText(
                tile, path.stem, (6, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (20, 20, 20), 1,
            )
            tiles.append(tile)

        cols = 3
        rows = (len(tiles) + cols - 1) // cols
        sheet = cv2.copyMakeBorder(
            cv2.vconcat([
                cv2.hconcat(tiles[i:i+cols])
                for i in range(0, len(tiles), cols)
            ]),
            0, 0, 0, 0, cv2.BORDER_CONSTANT, value=(230, 230, 230)
        )
        cv2.imwrite(str(OUT / f"contact_sheet_{model_name.lower().replace('-', '_')}.jpg"), sheet)

    except Exception as exc:
        print(f"MODEL FAILED: {type(exc).__name__}: {exc}")
        if model_name == "YOLO-World":
            print("Continue with existing models; do not start another installation.")

if not all_rows:
    raise SystemExit("No detector completed. Check the errors above.")

csv_path = OUT / "metrics.csv"
json_path = OUT / "metrics.json"

with csv_path.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=all_rows[0].keys())
    writer.writeheader()
    writer.writerows(all_rows)

json_path.write_text(json.dumps({
    "images": len(images),
    "ground_truth_boxes": total_gt,
    "models_completed": successful_models,
    "note": "Small diagnostic sample from two videos, not an independent benchmark.",
    "results": all_rows,
}, indent=2))

print("\nCOMPARISON FINISHED")
print(f"Models completed: {successful_models}")
print(f"Metrics CSV: {csv_path}")
print(f"Metrics JSON: {json_path}")
print(f"Visual contact sheets: {OUT}")
