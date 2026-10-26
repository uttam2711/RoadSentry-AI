from pathlib import Path
import csv
import json
import cv2
from ultralytics import YOLO

ROOT = Path(".")
IMG_DIR = ROOT / "data/processed/roadsentry_real_eval_v1/images"
LBL_DIR = ROOT / "data/processed/roadsentry_real_eval_v1/labels"

MODELS = {
    "YOLO26n": ROOT / "runs/roadsentry/baseline_yolo26n_9class/weights/best.pt",
    "YOLO11s": ROOT / "runs/roadsentry/yolo11s_rdd_warmstart_9class_640_20261021/weights/best.pt",
}

CONF_THRESHOLDS = [0.15, 0.20, 0.25, 0.30, 0.35]
IOU_THRESHOLDS = [0.25, 0.50]
OUT_DIR = ROOT / "outputs/demo/real_road_eval_v1_updated_20261021"
OUT_DIR.mkdir(parents=True, exist_ok=True)

image_paths = sorted(
    p for p in IMG_DIR.iterdir()
    if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
)

if not image_paths:
    raise SystemExit(f"No images found in {IMG_DIR}")

# This folder is D40-only: every annotated box is treated as a D40 ground truth.
gt_by_stem = {}
missing = []

for img_path in image_paths:
    label_path = LBL_DIR / f"{img_path.stem}.txt"
    if not label_path.exists():
        missing.append(img_path.name)
        continue

    img = cv2.imread(str(img_path))
    if img is None:
        raise SystemExit(f"Cannot read image: {img_path}")

    h, w = img.shape[:2]
    boxes = []

    for line in label_path.read_text().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue

        _, xc, yc, bw, bh = map(float, parts[:5])
        boxes.append([
            (xc - bw / 2) * w,
            (yc - bh / 2) * h,
            (xc + bw / 2) * w,
            (yc + bh / 2) * h,
        ])

    gt_by_stem[img_path.stem] = boxes

if missing:
    raise SystemExit("Missing label files for: " + ", ".join(missing))

total_gt = sum(len(v) for v in gt_by_stem.values())
if total_gt == 0:
    raise SystemExit("No ground-truth boxes found. Check the annotation files.")

def box_iou(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0

def evaluate(pred_by_stem, conf_threshold, iou_threshold):
    tp = fp = fn = 0

    for stem, gt_boxes in gt_by_stem.items():
        preds = [
            p for p in pred_by_stem.get(stem, [])
            if p["conf"] >= conf_threshold
        ]
        preds.sort(key=lambda p: p["conf"], reverse=True)
        matched = set()

        for pred in preds:
            best_iou = 0.0
            best_idx = None

            for idx, gt in enumerate(gt_boxes):
                if idx in matched:
                    continue
                score = box_iou(pred["box"], gt)
                if score > best_iou:
                    best_iou = score
                    best_idx = idx

            if best_idx is not None and best_iou >= iou_threshold:
                tp += 1
                matched.add(best_idx)
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
device = 0  # Use the NVIDIA GPU

print(f"Images: {len(image_paths)}")
print(f"Annotated D40 boxes: {total_gt}")
print("Running inference; models and annotations will not be modified.")

for model_name, weights_path in MODELS.items():
    if not weights_path.exists():
        raise SystemExit(f"Checkpoint not found: {weights_path}")

    print(f"\nLoading {model_name}: {weights_path}")
    model = YOLO(str(weights_path))
    names = model.names
    items = names.items() if isinstance(names, dict) else enumerate(names)
    d40_ids = [
        int(idx) for idx, name in items
        if str(name).strip().upper() == "D40"
    ]

    if not d40_ids:
        raise SystemExit(
            f"{model_name} has no class named D40. Class names: {names}"
        )

    d40_id = d40_ids[0]
    print(f"D40 class index: {d40_id}")

    results = model.predict(
        source=[str(p) for p in image_paths],
        imgsz=640,
        conf=0.001,
        batch=8,
        device=device,
        max_det=300,
        verbose=False,
    )

    pred_by_stem = {p.stem: [] for p in image_paths}

    for result in results:
        stem = Path(result.path).stem
        if result.boxes is None:
            continue

        for box in result.boxes:
            cls_id = int(box.cls.item())
            if cls_id != d40_id:
                continue

            pred_by_stem[stem].append({
                "conf": float(box.conf.item()),
                "box": [float(x) for x in box.xyxy[0].tolist()],
            })

    for conf_threshold in CONF_THRESHOLDS:
        for iou_threshold in IOU_THRESHOLDS:
            metrics = evaluate(
                pred_by_stem, conf_threshold, iou_threshold
            )
            row = {
                "model": model_name,
                "confidence": conf_threshold,
                "match_IoU": iou_threshold,
                **metrics,
            }
            all_rows.append(row)

            print(
                f"{model_name:8s} conf={conf_threshold:.2f} "
                f"IoU={iou_threshold:.2f} "
                f"TP={metrics['TP']:3d} FP={metrics['FP']:3d} "
                f"FN={metrics['FN']:3d} "
                f"P={metrics['precision']:.3f} "
                f"R={metrics['recall']:.3f} "
                f"F1={metrics['F1']:.3f}"
            )

csv_path = OUT_DIR / "updated_metrics.csv"
json_path = OUT_DIR / "updated_metrics.json"

with csv_path.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=all_rows[0].keys())
    writer.writeheader()
    writer.writerows(all_rows)

json_path.write_text(json.dumps({
    "images": len(image_paths),
    "ground_truth_boxes": total_gt,
    "note": "Small diagnostic set from two videos; not an independent benchmark.",
    "results": all_rows,
}, indent=2))

print("\nEvaluation complete.")
print(f"CSV:  {csv_path}")
print(f"JSON: {json_path}")
