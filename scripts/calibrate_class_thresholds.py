from pathlib import Path
import csv
import json

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ROOT / "runs/roadsentry/baseline_yolo26n_9class/weights/best.pt"
IMAGES = ROOT / "data/processed/roadsentry_yolo_multicountry_v2/images/val"
LABELS = ROOT / "data/processed/roadsentry_yolo_multicountry_v2/labels/val"
OUTPUT = ROOT / "outputs/demo/threshold_calibration_640_20261016"

TARGET_IDS = [0, 1, 2, 3]
THRESHOLDS = [round(i / 100, 2) for i in range(5, 51, 5)]
BASELINE_THRESHOLD = 0.25
INFERENCE_CONFIDENCE_FLOOR = 0.001
INFERENCE_SIZE = 640
MATCH_IOU = 0.50


def box_iou(a, b):
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


def read_ground_truth(path, width, height, target_id):
    boxes = []
    if not path.is_file():
        raise FileNotFoundError(f"Missing label file: {path}")

    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue

        try:
            cls = int(float(parts[0]))
            xc, yc, bw, bh = map(float, parts[1:5])
        except ValueError:
            continue

        if cls != target_id:
            continue

        boxes.append({
            "x1": (xc - bw / 2) * width,
            "y1": (yc - bh / 2) * height,
            "x2": (xc + bw / 2) * width,
            "y2": (yc + bh / 2) * height,
        })

    return boxes


def match_counts(ground_truth, predictions, threshold):
    selected = [
        p for p in predictions if p["confidence"] >= threshold
    ]

    possible_matches = []
    for gi, gt in enumerate(ground_truth):
        for pi, pred in enumerate(selected):
            overlap = box_iou(gt, pred)
            if overlap >= MATCH_IOU:
                possible_matches.append((overlap, gi, pi))

    # Greedily match the highest-IoU pairs, one prediction per GT.
    possible_matches.sort(reverse=True)
    matched_gt = set()
    matched_pred = set()

    for _, gi, pi in possible_matches:
        if gi in matched_gt or pi in matched_pred:
            continue
        matched_gt.add(gi)
        matched_pred.add(pi)

    tp = len(matched_gt)
    fp = len(selected) - len(matched_pred)
    fn = len(ground_truth) - len(matched_gt)
    return tp, fp, fn


def metrics_from_counts(tp, fp, fn):
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0.0
    )
    return {
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def main():
    for path in (WEIGHTS, IMAGES, LABELS):
        if not path.exists():
            raise FileNotFoundError(path)

    if OUTPUT.exists():
        raise FileExistsError(
            f"Output already exists: {OUTPUT}. "
            "Rename OUTPUT in this script to preserve prior results."
        )

    model = YOLO(str(WEIGHTS))
    names = model.names

    image_paths = sorted(
        p for p in IMAGES.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if not image_paths:
        raise RuntimeError(f"No validation images found in {IMAGES}")

    # Independent counts for every class and confidence threshold.
    counts = {
        (cls, threshold): [0, 0, 0]
        for cls in TARGET_IDS
        for threshold in THRESHOLDS
    }

    print("=== ROAD SENTRY CLASS THRESHOLD CALIBRATION ===")
    print("Checkpoint:", WEIGHTS)
    print("Images:", IMAGES)
    print("Validation images:", len(image_paths))
    print("Inference size:", INFERENCE_SIZE)
    print("Inference confidence floor:", INFERENCE_CONFIDENCE_FLOOR)
    print("Matching IoU:", MATCH_IOU)
    print("Thresholds:", THRESHOLDS)
    print("Classes:", [names[c] for c in TARGET_IDS])
    print("Training: NO")

    results = model.predict(
        source=str(IMAGES),
        imgsz=INFERENCE_SIZE,
        conf=INFERENCE_CONFIDENCE_FLOOR,
        iou=0.7,
        max_det=300,
        classes=TARGET_IDS,
        batch=8,
        device=0,
        stream=True,
        verbose=False,
        save=False,
    )

    processed = 0
    for result in results:
        processed += 1
        image_path = Path(result.path)
        label_path = LABELS / f"{image_path.stem}.txt"
        height, width = result.orig_shape

        predictions_by_class = {cls: [] for cls in TARGET_IDS}
        if result.boxes is not None and len(result.boxes):
            coords = result.boxes.xyxy.cpu().tolist()
            classes = result.boxes.cls.cpu().tolist()
            scores = result.boxes.conf.cpu().tolist()

            for xyxy, cls, score in zip(coords, classes, scores):
                cls = int(cls)
                if cls not in predictions_by_class:
                    continue
                predictions_by_class[cls].append({
                    "x1": float(xyxy[0]),
                    "y1": float(xyxy[1]),
                    "x2": float(xyxy[2]),
                    "y2": float(xyxy[3]),
                    "confidence": float(score),
                })

        for cls in TARGET_IDS:
            gt = read_ground_truth(label_path, width, height, cls)
            predictions = predictions_by_class[cls]

            for threshold in THRESHOLDS:
                tp, fp, fn = match_counts(gt, predictions, threshold)
                running = counts[(cls, threshold)]
                running[0] += tp
                running[1] += fp
                running[2] += fn

        if processed % 250 == 0 or processed == len(image_paths):
            print(f"Processed {processed}/{len(image_paths)}")

    rows = []
    recommendations = []

    for cls in TARGET_IDS:
        class_name = (
            names.get(cls, str(cls))
            if isinstance(names, dict) else names[cls]
        )
        class_rows = []

        for threshold in THRESHOLDS:
            tp, fp, fn = counts[(cls, threshold)]
            row = {
                "class_id": cls,
                "class_name": str(class_name),
                "threshold": threshold,
                **metrics_from_counts(tp, fp, fn),
            }
            class_rows.append(row)
            rows.append(row)

        baseline = next(
            row for row in class_rows
            if abs(row["threshold"] - BASELINE_THRESHOLD) < 1e-9
        )

        # Maximize F1, breaking ties by precision, then higher threshold.
        recommended = max(
            class_rows,
            key=lambda row: (
                row["f1"], row["precision"], row["threshold"]
            ),
        )

        recommendations.append({
            "class_id": cls,
            "class_name": str(class_name),
            "baseline_threshold": BASELINE_THRESHOLD,
            "baseline_at_0_25": baseline,
            "best_f1_threshold_on_validation": recommended["threshold"],
            "best_f1_metrics": recommended,
        })

    OUTPUT.mkdir(parents=True)
    csv_path = OUTPUT / "threshold_sweep_metrics.csv"
    json_path = OUTPUT / "threshold_recommendations.json"

    fields = [
        "class_id", "class_name", "threshold",
        "true_positives", "false_positives", "false_negatives",
        "precision", "recall", "f1",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "checkpoint": str(WEIGHTS),
        "validation_images": len(image_paths),
        "image_size": INFERENCE_SIZE,
        "matching_iou": MATCH_IOU,
        "confidence_floor_for_inference": INFERENCE_CONFIDENCE_FLOOR,
        "thresholds_tested": THRESHOLDS,
        "selection_method": (
            "Maximize validation F1; break ties by precision, then threshold."
        ),
        "warning": (
            "Thresholds are calibrated on this validation split, so the "
            "best-F1 results may be optimistic. Verify selected thresholds "
            "on an independent test set before final performance claims."
        ),
        "recommendations": recommendations,
    }
    json_path.write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )

    print("\n=== THRESHOLD RESULTS ===")
    print("Class | Best threshold | P | R | F1 | Baseline F1 at 0.25")
    for item in recommendations:
        best = item["best_f1_metrics"]
        baseline = item["baseline_at_0_25"]
        print(
            f'{item["class_name"]} | '
            f'{item["best_f1_threshold_on_validation"]:.2f} | '
            f'{best["precision"]:.3f} | {best["recall"]:.3f} | '
            f'{best["f1"]:.3f} | {baseline["f1"]:.3f}'
        )

    print("\nCSV:", csv_path)
    print("JSON:", json_path)
    print("Calibration complete. The model checkpoint was not changed.")


if __name__ == "__main__":
    main()
