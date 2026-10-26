from pathlib import Path
import json
import random

import cv2
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/processed/roadsentry_yolo_multicountry_v2"
WEIGHTS = ROOT / "runs/roadsentry/baseline_yolo26n_9class/weights/best.pt"
OUTPUT = ROOT / "outputs/demo/validation_examples"

CLASS_NAMES = {0: "D00", 1: "D10", 2: "D20", 3: "D40"}
MAX_IMAGES_PER_CLASS = 80
SEED = 2026
IMAGE_SIZE = 960
CONFIDENCE = 0.25


def read_ground_truth(label_path, width, height):
    boxes = []
    if not label_path.is_file():
        return boxes

    for line in label_path.read_text().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue

        try:
            cls = int(float(parts[0]))
            xc, yc, bw, bh = map(float, parts[1:5])
        except ValueError:
            continue

        if cls not in CLASS_NAMES:
            continue

        boxes.append({
            "class_id": cls,
            "class_name": CLASS_NAMES[cls],
            "x1": (xc - bw / 2) * width,
            "y1": (yc - bh / 2) * height,
            "x2": (xc + bw / 2) * width,
            "y2": (yc + bh / 2) * height,
        })
    return boxes


def iou(a, b):
    x1 = max(a["x1"], b["x1"])
    y1 = max(a["y1"], b["y1"])
    x2 = min(a["x2"], b["x2"])
    y2 = min(a["y2"], b["y2"])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = max(0, a["x2"] - a["x1"]) * max(0, a["y2"] - a["y1"])
    area_b = max(0, b["x2"] - b["x1"]) * max(0, b["y2"] - b["y1"])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def draw_box(image, box, color, label):
    x1, y1 = round(box["x1"]), round(box["y1"])
    x2, y2 = round(box["x2"]), round(box["y2"])
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)
    y_text = max(24, y1 - 8)
    cv2.putText(
        image, label, (max(3, x1), y_text),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA
    )


def main():
    images_dir = DATA / "images/val"
    labels_dir = DATA / "labels/val"

    for path in (images_dir, labels_dir, WEIGHTS):
        if not path.exists():
            raise FileNotFoundError(path)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)
    images_by_class = {c: [] for c in CLASS_NAMES}

    for image_path in images_dir.iterdir():
        if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue

        label_path = labels_dir / f"{image_path.stem}.txt"
        if not label_path.is_file():
            continue

        found_classes = set()
        for line in label_path.read_text().splitlines():
            parts = line.split()
            if not parts:
                continue
            try:
                cls = int(float(parts[0]))
            except ValueError:
                continue
            if cls in CLASS_NAMES:
                found_classes.add(cls)

        for cls in found_classes:
            images_by_class[cls].append(image_path)

    selected = set()
    print("=== VALIDATION CANDIDATES ===")
    for cls, name in CLASS_NAMES.items():
        candidates = images_by_class[cls]
        rng.shuffle(candidates)
        chosen = candidates[:MAX_IMAGES_PER_CLASS]
        selected.update(chosen)
        print(
            f"{name}: {len(candidates)} positive images available; "
            f"sampling {len(chosen)}"
        )

    selected = sorted(selected)
    print("\nUnique images to evaluate:", len(selected))
    print("Image size:", IMAGE_SIZE, "| Confidence threshold:", CONFIDENCE)

    model = YOLO(str(WEIGHTS))
    best = {cls: None for cls in CLASS_NAMES}

    for index, image_path in enumerate(selected, start=1):
        image = cv2.imread(str(image_path))
        if image is None:
            print("Skipping unreadable image:", image_path.name)
            continue

        height, width = image.shape[:2]
        label_path = labels_dir / f"{image_path.stem}.txt"
        gt_boxes = read_ground_truth(label_path, width, height)
        gt_boxes = [b for b in gt_boxes if b["class_id"] in CLASS_NAMES]

        result = model.predict(
            source=image,
            imgsz=IMAGE_SIZE,
            conf=CONFIDENCE,
            classes=list(CLASS_NAMES),
            device=0,
            verbose=False,
        )[0]

        predictions = []
        if result.boxes is not None and len(result.boxes):
            coords = result.boxes.xyxy.cpu().tolist()
            classes = result.boxes.cls.cpu().tolist()
            scores = result.boxes.conf.cpu().tolist()

            for xyxy, cls, score in zip(coords, classes, scores):
                cls = int(cls)
                if cls not in CLASS_NAMES:
                    continue
                predictions.append({
                    "class_id": cls,
                    "class_name": CLASS_NAMES[cls],
                    "confidence": float(score),
                    "x1": xyxy[0], "y1": xyxy[1],
                    "x2": xyxy[2], "y2": xyxy[3],
                })

        for gt in gt_boxes:
            matching = [
                pred for pred in predictions
                if pred["class_id"] == gt["class_id"]
            ]
            if not matching:
                continue

            pred = max(matching, key=lambda item: iou(gt, item))
            overlap = iou(gt, pred)
            cls = gt["class_id"]

            candidate = {
                "class_id": cls,
                "class_name": CLASS_NAMES[cls],
                "image": str(image_path),
                "confidence": round(pred["confidence"], 5),
                "iou": round(overlap, 5),
                "ground_truth": gt,
                "prediction": pred,
            }

            current = best[cls]
            if current is None or (
                candidate["iou"], candidate["confidence"]
            ) > (current["iou"], current["confidence"]):
                best[cls] = candidate

        if index % 25 == 0 or index == len(selected):
            print(f"Processed {index}/{len(selected)} images")

    saved = []
    for cls, name in CLASS_NAMES.items():
        candidate = best[cls]
        if candidate is None:
            print(f"\n{name}: no same-class prediction found in the sample")
            continue

        image = cv2.imread(candidate["image"])
        gt = candidate["ground_truth"]
        pred = candidate["prediction"]

        # Green = ground truth; magenta = model prediction.
        draw_box(image, gt, (0, 220, 0), f"GT {name}")
        draw_box(
            image, pred, (220, 0, 220),
            f"Pred {name} {pred['confidence']:.2f} | IoU {candidate['iou']:.2f}"
        )

        output_path = OUTPUT / f"candidate_{name}.jpg"
        if not cv2.imwrite(str(output_path), image):
            raise RuntimeError(f"Could not save {output_path}")

        candidate["output_image"] = str(output_path)
        saved.append(candidate)

        print(
            f"\n{name}: IoU={candidate['iou']:.3f}, "
            f"confidence={candidate['confidence']:.3f}"
        )
        print("Source:", Path(candidate["image"]).name)
        print("Evidence:", output_path)

    summary_path = OUTPUT / "validation_example_summary.json"
    summary_path.write_text(
        json.dumps({
            "weights": str(WEIGHTS),
            "validation_images_sampled": len(selected),
            "sample_seed": SEED,
            "image_size": IMAGE_SIZE,
            "confidence_threshold": CONFIDENCE,
            "selection_note": (
                "Best same-class IoU example found in the sampled validation "
                "images. This is curated evidence, not a full validation metric."
            ),
            "examples": saved,
        }, indent=2) + "\n",
        encoding="utf-8",
    )

    print("\nSummary:", summary_path)
    print("Examples saved:", len(saved))


if __name__ == "__main__":
    main()
