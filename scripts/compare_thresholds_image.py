from pathlib import Path
import argparse
import json

import cv2
import torch
from ultralytics import YOLO


def collect_detections(result, model_names, thresholds=None):
    detections = []
    boxes = result.boxes

    if boxes is None or len(boxes) == 0:
        return detections

    coordinates = boxes.xyxy.cpu().tolist()
    class_ids = boxes.cls.cpu().tolist()
    confidences = boxes.conf.cpu().tolist()

    for xyxy, class_id, score in zip(coordinates, class_ids, confidences):
        class_id = int(class_id)
        class_name = str(model_names[class_id])
        score = float(score)

        if thresholds is not None:
            minimum = thresholds.get(class_name, 0.25)
            if score < minimum:
                continue

        detections.append({
            "class_id": class_id,
            "class_name": class_name,
            "confidence": round(score, 6),
            "bbox_xyxy": [round(float(v), 2) for v in xyxy],
        })

    return detections


def save_annotated(source, detections, destination):
    image = cv2.imread(str(source))
    if image is None:
        raise RuntimeError(f"Could not read image: {source}")

    class_colors = {
        0: (0, 200, 255),
        1: (255, 180, 0),
        2: (0, 220, 0),
        3: (255, 0, 220),
    }

    for det in detections:
        x1, y1, x2, y2 = map(int, det["bbox_xyxy"])
        class_id = det["class_id"]
        color = class_colors.get(class_id, (255, 255, 255))
        label = f'{det["class_name"]} {det["confidence"]:.2f}'

        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        text_y = max(20, y1 - 8)
        cv2.putText(
            image, label, (x1, text_y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(destination), image):
        raise RuntimeError(f"Could not save image: {destination}")


def main():
    root = Path(__file__).resolve().parents[1]

    parser = argparse.ArgumentParser(
        description="Compare baseline and experimental class thresholds on one image."
    )
    parser.add_argument("--source", required=True, help="Input image path")
    parser.add_argument(
        "--weights",
        default=str(root / "runs/roadsentry/baseline_yolo26n_9class/weights/best.pt")
    )
    parser.add_argument(
        "--config",
        default=str(root / "configs/class_thresholds_experimental_640.json")
    )
    parser.add_argument(
        "--output",
        default=str(root / "outputs/demo/threshold_image_comparison")
    )
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    source = Path(args.source).expanduser().resolve()
    weights = Path(args.weights).expanduser().resolve()
    config_path = Path(args.config).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()

    for path in (source, weights, config_path):
        if not path.is_file():
            parser.error(f"File not found: {path}")

    config = json.loads(config_path.read_text(encoding="utf-8"))
    thresholds = config["class_thresholds"]
    baseline_conf = float(config["baseline_global_threshold"])
    minimum_conf = min(thresholds.values())

    if args.imgsz < 32:
        parser.error("--imgsz must be at least 32")

    device = args.device
    if device is None:
        device = 0 if torch.cuda.is_available() else "cpu"

    model = YOLO(str(weights))
    target_classes = [0, 1, 2, 3]

    baseline_result = model.predict(
        source=str(source),
        imgsz=args.imgsz,
        conf=baseline_conf,
        classes=target_classes,
        device=device,
        verbose=False,
        save=False,
    )[0]

    experimental_result = model.predict(
        source=str(source),
        imgsz=args.imgsz,
        conf=minimum_conf,
        classes=target_classes,
        device=device,
        verbose=False,
        save=False,
    )[0]

    baseline = collect_detections(baseline_result, model.names)
    experimental = collect_detections(
        experimental_result, model.names, thresholds=thresholds
    )

    output.mkdir(parents=True, exist_ok=True)
    baseline_image = output / f"{source.stem}_baseline_conf025.jpg"
    experimental_image = output / f"{source.stem}_experimental_thresholds.jpg"

    save_annotated(source, baseline, baseline_image)
    save_annotated(source, experimental, experimental_image)

    def counts_by_class(items):
        counts = {}
        for item in items:
            name = item["class_name"]
            counts[name] = counts.get(name, 0) + 1
        return counts

    report = {
        "source_image": str(source),
        "weights": str(weights),
        "image_size": args.imgsz,
        "device": str(device),
        "baseline_global_threshold": baseline_conf,
        "experimental_class_thresholds": thresholds,
        "baseline_detection_count": len(baseline),
        "experimental_detection_count": len(experimental),
        "baseline_counts_by_class": counts_by_class(baseline),
        "experimental_counts_by_class": counts_by_class(experimental),
        "baseline_detections": baseline,
        "experimental_detections": experimental,
        "baseline_annotated_image": str(baseline_image),
        "experimental_annotated_image": str(experimental_image),
        "warning": (
            "This is an experimental visual comparison, not an independent "
            "accuracy evaluation. Thresholds were explored on validation data."
        ),
    }

    json_path = output / f"{source.stem}_comparison.json"
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print("\n=== THRESHOLD COMPARISON ===")
    print("Image:", source)
    print("Device:", device)
    print("Baseline detections:", len(baseline), counts_by_class(baseline))
    print("Experimental detections:", len(experimental), counts_by_class(experimental))
    print("Baseline annotated:", baseline_image)
    print("Experimental annotated:", experimental_image)
    print("JSON comparison:", json_path)
    print("\nReminder: inspect both images for false positives and missed defects.")


if __name__ == "__main__":
    main()
