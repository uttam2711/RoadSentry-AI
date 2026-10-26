from pathlib import Path
import csv
import json

import numpy as np
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ROOT / "runs/roadsentry/baseline_yolo26n_9class/weights/best.pt"
DATA = ROOT / "data/processed/roadsentry_yolo_multicountry_v2/data.yaml"
RUN_DIR = ROOT / "runs/roadsentry/full_validation_baseline_640_20261015"

IMAGE_SIZE = 640
BATCH_SIZE = 8
DEVICE = 0


def scalar(value):
    return float(value)


def main():
    for path in (WEIGHTS, DATA):
        if not path.is_file():
            raise FileNotFoundError(f"Required file missing: {path}")

    if RUN_DIR.exists():
        raise FileExistsError(
            f"Evaluation folder already exists: {RUN_DIR}\n"
            "Rename RUN_DIR in this script before running again."
        )

    print("=== ROAD SENTRY FULL VALIDATION ===")
    print("Checkpoint:", WEIGHTS)
    print("Dataset:", DATA)
    print("Split: val")
    print("Image size:", IMAGE_SIZE)
    print("Batch size:", BATCH_SIZE)
    print("Device: CUDA GPU 0")
    print("Output:", RUN_DIR)
    print("\nStarting evaluation. No training will be performed.")

    model = YOLO(str(WEIGHTS))
    metrics = model.val(
        data=str(DATA),
        split="val",
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        device=DEVICE,
        project=str(RUN_DIR.parent),
        name=RUN_DIR.name,
        plots=True,
        verbose=True,
        save_json=False,
    )

    box = metrics.box
    class_indices = np.asarray(box.ap_class_index, dtype=int)
    precisions = np.asarray(box.p)
    recalls = np.asarray(box.r)
    ap50_values = np.asarray(box.ap50)
    ap_values = np.asarray(box.ap)

    class_metrics = []
    for i, class_id in enumerate(class_indices):
        if isinstance(model.names, dict):
            class_name = model.names.get(int(class_id), str(class_id))
        else:
            class_name = model.names[int(class_id)]

        ap_per_threshold = ap_values[i]
        class_metrics.append({
            "class_id": int(class_id),
            "class_name": str(class_name),
            "precision": scalar(precisions[i]),
            "recall": scalar(recalls[i]),
            "mAP50": scalar(ap50_values[i]),
            "mAP50_95": scalar(np.mean(ap_per_threshold)),
        })

    summary = {
        "checkpoint": str(WEIGHTS),
        "dataset_yaml": str(DATA),
        "split": "val",
        "image_size": IMAGE_SIZE,
        "batch_size": BATCH_SIZE,
        "device": str(DEVICE),
        "class_metrics": class_metrics,
        "overall": {
            "precision": scalar(box.mp),
            "recall": scalar(box.mr),
            "mAP50": scalar(box.map50),
            "mAP50_95": scalar(box.map),
        },
        "note": (
            "Metrics are from the full validation split using the saved "
            "checkpoint. Curated visual examples are not full-split metrics."
        ),
    }

    RUN_DIR.mkdir(parents=True, exist_ok=True)
    json_path = RUN_DIR / "full_validation_metrics.json"
    csv_path = RUN_DIR / "per_class_metrics.csv"

    json_path.write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )

    fields = [
        "class_id", "class_name", "precision", "recall",
        "mAP50", "mAP50_95",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(class_metrics)

    print("\n=== OVERALL FULL-VALIDATION METRICS ===")
    for name, value in summary["overall"].items():
        print(f"{name}: {value:.4f}")

    print("\n=== PER-CLASS METRICS ===")
    print("Class | Precision | Recall | mAP50 | mAP50-95")
    for row in class_metrics:
        print(
            f'{row["class_name"]} | {row["precision"]:.4f} | '
            f'{row["recall"]:.4f} | {row["mAP50"]:.4f} | '
            f'{row["mAP50_95"]:.4f}'
        )

    print("\nMetrics JSON:", json_path)
    print("Per-class CSV:", csv_path)
    print("Plots and validation artifacts:", RUN_DIR)


if __name__ == "__main__":
    main()
