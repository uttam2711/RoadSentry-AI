from pathlib import Path
import csv
import json
import random
import os

from ultralytics import YOLO

from calibrate_class_thresholds import (
    WEIGHTS, IMAGES, LABELS, TARGET_IDS, THRESHOLDS,
    BASELINE_THRESHOLD, INFERENCE_CONFIDENCE_FLOOR,
    INFERENCE_SIZE, MATCH_IOU, read_ground_truth,
    match_counts, metrics_from_counts,
)

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs/demo/threshold_confirmation_640_20261016_safe"
SEED = 20261016
CHECKPOINT_EVERY = 50
SPLITS = ("calibration", "holdout")


def empty_state():
    counts = {}
    for split in SPLITS:
        counts[split] = {}
        for cls in TARGET_IDS:
            for threshold in THRESHOLDS:
                counts[split][f"{cls}:{threshold:.2f}"] = [0, 0, 0]
    return {"processed_stems": [], "counts": counts}


def save_state(path, state):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(state), encoding="utf-8")
    os.replace(temp, path)


def main():
    for path in (WEIGHTS, IMAGES, LABELS):
        if not path.exists():
            raise FileNotFoundError(path)

    images = sorted(
        p for p in IMAGES.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    shuffled = images.copy()
    random.Random(SEED).shuffle(shuffled)
    midpoint = len(shuffled) // 2
    calibration_stems = {p.stem for p in shuffled[:midpoint]}

    OUTPUT.mkdir(parents=True, exist_ok=True)
    state_path = OUTPUT / "progress_checkpoint.json"
    final_json = OUTPUT / "confirmation_summary.json"

    if final_json.is_file():
        raise FileExistsError(
            f"Final results already exist at {final_json}; not overwriting."
        )

    if state_path.is_file():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        print("Resuming from checkpoint.")
    else:
        other_files = list(OUTPUT.iterdir())
        if other_files:
            raise RuntimeError(
                f"Output folder exists without a checkpoint: {OUTPUT}"
            )
        state = empty_state()
        save_state(state_path, state)

    processed_stems = set(state["processed_stems"])
    todo = [p for p in images if p.stem not in processed_stems]

    print("=== RESUMABLE THRESHOLD CONFIRMATION ===")
    print("Total images:", len(images))
    print("Calibration images:", len(calibration_stems))
    print("Holdout images:", len(images) - len(calibration_stems))
    print("Already checkpointed:", len(processed_stems))
    print("Remaining:", len(todo))
    print("Inference size:", INFERENCE_SIZE, "| IoU:", MATCH_IOU)
    print("Batching: one image per prediction")
    print("Training: NO")

    model = YOLO(str(WEIGHTS))

    for index, image_path in enumerate(todo, start=1):
        result = model.predict(
            source=str(image_path),
            imgsz=INFERENCE_SIZE,
            conf=INFERENCE_CONFIDENCE_FLOOR,
            iou=0.7,
            max_det=300,
            classes=TARGET_IDS,
            device=0,
            verbose=False,
            save=False,
        )[0]

        height, width = result.orig_shape
        label_path = LABELS / f"{image_path.stem}.txt"
        split = (
            "calibration"
            if image_path.stem in calibration_stems
            else "holdout"
        )

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
                key = f"{cls}:{threshold:.2f}"
                running = state["counts"][split][key]
                running[0] += tp
                running[1] += fp
                running[2] += fn

        processed_stems.add(image_path.stem)

        if index % CHECKPOINT_EVERY == 0 or index == len(todo):
            state["processed_stems"] = sorted(processed_stems)
            save_state(state_path, state)
            print(
                f"Checkpoint saved: {len(processed_stems)}/{len(images)}",
                flush=True,
            )

    rows = []
    recommendations = []
    names = model.names

    for cls in TARGET_IDS:
        class_name = (
            names.get(cls, str(cls))
            if isinstance(names, dict) else names[cls]
        )
        calibration_rows = {}

        for threshold in THRESHOLDS:
            tp, fp, fn = state["counts"]["calibration"][
                f"{cls}:{threshold:.2f}"
            ]
            calibration_rows[threshold] = metrics_from_counts(tp, fp, fn)

        selected_threshold = max(
            THRESHOLDS,
            key=lambda threshold: (
                calibration_rows[threshold]["f1"],
                calibration_rows[threshold]["precision"],
                threshold,
            ),
        )

        for threshold in THRESHOLDS:
            for split in SPLITS:
                tp, fp, fn = state["counts"][split][
                    f"{cls}:{threshold:.2f}"
                ]
                rows.append({
                    "split": split,
                    "class_id": cls,
                    "class_name": str(class_name),
                    "threshold": threshold,
                    **metrics_from_counts(tp, fp, fn),
                })

        tp, fp, fn = state["counts"]["holdout"][
            f"{cls}:{selected_threshold:.2f}"
        ]
        candidate = metrics_from_counts(tp, fp, fn)

        tp, fp, fn = state["counts"]["holdout"][
            f"{cls}:{BASELINE_THRESHOLD:.2f}"
        ]
        baseline = metrics_from_counts(tp, fp, fn)

        recommendations.append({
            "class_id": cls,
            "class_name": str(class_name),
            "threshold_selected_on_calibration": selected_threshold,
            "holdout_f1_at_selected_threshold": candidate["f1"],
            "holdout_f1_at_global_025": baseline["f1"],
            "holdout_f1_delta": candidate["f1"] - baseline["f1"],
            "holdout_precision_at_selected_threshold": candidate["precision"],
            "holdout_recall_at_selected_threshold": candidate["recall"],
        })

    fields = [
        "split", "class_id", "class_name", "threshold",
        "true_positives", "false_positives", "false_negatives",
        "precision", "recall", "f1",
    ]
    with (OUTPUT / "split_threshold_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "checkpoint": str(WEIGHTS),
        "validation_images": len(images),
        "calibration_images": len(calibration_stems),
        "holdout_images": len(images) - len(calibration_stems),
        "seed": SEED,
        "image_size": INFERENCE_SIZE,
        "matching_iou": MATCH_IOU,
        "note": (
            "Confirmation split within the validation dataset, not an "
            "untouched test set. Earlier threshold exploration used the "
            "full validation split."
        ),
        "recommendations": recommendations,
    }
    final_json.write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )

    print("\n=== HOLDOUT CONFIRMATION RESULTS ===")
    print("Class | Threshold | Holdout F1 | F1 at 0.25 | Delta")
    for row in recommendations:
        print(
            f'{row["class_name"]} | '
            f'{row["threshold_selected_on_calibration"]:.2f} | '
            f'{row["holdout_f1_at_selected_threshold"]:.3f} | '
            f'{row["holdout_f1_at_global_025"]:.3f} | '
            f'{row["holdout_f1_delta"]:+.3f}'
        )

    print("\nCSV:", OUTPUT / "split_threshold_metrics.csv")
    print("JSON:", final_json)
    print("All images checkpointed. Model weights were not changed.")


if __name__ == "__main__":
    main()
