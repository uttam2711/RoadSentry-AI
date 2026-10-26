from pathlib import Path
from collections import defaultdict
import json
import math
import time

import cv2
import numpy as np
from ultralytics import YOLO

ROOT = Path.cwd()
DATA = ROOT / "data/processed/roadsentry_yolo_multicountry_v2"
IMAGE_DIR = DATA / "images/val"
LABEL_DIR = DATA / "labels/val"
WEIGHTS = ROOT / "runs/roadsentry/baseline_yolo26n_9class/weights/best.pt"
OUT_DIR = ROOT / "runs/roadsentry/tiling_448_validation_20261018"

IMG_SIZE = 640
TILE_SIZE = 448
OVERLAP = 0.25
CONF = 0.001
PRED_IOU = 0.70
MERGE_IOU = 0.55
REPORT_CONF = 0.25
BATCH = 16
IMAGE_CHUNK = 24
IOU_THRESHOLDS = np.arange(0.50, 0.951, 0.05)


def empty_pred():
    return (
        np.zeros((0, 4), dtype=np.float32),
        np.zeros(0, dtype=np.float32),
        np.zeros(0, dtype=np.int32),
    )


def unpack_result(result):
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return empty_pred()
    return (
        boxes.xyxy.cpu().numpy().astype(np.float32),
        boxes.conf.cpu().numpy().astype(np.float32),
        boxes.cls.cpu().numpy().astype(np.int32),
    )


def iou_matrix(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    area_a = np.maximum(0, a[:, 2] - a[:, 0]) * np.maximum(0, a[:, 3] - a[:, 1])
    area_b = np.maximum(0, b[:, 2] - b[:, 0]) * np.maximum(0, b[:, 3] - b[:, 1])
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-9)


def tile_starts(length):
    if length <= TILE_SIZE:
        return [0]
    max_start = length - TILE_SIZE
    stride = max(1, int(TILE_SIZE * (1 - OVERLAP)))
    starts = list(range(0, max_start + 1, stride))
    if starts[-1] != max_start:
        starts.append(max_start)
    return sorted(set(starts))


def read_gt(label_path, width, height):
    boxes, classes = [], []
    for line in label_path.read_text().splitlines():
        p = line.split()
        if len(p) < 5:
            continue
        try:
            cls = int(p[0])
            cx, cy, bw, bh = map(float, p[1:5])
        except ValueError:
            continue

        box = [
            max(0.0, (cx - bw / 2) * width),
            max(0.0, (cy - bh / 2) * height),
            min(float(width), (cx + bw / 2) * width),
            min(float(height), (cy + bh / 2) * height),
        ]
        if box[2] > box[0] and box[3] > box[1]:
            boxes.append(box)
            classes.append(cls)

    return (
        np.asarray(boxes, dtype=np.float32).reshape(-1, 4),
        np.asarray(classes, dtype=np.int32),
    )


def classwise_nms(items):
    if not items:
        return empty_pred()

    boxes = np.asarray([x[0] for x in items], dtype=np.float32)
    scores = np.asarray([x[1] for x in items], dtype=np.float32)
    classes = np.asarray([x[2] for x in items], dtype=np.int32)
    keep = []

    for cls in np.unique(classes):
        ids = np.flatnonzero(classes == cls)
        ids = ids[np.argsort(scores[ids])[::-1]]

        while len(ids):
            best = int(ids[0])
            keep.append(best)
            rest = ids[1:]
            if not len(rest):
                break
            overlaps = iou_matrix(boxes[best:best + 1], boxes[rest])[0]
            ids = rest[overlaps <= MERGE_IOU]

    keep = np.asarray(keep, dtype=int)
    return boxes[keep], scores[keep], classes[keep]


def matched_count(pred_boxes, gt_boxes, threshold):
    if len(pred_boxes) == 0 or len(gt_boxes) == 0:
        return 0
    overlaps = iou_matrix(pred_boxes, gt_boxes)
    pairs = np.argwhere(overlaps >= threshold)
    if not len(pairs):
        return 0

    order = np.argsort(overlaps[pairs[:, 0], pairs[:, 1]])[::-1]
    used_p, used_g = set(), set()
    count = 0
    for k in order:
        pi, gi = map(int, pairs[k])
        if pi not in used_p and gi not in used_g:
            used_p.add(pi)
            used_g.add(gi)
            count += 1
    return count


def compute_metrics(preds, gts, names):
    gt_counts = Counter()
    pred_records = defaultdict(list)
    total_gt = 0

    for image_idx, (gt_boxes, gt_cls) in enumerate(gts):
        total_gt += len(gt_cls)
        for cls in gt_cls:
            gt_counts[int(cls)] += 1

        boxes, scores, classes = preds[image_idx]
        for j, cls in enumerate(classes):
            pred_records[int(cls)].append(
                (float(scores[j]), image_idx, boxes[j].copy())
            )

    class_ids = sorted(set(gt_counts) | set(pred_records))
    per_class = {}
    aps50, aps_all = [], []
    global_tp, global_pred_count = 0, 0

    for cls in class_ids:
        records = pred_records.get(cls, [])
        gt_total = int(gt_counts.get(cls, 0))
        records.sort(key=lambda r: r[0], reverse=True)
        record_ids_by_image = defaultdict(list)
        correct = np.zeros((len(records), len(IOU_THRESHOLDS)), dtype=bool)

        for rid, (_, image_idx, _) in enumerate(records):
            record_ids_by_image[image_idx].append(rid)

        for image_idx, record_ids in record_ids_by_image.items():
            gt_boxes, gt_cls = gts[image_idx]
            class_gt = gt_boxes[gt_cls == cls]
            if len(class_gt) == 0:
                continue
            pred_boxes = np.stack([records[r][2] for r in record_ids])
            overlaps = iou_matrix(pred_boxes, class_gt)

            for ti, threshold in enumerate(IOU_THRESHOLDS):
                pairs = np.argwhere(overlaps >= threshold)
                if not len(pairs):
                    continue
                order = np.argsort(overlaps[pairs[:, 0], pairs[:, 1]])[::-1]
                used_p, used_g = set(), set()
                for k in order:
                    pi, gi = map(int, pairs[k])
                    if pi not in used_p and gi not in used_g:
                        used_p.add(pi)
                        used_g.add(gi)
                        correct[record_ids[pi], ti] = True

        class_ap = []
        if gt_total > 0 and len(records):
            conf_order = np.argsort(
                [-r[0] for r in records]
            )
            for ti in range(len(IOU_THRESHOLDS)):
                tp = correct[conf_order, ti].astype(float)
                fp = 1.0 - tp
                tp_cum = np.cumsum(tp)
                fp_cum = np.cumsum(fp)
                recall = tp_cum / max(gt_total, 1)
                precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-12)

                mrec = np.concatenate(([0.0], recall, [1.0]))
                mpre = np.concatenate(([1.0], precision, [0.0]))
                mpre = np.maximum.accumulate(mpre[::-1])[::-1]
                grid = np.linspace(0.0, 1.0, 101)
                ap = float(np.trapezoid(np.interp(grid, mrec, mpre), grid))
                class_ap.append(ap)

        ap50 = class_ap[0] if class_ap and gt_total > 0 else None
        ap_all = float(np.mean(class_ap)) if class_ap and gt_total > 0 else None
        if ap50 is not None:
            aps50.append(ap50)
            aps_all.append(ap_all)

        selected_by_image = defaultdict(list)
        for score, image_idx, box in records:
            if score >= REPORT_CONF:
                selected_by_image[image_idx].append(box)

        selected_count = sum(len(v) for v in selected_by_image.values())
        true_positives = 0
        for image_idx, chosen in selected_by_image.items():
            gt_boxes, gt_cls = gts[image_idx]
            class_gt = gt_boxes[gt_cls == cls]
            true_positives += matched_count(
                np.asarray(chosen, dtype=np.float32).reshape(-1, 4),
                class_gt,
                0.50,
            )

        global_tp += true_positives
        global_pred_count += selected_count

        per_class[names.get(cls, str(cls))] = {
            "ground_truth": gt_total,
            "detections_at_conf_0.25": selected_count,
            "precision_at_conf_0.25": (
                true_positives / selected_count if selected_count else 0.0
            ),
            "recall_at_conf_0.25": (
                true_positives / gt_total if gt_total else None
            ),
            "mAP50_custom": ap50,
            "mAP50_95_custom": ap_all,
        }

    return {
        "precision_at_conf_0.25": (
            global_tp / global_pred_count if global_pred_count else 0.0
        ),
        "recall_at_conf_0.25": global_tp / total_gt if total_gt else 0.0,
        "mAP50_custom": float(np.mean(aps50)) if aps50 else 0.0,
        "mAP50_95_custom": float(np.mean(aps_all)) if aps_all else 0.0,
        "ground_truth_boxes": total_gt,
        "detections_at_conf_0.25": global_pred_count,
        "per_class": per_class,
    }


# Counter is imported here to keep the rest of the script grouped by task.
from collections import Counter

def main():
    if not WEIGHTS.is_file():
        raise SystemExit(f"Missing checkpoint: {WEIGHTS}")
    if not IMAGE_DIR.is_dir() or not LABEL_DIR.is_dir():
        raise SystemExit(f"Missing validation image/label folders under {DATA}")
    if OUT_DIR.exists():
        raise SystemExit(f"Output already exists; refusing to overwrite: {OUT_DIR}")

    paths = sorted(
        p for p in IMAGE_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    )
    if not paths:
        raise SystemExit("No validation images found.")

    labels = [LABEL_DIR / f"{p.stem}.txt" for p in paths]
    missing = [p for p in labels if not p.is_file()]
    if missing:
        raise SystemExit(f"Missing {len(missing)} validation label files; stopping.")

    print("Loading frozen baseline checkpoint:", WEIGHTS)
    model = YOLO(str(WEIGHTS))
    names = model.names if isinstance(model.names, dict) else dict(enumerate(model.names))

    gts, baseline_preds = [], []
    tiled_items = [[] for _ in paths]
    total_tiles = 0
    started = time.time()

    for start in range(0, len(paths), IMAGE_CHUNK):
        batch_paths = paths[start:start + IMAGE_CHUNK]
        imgs, tile_imgs, tile_map = [], [], []
        batch_gts = []

        for offset, path in enumerate(batch_paths):
            image = cv2.imread(str(path))
            if image is None:
                raise SystemExit(f"Unreadable image: {path}")
            h, w = image.shape[:2]
            batch_idx = start + offset
            imgs.append(image)
            batch_gts.append(read_gt(labels[batch_idx], w, h))

            xs = tile_starts(w)
            ys = tile_starts(h)
            for y0 in ys:
                for x0 in xs:
                    crop = image[y0:min(y0 + TILE_SIZE, h),
                                 x0:min(x0 + TILE_SIZE, w)]
                    tile_imgs.append(crop)
                    tile_map.append((batch_idx, x0, y0, w, h))

        baseline_results = model.predict(
            source=imgs, imgsz=IMG_SIZE, conf=CONF, iou=PRED_IOU,
            max_det=300, device=0, batch=BATCH, verbose=False
        )
        for result in baseline_results:
            baseline_preds.append(unpack_result(result))

        tile_results = model.predict(
            source=tile_imgs, imgsz=IMG_SIZE, conf=CONF, iou=PRED_IOU,
            max_det=300, device=0, batch=BATCH, verbose=False
        )
        for result, mapping in zip(tile_results, tile_map):
            image_idx, x0, y0, w, h = mapping
            boxes, scores, classes = unpack_result(result)
            for box, score, cls in zip(boxes, scores, classes):
                box = box.copy()
                box[[0, 2]] += x0
                box[[1, 3]] += y0
                box[[0, 2]] = np.clip(box[[0, 2]], 0, w)
                box[[1, 3]] = np.clip(box[[1, 3]], 0, h)
                if box[2] > box[0] and box[3] > box[1]:
                    tiled_items[image_idx].append((box, float(score), int(cls)))

        gts.extend(batch_gts)
        total_tiles += len(tile_imgs)
        done = min(start + len(batch_paths), len(paths))
        print(f"Processed {done}/{len(paths)} images; {total_tiles} tiles",
              flush=True)

    tiled_preds = [classwise_nms(items) for items in tiled_items]

    print("\nComputing paired metrics...")
    baseline_metrics = compute_metrics(baseline_preds, gts, names)
    tiled_metrics = compute_metrics(tiled_preds, gts, names)
    elapsed = time.time() - started

    report = {
        "method": (
            "Paired custom evaluator: baseline full-image inference versus "
            "448px tiled inference with cross-tile classwise NMS. Use the "
            "relative comparison here; these custom AP values are not intended "
            "to be numerically identical to Ultralytics model.val metrics."
        ),
        "validation_images": len(paths),
        "image_size": IMG_SIZE,
        "tile_size": TILE_SIZE,
        "nominal_overlap": OVERLAP,
        "tile_prediction_confidence": CONF,
        "within_tile_nms_iou": PRED_IOU,
        "cross_tile_merge_iou": MERGE_IOU,
        "final_precision_recall_confidence": REPORT_CONF,
        "elapsed_seconds": round(elapsed, 2),
        "baseline": baseline_metrics,
        "tiled": tiled_metrics,
    }

    OUT_DIR.mkdir(parents=True)
    output_file = OUT_DIR / "paired_metrics.json"
    output_file.write_text(json.dumps(report, indent=2) + "\n")

    print("\n=== PAIRED EXPERIMENT RESULTS ===")
    print(f'{"Metric":<27} {"Baseline":>12} {"Tiled":>12} {"Delta":>12}')
    for key in (
        "precision_at_conf_0.25",
        "recall_at_conf_0.25",
        "mAP50_custom",
        "mAP50_95_custom",
    ):
        a = baseline_metrics[key]
        b = tiled_metrics[key]
        print(f"{key:<27} {a:>12.4f} {b:>12.4f} {b-a:>+12.4f}")

    print("\n=== PER-CLASS RECALL AT CONFIDENCE 0.25 ===")
    bclasses = baseline_metrics["per_class"]
    tclasses = tiled_metrics["per_class"]
    print(f'{"Class":<8} {"Baseline":>12} {"Tiled":>12} {"Delta":>12}')
    for cls in bclasses:
        a = bclasses[cls]["recall_at_conf_0.25"]
        b = tclasses.get(cls, {}).get("recall_at_conf_0.25")
        if a is not None and b is not None:
            print(f"{cls:<8} {a:>12.4f} {b:>12.4f} {b-a:>+12.4f}")

    print(f"\nElapsed: {elapsed/60:.1f} minutes")
    print("Results:", output_file)
    print("Checkpoint and original metrics were not changed.")
    print("NOTE: this validation comparison is exploratory, not independent test evidence.")


if __name__ == "__main__":
    main()
