from pathlib import Path
from collections import defaultdict
import csv
import json
import os
import shutil

import yaml
from ultralytics import YOLO

ROOT = Path.cwd()
DATASET = ROOT / "data/processed/roadsentry_yolo_multicountry_v2"
MANIFEST = DATASET / "manifest.csv"
BASE_YAML = DATASET / "data.yaml"
WEIGHTS = ROOT / "runs/roadsentry/baseline_yolo26n_9class/weights/best.pt"
EVAL_ROOT = ROOT / "data/processed/country_eval"
RUN_DIR = ROOT / "runs/roadsentry/country_eval"

for p in (MANIFEST, BASE_YAML, WEIGHTS):
    if not p.is_file():
        raise FileNotFoundError(p)

base = yaml.safe_load(BASE_YAML.read_text())
names = {int(k): v for k, v in base["names"].items()}
nc = len(names)

with MANIFEST.open(newline="", encoding="utf-8-sig") as f:
    rows = list(csv.DictReader(f))

val_rows = [r for r in rows if r["split"].lower() == "val"]
image_dir = DATASET / "images/val"
label_dir = DATASET / "labels/val"

images_by_stem = {
    p.stem: p
    for p in image_dir.iterdir()
    if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
}

by_country = defaultdict(list)
for row in val_rows:
    country = row["country"]
    stem = f'{country}__{row["source_stem"]}'
    image = images_by_stem.get(stem)
    if image is None:
        raise RuntimeError(f"Image missing for manifest entry: {stem}")

    label = label_dir / f"{stem}.txt"
    if not label.is_file():
        raise RuntimeError(f"Prepared label missing: {label}")

    by_country[country].append((image.resolve(), label.resolve()))

if EVAL_ROOT.exists():
    shutil.rmtree(EVAL_ROOT)

summary = []
per_class = []

for country, pairs in sorted(by_country.items()):
    country_root = EVAL_ROOT / country.lower()
    country_images = country_root / "images" / "val"
    country_labels = country_root / "labels" / "val"
    country_images.mkdir(parents=True)
    country_labels.mkdir(parents=True)

    for image, label in pairs:
        image_link = country_images / image.name
        label_link = country_labels / image.with_suffix(".txt").name
        os.link(image, image_link)
        os.link(label, label_link)

    country_yaml = country_root / "data.yaml"
    config = {
        "path": str(country_root.resolve()),
        "train": "images/val",
        "val": "images/val",
        "nc": nc,
        "names": names,
    }
    country_yaml.write_text(yaml.safe_dump(config, sort_keys=False))

    # Check that this country's prepared labels contain objects.
    box_count = 0
    for _, label in pairs:
        for line in label.read_text().splitlines():
            if line.strip():
                box_count += 1

    print(f"\n=== {country.upper()} ===")
    print("Validation images:", len(pairs))
    print("Annotation boxes:", box_count)

    if box_count == 0:
        raise RuntimeError(f"No annotation boxes found for {country}")

    model = YOLO(str(WEIGHTS))
    metrics = model.val(
        data=str(country_yaml),
        imgsz=640,
        batch=8,
        device=0,
        workers=2,
        plots=False,
        project=str(RUN_DIR),
        name=country.lower(),
        exist_ok=True,
        verbose=True,
    )

    box = metrics.box
    summary.append({
        "country": country,
        "images": len(pairs),
        "annotation_boxes": box_count,
        "precision": float(box.mp),
        "recall": float(box.mr),
        "mAP50": float(box.map50),
        "mAP50_95": float(box.map),
    })

    for j, class_id in enumerate(box.ap_class_index):
        class_id = int(class_id)
        per_class.append({
            "country": country,
            "class_id": class_id,
            "class_name": names[class_id],
            "precision": float(box.p[j]),
            "recall": float(box.r[j]),
            "mAP50": float(box.ap50[j]),
            "mAP50_95": float(box.maps[class_id]),
        })

out = ROOT / "outputs/reports"
out.mkdir(parents=True, exist_ok=True)

with (out / "country_eval_summary.csv").open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
    writer.writeheader()
    writer.writerows(summary)

fields = ["country", "class_id", "class_name", "precision", "recall", "mAP50", "mAP50_95"]
with (out / "country_eval_per_class.csv").open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(per_class)

(out / "country_eval.json").write_text(json.dumps({
    "overall_by_country": summary,
    "per_class": per_class
}, indent=2))

print("\n=== COUNTRY-WISE SUMMARY ===")
for r in summary:
    print(
        f'{r["country"]:8s} | Images: {r["images"]:4d} | '
        f'Boxes: {r["annotation_boxes"]:5d} | '
        f'P: {r["precision"]:.3f} | R: {r["recall"]:.3f} | '
        f'mAP50: {r["mAP50"]:.3f} | mAP50-95: {r["mAP50_95"]:.3f}'
    )

print("\nReports saved in:", out)
