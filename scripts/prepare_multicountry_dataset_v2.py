
from collections import Counter, defaultdict
from pathlib import Path
import csv
import json
import math
import random
import re
import sys
import xml.etree.ElementTree as ET

import cv2


ROOT = Path(__file__).resolve().parents[1]

COUNTRIES = {
    "India": ROOT / "data/raw/India_train_data",
    "Japan": ROOT / "data/raw/Japan",
    "Czech": ROOT / "data/raw/Czech",
}

# Target classes first: these are the four classes shown to users.
TARGET_CLASSES = ("D00", "D10", "D20", "D40")

# Other documented annotation classes; train them as auxiliary classes.
AUXILIARY_CLASSES = ("D01", "D11", "D43", "D44", "D50")
MODEL_CLASSES = TARGET_CLASSES + AUXILIARY_CLASSES
CLASS_TO_ID = {name: idx for idx, name in enumerate(MODEL_CLASSES)}

OUT_DIR = ROOT / "data/processed/roadsentry_yolo_multicountry_v2"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
BLOCK_SIZE = 250
VAL_RATIO = 0.20
SEED = 2026


def get_numeric_id(stem):
    match = re.search(r"(\d+)$", stem)
    if not match:
        raise ValueError(f"Cannot extract numeric ID from {stem}")
    return int(match.group(1))


def scan_country(country, base, audit):
    image_dir = base / "images"
    xml_dir = base / "annotations/xmls"

    if not image_dir.is_dir() or not xml_dir.is_dir():
        raise FileNotFoundError(
            f"{country}: expected {image_dir} and {xml_dir}"
        )

    image_paths = [
        p for p in image_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]
    xml_paths = list(xml_dir.rglob("*.xml"))

    images = {p.stem: p for p in image_paths}
    xmls = {p.stem: p for p in xml_paths}

    if len(images) != len(image_paths) or len(xmls) != len(xml_paths):
        raise ValueError(f"{country}: duplicate filename stems")

    if images.keys() != xmls.keys():
        raise ValueError(
            f"{country}: image/XML mismatch; "
            f"{len(images.keys() - xmls.keys())} missing XML, "
            f"{len(xmls.keys() - images.keys())} missing images"
        )

    audit[country]["source_images"] = len(images)
    audit[country]["source_xmls"] = len(xmls)

    records = []

    for index, stem in enumerate(sorted(images), start=1):
        image_path = images[stem]
        xml_path = xmls[stem]

        try:
            root = ET.parse(xml_path).getroot()
        except (ET.ParseError, OSError) as exc:
            audit[country]["invalid_xml_skipped"] += 1
            audit["skip_reasons"]["invalid_xml"] += 1
            continue

        declared_name = (root.findtext("filename") or "").strip()
        if declared_name and Path(declared_name).stem != stem:
            audit[country]["filename_mismatch_skipped"] += 1
            audit["skip_reasons"]["filename_mismatch"] += 1
            continue

        image = cv2.imread(str(image_path))
        if image is None:
            audit[country]["unreadable_image_skipped"] += 1
            audit["skip_reasons"]["unreadable_image"] += 1
            continue

        height, width = image.shape[:2]
        objects = root.findall("object")

        # Empty annotations are legitimate negative examples.
        if not objects:
            records.append({
                "country": country,
                "stem": stem,
                "image_path": image_path,
                "numeric_id": get_numeric_id(stem),
                "boxes": [],
                "counts": Counter(),
            })
            audit[country]["empty_images_kept"] += 1
            continue

        labels = [
            (obj.findtext("name") or "").strip()
            for obj in objects
        ]

        # Do not silently convert unsupported labels to background.
        unknown = {
            label for label in labels
            if label not in CLASS_TO_ID
        }

        if unknown:
            audit[country]["unsupported_image_skipped"] += 1
            audit["skip_reasons"]["unsupported_label_image"] += 1
            audit["unsupported_labels"].update(unknown)
            continue

        boxes = []
        counts = Counter()
        invalid = False

        for obj in objects:
            label = (obj.findtext("name") or "").strip()
            box = obj.find("bndbox")

            try:
                if box is None:
                    raise ValueError("missing bounding box")

                x1 = float(box.findtext("xmin"))
                y1 = float(box.findtext("ymin"))
                x2 = float(box.findtext("xmax"))
                y2 = float(box.findtext("ymax"))

                if not all(map(math.isfinite, (x1, y1, x2, y2))):
                    raise ValueError("non-finite coordinates")

                if not (
                    0 <= x1 < x2 <= width
                    and 0 <= y1 < y2 <= height
                ):
                    raise ValueError("out-of-range or zero-area box")

            except (TypeError, ValueError):
                invalid = True
                break

            class_id = CLASS_TO_ID[label]

            # Convert Pascal VOC coordinates to normalized YOLO format.
            boxes.append((
                class_id,
                (x1 + x2) / (2 * width),
                (y1 + y2) / (2 * height),
                (x2 - x1) / width,
                (y2 - y1) / height,
            ))
            counts[label] += 1

        # Skip the complete image, not only the invalid object.
        if invalid:
            audit[country]["invalid_box_image_skipped"] += 1
            audit["skip_reasons"]["invalid_box_image"] += 1
            continue

        audit[country]["images_with_objects_kept"] += 1

        records.append({
            "country": country,
            "stem": stem,
            "image_path": image_path,
            "numeric_id": get_numeric_id(stem),
            "boxes": boxes,
            "counts": counts,
        })

        if index % 3000 == 0:
            print(f"{country}: scanned {index}/{len(images)} images")

    return records


def assign_splits(records):
    # Keep nearby numerical ID blocks within the same split, per country.
    groups = defaultdict(list)

    for record in records:
        group = (
            record["country"],
            record["numeric_id"] // BLOCK_SIZE,
        )
        groups[group].append(record)

    rng = random.Random(SEED)
    validation_groups = set()

    for country in COUNTRIES:
        country_groups = sorted(
            group for group in groups
            if group[0] == country
        )

        if len(country_groups) < 2:
            raise RuntimeError(
                f"Insufficient groups for splitting {country}"
            )

        rng.shuffle(country_groups)

        count = max(1, round(len(country_groups) * VAL_RATIO))
        validation_groups.update(country_groups[:count])

    splits = {"train": [], "val": []}

    for record in records:
        group = (
            record["country"],
            record["numeric_id"] // BLOCK_SIZE,
        )
        split = "val" if group in validation_groups else "train"
        splits[split].append(record)

    return splits


def write_yaml():
    names = "\n".join(
        f"  {idx}: {name}"
        for idx, name in enumerate(MODEL_CLASSES)
    )

    content = (
        f"path: {OUT_DIR.as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        f"names:\n{names}\n"
    )

    (OUT_DIR / "data.yaml").write_text(
        content, encoding="utf-8"
    )


def main():
    if OUT_DIR.exists() and any(OUT_DIR.iterdir()):
        raise FileExistsError(
            f"Output folder already contains files: {OUT_DIR}\n"
            "Inspect the existing folder before rerunning."
        )

    audit = {
        country: Counter()
        for country in COUNTRIES
    }
    audit["skip_reasons"] = Counter()
    audit["unsupported_labels"] = Counter()

    records = []

    for country, base in COUNTRIES.items():
        print(f"\nScanning {country}...")
        country_records = scan_country(country, base, audit)
        print(
            f"{country}: retained {len(country_records)} images"
        )
        records.extend(country_records)

    if not records:
        raise RuntimeError("No usable images found.")

    splits = assign_splits(records)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report = {
        "target_classes": list(TARGET_CLASSES),
        "auxiliary_classes": list(AUXILIARY_CLASSES),
        "model_classes": list(MODEL_CLASSES),
        "seed": SEED,
        "numeric_id_block_size": BLOCK_SIZE,
        "validation_ratio_requested": VAL_RATIO,
        "source_audit": {
            country: dict(values)
            for country, values in audit.items()
            if country not in ("skip_reasons", "unsupported_labels")
        },
        "skip_reasons": dict(audit["skip_reasons"]),
        "unsupported_labels": dict(audit["unsupported_labels"]),
        "retained_images": len(records),
        "splits": {},
    }

    manifest = OUT_DIR / "manifest.csv"

    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "country", "source_stem", "split",
            "target_box_count", "all_box_count", "has_target_box",
        ])

        for split, items in splits.items():
            image_dir = OUT_DIR / "images" / split
            label_dir = OUT_DIR / "labels" / split
            image_dir.mkdir(parents=True, exist_ok=True)
            label_dir.mkdir(parents=True, exist_ok=True)

            class_counts = Counter()
            country_counts = Counter()
            empty_count = 0

            for record in items:
                country = record["country"]
                stem = record["stem"]
                out_stem = f"{country}__{stem}"
                image_out = image_dir / (
                    out_stem + record["image_path"].suffix.lower()
                )
                label_out = label_dir / f"{out_stem}.txt"

                # Symlink avoids duplicating gigabytes of source images.
                image_out.symlink_to(record["image_path"].resolve())

                label_lines = [
                    f"{class_id} {cx:.6f} {cy:.6f} "
                    f"{bw:.6f} {bh:.6f}"
                    for class_id, cx, cy, bw, bh in record["boxes"]
                ]

                label_out.write_text(
                    "\n".join(label_lines)
                    + ("\n" if label_lines else ""),
                    encoding="utf-8",
                )

                class_counts.update(record["counts"])
                country_counts[country] += 1

                target_box_count = sum(
                    record["counts"][name]
                    for name in TARGET_CLASSES
                )

                if not record["boxes"]:
                    empty_count += 1

                writer.writerow([
                    country,
                    stem,
                    split,
                    target_box_count,
                    len(record["boxes"]),
                    int(target_box_count > 0),
                ])

            report["splits"][split] = {
                "images": len(items),
                "empty_images": empty_count,
                "images_by_country": dict(country_counts),
                "boxes_by_class": {
                    name: class_counts[name]
                    for name in MODEL_CLASSES
                },
                "target_boxes": sum(
                    class_counts[name] for name in TARGET_CLASSES
                ),
            }

    write_yaml()

    (OUT_DIR / "preparation_report.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("\n========== V2 PREPARATION COMPLETE ==========")
    print("Output:", OUT_DIR)
    print("Retained images:", len(records))
    print("Skip reasons:", dict(audit["skip_reasons"]))
    print("Unsupported labels:", dict(audit["unsupported_labels"]))

    for split, values in report["splits"].items():
        print(f"\n{split.upper()}:")
        print("Images:", values["images"])
        print("By country:", values["images_by_country"])
        print("Target boxes:", values["target_boxes"])
        print("All class boxes:", values["boxes_by_class"])

    print("\nInspect preparation_report.json before training.")
    print("The filename-block split is only a proxy for sequence separation.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
