
from collections import Counter
from pathlib import Path
import csv
import json
import random
import re
import shutil
import sys
import xml.etree.ElementTree as ET

import cv2


ROOT = Path(__file__).resolve().parents[1]

COUNTRIES = {
    "India": ROOT / "data/raw/India_train_data",
    "Japan": ROOT / "data/raw/Japan",
    "Czech": ROOT / "data/raw/Czech",
}

TARGET_CLASSES = ("D00", "D10", "D20", "D40")
CLASS_TO_ID = {name: idx for idx, name in enumerate(TARGET_CLASSES)}

LABEL_MAP = ROOT / "data/raw/label_map.pbtxt"
OUT_DIR = ROOT / "data/processed/roadsentry_yolo_multicountry"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
VAL_RATIO = 0.20
SEED = 2026
BLOCK_SIZE = 250


def numeric_id(stem):
    match = re.search(r"(\d+)$", stem)
    if not match:
        raise ValueError(f"Cannot extract numeric ID from {stem}")
    return int(match.group(1))


def validate_label_map():
    if not LABEL_MAP.is_file():
        raise FileNotFoundError(f"Missing label map: {LABEL_MAP}")

    content = LABEL_MAP.read_text(encoding="utf-8")
    names = re.findall(
        r"\bname\s*:\s*['\"]([^'\"]+)['\"]", content
    )

    if tuple(names) != TARGET_CLASSES:
        raise ValueError(
            f"Unexpected label map classes: {names}; "
            f"expected {TARGET_CLASSES}"
        )


def scan_country(country, base, stats):
    image_dir = base / "images"
    xml_dir = base / "annotations/xmls"

    if not image_dir.is_dir() or not xml_dir.is_dir():
        raise FileNotFoundError(
            f"{country}: expected {image_dir} and {xml_dir}"
        )

    images = {
        p.stem: p
        for p in image_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    }

    xmls = {p.stem: p for p in xml_dir.rglob("*.xml")}

    if len(images) != sum(
        1 for p in image_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ):
        raise ValueError(f"{country}: duplicate image filename stems")

    if len(xmls) != len(list(xml_dir.rglob("*.xml"))):
        raise ValueError(f"{country}: duplicate XML filename stems")

    if images.keys() != xmls.keys():
        raise ValueError(
            f"{country}: images/XML mismatch; "
            f"{len(images.keys() - xmls.keys())} missing XML, "
            f"{len(xmls.keys() - images.keys())} missing images"
        )

    records = []

    for index, stem in enumerate(sorted(images), 1):
        image_path = images[stem]
        xml_path = xmls[stem]

        try:
            root = ET.parse(xml_path).getroot()
        except (ET.ParseError, OSError) as exc:
            stats[country]["invalid_xml"] += 1
            stats["skip_reasons"]["invalid_xml"] += 1
            continue

        image = cv2.imread(str(image_path))
        if image is None:
            stats[country]["unreadable_images"] += 1
            stats["skip_reasons"]["unreadable_image"] += 1
            continue

        height, width = image.shape[:2]
        declared_filename = (root.findtext("filename") or "").strip()

        if declared_filename and Path(declared_filename).stem != stem:
            stats[country]["filename_mismatch"] += 1
            stats["skip_reasons"]["filename_mismatch"] += 1
            continue

        objects = root.findall("object")
        labels = [
            (obj.findtext("name") or "").strip()
            for obj in objects
        ]

        # Empty XML files are retained as negative examples.
        if not objects:
            records.append({
                "country": country,
                "stem": stem,
                "image_path": image_path,
                "numeric_id": numeric_id(stem),
                "boxes": [],
                "class_counts": Counter(),
            })
            stats[country]["empty_xml"] += 1
            continue

        if any(not label for label in labels):
            stats["skip_reasons"]["missing_label"] += 1
            continue

        label_set = set(labels)
        target_labels = label_set & set(TARGET_CLASSES)
        non_target_labels = label_set - set(TARGET_CLASSES)

        if target_labels and non_target_labels:
            stats[country]["mixed_excluded"] += 1
            stats["skip_reasons"]["mixed_labels"] += 1
            continue

        if not target_labels:
            stats[country]["other_only_excluded"] += 1
            stats["skip_reasons"]["other_only"] += 1
            continue

        boxes = []
        class_counts = Counter()
        invalid = False

        for obj in objects:
            label = (obj.findtext("name") or "").strip()
            box = obj.find("bndbox")

            try:
                if box is None:
                    raise ValueError("missing bndbox")

                x1 = float(box.findtext("xmin"))
                y1 = float(box.findtext("ymin"))
                x2 = float(box.findtext("xmax"))
                y2 = float(box.findtext("ymax"))

                if not (
                    0 <= x1 < x2 <= width
                    and 0 <= y1 < y2 <= height
                ):
                    raise ValueError("invalid box coordinates")

            except (TypeError, ValueError):
                invalid = True
                break

            class_id = CLASS_TO_ID[label]

            boxes.append((
                class_id,
                (x1 + x2) / (2 * width),
                (y1 + y2) / (2 * height),
                (x2 - x1) / width,
                (y2 - y1) / height,
            ))
            class_counts[label] += 1

        if invalid:
            stats[country]["invalid_boxes"] += 1
            stats["skip_reasons"]["invalid_boxes"] += 1
            continue

        records.append({
            "country": country,
            "stem": stem,
            "image_path": image_path,
            "numeric_id": numeric_id(stem),
            "boxes": boxes,
            "class_counts": class_counts,
        })

        stats[country]["target_only"] += 1

        if index % 2000 == 0:
            print(f"{country}: scanned {index}/{len(images)} images")

    return records


def make_split(records):
    groups = {}

    for record in records:
        group = (
            record["country"],
            record["numeric_id"] // BLOCK_SIZE,
        )
        groups.setdefault(group, []).append(record)

    rng = random.Random(SEED)
    validation_groups = set()

    for country in COUNTRIES:
        country_groups = sorted(
            group for group in groups
            if group[0] == country
        )

        if len(country_groups) < 2:
            raise RuntimeError(
                f"Not enough ID groups in {country} for splitting"
            )

        rng.shuffle(country_groups)
        n_val = max(1, round(len(country_groups) * VAL_RATIO))
        validation_groups.update(country_groups[:n_val])

    splits = {"train": [], "val": []}

    for record in records:
        group = (
            record["country"],
            record["numeric_id"] // BLOCK_SIZE,
        )
        split = "val" if group in validation_groups else "train"
        splits[split].append(record)

    return splits


def main():
    validate_label_map()

    if OUT_DIR.exists() and any(OUT_DIR.iterdir()):
        raise FileExistsError(
            f"Output directory already contains files: {OUT_DIR}\n"
            "Inspect or rename it before creating a new dataset."
        )

    stats = {
        country: Counter()
        for country in COUNTRIES
    }
    stats["skip_reasons"] = Counter()

    records = []

    for country, base in COUNTRIES.items():
        print(f"\nScanning {country}...")
        country_records = scan_country(country, base, stats)
        print(
            f"{country}: retained {len(country_records)} images"
        )
        records.extend(country_records)

    if not records:
        raise RuntimeError("No records available for preparation.")

    splits = make_split(records)
    report = {
        "seed": SEED,
        "validation_ratio_requested": VAL_RATIO,
        "numeric_id_block_size": BLOCK_SIZE,
        "target_classes": list(TARGET_CLASSES),
        "total_retained_images": len(records),
        "country_audit": {
            country: dict(values)
            for country, values in stats.items()
            if country != "skip_reasons"
        },
        "skip_reasons": dict(stats["skip_reasons"]),
        "splits": {},
    }

    # Write into a new output directory; never alter source data.
    for split, items in splits.items():
        images_out = OUT_DIR / "images" / split
        labels_out = OUT_DIR / "labels" / split
        images_out.mkdir(parents=True, exist_ok=True)
        labels_out.mkdir(parents=True, exist_ok=True)

        class_totals = Counter()
        country_totals = Counter()
        empty_count = 0

        for record in items:
            country = record["country"]
            stem = record["stem"]
            output_stem = f"{country}__{stem}"

            shutil.copy2(
                record["image_path"],
                images_out / f"{output_stem}{record['image_path'].suffix.lower()}",
            )

            lines = [
                f"{class_id} {cx:.6f} {cy:.6f} "
                f"{bw:.6f} {bh:.6f}"
                for class_id, cx, cy, bw, bh in record["boxes"]
            ]

            (labels_out / f"{output_stem}.txt").write_text(
                "\n".join(lines) + ("\n" if lines else ""),
                encoding="utf-8",
            )

            if lines:
                class_totals.update(record["class_counts"])
            else:
                empty_count += 1

            country_totals[country] += 1

        report["splits"][split] = {
            "images": len(items),
            "images_without_target_boxes": empty_count,
            "boxes_per_class": dict(class_totals),
            "images_per_country": dict(country_totals),
        }

    (OUT_DIR / "data.yaml").write_text(
        f"path: {OUT_DIR.as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n"
        "  0: D00\n"
        "  1: D10\n"
        "  2: D20\n"
        "  3: D40\n",
        encoding="utf-8",
    )

    with (OUT_DIR / "manifest.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "country", "source_stem", "split",
            "target_box_count", "has_target_boxes",
        ])

        for split, items in splits.items():
            for record in items:
                writer.writerow([
                    record["country"],
                    record["stem"],
                    split,
                    len(record["boxes"]),
                    int(bool(record["boxes"])),
                ])

    report["warning"] = (
        "Numeric ID blocks reduce one potential source of leakage but do "
        "not prove independent video sequences or acquisition sessions."
    )

    (OUT_DIR / "preparation_report.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("\n========== PREPARATION COMPLETE ==========")
    print("Output:", OUT_DIR)
    print("Retained images:", len(records))

    for split, values in report["splits"].items():
        print(f"\n{split.upper()}:")
        print("Images:", values["images"])
        print("Empty-target images:", values["images_without_target_boxes"])
        print("Country counts:", values["images_per_country"])
        print("Class boxes:", values["boxes_per_class"])

    print("\nExcluded reasons:", report["skip_reasons"])
    print("Inspect preparation_report.json before training.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
