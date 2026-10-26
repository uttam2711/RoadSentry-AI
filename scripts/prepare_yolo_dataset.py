
from collections import Counter, defaultdict
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

IMAGE_DIR = ROOT / "data/raw/India_train_data/images"
XML_DIR = ROOT / "data/raw/India_train_data/annotations/xmls"
LABEL_MAP = ROOT / "data/raw/label_map.pbtxt"

OUT_DIR = ROOT / "data/processed/roadsentry_yolo"

# Order must match the four classes in the official label map.
TARGET_CLASSES = ("D00", "D10", "D20", "D40")
CLASS_TO_ID = {name: i for i, name in enumerate(TARGET_CLASSES)}

VAL_RATIO = 0.20
SEED = 2026

# Keep blocks of nearby numerical image IDs together across the split.
GROUP_SIZES_TO_TRY = (250, 100)
MAX_SPLIT_TRIALS = 12000

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def read_label_map():
    text = LABEL_MAP.read_text(encoding="utf-8")
    items = []

    for block in re.findall(r"item\s*\{(.*?)\}", text, flags=re.S):
        id_match = re.search(r"\bid\s*:\s*(\d+)", block)
        name_match = re.search(
            r"\bname\s*:\s*['\"]([^'\"]+)['\"]", block
        )

        if id_match and name_match:
            items.append((int(id_match.group(1)), name_match.group(1)))

    names = tuple(name for _, name in sorted(items))

    if names != TARGET_CLASSES:
        raise ValueError(
            f"Unexpected label map order: {names}. "
            f"Expected: {TARGET_CLASSES}"
        )

    return names


def numeric_image_id(stem):
    match = re.search(r"(\d+)$", stem)

    if not match:
        raise ValueError(
            f"Cannot derive numeric ID from image filename: {stem}"
        )

    return int(match.group(1))


def inspect_source():
    for folder in (IMAGE_DIR, XML_DIR):
        if not folder.is_dir():
            raise FileNotFoundError(f"Required folder not found: {folder}")

    read_label_map()

    image_paths = sorted(
        path for path in IMAGE_DIR.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )

    xml_paths = sorted(XML_DIR.rglob("*.xml"))

    images = {path.stem: path for path in image_paths}
    xmls = {path.stem: path for path in xml_paths}

    if len(images) != len(image_paths):
        raise ValueError("Duplicate image filename stems detected.")

    if len(xmls) != len(xml_paths):
        raise ValueError("Duplicate XML filename stems detected.")

    missing_xml = sorted(images.keys() - xmls.keys())
    missing_images = sorted(xmls.keys() - images.keys())

    if missing_xml or missing_images:
        raise ValueError(
            f"Image/XML mismatch: {len(missing_xml)} images lack XML; "
            f"{len(missing_images)} XML files lack images."
        )

    records = []
    target_totals = Counter()
    ignored_totals = Counter()
    empty_source_xmls = 0

    print(f"Parsing {len(image_paths)} matched image/XML pairs...")

    for index, (stem, image_path) in enumerate(sorted(images.items()), 1):
        xml_path = xmls[stem]

        try:
            root = ET.parse(xml_path).getroot()
        except (ET.ParseError, OSError) as exc:
            raise ValueError(f"Invalid XML {xml_path}: {exc}") from exc

        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Unreadable image: {image_path}")

        height, width = image.shape[:2]

        declared_name = (root.findtext("filename") or "").strip()
        if declared_name and Path(declared_name).stem != stem:
            raise ValueError(
                f"Filename mismatch in {xml_path.name}: {declared_name}"
            )

        objects = root.findall("object")
        if not objects:
            empty_source_xmls += 1

        yolo_lines = []
        class_counts = Counter()
        unknown_counts = Counter()

        for obj in objects:
            label = (obj.findtext("name") or "").strip()

            if label not in CLASS_TO_ID:
                # Preserve source XML, but do not train an unsupported class.
                ignored_totals[label or "<missing-label>"] += 1
                unknown_counts[label or "<missing-label>"] += 1
                continue

            box = obj.find("bndbox")
            if box is None:
                raise ValueError(f"Missing bounding box in {xml_path.name}")

            try:
                x1 = float(box.findtext("xmin"))
                y1 = float(box.findtext("ymin"))
                x2 = float(box.findtext("xmax"))
                y2 = float(box.findtext("ymax"))
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid coordinates in {xml_path.name}"
                ) from exc

            if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
                raise ValueError(
                    f"Out-of-range or degenerate box in {xml_path.name}: "
                    f"{x1}, {y1}, {x2}, {y2} for {width}x{height}"
                )

            # Pascal VOC absolute coordinates -> YOLO normalized format.
            center_x = ((x1 + x2) / 2) / width
            center_y = ((y1 + y2) / 2) / height
            box_width = (x2 - x1) / width
            box_height = (y2 - y1) / height

            class_id = CLASS_TO_ID[label]
            yolo_lines.append(
                f"{class_id} {center_x:.6f} {center_y:.6f} "
                f"{box_width:.6f} {box_height:.6f}"
            )

            class_counts[label] += 1
            target_totals[label] += 1

        records.append({
            "stem": stem,
            "image_path": image_path,
            "xml_path": xml_path,
            "group_id": numeric_image_id(stem),
            "yolo_lines": yolo_lines,
            "class_counts": class_counts,
            "ignored_counts": unknown_counts,
        })

        if index % 1000 == 0:
            print(f"Parsed {index}/{len(image_paths)} pairs...")

    if len(records) != 7706:
        print(
            f"NOTICE: Found {len(records)} pairs, not the previously "
            "observed 7706. Continuing with verified files."
        )

    return records, target_totals, ignored_totals, empty_source_xmls


def choose_blocked_split(records, target_totals):
    """Find a deterministic split holding numeric ID blocks together."""

    ratio = VAL_RATIO
    best_fallback = None
    rng = random.Random(SEED)

    for block_size in GROUP_SIZES_TO_TRY:
        group_records = defaultdict(list)
        group_counts = defaultdict(Counter)

        for record in records:
            group = record["group_id"] // block_size
            group_records[group].append(record)
            group_counts[group].update(record["class_counts"])

        groups = list(group_records)
        if len(groups) < 5:
            continue

        best_valid = None
        best_score = float("inf")

        for _ in range(MAX_SPLIT_TRIALS):
            val_groups = {
                group for group in groups if rng.random() < ratio
            }

            if not val_groups or len(val_groups) == len(groups):
                continue

            val_count = sum(len(group_records[g]) for g in val_groups)
            val_boxes = Counter()
            for group in val_groups:
                val_boxes.update(group_counts[group])

            image_fraction = val_count / len(records)

            score = 5 * abs(image_fraction - ratio)

            class_deviations = []
            class_present_in_both = True

            for label in TARGET_CLASSES:
                total = target_totals[label]
                if total == 0:
                    continue

                val_class_count = val_boxes[label]

                if val_class_count == 0 or val_class_count == total:
                    class_present_in_both = False

                class_deviations.append(
                    abs(val_class_count / total - ratio)
                )

            if class_deviations:
                score += sum(class_deviations) / len(class_deviations)

            candidate = (score, val_groups, block_size)

            if score < (best_fallback[0] if best_fallback else float("inf")):
                best_fallback = candidate

            if class_present_in_both and score < best_score:
                best_valid = candidate
                best_score = score

        if best_valid is not None:
            print(f"Using numeric ID blocks of {block_size}.")
            return best_valid[1], block_size

    if best_fallback is not None:
        print(
            "WARNING: Could not balance every class across both splits. "
            "Using the best available blocked split; inspect its report."
        )
        return best_fallback[1], best_fallback[2]

    raise RuntimeError("Unable to create a non-empty train/validation split.")


def main():
    if not LABEL_MAP.is_file():
        raise FileNotFoundError(f"Label map not found: {LABEL_MAP}")

    if OUT_DIR.exists() and any(OUT_DIR.iterdir()):
        raise FileExistsError(
            f"Output folder is not empty: {OUT_DIR}\n"
            "Inspect its contents before moving or deleting anything."
        )

    records, target_totals, ignored_totals, empty_xmls = inspect_source()

    if not records:
        raise RuntimeError("No verified image/XML pairs found.")

    val_groups, block_size = choose_blocked_split(records, target_totals)

    splits = {
        "train": [],
        "val": [],
    }

    for record in records:
        group = record["group_id"] // block_size
        split = "val" if group in val_groups else "train"
        splits[split].append(record)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Copy images and write one YOLO label file per image.
    # Empty label files intentionally represent images with no target-class
    # boxes; they are useful negative examples for the four-class task.
    split_box_counts = {}
    split_empty_target_images = {}

    for split, split_records in splits.items():
        image_out = OUT_DIR / "images" / split
        label_out = OUT_DIR / "labels" / split
        image_out.mkdir(parents=True, exist_ok=True)
        label_out.mkdir(parents=True, exist_ok=True)

        box_counts = Counter()
        empty_target_count = 0

        for record in split_records:
            destination_image = image_out / record["image_path"].name
            destination_label = label_out / f"{record['stem']}.txt"

            shutil.copy2(record["image_path"], destination_image)

            label_text = "\n".join(record["yolo_lines"])
            destination_label.write_text(
                label_text + ("\n" if label_text else ""),
                encoding="utf-8",
            )

            box_counts.update(record["class_counts"])

            if not record["yolo_lines"]:
                empty_target_count += 1

        split_box_counts[split] = dict(box_counts)
        split_empty_target_images[split] = empty_target_count

    dataset_yaml = OUT_DIR / "data.yaml"
    dataset_yaml.write_text(
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

    manifest_path = OUT_DIR / "split_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "image", "split", "numeric_id_group",
            "D00_boxes", "D10_boxes", "D20_boxes", "D40_boxes",
            "has_target_box",
        ])

        for split, split_records in splits.items():
            for record in split_records:
                counts = record["class_counts"]
                writer.writerow([
                    record["image_path"].name,
                    split,
                    record["group_id"] // block_size,
                    counts["D00"],
                    counts["D10"],
                    counts["D20"],
                    counts["D40"],
                    int(bool(record["yolo_lines"])),
                ])

    report = {
        "source": str(IMAGE_DIR),
        "images": len(records),
        "empty_source_xmls": empty_xmls,
        "target_classes": list(TARGET_CLASSES),
        "target_box_totals": dict(target_totals),
        "ignored_non_target_box_totals": dict(ignored_totals),
        "validation_ratio_requested": VAL_RATIO,
        "block_size": block_size,
        "seed": SEED,
        "split_images": {
            split: len(items) for split, items in splits.items()
        },
        "split_box_counts": split_box_counts,
        "split_images_without_target_boxes": split_empty_target_images,
        "warning": (
            "Numeric filename blocks are a proxy for acquisition groups; "
            "they do not prove temporal independence."
        ),
    }

    (OUT_DIR / "preparation_report.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("\n========== PREPARATION REPORT ==========")
    print(f"Total images: {len(records)}")
    print(f"Source XML files with no objects: {empty_xmls}")
    print(f"Output: {OUT_DIR}")
    print(f"Dataset YAML: {dataset_yaml}")

    print("\nTarget-class boxes:")
    for label in TARGET_CLASSES:
        print(
            f"  {label}: {target_totals[label]} "
            f"(train={split_box_counts['train'].get(label, 0)}, "
            f"val={split_box_counts['val'].get(label, 0)})"
        )

    print("\nExcluded non-target labels:")
    for label, count in ignored_totals.most_common():
        print(f"  {label}: {count}")

    print("\nImages per split:")
    for split in ("train", "val"):
        print(
            f"  {split}: {len(splits[split])} images, "
            f"{split_empty_target_images[split]} without target boxes"
        )

    print("\nRESULT: Dataset conversion finished.")
    print("Review preparation_report.json before training.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        sys.exit(1)
