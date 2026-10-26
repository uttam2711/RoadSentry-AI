
from collections import Counter
from pathlib import Path
import xml.etree.ElementTree as ET

import cv2

ROOT = Path("data/raw")

COUNTRIES = {
    "India": ROOT / "India_train_data",
    "Japan": ROOT / "Japan",
    "Czech": ROOT / "Czech",
}

TARGETS = {"D00", "D10", "D20", "D40"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def audit(country, base):
    image_dir = base / "images"
    xml_dir = base / "annotations" / "xmls"

    images = {
        p.stem: p
        for p in image_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    }

    xmls = {p.stem: p for p in xml_dir.rglob("*.xml")}

    result = Counter()
    target_boxes = Counter()
    other_boxes = Counter()
    invalid_examples = []
    mixed_examples = []

    for stem, xml_path in xmls.items():
        if stem not in images:
            result["missing_image"] += 1
            continue

        try:
            root = ET.parse(xml_path).getroot()
        except (ET.ParseError, OSError):
            result["invalid_xml"] += 1
            continue

        image = cv2.imread(str(images[stem]))
        if image is None:
            result["unreadable_image"] += 1
            continue

        height, width = image.shape[:2]
        target_labels = set()
        other_labels = set()
        objects = root.findall("object")

        if not objects:
            result["empty_xml"] += 1

        for obj in objects:
            label = (obj.findtext("name") or "").strip()
            box = obj.find("bndbox")

            if label in TARGETS:
                target_boxes[label] += 1
                target_labels.add(label)
            else:
                other_boxes[label or "<missing-label>"] += 1
                other_labels.add(label or "<missing-label>")

            try:
                if box is None:
                    raise ValueError("missing bndbox")

                coords = [
                    float(box.findtext("xmin")),
                    float(box.findtext("ymin")),
                    float(box.findtext("xmax")),
                    float(box.findtext("ymax")),
                ]
                x1, y1, x2, y2 = coords

                if not (
                    0 <= x1 < x2 <= width
                    and 0 <= y1 < y2 <= height
                ):
                    raise ValueError(
                        f"invalid coordinates {coords}, image={width}x{height}"
                    )

            except (TypeError, ValueError) as exc:
                if len(invalid_examples) < 20:
                    invalid_examples.append(
                        f"{country}/{xml_path.name}: {label}: {exc}"
                    )
                result["invalid_boxes"] += 1

        if target_labels and other_labels:
            result["mixed_target_and_other"] += 1
            if len(mixed_examples) < 5:
                mixed_examples.append(
                    f"{country}/{xml_path.name}: "
                    f"targets={sorted(target_labels)}, "
                    f"others={sorted(other_labels)}"
                )
        elif target_labels:
            result["target_only"] += 1
        elif objects:
            result["other_only"] += 1

    print(f"\n========== {country.upper()} ==========")
    print(f"Images: {len(images)}")
    print(f"XML files: {len(xmls)}")

    for name in (
        "empty_xml",
        "target_only",
        "other_only",
        "mixed_target_and_other",
        "missing_image",
        "invalid_xml",
        "unreadable_image",
        "invalid_boxes",
    ):
        print(f"{name}: {result[name]}")

    print("Target boxes:", dict(target_boxes))
    print("Other boxes:", dict(other_boxes))

    if mixed_examples:
        print("Mixed-label examples:")
        for item in mixed_examples:
            print(" ", item)

    if invalid_examples:
        print("Invalid-box examples:")
        for item in invalid_examples:
            print(" ", item)

    return result


def main():
    totals = Counter()

    for country, base in COUNTRIES.items():
        totals.update(audit(country, base))

    print("\n========== ALL COUNTRIES ==========")
    for name in (
        "empty_xml",
        "target_only",
        "other_only",
        "mixed_target_and_other",
        "invalid_boxes",
    ):
        print(f"{name}: {totals[name]}")


if __name__ == "__main__":
    main()
