
from collections import Counter
from pathlib import Path
import json
import xml.etree.ElementTree as ET

ROOT = Path("data/raw")

COUNTRIES = {
    "India": ROOT / "India_train_data",
    "Japan": ROOT / "Japan",
    "Czech": ROOT / "Czech",
}

TARGET_CLASSES = {"D00", "D10", "D20", "D40"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def audit_country(country, base):
    image_dir = base / "images"
    xml_dir = base / "annotations" / "xmls"

    result = {
        "images": 0,
        "xml_files": 0,
        "missing_xml": 0,
        "missing_images": 0,
        "invalid_xml": 0,
        "invalid_boxes": 0,
        "empty_xml": 0,
        "images_without_target_boxes": 0,
        "images_with_target_boxes": 0,
        "images_with_only_non_target_boxes": 0,
        "classes": Counter(),
        "target_boxes": Counter(),
        "non_target_boxes": Counter(),
    }

    if not image_dir.is_dir() or not xml_dir.is_dir():
        result["error"] = (
            f"Required folders missing: {image_dir} or {xml_dir}"
        )
        return result

    images = [
        p for p in image_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]
    xml_files = list(xml_dir.rglob("*.xml"))

    image_stems = {p.stem for p in images}
    xml_stems = {p.stem for p in xml_files}

    result["images"] = len(images)
    result["xml_files"] = len(xml_files)
    result["missing_xml"] = len(image_stems - xml_stems)
    result["missing_images"] = len(xml_stems - image_stems)

    for xml_path in xml_files:
        try:
            root = ET.parse(xml_path).getroot()
        except (ET.ParseError, OSError):
            result["invalid_xml"] += 1
            continue

        objects = root.findall("object")

        if not objects:
            result["empty_xml"] += 1

        target_in_image = 0
        non_target_in_image = 0

        try:
            width = int(root.findtext("size/width"))
            height = int(root.findtext("size/height"))
        except (TypeError, ValueError):
            width = height = None

        for obj in objects:
            label = (obj.findtext("name") or "").strip()

            if not label:
                result["invalid_boxes"] += 1
                continue

            result["classes"][label] += 1

            if label in TARGET_CLASSES:
                result["target_boxes"][label] += 1
                target_in_image += 1
            else:
                result["non_target_boxes"][label] += 1
                non_target_in_image += 1

            box = obj.find("bndbox")
            try:
                x1 = float(box.findtext("xmin"))
                y1 = float(box.findtext("ymin"))
                x2 = float(box.findtext("xmax"))
                y2 = float(box.findtext("ymax"))

                valid = (
                    x1 >= 0 and y1 >= 0
                    and x1 < x2 and y1 < y2
                )

                if width is not None and height is not None:
                    valid = valid and x2 <= width and y2 <= height

                if not valid:
                    result["invalid_boxes"] += 1

            except (AttributeError, TypeError, ValueError):
                result["invalid_boxes"] += 1

        if target_in_image:
            result["images_with_target_boxes"] += 1
        else:
            result["images_without_target_boxes"] += 1

            if non_target_in_image:
                result["images_with_only_non_target_boxes"] += 1

    # Convert counters to ordinary dictionaries for JSON output.
    for key in ("classes", "target_boxes", "non_target_boxes"):
        result[key] = dict(result[key].most_common())

    return result


def main():
    report = {}

    for country, folder in COUNTRIES.items():
        report[country] = audit_country(country, folder)

    print("========== MULTI-COUNTRY AUDIT ==========")

    for country, data in report.items():
        print(f"\n--- {country} ---")

        for key in (
            "images", "xml_files", "missing_xml", "missing_images",
            "invalid_xml", "invalid_boxes", "empty_xml",
            "images_with_target_boxes",
            "images_without_target_boxes",
            "images_with_only_non_target_boxes",
        ):
            if key in data:
                print(f"{key}: {data[key]}")

        print("Target boxes:", data.get("target_boxes", {}))
        print("Other classes:", data.get("non_target_boxes", {}))

        if data.get("error"):
            print("ERROR:", data["error"])

    output = Path("outputs/reports/multicountry_audit.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"\nFull report saved to: {output}")


if __name__ == "__main__":
    main()
