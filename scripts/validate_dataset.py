
from collections import Counter
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import cv2


# Dataset locations
DATA_ROOT = Path("data/raw/India_train_data")
IMAGE_DIR = DATA_ROOT / "images"
ANNOTATION_DIR = DATA_ROOT / "annotations" / "xmls"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def main():
    if not IMAGE_DIR.is_dir() or not ANNOTATION_DIR.is_dir():
        print("ERROR: Dataset folders were not found.")
        print(f"Images:      {IMAGE_DIR}")
        print(f"Annotations: {ANNOTATION_DIR}")
        sys.exit(1)

    images = sorted(
        p for p in IMAGE_DIR.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    xml_files = sorted(ANNOTATION_DIR.rglob("*.xml"))

    # Index images by filename stem, e.g. India_009638
    images_by_stem = {}
    duplicate_image_stems = []

    for image_path in images:
        if image_path.stem in images_by_stem:
            duplicate_image_stems.append(image_path.stem)
        else:
            images_by_stem[image_path.stem] = image_path

    xml_stems = {p.stem for p in xml_files}
    image_stems = set(images_by_stem)

    missing_annotations = sorted(image_stems - xml_stems)
    missing_images = sorted(xml_stems - image_stems)

    class_counts = Counter()
    object_count = 0
    empty_annotations = 0
    unreadable_images = []
    invalid_xml = []
    invalid_boxes = []
    filename_mismatches = []
    dimension_mismatches = []
    boxes_outside_image = []

    print("Validating images and Pascal VOC annotations...")
    print(f"Image files:      {len(images)}")
    print(f"XML annotations:  {len(xml_files)}")

    for index, xml_path in enumerate(xml_files, start=1):
        image_path = images_by_stem.get(xml_path.stem)

        if image_path is None:
            continue

        # Parse XML
        try:
            root = ET.parse(xml_path).getroot()
        except (ET.ParseError, OSError) as exc:
            invalid_xml.append(f"{xml_path}: {exc}")
            continue

        # Confirm that the XML refers to the corresponding image
        xml_filename = (root.findtext("filename") or "").strip()
        if xml_filename and Path(xml_filename).stem != image_path.stem:
            filename_mismatches.append(
                f"{xml_path.name}: filename says {xml_filename}"
            )

        # Decode the actual image and obtain true dimensions
        image = cv2.imread(str(image_path))
        if image is None:
            unreadable_images.append(str(image_path))
            continue

        height, width = image.shape[:2]

        xml_width = root.findtext("size/width")
        xml_height = root.findtext("size/height")

        try:
            if xml_width and xml_height:
                if (int(xml_width), int(xml_height)) != (width, height):
                    dimension_mismatches.append(
                        f"{xml_path.name}: XML {xml_width}x{xml_height}, "
                        f"actual {width}x{height}"
                    )
        except ValueError:
            dimension_mismatches.append(
                f"{xml_path.name}: invalid XML image dimensions"
            )

        objects = root.findall("object")
        if not objects:
            empty_annotations += 1

        for obj in objects:
            label = (obj.findtext("name") or "").strip()
            box = obj.find("bndbox")

            if not label or box is None:
                invalid_boxes.append(
                    f"{xml_path.name}: missing label or bounding box"
                )
                continue

            try:
                x_min = float(box.findtext("xmin"))
                y_min = float(box.findtext("ymin"))
                x_max = float(box.findtext("xmax"))
                y_max = float(box.findtext("ymax"))
            except (TypeError, ValueError):
                invalid_boxes.append(
                    f"{xml_path.name}: non-numeric box coordinates"
                )
                continue

            if x_min >= x_max or y_min >= y_max:
                invalid_boxes.append(
                    f"{xml_path.name}: degenerate box "
                    f"({x_min}, {y_min}, {x_max}, {y_max})"
                )
                continue

            if (
                x_min < 0 or y_min < 0
                or x_max > width or y_max > height
            ):
                boxes_outside_image.append(
                    f"{xml_path.name}: box outside {width}x{height}"
                )

            class_counts[label] += 1
            object_count += 1

        if index % 1000 == 0:
            print(f"Processed {index}/{len(xml_files)} XML files...")

    # Report
    print("\n========== DATASET VALIDATION REPORT ==========")
    print(f"Images:                   {len(images)}")
    print(f"XML files:                {len(xml_files)}")
    print(f"Images without XML:       {len(missing_annotations)}")
    print(f"XML without images:       {len(missing_images)}")
    print(f"Duplicate image stems:    {len(duplicate_image_stems)}")
    print(f"Invalid XML files:        {len(invalid_xml)}")
    print(f"Unreadable images:        {len(unreadable_images)}")
    print(f"Filename mismatches:      {len(filename_mismatches)}")
    print(f"Dimension mismatches:     {len(dimension_mismatches)}")
    print(f"Invalid bounding boxes:   {len(invalid_boxes)}")
    print(f"Boxes outside image:      {len(boxes_outside_image)}")
    print(f"Annotations with no objects: {empty_annotations}")
    print(f"Total valid boxes counted: {object_count}")

    print("\n========== CLASS DISTRIBUTION ==========")
    if class_counts:
        for label, count in class_counts.most_common():
            print(f"{label}: {count}")
    else:
        print("No valid labelled objects found.")

    issue_groups = {
        "Missing annotations": missing_annotations,
        "Missing images": missing_images,
        "Duplicate image stems": duplicate_image_stems,
        "Invalid XML": invalid_xml,
        "Unreadable images": unreadable_images,
        "Filename mismatches": filename_mismatches,
        "Dimension mismatches": dimension_mismatches,
        "Invalid boxes": invalid_boxes,
        "Boxes outside image": boxes_outside_image,
    }

    for title, items in issue_groups.items():
        if items:
            print(f"\n--- {title}: first 5 ---")
            for item in items[:5]:
                print(item)

    critical_issues = (
        missing_annotations
        or missing_images
        or duplicate_image_stems
        or invalid_xml
        or unreadable_images
        or filename_mismatches
        or invalid_boxes
    )

    if critical_issues:
        print("\nRESULT: CHECK REQUIRED — resolve reported issues.")
        sys.exit(1)

    print("\nRESULT: STRUCTURAL CHECK PASSED.")
    print("Review dimension mismatches and out-of-image boxes before training.")


if __name__ == "__main__":
    main()
