
from collections import Counter, defaultdict
from pathlib import Path
import random
import re
import xml.etree.ElementTree as ET

import cv2


DATA_ROOT = Path("data/raw/India_train_data")
IMAGE_DIR = DATA_ROOT / "images"
XML_DIR = DATA_ROOT / "annotations" / "xmls"
LABEL_MAP = Path("data/raw/label_map.pbtxt")
OUTPUT_DIR = Path("outputs/annotated/dataset_audit")

SEED = 2026
EMPTY_PREVIEWS = 4


def load_label_map():
    if not LABEL_MAP.is_file():
        print(f"WARNING: Label map not found: {LABEL_MAP}")
        return set()

    text = LABEL_MAP.read_text(encoding="utf-8", errors="replace")
    return set(re.findall(r"\bname\s*:\s*['\"]([^'\"]+)['\"]", text))


def main():
    rng = random.Random(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    images = {
        p.stem: p
        for p in IMAGE_DIR.rglob("*")
        if p.is_file()
        and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    }

    by_class = defaultdict(list)
    empty_xmls = []
    class_counts = Counter()
    bad_xmls = []

    for xml_path in sorted(XML_DIR.rglob("*.xml")):
        try:
            root = ET.parse(xml_path).getroot()
        except (ET.ParseError, OSError) as exc:
            bad_xmls.append(f"{xml_path.name}: {exc}")
            continue

        image_path = images.get(xml_path.stem)
        if image_path is None:
            continue

        objects = root.findall("object")

        if not objects:
            empty_xmls.append((xml_path, image_path))
            continue

        labels_in_image = set()

        for obj in objects:
            label = (obj.findtext("name") or "").strip()
            box = obj.find("bndbox")

            if label and box is not None:
                class_counts[label] += 1
                labels_in_image.add(label)

        for label in labels_in_image:
            by_class[label].append((xml_path, image_path))

    map_classes = load_label_map()
    dataset_classes = set(by_class)

    print("========== CLASS AUDIT ==========")
    print("Label map classes:", sorted(map_classes))
    print("Dataset classes:", sorted(dataset_classes))
    print("Classes missing from label map:", sorted(dataset_classes - map_classes))
    print("Classes absent from this dataset:", sorted(map_classes - dataset_classes))
    print("Empty annotations:", len(empty_xmls))
    print("XML parsing errors:", len(bad_xmls))

    print("\n========== BOX COUNTS ==========")
    for label, count in class_counts.most_common():
        print(f"{label}: {count}")

    def create_preview(xml_path, image_path, label):
        image = cv2.imread(str(image_path))

        if image is None:
            print("UNREADABLE IMAGE:", image_path)
            return

        root = ET.parse(xml_path).getroot()

        for obj in root.findall("object"):
            name = (obj.findtext("name") or "UNKNOWN").strip()
            box = obj.find("bndbox")

            if box is None:
                continue

            try:
                x1 = int(float(box.findtext("xmin")))
                y1 = int(float(box.findtext("ymin")))
                x2 = int(float(box.findtext("xmax")))
                y2 = int(float(box.findtext("ymax")))
            except (TypeError, ValueError):
                continue

            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                image,
                name,
                (x1, max(18, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

        if label == "EMPTY":
            cv2.putText(
                image,
                "NO OBJECTS IN XML",
                (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )

        filename = f"{label}__{image_path.stem}.jpg"
        destination = OUTPUT_DIR / filename

        if not cv2.imwrite(str(destination), image):
            print("FAILED TO SAVE:", destination)
        else:
            print("Saved:", destination)

    print("\n========== GENERATING PREVIEWS ==========")

    # One deterministic example per class, including rare classes.
    for label in sorted(by_class):
        xml_path, image_path = rng.choice(by_class[label])
        create_preview(xml_path, image_path, label)

    # Sample apparently negative images for manual inspection.
    selected_empty = rng.sample(
        empty_xmls, min(EMPTY_PREVIEWS, len(empty_xmls))
    )

    for xml_path, image_path in selected_empty:
        create_preview(xml_path, image_path, "EMPTY")

    if "D0w0" in by_class:
        print("\nWARNING: Inspect the D0w0 preview manually.")

    print(f"\nOpen the generated images in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
