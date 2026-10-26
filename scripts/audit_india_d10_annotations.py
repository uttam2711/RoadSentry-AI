from pathlib import Path
from collections import Counter
import csv
import xml.etree.ElementTree as ET

ROOT = Path.cwd()
DATASET = ROOT / "data/processed/roadsentry_yolo_multicountry_v2"
XML_DIR = ROOT / "data/raw/India_train_data/annotations/xmls"
LABEL_DIR = DATASET / "labels/train"
MANIFEST = DATASET / "manifest.csv"

CLASS_IDS = {"D00": 0, "D10": 1, "D20": 2, "D40": 3}

with MANIFEST.open(newline="", encoding="utf-8-sig") as f:
    rows = list(csv.DictReader(f))

india_train = [
    r for r in rows
    if r["country"] == "India" and r["split"].lower() == "train"
]

results = []
missing_xml = []
invalid_yolo = []
xml_d10_total = 0
yolo_d10_total = 0

for row in india_train:
    source_stem = row["source_stem"]
    prepared_stem = f"India__{source_stem}"
    label_path = LABEL_DIR / f"{prepared_stem}.txt"

    if not label_path.is_file():
        continue

    yolo_lines = [
        line.split()
        for line in label_path.read_text().splitlines()
        if line.strip()
    ]

    yolo_counts = Counter()
    for parts in yolo_lines:
        if len(parts) != 5:
            invalid_yolo.append((prepared_stem, "wrong field count", parts))
            continue

        try:
            cls = int(parts[0])
            coords = [float(v) for v in parts[1:]]
        except ValueError:
            invalid_yolo.append((prepared_stem, "non-numeric value", parts))
            continue

        if not all(0 <= v <= 1 for v in coords):
            invalid_yolo.append((prepared_stem, "coordinate outside 0..1", parts))
        if coords[2] <= 0 or coords[3] <= 0:
            invalid_yolo.append((prepared_stem, "non-positive box size", parts))

        yolo_counts[cls] += 1

    xml_path = XML_DIR / f"{source_stem}.xml"
    if not xml_path.is_file():
        missing_xml.append(str(xml_path))
        continue

    try:
        tree = ET.parse(xml_path)
    except ET.ParseError as e:
        missing_xml.append(f"{xml_path} [XML parse error: {e}]")
        continue

    xml_counts = Counter(
        obj.findtext("name", default="").strip()
        for obj in tree.getroot().findall("object")
    )

    original_d10 = xml_counts["D10"]
    prepared_d10 = yolo_counts[CLASS_IDS["D10"]]

    xml_d10_total += original_d10
    yolo_d10_total += prepared_d10

    if original_d10 != prepared_d10:
        results.append((
            prepared_stem, original_d10, prepared_d10,
            dict(xml_counts), dict(yolo_counts)
        ))

print("=== INDIA D10 XML vs YOLO AUDIT ===")
print("India training images in manifest:", len(india_train))
print("Original XML D10 boxes:", xml_d10_total)
print("Prepared YOLO D10 boxes:", yolo_d10_total)
print("Images with D10 count mismatch:", len(results))
print("Missing or unreadable XML files:", len(missing_xml))
print("Invalid YOLO label rows/boxes:", len(invalid_yolo))

print("\n=== D10 COUNT MISMATCHES ===")
for stem, original, prepared, xml_counts, yolo_counts in results[:30]:
    print(
        f"{stem}: XML D10={original}, YOLO D10={prepared}; "
        f"XML classes={xml_counts}; YOLO class counts={yolo_counts}"
    )

print("\n=== MISSING/UNREADABLE XML EXAMPLES ===")
for item in missing_xml[:10]:
    print(item)

print("\n=== INVALID YOLO LABEL EXAMPLES ===")
for item in invalid_yolo[:10]:
    print(item)

print("\n=== RESULT ===")
if missing_xml or invalid_yolo or results:
    print("INVESTIGATE: at least one issue needs inspection.")
else:
    print("D10 object counts match for all audited India training images.")
