from pathlib import Path
import csv
import xml.etree.ElementTree as ET
import statistics

ROOT = Path.cwd()
DATASET = ROOT / "data/processed/roadsentry_yolo_multicountry_v2"
XML_DIR = ROOT / "data/raw/India_train_data/annotations/xmls"
LABEL_DIR = DATASET / "labels/train"
MANIFEST = DATASET / "manifest.csv"

records = []

with MANIFEST.open(newline="", encoding="utf-8-sig") as f:
    rows = list(csv.DictReader(f))

for row in rows:
    if row["country"] != "India" or row["split"].lower() != "train":
        continue

    stem = row["source_stem"]
    prepared = f"India__{stem}"
    xml_path = XML_DIR / f"{stem}.xml"
    label_path = LABEL_DIR / f"{prepared}.txt"

    if not xml_path.is_file() or not label_path.is_file():
        continue

    root = ET.parse(xml_path).getroot()
    size = root.find("size")
    if size is None:
        continue

    width = float(size.findtext("width", "0"))
    height = float(size.findtext("height", "0"))
    if width <= 0 or height <= 0:
        continue

    xml_boxes = []
    for obj in root.findall("object"):
        if obj.findtext("name", "").strip() != "D10":
            continue
        b = obj.find("bndbox")
        if b is None:
            continue
        try:
            x1 = float(b.findtext("xmin"))
            y1 = float(b.findtext("ymin"))
            x2 = float(b.findtext("xmax"))
            y2 = float(b.findtext("ymax"))
        except (TypeError, ValueError):
            continue
        xml_boxes.append((x1, y1, x2, y2))

    yolo_boxes = []
    for line in label_path.read_text().splitlines():
        parts = line.split()
        if len(parts) != 5 or parts[0] != "1":
            continue
        xc, yc, bw, bh = map(float, parts[1:])
        x1 = (xc - bw / 2) * width
        y1 = (yc - bh / 2) * height
        x2 = (xc + bw / 2) * width
        y2 = (yc + bh / 2) * height
        yolo_boxes.append((x1, y1, x2, y2))

    if len(xml_boxes) != len(yolo_boxes):
        print("COUNT MISMATCH:", prepared, len(xml_boxes), len(yolo_boxes))
        continue

    for a, b in zip(xml_boxes, yolo_boxes):
        x1, y1, x2, y2 = b
        w = x2 - x1
        h = y2 - y1
        records.append({
            "stem": prepared,
            "width": w,
            "height": h,
            "aspect": w / h if h > 0 else float("inf"),
            "area_fraction": (w * h) / (width * height),
            "edge": min(x1, y1, width - x2, height - y2) < 2,
            "coordinate_error": max(abs(a[i] - b[i]) for i in range(4)),
        })

if not records:
    raise SystemExit("No D10 boxes available for geometry audit.")

def report(name, values):
    values = sorted(values)
    print(
        f"{name}: min={min(values):.4f}, "
        f"median={statistics.median(values):.4f}, "
        f"max={max(values):.4f}"
    )

print("=== INDIA D10 BOX GEOMETRY ===")
print("D10 boxes checked:", len(records))
report("Box width in pixels", [r["width"] for r in records])
report("Box height in pixels", [r["height"] for r in records])
report("Width / height ratio", [r["aspect"] for r in records])
report("Fraction of image area", [r["area_fraction"] for r in records])
report("XML-to-YOLO coordinate error (pixels)", [r["coordinate_error"] for r in records])
print("Boxes touching/near image edge:", sum(r["edge"] for r in records))

print("\n=== WIDEST BOXES ===")
for r in sorted(records, key=lambda x: x["aspect"], reverse=True)[:10]:
    print(
        f'{r["stem"]}: width={r["width"]:.1f}px, '
        f'height={r["height"]:.1f}px, '
        f'aspect={r["aspect"]:.1f}, '
        f'coordinate_error={r["coordinate_error"]:.4f}px'
    )
