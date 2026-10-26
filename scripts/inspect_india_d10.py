from pathlib import Path
import math
import cv2

ROOT = Path("data/processed/roadsentry_yolo_multicountry_v2")
IMAGE_DIR = ROOT / "images/train"
LABEL_DIR = ROOT / "labels/train"
OUT = Path("outputs/reports/india_d10_train_contact_sheet.jpg")

CELL_W, CELL_H = 320, 220
COLS = 5
D10_ID = 1

items = []

for label_path in sorted(LABEL_DIR.glob("*.txt")):
    lines = label_path.read_text().splitlines()
    d10_boxes = []

    for line in lines:
        parts = line.split()
        if len(parts) != 5:
            continue
        if parts[0] == str(D10_ID):
            try:
                d10_boxes.append([float(x) for x in parts[1:]])
            except ValueError:
                continue

    if not d10_boxes or not label_path.stem.startswith("India__"):
        continue

    image_path = next(
        (IMAGE_DIR / f"{label_path.stem}{ext}"
         for ext in (".jpg", ".jpeg", ".png", ".webp", ".bmp")
         if (IMAGE_DIR / f"{label_path.stem}{ext}").is_file()),
        None
    )

    if image_path is not None:
        items.append((image_path, label_path.stem, d10_boxes))

if not items:
    raise SystemExit("No Indian D10 training images found. No files changed.")

rows = math.ceil(len(items) / COLS)
sheet = 255 * __import__("numpy").ones(
    (rows * CELL_H, COLS * CELL_W, 3), dtype="uint8"
)

for i, (image_path, stem, boxes) in enumerate(items):
    image = cv2.imread(str(image_path))
    if image is None:
        continue

    h, w = image.shape[:2]

    for xc, yc, bw, bh in boxes:
        x1 = max(0, int((xc - bw / 2) * w))
        y1 = max(0, int((yc - bh / 2) * h))
        x2 = min(w - 1, int((xc + bw / 2) * w))
        y2 = min(h - 1, int((yc + bh / 2) * h))
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)

    thumb = cv2.resize(image, (CELL_W, CELL_H - 30))
    x = (i % COLS) * CELL_W
    y = (i // COLS) * CELL_H

    sheet[y:y + CELL_H - 30, x:x + CELL_W] = thumb
    cv2.putText(
        sheet, f"{stem} | D10 boxes: {len(boxes)}",
        (x + 4, y + CELL_H - 8),
        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1, cv2.LINE_AA
    )

OUT.parent.mkdir(parents=True, exist_ok=True)
if not cv2.imwrite(str(OUT), sheet):
    raise SystemExit("Failed to save contact sheet.")

print("Indian D10 training images:", len(items))
print("D10 boxes:", sum(len(x[2]) for x in items))
print("Contact sheet:", OUT.resolve())
