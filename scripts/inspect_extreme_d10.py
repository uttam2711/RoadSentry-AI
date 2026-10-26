from pathlib import Path
import cv2
import numpy as np

root = Path("data/processed/roadsentry_yolo_multicountry_v2")
images = root / "images/train"
labels = root / "labels/train"
out = Path("outputs/reports/india_d10_extreme_boxes.jpg")

names = [
    "India__India_008307",
    "India__India_002404",
    "India__India_003885",
    "India__India_007329",
    "India__India_008801",
    "India__India_007116",
    "India__India_000193",
]

extensions = [".jpg", ".jpeg", ".png", ".webp", ".bmp"]
thumb_w, thumb_h = 640, 400
canvas = np.full((len(names) * thumb_h, thumb_w, 3), 255, dtype=np.uint8)

for i, name in enumerate(names):
    image_path = next(
        (images / f"{name}{ext}" for ext in extensions
         if (images / f"{name}{ext}").is_file()),
        None
    )
    label_path = labels / f"{name}.txt"

    if image_path is None or not label_path.is_file():
        print("Missing:", name)
        continue

    image = cv2.imread(str(image_path))
    h, w = image.shape[:2]

    for line in label_path.read_text().splitlines():
        p = line.split()
        if len(p) != 5 or p[0] != "1":
            continue
        xc, yc, bw, bh = map(float, p[1:])
        x1, y1 = int((xc-bw/2)*w), int((yc-bh/2)*h)
        x2, y2 = int((xc+bw/2)*w), int((yc+bh/2)*h)
        cv2.rectangle(image, (x1,y1), (x2,y2), (0,0,255), 3)

    scale = min(thumb_w / w, (thumb_h - 35) / h)
    resized = cv2.resize(image, (int(w*scale), int(h*scale)))
    x = (thumb_w - resized.shape[1]) // 2
    y = i * thumb_h
    canvas[y:y+resized.shape[0], x:x+resized.shape[1]] = resized
    cv2.putText(canvas, name, (8, y+thumb_h-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)

out.parent.mkdir(parents=True, exist_ok=True)
cv2.imwrite(str(out), canvas)
print("Saved:", out.resolve())
