from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO

root = Path(".")
img_dir = root / "data/processed/roadsentry_real_eval_v1/images"
lbl_dir = root / "data/processed/roadsentry_real_eval_v1/labels"
weights = root / "runs/roadsentry/yolo11s_rdd_warmstart_9class_640_20261021/weights/best.pt"
out_dir = root / "outputs/demo/real_road_eval_v1_updated_20261021"
overlay_dir = out_dir / "overlays"
overlay_dir.mkdir(parents=True, exist_ok=True)

images = sorted(
    p for p in img_dir.iterdir()
    if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
)

model = YOLO(str(weights))
names = model.names
items = names.items() if isinstance(names, dict) else enumerate(names)
d40_ids = [
    int(i) for i, name in items
    if str(name).strip().upper() == "D40"
]
if not d40_ids:
    raise SystemExit(f"No D40 class found: {names}")
d40_id = d40_ids[0]

results = model.predict(
    source=[str(p) for p in images],
    imgsz=640,
    conf=0.15,
    batch=8,
    device=0,
    max_det=300,
    verbose=False,
)

result_by_stem = {Path(r.path).stem: r for r in results}
tiles = []

for img_path in images:
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"Skipping unreadable image: {img_path}")
        continue

    h, w = img.shape[:2]
    canvas = img.copy()
    label_path = lbl_dir / f"{img_path.stem}.txt"
    gt_count = 0

    # Ground-truth boxes: GREEN
    if label_path.exists():
        for line in label_path.read_text().splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue
            _, xc, yc, bw, bh = map(float, parts[:5])
            x1 = int((xc - bw / 2) * w)
            y1 = int((yc - bh / 2) * h)
            x2 = int((xc + bw / 2) * w)
            y2 = int((yc + bh / 2) * h)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 0), 2)
            gt_count += 1

    # YOLO11s predictions: RED
    result = result_by_stem.get(img_path.stem)
    pred_count = 0

    if result is not None and result.boxes is not None:
        for box in result.boxes:
            if int(box.cls.item()) != d40_id:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = float(box.conf.item())
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(
                canvas, f"D40 {conf:.2f}", (x1, max(20, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2
            )
            pred_count += 1

    cv2.putText(
        canvas, f"GT: {gt_count} | Predictions: {pred_count}",
        (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2
    )
    cv2.imwrite(str(overlay_dir / f"{img_path.stem}_overlay.jpg"), canvas)

    # Thumbnail for contact sheet
    tile_w, tile_h = 480, 300
    scale = min((tile_w - 12) / w, (tile_h - 38) / h)
    resized = cv2.resize(
        canvas, (int(w * scale), int(h * scale)),
        interpolation=cv2.INTER_AREA
    )
    tile = np.full((tile_h, tile_w, 3), 245, dtype=np.uint8)
    rh, rw = resized.shape[:2]
    tile[32:32 + rh, (tile_w - rw) // 2:(tile_w - rw) // 2 + rw] = resized
    cv2.putText(
        tile, img_path.stem, (8, 22),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 20, 20), 1
    )
    tiles.append(tile)

cols = 3
rows = (len(tiles) + cols - 1) // cols
sheet = np.full((rows * 300, cols * 480, 3), 230, dtype=np.uint8)

for i, tile in enumerate(tiles):
    y = (i // cols) * 300
    x = (i % cols) * 480
    sheet[y:y + 300, x:x + 480] = tile

sheet_path = out_dir / "updated_prediction_vs_ground_truth.jpg"
cv2.imwrite(str(sheet_path), sheet)

print(f"Frames reviewed: {len(tiles)}")
print(f"Contact sheet: {sheet_path}")
print(f"Full-size overlays: {overlay_dir}")
