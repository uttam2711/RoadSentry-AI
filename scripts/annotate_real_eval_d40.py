from pathlib import Path
import csv
import cv2

ROOT = Path.cwd()
IMAGE_DIR = ROOT / "data/processed/roadsentry_real_eval_v1/images"
LABEL_DIR = ROOT / "data/processed/roadsentry_real_eval_v1/labels"
MANIFEST = ROOT / "data/processed/roadsentry_real_eval_v1/frame_manifest.csv"

LABEL_DIR.mkdir(parents=True, exist_ok=True)
images = sorted(
    p for p in IMAGE_DIR.iterdir()
    if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
)

if not images:
    raise SystemExit(f"No frames found in {IMAGE_DIR}")

index = 0
window = "RoadSentry - D40 Pothole Annotation"
cv2.namedWindow(window, cv2.WINDOW_NORMAL)

def load_boxes(label_path, w, h):
    boxes = []
    if label_path.exists():
        for line in label_path.read_text().splitlines():
            parts = line.split()
            if len(parts) != 5:
                continue
            cls, xc, yc, bw, bh = map(float, parts)
            if int(cls) != 3:
                continue
            boxes.append([
                max(0, int((xc - bw / 2) * w)),
                max(0, int((yc - bh / 2) * h)),
                min(w - 1, int((xc + bw / 2) * w)),
                min(h - 1, int((yc + bh / 2) * h)),
            ])
    return boxes

def save_boxes(path, boxes, w, h):
    lines = []
    for x1, y1, x2, y2 in boxes:
        x1, x2 = sorted((max(0, min(w-1, x1)), max(0, min(w-1, x2))))
        y1, y2 = sorted((max(0, min(h-1, y1)), max(0, min(h-1, y2))))
        if x2 - x1 < 3 or y2 - y1 < 3:
            continue
        xc = ((x1 + x2) / 2) / w
        yc = ((y1 + y2) / 2) / h
        bw = (x2 - x1) / w
        bh = (y2 - y1) / h
        lines.append(f"3 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
    path.write_text("\n".join(lines) + ("\n" if lines else ""))

def draw_callback(event, x, y, flags, param):
    drag = param["drag"]
    scale = param["scale"]
    w, h = param["w"], param["h"]

    if event == cv2.EVENT_LBUTTONDOWN:
        drag["start"] = (x, y)
        drag["current"] = (x, y)

    elif event == cv2.EVENT_MOUSEMOVE and drag["start"] is not None:
        drag["current"] = (x, y)

    elif event == cv2.EVENT_LBUTTONUP and drag["start"] is not None:
        sx, sy = drag["start"]
        x1 = max(0, min(w-1, int(min(sx, x) / scale)))
        y1 = max(0, min(h-1, int(min(sy, y) / scale)))
        x2 = max(0, min(w-1, int(max(sx, x) / scale)))
        y2 = max(0, min(h-1, int(max(sy, y) / scale)))

        if x2 - x1 >= 5 and y2 - y1 >= 5:
            param["boxes"].append([x1, y1, x2, y2])

        drag["start"] = None
        drag["current"] = None

def review_one(i):
    image_path = images[i]
    label_path = LABEL_DIR / f"{image_path.stem}.txt"

    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError(f"Cannot read {image_path}")

    h, w = image.shape[:2]
    scale = min(1280 / w, 720 / h, 1.0)
    dw, dh = int(w * scale), int(h * scale)
    boxes = load_boxes(label_path, w, h)
    reviewed = label_path.exists()
    drag = {"start": None, "current": None}

    param = {
        "drag": drag, "scale": scale, "w": w, "h": h,
        "boxes": boxes,
    }
    cv2.setMouseCallback(window, draw_callback, param)

    while True:
        canvas = cv2.resize(image, (dw, dh))

        for n, (x1, y1, x2, y2) in enumerate(boxes, start=1):
            p1 = (int(x1 * scale), int(y1 * scale))
            p2 = (int(x2 * scale), int(y2 * scale))
            cv2.rectangle(canvas, p1, p2, (0, 255, 0), 2)
            cv2.putText(
                canvas, f"D40 #{n}", (p1[0], max(20, p1[1]-6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2
            )

        status = "REVIEWED" if reviewed else "NOT REVIEWED"
        lines = [
            f"{i+1}/{len(images)}  {image_path.name}  |  {status}",
            "Drag: box definite potholes | S: save/next | 0: no pothole",
            "U: undo | C: clear boxes | P: previous | Q: save/quit",
        ]
        for j, line in enumerate(lines):
            cv2.putText(
                canvas, line, (10, 25 + 27*j),
                cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 255, 255), 2,
                cv2.LINE_AA
            )

        if drag["start"] is not None and drag["current"] is not None:
            cv2.rectangle(
                canvas, drag["start"], drag["current"], (255, 255, 0), 2
            )

        cv2.imshow(window, canvas)
        key = cv2.waitKey(30) & 0xFF

        if key in (ord("u"), ord("U")) and boxes:
            boxes.pop()
        elif key in (ord("c"), ord("C")):
            boxes.clear()
        elif key in (ord("s"), ord("S")):
            save_boxes(label_path, boxes, w, h)
            return "next"
        elif key == ord("0"):
            boxes.clear()
            save_boxes(label_path, boxes, w, h)
            return "next"
        elif key in (ord("p"), ord("P")):
            save_boxes(label_path, boxes, w, h)
            return "previous"
        elif key in (ord("q"), ord("Q"), 27):
            save_boxes(label_path, boxes, w, h)
            return "quit"

def main():
    global index
    while 0 <= index < len(images):
        action = review_one(index)
        if action == "quit":
            break
        if action == "previous":
            index = max(0, index - 1)
        else:
            index += 1

    cv2.destroyAllWindows()

    reviewed = sum(
        (LABEL_DIR / f"{p.stem}.txt").exists() for p in images
    )
    print(f"\nReviewed frames: {reviewed}/{len(images)}")
    print("Image directory:", IMAGE_DIR)
    print("D40 labels:", LABEL_DIR)

    if MANIFEST.exists():
        with MANIFEST.open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        for row in rows:
            label = LABEL_DIR / f"{Path(row['image']).stem}.txt"
            row["review_status"] = (
                f"REVIEWED_D40_{len(label.read_text().splitlines())}"
                if label.exists() else "NOT_REVIEWED"
            )
        with MANIFEST.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print("Manifest updated:", MANIFEST)

if __name__ == "__main__":
    main()
