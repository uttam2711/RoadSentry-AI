from pathlib import Path
import cv2

ROOT = Path("data/raw/external_eval/independent_sample_v1")
FRAMES = sorted((ROOT / "frames").glob("frame_*s.jpg"))
OUT = ROOT / "labels_gt"

CLASSES = {
    0: "D00 - longitudinal crack",
    1: "D10 - transverse crack",
    2: "D20 - alligator cracking",
    3: "D40 - pothole",
}

if not FRAMES:
    raise SystemExit(f"No extracted frames found in {ROOT / 'frames'}")
OUT.mkdir(parents=True, exist_ok=True)
index = 0
boxes = []
current_class = None
drawing = False
start = None
end = None
scale = 1.0
display = None


def mouse(event, x, y, flags, param):
    global drawing, start, end, boxes, current_class

    if event == cv2.EVENT_LBUTTONDOWN:
        if current_class is None:
            print("Select a class first: press 0, 1, 2, or 3.")
            return
        drawing = True
        start = (x, y)
        end = (x, y)

    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        end = (x, y)

    elif event == cv2.EVENT_LBUTTONUP and drawing:
        drawing = False
        end = (x, y)

        x1, x2 = sorted((start[0], end[0]))
        y1, y2 = sorted((start[1], end[1]))

        if x2 - x1 >= 5 and y2 - y1 >= 5:
            boxes.append((x1, y1, x2, y2, current_class))
            print("Added:", CLASSES[current_class])


def save_labels(frame_path, display_width, display_height, annotations):
    path = OUT / f"{frame_path.stem}.txt"
    lines = []

    for x1, y1, x2, y2, cls in annotations:
        xc = ((x1 + x2) / 2) / display_width
        yc = ((y1 + y2) / 2) / display_height
        bw = (x2 - x1) / display_width
        bh = (y2 - y1) / display_height
        lines.append(f"{cls} {xc:.7f} {yc:.7f} {bw:.7f} {bh:.7f}")

    path.write_text("".join(line + "\n" for line in lines))
    print(f"Saved {len(lines)} boxes: {path}")


def render(image, frame_path, display_width, display_height):
    canvas = image.copy()

    for x1, y1, x2, y2, cls in boxes:
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            canvas, CLASSES[cls].split(" - ")[0],
            (x1, max(18, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
        )

    if drawing and start and end:
        cv2.rectangle(canvas, start, end, (0, 255, 255), 2)

    selected = CLASSES[current_class] if current_class is not None else "None"
    lines = [
        f"{frame_path.name} | {index + 1}/{len(FRAMES)}",
        f"Selected class: {selected}",
        "Keys 0-3 select class | ENTER save+next | U undo",
        "N save empty+next | R clear boxes | Q save+quit",
    ]

    y = 22
    for line in lines:
        cv2.putText(
            canvas, line, (10, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.48,
            (0, 0, 0), 3, cv2.LINE_AA
        )
        cv2.putText(
            canvas, line, (10, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.48,
            (255, 255, 255), 1, cv2.LINE_AA
        )
        y += 23

    return canvas


window = "RoadSentry Ground Truth Annotation"
cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
cv2.setMouseCallback(window, mouse)

try:
    while index < len(FRAMES):
        frame_path = FRAMES[index]
        original = cv2.imread(str(frame_path))
        if original is None:
            print("Skipping unreadable frame:", frame_path)
            index += 1
            continue

        h, w = original.shape[:2]
        scale = min(1200 / w, 760 / h, 1.0)
        display_width = int(w * scale)
        display_height = int(h * scale)
        image = cv2.resize(original, (display_width, display_height))

        # Restore previously saved boxes so annotation can resume.
        label_file = OUT / f"{frame_path.stem}.txt"
        boxes = []
        if label_file.is_file():
            for line in label_file.read_text().splitlines():
                parts = line.split()
                if len(parts) < 5:
                    continue
                try:
                    cls = int(parts[0])
                    xc, yc, bw, bh = map(float, parts[1:5])
                except ValueError:
                    continue
                x1 = max(0, int((xc - bw / 2) * display_width))
                y1 = max(0, int((yc - bh / 2) * display_height))
                x2 = min(display_width - 1, int((xc + bw / 2) * display_width))
                y2 = min(display_height - 1, int((yc + bh / 2) * display_height))
                if x2 > x1 and y2 > y1:
                    boxes.append((x1, y1, x2, y2, cls))
            print(f"Loaded {len(boxes)} existing boxes for {frame_path.name}")

        current_class = None
        drawing = False
        start = end = None

        print(f"\nFrame {index + 1}/{len(FRAMES)}: {frame_path.name}")
        while True:
            canvas = render(image, frame_path, display_width, display_height)
            cv2.imshow(window, canvas)
            key = cv2.waitKey(20) & 0xFF

            if key in (ord("0"), ord("1"), ord("2"), ord("3")):
                current_class = int(chr(key))
                print("Selected:", CLASSES[current_class])

            elif key in (ord("u"), ord("U")):
                if boxes:
                    print("Undo:", CLASSES[boxes.pop()[4]])
                else:
                    print("No boxes to undo.")

            elif key in (ord("r"), ord("R")):
                boxes.clear()
                print("Cleared all boxes for this frame.")

            elif key in (10, 13):  # Enter
                save_labels(frame_path, display_width, display_height, boxes)
                index += 1
                break

            elif key in (ord("n"), ord("N")):
                boxes.clear()
                save_labels(frame_path, display_width, display_height, boxes)
                index += 1
                break

            elif key in (ord("q"), ord("Q"), 27):
                save_labels(frame_path, display_width, display_height, boxes)
                print("Stopped. Completed labels have been preserved.")
                raise SystemExit(0)

finally:
    cv2.destroyAllWindows()

print("\nAnnotation pass completed.")
print("Labels saved in:", OUT.resolve())
print("No model predictions or weight changes were made.")
