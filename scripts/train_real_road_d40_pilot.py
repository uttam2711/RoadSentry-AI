from pathlib import Path
from collections import Counter
import random
import shutil
import yaml
import torch
from ultralytics import YOLO

ROOT = Path.cwd()
IMG_DIR = ROOT / "data/processed/roadsentry_real_eval_v1/images"
LBL_DIR = ROOT / "data/processed/roadsentry_real_eval_v1/labels"
START_WEIGHTS = ROOT / "models/external/yolo11s_rdd2022_seed0_best.pt"
DATA_DIR = ROOT / "data/processed/roadsentry_d40_pilot_20261021"

random.seed(42)

if not IMG_DIR.exists() or not LBL_DIR.exists():
    raise SystemExit("Annotation image/label directories were not found.")

if not START_WEIGHTS.exists():
    raise SystemExit(f"Starting checkpoint not found: {START_WEIGHTS}")

if DATA_DIR.exists():
    raise SystemExit(
        f"Dataset already exists: {DATA_DIR}\n"
        "Stopping safely; existing data will not be overwritten."
    )

images = sorted(
    p for p in IMG_DIR.iterdir()
    if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
)
if len(images) < 10:
    raise SystemExit(f"Only {len(images)} images found; expected at least 10.")

# Verify annotations and determine the original D40 class ID.
class_ids = set()
label_contents = {}

for img in images:
    label = LBL_DIR / f"{img.stem}.txt"
    if not label.exists():
        raise SystemExit(f"Missing annotation file: {label}")

    lines = [s.strip() for s in label.read_text().splitlines() if s.strip()]
    checked = []

    for line in lines:
        parts = line.split()
        if len(parts) != 5:
            raise SystemExit(f"Invalid YOLO annotation in {label}: {line}")

        cls = int(float(parts[0]))
        coords = [float(v) for v in parts[1:]]
        if not all(0 <= v <= 1 for v in coords):
            raise SystemExit(f"Coordinates outside 0..1 in {label}: {line}")
        class_ids.add(cls)
        checked.append((cls, coords))

    label_contents[img.stem] = checked

# D40-only annotation files may use class 0 or class 3.
# Do not guess if other class IDs are present.
if not class_ids.issubset({0, 3}):
    raise SystemExit(
        f"Unexpected annotation class IDs: {sorted(class_ids)}. "
        "Stopping rather than silently relabeling."
    )

# Split within both source-video groups so both types appear in train/val.
groups = {}
for img in images:
    if img.stem.startswith("rainwater_"):
        group = "rainwater"
    elif img.stem.startswith("roadstate_"):
        group = "roadstate"
    else:
        group = img.stem.split("_")[0]
    groups.setdefault(group, []).append(img)

train_images, val_images = [], []
for group, members in sorted(groups.items()):
    members = list(members)
    random.shuffle(members)
    n_val = max(1, round(len(members) * 0.20))
    if len(members) >= 4:
        n_val = min(n_val, len(members) - 2)
    else:
        n_val = 1 if len(members) > 1 else 0

    val_images.extend(members[:n_val])
    train_images.extend(members[n_val:])

if not train_images or not val_images:
    raise SystemExit("Could not create a valid train/validation split.")

for split in ("train", "val"):
    (DATA_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

def prepare_split(members, split):
    boxes = 0
    for img in members:
        shutil.copy2(img, DATA_DIR / "images" / split / img.name)
        out_label = DATA_DIR / "labels" / split / f"{img.stem}.txt"

        # Remap the pothole class to 0 for this dedicated one-class model.
        rows = []
        for _, coords in label_contents[img.stem]:
            rows.append("0 " + " ".join(f"{v:.6f}" for v in coords))
        out_label.write_text("\n".join(rows) + ("\n" if rows else ""))
        boxes += len(rows)
    return boxes

train_boxes = prepare_split(train_images, "train")
val_boxes = prepare_split(val_images, "val")

data_yaml = DATA_DIR / "data.yaml"
data_yaml.write_text(yaml.safe_dump({
    "path": str(DATA_DIR.resolve()),
    "train": "images/train",
    "val": "images/val",
    "names": {0: "D40"},
}, sort_keys=False))

print(f"Training images: {len(train_images)}; pothole boxes: {train_boxes}")
print(f"Validation images: {len(val_images)}; pothole boxes: {val_boxes}")
print(f"Dataset: {DATA_DIR}")
print("Starting a separate one-class fine-tuning experiment...")

device = 0 if torch.cuda.is_available() else "cpu"
model = YOLO(str(START_WEIGHTS))

model.train(
    data=str(data_yaml),
    imgsz=640,
    epochs=15,
    patience=5,
    batch=4,
    workers=2,
    device=device,
    project=str(ROOT / "runs/roadsentry"),
    name="d40_realroad_specialist_pilot_20261021",
    exist_ok=False,
    pretrained=True,
    seed=42,
    deterministic=True,
    cos_lr=True,
    close_mosaic=3,
    mosaic=0.5,
    mixup=0.0,
    fliplr=0.5,
    flipud=0.0,
    degrees=2.0,
    translate=0.05,
    scale=0.3,
    hsv_h=0.015,
    hsv_s=0.5,
    hsv_v=0.3,
    amp=True,
    plots=True,
    save=True,
)

print("\nPILOT TRAINING FINISHED")
print("Check the run's weights/best.pt and validation metrics.")
