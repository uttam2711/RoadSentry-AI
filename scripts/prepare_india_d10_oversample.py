from pathlib import Path
import csv
import os
import shutil
import yaml

ROOT = Path.cwd()
SRC = ROOT / "data/processed/roadsentry_yolo_multicountry_v2"
DST = ROOT / "data/processed/roadsentry_yolo_india_d10_v1"
REPEATS = 3  # Three extra copies of each Indian D10-positive image

if DST.exists():
    raise SystemExit(
        f"STOP: {DST} already exists. "
        "Inspect it before deciding whether to rebuild."
    )

with (SRC / "manifest.csv").open(
    newline="", encoding="utf-8-sig"
) as f:
    rows = list(csv.DictReader(f))

# Identify Indian training images containing D10 (class ID 1).
d10_stems = set()
for row in rows:
    if row["country"] != "India" or row["split"].lower() != "train":
        continue
    stem = f'India__{row["source_stem"]}'
    label = SRC / "labels/train" / f"{stem}.txt"
    if label.is_file() and any(
        line.split() and line.split()[0] == "1"
        for line in label.read_text().splitlines()
    ):
        d10_stems.add(stem)

if not d10_stems:
    raise SystemExit("No Indian D10-positive training images found.")

try:
    for split in ("train", "val"):
        for kind in ("images", "labels"):
            src_dir = SRC / kind / split
            dst_dir = DST / kind / split
            dst_dir.mkdir(parents=True, exist_ok=True)

            for source in src_dir.iterdir():
                if not source.is_file():
                    continue
                target = dst_dir / source.name
                os.symlink(source.resolve(), target)

    # Additional train-only entries. Original files remain unchanged.
    for stem in sorted(d10_stems):
        image_matches = [
            p for p in (DST / "images/train").glob(f"{stem}.*")
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
        ]
        label = DST / "labels/train" / f"{stem}.txt"

        if len(image_matches) != 1 or not label.is_file():
            raise RuntimeError(f"Could not safely locate pair for {stem}")

        image = image_matches[0]
        for repeat in range(1, REPEATS + 1):
            new_stem = f"{stem}__repeat{repeat}"
            os.symlink(
                image.resolve(),
                DST / "images/train" / f"{new_stem}{image.suffix}"
            )
            os.symlink(
                label.resolve(),
                DST / "labels/train" / f"{new_stem}.txt"
            )

    original_yaml = yaml.safe_load((SRC / "data.yaml").read_text())
    config = {
        "path": str(DST.resolve()),
        "train": "images/train",
        "val": "images/val",
        "nc": original_yaml["nc"],
        "names": original_yaml["names"],
    }
    (DST / "data.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False)
    )

    print("Prepared separate oversampling dataset.")
    print("Indian D10-positive source images:", len(d10_stems))
    print("Additional train entries:", len(d10_stems) * REPEATS)
    print("Original validation images preserved:", len(list((DST / "images/val").iterdir())))
    print("Config:", DST / "data.yaml")

except Exception:
    shutil.rmtree(DST, ignore_errors=True)
    raise
