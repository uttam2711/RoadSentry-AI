from pathlib import Path
import csv
import os
import yaml

ROOT = Path.cwd()
SRC = ROOT / "data/processed/roadsentry_yolo_multicountry_v2"
DST = ROOT / "data/processed/roadsentry_yolo_india_d10_pilot_v1"

if DST.exists():
    raise SystemExit(f"STOP: destination already exists: {DST}")

with (SRC / "manifest.csv").open(
    newline="", encoding="utf-8-sig"
) as f:
    rows = list(csv.DictReader(f))

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

if len(d10_stems) != 45:
    raise SystemExit(f"Expected 45 Indian D10 images, found {len(d10_stems)}")

try:
    for split in ("train", "val"):
        for kind in ("images", "labels"):
            src_dir = SRC / kind / split
            dst_dir = DST / kind / split
            dst_dir.mkdir(parents=True, exist_ok=True)
            for source in src_dir.iterdir():
                if source.is_file():
                    os.symlink(source.resolve(), dst_dir / source.name)

    for stem in sorted(d10_stems):
        images = [
            p for p in (DST / "images/train").glob(f"{stem}.*")
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
        ]
        label = DST / "labels/train" / f"{stem}.txt"
        if len(images) != 1 or not label.is_file():
            raise RuntimeError(f"Invalid image-label pair: {stem}")

        image = images[0]
        os.symlink(
            image.resolve(),
            DST / "images/train" / f"{stem}__repeat1{image.suffix}"
        )
        os.symlink(
            label.resolve(),
            DST / "labels/train" / f"{stem}__repeat1.txt"
        )

    base = yaml.safe_load((SRC / "data.yaml").read_text())
    config = {
        "path": str(DST.resolve()),
        "train": "images/train",
        "val": "images/val",
        "nc": base["nc"],
        "names": base["names"],
    }
    (DST / "data.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False)
    )

    print("Pilot dataset created:", DST)
    print("Unique Indian D10-positive images:", len(d10_stems))
    print("Additional training entries:", len(d10_stems))
    print("Validation images:", len(list((DST / "images/val").iterdir())))

except Exception:
    import shutil
    shutil.rmtree(DST, ignore_errors=True)
    raise
