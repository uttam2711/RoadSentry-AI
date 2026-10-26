# RoadSentry AI

Road-damage detection using existing YOLO models, full-frame inference,
and overlapping-tile pothole detection.

## Requirements
- Linux or WSL
- Python environment with the project's installed dependencies
- NVIDIA GPU recommended
- Existing model checkpoints under `runs/roadsentry/`

## Run
./run_roadsentry.sh \
  data/raw/external_eval/rainwater_potholes.webm \
  outputs/final_delivery/rainwater_potholes/new_run.mp4

## Selected inference settings
- Full-frame pothole confidence: 0.08
- Tiled pothole confidence: 0.20
- Tile size: 512
- Tile overlap: 45%
- Tile inference size: 640

## Outputs
The runner creates an annotated MP4, an event CSV, and a JSON report.

## Important limitations
Event counts are frame-level detections, not unique potholes.
The selected settings were visually reviewed on two videos; the reported
detection counts do not establish measured precision or recall.
Independent ground-truth evaluation is needed before making accuracy claims.
