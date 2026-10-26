# RoadSentry AI — Road Damage Inspection

RoadSentry AI is a computer-vision prototype for identifying potential road
damage in images and videos. It uses a trained Ultralytics YOLO model and
produces annotated images, frame-level video detections, structured JSON/CSV
evidence, heuristic temporal tracks, and an HTML inspection report.

**Status:** Research/demo prototype. Model detections require human review.
This project is not an autonomous road-safety or maintenance decision system.

## Current capabilities

- Run inference on a single road image.
- Analyze a video at a configurable frame interval.
- Export detection classes, confidence scores, bounding boxes and timestamps.
- Group nearby same-class video detections into estimated tracks.
- Generate an HTML report with validation metrics and curated examples.

Track grouping is heuristic. A track is not a verified unique physical defect.
False positives and missed defects are possible.

## Model and dataset

Baseline checkpoint:

`runs/roadsentry/baseline_yolo26n_9class/weights/best.pt`

The checkpoint uses nine dataset classes:

`D00, D10, D20, D40, D01, D11, D43, D44, D50`

The current video runner filters for class IDs 0–3:
`D00, D10, D20, D40`.

Prepared dataset location, when available locally:

`data/processed/roadsentry_yolo_multicountry_v2/`

The prepared dataset contains 21,039 retained images, with 16,803
training images and 4,236 validation images. The dataset combines India,
Japan and Czech data. Dataset files are not included automatically by this
README; paths and image availability depend on the local workspace.

## Baseline validation results

The saved baseline checkpoint was evaluated on the full 4,236-image
validation split at image size 640.

| Metric | Result |
|---|---:|
| Precision | 0.5820 |
| Recall | 0.5356 |
| mAP@0.5 | 0.5737 |
| mAP@0.5:0.95 | 0.2982 |

Full metrics:
`runs/roadsentry/full_validation_baseline_640_20261015/full_validation_metrics.json`

Per-class metrics:
`runs/roadsentry/full_validation_baseline_640_20261015/per_class_metrics.csv`

These are validation results for this checkpoint and evaluation setup.
They are not independent test-set results or a guarantee of real-world
performance. The video demo uses image size 960, so its output is a
separate demonstration run.

## Tested environment

The existing development environment reported:

- Ubuntu under WSL
- Python 3.13.16
- PyTorch 2.13.0+cu130
- Ultralytics 8.4.177
- NVIDIA GeForce RTX 3050 6GB Laptop GPU
- CUDA available in the tested environment

Other environments may require a different PyTorch build.

## Setup

Activate the existing project environment:

```bash
cd ~/projects/roadsentry-ai
source .venv/bin/activate
```

For a fresh environment, use a compatible Python version and install the
dependencies in `requirements.txt`. For NVIDIA GPU support, install a
PyTorch build compatible with your operating system, Python version and
GPU using the official PyTorch instructions:

https://pytorch.org/get-started/locally/

Then install the project dependencies:

```bash
python -m pip install -r requirements.txt
```

Check the environment:

```bash
python -c "import torch, ultralytics, cv2; print('PyTorch:', torch.__version__); print('Ultralytics:', ultralytics.__version__); print('OpenCV:', cv2.__version__); print('CUDA:', torch.cuda.is_available())"
```

## Run inference on one image

Example using an image in the prepared validation dataset:

```bash
cd ~/projects/roadsentry-ai
source .venv/bin/activate

PYTHONPATH="$PWD/src" python scripts/run_sample.py \
  --source "samples/Japan_003532.jpg" \
  --imgsz 640 \
  --conf 0.25
```

By default, annotated output and JSON results are saved under:

`outputs/demo/image_inference/`

Use `python scripts/run_sample.py --help` to see the supported options.

## Existing video demo and inspection report

The current generated report is:

`outputs/demo/video_inspection_targets_960/RoadSentry_inspection_report.html`

The associated H.264 video is:

`outputs/demo/video_inspection_targets_960/Cars_Moving_into_pothole_roadsentry_h264.mp4`

Open the report from WSL on Windows:

```bash
explorer.exe "$(wslpath -w outputs/demo/video_inspection_targets_960/RoadSentry_inspection_report.html)"
```

To inspect video runner options:

```bash
python scripts/inspect_video_targets.py --help
```

Regenerating video outputs can overwrite existing event files. Preserve a
copy of the current demo outputs before rerunning inference. The report
builder uses existing JSON and track data; rebuild the report only after
confirming those source files correspond to the intended demo run.

## Run the API service

RoadSentry exposes a local HTTP API for image-based road-damage inference.

Start the service from the project root:

```bash
cd ~/projects/roadsentry-ai
source .venv/bin/activate

python -m pip install -r requirements.txt
PYTHONPATH="$PWD/src" python -m uvicorn roadsentry.api.main:app \
  --host 127.0.0.1 --port 8000
```

The API uses the baseline checkpoint at:

`runs/roadsentry/baseline_yolo26n_9class/weights/best.pt`

To check service health, open `http://127.0.0.1:8000/health`.

Interactive API documentation is available at
`http://127.0.0.1:8000/docs`.

To test image inference, use the Swagger interface:

1. Open `/docs` in your browser.
2. Expand `POST /predict` and select **Try it out**.
3. Upload a JPEG, PNG or WebP image, then select **Execute**.
4. Inspect the returned JSON detections, confidence scores and bounding boxes.

The API limits uploaded images to 10 MiB. Predictions are advisory and
require human review. This local setup is not a cloud deployment.

Stop the service with `Ctrl+C` in the terminal running Uvicorn.

## Key project files

- `src/roadsentry/vision/detector.py` — image detection wrapper
- `scripts/run_sample.py` — single-image inference
- `scripts/inspect_video_targets.py` — video inference and frame-level events
- `scripts/summarize_tracks.py` — heuristic track summary
- `scripts/build_inspection_report.py` — HTML inspection report
- `configs/class_thresholds_experimental_640.json` — experimental settings;
  not the production configuration
- `outputs/demo/validation_examples/` — curated validation evidence
- `runs/roadsentry/full_validation_baseline_640_20261015/` — baseline metrics

## Interpretation and limitations

- Detection confidence is not the probability that a prediction is correct.
- Frame-level detections may repeat the same physical road feature.
- Heuristic tracks can merge or split detections incorrectly.
- Road texture, shadows, markings and image boundaries can cause false
  positives; small or difficult defects can be missed.
- A human must verify candidate defects before taking action.
- Do not interpret curated example images as full-dataset accuracy.

The baseline model is preserved as the reference checkpoint. Experimental
thresholds must not be described as production improvements without suitable
evaluation.

## Run the inspection triage workflow

Run the explainable inspection prototype on a sample image:

```bash
cd ~/projects/roadsentry-ai
source .venv/bin/activate

PYTHONPATH="$PWD/src" python scripts/run_inspection_agent.py \
  --image outputs/demo/validation_examples/candidate_D20.jpg \
  --output-dir outputs/demo/agent_inspection
```

The workflow produces JSON and HTML inspection reports. Triage scores are prototype heuristics, not validated road-severity measurements. All detections require human review.