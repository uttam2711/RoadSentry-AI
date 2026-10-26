#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if [[ -f ".venv/bin/activate" ]]; then
    source .venv/bin/activate
fi

python scripts/inspect_video_multiscale_d40.py \
  --source "${1:?Usage: ./run_roadsentry.sh path/to/video.webm output.mp4}" \
  --output "${2:?Usage: ./run_roadsentry.sh path/to/video.webm output.mp4}" \
  --full-conf 0.08 \
  --conf 0.20 \
  --tile 512 \
  --overlap 0.45 \
  --tile-imgsz 640
