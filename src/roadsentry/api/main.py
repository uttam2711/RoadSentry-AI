import os
import tempfile
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from roadsentry.vision.detector import RoadDamageDetector

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_WEIGHTS = (
    PROJECT_ROOT
    / "runs"
    / "roadsentry"
    / "baseline_yolo26n_9class"
    / "weights"
    / "best.pt"
)

WEIGHTS_PATH = Path(
    os.environ.get("ROADSENTRY_WEIGHTS", str(DEFAULT_WEIGHTS))
).expanduser().resolve()

app = FastAPI(
    title="RoadSentry AI",
    description=(
        "Road-damage detection API. Predictions are advisory "
        "and require human review."
    ),
    version="1.0.0",
)

@lru_cache(maxsize=1)
def get_detector():
    return RoadDamageDetector(
        weights=WEIGHTS_PATH,
        confidence=0.25,
        image_size=640,
    )


@app.get("/")
def root():
    return {
        "service": "RoadSentry AI",
        "status": "running",
        "docs": "/docs",
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_weights_configured": WEIGHTS_PATH.is_file(),
        "model_weights": str(WEIGHTS_PATH),
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    allowed_types = {"image/jpeg", "image/png", "image/webp"}
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=415,
            detail="Upload a JPEG, PNG, or WebP image.",
        )

    suffixes = {
        "image/jpeg": ".jpg",
        "image/png": ".png",
        "image/webp": ".webp",
    }
    temp_path = None

    try:
        # Bound the upload size to 10 MiB.
        content = await file.read(10 * 1024 * 1024 + 1)
        if not content:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")
        if len(content) > 10 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="Image exceeds 10 MiB.")

        with tempfile.NamedTemporaryFile(
            suffix=suffixes[file.content_type], delete=False
        ) as temp_file:
            temp_file.write(content)
            temp_path = Path(temp_file.name)

        # Verify the file is a decodable image before running inference.
        import cv2
        image = cv2.imread(str(temp_path))
        if image is None:
            raise HTTPException(
                status_code=400,
                detail="The uploaded file is not a valid image.",
            )

        detector = get_detector()
        result = detector.predict(temp_path)
        result["uploaded_filename"] = file.filename
        result["note"] = (
            "Prototype output; detections require human verification."
        )
        return JSONResponse(content=result)

    except HTTPException:
        raise
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Inference failed: {type(exc).__name__}: {exc}",
        ) from exc
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
        await file.close()
