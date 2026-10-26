"""OpenCV-based visual analysis to guide targeted road-damage reinspection."""

from pathlib import Path

import cv2
import numpy as np


def inspect_candidate_crop(image_path, detection, output_dir, index=0):
    """Measure a candidate region and decide whether to reinspect it.

    The thresholds are transparent prototype heuristics, not validated
    measures of physical damage severity or road safety.
    """
    image_path = Path(image_path).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"OpenCV could not read image: {image_path}")

    height, width = image.shape[:2]
    bbox = detection.get("bbox_xyxy")

    if not bbox or len(bbox) != 4:
        raise ValueError("Detection must contain four bbox_xyxy coordinates")

    x1, y1, x2, y2 = map(float, bbox)
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(width, int(np.ceil(x2))), min(height, int(np.ceil(y2)))

    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Invalid detection bounding box: {bbox}")

    box_width = x2 - x1
    box_height = y2 - y1

    # Expand the candidate region to retain some surrounding road context.
    pad_x = max(8, int(box_width * 0.25))
    pad_y = max(8, int(box_height * 0.25))

    crop_x1 = max(0, x1 - pad_x)
    crop_y1 = max(0, y1 - pad_y)
    crop_x2 = min(width, x2 + pad_x)
    crop_y2 = min(height, y2 + pad_y)

    crop = image[crop_y1:crop_y2, crop_x1:crop_x2]
    if crop.size == 0:
        raise ValueError("OpenCV produced an empty candidate crop")

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    # OpenCV 5 measurements used as evidence for the next action.
    contrast_std = float(np.std(gray))
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    edges = cv2.Canny(gray, 80, 160)
    edge_density = float(np.count_nonzero(edges) / edges.size)

    crop_path = output_dir / f"candidate_{index + 1:04d}.jpg"
    if not cv2.imwrite(str(crop_path), crop):
        raise RuntimeError(f"Could not save candidate crop: {crop_path}")

    confidence = float(detection.get("confidence", 0.0))
    reasons = []
    opencv_flags = []

    # Experimental selective-reinspection policy.
    # Thresholds are not validated on an independent test set.
    if confidence < 0.45:
        reasons.append("detector_confidence_below_0.45")

    if min(box_width, box_height) < 48:
        reasons.append("candidate_box_dimension_below_48px")

    if sharpness < 50.0:
        opencv_flags.append("opencv_low_sharpness")

    if contrast_std < 22.0:
        opencv_flags.append("opencv_low_contrast")

    if edge_density < 0.015:
        opencv_flags.append("opencv_low_edge_density")

    if contrast_std < 22.0 and confidence < 0.60:
        reasons.append("low_contrast_with_confidence_below_0.60")

    return {
        "tool": "opencv_candidate_crop_analysis",
        "opencv_version": cv2.__version__,
        "source_image": str(image_path),
        "crop_path": str(crop_path),
        "crop_bbox_xyxy": [
            crop_x1, crop_y1, crop_x2, crop_y2
        ],
        "candidate_box_width_px": box_width,
        "candidate_box_height_px": box_height,
        "candidate_area_fraction": round(
            (box_width * box_height) / (width * height), 6
        ),
        "crop_width_px": int(crop.shape[1]),
        "crop_height_px": int(crop.shape[0]),
        "grayscale_contrast_std": round(contrast_std, 3),
        "laplacian_sharpness_variance": round(sharpness, 3),
        "canny_edge_density": round(edge_density, 5),
        "opencv_review_flags": opencv_flags,
        "reinspection_reasons": reasons,
        "reinspection_required": bool(reasons),
        "next_action": (
            "TARGETED_CROP_REINSPECTION"
            if reasons
            else "RECORD_EVIDENCE_AND_REQUEST_HUMAN_REVIEW"
        ),
        "thresholds_are_heuristic": True,
        "decision_policy": {
            "name": "selective_reinspection_v1",
            "confidence_trigger_below": 0.45,
            "box_minimum_dimension_px": 48,
            "low_contrast_std_below": 22.0,
            "low_contrast_confidence_below": 0.60,
            "validated": False,
        },
        "human_review_required": True,
    }
