"""Orchestrate road-damage detection with active visual reinspection."""

from datetime import datetime, timezone
from pathlib import Path

import cv2

from roadsentry.agent.inspection_tools import build_inspection_record
from roadsentry.agent.opencv_active_perception import inspect_candidate_crop
from roadsentry.reporting.report_generator import (
    write_html_report,
    write_json_report,
)
from roadsentry.video.temporal_analysis import group_frame_detections
from roadsentry.vision.detector import RoadDamageDetector


class InspectionOrchestrator:
    """Run detection, inspect visual evidence, and log follow-up actions."""

    def __init__(self, detector=None):
        if detector is not None:
            self.detector = detector
            return

        import os

        project_root = Path(__file__).resolve().parents[3]
        default_weights = (
            project_root
            / "runs"
            / "roadsentry"
            / "baseline_yolo26n_9class"
            / "weights"
            / "best.pt"
        )
        weights = Path(
            os.environ.get("ROADSENTRY_WEIGHTS", str(default_weights))
        ).expanduser().resolve()

        self.detector = RoadDamageDetector(
            weights=weights,
            confidence=0.25,
            image_size=640,
        )

    @staticmethod
    def _box_iou(box_a, box_b):
        """Calculate intersection-over-union for two xyxy boxes."""
        ax1, ay1, ax2, ay2 = map(float, box_a)
        bx1, by1, bx2, by2 = map(float, box_b)

        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)

        intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        union = area_a + area_b - intersection

        return intersection / union if union > 0 else 0.0

    def _run_active_perception(self, image_path, detection, output_dir, index):
        """Let OpenCV evidence determine whether a second detector pass runs."""
        try:
            visual = inspect_candidate_crop(
                image_path=image_path,
                detection=detection,
                output_dir=output_dir / "opencv_candidates",
                index=index,
            )
        except Exception as exc:
            return {
                "tool": "opencv_candidate_crop_analysis",
                "analysis_status": "FAILED",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "human_review_required": True,
            }, {
                "decision": "REQUEST_HUMAN_REVIEW",
                "decision_reasons": ["opencv_analysis_failed"],
                "second_pass": {
                    "attempted": False,
                    "status": "NOT_RUN_OPENCV_FAILED",
                },
            }

        if not visual["reinspection_required"]:
            return visual, {
                "decision": "RECORD_EVIDENCE_AND_REQUEST_HUMAN_REVIEW",
                "decision_reasons": visual["reinspection_reasons"],
                "second_pass": {
                    "attempted": False,
                    "status": "SKIPPED_BY_HEURISTIC",
                },
            }

        # This action is triggered by the OpenCV analysis above.
        # Run the same baseline detector on the expanded candidate crop.
        try:
            crop_prediction = self.detector.predict(visual["crop_path"])
            offset_x, offset_y = visual["crop_bbox_xyxy"][:2]

            second_pass_detections = []
            matching_detections = []

            for candidate in crop_prediction.get("detections", []):
                crop_box = candidate.get("bbox_xyxy", [])
                if len(crop_box) != 4:
                    continue

                mapped_box = [
                    round(float(crop_box[0]) + offset_x, 2),
                    round(float(crop_box[1]) + offset_y, 2),
                    round(float(crop_box[2]) + offset_x, 2),
                    round(float(crop_box[3]) + offset_y, 2),
                ]

                same_class = (
                    candidate.get("class_name") == detection.get("class_name")
                )
                overlap = self._box_iou(
                    detection["bbox_xyxy"], mapped_box
                )
                matched = same_class and overlap >= 0.10

                item = {
                    "class_name": candidate.get("class_name"),
                    "confidence": candidate.get("confidence"),
                    "bbox_xyxy_original_image": mapped_box,
                    "same_class_as_primary": same_class,
                    "iou_with_primary": round(overlap, 4),
                    "overlaps_primary_candidate": matched,
                }
                second_pass_detections.append(item)

                if matched:
                    matching_detections.append(item)

            if matching_detections:
                outcome = "SAME_CLASS_OVERLAPPING_DETECTION_FOUND"
            elif second_pass_detections:
                outcome = "NO_MATCHING_CANDIDATE_FOUND"
            else:
                outcome = "NO_DETECTIONS_IN_REINSPECTION"

            return visual, {
                "decision": "TARGETED_CROP_REINSPECTION",
                "decision_reasons": visual["reinspection_reasons"],
                "second_pass": {
                    "attempted": True,
                    "status": "COMPLETED",
                    "outcome": outcome,
                    "image_size": crop_prediction.get("image_size"),
                    "detection_count": len(second_pass_detections),
                    "same_class_overlap_count": len(matching_detections),
                    "detections": second_pass_detections,
                    "iou_match_threshold": 0.10,
                    "iou_threshold_is_heuristic": True,
                    "original_detection_preserved": True,
                },
            }

        except Exception as exc:
            return visual, {
                "decision": "TARGETED_CROP_REINSPECTION",
                "decision_reasons": visual["reinspection_reasons"],
                "second_pass": {
                    "attempted": True,
                    "status": "FAILED",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "human_review_required": True,
                },
            }

    def inspect_image(self, image_path, output_dir):
        image_path = Path(image_path).expanduser().resolve()
        output_dir = Path(output_dir).expanduser().resolve()

        if not image_path.is_file():
            raise FileNotFoundError(f"Image not found: {image_path}")

        output_dir.mkdir(parents=True, exist_ok=True)

        prediction = self.detector.predict(image_path)
        detections = prediction.get("detections", [])
        inspections = []

        for index, detection in enumerate(detections):
            inspection = build_inspection_record(
                detection, image_path, index
            )

            visual, action_trace = self._run_active_perception(
                image_path=image_path,
                detection=detection,
                output_dir=output_dir,
                index=index,
            )

            inspection["opencv_analysis"] = visual
            inspection["agentic_action_trace"] = action_trace
            inspections.append(inspection)

        inspections.sort(
            key=lambda item: item["triage_score"], reverse=True
        )

        second_passes = [
            item["agentic_action_trace"]["second_pass"]
            for item in inspections
        ]

        report = {
            "schema_version": "1.1",
            "workflow": "roadsentry_active_inspection",
            "source": str(image_path),
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": "COMPLETED_REQUIRES_HUMAN_REVIEW",
            "inspections": inspections,
            "tracks": [],
            "metadata": {
                "image_width": prediction.get("image_width"),
                "image_height": prediction.get("image_height"),
                "detection_count": len(detections),
                "model_weights": prediction.get("weights"),
                "confidence_threshold": prediction.get("confidence_threshold"),
                "image_size": prediction.get("image_size"),
                "device": prediction.get("device"),
                "opencv_version": cv2.__version__,
                "opencv_analysis_count": sum(
                    item.get("opencv_analysis", {}).get("analysis_status")
                    != "FAILED"
                    for item in inspections
                ),
                "targeted_reinspection_attempts": sum(
                    item.get("attempted", False) for item in second_passes
                ),
                "targeted_reinspection_failures": sum(
                    item.get("status") == "FAILED" for item in second_passes
                ),
                "triage_is_not_severity": True,
                "second_pass_does_not_replace_primary_detection": True,
                "human_review_required": True,
            },
        }

        json_path = write_json_report(
            report, output_dir / "inspection_report.json"
        )
        html_path = write_html_report(
            report, output_dir / "inspection_report.html"
        )

        return {
            "report": report,
            "json_report": str(json_path),
            "html_report": str(html_path),
        }

    @staticmethod
    def group_video_frames(frame_results):
        return group_frame_detections(frame_results)
