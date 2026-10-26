from pathlib import Path
import json

import torch
from ultralytics import YOLO


class RoadDamageDetector:
    """Road damage inference wrapper around a trained Ultralytics model."""

    def __init__(
        self,
        weights,
        confidence=0.25,
        image_size=640,
        device=None,
    ):
        self.weights = Path(weights).expanduser().resolve()
        if not self.weights.is_file():
            raise FileNotFoundError(
                f"Model weights not found: {self.weights}"
            )

        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")
        if image_size < 32:
            raise ValueError("image_size must be at least 32")

        self.confidence = float(confidence)
        self.image_size = int(image_size)
        self.device = (
            device
            if device is not None
            else (0 if torch.cuda.is_available() else "cpu")
        )
        self.model = YOLO(str(self.weights))

    def predict(self, source):
        """Run inference on an image path and return structured detections."""
        source = Path(source).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(f"Input image not found: {source}")

        results = self.model.predict(
            source=str(source),
            imgsz=self.image_size,
            conf=self.confidence,
            device=self.device,
            verbose=False,
            save=False,
        )

        result = results[0]
        detections = []

        if result.boxes is not None:
            boxes = result.boxes
            coordinates = boxes.xyxy.cpu().tolist()
            class_ids = boxes.cls.cpu().tolist()
            confidences = boxes.conf.cpu().tolist()

            for xyxy, class_id, score in zip(
                coordinates, class_ids, confidences
            ):
                class_id = int(class_id)
                detections.append({
                    "class_id": class_id,
                    "class_name": self.model.names[class_id],
                    "confidence": round(float(score), 6),
                    "bbox_xyxy": [round(float(v), 2) for v in xyxy],
                })

        return {
            "image": str(source),
            "image_width": int(result.orig_shape[1]),
            "image_height": int(result.orig_shape[0]),
            "weights": str(self.weights),
            "confidence_threshold": self.confidence,
            "image_size": self.image_size,
            "device": str(self.device),
            "detection_count": len(detections),
            "detections": detections,
        }

    def annotate(self, source, output_dir):
        """Save a visual prediction and JSON detection record."""
        source = Path(source).expanduser().resolve()
        output_dir = Path(output_dir).expanduser().resolve()

        record = self.predict(source)
        output_dir.mkdir(parents=True, exist_ok=True)

        results = self.model.predict(
            source=str(source),
            imgsz=self.image_size,
            conf=self.confidence,
            device=self.device,
            verbose=False,
            save=False,
        )
        annotated_path = output_dir / f"{source.stem}_annotated.jpg"
        if not results[0].save(filename=str(annotated_path)):
            raise RuntimeError("Ultralytics did not save the annotated image")

        record["annotated_image"] = str(annotated_path)
        json_path = output_dir / f"{source.stem}_detections.json"
        json_path.write_text(json.dumps(record, indent=2) + "\n")

        return record
