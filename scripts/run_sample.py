from pathlib import Path
import argparse
import json

from roadsentry.vision import RoadDamageDetector


def main():
    root = Path(__file__).resolve().parents[1]
    default_weights = (
        root / "runs/roadsentry/baseline_yolo26n_9class/weights/best.pt"
    )

    parser = argparse.ArgumentParser(
        description="Run RoadSentry road-damage inference on one image."
    )
    parser.add_argument("--source", required=True, help="Path to an input image")
    parser.add_argument(
        "--weights", default=str(default_weights),
        help="Model checkpoint path"
    )
    parser.add_argument(
        "--output", default=str(root / "outputs/demo/image_inference"),
        help="Directory for annotated image and JSON results"
    )
    parser.add_argument(
        "--conf", type=float, default=0.25,
        help="Detection confidence threshold (0 to 1)"
    )
    parser.add_argument(
        "--imgsz", type=int, default=640,
        help="Inference image size"
    )
    parser.add_argument(
        "--device", default=None,
        help="Device such as 0 or cpu; default automatically selects CUDA"
    )
    args = parser.parse_args()

    detector = RoadDamageDetector(
        weights=args.weights,
        confidence=args.conf,
        image_size=args.imgsz,
        device=args.device,
    )
    record = detector.annotate(args.source, args.output)

    print("\n=== ROADSENTRY INFERENCE ===")
    print("Image:", record["image"])
    print("Image size:", f'{record["image_width"]}x{record["image_height"]}')
    print("Model:", record["weights"])
    print("Device:", record["device"])
    print("Detections:", record["detection_count"])

    for item in record["detections"]:
        print(
            f'  {item["class_name"]}: '
            f'{item["confidence"]:.3f} '
            f'bbox={item["bbox_xyxy"]}'
        )

    if not record["detections"]:
        print("  No detections exceeded the confidence threshold.")

    print("Annotated image:", record["annotated_image"])
    print(
        "JSON report:",
        str(Path(args.output).expanduser().resolve()
            / f'{Path(args.source).stem}_detections.json')
    )


if __name__ == "__main__":
    main()
