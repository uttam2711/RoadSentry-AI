"""Run RoadSentry's explainable inspection workflow on one image."""

import argparse
import json
import os
from pathlib import Path

from roadsentry.agent.orchestrator import InspectionOrchestrator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Input image path")
    parser.add_argument(
        "--output-dir", default="outputs/demo/agent_inspection",
        help="Directory for JSON and HTML reports",
    )
    parser.add_argument(
        "--weights", default=None,
        help="Optional path to detector checkpoint",
    )
    args = parser.parse_args()

    if args.weights:
        os.environ["ROADSENTRY_WEIGHTS"] = str(Path(args.weights).resolve())

    workflow = InspectionOrchestrator()
    result = workflow.inspect_image(args.image, args.output_dir)
    print(json.dumps({
        "status": result["report"]["status"],
        "detections": result["report"]["metadata"]["detection_count"],
        "json_report": result["json_report"],
        "html_report": result["html_report"],
    }, indent=2))


if __name__ == "__main__":
    main()
