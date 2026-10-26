"""Lightweight temporal grouping for detections from sampled video frames."""

from collections import defaultdict


def group_frame_detections(frame_results, max_frame_gap=3):
    """Group same-class detections across nearby frames.

    frame_results: iterable of {"frame_index": int, "detections": list}.
    This simple grouping is a review aid, not a validated object tracker.
    """
    observations = []
    for frame in frame_results:
        frame_index = int(frame["frame_index"])
        for detection in frame.get("detections", []):
            item = dict(detection)
            item["frame_index"] = frame_index
            observations.append(item)

    observations.sort(key=lambda item: (item.get("class_name", ""), item["frame_index"]))
    groups = []
    by_class = defaultdict(list)

    for item in observations:
        by_class[str(item.get("class_name", "UNKNOWN"))].append(item)

    for class_name, items in by_class.items():
        current = []
        last_frame = None
        for item in items:
            frame_index = item["frame_index"]
            if current and frame_index - last_frame > max_frame_gap:
                groups.append(_summarize_group(class_name, current))
                current = []
            current.append(item)
            last_frame = frame_index
        if current:
            groups.append(_summarize_group(class_name, current))

    groups.sort(key=lambda group: group["start_frame"])
    for index, group in enumerate(groups, 1):
        group["track_id"] = f"TRK-{index:04d}"
        group["status"] = "UNVERIFIED_REVIEW_REQUIRED"
    return groups


def _summarize_group(class_name, observations):
    confidences = [float(item.get("confidence", 0.0)) for item in observations]
    return {
        "class_name": class_name,
        "start_frame": min(item["frame_index"] for item in observations),
        "end_frame": max(item["frame_index"] for item in observations),
        "observation_count": len(observations),
        "max_confidence": round(max(confidences, default=0.0), 4),
        "mean_confidence": round(
            sum(confidences) / len(confidences), 4
        ) if confidences else 0.0,
        "observations": observations,
        "method": "same-class temporal grouping; not identity tracking",
    }
