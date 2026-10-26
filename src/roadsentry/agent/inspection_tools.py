"""Explainable triage helpers for road-damage inspection candidates."""

from datetime import datetime, timezone


DAMAGE_GUIDANCE = {
    "D00": "Longitudinal crack candidate",
    "D10": "Transverse crack candidate",
    "D20": "Alligator crack candidate",
    "D40": "Pothole candidate",
    "D01": "Longitudinal-cra​​ck-related candidate",
    "D11": "Transverse-crack-related candidate",
    "D43": "Crossing or related damage candidate",
    "D44": "Other damage candidate",
    "D50": "Other road-surface damage candidate",
}


def score_detection(detection):
    """Return a transparent triage score; this is not a safety assessment."""
    class_name = str(detection.get("class_name", "UNKNOWN"))
    confidence = float(detection.get("confidence", 0.0))
    confidence = max(0.0, min(1.0, confidence))

    # These are configurable prototype triage weights, not learned severity.
    class_weight = {
        "D40": 1.00,
        "D20": 0.85,
        "D00": 0.65,
        "D10": 0.65,
        "D01": 0.65,
        "D11": 0.65,
        "D43": 0.70,
        "D44": 0.70,
        "D50": 0.70,
    }.get(class_name, 0.50)

    score = round(100.0 * confidence * class_weight, 2)
    if score >= 60:
        priority = "HIGH"
    elif score >= 35:
        priority = "MEDIUM"
    else:
        priority = "LOW"

    return {
        "triage_score": score,
        "priority": priority,
        "class_guidance": DAMAGE_GUIDANCE.get(
            class_name, "Unmapped damage class; manual inspection needed"
        ),
        "reason": (
            f"Confidence {confidence:.3f} × class triage weight "
            f"{class_weight:.2f} = {score:.2f}/100. "
            "This score ranks review candidates; it does not establish "
            "physical severity or road safety."
        ),
        "human_review_required": True,
    }


def build_inspection_record(detection, source, index=0):
    """Attach stable-in-result ID, source evidence, and triage explanation."""
    result = dict(detection)
    result.update(score_detection(detection))
    result["inspection_id"] = f"INS-{index + 1:04d}"
    result["source"] = str(source)
    result["created_at_utc"] = datetime.now(timezone.utc).isoformat()
    result["review_status"] = "PENDING_HUMAN_REVIEW"
    return result
