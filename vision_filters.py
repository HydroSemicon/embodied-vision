"""Pure filters used before detections enter DeepSORT."""

from __future__ import annotations


def is_plausible_person_detection(
    *,
    width: float,
    height: float,
    confidence: float,
    frame_width: int,
    frame_height: int,
    minimum_confidence: float,
    minimum_width: int,
    minimum_height: int,
    minimum_area_ratio: float,
) -> bool:
    """Reject tiny/weak person detections that commonly come from background objects."""
    if confidence < minimum_confidence:
        return False
    if width < minimum_width or height < minimum_height:
        return False
    frame_area = max(1.0, float(frame_width * frame_height))
    return (width * height) / frame_area >= minimum_area_ratio
