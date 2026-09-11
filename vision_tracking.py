"""Small adapters around Ultralytics BoT-SORT tracking results."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass


@dataclass(frozen=True)
class TrackObservation:
    track_id: str
    bounds: tuple[float, float, float, float]
    confidence: float
    class_id: int


def observations_from_result(result) -> list[TrackObservation]:
    """Convert an Ultralytics tracked ``Results`` object to plain values."""
    boxes = getattr(result, "boxes", None)
    if boxes is None or not bool(getattr(boxes, "is_track", False)):
        return []
    identifiers = getattr(boxes, "id", None)
    if identifiers is None:
        return []

    xyxy = boxes.xyxy.detach().cpu().tolist()
    confidences = boxes.conf.detach().cpu().tolist()
    classes = boxes.cls.detach().cpu().tolist()
    track_ids = identifiers.detach().cpu().tolist()
    return [
        TrackObservation(
            track_id=str(int(track_id)),
            bounds=tuple(float(value) for value in bounds),
            confidence=float(confidence),
            class_id=int(class_id),
        )
        for bounds, confidence, class_id, track_id in zip(
            xyxy,
            confidences,
            classes,
            track_ids,
            strict=True,
        )
    ]


class TrackLifecycle:
    """Keep logical tracks alive briefly while BoT-SORT bridges occlusions."""

    def __init__(self, max_missing_frames: int) -> None:
        self.max_missing_frames = max(0, int(max_missing_frames))
        self.frame_index = 0
        self.last_seen: dict[str, int] = {}
        self.active_ids: set[str] = set()

    def update(self, observed_ids: Iterable[str]) -> tuple[set[str], set[str]]:
        self.frame_index += 1
        for track_id in observed_ids:
            self.last_seen[str(track_id)] = self.frame_index

        next_active = {
            track_id
            for track_id, last_seen in self.last_seen.items()
            if self.frame_index - last_seen <= self.max_missing_frames
        }
        disappeared = self.active_ids - next_active
        for track_id in disappeared:
            self.last_seen.pop(track_id, None)
        self.active_ids = next_active
        return set(next_active), disappeared

    def staleness(self, track_id: str) -> int:
        last_seen = self.last_seen.get(str(track_id))
        if last_seen is None:
            return self.max_missing_frames + 1
        return max(0, self.frame_index - last_seen)
