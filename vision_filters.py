"""Pure filters used around person tracking and identity assignment."""

from __future__ import annotations

from typing import Iterable, Optional


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


def resolve_duplicate_identity_track(
    *,
    candidate_track_id: str,
    existing_tracks: Iterable[tuple[str, int]],
) -> Optional[tuple[str, str]]:
    """Return ``(kept, dropped)`` for duplicate tracks of one person.

    A currently updated owner wins. If every existing owner is stale, the new
    candidate wins so the visible box follows the person's current location.
    """
    owners = list(existing_tracks)
    if not owners:
        return None
    owner_track_id, owner_staleness = min(
        owners,
        key=lambda item: (max(0, int(item[1])), item[0]),
    )
    if owner_staleness == 0:
        return owner_track_id, candidate_track_id
    return candidate_track_id, owner_track_id
