from __future__ import annotations

import os
import threading
import time
import uuid
from collections import defaultdict, deque
from datetime import UTC, datetime
from pathlib import Path
from queue import Empty, Full, Queue

import cv2  # type: ignore
import requests  # type: ignore
import yaml  # type: ignore
from flask import Flask, Response, jsonify, render_template, request  # type: ignore
from ultralytics import YOLO  # type: ignore

from face_registry import (
    FaceEncoding,
    FaceMatch,
    FaceRegistry,
    InsightFaceEncoder,
    resolve_face_vote,
    select_representative_encodings,
)
from vision_filters import (
    is_plausible_person_detection,
    resolve_duplicate_identity_track,
)
from vision_tracking import TrackLifecycle, observations_from_result

BASE_DIRECTORY = Path(__file__).resolve().parent


def env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


EVENT_SEND_ENABLED = env_bool("VISION_EVENT_SEND_ENABLED", True)
DISPLAY_ENABLED = env_bool("VISION_DISPLAY_ENABLED", True)
FACE_RECOGNITION_ENABLED = env_bool("FACE_RECOGNITION_ENABLED", True)
FACE_EVENT_SEND_DEFAULT_ENABLED = env_bool("FACE_EVENT_SEND_ENABLED", True)
FACE_REQUIRE_GPU = env_bool("FACE_REQUIRE_GPU", True)

EVENT_URL = os.getenv("VISION_EVENT_URL", "http://localhost:3000/yolo_event")
EVENT_REQUEST_TIMEOUT_SECONDS = float(os.getenv("VISION_EVENT_REQUEST_TIMEOUT_SECONDS", "10.0"))
HTTP_HOST = os.getenv("VISION_HTTP_HOST", "127.0.0.1")
HTTP_PORT = int(os.getenv("VISION_HTTP_PORT", "5000"))
CAMERA_INDEX = int(os.getenv("VISION_CAMERA_INDEX", "0"))
YOLO_MODEL_PATH = os.getenv("VISION_YOLO_MODEL", str(BASE_DIRECTORY / "yolov8n.pt"))
YOLO_CONFIDENCE_THRESHOLD = max(
    0.0,
    min(1.0, float(os.getenv("VISION_YOLO_CONFIDENCE", "0.50"))),
)
YOLO_MIN_PERSON_WIDTH = int(os.getenv("VISION_YOLO_MIN_PERSON_WIDTH", "40"))
YOLO_MIN_PERSON_HEIGHT = int(os.getenv("VISION_YOLO_MIN_PERSON_HEIGHT", "80"))
YOLO_MIN_PERSON_AREA_RATIO = float(
    os.getenv("VISION_YOLO_MIN_PERSON_AREA_RATIO", "0.002")
)
YOLO_TRACK_CONFIDENCE_THRESHOLD = max(
    0.0,
    min(
        YOLO_CONFIDENCE_THRESHOLD,
        float(os.getenv("VISION_YOLO_TRACK_CONFIDENCE", "0.10")),
    ),
)

FACE_MODEL_NAME = os.getenv("FACE_MODEL_NAME", "buffalo_l")
FACE_DETECTOR_BACKEND = "SCRFD"
FACE_MODEL_DEFAULT_THRESHOLDS = {
    "buffalo_l": 0.55,
    "antelopev2": 0.55,
    "buffalo_m": 0.55,
    "buffalo_s": 0.55,
}
FACE_MATCH_THRESHOLD = float(
    os.getenv(
        "FACE_MATCH_THRESHOLD",
        str(FACE_MODEL_DEFAULT_THRESHOLDS.get(FACE_MODEL_NAME, 0.40)),
    )
)
FACE_ANALYSIS_INTERVAL_SECONDS = float(os.getenv("FACE_ANALYSIS_INTERVAL_SECONDS", "0.25"))
FACE_CONFIRMATIONS = int(os.getenv("FACE_CONFIRMATIONS", "3"))
FACE_RECOGNITION_WINDOW = int(os.getenv("FACE_RECOGNITION_WINDOW", "8"))
FACE_RECOGNITION_MIN_VOTE_RATIO = float(
    os.getenv("FACE_RECOGNITION_MIN_VOTE_RATIO", "0.6")
)
FACE_CONSECUTIVE_CONFIRMATIONS = int(
    os.getenv("FACE_CONSECUTIVE_CONFIRMATIONS", "2")
)
FACE_UNKNOWN_CONFIRMATIONS = int(os.getenv("FACE_UNKNOWN_CONFIRMATIONS", "40"))
FACE_UNKNOWN_MIN_SECONDS = float(os.getenv("FACE_UNKNOWN_MIN_SECONDS", "15.0"))
FACE_MIN_PERSON_WIDTH = int(os.getenv("FACE_MIN_PERSON_WIDTH", "100"))
FACE_MIN_PERSON_HEIGHT = int(os.getenv("FACE_MIN_PERSON_HEIGHT", "140"))
FACE_MIN_FACE_SIZE = int(os.getenv("FACE_MIN_FACE_SIZE", "70"))
FACE_MIN_DETECTION_CONFIDENCE = float(
    os.getenv("FACE_MIN_DETECTION_CONFIDENCE", "0.70")
)
FACE_MIN_BLUR_VARIANCE = float(os.getenv("FACE_MIN_BLUR_VARIANCE", "35.0"))
FACE_MIN_BRIGHTNESS = float(os.getenv("FACE_MIN_BRIGHTNESS", "35.0"))
FACE_MAX_BRIGHTNESS = float(os.getenv("FACE_MAX_BRIGHTNESS", "225.0"))
FACE_INPUT_MAX_DIMENSION = int(os.getenv("FACE_INPUT_MAX_DIMENSION", "640"))
FACE_DETECTION_SIZE = int(os.getenv("FACE_DETECTION_SIZE", "640"))
FACE_MODEL_ROOT = os.getenv("FACE_MODEL_ROOT") or None
FACE_EXECUTION_PROVIDERS = tuple(
    provider.strip()
    for provider in os.getenv(
        "FACE_EXECUTION_PROVIDERS",
        "CUDAExecutionProvider,CPUExecutionProvider",
    ).split(",")
    if provider.strip()
)
FACE_PERSON_CROP_TOP_RATIO = max(
    0.25,
    min(1.0, float(os.getenv("FACE_PERSON_CROP_TOP_RATIO", "0.75"))),
)
FACE_OBSERVATION_BUFFER_SIZE = int(os.getenv("FACE_OBSERVATION_BUFFER_SIZE", "64"))
FACE_ENROLLMENT_SECONDS = float(os.getenv("FACE_ENROLLMENT_SECONDS", "30.0"))
FACE_ENROLLMENT_MIN_SECONDS = float(os.getenv("FACE_ENROLLMENT_MIN_SECONDS", "8.0"))
FACE_ENROLLMENT_MIN_SAMPLES = int(os.getenv("FACE_ENROLLMENT_MIN_SAMPLES", "4"))
FACE_ENROLLMENT_TARGET_SAMPLES = int(os.getenv("FACE_ENROLLMENT_TARGET_SAMPLES", "8"))
FACE_ENROLLMENT_EXEMPLARS = int(os.getenv("FACE_ENROLLMENT_EXEMPLARS", "8"))
FACE_ENROLLMENT_CLUSTER_DISTANCE = float(
    os.getenv(
        "FACE_ENROLLMENT_CLUSTER_DISTANCE",
        str(
            min(
                0.65,
                max(FACE_MATCH_THRESHOLD + 0.10, FACE_MATCH_THRESHOLD * 1.5),
            )
        ),
    )
)
FACE_MAX_EMBEDDINGS_PER_PERSON = int(
    os.getenv("FACE_MAX_EMBEDDINGS_PER_PERSON", "12")
)
TRACK_MAX_AGE = max(1, int(os.getenv("TRACK_MAX_AGE", "90")))
TRACKER_CONFIG_PATH = os.getenv(
    "VISION_TRACKER_CONFIG",
    str(BASE_DIRECTORY / "botsort.yaml"),
)
UI_STREAM_FPS = max(1.0, min(120.0, float(os.getenv("VISION_UI_STREAM_FPS", "30"))))
UI_JPEG_QUALITY = max(50, min(100, int(os.getenv("VISION_UI_JPEG_QUALITY", "90"))))
DEFAULT_FACE_REGISTRY_PATH = (
    BASE_DIRECTORY / "data" / f"face_registry_{FACE_MODEL_NAME.lower()}.json"
)
FACE_REGISTRY_PATH = Path(
    os.getenv(
        "FACE_REGISTRY_PATH",
        str(DEFAULT_FACE_REGISTRY_PATH),
    )
)


def prepare_tracker_config() -> str:
    """Create an effective BoT-SORT config using the runtime thresholds."""
    source = Path(TRACKER_CONFIG_PATH)
    if not source.exists():
        raise FileNotFoundError(f"tracker config does not exist: {source}")
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("tracker_type") != "botsort":
        raise ValueError("VISION_TRACKER_CONFIG must contain a BoT-SORT YAML config")
    payload["track_high_thresh"] = YOLO_CONFIDENCE_THRESHOLD
    payload["new_track_thresh"] = YOLO_CONFIDENCE_THRESHOLD
    payload["track_low_thresh"] = YOLO_TRACK_CONFIDENCE_THRESHOLD
    payload["track_buffer"] = TRACK_MAX_AGE

    runtime_path = BASE_DIRECTORY / "data" / "botsort_runtime.yaml"
    runtime_path.parent.mkdir(parents=True, exist_ok=True)
    runtime_path.write_text(
        yaml.safe_dump(payload, sort_keys=False),
        encoding="utf-8",
    )
    return str(runtime_path)

app = Flask(__name__)
frame_lock = threading.Lock()
frame_condition = threading.Condition(frame_lock)
latest_frame = None
latest_annotated_frame = None
latest_frame_sequence = 0
event_queue: Queue[dict] = Queue(maxsize=100)
face_event_state_lock = threading.RLock()
face_event_send_enabled = FACE_EVENT_SEND_DEFAULT_ENABLED


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def face_events_enabled() -> bool:
    with face_event_state_lock:
        return EVENT_SEND_ENABLED and face_event_send_enabled


def set_face_events_enabled(enabled: bool) -> dict:
    global face_event_send_enabled

    with face_event_state_lock:
        face_event_send_enabled = enabled

    dropped = 0
    if not enabled:
        while True:
            try:
                event_queue.get_nowait()
            except Empty:
                break
            event_queue.task_done()
            dropped += 1

    state = {
        "enabled": face_event_send_enabled,
        "effective_enabled": face_events_enabled(),
        "master_event_delivery_enabled": EVENT_SEND_ENABLED,
        "queued_events_dropped": dropped,
    }
    print("face event delivery changed:", state)
    return state


def pending_identity(status: str = "pending") -> dict:
    return {
        "status": status,
        "person_id": None,
        "name": None,
        "distance": None,
        "threshold": FACE_MATCH_THRESHOLD,
    }


def event_message(event_type: str, identity: dict) -> str:
    name = identity.get("name")
    messages = {
        "person_unknown": "A visible face is not registered.",
        "person_disappeared": "A tracked person has disappeared.",
        "person_enrolled": f"The visible person was registered as {name}.",
        "person_recognized": f"The visible registered person is {name}.",
    }
    return messages[event_type]


def send_person_event(
    event_type: str,
    track_id: str,
    *,
    position: str | None = None,
    identity: dict | None = None,
) -> bool:
    identity_payload = identity or pending_identity(
        "pending" if FACE_RECOGNITION_ENABLED else "unavailable"
    )
    event = {
        "event_id": uuid.uuid4().hex,
        "source": "deepsort",
        "type": event_type,
        "track_id": str(track_id),
        "timestamp": utc_now(),
        "identity": identity_payload,
        "message": event_message(event_type, identity_payload),
    }
    if position is not None:
        event["position"] = position

    payload = {"event": event}
    if not face_events_enabled():
        print(
            "face event skipped: disabled",
            {"type": event_type, "track_id": str(track_id)},
        )
        return False

    try:
        event_queue.put_nowait(payload)
        return True
    except Full:
        print("vision event dropped: delivery queue is full", payload)
        return False


def run_event_sender() -> None:
    retry_delays = (0.0, 0.2, 0.5, 1.0)
    while True:
        payload = event_queue.get()
        if not face_events_enabled():
            print(
                "queued face event dropped: disabled",
                payload.get("event", {}).get("event_id"),
            )
            event_queue.task_done()
            continue

        delivered = False
        last_error = None
        for delay in retry_delays:
            if delay:
                time.sleep(delay)
            if not face_events_enabled():
                break
            try:
                response = requests.post(
                    EVENT_URL,
                    json=payload,
                    timeout=EVENT_REQUEST_TIMEOUT_SECONDS,
                )
                response.raise_for_status()
                print("vision event sent:", payload)
                delivered = True
                break
            except requests.RequestException as error:
                last_error = error
        if not delivered and face_events_enabled():
            print("vision event failed after retries:", last_error, payload)
        elif not delivered:
            print(
                "face event delivery stopped: disabled",
                payload.get("event", {}).get("event_id"),
            )
        event_queue.task_done()


threading.Thread(target=run_event_sender, daemon=True, name="vision-events").start()


class FaceRecognitionService:
    """Runs InsightFace on one background thread and binds identities to tracks."""

    def __init__(self, registry: FaceRegistry, enabled: bool) -> None:
        self.registry = registry
        self.enabled = enabled
        self.jobs: Queue[tuple[str, object, float]] = Queue(maxsize=2)
        self.lock = threading.RLock()
        self.sample_condition = threading.Condition(self.lock)
        self.worker_status = "disabled" if not enabled else "starting"
        self.worker_error: str | None = None
        self.last_submitted: dict[str, float] = {}
        self.latest_embeddings: dict[str, list[float]] = {}
        self.identities: dict[str, dict] = {}
        self.active_tracks: set[str] = set()
        self.track_staleness: dict[str, int] = {}
        self.superseded_tracks: set[str] = set()
        self.candidate_history: dict[str, deque[FaceMatch | None]] = defaultdict(
            lambda: deque(
                maxlen=max(
                    1,
                    FACE_RECOGNITION_WINDOW,
                    FACE_UNKNOWN_CONFIRMATIONS,
                )
            )
        )
        self.recent_encodings: dict[str, deque[FaceEncoding]] = defaultdict(
            lambda: deque(maxlen=max(1, FACE_OBSERVATION_BUFFER_SIZE))
        )
        self.first_quality_sample_at: dict[str, float] = {}
        self.last_face_quality: dict[str, dict] = {}
        self.last_face_rejections: dict[str, dict] = {}
        self.enrollment_sessions: dict[str, dict] = {}
        self.enrollment_results: dict[str, dict] = {}
        self.identity_event_tracks: set[str] = set()
        self.best_candidate_distances: dict[str, float] = {}
        self.latest_candidates: dict[str, dict] = {}
        self.best_candidates: dict[str, dict] = {}
        self.last_face_sample_at: dict[str, str] = {}
        self.last_failure_log: dict[str, float] = {}
        self.execution_providers: list[str] = []

    def start(self) -> None:
        if self.enabled:
            threading.Thread(target=self._run, daemon=True, name="face-recognition").start()
            threading.Thread(
                target=self._run_enrollment_monitor,
                daemon=True,
                name="face-enrollment",
            ).start()

    def _run(self) -> None:
        try:
            encoder = InsightFaceEncoder(
                model_name=FACE_MODEL_NAME,
                providers=FACE_EXECUTION_PROVIDERS,
                model_root=FACE_MODEL_ROOT,
                detection_size=FACE_DETECTION_SIZE,
                require_accelerator=FACE_REQUIRE_GPU,
                min_face_size=FACE_MIN_FACE_SIZE,
                min_detection_confidence=FACE_MIN_DETECTION_CONFIDENCE,
                min_blur_variance=FACE_MIN_BLUR_VARIANCE,
                min_brightness=FACE_MIN_BRIGHTNESS,
                max_brightness=FACE_MAX_BRIGHTNESS,
                max_input_dimension=FACE_INPUT_MAX_DIMENSION,
            )
        except Exception as error:
            with self.lock:
                self.worker_status = "error"
                self.worker_error = str(error)
            print("face recognition unavailable:", error)
            return

        with self.lock:
            self.worker_status = "ready"
            self.execution_providers = list(encoder.providers)
        print(
            "face recognition ready:",
            {
                "model": FACE_MODEL_NAME,
                "detector": FACE_DETECTOR_BACKEND,
                "providers": encoder.providers,
            },
        )

        while True:
            try:
                track_id, person_crop, captured_at = self.jobs.get(timeout=1.0)
            except Empty:
                continue

            try:
                encoding = encoder.encode(person_crop)
                self._process_encoding(track_id, encoding, captured_at)
            except Exception as error:
                now = time.monotonic()
                with self.lock:
                    self.last_face_rejections[track_id] = {
                        "reason": str(error),
                        "timestamp": utc_now(),
                    }
                    last_log = self.last_failure_log.get(track_id, 0.0)
                    if now - last_log >= 10.0:
                        print(f"face not usable for track {track_id}:", error)
                        self.last_failure_log[track_id] = now
            finally:
                self.jobs.task_done()

    def _process_encoding(
        self,
        track_id: str,
        encoding: FaceEncoding,
        captured_at: float,
    ) -> None:
        nearest = self.registry.nearest(encoding.embedding)
        match = (
            nearest
            if nearest is not None and nearest.distance <= self.registry.threshold
            else None
        )
        event_to_send = None

        with self.lock:
            if track_id not in self.active_tracks:
                return
            self.latest_embeddings[track_id] = list(encoding.embedding)
            enrollment = self.enrollment_sessions.get(track_id)
            if not enrollment or captured_at >= enrollment["started_at"]:
                self.recent_encodings[track_id].append(encoding)
            self.first_quality_sample_at.setdefault(track_id, time.monotonic())
            self.last_face_sample_at[track_id] = utc_now()
            self.last_face_quality[track_id] = encoding.public_quality()
            self.last_face_rejections.pop(track_id, None)
            if nearest is None:
                self.latest_candidates.pop(track_id, None)
            else:
                candidate = {
                    "person_id": nearest.person_id,
                    "name": nearest.name,
                    "distance": round(nearest.distance, 4),
                    "threshold": nearest.threshold,
                    "within_threshold": nearest.distance <= nearest.threshold,
                }
                self.latest_candidates[track_id] = candidate

            if nearest is not None:
                previous_best = self.best_candidate_distances.get(track_id)
                if previous_best is None or nearest.distance < previous_best:
                    self.best_candidate_distances[track_id] = nearest.distance
                    self.best_candidates[track_id] = candidate.copy()
                    print(
                        f"face candidate for track {track_id}: "
                        f"{nearest.name} distance={nearest.distance:.4f} "
                        f"threshold={self.registry.threshold:.4f}"
                    )

            history = self.candidate_history[track_id]
            history.append(match)
            self.sample_condition.notify_all()

            # Enrollment owns the identity decision while it is collecting.
            # This prevents an "unknown" event from racing the later
            # "person_enrolled" event for the same visible person.
            if enrollment:
                return

            existing = self.identities.get(track_id)
            if existing and existing.get("status") == "recognized":
                return

            vote = resolve_face_vote(
                history,
                window_size=FACE_RECOGNITION_WINDOW,
                min_votes=FACE_CONFIRMATIONS,
                min_vote_ratio=FACE_RECOGNITION_MIN_VOTE_RATIO,
                consecutive_matches=FACE_CONSECUTIVE_CONFIRMATIONS,
            )
            if vote:
                identity = {
                    "status": "recognized",
                    "person_id": vote.person_id,
                    "name": vote.name,
                    "distance": round(vote.distance, 4),
                    "threshold": vote.threshold,
                }
                identity_owners = [
                    owner_track_id
                    for owner_track_id, owner_identity in self.identities.items()
                    if (
                        owner_track_id != track_id
                        and owner_track_id in self.active_tracks
                        and owner_track_id not in self.superseded_tracks
                        and owner_identity.get("status") == "recognized"
                        and owner_identity.get("person_id") == vote.person_id
                    )
                ]
                duplicate_resolution = resolve_duplicate_identity_track(
                    candidate_track_id=track_id,
                    existing_tracks=(
                        (
                            owner_track_id,
                            self.track_staleness.get(owner_track_id, 0),
                        )
                        for owner_track_id in identity_owners
                    ),
                )

                if duplicate_resolution is None:
                    self.identities[track_id] = identity
                    event_to_send = ("person_recognized", identity.copy())
                elif duplicate_resolution[0] != track_id:
                    # Both boxes are current detections. Keep the identity owner
                    # that was already visible and retire the duplicate track.
                    owner_track_id = duplicate_resolution[0]
                    self.identities[track_id] = identity
                    self.superseded_tracks.add(track_id)
                    print(
                        f"duplicate face track suppressed: {vote.name} "
                        f"kept={owner_track_id}, dropped={track_id}"
                    )
                else:
                    # The old owner is only a Kalman prediction. Move the
                    # logical person to the fresh track without emitting a
                    # disappearance/reappearance pair.
                    owner_track_id = duplicate_resolution[1]
                    self.identities[track_id] = identity
                    self.superseded_tracks.add(owner_track_id)
                    if owner_track_id in self.identity_event_tracks:
                        self.identity_event_tracks.discard(owner_track_id)
                        self.identity_event_tracks.add(track_id)
                    print(
                        f"face identity track transferred: {vote.name} "
                        f"old={owner_track_id}, new={track_id}"
                    )

            if event_to_send is None:
                required = max(1, FACE_UNKNOWN_CONFIRMATIONS)
                first_sample_at = self.first_quality_sample_at[track_id]
                enough_time = (
                    time.monotonic() - first_sample_at >= FACE_UNKNOWN_MIN_SECONDS
                )
                unknown_evidence = list(history)[-required:]
                consistently_unknown = (
                    len(unknown_evidence) >= required
                    and all(candidate is None for candidate in unknown_evidence)
                )
                if (
                    enough_time
                    and consistently_unknown
                    and not (existing and existing.get("status") == "unknown")
                ):
                    self.identities[track_id] = pending_identity("unknown")
                    event_to_send = (
                        "person_unknown",
                        self.identities[track_id].copy(),
                    )

        if event_to_send:
            queued = send_person_event(
                event_to_send[0],
                track_id,
                identity=event_to_send[1],
            )
            if queued:
                with self.lock:
                    if track_id in self.active_tracks:
                        self.identity_event_tracks.add(track_id)

    def submit(self, track_id: str, person_crop) -> bool:
        if not self.enabled:
            return False

        now = time.monotonic()
        with self.lock:
            self.active_tracks.add(track_id)
            if self.worker_status == "error":
                return False
            if (
                self.identities.get(track_id, {}).get("status") == "recognized"
                and track_id not in self.enrollment_sessions
            ):
                return False
            if now - self.last_submitted.get(track_id, 0.0) < FACE_ANALYSIS_INTERVAL_SECONDS:
                return False
            self.last_submitted[track_id] = now

        try:
            self.jobs.put_nowait((track_id, person_crop.copy(), now))
            return True
        except Full:
            try:
                self.jobs.get_nowait()
                self.jobs.task_done()
                self.jobs.put_nowait((track_id, person_crop.copy(), now))
                return True
            except (Empty, Full):
                return False

    def mark_active(self, track_id: str, time_since_update: int = 0) -> None:
        with self.lock:
            self.active_tracks.add(track_id)
            self.track_staleness[track_id] = max(0, int(time_since_update))

    def is_superseded(self, track_id: str) -> bool:
        with self.lock:
            return track_id in self.superseded_tracks

    def prune_superseded_tracks(self, tracker_track_ids: set[str]) -> None:
        with self.lock:
            self.superseded_tracks.intersection_update(tracker_track_ids)

    def identity_for(self, track_id: str) -> dict:
        with self.lock:
            identity = self.identities.get(track_id)
            if identity:
                return identity.copy()
            if not self.enabled or self.worker_status == "error":
                return pending_identity("unavailable")
            return pending_identity()

    def identity_event_announced(self, track_id: str) -> bool:
        with self.lock:
            return track_id in self.identity_event_tracks

    def enroll(self, track_id: str, name: str) -> dict:
        started_at = time.monotonic()
        deadline = started_at + FACE_ENROLLMENT_SECONDS

        with self.sample_condition:
            if not self.enabled or self.worker_status != "ready":
                raise LookupError("face recognition worker is not ready")
            if track_id not in self.active_tracks:
                raise LookupError("track is not active")
            if track_id in self.enrollment_sessions:
                raise LookupError("an enrollment session is already active for this track")

            self.recent_encodings[track_id].clear()
            self.enrollment_results.pop(track_id, None)
            self.enrollment_sessions[track_id] = {
                "name": name,
                "started_at": started_at,
                "deadline": deadline,
            }
            self.last_submitted[track_id] = 0.0
            print(
                f"face enrollment started for track {track_id}: "
                "look forward, then slowly turn left and right"
            )
            self.sample_condition.notify_all()
        return {
            "status": "collecting",
            "track_id": track_id,
            "name": name,
            "sample_count": 0,
            "target_samples": FACE_ENROLLMENT_TARGET_SAMPLES,
            "minimum_samples": FACE_ENROLLMENT_MIN_SAMPLES,
            "seconds_remaining": round(FACE_ENROLLMENT_SECONDS, 1),
        }

    def _run_enrollment_monitor(self) -> None:
        while True:
            ready_sessions: list[tuple[str, str, list[FaceEncoding]]] = []
            with self.sample_condition:
                now = time.monotonic()
                for track_id, session in list(self.enrollment_sessions.items()):
                    samples = list(self.recent_encodings.get(track_id, ()))
                    elapsed = now - session["started_at"]
                    target_reached = (
                        elapsed >= FACE_ENROLLMENT_MIN_SECONDS
                        and len(samples) >= FACE_ENROLLMENT_TARGET_SAMPLES
                    )
                    if target_reached or now >= session["deadline"]:
                        self.enrollment_sessions.pop(track_id, None)
                        ready_sessions.append((track_id, session["name"], samples))

                if not ready_sessions:
                    self.sample_condition.wait(timeout=0.25)
                    continue

            for track_id, name, samples in ready_sessions:
                try:
                    self._finalize_enrollment(track_id, name, samples)
                except Exception as error:
                    result = {
                        "status": "failed",
                        "name": name,
                        "message": f"failed to save enrollment: {error}",
                        "captured_sample_count": len(samples),
                        "selected_sample_count": 0,
                        "completed_at": utc_now(),
                    }
                    with self.lock:
                        if track_id in self.active_tracks:
                            self.enrollment_results[track_id] = result
                    print(f"face enrollment failed for track {track_id}:", error)

    def _finalize_enrollment(
        self,
        track_id: str,
        name: str,
        samples: list[FaceEncoding],
    ) -> None:
        selected = select_representative_encodings(
            samples,
            limit=max(1, FACE_ENROLLMENT_EXEMPLARS),
            cluster_distance=FACE_ENROLLMENT_CLUSTER_DISTANCE,
        )
        if len(selected) < FACE_ENROLLMENT_MIN_SAMPLES:
            message = (
                "not enough consistent high-quality face samples; "
                f"captured {len(samples)}, selected {len(selected)}, "
                f"need {FACE_ENROLLMENT_MIN_SAMPLES}"
            )
            result = {
                "status": "failed",
                "name": name,
                "message": message,
                "captured_sample_count": len(samples),
                "selected_sample_count": len(selected),
                "completed_at": utc_now(),
            }
            with self.lock:
                if track_id in self.active_tracks:
                    self.enrollment_results[track_id] = result
            print(f"face enrollment failed for track {track_id}: {message}")
            return

        enrolled = self.registry.enroll_many(
            name,
            [encoding.embedding for encoding in selected],
        )
        identity = {
            "status": "recognized",
            "person_id": enrolled["person_id"],
            "name": enrolled["name"],
            "distance": 0.0,
            "threshold": FACE_MATCH_THRESHOLD,
        }
        result = {
            "status": "completed",
            "name": enrolled["name"],
            "person_id": enrolled["person_id"],
            "captured_sample_count": len(samples),
            "selected_sample_count": len(selected),
            "embedding_count": enrolled["embedding_count"],
            "completed_at": utc_now(),
        }
        with self.lock:
            if track_id not in self.active_tracks:
                print(
                    f"face enrollment saved after track {track_id} disappeared: "
                    f"captured={len(samples)}, selected={len(selected)}"
                )
                return
            self.identities[track_id] = identity
            self.candidate_history[track_id].clear()
            self.enrollment_results[track_id] = result
            print(
                f"face enrollment completed for track {track_id}: "
                f"captured={len(samples)}, selected={len(selected)}, "
                f"stored={enrolled['embedding_count']}"
            )

        queued = send_person_event(
            "person_enrolled",
            track_id,
            identity=identity.copy(),
        )
        if queued:
            with self.lock:
                if track_id in self.active_tracks:
                    self.identity_event_tracks.add(track_id)

    def forget_track(self, track_id: str) -> None:
        with self.sample_condition:
            self.active_tracks.discard(track_id)
            self.track_staleness.pop(track_id, None)
            self.last_submitted.pop(track_id, None)
            self.latest_embeddings.pop(track_id, None)
            self.identities.pop(track_id, None)
            self.candidate_history.pop(track_id, None)
            self.recent_encodings.pop(track_id, None)
            self.first_quality_sample_at.pop(track_id, None)
            self.last_face_quality.pop(track_id, None)
            self.last_face_rejections.pop(track_id, None)
            self.enrollment_sessions.pop(track_id, None)
            self.enrollment_results.pop(track_id, None)
            self.identity_event_tracks.discard(track_id)
            self.best_candidate_distances.pop(track_id, None)
            self.latest_candidates.pop(track_id, None)
            self.best_candidates.pop(track_id, None)
            self.last_face_sample_at.pop(track_id, None)
            self.last_failure_log.pop(track_id, None)
            self.sample_condition.notify_all()

    def track_states(self) -> list[dict]:
        now = time.monotonic()
        with self.lock:
            states = []
            for track_id in sorted(self.active_tracks):
                enrollment = self.enrollment_sessions.get(track_id)
                enrollment_result = self.enrollment_results.get(track_id)
                enrollment_state = None
                if enrollment:
                    enrollment_state = {
                        "active": True,
                        "name": enrollment["name"],
                        "sample_count": len(self.recent_encodings.get(track_id, ())),
                        "target_samples": FACE_ENROLLMENT_TARGET_SAMPLES,
                        "seconds_remaining": round(
                            max(0.0, enrollment["deadline"] - now),
                            1,
                        ),
                    }
                states.append(
                    {
                        "track_id": track_id,
                        "identity": self.identity_for(track_id),
                        "face_sample_ready": track_id in self.latest_embeddings,
                        "last_face_sample_at": self.last_face_sample_at.get(track_id),
                        "face_quality": (
                            self.last_face_quality.get(track_id, {}).copy() or None
                        ),
                        "face_rejection": (
                            self.last_face_rejections.get(track_id, {}).copy() or None
                        ),
                        "latest_candidate": (
                            self.latest_candidates.get(track_id, {}).copy() or None
                        ),
                        "best_candidate": (
                            self.best_candidates.get(track_id, {}).copy() or None
                        ),
                        "evidence_samples": len(
                            self.candidate_history.get(track_id, ())
                        ),
                        "enrollment": enrollment_state,
                        "enrollment_result": (
                            enrollment_result.copy() if enrollment_result else None
                        ),
                    }
                )
            return states

    def status(self) -> dict:
        with self.lock:
            return {
                "enabled": self.enabled,
                "state": self.worker_status,
                "error": self.worker_error,
                "model": FACE_MODEL_NAME,
                "detector": FACE_DETECTOR_BACKEND,
                "execution_providers": list(self.execution_providers),
                "gpu_required": FACE_REQUIRE_GPU,
                "registered_people": len(self.registry.list_people()),
                "active_enrollments": len(self.enrollment_sessions),
                "suppressed_duplicate_tracks": len(self.superseded_tracks),
                "quality_filter": {
                    "min_face_size": FACE_MIN_FACE_SIZE,
                    "min_detection_confidence": FACE_MIN_DETECTION_CONFIDENCE,
                    "min_blur_variance": FACE_MIN_BLUR_VARIANCE,
                    "brightness_range": [
                        FACE_MIN_BRIGHTNESS,
                        FACE_MAX_BRIGHTNESS,
                    ],
                    "max_input_dimension": FACE_INPUT_MAX_DIMENSION,
                    "detection_size": FACE_DETECTION_SIZE,
                },
                "enrollment": {
                    "maximum_seconds": FACE_ENROLLMENT_SECONDS,
                    "minimum_seconds": FACE_ENROLLMENT_MIN_SECONDS,
                    "minimum_samples": FACE_ENROLLMENT_MIN_SAMPLES,
                    "target_samples": FACE_ENROLLMENT_TARGET_SAMPLES,
                },
            }


face_registry = FaceRegistry(
    FACE_REGISTRY_PATH,
    model_name=FACE_MODEL_NAME,
    threshold=FACE_MATCH_THRESHOLD,
    max_embeddings_per_person=FACE_MAX_EMBEDDINGS_PER_PERSON,
)
face_service = FaceRecognitionService(face_registry, FACE_RECOGNITION_ENABLED)


@app.route("/")
def index():
    return render_template("dashboard.html")


@app.route("/status")
def status():
    with frame_lock:
        frame_ready = latest_frame is not None
    response = {
        "camera": "ready" if frame_ready else "no_frame",
        "event_delivery": face_events_enabled(),
        "face_events": {
            "enabled": face_event_send_enabled,
            "effective_enabled": face_events_enabled(),
            "master_event_delivery_enabled": EVENT_SEND_ENABLED,
            "queued_events": event_queue.qsize(),
        },
        "face_recognition": face_service.status(),
        "ui_stream": {
            "fps_limit": UI_STREAM_FPS,
            "jpeg_quality": UI_JPEG_QUALITY,
        },
    }
    return jsonify(response), 200 if frame_ready else 503


@app.route("/face-events", methods=["GET", "POST"])
def face_event_switch():
    if request.method == "GET":
        return jsonify(
            {
                "enabled": face_event_send_enabled,
                "effective_enabled": face_events_enabled(),
                "master_event_delivery_enabled": EVENT_SEND_ENABLED,
                "queued_events": event_queue.qsize(),
            }
        )

    payload = request.get_json(silent=True)
    if not isinstance(payload, dict) or set(payload) != {"enabled"}:
        return jsonify({"error": "body must contain exactly enabled"}), 400
    if not isinstance(payload["enabled"], bool):
        return jsonify({"error": "enabled must be a boolean"}), 400
    return jsonify(set_face_events_enabled(payload["enabled"]))


def jpeg_response(frame) -> tuple[Response, int] | Response:
    if frame is None:
        return Response("no frame", mimetype="text/plain"), 503
    encoded, jpeg = cv2.imencode(".jpg", frame)
    if not encoded:
        return Response("encode failed", mimetype="text/plain"), 500
    return Response(jpeg.tobytes(), mimetype="image/jpeg")


@app.route("/snapshot")
def snapshot():
    with frame_lock:
        frame = None if latest_frame is None else latest_frame.copy()
    return jpeg_response(frame)


@app.route("/snapshot/annotated")
def annotated_snapshot():
    with frame_lock:
        frame = (
            None if latest_annotated_frame is None else latest_annotated_frame.copy()
        )
    return jpeg_response(frame)


def annotated_frame_stream():
    last_sequence = -1
    next_frame_at = 0.0
    frame_interval = 1.0 / UI_STREAM_FPS

    while True:
        remaining = next_frame_at - time.monotonic()
        if remaining > 0:
            time.sleep(remaining)

        with frame_condition:
            frame_condition.wait_for(
                lambda previous_sequence=last_sequence: (
                    latest_annotated_frame is not None
                    and latest_frame_sequence != previous_sequence
                ),
                timeout=1.0,
            )
            if (
                latest_annotated_frame is None
                or latest_frame_sequence == last_sequence
            ):
                continue
            frame = latest_annotated_frame.copy()
            last_sequence = latest_frame_sequence

        encoded, jpeg = cv2.imencode(
            ".jpg",
            frame,
            [cv2.IMWRITE_JPEG_QUALITY, UI_JPEG_QUALITY],
        )
        if not encoded:
            continue

        image = jpeg.tobytes()
        next_frame_at = time.monotonic() + frame_interval
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n"
            + f"Content-Length: {len(image)}\r\n\r\n".encode("ascii")
            + image
            + b"\r\n"
        )


@app.route("/stream/annotated")
def annotated_stream():
    response = Response(
        annotated_frame_stream(),
        content_type="multipart/x-mixed-replace; boundary=frame",
    )
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    return response


@app.route("/faces", methods=["GET"])
def registered_faces():
    return jsonify({"people": face_registry.list_people()})


@app.route("/tracks", methods=["GET"])
def face_track_states():
    return jsonify({"tracks": face_service.track_states()})


@app.route("/faces/enroll", methods=["POST"])
def enroll_face():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "body must be a JSON object"}), 400
    if set(payload) != {"track_id", "name"}:
        return jsonify({"error": "body must contain exactly track_id and name"}), 400

    track_id = str(payload["track_id"]).strip()
    name = payload["name"]
    if not track_id:
        return jsonify({"error": "track_id must not be empty"}), 400
    if not isinstance(name, str) or not name.strip() or len(name.strip()) > 80:
        return jsonify({"error": "name must be a non-empty string of at most 80 characters"}), 400
    if any(ord(character) < 32 for character in name):
        return jsonify({"error": "name must not contain control characters"}), 400

    try:
        enrolled = face_service.enroll(track_id, name.strip())
    except LookupError as error:
        return jsonify({"error": str(error)}), 409
    except ValueError as error:
        return jsonify({"error": str(error)}), 400
    return jsonify({"status": "collecting", "person": enrolled}), 202


def run_server() -> None:
    app.run(host=HTTP_HOST, port=HTTP_PORT, debug=False, use_reloader=False)


threading.Thread(target=run_server, daemon=True, name="vision-http").start()

model = YOLO(YOLO_MODEL_PATH)
person_class_ids = [
    int(class_id)
    for class_id, class_name in model.names.items()
    if class_name == "person"
]
effective_tracker_config_path = prepare_tracker_config()
face_service.start()
cap = cv2.VideoCapture(CAMERA_INDEX)
prev_active_ids: set[str] = set()
track_positions: dict[str, str] = {}
track_lifecycle = TrackLifecycle(max_missing_frames=TRACK_MAX_AGE)


def clamp_bbox(bounds, frame_shape) -> tuple[int, int, int, int]:
    height, width = frame_shape[:2]
    left, top, right, bottom = (int(value) for value in bounds)
    return (
        max(0, min(width, left)),
        max(0, min(height, top)),
        max(0, min(width, right)),
        max(0, min(height, bottom)),
    )


def horizontal_position(left: int, right: int, frame_width: int) -> str:
    center_ratio = ((left + right) / 2) / max(1, frame_width)
    if center_ratio < 1 / 3:
        return "left"
    if center_ratio > 2 / 3:
        return "right"
    return "center"


def identity_label(identity: dict) -> str:
    if identity.get("status") == "recognized":
        return str(identity["name"])
    if identity.get("status") == "unknown":
        return "Unknown"
    if identity.get("status") == "unavailable":
        return "Face unavailable"
    return "Identifying"


try:
    while True:
        captured, frame = cap.read()
        if not captured:
            print("camera capture failed")
            break

        raw_snapshot = frame.copy()
        result = model.track(
            frame,
            persist=True,
            tracker=effective_tracker_config_path,
            verbose=False,
            classes=person_class_ids or None,
            conf=YOLO_TRACK_CONFIDENCE_THRESHOLD,
        )[0]
        observed_ids: set[str] = set()

        for observation in observations_from_result(result):
            x1, y1, x2, y2 = observation.bounds
            confidence = observation.confidence
            class_id = observation.class_id
            label = model.names[class_id]
            if label != "person":
                continue
            width = x2 - x1
            height = y2 - y1
            if not is_plausible_person_detection(
                width=width,
                height=height,
                confidence=confidence,
                frame_width=frame.shape[1],
                frame_height=frame.shape[0],
                minimum_confidence=YOLO_TRACK_CONFIDENCE_THRESHOLD,
                minimum_width=YOLO_MIN_PERSON_WIDTH,
                minimum_height=YOLO_MIN_PERSON_HEIGHT,
                minimum_area_ratio=YOLO_MIN_PERSON_AREA_RATIO,
            ):
                continue
            track_id = observation.track_id
            observed_ids.add(track_id)
            if face_service.is_superseded(track_id):
                continue

            face_service.mark_active(track_id, 0)
            left, top, right, bottom = clamp_bbox(observation.bounds, frame.shape)
            position = horizontal_position(left, right, frame.shape[1])
            track_positions[track_id] = position
            identity = face_service.identity_for(track_id)

            if (
                right - left >= FACE_MIN_PERSON_WIDTH
                and bottom - top >= FACE_MIN_PERSON_HEIGHT
            ):
                face_search_bottom = min(
                    bottom,
                    top + max(1, round((bottom - top) * FACE_PERSON_CROP_TOP_RATIO)),
                )
                person_crop = raw_snapshot[top:face_search_bottom, left:right]
                if person_crop.size:
                    face_service.submit(track_id, person_crop)

            color = (0, 200, 0) if identity.get("status") == "recognized" else (0, 200, 255)
            cv2.putText(
                frame,
                f"ID {track_id}: {identity_label(identity)}",
                (left, max(20, top - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                color,
                2,
            )
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

        tracker_active_ids, _ = track_lifecycle.update(observed_ids)
        face_service.prune_superseded_tracks(tracker_active_ids)
        for stale_id in tracker_active_ids - observed_ids:
            if not face_service.is_superseded(stale_id):
                face_service.mark_active(
                    stale_id,
                    track_lifecycle.staleness(stale_id),
                )
        current_active_ids = {
            track_id
            for track_id in tracker_active_ids
            if not face_service.is_superseded(track_id)
        }

        for disappeared_id in prev_active_ids - current_active_ids:
            if face_service.identity_event_announced(disappeared_id):
                send_person_event(
                    "person_disappeared",
                    disappeared_id,
                    position=track_positions.get(disappeared_id),
                    identity=face_service.identity_for(disappeared_id),
                )
            face_service.forget_track(disappeared_id)
            track_positions.pop(disappeared_id, None)

        prev_active_ids = current_active_ids

        with frame_condition:
            latest_frame = raw_snapshot
            latest_annotated_frame = frame.copy()
            latest_frame_sequence += 1
            frame_condition.notify_all()

        if DISPLAY_ENABLED:
            event_label = "ON" if face_events_enabled() else "OFF"
            event_color = (0, 200, 0) if face_events_enabled() else (0, 0, 255)
            cv2.putText(
                frame,
                f"Face events: {event_label} (E to toggle)",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                event_color,
                2,
            )
            cv2.imshow("Embodied Vision - BoT-SORT + InsightFace", frame)
            pressed_key = cv2.waitKey(1) & 0xFF
            if pressed_key == 27:
                break
            if pressed_key in (ord("e"), ord("E")):
                set_face_events_enabled(not face_events_enabled())
finally:
    cap.release()
    if DISPLAY_ENABLED:
        cv2.destroyAllWindows()
