"""Persistent face embeddings and an InsightFace adapter.

The registry deliberately stores embeddings only. Camera frames and face crops stay
in memory and are never written to disk by this module.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import threading
import uuid
from statistics import median
from typing import Iterable, Optional


REGISTRY_VERSION = 1


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalise_embedding(values: Iterable[float]) -> list[float]:
    embedding = [float(value) for value in values]
    if not embedding or not all(math.isfinite(value) for value in embedding):
        raise ValueError("embedding must contain finite numbers")

    norm = math.sqrt(sum(value * value for value in embedding))
    if norm <= 0:
        raise ValueError("embedding norm must be positive")
    return [value / norm for value in embedding]


@dataclass(frozen=True)
class FaceMatch:
    person_id: str
    name: str
    distance: float
    threshold: float


@dataclass(frozen=True)
class FaceEncoding:
    embedding: tuple[float, ...]
    detection_confidence: float
    face_width: int
    face_height: int
    blur_variance: float
    brightness: float
    quality: float

    def public_quality(self) -> dict:
        return {
            "detection_confidence": round(self.detection_confidence, 4),
            "face_width": self.face_width,
            "face_height": self.face_height,
            "blur_variance": round(self.blur_variance, 1),
            "brightness": round(self.brightness, 1),
            "quality": round(self.quality, 3),
        }


@dataclass(frozen=True)
class FaceVote:
    person_id: str
    name: str
    distance: float
    threshold: float
    votes: int
    window: int


def cosine_distance(left: Iterable[float], right: Iterable[float]) -> float:
    left_normalised = _normalise_embedding(left)
    right_normalised = _normalise_embedding(right)
    if len(left_normalised) != len(right_normalised):
        raise ValueError("embedding dimensions must match")
    similarity = sum(
        left_value * right_value
        for left_value, right_value in zip(left_normalised, right_normalised)
    )
    return 1.0 - max(-1.0, min(1.0, similarity))


def select_representative_encodings(
    encodings: Iterable[FaceEncoding],
    *,
    limit: int,
    cluster_distance: float,
    duplicate_distance: float = 0.015,
) -> list[FaceEncoding]:
    """Remove isolated samples, then retain high-quality diverse exemplars."""
    samples = list(encodings)
    if not samples or limit < 1:
        return []
    if len(samples) == 1:
        return samples

    neighbours: list[set[int]] = [set() for _ in samples]
    for left_index, left in enumerate(samples):
        for right_index in range(left_index + 1, len(samples)):
            distance = cosine_distance(
                left.embedding,
                samples[right_index].embedding,
            )
            if distance <= cluster_distance:
                neighbours[left_index].add(right_index)
                neighbours[right_index].add(left_index)

    components: list[list[int]] = []
    remaining = set(range(len(samples)))
    while remaining:
        seed = remaining.pop()
        component = [seed]
        pending = [seed]
        while pending:
            current = pending.pop()
            connected = neighbours[current] & remaining
            remaining.difference_update(connected)
            component.extend(connected)
            pending.extend(connected)
        components.append(component)

    component = max(
        components,
        key=lambda indexes: (
            len(indexes),
            sum(samples[index].quality for index in indexes) / len(indexes),
        ),
    )
    candidates = [samples[index] for index in component]
    first = max(candidates, key=lambda sample: sample.quality)
    selected = [first]
    candidates.remove(first)

    while candidates and len(selected) < limit:
        ranked = []
        for candidate in candidates:
            nearest_selected = min(
                cosine_distance(candidate.embedding, item.embedding)
                for item in selected
            )
            ranked.append(
                (
                    nearest_selected * 0.75 + candidate.quality * 0.25,
                    nearest_selected,
                    candidate,
                )
            )
        _, nearest_selected, chosen = max(ranked, key=lambda item: item[0])
        candidates.remove(chosen)
        if nearest_selected >= duplicate_distance:
            selected.append(chosen)

    return selected


def resolve_face_vote(
    candidates: Iterable[Optional[FaceMatch]],
    *,
    window_size: int,
    min_votes: int,
    min_vote_ratio: float,
    consecutive_matches: int = 0,
) -> Optional[FaceVote]:
    recent = list(candidates)[-max(1, window_size) :]
    if not recent:
        return None

    consecutive_matches = max(0, consecutive_matches)
    if consecutive_matches and len(recent) >= consecutive_matches:
        tail = recent[-consecutive_matches:]
        if tail[0] is not None and all(
            candidate is not None and candidate.person_id == tail[0].person_id
            for candidate in tail
        ):
            representative = min(tail, key=lambda candidate: candidate.distance)
            return FaceVote(
                person_id=representative.person_id,
                name=representative.name,
                distance=median(candidate.distance for candidate in tail),
                threshold=representative.threshold,
                votes=consecutive_matches,
                window=consecutive_matches,
            )

    vote_counts = Counter(
        candidate.person_id for candidate in recent if candidate is not None
    )
    if not vote_counts:
        return None

    winner_id, winner_votes = vote_counts.most_common(1)[0]
    if winner_votes < max(1, min_votes) or winner_votes / len(recent) < min_vote_ratio:
        return None
    winner_matches = [
        candidate
        for candidate in recent
        if candidate is not None and candidate.person_id == winner_id
    ]
    representative = min(winner_matches, key=lambda candidate: candidate.distance)
    return FaceVote(
        person_id=representative.person_id,
        name=representative.name,
        distance=median(candidate.distance for candidate in winner_matches),
        threshold=representative.threshold,
        votes=winner_votes,
        window=len(recent),
    )


class FaceRegistry:
    """Thread-safe JSON registry for named face embeddings."""

    def __init__(
        self,
        path: str | Path,
        *,
        model_name: str,
        threshold: float,
        max_embeddings_per_person: int = 5,
    ) -> None:
        if threshold <= 0:
            raise ValueError("threshold must be positive")
        if max_embeddings_per_person < 1:
            raise ValueError("max_embeddings_per_person must be at least 1")

        self.path = Path(path)
        self.model_name = model_name
        self.threshold = float(threshold)
        self.max_embeddings_per_person = max_embeddings_per_person
        self._lock = threading.RLock()
        self._people: dict[str, dict] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return

        payload = json.loads(self.path.read_text(encoding="utf-8"))
        if payload.get("version") != REGISTRY_VERSION:
            raise ValueError(f"unsupported face registry version: {payload.get('version')}")
        if payload.get("model") != self.model_name:
            raise ValueError(
                "face registry model does not match configuration: "
                f"{payload.get('model')} != {self.model_name}"
            )

        people = payload.get("people")
        if not isinstance(people, list):
            raise ValueError("face registry people must be a list")

        loaded: dict[str, dict] = {}
        for person in people:
            person_id = str(person["person_id"])
            name = str(person["name"]).strip()
            embeddings = [
                _normalise_embedding(embedding)
                for embedding in person.get("embeddings", [])
            ]
            if not person_id or not name or not embeddings:
                continue
            loaded[person_id] = {
                "person_id": person_id,
                "name": name,
                "created_at": person.get("created_at") or _utc_now(),
                "updated_at": person.get("updated_at") or _utc_now(),
                "embeddings": embeddings[-self.max_embeddings_per_person :],
            }
        self._people = loaded

    def _save_locked(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": REGISTRY_VERSION,
            "model": self.model_name,
            "distance_metric": "cosine",
            "people": list(self._people.values()),
        }
        temporary_path = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary_path.replace(self.path)

    def list_people(self) -> list[dict]:
        with self._lock:
            return [
                {
                    "person_id": person["person_id"],
                    "name": person["name"],
                    "created_at": person["created_at"],
                    "updated_at": person["updated_at"],
                    "embedding_count": len(person["embeddings"]),
                }
                for person in sorted(
                    self._people.values(), key=lambda item: item["name"].casefold()
                )
            ]

    def match(self, embedding: Iterable[float]) -> Optional[FaceMatch]:
        best = self.nearest(embedding)
        if best is None or best.distance > self.threshold:
            return None
        return best

    def nearest(self, embedding: Iterable[float]) -> Optional[FaceMatch]:
        """Return the nearest registered face even when it exceeds the threshold."""
        candidate = _normalise_embedding(embedding)
        best: Optional[FaceMatch] = None

        with self._lock:
            for person in self._people.values():
                for reference in person["embeddings"]:
                    if len(candidate) != len(reference):
                        continue
                    similarity = sum(
                        left * right for left, right in zip(candidate, reference)
                    )
                    distance = 1.0 - max(-1.0, min(1.0, similarity))
                    if best is None or distance < best.distance:
                        best = FaceMatch(
                            person_id=person["person_id"],
                            name=person["name"],
                            distance=distance,
                            threshold=self.threshold,
                        )

        return best

    def enroll(self, name: str, embedding: Iterable[float]) -> dict:
        return self.enroll_many(name, [embedding])

    def enroll_many(
        self,
        name: str,
        embeddings: Iterable[Iterable[float]],
    ) -> dict:
        clean_name = name.strip()
        if not clean_name:
            raise ValueError("name must not be empty")

        normalised_embeddings = [
            _normalise_embedding(embedding) for embedding in embeddings
        ]
        if not normalised_embeddings:
            raise ValueError("at least one embedding is required")
        now = _utc_now()

        with self._lock:
            person = next(
                (
                    item
                    for item in self._people.values()
                    if item["name"].casefold() == clean_name.casefold()
                ),
                None,
            )
            if person is None:
                person_id = uuid.uuid4().hex
                person = {
                    "person_id": person_id,
                    "name": clean_name,
                    "created_at": now,
                    "updated_at": now,
                    "embeddings": [],
                }
                self._people[person_id] = person

            person["name"] = clean_name
            person["updated_at"] = now
            added_count = 0
            for normalised in normalised_embeddings:
                if any(
                    cosine_distance(normalised, existing) < 0.015
                    for existing in person["embeddings"]
                ):
                    continue
                person["embeddings"].append(normalised)
                added_count += 1
            person["embeddings"] = person["embeddings"][-self.max_embeddings_per_person :]
            self._save_locked()

            return {
                "person_id": person["person_id"],
                "name": person["name"],
                "embedding_count": len(person["embeddings"]),
                "added_embedding_count": added_count,
            }


class InsightFaceEncoder:
    """Lazy InsightFace wrapper with explicit quality gates.

    ``analysis_app`` exists for deterministic unit tests. Production creates a
    ``FaceAnalysis`` pipeline, which performs SCRFD detection, alignment and
    buffalo-family recognition in one pass.
    """

    def __init__(
        self,
        *,
        model_name: str,
        providers: Optional[Iterable[str]] = None,
        model_root: Optional[str] = None,
        detection_size: int = 640,
        require_accelerator: bool = False,
        min_face_size: int = 70,
        min_detection_confidence: float = 0.90,
        min_blur_variance: float = 35.0,
        min_brightness: float = 35.0,
        max_brightness: float = 225.0,
        max_input_dimension: int = 640,
        analysis_app=None,
    ) -> None:
        self.model_name = model_name
        self.detector_backend = "SCRFD"
        self.min_face_size = min_face_size
        self.min_detection_confidence = min_detection_confidence
        self.min_blur_variance = min_blur_variance
        self.min_brightness = min_brightness
        self.max_brightness = max_brightness
        self.max_input_dimension = max_input_dimension
        self.detection_size = max(128, int(detection_size))

        if analysis_app is not None:
            self._analysis = analysis_app
            self.providers = ["test"]
            return

        import onnxruntime as ort  # type: ignore
        from insightface.app import FaceAnalysis  # type: ignore

        available = ort.get_available_providers()
        requested = list(providers or ("CUDAExecutionProvider", "CPUExecutionProvider"))
        selected = [provider for provider in requested if provider in available]
        if not selected:
            selected = list(available)
        accelerators = {
            "CUDAExecutionProvider",
            "TensorrtExecutionProvider",
        }
        if require_accelerator and not accelerators.intersection(selected):
            raise RuntimeError(
                "GPU face recognition is required, but ONNX Runtime exposes only "
                f"{available}. Remove the CPU 'onnxruntime' package and install a "
                "CUDA/JetPack-compatible 'onnxruntime-gpu' build, or set "
                "FACE_REQUIRE_GPU=false to allow CPU fallback."
            )
        if not selected:
            raise RuntimeError("ONNX Runtime has no available execution provider")

        options = {
            "name": model_name,
            "providers": selected,
            "allowed_modules": ["detection", "recognition"],
        }
        if model_root:
            options["root"] = model_root
        self._analysis = FaceAnalysis(**options)
        self._analysis.prepare(
            ctx_id=(
                0
                if accelerators.intersection(selected)
                else -1
            ),
            det_size=(self.detection_size, self.detection_size),
            det_thresh=self.min_detection_confidence,
        )
        actual_providers: list[str] = []
        for model in self._analysis.models.values():
            session = getattr(model, "session", None)
            if session is None or not hasattr(session, "get_providers"):
                continue
            for provider in session.get_providers():
                if provider not in actual_providers:
                    actual_providers.append(provider)
        self.providers = actual_providers or selected
        if require_accelerator and not accelerators.intersection(self.providers):
            raise RuntimeError(
                "InsightFace sessions fell back to CPU even though GPU was "
                f"requested; active providers are {self.providers}. Check CUDA, "
                "cuDNN and ONNX Runtime GPU compatibility."
            )

    @staticmethod
    def _face_value(face, name: str, default=None):
        if hasattr(face, name):
            return getattr(face, name)
        if isinstance(face, dict):
            return face.get(name, default)
        return default

    def encode(self, image) -> FaceEncoding:
        import cv2  # type: ignore

        image_height, image_width = image.shape[:2]
        longest_edge = max(image_width, image_height)
        if self.max_input_dimension > 0 and longest_edge > self.max_input_dimension:
            scale = self.max_input_dimension / longest_edge
            image = cv2.resize(
                image,
                (max(1, round(image_width * scale)), max(1, round(image_height * scale))),
                interpolation=cv2.INTER_AREA,
            )

        faces = self._analysis.get(image)
        if not faces:
            raise ValueError("no face detected")

        selected_face = max(
            faces,
            key=lambda face: (
                float(self._face_value(face, "det_score", 0.0) or 0.0)
                * max(
                    0.0,
                    float(self._face_value(face, "bbox", [0, 0, 0, 0])[2])
                    - float(self._face_value(face, "bbox", [0, 0, 0, 0])[0]),
                )
                * max(
                    0.0,
                    float(self._face_value(face, "bbox", [0, 0, 0, 0])[3])
                    - float(self._face_value(face, "bbox", [0, 0, 0, 0])[1]),
                )
            ),
        )
        bbox = self._face_value(selected_face, "bbox")
        if bbox is None or len(bbox) < 4:
            raise ValueError("face detector returned no bounding box")
        image_height, image_width = image.shape[:2]
        left = max(0, min(image_width, int(round(float(bbox[0])))))
        top = max(0, min(image_height, int(round(float(bbox[1])))))
        right = max(left, min(image_width, int(round(float(bbox[2])))))
        bottom = max(top, min(image_height, int(round(float(bbox[3])))))
        face_width = right - left
        face_height = bottom - top
        confidence = float(self._face_value(selected_face, "det_score", 0.0) or 0.0)

        if min(face_width, face_height) < self.min_face_size:
            raise ValueError(
                f"face too small: {face_width}x{face_height}, "
                f"minimum={self.min_face_size}"
            )
        if confidence < self.min_detection_confidence:
            raise ValueError(
                f"face confidence too low: {confidence:.3f}, "
                f"minimum={self.min_detection_confidence:.3f}"
            )

        face_crop = image[top:bottom, left:right]
        if not face_crop.size:
            raise ValueError("detected face crop is empty")
        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        blur_variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        brightness = float(gray.mean())
        if blur_variance < self.min_blur_variance:
            raise ValueError(
                f"face too blurry: {blur_variance:.1f}, "
                f"minimum={self.min_blur_variance:.1f}"
            )
        if not self.min_brightness <= brightness <= self.max_brightness:
            raise ValueError(
                f"face exposure out of range: {brightness:.1f}, "
                f"expected={self.min_brightness:.1f}-{self.max_brightness:.1f}"
            )

        size_score = min(1.0, min(face_width, face_height) / 180.0)
        sharpness_score = min(1.0, blur_variance / 250.0)
        exposure_score = max(0.0, 1.0 - abs(brightness - 130.0) / 130.0)
        quality = (
            confidence * 0.45
            + size_score * 0.25
            + sharpness_score * 0.20
            + exposure_score * 0.10
        )
        embedding = self._face_value(selected_face, "normed_embedding")
        if embedding is None:
            embedding = self._face_value(selected_face, "embedding")
        if embedding is None:
            raise ValueError("face recognizer returned no embedding")
        return FaceEncoding(
            embedding=tuple(_normalise_embedding(embedding)),
            detection_confidence=confidence,
            face_width=face_width,
            face_height=face_height,
            blur_variance=blur_variance,
            brightness=brightness,
            quality=quality,
        )
