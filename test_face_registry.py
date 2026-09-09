import json
import tempfile
import unittest
from pathlib import Path

from face_registry import (
    FaceEncoding,
    FaceMatch,
    FaceRegistry,
    resolve_face_vote,
    select_representative_encodings,
)
from vision_filters import is_plausible_person_detection


class FaceRegistryTests(unittest.TestCase):
    def make_registry(self, directory: str) -> FaceRegistry:
        return FaceRegistry(
            Path(directory) / "faces.json",
            model_name="test-model",
            threshold=0.2,
            max_embeddings_per_person=2,
        )

    def test_enroll_match_and_reload(self):
        with tempfile.TemporaryDirectory() as directory:
            registry = self.make_registry(directory)
            enrolled = registry.enroll("Alice", [1.0, 0.0, 0.0])

            match = registry.match([0.99, 0.01, 0.0])
            self.assertIsNotNone(match)
            self.assertEqual(match.person_id, enrolled["person_id"])
            self.assertEqual(match.name, "Alice")

            reloaded = self.make_registry(directory)
            self.assertEqual(reloaded.list_people()[0]["name"], "Alice")

    def test_unknown_embedding_does_not_match(self):
        with tempfile.TemporaryDirectory() as directory:
            registry = self.make_registry(directory)
            registry.enroll("Alice", [1.0, 0.0])
            self.assertIsNone(registry.match([0.0, 1.0]))
            nearest = registry.nearest([0.0, 1.0])
            self.assertIsNotNone(nearest)
            self.assertEqual(nearest.name, "Alice")

    def test_same_name_adds_limited_exemplars(self):
        with tempfile.TemporaryDirectory() as directory:
            registry = self.make_registry(directory)
            first = registry.enroll("Alice", [1.0, 0.0, 0.0])
            second = registry.enroll("alice", [0.8, 0.6, 0.0])
            third = registry.enroll("Alice", [0.6, 0.8, 0.0])

            self.assertEqual(first["person_id"], second["person_id"])
            self.assertEqual(second["person_id"], third["person_id"])
            self.assertEqual(third["embedding_count"], 2)

            payload = json.loads(
                (Path(directory) / "faces.json").read_text(encoding="utf-8")
            )
            self.assertEqual(len(payload["people"][0]["embeddings"]), 2)

    def test_enroll_many_deduplicates_near_identical_embeddings(self):
        with tempfile.TemporaryDirectory() as directory:
            registry = self.make_registry(directory)
            enrolled = registry.enroll_many(
                "Alice",
                [[1.0, 0.0, 0.0], [0.999, 0.001, 0.0], [0.8, 0.6, 0.0]],
            )
            self.assertEqual(enrolled["embedding_count"], 2)
            self.assertEqual(enrolled["added_embedding_count"], 2)

    def test_representative_selection_rejects_isolated_outlier(self):
        def sample(embedding, quality):
            return FaceEncoding(
                embedding=tuple(embedding),
                detection_confidence=0.99,
                face_width=120,
                face_height=120,
                blur_variance=100.0,
                brightness=120.0,
                quality=quality,
            )

        consistent = [
            sample([1.0, 0.0, 0.0], 0.9),
            sample([0.98, 0.2, 0.0], 0.8),
            sample([0.95, 0.31, 0.0], 0.85),
        ]
        outlier = sample([0.0, 1.0, 0.0], 0.95)
        selected = select_representative_encodings(
            [*consistent, outlier],
            limit=3,
            cluster_distance=0.2,
        )

        self.assertGreaterEqual(len(selected), 2)
        self.assertNotIn(outlier, selected)

    def test_face_vote_tolerates_occasional_bad_frames(self):
        alice = FaceMatch("alice-id", "Alice", 0.21, 0.3)
        vote = resolve_face_vote(
            [None, alice, None, alice, alice],
            window_size=5,
            min_votes=3,
            min_vote_ratio=0.5,
        )
        self.assertIsNotNone(vote)
        self.assertEqual(vote.person_id, "alice-id")
        self.assertEqual(vote.votes, 3)

    def test_face_vote_rejects_split_identity_evidence(self):
        alice = FaceMatch("alice-id", "Alice", 0.21, 0.3)
        bob = FaceMatch("bob-id", "Bob", 0.20, 0.3)
        vote = resolve_face_vote(
            [alice, bob, alice, bob],
            window_size=4,
            min_votes=3,
            min_vote_ratio=0.5,
        )
        self.assertIsNone(vote)

    def test_two_consecutive_matches_allow_fast_recognition(self):
        alice = FaceMatch("alice-id", "Alice", 0.24, 0.3)
        vote = resolve_face_vote(
            [None, None, None, alice, alice],
            window_size=8,
            min_votes=3,
            min_vote_ratio=0.6,
            consecutive_matches=2,
        )
        self.assertIsNotNone(vote)
        self.assertEqual(vote.person_id, "alice-id")
        self.assertEqual(vote.votes, 2)

    def test_consecutive_matches_must_have_same_identity(self):
        alice = FaceMatch("alice-id", "Alice", 0.21, 0.3)
        bob = FaceMatch("bob-id", "Bob", 0.20, 0.3)
        vote = resolve_face_vote(
            [None, None, alice, bob],
            window_size=8,
            min_votes=3,
            min_vote_ratio=0.6,
            consecutive_matches=2,
        )
        self.assertIsNone(vote)

    def test_tiny_background_detection_is_rejected(self):
        accepted = is_plausible_person_detection(
            width=25,
            height=20,
            confidence=0.8,
            frame_width=640,
            frame_height=480,
            minimum_confidence=0.5,
            minimum_width=40,
            minimum_height=80,
            minimum_area_ratio=0.002,
        )
        self.assertFalse(accepted)

    def test_normal_person_detection_is_accepted(self):
        accepted = is_plausible_person_detection(
            width=180,
            height=300,
            confidence=0.8,
            frame_width=640,
            frame_height=480,
            minimum_confidence=0.5,
            minimum_width=40,
            minimum_height=80,
            minimum_area_ratio=0.002,
        )
        self.assertTrue(accepted)


if __name__ == "__main__":
    unittest.main()
