"""Unit tests for the calibrated three-state decision layer."""

import json
import tempfile
import unittest
from pathlib import Path

from src.inference.decision import (
    DecisionStatus,
    decide_prediction,
    load_thresholds,
)


class TestDecisionLayer(unittest.TestCase):
    def setUp(self):
        self.class_names = {0: "beagle", 1: "poodle", 2: "collie"}
        self.policy = {
            "ood_threshold": 0.80,
            "uncertain_threshold": 0.55,
            "breed_confidence_threshold": 0.45,
            "score_direction": "higher_is_ood",
        }

    def test_dog_returns_breed_prediction(self):
        result = decide_prediction([0.70, 0.20, 0.10], 0.20, self.class_names, self.policy)

        self.assertIs(result.status, DecisionStatus.DOG)
        self.assertEqual(result.prediction["breed"], "beagle")
        self.assertEqual(result.prediction["index"], 0)
        self.assertEqual(result.as_ui_dict([result.prediction])["top5"], [result.prediction])

    def test_strong_ood_is_checked_before_uncertainty(self):
        result = decide_prediction([0.50, 0.30, 0.20], 0.90, {}, self.policy)

        self.assertIs(result.status, DecisionStatus.NOT_DOG)
        self.assertEqual(result.prediction, "not_dog")
        self.assertEqual(result.as_ui_dict([{"breed": "beagle"}])["top5"], [])

    def test_uncertain_does_not_return_a_breed(self):
        result = decide_prediction([0.70, 0.20, 0.10], 0.60, self.class_names, self.policy)

        self.assertIs(result.status, DecisionStatus.UNCERTAIN)
        self.assertEqual(result.prediction, "uncertain")
        self.assertIsNone(result.is_dog)
        self.assertEqual(result.as_ui_dict([{"breed": "beagle"}])["top5"], [])

    def test_low_breed_confidence_is_uncertain(self):
        result = decide_prediction([0.40, 0.35, 0.25], 0.10, self.class_names, self.policy)

        self.assertIs(result.status, DecisionStatus.UNCERTAIN)
        self.assertEqual(result.prediction, "uncertain")

    def test_threshold_is_loaded_from_json_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ood_thresholds.json"
            path.write_text(json.dumps(self.policy), encoding="utf-8")

            loaded = load_thresholds(path)
            result = decide_prediction([0.90, 0.05, 0.05], 0.75, self.class_names, path)

        self.assertEqual(loaded["ood_threshold"], 0.80)
        self.assertIs(result.status, DecisionStatus.UNCERTAIN)

    def test_lower_score_direction_is_supported(self):
        policy = {
            "threshold": 0.20,
            "uncertain_threshold": 0.35,
            "score_direction": "lower_is_ood",
        }

        self.assertIs(
            decide_prediction([0.90, 0.05, 0.05], 0.10, self.class_names, policy).status,
            DecisionStatus.NOT_DOG,
        )
        self.assertIs(
            decide_prediction([0.90, 0.05, 0.05], 0.30, self.class_names, policy).status,
            DecisionStatus.UNCERTAIN,
        )

    def test_missing_calibrated_threshold_fails_without_a_numeric_fallback(self):
        with self.assertRaises(ValueError):
            decide_prediction([0.90, 0.05, 0.05], 0.10, self.class_names, {})


if __name__ == "__main__":
    unittest.main()
