"""Tests for Task 15 OOD threshold calibration."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.ood.calibration import (
    MAX_F1,
    MIN_WEIGHTED_ERROR,
    STRATEGIES,
    TARGET_ID_TPR,
    calibrate_threshold,
    calibrate_thresholds,
    save_thresholds,
)


class OODThresholdCalibrationTests(unittest.TestCase):
    def setUp(self):
        self.id_scores = np.array([0.10, 0.20, 0.30, 0.40, 0.50])
        self.ood_scores = np.array([0.70, 0.80, 0.90, 1.00])

    def test_all_three_strategies_are_returned_and_use_calibration_metrics(self):
        artifact = calibrate_thresholds(self.id_scores, self.ood_scores)

        self.assertEqual(set(artifact["strategies"]), set(STRATEGIES))
        self.assertEqual(artifact["method"], MAX_F1)
        self.assertEqual(artifact["data_sources"]["id"], "ID validation")
        self.assertEqual(artifact["data_sources"]["ood"], "OOD calibration")
        self.assertFalse(artifact["data_sources"]["final_test_used"])
        self.assertAlmostEqual(artifact["metrics"]["ood_f1"], 1.0)
        self.assertAlmostEqual(
            artifact["strategies"][TARGET_ID_TPR]["metrics"]["id_tpr"], 1.0
        )

    def test_target_id_tpr_keeps_at_least_the_requested_id_fraction(self):
        result = calibrate_thresholds(
            self.id_scores,
            self.ood_scores,
            method=TARGET_ID_TPR,
            target_tpr=0.95,
        )
        self.assertGreaterEqual(result["metrics"]["id_tpr"], 0.95)

    def test_weighted_error_uses_explicit_false_acceptance_and_rejection_weights(self):
        result = calibrate_thresholds(
            self.id_scores,
            np.array([0.25, 0.35, 0.45, 0.55]),
            method=MIN_WEIGHTED_ERROR,
            false_acceptance_weight=4.0,
            false_rejection_weight=1.0,
        )
        metrics = result["metrics"]
        expected = (
            4.0 * metrics["false_acceptance_rate"]
            + metrics["false_rejection_rate"]
        ) / 5.0
        self.assertAlmostEqual(metrics["weighted_error"], 5.0 * expected)
        self.assertAlmostEqual(metrics["weighted_error_rate"], expected)

    def test_lower_is_ood_returns_threshold_in_raw_score_units(self):
        result = calibrate_thresholds(
            1.0 - self.id_scores,
            1.0 - self.ood_scores,
            score_direction="lower_is_ood",
        )
        self.assertEqual(result["score_direction"], "lower_is_ood")
        self.assertAlmostEqual(result["metrics"]["ood_f1"], 1.0)
        self.assertGreater(result["threshold"], 0.0)

    def test_legacy_api_keeps_ood_threshold_alias(self):
        result = calibrate_threshold(self.id_scores, self.ood_scores)
        self.assertEqual(result["ood_threshold"], result["threshold"])
        self.assertEqual(result["target_id_acceptance"], 0.95)

    def test_strategy_alias_is_accepted(self):
        result = calibrate_thresholds(
            self.id_scores,
            self.ood_scores,
            strategy="max_f1",
        )
        self.assertEqual(result["method"], MAX_F1)

    def test_json_artifact_is_serializable_and_rejects_test_data_keys(self):
        artifact = calibrate_thresholds(self.id_scores, self.ood_scores)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ood_thresholds.json"
            save_thresholds(path, artifact)
            loaded = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(loaded["data_sources"]["final_test_used"], False)
        self.assertNotIn("test_scores", loaded)


if __name__ == "__main__":
    unittest.main()
