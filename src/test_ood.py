"""Fast unit tests for the standalone OOD and decision layers."""

import unittest

import numpy as np

from src.inference.decision import DecisionStatus, decide_prediction
from src.ood.calibration import calibrate_threshold
from src.ood.embedding_distance import cosine_distance, mahalanobis_distance
from src.ood.energy import energy_score, energy_ood_score, evaluate_energy_grid
from src.ood.entropy import entropy_ood_score
from src.ood.evaluate_ood import evaluate_ood_scores, rejection_error_rates
from src.ood.msp import msp_ood_score


class TestOODScorers(unittest.TestCase):
    def test_scores_are_finite_and_ood_oriented(self):
        probabilities = np.array([[0.9, 0.1], [0.5, 0.5]], dtype=np.float64)
        logits = np.array([[4.0, 0.0], [0.0, 0.0]], dtype=np.float64)
        self.assertLess(msp_ood_score(probabilities)[0], msp_ood_score(probabilities)[1])
        self.assertLess(entropy_ood_score(probabilities)[0], entropy_ood_score(probabilities)[1])
        self.assertLess(energy_ood_score(logits)[0], energy_ood_score(logits)[1])

    def test_energy_logits_temperature_validation(self):
        self.assertTrue(np.isfinite(energy_score(np.array([4.0, 0.0]), 2.0)))
        with self.assertRaises(ValueError):
            energy_score(np.array([1.0, 2.0]), 0.0)

    def test_temperature_grid_is_analysis_only(self):
        result = evaluate_energy_grid(
            np.array([[4.0, 0.0], [3.0, 0.0]]),
            np.array([[0.0, 0.0], [1.0, 0.0]]),
        )
        self.assertEqual(set(result["temperatures"]), {"1.0", "2.0", "5.0", "10.0"})
        self.assertIn(result["best_temperature"], {1.0, 2.0, 5.0, 10.0})
        self.assertFalse(result["production_selected"])

    def test_embedding_scores(self):
        features = np.array([[1.0, 0.0], [0.0, 1.0]])
        centers = np.array([[1.0, 0.0]])
        self.assertAlmostEqual(float(cosine_distance(features, centers)[0]), 0.0)
        self.assertAlmostEqual(float(cosine_distance(features, centers)[1]), 1.0)
        self.assertTrue(np.all(mahalanobis_distance(features, np.zeros(2), np.eye(2)) >= 0))


class TestCalibrationAndDecision(unittest.TestCase):
    def test_threshold_and_metrics(self):
        calibration = calibrate_threshold(np.array([0.1, 0.2, 0.3]), np.array([0.7, 0.8]))
        self.assertGreater(calibration["ood_threshold"], 0.0)
        metrics = evaluate_ood_scores(np.array([0.1, 0.2]), np.array([0.8, 0.9]))
        self.assertAlmostEqual(metrics["auroc"], 1.0)
        self.assertAlmostEqual(metrics["fpr95"], 0.0)
        errors = rejection_error_rates(np.array([0.1, 0.2]), np.array([0.8, 0.9]), 0.5)
        self.assertEqual(errors["false_acceptance_rate"], 0.0)
        self.assertEqual(errors["false_rejection_rate"], 0.0)

    def test_three_state_policy(self):
        names = {0: "a", 1: "b"}
        thresholds = {"uncertain": 0.4, "not_dog": 0.8, "breed_confidence": 0.45}
        self.assertIs(decide_prediction([0.9, 0.1], 0.1, names, thresholds).status, DecisionStatus.DOG)
        self.assertIs(decide_prediction([0.6, 0.4], 0.5, names, thresholds).status, DecisionStatus.UNCERTAIN)
        self.assertIs(decide_prediction([0.9, 0.1], 0.9, names, thresholds).status, DecisionStatus.NOT_DOG)


if __name__ == "__main__":
    unittest.main()
