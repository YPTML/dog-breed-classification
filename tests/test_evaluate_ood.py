"""Unit tests for OOD metric arithmetic and score-direction handling."""

import unittest

import numpy as np

from src.ood.evaluate_ood import (
    DETECTORS,
    evaluate_detector,
    evaluate_ood_scores,
    rejection_error_rates,
)


class OODEvaluationMetricTests(unittest.TestCase):
    def test_perfect_separation_has_perfect_ranking_and_pr_metrics(self):
        metrics = evaluate_ood_scores(
            [0.10, 0.20, 0.30],
            [0.70, 0.80, 0.90],
            higher_score_is_ood=True,
        )

        for name in ("auroc", "fpr95", "aupr_in", "aupr_out"):
            self.assertAlmostEqual(metrics[name], 1.0 if name != "fpr95" else 0.0)
        self.assertAlmostEqual(metrics["ood_precision"], 1.0)
        self.assertAlmostEqual(metrics["ood_recall"], 1.0)
        self.assertAlmostEqual(metrics["ood_f1"], 1.0)

    def test_rank_and_average_precision_calculations_handle_interleaved_scores(self):
        metrics = evaluate_ood_scores(
            [0.10, 0.40],
            [0.30, 0.80],
            higher_score_is_ood=True,
        )

        self.assertAlmostEqual(metrics["auroc"], 0.75)
        self.assertAlmostEqual(metrics["fpr95"], 0.50)
        self.assertAlmostEqual(metrics["aupr_out"], 5 / 6)
        self.assertAlmostEqual(metrics["aupr_in"], 5 / 6)

    def test_msp_direction_is_inverted_from_raw_confidence_scores(self):
        raw_msp = evaluate_detector(
            "msp",
            np.array([0.95, 0.85, 0.75]),
            np.array([0.35, 0.25, 0.15]),
        )
        explicit_direction = evaluate_ood_scores(
            [0.95, 0.85, 0.75],
            [0.35, 0.25, 0.15],
            higher_score_is_ood=False,
        )

        self.assertFalse(DETECTORS["msp"].higher_score_is_ood)
        self.assertEqual(raw_msp["higher_score_is_ood"], False)
        for name in ("auroc", "fpr95", "aupr_in", "aupr_out", "ood_f1"):
            self.assertAlmostEqual(raw_msp[name], explicit_direction[name])
        self.assertAlmostEqual(raw_msp["auroc"], 1.0)

    def test_precision_recall_and_f1_use_a_provided_threshold(self):
        metrics = evaluate_ood_scores(
            [0.10, 0.80, 0.20],
            [0.90, 0.70],
            higher_score_is_ood=True,
            threshold=0.50,
        )

        # Predictions: one ID false positive and two OOD true positives.
        self.assertEqual(metrics["true_positive"], 2)
        self.assertEqual(metrics["false_positive"], 1)
        self.assertEqual(metrics["false_negative"], 0)
        self.assertAlmostEqual(metrics["ood_precision"], 2 / 3)
        self.assertAlmostEqual(metrics["ood_recall"], 1.0)
        self.assertAlmostEqual(metrics["ood_f1"], 0.8)

    def test_all_supported_detectors_declare_a_direction(self):
        self.assertEqual(
            set(DETECTORS), {"msp", "entropy", "energy", "cosine", "mahalanobis"}
        )
        self.assertEqual(
            {name: spec.higher_score_is_ood for name, spec in DETECTORS.items()},
            {
                "msp": False,
                "entropy": True,
                "energy": True,
                "cosine": True,
                "mahalanobis": True,
            },
        )

    def test_rejection_error_rates_respect_lower_is_ood_direction(self):
        errors = rejection_error_rates(
            [0.80, 0.90],
            [0.10, 0.20],
            threshold=0.50,
            higher_score_is_ood=False,
        )
        self.assertEqual(errors["false_acceptance_rate"], 0.0)
        self.assertEqual(errors["false_rejection_rate"], 0.0)

    def test_invalid_inputs_are_rejected(self):
        with self.assertRaises(ValueError):
            evaluate_ood_scores([], [0.1], higher_score_is_ood=True)
        with self.assertRaises(ValueError):
            evaluate_ood_scores([0.1], [np.nan], higher_score_is_ood=True)
        with self.assertRaises(ValueError):
            evaluate_ood_scores([0.1], [0.9], detector="unknown")
        with self.assertRaises(ValueError):
            evaluate_ood_scores(
                [0.1], [0.9], detector="msp", higher_score_is_ood=True
            )


if __name__ == "__main__":
    unittest.main()
