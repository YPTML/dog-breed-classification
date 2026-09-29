"""Regression tests for the full TASK 16 OOD benchmark."""

import unittest

import numpy as np

from src.ood.benchmark import SUPPORTED_METHODS, benchmark_methods


class OODBenchmarkTests(unittest.TestCase):
    def test_all_methods_use_the_same_metric_protocol(self):
        id_probabilities = np.asarray(
            [[0.90, 0.08, 0.01, 0.01], [0.80, 0.10, 0.05, 0.05]], dtype=np.float64
        )
        ood_probabilities = np.asarray(
            [[0.26, 0.25, 0.25, 0.24], [0.30, 0.25, 0.25, 0.20]], dtype=np.float64
        )
        id_logits = np.log(id_probabilities)
        ood_logits = np.log(ood_probabilities)
        id_embeddings = np.asarray([[1.0, 0.0], [0.9, 0.1]], dtype=np.float64)
        ood_embeddings = np.asarray([[0.0, 1.0], [0.1, 0.9]], dtype=np.float64)
        results = benchmark_methods(
            id_probabilities,
            ood_probabilities,
            id_logits=id_logits,
            ood_logits=ood_logits,
            id_embeddings=id_embeddings,
            ood_embeddings=ood_embeddings,
            centroids=np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64),
            mahalanobis_mean=np.zeros(2, dtype=np.float64),
            mahalanobis_covariance=np.eye(2, dtype=np.float64),
        )

        self.assertEqual(tuple(results), SUPPORTED_METHODS)
        for name, metrics in results.items():
            self.assertEqual(metrics["status"], "evaluated", name)
            for field in (
                "auroc",
                "fpr95",
                "aupr_in",
                "aupr_out",
                "ood_f1",
                "false_acceptance_rate",
                "false_rejection_rate",
            ):
                self.assertIn(field, metrics, name)
                self.assertTrue(np.isfinite(metrics[field]), (name, field))

    def test_energy_is_unavailable_without_logits_and_embeddings_are_explicit(self):
        probabilities = np.asarray([[0.8, 0.2], [0.7, 0.3]], dtype=np.float64)
        results = benchmark_methods(probabilities, probabilities.copy())

        self.assertEqual(results["energy"]["status"], "unavailable")
        self.assertIn("pre-softmax", results["energy"]["status_reason"])
        self.assertEqual(results["cosine"]["status"], "unavailable")
        self.assertEqual(results["mahalanobis"]["status"], "unavailable")

    def test_missing_ood_does_not_claim_separation(self):
        probabilities = np.asarray([[0.8, 0.2], [0.7, 0.3]], dtype=np.float64)
        empty_ood = np.empty((0, 2), dtype=np.float64)
        results = benchmark_methods(probabilities, empty_ood)

        self.assertEqual(results["msp"]["status"], "unavailable")
        self.assertIsNone(results["msp"]["auroc"])
        self.assertIsNone(results["msp"]["false_acceptance_rate"])


if __name__ == "__main__":
    unittest.main()
