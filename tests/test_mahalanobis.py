"""Mahalanobis fitting and singular-covariance regressions."""

import unittest

import numpy as np

from src.ood.embedding_distance import (
    fit_mahalanobis_statistics,
    mahalanobis_distance,
    regularize_covariance,
)


class MahalanobisTests(unittest.TestCase):
    def test_covariance_is_fit_from_training_embeddings_only(self):
        training = np.array([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
        mean, covariance = fit_mahalanobis_statistics(training)

        np.testing.assert_allclose(mean, [2.0, 0.0])
        np.testing.assert_allclose(covariance, covariance.T)
        self.assertEqual(covariance.shape, (2, 2))
        self.assertTrue(np.isfinite(covariance).all())
        # The second direction has zero training variance; the ridge repairs it.
        self.assertGreater(covariance[1, 1], 0.0)

    def test_singular_and_ill_conditioned_covariance_stays_finite(self):
        covariance = np.array([[1.0, 0.0], [0.0, 0.0]])
        scores = mahalanobis_distance(
            np.array([[1.0, 2.0], [0.0, 0.0]]),
            np.zeros(2),
            covariance,
            regularization=0.0,
        )
        self.assertTrue(np.isfinite(scores).all())
        self.assertAlmostEqual(float(scores[0]), 1.0)
        self.assertAlmostEqual(float(scores[1]), 0.0)

    def test_multiple_means_use_nearest_mahalanobis_distance(self):
        scores = mahalanobis_distance(
            np.array([[0.2, 0.0], [4.0, 0.0]]),
            np.array([[0.0, 0.0], [5.0, 0.0]]),
            np.eye(2),
        )
        np.testing.assert_allclose(scores, [0.04, 1.0], rtol=0.0, atol=2e-6)

    def test_regularization_validation(self):
        with self.assertRaises(ValueError):
            regularize_covariance(np.eye(2), -1.0)
        with self.assertRaises(ValueError):
            mahalanobis_distance(
                np.zeros((1, 2)), np.zeros(2), np.eye(2), pinv_rcond=-1.0
            )


if __name__ == "__main__":
    unittest.main()
