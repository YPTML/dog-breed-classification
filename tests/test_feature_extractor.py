"""Integration checks for embeddings from the production Keras classifier."""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from src.models.feature_extractor import (
    build_feature_extractor,
    extract_embeddings,
)


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "outputs" / "deployment_ready" / "dog_breed_model.keras"


class FeatureExtractorTests(unittest.TestCase):
    """Validate shape, numerical health, batching, determinism, and immutability."""

    @classmethod
    def setUpClass(cls) -> None:
        try:
            import tensorflow as tf
        except ImportError as exc:  # pragma: no cover - dependency is pinned
            raise unittest.SkipTest(f"TensorFlow is unavailable: {exc}") from exc
        if not MODEL_PATH.is_file():
            raise unittest.SkipTest(f"production model not found: {MODEL_PATH}")

        cls.tf = tf
        preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
        cls.model = tf.keras.models.load_model(
            MODEL_PATH,
            custom_objects={"preprocess_input": preprocess_input},
            compile=False,
            safe_mode=True,
        )
        cls.extractor = build_feature_extractor(cls.model)
        cls.images = np.random.default_rng(42).uniform(
            0.0, 255.0, size=(3, 224, 224, 3)
        ).astype(np.float32)

    def test_output_shape_and_finite_values(self) -> None:
        embeddings = extract_embeddings(self.extractor, self.images[:1])
        self.assertEqual(embeddings.shape, (1, 1280))
        self.assertFalse(np.isnan(embeddings).any())
        self.assertFalse(np.isinf(embeddings).any())
        self.assertEqual(self.extractor.embedding_dim, 1280)

    def test_batch_inference(self) -> None:
        embeddings = extract_embeddings(self.extractor, self.images)
        self.assertEqual(embeddings.shape, (3, 1280))
        for index in range(self.images.shape[0]):
            single = extract_embeddings(self.extractor, self.images[index : index + 1])
            # oneDNN may use a different reduction order for batch size 3 vs 1;
            # keep this check tight while allowing that bounded float32 drift.
            np.testing.assert_allclose(embeddings[index], single[0], rtol=2e-4, atol=2e-5)

    def test_deterministic_inference_and_weights_unchanged(self) -> None:
        before = [np.array(weight.numpy(), copy=True) for weight in self.model.weights]
        first = np.asarray(self.extractor(self.images, training=False))
        second = np.asarray(self.extractor(self.images, training=False))
        np.testing.assert_array_equal(first, second)
        for initial, current in zip(before, self.model.weights):
            np.testing.assert_array_equal(initial, current.numpy())


if __name__ == "__main__":
    unittest.main()
