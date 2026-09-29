"""Regressions discovered by the final forensic audit (no threshold fitting)."""
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import tensorflow as tf
from PIL import Image

import app
from src.inference.oe_v2_predictor import (
    EXPECTED_CENTROIDS_SHA256,
    EXPECTED_CENTROID_SHAPE,
    EXPECTED_MODEL_SHA256,
    LOCKED_T_NOT_DOG,
    LOCKED_T_UNCERTAIN,
    compute_file_sha256,
    decide_tri_state,
)
from src.inference.predictor import Predictor
from src.inference.preprocessing import preprocess_image
from src.ood.embedding_distance import load_centroid_artifacts, mahalanobis_distance
from src.ood.energy import build_logits_model, energy_score
from src.ood.entropy import predictive_entropy


class AuditRegressions(unittest.TestCase):
    def test_jpeg_preprocessing_matches_training(self):
        pixels = np.random.default_rng(7).integers(0, 256, (341, 517, 3), dtype=np.uint8)
        buffer = io.BytesIO()
        Image.fromarray(pixels).save(buffer, format='JPEG')
        data = buffer.getvalue()
        expected = tf.image.resize(tf.image.decode_jpeg(data, channels=3), (224, 224)).numpy()[None]
        np.testing.assert_array_equal(preprocess_image(data), expected)
        np.testing.assert_array_equal(app.preprocess_uploaded_image(data), expected)

    def test_locked_artifact_integrity(self):
        self.assertEqual(compute_file_sha256(app.MODEL_PATH), EXPECTED_MODEL_SHA256)
        self.assertEqual(compute_file_sha256(app.CENTROIDS_PATH), EXPECTED_CENTROIDS_SHA256)

    def test_locked_threshold_order_and_boundaries(self):
        self.assertLess(LOCKED_T_UNCERTAIN, LOCKED_T_NOT_DOG)
        cases = (
            (np.nextafter(LOCKED_T_UNCERTAIN, 0), 'DOG'),
            (LOCKED_T_UNCERTAIN, 'UNCERTAIN'),
            (np.nextafter(LOCKED_T_NOT_DOG, 0), 'UNCERTAIN'),
            (LOCKED_T_NOT_DOG, 'NOT_DOG'),
        )
        for score, expected in cases:
            with self.subTest(score=score):
                self.assertEqual(decide_tri_state(score), expected)

    def test_runtime_centroid_contract(self):
        self.assertEqual(app.predictor.centroids.shape, EXPECTED_CENTROID_SHAPE)
        np.testing.assert_allclose(
            np.linalg.norm(app.predictor.centroids, axis=1),
            np.ones(EXPECTED_CENTROID_SHAPE[0]),
            atol=1e-4,
        )

    def test_predictor_loads_once_and_reuses_it(self):
        probability = np.array([[.6, .2, .1, .06, .04]], dtype=np.float32)
        model = Mock(return_value=probability)
        loader = Mock(return_value=model)
        policy = {'uncertain': 0.55, 'not_dog': 0.8, 'breed_confidence': 0.45}
        predictor = Predictor(model_path='placeholder.keras', model_loader=loader, class_names=['a','b','c','d','e'], config={'auto_build_logits':False,'auto_build_embedding':False}, thresholds=policy)
        for _ in range(2):
            result = predictor.predict_batch(np.zeros((1,224,224,3), np.float32))
            self.assertEqual(result['decision'], 'DOG')
            self.assertEqual(len({item['breed'] for item in result['top5']}),5)
        self.assertEqual(loader.call_count,1)
        self.assertEqual(model.call_count,2)

    def test_energy_uses_exact_logits(self):
        batch = np.zeros((1,224,224,3), dtype=np.float32)
        logits = build_logits_model(app.model)(batch, training=False).numpy()
        probabilities = app.model(batch, training=False).numpy()
        np.testing.assert_allclose(tf.nn.softmax(logits).numpy(), probabilities, atol=1e-6)
        expected = -tf.reduce_logsumexp(logits.astype(np.float64),axis=-1).numpy()
        np.testing.assert_allclose(energy_score(logits,1), expected, atol=1e-10)
        self.assertFalse(np.allclose(energy_score(logits),energy_score(probabilities)))

    def test_entropy_zero_probability_and_mahalanobis_nonfinite(self):
        self.assertEqual(predictive_entropy(np.array([1.,0.])),0)
        self.assertTrue(np.isfinite(mahalanobis_distance([[1.,2.]], [0.,0.], np.zeros((2,2)))).all())
        for bad in (float('nan'),float('inf')):
            with self.assertRaises(ValueError):
                mahalanobis_distance([[1.,bad]], [0.,0.], np.eye(2))

    def test_bad_and_missing_centroid_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)/'centroids.npy'
            p.write_bytes(b'not numpy')
            meta = app.BASE_DIR/'outputs/calibration/class_centroids.json'
            with self.assertRaises(ValueError):
                load_centroid_artifacts(p, meta, app.CLASS_NAMES_PATH)
            with self.assertRaises(FileNotFoundError):
                load_centroid_artifacts(p.with_name('missing.npy'),meta,app.CLASS_NAMES_PATH)


if __name__ == '__main__':
    unittest.main()
