"""Regression coverage for inference and offline image-loader failures."""

import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import tensorflow as tf
from PIL import Image

from src.inference.oe_v2_predictor import (
    EXPECTED_CENTROID_SHAPE,
    EXPECTED_CENTROIDS_SHA256,
    EXPECTED_EMBEDDING_DIM,
    EXPECTED_MODEL_SHA256,
    EXPECTED_NUM_CLASSES,
    LOCKED_T_NOT_DOG,
    LOCKED_T_UNCERTAIN,
    OEV2Predictor,
    decide_tri_state,
)
from src.inference.preprocessing import InvalidImageError, preprocess_image
from src.ood import benchmark, embedding_distance, energy, hard_evaluation, msp
from src.ood.image_loading import _decode_webp, preprocess_ood_image


class ImageFailureTests(unittest.TestCase):
    def test_invalid_image_inputs_use_specific_error(self):
        for value in (b"", b"not an image", "not bytes", None):
            with self.subTest(value=value), self.assertRaises(InvalidImageError):
                preprocess_image(value)

    def test_dimensions_and_pillow_bombs_use_specific_error(self):
        buffer = io.BytesIO()
        Image.new("RGB", (8, 8)).save(buffer, format="PNG")
        with self.assertRaises(InvalidImageError):
            preprocess_image(buffer.getvalue(), max_image_pixels=16)
        for limit in (40, 20):
            with self.subTest(limit=limit), patch.object(Image, "MAX_IMAGE_PIXELS", limit):
                with self.assertRaises(InvalidImageError):
                    preprocess_image(buffer.getvalue())

    def test_tensorflow_decode_errors_use_specific_error(self):
        buffer = io.BytesIO()
        Image.new("RGB", (8, 8)).save(buffer, format="PNG")
        with patch.object(
            tf.image, "decode_png", side_effect=tf.errors.InvalidArgumentError(None, None, "bad pixels")
        ):
            with self.assertRaises(InvalidImageError):
                preprocess_image(buffer.getvalue())


class OfflineImageLoaderTests(unittest.TestCase):
    def test_webp_decoder_preserves_trailing_nul_bytes(self):
        payload = (Path(__file__).parent / "fixtures/shiba_test.webp").read_bytes()
        self.assertTrue(payload.endswith(b"\x00"))
        with Image.open(io.BytesIO(payload)) as source:
            expected = np.asarray(source.convert("RGB"))
        for dtype in (None, object):
            with self.subTest(dtype=dtype):
                np.testing.assert_array_equal(_decode_webp(np.asarray(payload, dtype=dtype)), expected)

    def test_webp_loads_in_all_offline_dataset_builders(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "sample.webp"
            pixels = np.random.default_rng(2).integers(0, 256, (19, 31, 3), dtype=np.uint8)
            Image.fromarray(pixels).save(path, format="WEBP", lossless=True)
            expected = tf.image.resize(pixels, (224, 224)).numpy()[None]
            builders = {
                "msp": lambda: msp._build_ood_dataset([path], 1),
                "energy": lambda: energy._ood_calibration_dataset(root, 1)[0],
                "embedding": lambda: embedding_distance._ood_calibration_dataset(root, 1)[0],
                "benchmark": lambda: benchmark._ood_dataset(root, 1)[0],
                "hard_evaluation": lambda: hard_evaluation._image_dataset(tf, [path], 1),
                # Entropy maps this same common helper over its dataset.
                "entropy_decoder": lambda: tf.data.Dataset.from_tensor_slices([str(path)]).map(
                    preprocess_ood_image
                ).batch(1),
            }
            for name, build in builders.items():
                with self.subTest(loader=name):
                    actual = next(iter(build())).numpy()
                    self.assertEqual(actual.dtype, np.float32)
                    np.testing.assert_array_equal(actual, expected)

    def test_native_formats_keep_existing_tensorflow_pixels(self):
        with tempfile.TemporaryDirectory() as directory:
            pixels = np.random.default_rng(3).integers(0, 256, (27, 39, 3), dtype=np.uint8)
            for fmt in ("JPEG", "PNG", "BMP"):
                with self.subTest(format=fmt):
                    path = Path(directory) / ("image." + fmt.lower())
                    Image.fromarray(pixels).save(path, format=fmt)
                    expected = tf.image.resize(
                        tf.io.decode_image(path.read_bytes(), channels=3, expand_animations=False),
                        (224, 224),
                    ).numpy()
                    np.testing.assert_array_equal(preprocess_ood_image(path).numpy(), expected)

    def test_webp_signature_is_used_even_with_jpeg_extension(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "webp-content.jpg"
            Image.new("RGB", (5, 9), (24, 48, 96)).save(path, format="WEBP", lossless=True)
            actual = preprocess_ood_image(path).numpy()
            np.testing.assert_array_equal(actual, np.broadcast_to([24, 48, 96], actual.shape))


class OEV2OutputValidationTests(unittest.TestCase):
    def setUp(self):
        self.predictor = OEV2Predictor.__new__(OEV2Predictor)
        self.predictor.class_names = {i: f"breed_{i}" for i in range(EXPECTED_NUM_CLASSES)}
        self.predictor.centroids = np.zeros(EXPECTED_CENTROID_SHAPE, dtype=np.float32)
        self.predictor.centroids[:, 0] = 1
        self.predictor.t_uncertain = LOCKED_T_UNCERTAIN
        self.predictor.t_not_dog = LOCKED_T_NOT_DOG
        self.probabilities = np.zeros((1, EXPECTED_NUM_CLASSES), dtype=np.float32)
        self.probabilities[0, 0] = 1
        self.features = np.zeros((1, EXPECTED_EMBEDDING_DIM), dtype=np.float32)
        self.features[0, 0] = 1
        self.image = np.zeros((1, 224, 224, 3), dtype=np.float32)

    def test_malformed_probabilities_fail_instead_of_becoming_predictions(self):
        invalid = (
            np.full_like(self.probabilities, 0.5),
            np.concatenate((np.array([[1.1, -0.1]], dtype=np.float32), self.probabilities[:, 2:]), axis=1),
            self.probabilities[0],
            np.repeat(self.probabilities, 2, axis=0),
        )
        for probabilities in invalid:
            with self.subTest(shape=probabilities.shape, sum=probabilities.sum()):
                self.predictor.inference_model = Mock(return_value=(probabilities, self.features))
                with self.assertRaises(RuntimeError):
                    self.predictor.predict(self.image)

    def test_invalid_features_and_similarity_fail(self):
        for features in (self.features[0], self.features[:, :-1], np.full_like(self.features, np.nan)):
            with self.subTest(shape=features.shape):
                self.predictor.inference_model = Mock(return_value=(self.probabilities, features))
                with self.assertRaises(RuntimeError):
                    self.predictor.predict(self.image)
        self.predictor.inference_model = Mock(return_value=(self.probabilities, self.features))
        self.predictor.centroids[0, 0] = np.nan
        with self.assertRaises(RuntimeError):
            self.predictor.predict(self.image)

    def test_valid_and_zero_feature_predictions_remain_supported(self):
        for value, status in ((1, "DOG"), (0, "NOT_DOG")):
            self.features[0, 0] = value
            self.predictor.inference_model = Mock(return_value=(self.probabilities, self.features))
            result = self.predictor.predict(self.image)
            self.assertEqual(result["status"], status)
            self.assertEqual(len(result["top5"]), 5)

    def test_nonfinite_scores_and_invalid_thresholds_fail(self):
        for score in (np.nan, np.inf, -np.inf):
            with self.subTest(score=score), self.assertRaises(ValueError):
                decide_tri_state(score)
        for uncertain, not_dog in ((0.8, 0.2), (0.4, 0.4), (np.nan, 0.5), (0.3, np.inf)):
            with self.subTest(thresholds=(uncertain, not_dog)), self.assertRaises(ValueError):
                decide_tri_state(0.1, uncertain, not_dog)

    def test_invalid_class_indices_and_centroids_fail_at_startup(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model_path = root / "model.keras"
            model_path.touch()
            names_path = root / "classes.json"
            centroids_path = root / "centroids.npy"
            np.save(centroids_path, self.predictor.centroids)
            names_path.write_text(json.dumps({i + 1: name for i, name in self.predictor.class_names.items()}))
            with patch.object(tf.keras.models, "load_model") as loader:
                with self.assertRaisesRegex(RuntimeError, "indices"):
                    OEV2Predictor(model_path, centroids_path, names_path, verify_integrity=False)
                loader.assert_not_called()
            names_path.write_text(json.dumps(self.predictor.class_names))
            for value in (np.nan, np.inf, 0):
                centroids = self.predictor.centroids.copy()
                centroids[0, :] = value
                np.save(centroids_path, centroids)
                with self.subTest(centroid=value), patch.object(tf.keras.models, "load_model") as loader:
                    with self.assertRaisesRegex(RuntimeError, "centroids"):
                        OEV2Predictor(model_path, centroids_path, names_path, verify_integrity=False)
                    loader.assert_not_called()

    def test_duplicate_and_tampered_class_names_fail_at_startup(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model_path = root / "model.keras"
            model_path.touch()
            names_path = root / "classes.json"
            centroids_path = root / "centroids.npy"
            np.save(centroids_path, self.predictor.centroids)

            duplicate_names = dict(self.predictor.class_names)
            duplicate_names[1] = duplicate_names[0]
            names_path.write_text(json.dumps(duplicate_names), encoding="utf-8")
            with patch.object(tf.keras.models, "load_model") as loader:
                with self.assertRaisesRegex(RuntimeError, "unique"):
                    OEV2Predictor(model_path, centroids_path, names_path, verify_integrity=False)
                loader.assert_not_called()

            swapped_names = dict(self.predictor.class_names)
            swapped_names[0], swapped_names[1] = swapped_names[1], swapped_names[0]
            names_path.write_text(json.dumps(swapped_names), encoding="utf-8")
            with patch(
                "src.inference.oe_v2_predictor.compute_file_sha256",
                side_effect=[EXPECTED_MODEL_SHA256, EXPECTED_CENTROIDS_SHA256],
            ), patch.object(tf.keras.models, "load_model") as loader:
                with self.assertRaisesRegex(RuntimeError, "Class names SHA256"):
                    OEV2Predictor(model_path, centroids_path, names_path, verify_integrity=True)
                loader.assert_not_called()


class OODPackageCliTests(unittest.TestCase):
    def test_module_clis_do_not_preimport_the_module_being_executed(self):
        for module in ("src.ood.msp", "src.ood.entropy", "src.ood.energy"):
            with self.subTest(module=module):
                result = subprocess.run(
                    [sys.executable, "-m", module, "--help"],
                    cwd=Path(__file__).resolve().parents[1],
                    capture_output=True,
                    text=True,
                    timeout=30,
                    check=False,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertNotIn("RuntimeWarning", result.stderr)


if __name__ == "__main__":
    unittest.main()
