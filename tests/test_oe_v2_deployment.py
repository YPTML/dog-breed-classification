"""Comprehensive Deployment Integration & Integrity Tests for OE V2.

Verifies:
A. Model integrity & SHA256
B. Threshold boundaries (0.343357, 0.343358, 0.457512, 0.457513)
C. API response contracts for DOG, UNCERTAIN, and NOT_DOG
D. Exactly 120 output classes
E. Centroids shape (120, 1280) and SHA256
F. Production baseline model remains unchanged
G. Live Flask endpoints: GET /, GET /health, POST /predict error handling
"""

import hashlib
import io
import json
from pathlib import Path
import unittest

import numpy as np
from PIL import Image

import app
from src.inference.oe_v2_predictor import (
    OEV2Predictor,
    decide_tri_state,
    compute_file_sha256,
    EXPECTED_MODEL_SHA256,
    EXPECTED_CENTROIDS_SHA256,
    EXPECTED_NUM_CLASSES,
    EXPECTED_EMBEDDING_DIM,
    EXPECTED_CENTROID_SHAPE,
    LOCKED_DETECTOR,
    LOCKED_T_UNCERTAIN,
    LOCKED_T_NOT_DOG,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASELINE_MODEL_PATH = PROJECT_ROOT / "outputs/deployment_ready/dog_breed_model.keras"
EXPECTED_BASELINE_SHA256 = "4695eae55c8464c1a6be3f147828f901259fb2c2541c23c2e36e8b4eb95d5f3d"

STAGING_DIR = PROJECT_ROOT / "outputs/deployment_oe_v2"
STAGED_MODEL_PATH = STAGING_DIR / "dog_breed_model_oe_v2.keras"
STAGED_CENTROIDS_PATH = STAGING_DIR / "class_centroids.npy"
STAGED_CLASSES_PATH = STAGING_DIR / "class_names.json"
STAGED_MANIFEST_PATH = STAGING_DIR / "model_manifest.json"


def create_sample_png() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (32, 32), color=(100, 150, 200)).save(buffer, format="PNG")
    return buffer.getvalue()


class OEV2DeploymentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = app.app.test_client()

    # -----------------------------------------------------------------------
    # Requirement A: Model Integrity
    # -----------------------------------------------------------------------
    def test_a_model_integrity_and_hash(self):
        """Verify OE V2 model exists and matches locked SHA256 exactly."""
        self.assertTrue(STAGED_MODEL_PATH.exists(), f"Staged model missing: {STAGED_MODEL_PATH}")
        actual_sha = compute_file_sha256(STAGED_MODEL_PATH)
        self.assertEqual(
            actual_sha,
            EXPECTED_MODEL_SHA256,
            f"Staged model SHA256 mismatch: expected {EXPECTED_MODEL_SHA256}, got {actual_sha}",
        )

        # Check source checkpoint SHA256 as well
        source_model = PROJECT_ROOT / "outputs/models/oe_v2/stage_a_best.keras"
        self.assertTrue(source_model.exists())
        self.assertEqual(compute_file_sha256(source_model), EXPECTED_MODEL_SHA256)

    # -----------------------------------------------------------------------
    # Requirement B: Threshold Boundaries
    # -----------------------------------------------------------------------
    def test_b_threshold_boundaries(self):
        """Verify exact boundary behavior for locked tri-state thresholds."""
        # score = 0.343357 => DOG
        self.assertEqual(decide_tri_state(0.343357), "DOG")

        # score = 0.343358 => UNCERTAIN
        self.assertEqual(decide_tri_state(0.343358), "UNCERTAIN")

        # score = 0.457512 => UNCERTAIN
        self.assertEqual(decide_tri_state(0.457512), "UNCERTAIN")

        # score = 0.457513 => NOT_DOG
        self.assertEqual(decide_tri_state(0.457513), "NOT_DOG")

        # Edge cases
        self.assertEqual(decide_tri_state(0.0), "DOG")
        self.assertEqual(decide_tri_state(0.40), "UNCERTAIN")
        self.assertEqual(decide_tri_state(1.0), "NOT_DOG")

    # -----------------------------------------------------------------------
    # Requirement C: API Response Contracts
    # -----------------------------------------------------------------------
    def test_c_dog_response_contract(self):
        """Verify DOG response contract includes all required fields."""
        raw_res = {
            "status": "DOG",
            "final_status": "DOG",
            "breed": "golden_retriever",
            "confidence": 0.985,
            "top5": [
                {"rank": 1, "breed": "golden_retriever", "confidence": 0.985, "index": 45},
                {"rank": 2, "breed": "labrador_retriever", "confidence": 0.010, "index": 56},
            ],
            "probability_margin": 0.975,
            "ood_score": 0.2104,
            "T_uncertain": 0.343358,
            "T_not_dog": 0.457513,
        }
        res = app._api_prediction_response(raw_res)
        self.assertEqual(res["status"], "DOG")
        self.assertEqual(res["final_status"], "DOG")
        self.assertEqual(res["breed"], "golden_retriever")
        self.assertAlmostEqual(res["confidence"], 0.985)
        self.assertAlmostEqual(res["probability_margin"], 0.975)
        self.assertAlmostEqual(res["ood_score"], 0.2104)
        self.assertAlmostEqual(res["T_uncertain"], 0.343358)
        self.assertAlmostEqual(res["T_not_dog"], 0.457513)
        self.assertEqual(len(res["top5"]), 2)
        self.assertEqual(res["top5"][0]["breed"], "golden_retriever")

    def test_c_uncertain_response_contract(self):
        """Verify UNCERTAIN response contract includes all required fields."""
        raw_res = {
            "status": "UNCERTAIN",
            "final_status": "UNCERTAIN",
            "breed": "eskimo_dog",
            "confidence": 0.398673,
            "top5": [
                {"rank": 1, "breed": "eskimo_dog", "confidence": 0.398673, "index": 43},
                {"rank": 2, "breed": "siberian_husky", "confidence": 0.210076, "index": 101},
            ],
            "probability_margin": 0.188597,
            "ood_score": 0.41695,
            "T_uncertain": 0.343358,
            "T_not_dog": 0.457513,
            "message": "The image is ambiguous or outside the reliable dog-breed region.",
        }
        res = app._api_prediction_response(raw_res)
        self.assertEqual(res["status"], "UNCERTAIN")
        self.assertEqual(res["final_status"], "UNCERTAIN")
        self.assertEqual(res["breed"], "eskimo_dog")
        self.assertAlmostEqual(res["confidence"], 0.398673)
        self.assertAlmostEqual(res["probability_margin"], 0.188597)
        self.assertAlmostEqual(res["ood_score"], 0.41695)
        self.assertAlmostEqual(res["T_uncertain"], 0.343358)
        self.assertAlmostEqual(res["T_not_dog"], 0.457513)
        self.assertEqual(len(res["top5"]), 2)
        self.assertEqual(res["message"], "The image is ambiguous or outside the reliable dog-breed region.")

    def test_c_not_dog_response_contract(self):
        """Verify NOT_DOG response contract includes all required fields."""
        raw_res = {
            "status": "NOT_DOG",
            "final_status": "NOT_DOG",
            "breed": "chihuahua",
            "confidence": 0.15,
            "top5": [
                {"rank": 1, "breed": "chihuahua", "confidence": 0.15, "index": 1},
                {"rank": 2, "breed": "pug", "confidence": 0.08, "index": 2},
            ],
            "probability_margin": 0.07,
            "ood_score": 0.620,
            "T_uncertain": 0.343358,
            "T_not_dog": 0.457513,
            "message": "No dog was confidently detected.",
        }
        res = app._api_prediction_response(raw_res)
        self.assertEqual(res["status"], "NOT_DOG")
        self.assertEqual(res["final_status"], "NOT_DOG")
        self.assertEqual(res["breed"], "chihuahua")
        self.assertAlmostEqual(res["confidence"], 0.15)
        self.assertAlmostEqual(res["probability_margin"], 0.07)
        self.assertAlmostEqual(res["ood_score"], 0.620)
        self.assertAlmostEqual(res["T_uncertain"], 0.343358)
        self.assertAlmostEqual(res["T_not_dog"], 0.457513)
        self.assertEqual(len(res["top5"]), 2)
        self.assertEqual(res["message"], "No dog was confidently detected.")

    # -----------------------------------------------------------------------
    # Requirement D: Exactly 120 Output Classes
    # -----------------------------------------------------------------------
    def test_d_output_classes_count(self):
        """Verify model output shape is exactly (None, 120) and class_names count is 120."""
        self.assertEqual(app.model.output_shape, (None, EXPECTED_NUM_CLASSES))
        self.assertEqual(len(app.class_names), EXPECTED_NUM_CLASSES)
        self.assertEqual(set(app.class_names.keys()), set(range(EXPECTED_NUM_CLASSES)))

    # -----------------------------------------------------------------------
    # Requirement E: Centroids Shape and SHA256
    # -----------------------------------------------------------------------
    def test_e_centroids_shape_and_hash(self):
        """Verify class centroids shape is (120, 1280) and matches locked SHA256."""
        self.assertTrue(STAGED_CENTROIDS_PATH.exists())
        self.assertEqual(compute_file_sha256(STAGED_CENTROIDS_PATH), EXPECTED_CENTROIDS_SHA256)

        centroids = np.load(STAGED_CENTROIDS_PATH)
        self.assertEqual(centroids.shape, EXPECTED_CENTROID_SHAPE)
        self.assertEqual(centroids.dtype, np.float32)

    # -----------------------------------------------------------------------
    # Requirement F: Production Baseline Remains Unchanged
    # -----------------------------------------------------------------------
    def test_f_production_baseline_unchanged(self):
        """Verify the legacy production model was NOT overwritten and hash is unchanged."""
        self.assertTrue(
            BASELINE_MODEL_PATH.exists(),
            f"Production baseline model missing: {BASELINE_MODEL_PATH}",
        )
        actual_baseline_sha = compute_file_sha256(BASELINE_MODEL_PATH)
        self.assertEqual(
            actual_baseline_sha,
            EXPECTED_BASELINE_SHA256,
            "CRITICAL: Production baseline model has been modified or overwritten!",
        )

    # -----------------------------------------------------------------------
    # Requirement G: Live Flask Smoke Tests
    # -----------------------------------------------------------------------
    def test_g_get_index_200(self):
        """GET / returns 200."""
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)

    def test_g_get_health_200_and_payload(self):
        """GET /health returns 200 with required metadata."""
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data["status"], "ok")
        self.assertTrue(data["model_loaded"])
        self.assertTrue(data["ood_detector_loaded"])
        self.assertEqual(data["num_classes"], 120)
        self.assertEqual(data["detector"], "cosine")
        self.assertEqual(data["model_version"], "oe_v2")

    def test_g_predict_missing_file_400(self):
        """POST /predict without 'image' field returns 400."""
        response = self.client.post("/predict", data={}, content_type="multipart/form-data")
        self.assertEqual(response.status_code, 400)
        data = response.get_json()
        self.assertIn("error", data)

    def test_g_predict_empty_file_400(self):
        """POST /predict with empty file returns 400."""
        response = self.client.post(
            "/predict",
            data={"image": (io.BytesIO(b""), "empty.jpg")},
            content_type="multipart/form-data",
        )
        self.assertEqual(response.status_code, 400)
        data = response.get_json()
        self.assertIn("error", data)

    def test_g_predict_corrupt_file_400(self):
        """POST /predict with invalid non-image bytes returns 400."""
        response = self.client.post(
            "/predict",
            data={"image": (io.BytesIO(b"this is not an image file content"), "corrupt.jpg")},
            content_type="multipart/form-data",
        )
        self.assertEqual(response.status_code, 400)
        data = response.get_json()
        self.assertIn("error", data)

    def test_g_predict_real_dog_snoopy_garfield(self):
        """POST /predict on normal dog, Snoopy, and Garfield."""
        dog_path = PROJECT_ROOT / "data/train/fa79fd049300f5082393fa99a142cd7f.jpg"
        snoopy_path = PROJECT_ROOT / "data/ood_oe_v2/hard_test/cartoon_dog/snoopy/image_0001.jpg"
        garfield_path = PROJECT_ROOT / "data/ood_oe_v2/hard_test/cartoon_animal/garfield/image_0001.jpg"

        if dog_path.exists():
            with open(dog_path, "rb") as f:
                res = self.client.post(
                    "/predict",
                    data={"image": (io.BytesIO(f.read()), dog_path.name)},
                    content_type="multipart/form-data",
                )
            self.assertEqual(res.status_code, 200)
            data = res.get_json()
            self.assertEqual(data["status"], "DOG")
            self.assertEqual(data["breed"], "sealyham_terrier")
            self.assertGreater(data["confidence"], 0.90)
            self.assertEqual(len(data["top5"]), 5)
            self.assertLess(data["ood_score"], LOCKED_T_UNCERTAIN)

        if snoopy_path.exists():
            with open(snoopy_path, "rb") as f:
                res = self.client.post(
                    "/predict",
                    data={"image": (io.BytesIO(f.read()), snoopy_path.name)},
                    content_type="multipart/form-data",
                )
            self.assertEqual(res.status_code, 200)
            data = res.get_json()
            self.assertIn(data["status"], ("UNCERTAIN", "NOT_DOG"))
            self.assertIsNotNone(data["breed"])
            self.assertGreater(data["confidence"], 0.0)
            self.assertEqual(len(data["top5"]), 5)
            self.assertIn("probability_margin", data)
            self.assertIn("T_uncertain", data)
            self.assertIn("T_not_dog", data)
            self.assertGreaterEqual(data["ood_score"], LOCKED_T_UNCERTAIN)

        if garfield_path.exists():
            with open(garfield_path, "rb") as f:
                res = self.client.post(
                    "/predict",
                    data={"image": (io.BytesIO(f.read()), garfield_path.name)},
                    content_type="multipart/form-data",
                )
            self.assertEqual(res.status_code, 200)
            data = res.get_json()
            self.assertIn(data["status"], ("UNCERTAIN", "NOT_DOG"))
            self.assertIsNotNone(data["breed"])
            self.assertGreater(data["confidence"], 0.0)
            self.assertEqual(len(data["top5"]), 5)
            self.assertIn("probability_margin", data)
            self.assertIn("T_uncertain", data)
            self.assertIn("T_not_dog", data)
            self.assertGreaterEqual(data["ood_score"], LOCKED_T_UNCERTAIN)

    # -----------------------------------------------------------------------
    # Requirement H: Staging Bundle Manifest
    # -----------------------------------------------------------------------
    def test_h_manifest_integrity(self):
        """Verify model_manifest.json contents in staging bundle."""
        self.assertTrue(STAGED_MANIFEST_PATH.exists())
        with open(STAGED_MANIFEST_PATH, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        self.assertEqual(manifest["model_version"], "oe_v2")
        self.assertEqual(manifest["model_sha256"], EXPECTED_MODEL_SHA256)
        self.assertEqual(manifest["centroid_sha256"], EXPECTED_CENTROIDS_SHA256)
        self.assertEqual(manifest["class_count"], 120)
        self.assertEqual(manifest["embedding_dimension"], 1280)
        self.assertEqual(manifest["detector"], "cosine")
        self.assertAlmostEqual(manifest["T_uncertain"], 0.343358)
        self.assertAlmostEqual(manifest["T_not_dog"], 0.457513)


if __name__ == "__main__":
    unittest.main(verbosity=2)
