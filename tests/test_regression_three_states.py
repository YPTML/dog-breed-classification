"""Automated Regression Tests for Three-State Prediction Pipeline.

Verifies:
1. DOG state contract and end-to-end Flask /predict behavior.
2. UNCERTAIN state contract and end-to-end Flask /predict behavior using the real Shiba Inu image.
3. NOT_DOG state contract and end-to-end Flask /predict behavior using a non-dog image.
4. Validation that all 8 mandatory fields are present for all three states:
   - top1 breed (breed, top1_breed)
   - top1 confidence (confidence, top1_confidence)
   - top5 predictions (top5, top5_predictions)
   - top1-top2 probability margin (probability_margin, margin)
   - raw OOD score (ood_score, raw_ood_score)
   - T_uncertain
   - T_not_dog
   - final status (status, final_status)
"""

import io
import unittest
from pathlib import Path

import app
from src.inference.oe_v2_predictor import (
    LOCKED_T_UNCERTAIN,
    LOCKED_T_NOT_DOG,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOG_IMG = PROJECT_ROOT / "data/train/fa79fd049300f5082393fa99a142cd7f.jpg"
SHIBA_IMG = PROJECT_ROOT / "tests/fixtures/shiba_test.webp"
if not SHIBA_IMG.exists():
    SHIBA_IMG = Path(r"C:\Users\PC\Desktop\shiba-inu-dog-in-the-snow.webp")
NOT_DOG_IMG = PROJECT_ROOT / "data/ood_oe_v2/hard_test/cartoon_animal/garfield/image_0001.jpg"


class ThreeStateRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = app.app.test_client()

    def _verify_all_eight_fields(self, data, expected_status):
        """Helper to assert all 8 required fields from Task 3 are present and valid."""
        # 1. final status
        self.assertIn("status", data)
        self.assertIn("final_status", data)
        self.assertEqual(data["status"], expected_status)
        self.assertEqual(data["final_status"], expected_status)

        # 2. top1 breed
        self.assertIn("breed", data)
        self.assertIn("top1_breed", data)
        self.assertIsInstance(data["breed"], str)
        self.assertGreater(len(data["breed"].strip()), 0)
        self.assertEqual(data["top1_breed"], data["breed"])

        # 3. top1 confidence
        self.assertIn("confidence", data)
        self.assertIn("top1_confidence", data)
        self.assertIsInstance(data["confidence"], (float, int))
        self.assertGreaterEqual(data["confidence"], 0.0)
        self.assertLessEqual(data["confidence"], 1.0)
        self.assertAlmostEqual(data["top1_confidence"], data["confidence"])

        # 4. top5 predictions
        self.assertIn("top5", data)
        self.assertIn("top5_predictions", data)
        self.assertIsInstance(data["top5"], list)
        self.assertEqual(len(data["top5"]), 5)
        for i, item in enumerate(data["top5"]):
            self.assertEqual(item["rank"], i + 1)
            self.assertIn("breed", item)
            self.assertIn("confidence", item)
            self.assertIn("index", item)
            if i > 0:
                self.assertLessEqual(item["confidence"], data["top5"][i - 1]["confidence"])

        # 5. top1-top2 probability margin
        self.assertIn("probability_margin", data)
        self.assertIn("margin", data)
        self.assertIsInstance(data["probability_margin"], (float, int))
        self.assertGreaterEqual(data["probability_margin"], 0.0)
        expected_margin = data["top5"][0]["confidence"] - data["top5"][1]["confidence"]
        self.assertAlmostEqual(data["probability_margin"], expected_margin, places=5)
        self.assertAlmostEqual(data["margin"], data["probability_margin"])

        # 6. raw OOD score
        self.assertIn("ood_score", data)
        self.assertIn("raw_ood_score", data)
        self.assertIsInstance(data["ood_score"], (float, int))
        self.assertAlmostEqual(data["raw_ood_score"], data["ood_score"])

        # 7. T_uncertain
        self.assertIn("T_uncertain", data)
        self.assertAlmostEqual(data["T_uncertain"], LOCKED_T_UNCERTAIN, places=5)

        # 8. T_not_dog
        self.assertIn("T_not_dog", data)
        self.assertAlmostEqual(data["T_not_dog"], LOCKED_T_NOT_DOG, places=5)

    def test_state_dog_complete_pipeline(self):
        """Test complete /predict endpoint on a real in-distribution dog image."""
        self.assertTrue(DOG_IMG.exists(), f"Dog image missing: {DOG_IMG}")
        with open(DOG_IMG, "rb") as f:
            response = self.client.post(
                "/predict",
                data={"image": (io.BytesIO(f.read()), DOG_IMG.name)},
                content_type="multipart/form-data",
            )
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self._verify_all_eight_fields(data, "DOG")
        self.assertLess(data["ood_score"], LOCKED_T_UNCERTAIN)
        self.assertEqual(data["breed"], "sealyham_terrier")
        self.assertGreater(data["confidence"], 0.90)

    def test_state_uncertain_real_shiba_image(self):
        """Test complete /predict endpoint on the clear Shiba Inu test image."""
        self.assertTrue(SHIBA_IMG.exists(), f"Shiba test image missing: {SHIBA_IMG}")
        with open(SHIBA_IMG, "rb") as f:
            response = self.client.post(
                "/predict",
                data={"image": (io.BytesIO(f.read()), SHIBA_IMG.name)},
                content_type="multipart/form-data",
            )
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self._verify_all_eight_fields(data, "UNCERTAIN")

        # Verify OOD score falls in the UNCERTAIN region
        self.assertGreaterEqual(data["ood_score"], LOCKED_T_UNCERTAIN)
        self.assertLess(data["ood_score"], LOCKED_T_NOT_DOG)
        self.assertAlmostEqual(data["ood_score"], 0.41695, places=4)

        # Verify underlying breed classification is preserved
        self.assertEqual(data["breed"], "eskimo_dog")
        self.assertAlmostEqual(data["confidence"], 0.39867, places=3)
        self.assertIn("message", data)

    def test_state_not_dog_complete_pipeline(self):
        """Test complete /predict endpoint on a non-dog cartoon image."""
        self.assertTrue(NOT_DOG_IMG.exists(), f"Non-dog image missing: {NOT_DOG_IMG}")
        with open(NOT_DOG_IMG, "rb") as f:
            response = self.client.post(
                "/predict",
                data={"image": (io.BytesIO(f.read()), NOT_DOG_IMG.name)},
                content_type="multipart/form-data",
            )
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self._verify_all_eight_fields(data, "NOT_DOG")

        # Verify OOD score falls in the NOT_DOG region
        self.assertGreaterEqual(data["ood_score"], LOCKED_T_NOT_DOG)
        self.assertIn("message", data)


if __name__ == "__main__":
    unittest.main(verbosity=2)
