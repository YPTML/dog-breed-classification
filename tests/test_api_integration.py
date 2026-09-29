"""Flask client coverage for the OE V2 three-state prediction contract."""

import io
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image

import app


def image_bytes() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (8, 8), color=(80, 120, 160)).save(buffer, format="PNG")
    return buffer.getvalue()


class FakePredictor:
    def __init__(self, result):
        self.result = result
        self.model = SimpleNamespace(name="test-model")
        self.received = None

    def predict(self, image):
        self.received = image
        return self.result


class FlaskApiIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = app.app.test_client()

    def post_with_result(self, result):
        result = dict(result)
        result.setdefault("model", "test-model")
        predictor = FakePredictor(result)
        with patch.object(app, "predictor", predictor):
            response = self.client.post(
                "/predict",
                data={"image": (io.BytesIO(image_bytes()), "image.png")},
                content_type="multipart/form-data",
            )
        self.assertEqual(predictor.received, image_bytes())
        return response

    def test_health_contract(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data["status"], "ok")
        self.assertTrue(data["model_loaded"])
        self.assertTrue(data["ood_detector_loaded"])
        self.assertEqual(data["num_classes"], 120)
        self.assertEqual(data["detector"], "cosine")
        self.assertEqual(data["model_version"], "oe_v2")

    def test_dog_response(self):
        response = self.post_with_result(
            {
                "status": "DOG",
                "final_status": "DOG",
                "breed": "beagle",
                "confidence": 0.91,
                "top5": [
                    {"rank": 1, "breed": "beagle", "confidence": 0.91, "index": 10},
                    {"rank": 2, "breed": "basset", "confidence": 0.05, "index": 11},
                ],
                "probability_margin": 0.86,
                "ood_score": 0.25,
                "T_uncertain": 0.343358,
                "T_not_dog": 0.457513,
            }
        )
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data["status"], "DOG")
        self.assertEqual(data["final_status"], "DOG")
        self.assertEqual(data["breed"], "beagle")
        self.assertAlmostEqual(data["confidence"], 0.91, places=2)
        self.assertEqual(len(data["top5"]), 2)
        self.assertEqual(data["top5"][0]["breed"], "beagle")
        self.assertAlmostEqual(data["probability_margin"], 0.86, places=2)
        self.assertAlmostEqual(data["ood_score"], 0.25, places=2)
        self.assertAlmostEqual(data["T_uncertain"], 0.343358, places=5)
        self.assertAlmostEqual(data["T_not_dog"], 0.457513, places=5)

    def test_not_dog_response(self):
        response = self.post_with_result(
            {
                "status": "NOT_DOG",
                "final_status": "NOT_DOG",
                "breed": "chihuahua",
                "confidence": 0.12,
                "top5": [
                    {"rank": 1, "breed": "chihuahua", "confidence": 0.12, "index": 1},
                    {"rank": 2, "breed": "pug", "confidence": 0.08, "index": 2},
                ],
                "probability_margin": 0.04,
                "ood_score": 0.55,
                "T_uncertain": 0.343358,
                "T_not_dog": 0.457513,
                "message": "No dog was confidently detected.",
            }
        )
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data["status"], "NOT_DOG")
        self.assertEqual(data["final_status"], "NOT_DOG")
        self.assertEqual(data["breed"], "chihuahua")
        self.assertAlmostEqual(data["confidence"], 0.12, places=2)
        self.assertEqual(len(data["top5"]), 2)
        self.assertAlmostEqual(data["probability_margin"], 0.04, places=2)
        self.assertAlmostEqual(data["ood_score"], 0.55, places=2)
        self.assertAlmostEqual(data["T_uncertain"], 0.343358, places=5)
        self.assertAlmostEqual(data["T_not_dog"], 0.457513, places=5)
        self.assertEqual(data["message"], "No dog was confidently detected.")

    def test_uncertain_response(self):
        response = self.post_with_result(
            {
                "status": "UNCERTAIN",
                "final_status": "UNCERTAIN",
                "breed": "eskimo_dog",
                "confidence": 0.398,
                "top5": [
                    {"rank": 1, "breed": "eskimo_dog", "confidence": 0.398, "index": 43},
                    {"rank": 2, "breed": "siberian_husky", "confidence": 0.210, "index": 101},
                ],
                "probability_margin": 0.188,
                "ood_score": 0.417,
                "T_uncertain": 0.343358,
                "T_not_dog": 0.457513,
                "message": "The image is ambiguous or outside the reliable dog-breed region.",
            }
        )
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data["status"], "UNCERTAIN")
        self.assertEqual(data["final_status"], "UNCERTAIN")
        self.assertEqual(data["breed"], "eskimo_dog")
        self.assertAlmostEqual(data["confidence"], 0.398, places=2)
        self.assertEqual(len(data["top5"]), 2)
        self.assertAlmostEqual(data["probability_margin"], 0.188, places=2)
        self.assertAlmostEqual(data["ood_score"], 0.417, places=2)
        self.assertAlmostEqual(data["T_uncertain"], 0.343358, places=5)
        self.assertAlmostEqual(data["T_not_dog"], 0.457513, places=5)
        self.assertEqual(data["message"], "The image is ambiguous or outside the reliable dog-breed region.")


if __name__ == "__main__":
    unittest.main(verbosity=2)
