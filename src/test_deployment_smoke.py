import io
import unittest

import numpy as np
from PIL import Image

import app


class TestDeploymentSmoke(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = app.app.test_client()

    def test_01_synthetic_prediction_invariants(self):
        """Startup synthetic smoke test without touching the dataset."""
        dummy_input = np.random.uniform(0, 255, size=(1, 224, 224, 3)).astype(np.float32)
        top1, top5 = app.predict_image(dummy_input)

        self.assertIn("index", top1)
        self.assertIn("breed", top1)
        self.assertIn("confidence", top1)
        self.assertTrue(0 <= top1["index"] < 120)
        self.assertIsInstance(top1["breed"], str)
        self.assertFalse(np.isnan(top1["confidence"]))
        self.assertFalse(np.isinf(top1["confidence"]))

        self.assertEqual(len(top5), 5)
        for rank, item in enumerate(top5):
            self.assertIn("index", item)
            self.assertIn("breed", item)
            self.assertIn("confidence", item)
            self.assertFalse(np.isnan(item["confidence"]))
            self.assertFalse(np.isinf(item["confidence"]))
            if rank > 0:
                self.assertGreaterEqual(top5[rank - 1]["confidence"], item["confidence"])

    def test_02_flask_get_root(self):
        """GET / returns 200 and renders HTML."""
        res = self.client.get("/")
        self.assertEqual(res.status_code, 200)
        self.assertIn(b"Dog Breed", res.data)

    def test_03_flask_get_health(self):
        """GET /health returns 200 and validates all startup artifacts."""
        res = self.client.get("/health")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data.get("status"), "ok")
        self.assertTrue(data.get("model_loaded"))
        self.assertEqual(data.get("num_classes"), 120)
        self.assertTrue(data.get("ood_detector_loaded"))
        self.assertEqual(data.get("detector"), "cosine")
        self.assertEqual(data.get("model_version"), "oe_v2")

    def test_04_required_artifacts_and_relative_paths(self):
        """All runtime artifacts resolve from the application root."""
        required = (
            app.MODEL_PATH,
            app.CLASS_NAMES_PATH,
            app.CENTROIDS_PATH,
        )
        for path in required:
            with self.subTest(path=path.name):
                self.assertTrue(path.is_file(), path)
                self.assertTrue(path.resolve().is_relative_to(app.BASE_DIR.resolve()))

    def test_05_flask_post_predict_synthetic_jpeg(self):
        """POST /predict returns the documented three-state response."""
        img = Image.new("RGB", (300, 300), color=(120, 150, 180))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        buf.seek(0)

        res = self.client.post(
            "/predict",
            data={"image": (buf, "synthetic_dog.jpg")},
            content_type="multipart/form-data",
        )
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIn(data["status"], {"DOG", "UNCERTAIN", "NOT_DOG"})
        self.assertEqual(data["status"], data["decision"])
        self.assertIn("prediction", data)
        self.assertIn("top5", data)
        self.assertEqual(len(data["top5"]), 5)
        self.assertEqual(data["prediction"]["breed"], data["top5"][0]["breed"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
