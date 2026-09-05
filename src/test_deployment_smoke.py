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
        """22. Startup synthetic smoke test without dataset."""
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
        """23. Test GET / returns 200 and renders HTML."""
        res = self.client.get("/")
        self.assertEqual(res.status_code, 200)
        self.assertIn(b"Dog Breed", res.data)

    def test_03_flask_get_health(self):
        """23. Test GET /health returns 200 and valid JSON."""
        res = self.client.get("/health")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data.get("status"), "ok")
        self.assertTrue(data.get("model_loaded"))
        self.assertEqual(data.get("num_classes"), 120)

    def test_04_flask_post_predict_synthetic_jpeg(self):
        """23. Test POST /predict with synthetic in-memory JPEG returns 200."""
        img = Image.new("RGB", (300, 300), color=(120, 150, 180))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        buf.seek(0)
        
        res = self.client.post(
            "/predict",
            data={"image": (buf, "synthetic_dog.jpg")},
            content_type="multipart/form-data"
        )
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIn("prediction", data)
        self.assertIn("top5", data)
        self.assertEqual(len(data["top5"]), 5)
        self.assertEqual(data["prediction"]["index"], data["top5"][0]["index"])

if __name__ == "__main__":
    unittest.main(verbosity=2)
