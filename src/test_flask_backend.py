import io
import sys
import unittest
import numpy as np
from PIL import Image
from tests.response_assertions import assert_three_state_response

from app import (
    app,
    preprocess_uploaded_image,
    predict_image,
    class_names,
    model,
)


class TestFlaskBackend(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = app.test_client()

    def test_01_health_endpoint(self):
        """Test GET /health returns 200 and expected metadata."""
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data.get("status"), "ok")
        self.assertIs(data.get("model_loaded"), True)
        self.assertEqual(data.get("model_version"), "oe_v2")
        self.assertEqual(data.get("detector"), "cosine")
        self.assertEqual(data.get("num_classes"), 120)

    def test_02_predict_missing_image_field(self):
        """Test POST /predict without image form field returns 400."""
        response = self.client.post("/predict")
        self.assertEqual(response.status_code, 400)
        data = response.get_json()
        self.assertEqual(data.get("error"), "Missing image file")

    def test_03_predict_empty_file(self):
        """Test POST /predict with empty bytes / empty filename returns 400."""
        # Empty filename
        response1 = self.client.post(
            "/predict",
            data={"image": (io.BytesIO(b""), "")},
            content_type="multipart/form-data"
        )
        self.assertEqual(response1.status_code, 400)
        self.assertEqual(response1.get_json().get("error"), "Empty image file")

        # Non-empty filename but zero bytes
        response2 = self.client.post(
            "/predict",
            data={"image": (io.BytesIO(b""), "empty.jpg")},
            content_type="multipart/form-data"
        )
        self.assertEqual(response2.status_code, 400)
        self.assertEqual(response2.get_json().get("error"), "Empty image file")

    def test_04_predict_invalid_bytes(self):
        """Test POST /predict with invalid non-image payload returns 400."""
        corrupted_bytes = b"NOT_AN_IMAGE_RANDOM_TEXT_DATA_FOR_TESTING"
        response = self.client.post(
            "/predict",
            data={"image": (io.BytesIO(corrupted_bytes), "corrupted.jpg")},
            content_type="multipart/form-data"
        )
        self.assertEqual(response.status_code, 400)
        data = response.get_json()
        self.assertEqual(data.get("error"), "Invalid or unsupported image")

    def test_05_predict_valid_synthetic_jpeg(self):
        """Synthetic JPEG obeys the three-state API without forcing a breed."""
        # Generate synthetic 300x300 RGB image
        np.random.seed(42)
        raw_pixels = np.random.randint(0, 256, size=(300, 300, 3), dtype=np.uint8)
        img = Image.fromarray(raw_pixels, mode="RGB")
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        buf.seek(0)

        response = self.client.post(
            "/predict",
            data={"image": (buf, "synthetic.jpg")},
            content_type="multipart/form-data"
        )
        self.assertEqual(response.status_code, 200)
        data = response.get_json()

        assert_three_state_response(self, data, class_names)

    def test_06_predict_grayscale_image(self):
        """Test POST /predict with grayscale image converts to RGB and returns 200."""
        gray_pixels = np.random.randint(0, 256, size=(150, 150), dtype=np.uint8)
        img = Image.fromarray(gray_pixels, mode="L")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)

        response = self.client.post(
            "/predict",
            data={"image": (buf, "gray.png")},
            content_type="multipart/form-data"
        )
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        assert_three_state_response(self, data, class_names)

    def test_07_predict_rgba_png(self):
        """Test POST /predict with RGBA PNG converts to RGB and returns 200."""
        rgba_pixels = np.random.randint(0, 256, size=(200, 200, 4), dtype=np.uint8)
        img = Image.fromarray(rgba_pixels, mode="RGBA")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)

        response = self.client.post(
            "/predict",
            data={"image": (buf, "transparent.png")},
            content_type="multipart/form-data"
        )
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        assert_three_state_response(self, data, class_names)

    def test_08_preprocessing_invariants(self):
        """Verify preprocessing returns shape (1, 224, 224, 3), dtype float32, range [0, 255]."""
        raw_pixels = np.full((100, 100, 3), fill_value=200, dtype=np.uint8)
        img = Image.fromarray(raw_pixels, mode="RGB")
        buf = io.BytesIO()
        img.save(buf, format="JPEG")

        processed = preprocess_uploaded_image(buf.getvalue())
        self.assertEqual(processed.shape, (1, 224, 224, 3))
        self.assertEqual(processed.dtype, np.float32)
        self.assertGreaterEqual(float(np.min(processed)), 0.0)
        self.assertLessEqual(float(np.max(processed)), 255.0)
        # Verify pixels are NOT normalized to [0, 1]
        self.assertGreater(float(np.max(processed)), 1.0)

    def test_09_file_too_large_rejection(self):
        """Verify uploads exceeding 10 MB trigger HTTP 413 error handler."""
        # 11 MB of dummy data
        oversized_data = b"X" * (11 * 1024 * 1024)
        response = self.client.post(
            "/predict",
            data={"image": (io.BytesIO(oversized_data), "huge.jpg")},
            content_type="multipart/form-data"
        )
        self.assertEqual(response.status_code, 413)
        data = response.get_json()
        self.assertEqual(data.get("error"), "File too large")


def run_tests():
    print("=" * 70)
    print("RUNNING FLASK BACKEND TEST SUITE (Task 7.1)")
    print("=" * 70)
    suite = unittest.TestLoader().loadTestsFromTestCase(TestFlaskBackend)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    if not result.wasSuccessful():
        print("\nTEST SUITE FAILED!")
        sys.exit(1)
    print("\nALL TESTS PASSED SUCCESSFULLY!")


if __name__ == "__main__":
    run_tests()
