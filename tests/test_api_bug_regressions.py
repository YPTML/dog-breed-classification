"""API error classification regressions with real uploads and injected failures."""

import io
import unittest
from unittest.mock import patch

from PIL import Image, PngImagePlugin

import app


class PredictionErrorRegressions(unittest.TestCase):
    def setUp(self):
        self.client = app.app.test_client()
        image = io.BytesIO()
        Image.new("RGB", (8, 8), (80, 120, 160)).save(image, format="PNG")
        self.image_bytes = image.getvalue()

    def post(self, payload=None):
        return self.client.post(
            "/predict",
            data={"image": (io.BytesIO(self.image_bytes if payload is None else payload), "image.png")},
        )

    def test_model_errors_return_500_and_log_tracebacks(self):
        for error_type in (ValueError, OSError, SyntaxError, RuntimeError):
            with self.subTest(error_type=error_type.__name__):
                error = error_type("private model execution details")
                with patch.object(app.predictor, "inference_model", side_effect=error), \
                        self.assertLogs(app.logger, level="ERROR") as captured:
                    response = self.post()
                self.assertEqual(response.status_code, 500)
                self.assertEqual(response.get_json(), {"error": "Inference failed"})
                self.assertTrue(any(record.exc_info for record in captured.records))
                self.assertNotIn(str(error), response.get_data(as_text=True))

    def test_corrupt_upload_stays_400_without_running_model(self):
        with patch.object(app.predictor, "inference_model") as model:
            response = self.post(b"not an image")
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.get_json(), {"error": "Invalid or unsupported image"})
        model.assert_not_called()

    def test_pillow_decompression_warning_and_error_stay_400(self):
        for size in ((11, 10), (21, 10)):
            image = io.BytesIO()
            Image.new("RGB", size).save(image, format="PNG")
            with self.subTest(size=size), patch.object(Image, "MAX_IMAGE_PIXELS", 100), \
                    patch.object(app.predictor, "inference_model") as model:
                response = self.post(image.getvalue())
            self.assertEqual(response.status_code, 400)
            model.assert_not_called()

    def test_excessive_compressed_png_metadata_stays_400(self):
        metadata = PngImagePlugin.PngInfo()
        metadata.add_text("comment", "x" * 1024, zip=True)
        image = io.BytesIO()
        Image.new("RGB", (8, 8)).save(image, format="PNG", pnginfo=metadata)
        with patch.object(PngImagePlugin, "MAX_TEXT_CHUNK", 100), \
                patch.object(app.predictor, "inference_model") as model:
            response = self.post(image.getvalue())
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.get_json(), {"error": "Invalid or unsupported image"})
        model.assert_not_called()


if __name__ == "__main__":
    unittest.main()
