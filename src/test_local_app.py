"""Local API and inference regressions using only synthetic, in-memory images.

Run with ``python -m src.test_local_app`` in the project's virtual environment.
The production model is loaded normally once; edge cases replace only the Python
model reference with a mock. No dataset is read and no model artifact is written.
"""

import io
import json
import unittest
from unittest.mock import Mock, patch

import numpy as np
from flask import request
from PIL import Image, PngImagePlugin

import app as backend


def image_bytes(image_format="PNG", mode="RGB", size=(16, 12), color=None):
    """Encode a synthetic image without touching the filesystem."""
    colors = {"RGB": (240, 80, 20), "RGBA": (240, 80, 20, 120), "L": 180, "P": 1}
    with Image.new(mode, size, colors[mode] if color is None else color) as image:
        if mode == "P":
            image.putpalette([0, 0, 0, 240, 80, 20] + [0] * (768 - 6))
        buffer = io.BytesIO()
        image.save(buffer, format=image_format)
        return buffer.getvalue()


def multipart_bytes(payload, filename="synthetic.png", mime="image/png"):
    """Build a request body directly so the test client cannot spool it to disk."""
    boundary = "local-regression-boundary"
    prefix = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="image"; filename="{filename}"\r\n'
        f"Content-Type: {mime}\r\n\r\n"
    ).encode("ascii")
    return prefix + payload + f"\r\n--{boundary}--\r\n".encode("ascii"), (
        f"multipart/form-data; boundary={boundary}"
    )


def model_returning(probabilities=None):
    """Return a callable model stand-in with NumPy-compatible probabilities."""
    if probabilities is None:
        probabilities = np.zeros((1, 120), dtype=np.float32)
        probabilities[0, [17, 4, 33, 82, 119]] = [0.5, 0.2, 0.15, 0.1, 0.05]
    model = Mock(return_value=probabilities)
    model.name = "mobilenet_v2_transfer"
    return model


class TestLocalApp(unittest.TestCase):
    def setUp(self):
        self.client = backend.app.test_client()

    def post_image(self, payload, filename="synthetic.png", mime="image/png", **kwargs):
        body, content_type = multipart_bytes(payload, filename, mime)
        return self.client.post("/predict", data=body, content_type=content_type, **kwargs)

    def assert_json_error(self, response, status, message=None):
        self.assertEqual(response.status_code, status)
        self.assertTrue(response.is_json)
        self.assertIsInstance(response.get_json().get("error"), str)
        if message is not None:
            self.assertEqual(response.get_json(), {"error": message})

    def test_mapping_matches_unchanged_source_order(self):
        with backend.CLASS_NAMES_PATH.open(encoding="utf-8") as mapping_file:
            raw = json.load(mapping_file)
        expected = dict(enumerate(raw)) if isinstance(raw, list) else {
            int(index): breed for index, breed in raw.items()
        }
        self.assertEqual(list(backend.class_names.items()), list(expected.items()))
        self.assertEqual(set(expected), set(range(120)))
        self.assertEqual(len(set(expected.values())), 120)

    def test_health_does_not_infer(self):
        model = model_returning()
        with patch.object(backend, "model", model):
            response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["num_classes"], 120)
        self.assertIs(response.get_json()["model_loaded"], True)
        model.assert_not_called()

    def test_frontend_routes_do_not_infer(self):
        model = model_returning()
        with patch.object(backend, "model", model):
            for path, content_type in (
                ("/", "text/html"),
                ("/static/style.css", "text/css"),
                ("/static/app.js", "javascript"),
            ):
                with self.subTest(path=path):
                    response = self.client.get(path)
                    self.assertEqual(response.status_code, 200)
                    self.assertIn(content_type, response.content_type)
                    response.close()
        model.assert_not_called()

    def test_missing_empty_and_wrong_field_never_infer(self):
        model = model_returning()
        with patch.object(backend, "model", model):
            response = self.client.post(
                "/predict", data={"other": (io.BytesIO(image_bytes()), "wrong.png")}
            )
            self.assert_json_error(response, 400, "Missing image file")
            for payload, filename in ((b"", "empty.png"), (image_bytes(), "")):
                with self.subTest(filename=filename):
                    self.assert_json_error(
                        self.post_image(payload, filename), 400, "Empty image file"
                    )
        model.assert_not_called()

    def test_supported_formats_modes_and_tiny_images(self):
        for image_format, mode, size in (
            ("JPEG", "RGB", (1, 1)),
            ("PNG", "RGB", (1, 1)),
            ("PNG", "L", (7, 11)),
            ("PNG", "RGBA", (7, 11)),
            ("PNG", "P", (7, 11)),
            ("WEBP", "RGB", (7, 11)),
            ("WEBP", "RGBA", (7, 11)),
        ):
            with self.subTest(image_format=image_format, mode=mode, size=size):
                array = backend.preprocess_uploaded_image(image_bytes(image_format, mode, size))
                self.assertEqual(array.shape, (1, 224, 224, 3))
                self.assertEqual(array.dtype, np.float32)
                self.assertTrue(np.isfinite(array).all())
                self.assertGreaterEqual(float(array.min()), 0)
                self.assertLessEqual(float(array.max()), 255)
                self.assertGreater(float(array.max()), 1)
                if mode == "L":
                    np.testing.assert_array_equal(array[..., 0], array[..., 1])
                    np.testing.assert_array_equal(array[..., 1], array[..., 2])

    def test_preprocessing_preserves_raw_pixel_scale(self):
        for value in (0, 128, 255):
            with self.subTest(value=value):
                array = backend.preprocess_uploaded_image(
                    image_bytes(color=(value, value, value), size=(1, 1))
                )
                np.testing.assert_array_equal(array, np.full((1, 224, 224, 3), value, np.float32))

    def test_valid_bytes_override_filename_and_mime(self):
        model = model_returning()
        with patch.object(backend, "model", model):
            for filename, mime in (
                ("photo.txt", "text/plain"),
                ("../../outside.exe", "application/octet-stream"),
                ("no_extension", "image/jpeg"),
            ):
                with self.subTest(filename=filename, mime=mime):
                    response = self.post_image(image_bytes(), filename, mime)
                    self.assertEqual(response.status_code, 200)
                    self.assertEqual(response.get_json()["prediction"]["index"], 17)
        self.assertEqual(model.call_count, 3)

    def test_unsupported_decoded_formats_rejected(self):
        model = model_returning()
        with patch.object(backend, "model", model):
            for image_format in ("GIF", "BMP", "TIFF"):
                with self.subTest(image_format=image_format):
                    self.assert_json_error(
                        self.post_image(image_bytes(image_format)),
                        400,
                        "Invalid or unsupported image",
                    )
        model.assert_not_called()

    def test_corrupt_and_truncated_images_rejected(self):
        jpeg = image_bytes("JPEG", size=(64, 64))
        png = image_bytes("PNG", size=(64, 64))
        corrupt_png = bytearray(png)
        idat = corrupt_png.index(b"IDAT")
        corrupt_png[idat + 4] ^= 0xFF
        cases = {
            "random": b"not an image\x00\xff",
            "jpeg_header_only": jpeg[:40],
            "jpeg_missing_end": jpeg[:-16],
            "png_header_only": png[:40],
            "png_missing_end": png[:-20],
            "png_corrupt_chunk": bytes(corrupt_png),
        }
        model = model_returning()
        with patch.object(backend, "model", model):
            for name, payload in cases.items():
                with self.subTest(case=name):
                    self.assert_json_error(
                        self.post_image(payload), 400, "Invalid or unsupported image"
                    )
        model.assert_not_called()

    def test_exif_orientation_applied_before_resize(self):
        pixels = np.zeros((16, 32, 3), np.uint8)
        pixels[:, :16, 0] = 255
        pixels[:, 16:, 2] = 255
        with Image.fromarray(pixels) as image:
            orientation = Image.Exif()
            orientation[274] = 6  # Rotate 90 degrees clockwise for display.
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=100, subsampling=0, exif=orientation)
        array = backend.preprocess_uploaded_image(buffer.getvalue())
        self.assertGreater(float(array[0, :20, :, 0].mean()), 240)
        self.assertLess(float(array[0, :20, :, 2].mean()), 10)
        self.assertGreater(float(array[0, -20:, :, 2].mean()), 240)
        self.assertLess(float(array[0, -20:, :, 0].mean()), 10)

    def test_dimension_cap_rejects_before_pixel_decode(self):
        payload = image_bytes(size=(11, 10))
        model = model_returning()
        with patch.object(backend, "MAX_IMAGE_PIXELS", 100), patch.object(
            PngImagePlugin.PngImageFile, "load", side_effect=AssertionError("full decode happened")
        ) as decode, patch.object(backend, "model", model):
            self.assert_json_error(
                self.post_image(payload), 400, "Invalid or unsupported image"
            )
        decode.assert_not_called()
        model.assert_not_called()

    def test_pillow_bomb_warning_and_error_rejected(self):
        # Lower the library limit only within this test; large arrays are unnecessary.
        payloads = [image_bytes(size=(11, 10)), image_bytes(size=(21, 10))]
        model = model_returning()
        with patch.object(Image, "MAX_IMAGE_PIXELS", 100), patch.object(
            backend, "MAX_IMAGE_PIXELS", 10000
        ), patch.object(backend, "model", model):
            for payload in payloads:
                with self.subTest(compressed_bytes=len(payload)):
                    self.assert_json_error(
                        self.post_image(payload), 400, "Invalid or unsupported image"
                    )
        model.assert_not_called()

    def test_inference_rejects_wrong_input_shape(self):
        model = model_returning()
        with patch.object(backend, "model", model):
            for shape in ((224, 224, 3), (2, 224, 224, 3), (1, 223, 224, 3), (1, 224, 224, 4)):
                with self.subTest(shape=shape), self.assertRaises((ValueError, RuntimeError, TypeError)):
                    backend.predict_image(np.zeros(shape, np.float32))
        model.assert_not_called()

    def test_inference_rejects_wrong_input_dtype(self):
        model = model_returning()
        with patch.object(backend, "model", model):
            for dtype in (np.float64, np.uint8, np.int32):
                with self.subTest(dtype=dtype), self.assertRaises((ValueError, RuntimeError, TypeError)):
                    backend.predict_image(np.zeros((1, 224, 224, 3), dtype))
        model.assert_not_called()

    def test_inference_rejects_invalid_input_pixels(self):
        model = model_returning()
        with patch.object(backend, "model", model):
            for value in (np.nan, np.inf, -np.inf, -1, 256):
                array = np.zeros((1, 224, 224, 3), np.float32)
                array[0, 0, 0, 0] = value
                with self.subTest(value=value), self.assertRaises((ValueError, RuntimeError, TypeError)):
                    backend.predict_image(array)
        model.assert_not_called()

    def test_malformed_output_shapes_return_generic_500(self):
        for shape in ((120,), (1, 119), (2, 120), (1, 120, 1), (0, 120)):
            probabilities = np.full(shape, 1 / 120, dtype=np.float32)
            with self.subTest(shape=shape), patch.object(
                backend, "model", model_returning(probabilities)
            ), self.assertLogs(backend.logger, level="ERROR"):
                self.assert_json_error(self.post_image(image_bytes()), 500, "Inference failed")

    def test_nonfinite_probabilities_return_generic_500(self):
        for value in (np.nan, np.inf, -np.inf):
            probabilities = np.full((1, 120), 1 / 120, dtype=np.float32)
            probabilities[0, 17] = value
            with self.subTest(value=value), patch.object(
                backend, "model", model_returning(probabilities)
            ), self.assertLogs(backend.logger, level="ERROR"):
                self.assert_json_error(self.post_image(image_bytes()), 500, "Inference failed")

    def test_invalid_probability_ranges_and_sums_return_generic_500(self):
        cases = {
            "negative_and_over_one": np.array([[-0.1, 1.1] + [0] * 118], np.float32),
            "all_zero": np.zeros((1, 120), np.float32),
            "sum_half": np.full((1, 120), 0.5 / 120, np.float32),
            "sum_two": np.full((1, 120), 2 / 120, np.float32),
        }
        for name, probabilities in cases.items():
            with self.subTest(case=name), patch.object(
                backend, "model", model_returning(probabilities)
            ), self.assertLogs(backend.logger, level="ERROR"):
                self.assert_json_error(self.post_image(image_bytes()), 500, "Inference failed")

    def test_valid_inference_contract_and_ranking(self):
        model = model_returning()
        array = np.full((1, 224, 224, 3), 200, np.float32)
        with patch.object(backend, "model", model):
            top1, top5 = backend.predict_image(array)
        model.assert_called_once()
        self.assertIs(model.call_args.args[0], array)
        self.assertEqual(model.call_args.kwargs, {"training": False})
        self.assertEqual(top1, top5[0])
        self.assertEqual([item["index"] for item in top5], [17, 4, 33, 82, 119])
        self.assertEqual(len({item["index"] for item in top5}), 5)
        self.assertEqual([item["confidence"] for item in top5], sorted(
            [item["confidence"] for item in top5], reverse=True
        ))
        for item in top5:
            self.assertEqual(item["breed"], backend.class_names[item["index"]])
            self.assertGreaterEqual(item["confidence"], 0)
            self.assertLessEqual(item["confidence"], 1)

    def test_probability_ties_have_consistent_top1_and_top5(self):
        probabilities = np.full((1, 120), 1 / 120, np.float32)
        with patch.object(backend, "model", model_returning(probabilities)):
            top1, top5 = backend.predict_image(np.zeros((1, 224, 224, 3), np.float32))
        self.assertEqual(top1, top5[0])
        self.assertEqual([item["index"] for item in top5], [0, 1, 2, 3, 4])

    def test_internal_exceptions_do_not_leak_to_client(self):
        secret = "private E:\\sensitive\\model.keras environment-details"
        model = model_returning()
        model.side_effect = RuntimeError(secret)
        with patch.object(backend, "model", model), self.assertLogs(backend.logger, level="ERROR"):
            response = self.post_image(image_bytes())
        self.assert_json_error(response, 500, "Inference failed")
        self.assertNotIn(secret, response.get_data(as_text=True))
        # Exercise the global handler through a real route outside inference.
        with patch.object(backend, "render_template", side_effect=RuntimeError(secret)), patch.dict(
            backend.app.config, {"TESTING": False, "PROPAGATE_EXCEPTIONS": False}
        ), self.assertLogs(backend.app.logger, level="ERROR"):
            response = self.client.get("/")
        self.assert_json_error(response, 500)
        self.assertNotIn(secret, response.get_data(as_text=True))

    def test_malformed_request_returns_json_400(self):
        model = model_returning()
        with patch.object(backend, "model", model):
            response = self.client.post(
                "/predict", data=b"truncated multipart body", content_type="multipart/form-data"
            )
        self.assert_json_error(response, 400)
        model.assert_not_called()

    def test_limit_counts_complete_multipart_request(self):
        payload = image_bytes()
        body, content_type = multipart_bytes(payload)
        self.assertGreater(len(body), len(payload))
        model = model_returning()
        with patch.object(backend, "model", model):
            with patch.dict(backend.app.config, {"MAX_CONTENT_LENGTH": len(body)}):
                response = self.client.post("/predict", data=body, content_type=content_type)
                self.assertEqual(response.status_code, 200)
            with patch.dict(backend.app.config, {"MAX_CONTENT_LENGTH": len(body) - 1}):
                response = self.client.post("/predict", data=body, content_type=content_type)
                self.assert_json_error(response, 413, "File too large")
        model.assert_called_once()

    def test_large_upload_stays_in_memory(self):
        pixels = np.random.default_rng(42).integers(0, 256, (512, 512, 3), dtype=np.uint8)
        with Image.fromarray(pixels) as image:
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
        payload = buffer.getvalue()
        self.assertGreater(len(payload), 512 * 1024)  # Above Werkzeug's normal spool threshold.
        body, content_type = multipart_bytes(payload)
        with patch("werkzeug.formparser.SpooledTemporaryFile", side_effect=AssertionError("disk spool")):
            with backend.app.test_request_context(
                "/predict", method="POST", data=body, content_type=content_type
            ):
                stream = request.files["image"].stream
                self.assertIsInstance(stream, io.BytesIO)
                self.assertEqual(stream.read(), payload)
            self.assertTrue(stream.closed)

    def test_local_host_and_browser_origin_guards(self):
        model = model_returning()
        with patch.object(backend, "model", model):
            response = self.client.get("/health", base_url="http://attacker.invalid:5000")
            self.assert_json_error(response, 400)
            for headers in (
                {"Origin": "https://attacker.invalid"},
                {"Origin": "null"},
                {"Sec-Fetch-Site": "cross-site"},
            ):
                with self.subTest(headers=headers):
                    response = self.post_image(image_bytes(), headers=headers)
                    self.assert_json_error(response, 403)
            model.assert_not_called()
            for base_url in ("http://localhost:5000", "http://127.0.0.1:5000"):
                with self.subTest(base_url=base_url):
                    response = self.post_image(
                        image_bytes(), base_url=base_url,
                        headers={"Origin": base_url, "Sec-Fetch-Site": "same-origin"},
                    )
                    self.assertEqual(response.status_code, 200)


if __name__ == "__main__":
    unittest.main(verbosity=2)
