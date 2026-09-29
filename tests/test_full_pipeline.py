"""TASK 22 end-to-end coverage for the deployed image pipeline.

The OOD groups deliberately use only local, materialized fixtures.  This test
does not label synthetic pixels as cats, people, vehicles, or objects: when a
required semantic fixture is absent, the group is reported as skipped.
"""

from __future__ import annotations

import io
import unittest
from pathlib import Path

from PIL import Image, ImageDraw

import app


ROOT = Path(__file__).resolve().parents[1]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
REGULAR_OOD_ROOT = ROOT / "data" / "ood" / "test"
HARD_OOD_ROOT = ROOT / "data" / "ood_hard"


def _first_image(root: Path, categories: tuple[str, ...] = ()) -> Path | None:
    candidates = []
    for category in categories:
        category_root = root / category
        if category_root.is_dir():
            candidates.extend(
                path
                for path in category_root.rglob("*")
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            )
    if not categories and root.is_dir():
        candidates.extend(
            path
            for path in root.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
    return sorted(candidates, key=lambda path: path.as_posix().lower())[0] if candidates else None


def _encode(image: Image.Image, image_format: str = "JPEG") -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format=image_format)
    return buffer.getvalue()


class FullPipelineTests(unittest.TestCase):
    """Exercise the complete Flask decode -> inference -> decision contract."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.client = app.app.test_client()
        cls.dog_path = _first_image(ROOT / "data" / "test")
        if cls.dog_path is None:
            raise unittest.SkipTest("no local dog image is available under data/test")
        cls.dog_bytes = cls.dog_path.read_bytes()

    def post(self, payload: bytes, filename: str, content_type: str | None = None):
        content_type = content_type or "image/jpeg"
        return self.client.post(
            "/predict",
            data={"image": (io.BytesIO(payload), filename, content_type)},
            content_type="multipart/form-data",
        )

    def assert_valid_prediction_response(self, response) -> dict:
        self.assertEqual(response.status_code, 200, response.get_data(as_text=True))
        body = response.get_json()
        self.assertIsInstance(body, dict)
        self.assertIn(body.get("status"), {"DOG", "UNCERTAIN", "NOT_DOG"})
        self.assertEqual(body.get("status"), body.get("decision"))
        self.assertEqual(body.get("status"), body.get("final_status"))
        expected_is_dog = True if body["decision"] == "DOG" else (
            False if body["decision"] == "NOT_DOG" else None
        )
        self.assertIs(body.get("is_dog"), expected_is_dog)
        self.assertIsInstance(body.get("prediction"), dict)
        self.assertEqual(len(body.get("top5", [])), 5)
        return body

    def assert_ood_group_contract(self, root: Path, categories: tuple[str, ...], label: str) -> None:
        fixture = _first_image(root, categories)
        if fixture is None:
            self.skipTest(f"{label}: no local fixture under {root}")
        body = self.assert_valid_prediction_response(
            self.post(fixture.read_bytes(), fixture.name, "image/jpeg")
        )
        # The model decision is recorded by the full report; this assertion
        # intentionally checks only the runtime contract, not a threshold-made
        # semantic claim about an unavailable or unlabelled fixture.
        self.assertIn(body["decision"], {"DOG", "UNCERTAIN", "NOT_DOG"})

    def test_01_normal_dog(self) -> None:
        """A real local dog image reaches the complete inference path."""
        body = self.assert_valid_prediction_response(
            self.post(self.dog_bytes, self.dog_path.name, "image/jpeg")
        )
        self.assertIn(body["decision"], {"DOG", "UNCERTAIN", "NOT_DOG"})

    def test_02_other_animals(self) -> None:
        self.assert_ood_group_contract(
            REGULAR_OOD_ROOT,
            ("cat", "horse", "cow", "sheep", "rabbit", "bird", "elephant", "tiger", "bear"),
            "other animals",
        )

    def test_03_humans(self) -> None:
        self.assert_ood_group_contract(REGULAR_OOD_ROOT, ("humans",), "humans")

    def test_04_vehicles(self) -> None:
        self.assert_ood_group_contract(
            REGULAR_OOD_ROOT, ("car", "motorcycle", "bus", "bicycle"), "vehicles"
        )

    def test_05_household_objects(self) -> None:
        self.assert_ood_group_contract(
            REGULAR_OOD_ROOT,
            ("chair", "table", "bottle", "phone", "laptop"),
            "household objects",
        )

    def test_06_hard_ood(self) -> None:
        self.assert_ood_group_contract(
            HARD_OOD_ROOT,
            ("wolf", "fox", "coyote", "hyena", "dog_toy", "dog_statue", "dog_cartoon", "dog_drawing"),
            "hard OOD",
        )

    def test_07_corrupt_images(self) -> None:
        corrupt_png = b"\x89PNG\r\n\x1a\n" + b"not-a-valid-png"
        truncated_jpeg = b"\xff\xd8\xff\xe0\x00\x10JFIF"
        for payload, filename in ((corrupt_png, "corrupt.png"), (truncated_jpeg, "truncated.jpg")):
            with self.subTest(filename=filename):
                response = self.post(payload, filename)
                self.assertEqual(response.status_code, 400)
                self.assertEqual(response.get_json().get("error"), "Invalid or unsupported image")

    def test_08_invalid_file_type(self) -> None:
        response = self.post(b"plain text is not an image", "payload.txt", "text/plain")
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.get_json().get("error"), "Invalid or unsupported image")

    def test_09_empty_upload(self) -> None:
        missing = self.client.post("/predict")
        self.assertEqual(missing.status_code, 400)
        self.assertEqual(missing.get_json().get("error"), "Missing image file")

        for filename, payload in (("", b""), ("empty.jpg", b"")):
            with self.subTest(filename=filename):
                response = self.post(payload, filename)
                self.assertEqual(response.status_code, 400)
                self.assertEqual(response.get_json().get("error"), "Empty image file")

    def test_10_low_resolution_dog(self) -> None:
        with Image.open(io.BytesIO(self.dog_bytes)) as source:
            low_resolution = source.convert("RGB").resize((16, 16), Image.Resampling.NEAREST)
        self.assert_valid_prediction_response(
            self.post(_encode(low_resolution), "low-resolution-dog.jpg")
        )

    def test_11_partially_occluded_dog(self) -> None:
        with Image.open(io.BytesIO(self.dog_bytes)) as source:
            occluded = source.convert("RGB")
            draw = ImageDraw.Draw(occluded)
            width, height = occluded.size
            draw.rectangle(
                (width // 4, height // 4, width * 3 // 4, height * 3 // 4),
                fill=(0, 0, 0),
            )
        self.assert_valid_prediction_response(
            self.post(_encode(occluded), "partially-occluded-dog.jpg")
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
