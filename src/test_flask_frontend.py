import io
import sys
import unittest
from html.parser import HTMLParser

import numpy as np
from PIL import Image

from app import app, class_names


class PageStructure(HTMLParser):
    """Inspect served markup without installing a browser or HTML library."""

    def __init__(self):
        super().__init__()
        self.elements = []

    def handle_starttag(self, tag, attrs):
        self.elements.append((tag, dict(attrs)))


class TestFlaskFrontend(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = app.test_client()

    def test_01_index_endpoint(self):
        """Test GET / returns HTTP 200, text/html, and essential UI text."""
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("text/html", response.content_type)
        html_content = response.get_data(as_text=True)
        self.assertIn("Dog Breed Recognition", html_content)
        self.assertIn("Predict Breed", html_content)

    def test_02_static_css(self):
        """Test GET /static/style.css returns HTTP 200 and CSS content."""
        response = self.client.get("/static/style.css")
        self.addCleanup(response.close)
        self.assertEqual(response.status_code, 200)
        self.assertIn("text/css", response.content_type)
        css_content = response.get_data(as_text=True)
        self.assertIn("--primary-color", css_content)
        self.assertIn(".progress-fill", css_content)

    def test_03_static_js(self):
        """Test GET /static/app.js returns HTTP 200 and JavaScript content."""
        response = self.client.get("/static/app.js")
        self.addCleanup(response.close)
        self.assertEqual(response.status_code, 200)
        # Verify content type is javascript
        self.assertTrue(
            "javascript" in response.content_type or "text/plain" in response.content_type,
            f"Unexpected content type: {response.content_type}"
        )
        js_content = response.get_data(as_text=True)
        self.assertIn("imageInput", js_content)
        self.assertIn("predictBtn", js_content)
        self.assertIn("formatBreedName", js_content)

    def test_04_html_elements_present(self):
        """Test HTML contains required form inputs, preview, and result containers."""
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        html = response.get_data(as_text=True)

        # Form / Input elements
        self.assertIn('id="imageInput"', html)
        self.assertIn('accept="image/jpeg,image/png,image/webp"', html)
        self.assertNotIn('multiple', html)  # Must be single upload

        # Action button
        self.assertIn('id="predictBtn"', html)
        self.assertIn("Predict Breed", html)

        # Preview element
        self.assertIn('id="imagePreview"', html)
        self.assertIn('alt="Dog image preview"', html)

        # Prediction and Top-5 containers
        self.assertIn('id="resultContainer"', html)
        self.assertIn('id="top1Container"', html)
        self.assertIn('id="top5Container"', html)
        self.assertIn('id="errorContainer"', html)

    def test_05_predict_api_regression(self):
        """Test POST /predict API regression using synthetic JPEG payload."""
        # Create a small synthetic JPEG image (100x100 RGB)
        rng = np.random.default_rng(123)
        raw_pixels = rng.integers(0, 256, size=(100, 100, 3), dtype=np.uint8)
        img = Image.fromarray(raw_pixels)
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        buf.seek(0)

        response = self.client.post(
            "/predict",
            data={"image": (buf, "regression_test.jpg")},
            content_type="multipart/form-data"
        )
        self.assertEqual(response.status_code, 200)
        data = response.get_json()

        # Validate prediction response contract
        self.assertIn("model", data)
        self.assertEqual(data["model"], "mobilenet_v2_transfer")

        self.assertIn("prediction", data)
        top1 = data["prediction"]
        self.assertIn("index", top1)
        self.assertIn("breed", top1)
        self.assertIn("confidence", top1)
        self.assertEqual(top1["breed"], class_names[top1["index"]])

        self.assertIn("top5", data)
        top5 = data["top5"]
        self.assertEqual(len(top5), 5)
        for rank_item in top5:
            self.assertIn("index", rank_item)
            self.assertIn("breed", rank_item)
            self.assertIn("confidence", rank_item)
            self.assertEqual(rank_item["breed"], class_names[rank_item["index"]])

    def test_06_accessible_markup(self):
        """Controls and announcements have resolvable, unique semantic references."""
        page = PageStructure()
        page.feed(self.client.get("/").get_data(as_text=True))
        ids = [attrs["id"] for _, attrs in page.elements if "id" in attrs]
        self.assertEqual(len(ids), len(set(ids)), "Duplicate HTML IDs break labels and JS selectors")
        by_id = {attrs["id"]: attrs for _, attrs in page.elements if "id" in attrs}
        self.assertTrue(any(tag == "label" and attrs.get("for") == "imageInput"
                            for tag, attrs in page.elements))
        self.assertNotEqual(by_id["imageInput"].get("tabindex"), "-1")
        self.assertNotIn("hidden", by_id["imageInput"])
        self.assertEqual(by_id["resetBtn"].get("type"), "button")
        self.assertEqual(by_id["statusMessage"].get("role"), "status")
        self.assertEqual(by_id["statusMessage"].get("aria-live"), "polite")
        self.assertNotIn("hidden", by_id["statusMessage"].get("class", "").split())
        self.assertEqual(by_id["errorContainer"].get("role"), "alert")
        for _, attrs in page.elements:
            for reference in ("for", "aria-describedby", "aria-labelledby"):
                for target in attrs.get(reference, "").split():
                    self.assertIn(target, by_id)
            if "src" in attrs:
                self.assertTrue(attrs["src"], "Empty src can request the HTML page as an image")

    def test_07_assets_are_local(self):
        """The local UI has no external script, font, or stylesheet requirement."""
        page = PageStructure()
        page.feed(self.client.get("/").get_data(as_text=True))
        assets = [attrs[key] for tag, attrs in page.elements
                  for key in ("src", "href") if tag in ("script", "link") and key in attrs]
        self.assertGreaterEqual(len(assets), 2)
        for asset in assets:
            self.assertTrue(asset.startswith("/static/"), asset)


def run_tests():
    print("=" * 70)
    print("RUNNING FLASK FRONTEND TEST SUITE")
    print("=" * 70)
    suite = unittest.TestLoader().loadTestsFromTestCase(TestFlaskFrontend)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    if not result.wasSuccessful():
        print("\nFRONTEND TEST SUITE FAILED!")
        sys.exit(1)
    print("\nALL FRONTEND TESTS PASSED SUCCESSFULLY!")


if __name__ == "__main__":
    run_tests()
