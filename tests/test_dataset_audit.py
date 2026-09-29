"""Small adversarial fixtures for the read-only dataset audit."""
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from src.data.dataset_analysis import audit_project, render_report
from src.data.validate_dataset import inspect_image


class DatasetAuditTests(unittest.TestCase):
    def test_bad_images_missing_class_and_split_errors(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ("data/train", "data/test", "outputs/deployment_ready", "outputs/splits"):
                (root / name).mkdir(parents=True)
            (root / "outputs/deployment_ready/class_names.json").write_text(
                json.dumps(["breed_a", "breed_b"]), encoding="utf-8")
            (root / "data/labels.csv").write_text("id,breed\na,breed_a\nb,breed_b\n", encoding="utf-8")
            Image.new("RGB", (40, 20)).save(root / "data/train/a.jpg")
            broken = root / "data/train/bad.jpg"
            broken.write_bytes(b"not an image")
            truncated = root / "data/train/truncated.jpg"
            Image.new("RGB", (40, 20)).save(truncated)
            truncated.write_bytes(truncated.read_bytes()[:-2])
            self.assertEqual(inspect_image(broken)["status"], "cannot_open")
            self.assertEqual(inspect_image(truncated)["status"], "corrupt")
            header = "id,breed,img_name\n"
            (root / "outputs/splits/train.csv").write_text(header + "a,breed_a,a.jpg\na,breed_a,a.jpg\n", encoding="utf-8")
            (root / "outputs/splits/val.csv").write_text(header + "a,breed_b,a.jpg\n", encoding="utf-8")
            (root / "outputs/splits/test.csv").write_text(header, encoding="utf-8")
            before = {p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()}
            result = audit_project(root)
            after = {p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()}
            self.assertEqual(before, after)
            self.assertEqual(result["total_images"], 3)
            self.assertEqual(result["class_count"], 1)
            self.assertEqual(result["missing_classes"], ["breed_b"])
            self.assertEqual(result["missing_labeled_images"], ["b.jpg"])
            self.assertEqual(result["images_per_class_summary"]["min"], 0)
            self.assertEqual(result["pools"]["labeled_pool"]["aspect_ratio"]["mean"], 2)
            self.assertEqual(result["splits"]["overlap_ids"]["train/validation"], ["a"])
            self.assertEqual(result["splits"]["distribution"]["train"]["duplicate_ids"], {"a": 2})
            self.assertEqual(len(result["splits"]["distribution"]["validation"]["label_mismatches"]), 1)
            self.assertTrue(result["preservation"]["metadata_sha256_unchanged"])
            self.assertIn("DATASET AUDIT", render_report(result))

    def test_missing_dataset_still_reports(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = audit_project(Path(tmp))
            self.assertEqual(result["total_images"], 0)
            self.assertIsNone(result["images_per_class_summary"]["mean"])
            self.assertTrue(result["metadata_issues"])
            json.dumps(result, allow_nan=False)
            self.assertIn("DATASET AUDIT", render_report(result))


if __name__ == "__main__":
    unittest.main()
