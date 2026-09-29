"""Regression checks for read-only duplicate and leakage analysis."""
import csv
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image
from src.data.deduplicate import HashIndex, analyze, load_records, sha256_file


class DeduplicateTests(unittest.TestCase):
    def test_matches_and_read_only(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for split in ("train", "validation", "test"):
                (root / split).mkdir()
            original = root / "train/a.png"
            image = Image.new("L", (32, 32))
            image.putdata([(x * 7 + y * 3) % 256 for y in range(32) for x in range(32)])
            image.save(original)
            (root / "validation/copy.png").write_bytes(original.read_bytes())
            image.save(root / "test/reencoded.bmp")
            (root / "test/broken.png").write_bytes(b"not an image")
            records = load_records(root, None)
            before = {r["path"]: sha256_file(r["path"]) for r in records}
            counts = analyze(records, root / "report", threshold=0)
            self.assertEqual(counts, {"EXACT_DUPLICATE": 1, "NEAR_DUPLICATE": 2, "CROSS_SPLIT_DUPLICATE": 3})
            with (root / "report/duplicates.csv").open(newline="", encoding="utf-8") as handle:
                pairs = list(csv.DictReader(handle))
            self.assertEqual(len(pairs), 3)
            self.assertTrue(all(p["classification"] == "CROSS_SPLIT_DUPLICATE" for p in pairs))
            self.assertIn("dHash:", (root / "report/DUPLICATE_REPORT.md").read_text())
            self.assertEqual(before, {path: sha256_file(path) for path in before})
            with patch("src.data.deduplicate.Image", None):
                counts = analyze(records, root / "exact_only")
            self.assertEqual(counts["EXACT_DUPLICATE"], 1)
            self.assertEqual(counts["NEAR_DUPLICATE"], 0)

    def test_csv_membership_overlap_and_missing_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "a.png").write_bytes(b"identical bytes")
            for split in ("train", "val", "test"):
                (root / f"{split}.csv").write_text("img_name\na.png\na.png\n" if split != "test" else "img_name\nmissing.png\n")
            records = load_records(root, root)
            self.assertEqual(len(records), 3)
            counts = analyze(records, root / "report", near=False)
            self.assertEqual(counts["CROSS_SPLIT_DUPLICATE"], 1)
            self.assertIn("SHA256:", (root / "report/DUPLICATE_REPORT.md").read_text())

    def test_index_matches_brute_force(self):
        generator = random.Random(42)
        values = [generator.getrandbits(64) for _ in range(100)]
        values += [values[0] ^ 1, values[0] ^ 3, values[0] ^ 7]
        index = HashIndex()
        for value in values:
            index.add(value)
        for radius in (0, 1, 5, 64):
            for value in values[:5]:
                expected = {other for other in values if (value ^ other).bit_count() <= radius}
                self.assertEqual({other for other, _ in index.query(value, radius)}, expected)


if __name__ == "__main__":
    unittest.main()
