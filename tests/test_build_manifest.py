"""Manifest integrity checks using tiny independent image fixtures."""

import copy
import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from src.data.build_manifest import build_manifest, validate_manifest, write_manifest


class ManifestTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for directory in ("data/train", "data/test", "outputs/deployment_ready", "outputs/splits"):
            (self.root / directory).mkdir(parents=True)
        (self.root / "outputs/deployment_ready/class_names.json").write_text(
            json.dumps({"0": "beagle", "1": "pug"}), encoding="utf-8")
        self.labels = [{"id": "a", "breed": "beagle"}, {"id": "b", "breed": "pug"},
                       {"id": "c", "breed": "beagle"}]
        self.write_labels(self.labels)
        for row, split in zip(self.labels, ("train", "val", "test")):
            name = f"{row['id']}.jpg"
            Image.new("RGB", (13, 17), "red").save(self.root / "data/train" / name)
            with (self.root / f"outputs/splits/{split}.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["id", "breed", "img_name"])
                writer.writeheader()
                writer.writerow(dict(row, img_name=name))
        Image.new("RGB", (19, 23), "blue").save(self.root / "data/test/u.jpg")

    def write_labels(self, rows):
        with (self.root / "data/labels.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["id", "breed"])
            writer.writeheader()
            writer.writerows(rows)

    def test_deterministic_and_read_only(self):
        images = list((self.root / "data").rglob("*.jpg"))
        before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in images}
        rows = build_manifest(self.root)
        destination = self.root / "metadata/dataset_manifest.csv"
        write_manifest(rows, destination)
        first = destination.read_bytes()
        self.write_labels(list(reversed(self.labels)))
        write_manifest(build_manifest(self.root), destination)
        self.assertEqual(first, destination.read_bytes())
        self.assertEqual(before, {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in images})
        by_id = {row["image_id"]: row for row in rows}
        self.assertEqual([by_id[x]["split"] for x in "abc"], ["train", "val", "test"])
        self.assertEqual((by_id["u"]["label"], by_id["u"]["breed"], by_id["u"]["split"]),
                         ("", "", "unlabeled_test"))
        self.assertEqual((by_id["a"]["width"], by_id["a"]["height"]), (13, 17))
        self.assertEqual(by_id["a"]["sha256"], hashlib.sha256(before[self.root / "data/train/a.jpg"][0]).hexdigest())

    def test_required_validation_failures(self):
        original = build_manifest(self.root)
        cases = [("image_id", original[0]["image_id"], "Duplicate image_id"),
                 ("path", original[0]["path"], "Duplicate path"),
                 ("path", "data/train/missing.jpg", "Missing file"),
                 ("breed", "not_a_breed", "Invalid class"),
                 ("label", "not_a_breed", "Invalid class")]
        for field, value, message in cases:
            with self.subTest(field=field, message=message):
                rows = copy.deepcopy(original)
                rows[1][field] = value
                with self.assertRaisesRegex((ValueError, FileNotFoundError), message):
                    validate_manifest(rows, self.root, {"beagle", "pug"})

    def test_missing_labeled_image(self):
        (self.root / "data/train/a.jpg").unlink()
        with self.assertRaisesRegex(FileNotFoundError, "Missing file"):
            build_manifest(self.root)

    def test_invalid_source_class(self):
        self.labels[0]["breed"] = "typo"
        self.write_labels(self.labels)
        with self.assertRaisesRegex(ValueError, "Invalid class"):
            build_manifest(self.root)

    def test_duplicate_source_id(self):
        self.write_labels(self.labels + [self.labels[0]])
        with self.assertRaisesRegex(ValueError, "Duplicate image_id"):
            build_manifest(self.root)

    def test_partial_splits_fail(self):
        (self.root / "outputs/splits/val.csv").unlink()
        with self.assertRaises(FileNotFoundError):
            build_manifest(self.root)

    def test_unsplit_fallback(self):
        for path in (self.root / "outputs/splits").glob("*.csv"):
            path.unlink()
        self.assertEqual({r["split"] for r in build_manifest(self.root) if r["breed"]}, {"unsplit"})

    def test_unmapped_image_fails(self):
        Image.new("RGB", (10, 10)).save(self.root / "data/train/extra.jpg")
        with self.assertRaisesRegex(ValueError, "no metadata"):
            build_manifest(self.root)


if __name__ == "__main__":
    unittest.main()
