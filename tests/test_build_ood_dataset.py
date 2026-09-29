"""Regression tests for deterministic OOD materialisation and metadata."""

import csv
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from src.data.build_ood_dataset import build_ood_dataset


class OODBuilderTests(unittest.TestCase):
    def _image(self, path: Path, color: tuple[int, int, int]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (24, 19), color).save(path)

    def test_builds_disjoint_splits_hard_pool_and_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for index in range(4):
                self._image(root / "source/animals/cat" / f"cat-{index}.jpg", (index * 30, 20, 10))
            for index in range(2):
                self._image(root / "hard/dog_toy" / f"toy-{index}.jpg", (20, index * 40, 80))

            first = build_ood_dataset(root, "source", hard_source="hard", seed=7)
            manifest = root / "metadata/dataset_manifest.csv"
            with manifest.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual((first["calibration_count"], first["test_count"], first["hard_count"]), (2, 2, 2))
            self.assertEqual({row["distribution"] for row in rows}, {"ood"})
            self.assertEqual({row["ood_category"] for row in rows}, {"cat", "dog_toy"})
            self.assertEqual(len({row["sha256"] for row in rows}), len(rows))
            self.assertTrue((root / "data/ood/calibration").is_dir())
            self.assertTrue((root / "data/ood/test").is_dir())
            self.assertTrue((root / "data/ood_hard/dog_toy").is_dir())
            report_before = (root / "outputs/reports/OOD_DATASET_REPORT.md").read_bytes()
            manifest_before = manifest.read_bytes()

            second = build_ood_dataset(root, "source", hard_source="hard", seed=7)
            self.assertEqual(first["calibration_count"], second["calibration_count"])
            self.assertEqual(report_before, (root / "outputs/reports/OOD_DATASET_REPORT.md").read_bytes())
            self.assertEqual(manifest_before, manifest.read_bytes())

    def test_rejects_exact_training_leak(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source/animals/cat/cat.jpg"
            self._image(source, (1, 2, 3))
            training = root / "data/train/training.jpg"
            training.parent.mkdir(parents=True)
            training.write_bytes(source.read_bytes())
            with self.assertRaisesRegex(ValueError, "training image"):
                build_ood_dataset(root, "source")

    def test_rejects_unknown_category(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._image(root / "source/unknown/item.jpg", (1, 2, 3))
            with self.assertRaisesRegex(ValueError, "Cannot infer OOD category"):
                build_ood_dataset(root, "source")


if __name__ == "__main__":
    unittest.main()
