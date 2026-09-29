"""Centroid arithmetic, class alignment, and held-out leakage regressions."""

import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from src.ood.class_centroids import (
    compute_class_centroids, load_class_order, load_training_rows,
)


class CentroidTests(unittest.TestCase):
    def test_mean_uses_samples_not_batch_means_and_preserves_class_order(self):
        class Extractor:
            embedding_dim = 2

            def __call__(self, images, training):
                assert training is False
                return images[:, 0, 0, :]

        batches = [([1, 0, 1], np.array([[1, 3], [8, 10], [3, 5]])[:, None, None, :]),
                   ([1, 0], np.array([[8, 10], [12, 14]])[:, None, None, :])]
        rows = [{"breed": name} for name in ["a", "b", "a", "a", "b"]]
        with patch("src.ood.class_centroids._iter_batches", return_value=iter(batches)):
            centers, counts = compute_class_centroids(
                Extractor(), rows, ["b", "a"], ".", batch_size=3
            )
        np.testing.assert_allclose(centers, [[10, 12], [4, 6]])
        self.assertEqual(counts, [2, 3])
        self.assertEqual(centers.dtype, np.float32)

    def test_missing_class_and_invalid_batch_fail_before_extraction(self):
        with self.assertRaisesRegex(ValueError, "missing"):
            compute_class_centroids(None, [{"breed": "a"}], ["a", "b"], ".")
        with self.assertRaisesRegex(ValueError, "batch_size"):
            compute_class_centroids(None, [{"breed": "a"}], ["a"], ".", batch_size=0)

    def test_split_overlap_and_duplicate_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)

            def write_split(name, rows):
                with (root / f"{name}.csv").open("w", newline="") as handle:
                    writer = csv.writer(handle)
                    writer.writerow(["id", "breed", "img_name"])
                    writer.writerows(rows)

            write_split("train", [["a", "breed_a", "a.jpg"]])
            write_split("val", [["v", "breed_a", "v.jpg"]])
            write_split("test", [["t", "breed_a", "t.jpg"]])
            self.assertEqual(len(load_training_rows(root / "train.csv")), 1)
            with self.assertRaisesRegex(ValueError, "training split train.csv"):
                load_training_rows(root / "test.csv")
            write_split("test", [["a", "breed_a", "a.jpg"]])
            with self.assertRaisesRegex(ValueError, "overlaps test"):
                load_training_rows(root / "train.csv")
            write_split("train", [["a", "breed_a", "a.jpg"]] * 2)
            with self.assertRaisesRegex(ValueError, "duplicate"):
                load_training_rows(root / "train.csv")

    def test_numeric_class_indices_define_row_order(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "classes.json"
            mapping = {str(i): f"breed_{i}" for i in reversed(range(120))}
            path.write_text(json.dumps(mapping), encoding="utf-8")
            self.assertEqual(load_class_order(path), [f"breed_{i}" for i in range(120)])
            mapping["120"] = mapping.pop("0")
            path.write_text(json.dumps(mapping), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "contiguous"):
                load_class_order(path)

    def test_saved_artifacts_match_training_split_and_production_model(self):
        root = Path(__file__).resolve().parents[1]
        array_path = root / "outputs/calibration/class_centroids.npy"
        metadata_path = array_path.with_suffix(".json")
        if not array_path.is_file() or not metadata_path.is_file():
            self.skipTest("centroid artifacts have not been generated")
        values = np.load(array_path, allow_pickle=False)
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        classes = load_class_order(root / "outputs/deployment_ready/class_names.json")
        train_path = root / "outputs/splits/train.csv"
        rows = load_training_rows(train_path)
        counts = [sum(row["breed"] == breed for row in rows) for breed in classes]
        self.assertEqual(values.shape, (120, 1280))
        self.assertEqual(values.dtype, np.float32)
        self.assertTrue(np.isfinite(values).all())
        self.assertEqual(metadata["embedding_dimension"], values.shape[1])
        self.assertEqual(metadata["num_classes"], len(classes))
        self.assertEqual(metadata["class_order"], classes)
        self.assertEqual(metadata["samples_per_class"], counts)
        self.assertTrue(all(count > 0 for count in counts))
        self.assertEqual(sum(counts), len(rows))
        self.assertEqual(metadata["source_split"], "train")
        self.assertEqual(metadata["train_csv_sha256"], hashlib.sha256(train_path.read_bytes()).hexdigest())
        model_path = root / "outputs/deployment_ready/dog_breed_model.keras"
        self.assertEqual(metadata["model_sha256"], hashlib.sha256(model_path.read_bytes()).hexdigest())


if __name__ == "__main__":
    unittest.main()
