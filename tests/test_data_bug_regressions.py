"""Small image fixtures for dataset-builder regressions."""

import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from src.data.build_manifest import build_manifest, write_manifest
from src.data.build_ood_dataset import build_ood_dataset


class OODBuilderRegressions(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def image(self, relative, color):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (16, 20), color).save(path)
        return path

    def manifest_rows(self):
        with (self.root / "metadata/dataset_manifest.csv").open(
            encoding="utf-8", newline=""
        ) as handle:
            return list(csv.DictReader(handle))

    def test_default_hard_source_is_included_with_regular_source(self):
        self.image("data/ood_source/cat/a.png", "red")
        self.image("data/ood_source/cat/b.png", "blue")
        self.image("data/ood_source/fox/mixed.png", "green")
        self.image("data/ood_hard_source/dog_toy/toy.png", "yellow")

        result = build_ood_dataset(self.root)

        self.assertEqual(result["candidate_count"], 4)
        self.assertEqual(result["hard_count"], 2)
        self.assertEqual(
            {row["ood_category"] for row in self.manifest_rows()},
            {"cat", "fox", "dog_toy"},
        )

    def test_custom_destinations_are_used_for_manifest_and_counts(self):
        self.image("source/cat/a.png", "red")
        self.image("source/cat/b.png", "blue")
        self.image("hard_source/fox/fox.png", "green")
        classes = self.root / "outputs/deployment_ready/class_names.json"
        classes.parent.mkdir(parents=True)
        classes.write_text(json.dumps({"0": "beagle"}), encoding="utf-8")
        self.image("data/train/dog.jpg", "white")
        (self.root / "data/labels.csv").write_text("id,breed\ndog,beagle\n", encoding="utf-8")
        id_rows = build_manifest(self.root)
        write_manifest(id_rows, self.root / "metadata/dataset_manifest.csv")

        result = build_ood_dataset(
            self.root,
            "source",
            hard_source="hard_source",
            destination="custom/regular",
            hard_destination="custom/hard",
        )

        self.assertEqual(
            (result["calibration_count"], result["test_count"], result["hard_count"]),
            (1, 1, 1),
        )
        manifest = self.manifest_rows()
        self.assertEqual(
            [{key: str(value) for key, value in row.items()} for row in id_rows],
            [row for row in manifest if row["distribution"] == "id"],
        )
        rows = [row for row in manifest if row["distribution"] == "ood"]
        self.assertEqual(len(rows), 3)
        self.assertTrue(all(row["path"].startswith("custom/") for row in rows))
        self.assertTrue(all((self.root / row["path"]).is_file() for row in rows))

    def test_collision_suffix_cannot_overwrite_existing_filename(self):
        paths = [
            self.image(f"hard_source/fox/source-{index}.png", color)
            for index, color in enumerate(("red", "green", "blue"))
        ]
        ordered = sorted(paths, key=lambda path: hashlib.sha256(path.read_bytes()).hexdigest())
        digests = {hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
        last_digest = hashlib.sha256(ordered[-1].read_bytes()).hexdigest()
        ordered[0].rename(ordered[0].with_name(f"image__{last_digest[:12]}.png"))
        ordered[1].rename(ordered[1].with_name("image.png"))
        nested = ordered[2].parent / "nested/image.png"
        nested.parent.mkdir()
        ordered[2].rename(nested)

        result = build_ood_dataset(self.root, hard_source="hard_source")

        self.assertEqual(result["hard_count"], 3)
        self.assertEqual({row["sha256"] for row in self.manifest_rows()}, digests)

    def test_external_destination_is_rejected_before_copying(self):
        self.image("source/cat/a.png", "red")
        with tempfile.TemporaryDirectory() as external:
            destination = Path(external) / "ood"
            with self.assertRaisesRegex(ValueError, "inside the project root"):
                build_ood_dataset(self.root, "source", destination=destination)
            self.assertFalse(destination.exists())


if __name__ == "__main__":
    unittest.main()
