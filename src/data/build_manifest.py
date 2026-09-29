"""Build read-only metadata for the local Kaggle Dog Breed Identification images.

Paths are relative to the project root. Labels are breed names from the shipped
class vocabulary; unlabeled Kaggle test images have empty label/breed fields.
Existing split CSVs are authoritative and must cover every labeled image once.
If none exist, labeled images use 'unsplit'; no random split is invented.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import struct
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import json
import os
from pathlib import Path
import re
import tempfile

try:
    from PIL import Image
except ImportError:  # pragma: no cover - exercised in minimal runtime environments
    Image = None


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIELDS = [
    "image_id", "path", "label", "breed", "distribution", "ood_category",
    "split", "source", "width", "height", "sha256",
]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".gif", ".jfif"}
SOURCE = "kaggle_dog_breed_identification"
OOD_SOURCE = "local_ood"
OOD_REGULAR_CATEGORIES = frozenset(
    ("cat", "horse", "cow", "sheep", "rabbit", "bird", "elephant", "tiger", "bear",
     "humans", "car", "motorcycle", "bus", "bicycle", "chair", "table", "bottle",
     "phone", "laptop", "food", "landscape")
)
OOD_HARD_CATEGORIES = frozenset(
    ("wolf", "fox", "coyote", "hyena", "dog_toy", "dog_statue", "dog_cartoon", "dog_drawing")
)


def read_csv(path: Path, required: set[str]) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Missing columns in {path}: {sorted(required)}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"Empty metadata: {path}")
    return rows


def load_classes(path: Path) -> set[str]:
    with path.open(encoding="utf-8") as handle:
        mapping = json.load(handle)
    names = list(mapping.values()) if isinstance(mapping, dict) else mapping
    if (not isinstance(names, list) or not names
            or any(not isinstance(name, str) or not name.strip() for name in names)
            or len(set(names)) != len(names)):
        raise ValueError(f"Invalid class vocabulary: {path}")
    return set(names)


def validate_manifest(rows: list[dict], root: Path, classes: set[str]) -> None:
    """Reject duplicate identities/paths, invalid classes and OOD leakage."""
    root = Path(root).resolve()
    seen_ids, seen_paths = set(), set()
    ood_hashes, train_hashes = set(), set()
    for row in rows:
        if set(row) != set(FIELDS):
            raise ValueError("Invalid manifest columns")
        image_id = row["image_id"]
        if not image_id or image_id in seen_ids:
            raise ValueError(f"Duplicate image_id or empty image_id: {image_id}")
        seen_ids.add(image_id)
        relative = Path(row["path"])
        path = (root / relative).resolve()
        if not row["path"] or relative.is_absolute() or not path.is_relative_to(root):
            raise ValueError(f"Invalid path: {row['path']}")
        key = os.path.normcase(str(path))
        if key in seen_paths:
            raise ValueError(f"Duplicate path: {row['path']}")
        seen_paths.add(key)
        if not path.is_file():
            raise FileNotFoundError(f"Missing file: {row['path']}")
        if row["distribution"] == "ood":
            is_regular = (len(relative.parts) >= 5 and relative.parts[:2] == ("data", "ood")
                          and relative.parts[2] in {"calibration", "test"}
                          and relative.parts[3] in OOD_REGULAR_CATEGORIES)
            is_hard = (len(relative.parts) >= 4 and relative.parts[:2] == ("data", "ood_hard")
                       and relative.parts[2] in OOD_HARD_CATEGORIES)
            expected_split = relative.parts[2] if is_regular else "hard" if is_hard else ""
            valid_class = (row["label"] == row["breed"] == "" and row["ood_category"] in
                           (OOD_REGULAR_CATEGORIES | OOD_HARD_CATEGORIES)
                           and row["split"] == expected_split and bool(row["source"])
                           and row["ood_category"] == (relative.parts[3] if is_regular else relative.parts[2] if is_hard else ""))
            if not valid_class:
                raise ValueError(f"Invalid OOD metadata or layout for {image_id}: {row}")
            if row["sha256"]:
                if row["sha256"] in ood_hashes:
                    raise ValueError(f"Duplicate OOD image content: {image_id}")
                ood_hashes.add(row["sha256"])
            continue
        if row["distribution"] != "id":
            raise ValueError(f"Invalid distribution for {image_id}: {row['distribution']}")
        unlabeled = row["split"] == "unlabeled_test" and relative.parts[:2] == ("data", "test")
        if unlabeled:
            valid_class = row["label"] == row["breed"] == ""
        else:
            valid_class = row["breed"] in classes and row["label"] == row["breed"]
        if not valid_class:
            raise ValueError(f"Invalid class for {image_id}: {row['label']!r}/{row['breed']!r}")
        if row["ood_category"] != "":
            raise ValueError(f"Invalid dog distribution/category: {image_id}")
        if row["split"] == "train" and row["sha256"]:
            train_hashes.add(row["sha256"])
    overlap = ood_hashes & train_hashes
    if overlap:
        raise ValueError(f"OOD image overlaps 120-class training content: {sorted(overlap)[0]}")


def load_splits(root: Path, labels: dict[str, str]) -> dict[str, str]:
    paths = {split: root / "outputs/splits" / f"{split}.csv" for split in ("train", "val", "test")}
    if not any(path.exists() for path in paths.values()):
        return dict.fromkeys(labels, "unsplit")
    result = {}
    for split, path in paths.items():
        for row in read_csv(path, {"id", "breed", "img_name"}):
            image_id = row["id"]
            if image_id in result:
                raise ValueError(f"Duplicate image_id in splits: {image_id}")
            if image_id not in labels or row["breed"] != labels[image_id]:
                raise ValueError(f"Invalid class or unknown ID in {path}: {image_id}")
            if row["img_name"] != f"{image_id}.jpg":
                raise ValueError(f"Invalid image path in {path}: {row['img_name']}")
            result[image_id] = split
    if set(result) != set(labels):
        raise ValueError("Split metadata does not cover all labeled images")
    return result


def build_manifest(
    root: Path = PROJECT_ROOT,
    *,
    progress: bool = False,
    row_start: int = 0,
    row_end: int | None = None,
) -> list[dict]:
    root = Path(root).resolve()
    classes = load_classes(root / "outputs/deployment_ready/class_names.json")
    labels = {}
    for row in read_csv(root / "data/labels.csv", {"id", "breed"}):
        image_id, breed = row["id"], row["breed"]
        if not image_id or not re.fullmatch(r"[A-Za-z0-9_-]+", image_id):
            raise ValueError(f"Invalid image_id: {image_id!r}")
        if image_id in labels:
            raise ValueError(f"Duplicate image_id in labels: {image_id}")
        if breed not in classes:
            raise ValueError(f"Invalid class: {breed!r}")
        labels[image_id] = breed
    splits = load_splits(root, labels)
    rows = []
    for image_id, breed in labels.items():
        rows.append(dict(zip(FIELDS, [image_id, f"data/train/{image_id}.jpg", breed,
                                     breed, "id", "", splits[image_id], SOURCE, "", "", ""])))
    expected = {row["path"] for row in rows}
    # OOD output is intentionally outside the 120-class label vocabulary. It
    # is still part of the unified manifest, but gets empty label/breed fields
    # and a separate distribution so training loaders can exclude it.
    for path in sorted((root / "data").rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        relative = path.relative_to(root).as_posix()
        if relative in expected:
            continue
        parts = relative.split("/")
        if parts[:2] == ["data", "ood"]:
            if len(parts) < 5 or parts[2] not in {"calibration", "test"} or parts[3] not in OOD_REGULAR_CATEGORIES:
                raise ValueError(f"Invalid OOD image layout: {relative}")
            digest = sha256_file(path)
            rows.append(dict(zip(FIELDS, [f"ood-{digest[:32]}", relative, "", "", "ood",
                                         parts[3], parts[2], OOD_SOURCE, "", "", ""])))
            continue
        if parts[:2] == ["data", "ood_hard"]:
            if len(parts) < 4 or parts[2] not in OOD_HARD_CATEGORIES:
                raise ValueError(f"Invalid hard OOD image layout: {relative}")
            digest = sha256_file(path)
            rows.append(dict(zip(FIELDS, [f"ood-{digest[:32]}", relative, "", "", "ood",
                                         parts[2], "hard", OOD_SOURCE, "", "", ""])))
            continue
        if path.parent != root / "data/test":
            raise ValueError(f"Image has no metadata; unsupported dataset layout: {relative}")
        rows.append(dict(zip(FIELDS, [path.stem, relative, "", "", "id", "",
                                     "unlabeled_test", SOURCE, "", "", ""])))
    rows.sort(key=lambda row: row["path"])
    validate_manifest(rows, root, classes)
    if row_start < 0 or row_end is not None and row_end < row_start:
        raise ValueError("Invalid row range")
    selected_rows = rows[row_start:row_end]
    if progress:
        print(f"Metadata validated; reading {len(selected_rows)} images...", flush=True)
    result = []
    with ThreadPoolExecutor(max_workers=16) as executor:
        for row in executor.map(partial(image_metadata, root=root), selected_rows):
            result.append(row)
            if progress and (len(result) % 1000 == 0 or len(result) == len(selected_rows)):
                print(f"Images processed: {len(result)}/{len(selected_rows)}", flush=True)
    # Re-check after hashes and dimensions are populated so exact OOD/train
    # leakage and duplicate OOD bytes cannot hide behind empty placeholders.
    validate_manifest(result, root, classes)
    return result


def image_metadata(row: dict, *, root: Path) -> dict:
    path = root / row["path"]
    content = path.read_bytes()
    row["width"], row["height"] = image_dimensions(content, path.suffix.lower())
    row["sha256"] = hashlib.sha256(content).hexdigest()
    return row


def image_dimensions(content: bytes, suffix: str) -> tuple[int, int]:
    """Read dimensions from common image headers without decoding pixel data."""
    if Image is not None:
        with Image.open(io.BytesIO(content)) as image:
            size = image.size
            image.verify()
        return int(size[0]), int(size[1])
    if suffix == ".png" and content[:8] == b"\x89PNG\r\n\x1a\n":
        return struct.unpack(">II", content[16:24])
    if suffix in {".jpg", ".jpeg", ".jfif"} and content[:2] == b"\xff\xd8":
        index = 2
        while index + 9 <= len(content):
            if content[index] != 0xFF:
                index += 1
                continue
            while index < len(content) and content[index] == 0xFF:
                index += 1
            marker = content[index]
            index += 1
            if marker in {0xD8, 0xD9}:
                continue
            if index + 2 > len(content):
                break
            length = struct.unpack(">H", content[index:index + 2])[0]
            if marker in set(range(0xC0, 0xC4)) | set(range(0xC5, 0xC8)) | set(range(0xC9, 0xCC)) | set(range(0xCD, 0xD0)):
                if index + 7 <= len(content):
                    height, width = struct.unpack(">HH", content[index + 3:index + 7])
                    return int(width), int(height)
                break
            index += length
    if suffix == ".gif" and content[:6] in {b"GIF87a", b"GIF89a"}:
        return struct.unpack("<HH", content[6:10])
    if suffix == ".bmp" and content[:2] == b"BM":
        return struct.unpack("<ii", content[18:26])[:2]
    raise ValueError(f"Unsupported or invalid image header: {suffix}")


def write_manifest(rows: list[dict], destination: Path) -> None:
    """Replace output atomically only after a complete successful build."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", newline="", encoding="utf-8",
                                         dir=destination.parent, delete=False) as handle:
            temporary = Path(handle.name)
            writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--validate-only", action="store_true", help="Validate the saved manifest without rewriting it")
    parser.add_argument("--chunk-start", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--chunk-end", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--chunk-output", type=Path, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()
    root = args.root.resolve()
    destination = root / "metadata/dataset_manifest.csv"
    try:
        if args.validate_only:
            rows = read_csv(destination, set(FIELDS))
            validate_manifest(rows, root, load_classes(root / "outputs/deployment_ready/class_names.json"))
        else:
            rows = build_manifest(root, progress=True, row_start=args.chunk_start, row_end=args.chunk_end)
            write_manifest(rows, args.chunk_output or destination)
    except (ValueError, OSError) as exc:
        parser.exit(1, f"Manifest validation failed: {exc}\n")
    print(f"Validated {len(rows)} images: {args.chunk_output or destination}")


if __name__ == "__main__":
    main()
