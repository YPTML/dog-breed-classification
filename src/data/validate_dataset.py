"""Validate image datasets and write a reproducible manifest.

The validator is intentionally independent of TensorFlow.  It can be run
before training and checks decode failures, dimensions, SHA-256 duplicates and
perceptual duplicates across ID and OOD splits.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import struct
import warnings
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageOps, UnidentifiedImageError


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
IMAGE_EXTENSIONS.update({".tif", ".tiff", ".gif", ".jfif"})


def jpeg_dimensions(content: bytes) -> tuple[int, int] | None:
    """Read JPEG dimensions from SOF markers without invoking libjpeg."""
    if len(content) < 4 or content[:2] != b"\xff\xd8":
        return None
    index = 2
    sof_markers = set(range(0xC0, 0xC4)) | set(range(0xC5, 0xC8)) | set(range(0xC9, 0xCC)) | set(range(0xCD, 0xD0))
    while index + 3 < len(content):
        if content[index] != 0xFF:
            index += 1
            continue
        while index < len(content) and content[index] == 0xFF:
            index += 1
        if index >= len(content):
            break
        marker = content[index]
        index += 1
        if marker in {0xD8, 0xD9} or 0xD0 <= marker <= 0xD7:
            continue
        if index + 2 > len(content):
            break
        length = struct.unpack(">H", content[index:index + 2])[0]
        if length < 2 or index + length > len(content):
            break
        if marker in sof_markers and length >= 7:
            height, width = struct.unpack(">HH", content[index + 3:index + 7])
            return (int(width), int(height)) if width and height else None
        index += length
    return None


def inspect_image(path: Path) -> dict:
    """Read only: distinguish open failures from verification failures.

    Dimensions are stored pixel dimensions (before EXIF orientation). Warnings
    are retained separately and do not imply that the image failed to decode.
    """
    result = {"path": path.as_posix(), "status": "ok", "warnings": []}
    stage = "open"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            if path.suffix.lower() in {".jpg", ".jpeg", ".jfif"}:
                stage = "header"
                content = path.read_bytes()
                if len(content) < 2 or content[:2] != b"\xff\xd8":
                    stage = "open"
                    raise OSError("file does not start with a JPEG SOI marker")
                dimensions = jpeg_dimensions(content)
                if dimensions is None:
                    raise OSError("invalid or incomplete JPEG header")
                if not content.rstrip().endswith(b"\xff\xd9"):
                    raise OSError("JPEG is missing the end-of-image marker")
                width, height = dimensions
                result.update(width=width, height=height, format="JPEG", mode="unknown",
                              aspect_ratio=width / height)
                return result
            with Image.open(path) as image:
                image_format = (image.format or "").upper()
                if image_format not in {"JPEG", "JPG"}:
                    stage = "verify"
                    image.verify()
                result.update(width=image.width, height=image.height,
                              format=image.format, mode=image.mode,
                              aspect_ratio=image.width / image.height)
            # Check the required JPEG end marker without rasterizing every
            # image; this catches the common truncated-file case safely.
            if image_format in {"JPEG", "JPG"}:
                with path.open("rb") as handle:
                    handle.seek(-2, 2)
                    if handle.read(2) != b"\xff\xd9":
                        raise OSError("JPEG is missing the end-of-image marker")
        except Exception as exc:
            # A single bad file must not prevent reporting the remaining pool.
            result.update(status="cannot_open" if stage == "open" else "corrupt",
                          stage=stage, error_type=type(exc).__name__, error=str(exc))
        result["warnings"] = [str(item.message) for item in caught]
    return result


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def perceptual_hash(path: Path, size: int = 8) -> str:
    """Return a small average hash suitable for duplicate triage."""
    with Image.open(path) as image:
        gray = ImageOps.grayscale(image).resize((size, size), Image.Resampling.BILINEAR)
        pixels = list(gray.getdata())
    average = sum(pixels) / len(pixels)
    return "".join("1" if pixel >= average else "0" for pixel in pixels)


def _images(root: Path) -> Iterable[Path]:
    if not root.exists():
        return ()
    return (path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)


def scan_dataset(root: str | Path, *, distribution: str, split: str, source: str = "local") -> tuple[list[dict], list[dict]]:
    """Scan one dataset root, returning manifest rows and QA failures."""
    root_path = Path(root)
    rows: list[dict] = []
    failures: list[dict] = []
    for path in sorted(_images(root_path), key=lambda item: item.as_posix().lower()):
        relative = path.relative_to(root_path).as_posix()
        parts = Path(relative).parts
        label = parts[0] if len(parts) > 1 else ""
        category = label if distribution == "ood" else ""
        row = {
            "image_id": hashlib.sha1(relative.encode("utf-8")).hexdigest()[:16],
            "path": path.as_posix(),
            "label": label if distribution == "id" else "not_dog",
            "breed": label if distribution == "id" else "",
            "distribution": distribution,
            "ood_category": category,
            "split": split,
            "source": source,
        }
        try:
            with Image.open(path) as image:
                image.verify()
            with Image.open(path) as image:
                width, height = image.size
            row.update({"width": int(width), "height": int(height), "sha256": sha256_file(path), "phash": perceptual_hash(path)})
            if width < 8 or height < 8:
                failures.append({"path": path.as_posix(), "reason": "low_resolution"})
        except (OSError, UnidentifiedImageError, ValueError) as exc:
            failures.append({"path": path.as_posix(), "reason": "decode_error", "detail": str(exc)})
            continue
        rows.append(row)
    return rows, failures


def find_duplicates(rows: list[dict]) -> dict[str, list[list[str]]]:
    duplicate_groups: dict[str, list[list[str]]] = {}
    for field in ("sha256", "phash"):
        groups: dict[str, list[str]] = {}
        for row in rows:
            value = row.get(field)
            if value:
                groups.setdefault(value, []).append(row["path"])
        duplicate_groups[field] = [paths for paths in groups.values() if len(paths) > 1]
    return duplicate_groups


def write_report(rows: list[dict], failures: list[dict], output_dir: str | Path) -> dict:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    manifest_path = destination / "dataset_manifest.csv"
    fields = ["image_id", "path", "label", "breed", "distribution", "ood_category", "split", "source", "width", "height", "sha256", "phash"]
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    duplicates = find_duplicates(rows)
    report = {
        "image_count": len(rows),
        "failure_count": len(failures),
        "failures": failures,
        "duplicate_sha256_groups": duplicates["sha256"],
        "duplicate_phash_groups": duplicates["phash"],
        "cross_split_duplicate_count": sum(1 for group in duplicates["sha256"] if len({next((r["split"] for r in rows if r["path"] == p), "") for p in group}) > 1),
    }
    with (destination / "dataset_qa.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
    return report


def main() -> None:
    import sys
    # Preserve the existing explicit ID/OOD CLI; default now audits this project.
    if not any(arg.split("=")[0] in {"--id-root", "--ood-root", "--output-dir"}
               for arg in sys.argv[1:]):
        from src.data.dataset_analysis import main as audit_main
        audit_main()
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--id-root", type=Path, default=Path("data/in_distribution"))
    parser.add_argument("--ood-root", type=Path, default=Path("data/ood"))
    parser.add_argument("--output-dir", type=Path, default=Path("metadata"))
    args = parser.parse_args()
    rows: list[dict] = []
    failures: list[dict] = []
    for root, distribution in ((args.id_root, "id"), (args.ood_root, "ood")):
        for split_dir in ("train", "val", "test", "calibration"):
            split_root = root / split_dir
            scanned, bad = scan_dataset(split_root, distribution=distribution, split=split_dir)
            rows.extend(scanned)
            failures.extend(bad)
    report = write_report(rows, failures, args.output_dir)
    print(json.dumps({k: report[k] for k in ("image_count", "failure_count", "cross_split_duplicate_count")}, indent=2))


if __name__ == "__main__":
    main()
