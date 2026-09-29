"""Read-only duplicate analysis using SHA256 and perceptual dHash.

Run ``python -m src.data.deduplicate`` for the project's CSV splits.
Use ``--folder-splits --image-root PATH`` for train/validation/test folders.
Near matches are candidates for manual review; no images are deleted.
"""

from __future__ import annotations

import csv
import argparse
import hashlib
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from itertools import combinations
from pathlib import Path

try:
    from PIL import Image, ImageOps
except ImportError:
    Image = ImageOps = None

PIL_DECODE_ERRORS = (OSError, ValueError)
if Image is not None:
    PIL_DECODE_ERRORS = PIL_DECODE_ERRORS + (Image.DecompressionBombError,)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".gif"}


def duplicate_groups(manifest_path: str | Path, field: str = "sha256") -> list[list[dict]]:
    if field not in {"sha256", "phash"}:
        raise ValueError("field must be sha256 or phash")
    groups: dict[str, list[dict]] = defaultdict(list)
    with Path(manifest_path).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get(field):
                groups[row[field]].append(row)
    return [rows for rows in groups.values() if len(rows) > 1]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def difference_hash(path: Path) -> int:
    """64-bit perceptual dHash, with EXIF orientation normalized."""
    with Image.open(path) as image:
        gray = ImageOps.exif_transpose(image).convert("L").resize((9, 8), Image.Resampling.LANCZOS)
        pixels = list(gray.tobytes())
    result = 0
    for y in range(8):
        for x in range(8):
            result = (result << 1) | (pixels[y * 9 + x] > pixels[y * 9 + x + 1])
    return result


def load_records(image_root: Path, splits_dir: Path | None) -> list[dict]:
    """Use CSV membership or explicit train/{val,validation}/test folders."""
    records = []
    for split, aliases in (("train", ("train",)), ("validation", ("val", "validation")), ("test", ("test",))):
        if splits_dir is not None:
            candidates = [splits_dir / (alias + ".csv") for alias in aliases]
            found = [path for path in candidates if path.is_file()]
            if len(found) != 1:
                raise ValueError(f"Expected exactly one CSV for {split}: {candidates}")
            with found[0].open(newline="", encoding="utf-8-sig") as handle:
                reader = csv.DictReader(handle)
                if "img_name" not in (reader.fieldnames or []):
                    raise ValueError(f"Missing img_name column: {found[0]}")
                for row in reader:
                    name = row["img_name"].strip()
                    if not name:
                        raise ValueError(f"Empty img_name in {found[0]}")
                    path = (image_root / name).resolve()
                    if not path.is_relative_to(image_root.resolve()):
                        raise ValueError(f"Image outside image root: {name}")
                    records.append({"path": path, "split": split})
        else:
            found = [image_root / alias for alias in aliases if (image_root / alias).is_dir()]
            if len(found) != 1:
                raise ValueError(f"Expected exactly one folder for {split} under {image_root}")
            for path in sorted(found[0].rglob("*")):
                if path.is_file() and path.suffix.lower() in EXTENSIONS:
                    records.append({"path": path.resolve(), "split": split})
    # Repeated metadata lines are not separate images; retain cross-split memberships.
    return list({(str(row["path"]), row["split"]): row for row in records}.values())


class HashIndex:
    """BK-tree for exhaustive Hamming-radius queries without an all-pairs scan."""

    def __init__(self):
        self.root = None

    def add(self, value: int) -> None:
        if self.root is None:
            self.root = (value, {})
            return
        node = self.root
        while True:
            distance = (value ^ node[0]).bit_count()
            if distance == 0:
                return
            if distance not in node[1]:
                node[1][distance] = (value, {})
                return
            node = node[1][distance]

    def query(self, value: int, radius: int):
        pending = [self.root] if self.root else []
        while pending:
            other, children = pending.pop()
            distance = (value ^ other).bit_count()
            if distance <= radius:
                yield other, distance
            pending.extend(child for edge, child in children.items() if distance - radius <= edge <= distance + radius)


def analyze(records: list[dict], output_dir: Path, threshold: int = 5, near: bool = True, progress: bool = False, workers: int = 8) -> dict:
    if not 0 <= threshold <= 64:
        raise ValueError("Hamming threshold must be between 0 and 64")
    if workers < 1:
        raise ValueError("workers must be positive")
    near_enabled = near and Image is not None
    cache, errors = {}, []
    scanned = []
    def hash_image(path):
        hashes, failures = {}, []
        try:
            hashes["sha256"] = sha256_file(path)
        except OSError as exc:
            failures.append(f"SHA256: {path}: {exc}")
        if near_enabled:
            try:
                hashes["dhash"] = difference_hash(path)
            except PIL_DECODE_ERRORS as exc:
                failures.append(f"dHash: {path}: {exc}")
        return path, hashes, failures

    paths = sorted({record["path"] for record in records})
    # Independent read-only work overlaps disk waits; map preserves report ordering.
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for number, (path, hashes, failures) in enumerate(executor.map(hash_image, paths), 1):
            cache[path] = hashes
            errors.extend(failures)
            if progress and (number % 50 == 0 or number == len(paths)):
                print(f"Hashed {number}/{len(paths)} images", flush=True)
    for record in sorted(records, key=lambda row: str(row["path"])):
        scanned.append({**record, **cache[record["path"]]})
    exact, perceptual = defaultdict(list), defaultdict(list)
    for row in scanned:
        if "sha256" in row:
            exact[row["sha256"]].append(row)
        if "sha256" in row and "dhash" in row:
            perceptual[row["dhash"]].append(row)
    counts = Counter(EXACT_DUPLICATE=0, NEAR_DUPLICATE=0, CROSS_SPLIT_DUPLICATE=0)
    split_pairs = Counter()
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "duplicates.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["classification", "match_type", "cross_split", "path_a", "split_a", "path_b", "split_b", "sha256_a", "sha256_b", "dhash_a", "dhash_b", "hamming_distance"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()

        def emit(a, b, match_type, distance=""):
            cross = a["split"] != b["split"]
            counts[match_type] += 1
            if cross:
                counts["CROSS_SPLIT_DUPLICATE"] += 1
                split_pairs[tuple(sorted((a["split"], b["split"])))] += 1
            writer.writerow(dict(classification="CROSS_SPLIT_DUPLICATE" if cross else match_type,
                                 match_type=match_type, cross_split=cross,
                                 path_a=a["path"], split_a=a["split"], path_b=b["path"], split_b=b["split"],
                                 sha256_a=a["sha256"], sha256_b=b["sha256"],
                                 dhash_a=f'{a["dhash"]:016x}' if "dhash" in a else "",
                                 dhash_b=f'{b["dhash"]:016x}' if "dhash" in b else "", hamming_distance=distance))

        for group in exact.values():
            for a, b in combinations(group, 2):
                emit(a, b, "EXACT_DUPLICATE", 0 if "dhash" in a and "dhash" in b else "")
        index = HashIndex()
        for value, group in perceptual.items():
            for a, b in combinations(group, 2):
                if a["sha256"] != b["sha256"]:
                    emit(a, b, "NEAR_DUPLICATE", 0)
            for other, distance in index.query(value, threshold):
                for a in group:
                    for b in perceptual[other]:
                        if a["sha256"] != b["sha256"]:
                            emit(a, b, "NEAR_DUPLICATE", distance)
            index.add(value)
    coverage = Counter(row["split"] for row in records)
    lines = ["# Dataset duplicate report", "", "Analysis only. No image files were modified or deleted.", "",
             f"- Unique paths: {len(cache)}; split memberships: {len(records)}.",
             f"- SHA256 successful memberships: {sum(len(group) for group in exact.values())}.",
             f"- dHash successful memberships: {sum(len(group) for group in perceptual.values())}.",
             f"- Near-duplicate detection: {'enabled (Pillow, 64-bit dHash)' if near_enabled else 'SKIPPED (disabled or Pillow unavailable)' }.",
             f"- Hamming distance threshold: <= {threshold}.", f"- Hash/decode errors: {len(errors)}.", "",
             "## Split coverage", "", "| Split | Memberships |", "| --- | ---: |"]
    for split in ("train", "validation", "test"):
        lines.append(f"| {split} | {coverage[split]} |")
    lines += ["", "## Findings", "", "| Category | Pairs |", "| --- | ---: |"]
    lines.extend(f"| {key} | {counts[key]} |" for key in counts)
    lines += ["", "CROSS_SPLIT_DUPLICATE is a subset of exact and near pairs, not an additional pair count.",
              "CSV classification marks cross-split pairs; match_type retains EXACT_DUPLICATE or NEAR_DUPLICATE.",
              "SHA256 matches establish byte-identical content. dHash matches are review candidates, not proof of identity.",
              "dHash can miss crops/rotations and can match unrelated images with similar structure. Review before changing splits.", "",
              "## Leakage candidates", "", "| Splits | Pairs |", "| --- | ---: |"]
    for a, b in combinations(("train", "validation", "test"), 2):
        lines.append(f"| {a} / {b} | {split_pairs[tuple(sorted((a, b)))]} |")
    lines += ["", "Full pair details: [duplicates.csv](duplicates.csv).", "", "## Errors and limitations", ""]
    lines.extend(f"- {error}" for error in errors)
    if not errors:
        lines.append("No file read or decode errors.")
    if not all(coverage[split] for split in ("train", "validation", "test")) or errors or not near_enabled:
        lines.append("Coverage is incomplete; this report cannot establish absence of leakage.")
    (output_dir / "DUPLICATE_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return dict(counts)


def main() -> None:
    parser = argparse.ArgumentParser(description="Read-only SHA256/dHash duplicate and split leakage analysis.")
    parser.add_argument("--image-root", type=Path, default=PROJECT_ROOT / "data/train")
    parser.add_argument("--splits-dir", type=Path, default=PROJECT_ROOT / "outputs/splits")
    parser.add_argument("--folder-splits", action="store_true", help="Read train/val-or-validation/test folders under image root instead of CSVs")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs/reports")
    parser.add_argument("--threshold", type=int, default=5)
    parser.add_argument("--no-near", action="store_true")
    parser.add_argument("--workers", type=int, default=8, help="Concurrent read-only image hashing workers (default: 8)")
    parser.add_argument("--limit", type=int, help="Analyze only the first N sorted split memberships (for bounded dry runs)")
    args = parser.parse_args()
    if not 0 <= args.threshold <= 64:
        parser.error("--threshold must be between 0 and 64")
    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    try:
        records = load_records(args.image_root, None if args.folder_splits else args.splits_dir)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    total_records = len(records)
    if args.limit is not None:
        records = sorted(records, key=lambda row: str(row["path"]))[:args.limit]
    print(analyze(records, args.output_dir, args.threshold, not args.no_near, progress=True, workers=args.workers))
    with (args.output_dir / "DUPLICATE_REPORT.md").open("a", encoding="utf-8") as handle:
        handle.write(f"\n## Input scope\n\nImage root: `{args.image_root.resolve()}`.\n\n")
        handle.write("Split source: folders.\n" if args.folder_splits else f"Split source: CSVs in `{args.splits_dir.resolve()}` (img_name column).\n")
        if args.limit is not None:
            handle.write(f"Bounded run: analyzed {len(records)} of {total_records} split memberships (`--limit {args.limit}`).\n")
        else:
            handle.write("Only these split memberships are analyzed. Other datasets, including competition submission images, are outside this report.\n")


if __name__ == "__main__":
    main()
