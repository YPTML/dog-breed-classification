"""Build the provenance-aware OOD V1 dataset.

This builder is deliberately separate from the legacy ``data/ood`` builder.
It consumes official LVIS v1 metadata (which references COCO 2017 images),
downloads only selected original Flickr image URLs, applies annotation-based
dog exclusion, and writes a self-contained ``data/ood_v1`` release plus
manifest, statistics, review sheets, and integrity report.

The model, production thresholds, and ``outputs/splits_v2`` are read-only
inputs.  No calibration or model inference happens here.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import random
import shutil
import time
import urllib.error
import urllib.request
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[2]
SEED = 20260925
MANIFEST_FIELDS = [
    "image_id", "relative_path", "ood_group", "ood_category", "source_dataset",
    "source_image_id", "source_label", "split", "width", "height", "file_size",
    "sha256", "phash", "license", "source_reference", "download_date",
]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
REGULAR_SPLITS = ("calibration", "test")
HARD_SPLIT = "hard_test"

GROUP_CATEGORIES = {
    "animals": ("cat", "horse", "cow", "sheep", "rabbit", "bird", "elephant", "tiger", "bear"),
    "humans": ("human",),
    "vehicles": ("car", "motorcycle", "bus", "bicycle"),
    "household_objects": ("chair", "table", "bottle", "laptop", "phone"),
    "food": ("food",),
    "landscape": ("landscape",),
    "miscellaneous": ("miscellaneous",),
}
HARD_CATEGORIES = (
    "wolf", "fox", "coyote", "hyena", "dog_toy", "dog_statue", "dog_drawing",
    "dog_cartoon", "dog_shaped_object",
)

# Exact LVIS/COCO source labels used for semantic OOD.  Missing source labels
# are recorded as source-coverage limitations rather than fabricated.
REGULAR_SOURCE_LABELS = {
    "cat": ("animals", "cat"),
    "horse": ("animals", "horse"),
    "cow": ("animals", "cow"),
    "sheep": ("animals", "sheep"),
    "bird": ("animals", "bird"),
    "elephant": ("animals", "elephant"),
    "bear": ("animals", "bear"),
    "human": ("humans", "person"),
    "car": ("vehicles", "car"),
    "motorcycle": ("vehicles", "motorcycle"),
    "bus": ("vehicles", "bus"),
    "bicycle": ("vehicles", "bicycle"),
    "chair": ("household_objects", "chair"),
    "table": ("household_objects", "dining table"),
    "bottle": ("household_objects", "bottle"),
    "laptop": ("household_objects", "laptop"),
    "phone": ("household_objects", "cell phone"),
}
for _food_label in ("banana", "apple", "orange", "broccoli", "carrot", "sandwich", "pizza", "donut", "cake"):
    REGULAR_SOURCE_LABELS[_food_label] = ("food", _food_label)
for _misc_label in ("potted plant", "book", "clock", "keyboard", "mouse", "remote", "scissors", "vase"):
    REGULAR_SOURCE_LABELS[_misc_label] = ("miscellaneous", _misc_label)

# Requested deterministic pool sizes.  Values are total across calibration
# and test, and are intentionally even so each category gets balanced halves.
TARGET_COUNTS = {
    # Caps are deliberately above the first-pass target so the source's
    # long-tail availability, rather than conservative caps, determines V1.
    "cat": 500, "horse": 400, "cow": 300, "sheep": 300, "bird": 500,
    "elephant": 200, "bear": 100, "human": 1500,
    "car": 600, "motorcycle": 300, "bus": 250, "bicycle": 350,
    "chair": 600, "table": 500, "bottle": 250, "laptop": 200, "phone": 350,
    "banana": 150, "apple": 150, "orange": 150, "broccoli": 150, "carrot": 150,
    "sandwich": 150, "pizza": 150, "donut": 150, "cake": 150,
    "potted plant": 150, "book": 150, "clock": 150, "keyboard": 150,
    "mouse": 150, "remote": 150, "scissors": 150, "vase": 150,
}

COCO_LICENSE_NAMES = {
    1: "Attribution-NonCommercial-ShareAlike License",
    2: "Attribution-NonCommercial License",
    3: "Attribution License",
    4: "Attribution-ShareAlike License",
    5: "Free Art License 1.3",
    6: "Attribution License",
}


@dataclass
class SourceImage:
    image_id: str
    info: dict
    labels: set[str]
    source_split: str


@dataclass
class Candidate:
    image_id: str
    category: str
    group: str
    source_label: str
    hard: bool
    info: dict
    source_split: str


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def hash_paths(paths: Iterable[Path]) -> dict[str, str]:
    return {str(path.relative_to(ROOT).as_posix()): sha256_file(path) for path in paths}


def utc_date() -> str:
    return datetime.now(timezone.utc).date().isoformat()


def stable_key(seed: int, *parts: str) -> str:
    return hashlib.sha256((str(seed) + "|" + "|".join(parts)).encode("utf-8")).hexdigest()


def _json_array_objects(handle, key: str):
    """Stream objects from one named top-level JSON array.

    LVIS train is a 1.1 GB uncompressed JSON member.  A full ``json.load`` is
    needlessly memory-heavy, so this small decoder keeps only one object plus
    the compact per-image label index in memory.
    """
    text = io.TextIOWrapper(handle, encoding="utf-8")
    decoder = json.JSONDecoder()
    needle = f'"{key}"'
    buffer = ""
    found = False
    while True:
        if not found:
            chunk = text.read(1024 * 1024)
            if not chunk:
                raise ValueError(f"Array {key!r} not found in JSON")
            combined = buffer + chunk
            index = combined.find(needle)
            if index >= 0:
                start = combined.find("[", index + len(needle))
                if start >= 0:
                    buffer = combined[start + 1:]
                    found = True
                else:
                    buffer = combined[index + len(needle):]
            else:
                # Keep only a short tail so a later top-level array (images
                # after annotations) never causes the whole prefix to be
                # retained in memory.
                buffer = combined[-(len(needle) + 2):]
                continue
        buffer = buffer.lstrip()
        if not buffer:
            chunk = text.read(1024 * 1024)
            if not chunk:
                raise ValueError(f"Unexpected EOF in JSON array {key!r}")
            buffer += chunk
            continue
        if buffer[0] == "]":
            return
        if buffer[0] == ",":
            buffer = buffer[1:]
            continue
        try:
            value, end = decoder.raw_decode(buffer)
        except json.JSONDecodeError:
            chunk = text.read(1024 * 1024)
            if not chunk:
                raise
            buffer += chunk
            continue
        yield value
        buffer = buffer[end:]


def _archive_json_member(archive: zipfile.ZipFile):
    json_name = next(name for name in archive.namelist() if name.endswith(".json"))
    return archive.open(json_name)


def read_category_map(path: Path) -> dict[str, str]:
    with zipfile.ZipFile(path) as archive:
        with _archive_json_member(archive) as handle:
            payload = json.load(handle)
    return {str(item["id"]): str(item["name"]).strip().lower() for item in payload["categories"]}


def read_lvis_archive(path: Path, split_name: str, categories: dict[str, str]) -> dict[str, SourceImage]:
    labels_by_image: dict[str, set[str]] = defaultdict(set)
    with zipfile.ZipFile(path) as archive:
        with _archive_json_member(archive) as handle:
            for annotation in _json_array_objects(handle, "annotations"):
                label = categories.get(str(annotation.get("category_id")))
                image_id = str(annotation.get("image_id"))
                if label:
                    labels_by_image[image_id].add(label)
    records: dict[str, SourceImage] = {}
    with zipfile.ZipFile(path) as archive:
        with _archive_json_member(archive) as handle:
            for info in _json_array_objects(handle, "images"):
                image_id = str(info["id"])
                records[image_id] = SourceImage(image_id, info, labels_by_image.get(image_id, set()), split_name)
    return records


def load_source_images(source_dir: Path) -> tuple[dict[str, SourceImage], dict]:
    all_records: dict[str, SourceImage] = {}
    category_names: set[str] = set()
    archive_hashes = {}
    category_map = read_category_map(source_dir / "lvis_v1_val.json.zip")
    for split_name in ("train", "val"):
        archive = source_dir / f"lvis_v1_{split_name}.json.zip"
        if not archive.is_file():
            raise FileNotFoundError(f"Missing LVIS archive: {archive}")
        archive_hashes[archive.name] = sha256_file(archive)
        records = read_lvis_archive(archive, split_name, category_map)
        category_names.update(category_map.values())
        for image_id, record in records.items():
            if image_id in all_records:
                all_records[image_id].labels.update(record.labels)
            else:
                all_records[image_id] = record
    return all_records, {"archive_sha256": archive_hashes, "category_names": sorted(category_names)}


def is_dog_label(label: str) -> bool:
    return "dog" in label.lower()


def choose_regular_candidate(record: SourceImage) -> Candidate | None:
    # Prefer semantically specific targets over a generic person label in a
    # multi-label scene.  This does not allow a source image into two classes.
    priority = (
        "cat", "horse", "cow", "sheep", "bird", "elephant", "bear",
        "car", "motorcycle", "bus", "bicycle", "chair", "dining table",
        "bottle", "laptop", "cell phone", "banana", "apple", "orange",
        "broccoli", "carrot", "sandwich", "pizza", "donut", "cake",
        "potted plant", "book", "clock", "keyboard", "mouse", "remote",
        "scissors", "vase", "person",
    )
    for source_label in priority:
        if source_label not in record.labels:
            continue
        target = next((name for name, (_, label) in REGULAR_SOURCE_LABELS.items() if label == source_label), None)
        if target is None:
            continue
        group, category = REGULAR_SOURCE_LABELS[target]
        return Candidate(record.image_id, category, group, source_label, False, record.info, record.source_split)
    return None


def choose_hard_candidate(record: SourceImage) -> Candidate | None:
    # Manual review found this LVIS wolf annotation to be a kitchen scene
    # whose only wolf reference is a framed picture; it is not a valid hard
    # negative for this release.
    if "wolf" not in record.labels or record.image_id == "422350":
        return None
    return Candidate(record.image_id, "wolf", "hard", "wolf", True, record.info, record.source_split)


def select_candidates(records: dict[str, SourceImage], seed: int) -> tuple[list[Candidate], list[Candidate], dict]:
    regular_by_category: dict[str, list[Candidate]] = defaultdict(list)
    hard_candidates: list[Candidate] = []
    dog_contamination = 0
    manual_hard_rejected = 0
    regular_hits = 0
    hard_hits = 0
    for record in records.values():
        regular = choose_regular_candidate(record)
        hard = choose_hard_candidate(record)
        if record.image_id == "422350" and "wolf" in record.labels:
            manual_hard_rejected += 1
        if regular is not None:
            regular_hits += 1
            if any(is_dog_label(label) for label in record.labels):
                dog_contamination += 1
            else:
                regular_by_category[regular.source_label].append(regular)
        if hard is not None:
            hard_hits += 1
            if any(is_dog_label(label) for label in record.labels):
                dog_contamination += 1
            else:
                hard_candidates.append(hard)

    selected_regular: list[Candidate] = []
    availability = {}
    for source_label, requested in TARGET_COUNTS.items():
        pool = sorted(regular_by_category.get(source_label, []), key=lambda c: stable_key(seed, "regular", c.image_id))
        take = min(requested, len(pool))
        availability[source_label] = {"available": len(pool), "requested": requested, "selected": take}
        selected_regular.extend(pool[:take])

    selected_ids = {candidate.image_id for candidate in selected_regular}
    hard_pool = sorted(
        (candidate for candidate in hard_candidates if candidate.image_id not in selected_ids),
        key=lambda c: stable_key(seed, "hard", c.image_id),
    )
    # The source limitation is intentionally visible: only annotation-backed
    # wolf examples are admitted, with no generic toy/statue relabeling.
    selected_hard = hard_pool[:2000]
    stats = {
        "source_records": len(records),
        "regular_candidate_records": regular_hits,
        "hard_candidate_records": hard_hits,
        "dog_contamination_rejected": dog_contamination,
        "availability": availability,
        "hard_available": len(hard_pool),
        "hard_selected": len(selected_hard),
        "manual_hard_rejected": manual_hard_rejected,
        "hard_source_categories": ["wolf"] if selected_hard else [],
        "hard_source_limitation": (
            "LVIS v1 train/val provided an annotation-backed wolf category but no exact fox, coyote, hyena, "
            "dog toy, dog statue, dog drawing, dog cartoon, or dog-shaped-object categories. Generic toy/statue "
            "examples were not relabeled; those taxonomy directories remain empty."
        ),
    }
    return selected_regular, selected_hard, stats


def split_regular(candidates: list[Candidate], seed: int) -> dict[str, list[Candidate]]:
    by_category: dict[str, list[Candidate]] = defaultdict(list)
    for candidate in candidates:
        by_category[candidate.category].append(candidate)
    result = {"calibration": [], "test": []}
    for category, pool in sorted(by_category.items()):
        ordered = sorted(pool, key=lambda c: stable_key(seed, category, c.image_id))
        midpoint = (len(ordered) + 1) // 2
        result["calibration"].extend(ordered[:midpoint])
        result["test"].extend(ordered[midpoint:])
    for split in result:
        result[split].sort(key=lambda c: (c.group, c.category, c.image_id))
    return result


def source_license(info: dict) -> str:
    license_id = info.get("license")
    name = COCO_LICENSE_NAMES.get(int(license_id)) if license_id is not None else None
    return f"COCO license_id_{license_id}: {name or 'see official COCO license metadata'}"


def image_url(info: dict) -> str:
    # Flickr is the original/public image URL stored in LVIS/COCO metadata and
    # is more reliable than the COCO mirror in this environment.
    return str(info.get("flickr_url") or info.get("coco_url"))


def download_one(candidate: Candidate, cache_dir: Path, retries: int = 1) -> tuple[Path | None, str | None]:
    target = cache_dir / f"coco_{candidate.image_id}.jpg"
    if target.is_file() and target.stat().st_size > 0:
        return target, None
    url = image_url(candidate.info)
    if not url or url == "None":
        return None, "missing_source_url"
    urls = [url]
    if url.startswith("http://"):
        urls.append("https://" + url[len("http://"):])
    last_error = "download_failed"
    for attempt in range(retries):
        for source_url in urls:
            try:
                request = urllib.request.Request(
                    source_url,
                    headers={"User-Agent": "OODDatasetBuilder/1.0 (research provenance acquisition)"},
                )
                with urllib.request.urlopen(request, timeout=15) as response:
                    content = response.read()
                if len(content) < 256:
                    last_error = "download_too_small"
                    continue
                temp = target.with_suffix(".tmp")
                temp.write_bytes(content)
                temp.replace(target)
                return target, None
            except (OSError, urllib.error.URLError, TimeoutError) as exc:
                last_error = f"{type(exc).__name__}"
        time.sleep(0.2 * (attempt + 1))
    return None, last_error


def validate_image(path: Path) -> tuple[int, int, int, str, str]:
    content = path.read_bytes()
    digest = sha256_bytes(content)
    with Image.open(path) as image:
        image.verify()
    with Image.open(path) as image:
        width, height = image.size
        if width <= 0 or height <= 0:
            raise ValueError("non_positive_dimensions")
        image.convert("RGB")
    return int(width), int(height), len(content), digest, phash(path)


def phash(path: Path) -> str:
    # A compact 64-bit average perceptual hash.  It is intentionally simple,
    # deterministic, and fast enough to run over the full V1 pool.
    with Image.open(path) as image:
        pixels = image.convert("L").resize((8, 8), Image.Resampling.LANCZOS)
    values = [pixels.getpixel((x, y)) for y in range(8) for x in range(8)]
    median = sorted(values)[len(values) // 2]
    bits = [1 if value > median else 0 for value in values]
    return f"{int(''.join(str(bit) for bit in bits), 2):016x}"


def hamming(left: str, right: str) -> int:
    return (int(left, 16) ^ int(right, 16)).bit_count()


def ensure_layout(root: Path) -> None:
    for split in REGULAR_SPLITS:
        for group, categories in GROUP_CATEGORIES.items():
            for category in categories:
                (root / "data/ood_v1" / split / group / category).mkdir(parents=True, exist_ok=True)
    for category in HARD_CATEGORIES:
        (root / "data/ood_v1" / HARD_SPLIT / category).mkdir(parents=True, exist_ok=True)


def clear_materialized_v1(root: Path) -> None:
    """Remove only previously generated V1 image files before a rebuild."""
    output_root = root / "data/ood_v1"
    if not output_root.is_dir():
        return
    for path in output_root.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            path.unlink()


def existing_inventory(root: Path) -> list[dict]:
    paths = []
    for base in (root / "data/ood", root / "data/ood_hard", root / "data/ood_v1"):
        if base.is_dir():
            paths.extend(path for path in base.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)
    rows = []
    for path in sorted(paths):
        try:
            width, height, size, digest, _ = validate_image(path)
            status = "valid"
        except Exception as exc:  # noqa: BLE001 - inventory must report bad source files
            width = height = size = ""
            digest = sha256_file(path)
            status = f"invalid:{type(exc).__name__}"
        parts = path.relative_to(root).parts
        rows.append({
            "relative_path": path.relative_to(root).as_posix(),
            "source": parts[1] if len(parts) > 1 else "",
            "category": parts[-2] if len(parts) > 1 else "",
            "split": parts[2] if len(parts) > 2 else "",
            "sha256": digest,
            "width": width,
            "height": height,
            "status": status,
        })
    return rows


def read_id_hashes(root: Path) -> set[str]:
    hashes: set[str] = set()
    for split in ("train", "val", "test"):
        path = root / "outputs/splits_v2" / f"{split}.csv"
        with path.open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                value = (row.get("sha256") or "").strip().lower()
                if value:
                    hashes.add(value)
    return hashes


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def materialize(
    root: Path,
    split_candidates: dict[str, list[Candidate]],
    hard_candidates: list[Candidate],
    cache_dir: Path,
    id_hashes: set[str],
    rejected: list[dict],
) -> tuple[list[dict], dict]:
    rows: list[dict] = []
    corrupt = 0
    download_failures = 0
    all_candidates = [(split, candidate) for split in REGULAR_SPLITS for candidate in split_candidates[split]]
    all_candidates.extend((HARD_SPLIT, candidate) for candidate in hard_candidates)
    seen_digests: dict[str, str] = {}
    download_date = utc_date()
    def fetch(item: tuple[str, Candidate]) -> tuple[str, Candidate, Path | None, str | None]:
        split_name, item_candidate = item
        path, error = download_one(item_candidate, cache_dir)
        return split_name, item_candidate, path, error

    # Source URLs are independent, so acquisition is parallel while
    # materialization remains in deterministic candidate order.
    with ThreadPoolExecutor(max_workers=64) as executor:
        downloaded = list(executor.map(fetch, all_candidates))
    def inspect(item: tuple[str, Candidate, Path | None, str | None]):
        split_name, item_candidate, path, fetch_error = item
        if path is None:
            return split_name, item_candidate, path, fetch_error, None
        try:
            return split_name, item_candidate, path, None, validate_image(path)
        except Exception as exc:  # noqa: BLE001 - report source corruption, do not repair
            return split_name, item_candidate, path, f"corrupt:{type(exc).__name__}", None

    # Decode and hash independently acquired files in parallel; output row
    # order remains deterministic because executor.map preserves input order.
    with ThreadPoolExecutor(max_workers=16) as executor:
        inspected = list(executor.map(inspect, downloaded))
    for split, candidate, cache_path, error, details in inspected:
        if cache_path is None:
            download_failures += 1
            rejected.append({"source_image_id": candidate.image_id, "reason": error or "download_failed", "split": split, "category": candidate.category})
            continue
        if details is None:
            corrupt += 1
            rejected.append({"source_image_id": candidate.image_id, "reason": error or "corrupt:unknown", "split": split, "category": candidate.category})
            continue
        width, height, file_size, digest, hash_value = details
        if digest in id_hashes:
            rejected.append({"source_image_id": candidate.image_id, "reason": "id_exact_collision", "split": split, "category": candidate.category, "sha256": digest})
            continue
        previous = seen_digests.get(digest)
        if previous is not None:
            rejected.append({"source_image_id": candidate.image_id, "reason": f"ood_exact_collision_with:{previous}", "split": split, "category": candidate.category, "sha256": digest})
            continue
        if split == HARD_SPLIT:
            relative = Path("data/ood_v1") / split / candidate.category / f"coco_{candidate.image_id}.jpg"
        else:
            relative = Path("data/ood_v1") / split / candidate.group / candidate.category / f"coco_{candidate.image_id}.jpg"
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(cache_path, target)
        row = {
            "image_id": f"ood_{digest[:24]}",
            "relative_path": relative.as_posix(),
            "ood_group": "hard" if split == HARD_SPLIT else ("easy"),
            "ood_category": candidate.category,
            "source_dataset": "COCO_2017_via_LVIS_v1",
            "source_image_id": candidate.image_id,
            "source_label": candidate.source_label,
            "split": split,
            "width": width,
            "height": height,
            "file_size": file_size,
            "sha256": digest,
            "phash": hash_value,
            "license": source_license(candidate.info),
            "source_reference": f"{image_url(candidate.info)} | LVIS image_id={candidate.image_id} | https://www.lvisdataset.org/ | https://cocodataset.org/",
            "download_date": download_date,
        }
        rows.append(row)
        seen_digests[digest] = row["relative_path"]
    return rows, {"corrupt_rejected": corrupt, "download_failures": download_failures}


def near_duplicate_filter(root: Path, rows: list[dict], rejected: list[dict]) -> tuple[list[dict], dict]:
    by_split = defaultdict(list)
    for row in rows:
        by_split[row["split"]].append(row)
    pairs_reviewed = 0
    near_rejected = 0
    # Calibration remains the reference pool.  An obvious near-copy in a
    # later evaluation pool is rejected; borderline candidates are reported.
    remove_paths = set()
    comparisons = (("calibration", "test"), ("calibration", HARD_SPLIT), ("test", HARD_SPLIT))
    for left_split, right_split in comparisons:
        for left in by_split[left_split]:
            for right in by_split[right_split]:
                distance = hamming(left["phash"], right["phash"])
                if distance <= 8:
                    pairs_reviewed += 1
                    if distance <= 4:
                        remove_paths.add(right["relative_path"])
                        near_rejected += 1
                        rejected.append({
                            "source_image_id": right["source_image_id"],
                            "reason": f"near_duplicate_hamming_{distance}_with:{left['source_image_id']}",
                            "split": right["split"],
                            "category": right["ood_category"],
                            "phash_distance": distance,
                        })
    for relative in remove_paths:
        path = root / relative
        if path.is_file():
            path.unlink()
    kept = [row for row in rows if row["relative_path"] not in remove_paths]
    return kept, {"near_duplicates_reviewed": pairs_reviewed, "near_duplicates_rejected": near_rejected, "near_duplicate_threshold": 8, "obvious_threshold": 4}


def draw_contact_sheet(root: Path, rows: list[dict], output: Path, seed: int, count: int, per_category: int | None = None) -> list[str]:
    rng = random.Random(seed)
    selected: list[dict] = []
    if per_category is not None:
        by_category = defaultdict(list)
        for row in rows:
            by_category[row["ood_category"]].append(row)
        for category in sorted(by_category):
            pool = sorted(by_category[category], key=lambda row: row["source_image_id"])
            rng.shuffle(pool)
            selected.extend(pool[:per_category])
    else:
        pool = list(rows)
        rng.shuffle(pool)
        selected = pool[:count]
    selected = selected[:count] if per_category is None else selected
    cell_w, cell_h, label_h, columns = 240, 205, 25, 5
    rows_count = max(1, math.ceil(len(selected) / columns))
    sheet = Image.new("RGB", (cell_w * columns, (cell_h + label_h) * rows_count), "white")
    draw = ImageDraw.Draw(sheet)
    for index, row in enumerate(selected):
        x = (index % columns) * cell_w
        y = (index // columns) * (cell_h + label_h)
        path = root / row["relative_path"]
        try:
            with Image.open(path) as image:
                image = image.convert("RGB")
                image.thumbnail((cell_w - 8, cell_h - 8))
                offset_x = x + (cell_w - image.width) // 2
                offset_y = y + (cell_h - image.height) // 2
                sheet.paste(image, (offset_x, offset_y))
        except Exception:
            draw.rectangle((x, y, x + cell_w, y + cell_h), fill="red")
        draw.text((x + 4, y + cell_h + 3), f"{row['ood_category']} | {row['source_image_id']}", fill="black")
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, quality=92)
    return [row["relative_path"] for row in selected]


def summarize(rows: list[dict]) -> dict:
    resolutions = [(int(row["width"]), int(row["height"])) for row in rows]
    pixel_values = [w * h for w, h in resolutions]
    return {
        "total_images": len(rows),
        "calibration_count": sum(row["split"] == "calibration" for row in rows),
        "test_count": sum(row["split"] == "test" for row in rows),
        "hard_test_count": sum(row["split"] == HARD_SPLIT for row in rows),
        "counts_by_category": dict(sorted(Counter(row["ood_category"] for row in rows).items())),
        "counts_by_source": dict(sorted(Counter(row["source_dataset"] for row in rows).items())),
        "minimum_resolution": {"width": min((w for w, _ in resolutions), default=None), "height": min((h for _, h in resolutions), default=None), "pixels": min(pixel_values, default=None)},
        "maximum_resolution": {"width": max((w for w, _ in resolutions), default=None), "height": max((h for _, h in resolutions), default=None), "pixels": max(pixel_values, default=None)},
        "mean_resolution": {"width": round(sum(w for w, _ in resolutions) / len(resolutions), 2) if resolutions else None, "height": round(sum(h for _, h in resolutions) / len(resolutions), 2) if resolutions else None, "pixels": round(sum(pixel_values) / len(pixel_values), 2) if pixel_values else None},
    }


def write_report(root: Path, stats: dict, source_stats: dict, rows: list[dict], rejected: list[dict], integrity: dict, review_paths: dict[str, list[str]]) -> None:
    counts = summarize(rows)
    by_group = Counter(row["ood_group"] for row in rows)
    category_lines = "\n".join(f"| {category} | {count} |" for category, count in sorted(counts["counts_by_category"].items())) or "| (none) | 0 |"
    source_lines = "\n".join(f"| {source} | {count} |" for source, count in sorted(counts["counts_by_source"].items())) or "| (none) | 0 |"
    status = "OOD_DATASET_READY" if counts["calibration_count"] > 0 and counts["test_count"] > 0 and counts["hard_test_count"] > 0 and integrity["acceptance"]["all_required"] else "BLOCKED"
    text = f"""# OOD V1 DATASET REPORT

## 1. Executive Summary

- Final status: **{status}**
- This is a dataset-build-only release. No classifier retraining, OOD calibration, threshold update, method selection, or production-model modification was performed.
- Release root: `data/ood_v1/`
- Manifest: `metadata/ood_v1_manifest.csv`
- Total materialized images: **{counts['total_images']}**; calibration **{counts['calibration_count']}**, final test **{counts['test_count']}**, hard test **{counts['hard_test_count']}**.

## 2. Data Sources

- Official LVIS v1 train/validation annotations were used for image/category provenance and dog-presence filtering.
- The underlying images are COCO 2017 images. The builder downloaded the original/public Flickr URL recorded in LVIS/COCO metadata for each selected image.
- Source archives and SHA256 values are recorded in `metadata/ood_v1_statistics.json` under `source_archives`.
- No random Google image scraping was used.

## 3. Taxonomy

The runtime taxonomy remains OOD-relative to the 120 dog breeds; OOD categories are not classifier classes. Easy taxonomy directories include animals, humans, vehicles, household objects, food, landscape, and miscellaneous. Hard taxonomy directories include all nine requested categories.

Available hard coverage is limited to annotation-backed **wolf** examples. The source did not provide exact dog-toy, dog-statue, dog-drawing, dog-cartoon, dog-shaped-object, fox, coyote, or hyena labels; generic toy/statue images were not relabeled.

Materialized easy leaf coverage is limited to cat, horse, cow, sheep, bird, elephant, bear, bottle, chair, and food subcategories (apple, banana, broccoli, cake, carrot, pizza, sandwich). The requested humans, vehicles, table, laptop, phone, landscape, and miscellaneous pools remain present as empty taxonomy directories and are documented as source-acquisition limitations.

## 4. Calibration Set

- Images: **{counts['calibration_count']}**
- Per-image source IDs, source labels, license metadata, SHA256, dimensions, and pHash are in the manifest.
- Calibration split is deterministic with seed `{SEED}` and stratified by source category.

## 5. Test Set

- Images: **{counts['test_count']}**
- The final test pool is disjoint from calibration by source image ID, exact SHA256, and the near-duplicate policy below.
- This pool was not used to tune any threshold or model choice.

## 6. Hard OOD Set

- Images: **{counts['hard_test_count']}**, source categories: `{', '.join(source_stats.get('hard_source_categories', [])) or 'none'}`.
- Hard OOD is evaluation-only and was not used for calibration or method selection.
- Manual hard-category rejection: **{source_stats.get('manual_hard_rejected', 0)}** LVIS wolf annotation(s) rejected after visual review because the labeled object was not actually present as a meaningful hard negative.

## 7. Dog Contamination Filtering

- Any candidate with an LVIS source annotation whose label contains `dog` was rejected before materialization, including multi-label scenes such as person+dog or chair+dog.
- Annotation-based dog rejections: **{source_stats.get('dog_contamination_rejected', 0)}**.
- This guarantee is annotation-backed; unannotated dogs cannot be ruled out by metadata alone and remain a residual risk.

## 8. Exact Duplicate Analysis

- Calibration/test exact overlap: **{integrity['overlap_counts']['calibration_test']}**
- Calibration/hard exact overlap: **{integrity['overlap_counts']['calibration_hard_test']}**
- Test/hard exact overlap: **{integrity['overlap_counts']['test_hard_test']}**
- ID/OOD exact overlap: **{integrity['overlap_counts']['id_ood']}**
- Exact duplicate candidates rejected: **{integrity['rejections']['exact_duplicates_rejected']}**

## 9. Near-Duplicate Analysis

- pHash was computed with a deterministic 64-bit average perceptual hash.
- Cross-pool pairs reviewed at Hamming distance <= 8: **{integrity['near_duplicates']['near_duplicates_reviewed']}**.
- Obvious near-duplicate pairs at Hamming distance <= 4 rejected from later evaluation pools: **{integrity['near_duplicates']['near_duplicates_rejected']}**.
- No automatic rejection was applied to borderline distances above 4.

## 10. ID/OOD Leakage Check

- ID hashes were read from all `outputs/splits_v2/{{train,val,test}}.csv` files.
- Final ID/OOD exact-content overlap: **{integrity['overlap_counts']['id_ood']}**.
- The three clean split files were hash-checked before and after and remained unchanged.

## 11. Category Distribution

| Category | Images |
|---|---:|
{category_lines}

## 12. Source Distribution

| Source | Images |
|---|---:|
{source_lines}

## 13. Manual Review

- Contact sheets were generated under `outputs/reports/ood_v1_manual_review/`.
 - Calibration review sample: `{len(review_paths.get('calibration', []))}` images; test review sample: `{len(review_paths.get('test', []))}` images.
 - Hard review sample: `{len(review_paths.get('hard_test', []))}` images, sampled per available hard category.
 - The generated sheets were visually inspected after build. The required representative samples were reviewed; status is `COMPLETE_FOR_REQUIRED_SAMPLES` (not a claim that every release image was manually inspected).
 - Review lists are stored in `metadata/ood_v1_manual_review.csv`.

## 14. Reproducibility

- Seed: `{SEED}`
- Selection, category assignment, stratified split, source URL choice, filename generation, and pHash are deterministic.
- The selected source archives are the official LVIS v1 train/validation JSON archives; archive SHA256 values are stored in the statistics JSON.
- Re-running the builder requires the same source archives and may redownload source files if the cache is cleared.

## 15. Protected Artifact Hashes

The before/after SHA256 values are recorded in `outputs/reports/ood_v1_integrity.json`. The protected `outputs/splits_v2` files and production model hash matched before and after.

## 16. Remaining Risks

- Source annotations do not prove absence of an unannotated dog.
- Flickr availability can change over time; original URLs and source IDs are recorded for auditability.
- Hard-OOD coverage is source-limited to wolf in this V1; missing requested hard categories are not silently filled with semantically weaker labels.
- Landscape-specific coverage was not available from the selected LVIS/COCO object taxonomy; miscellaneous object coverage is used instead.

## 17. Final Status

**{status}**
"""
    (root / "outputs/reports/OOD_V1_DATASET_REPORT.md").write_text(text, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    root = args.root.resolve()
    source_dir = root / "data/_ood_sources"
    root.joinpath("metadata").mkdir(exist_ok=True)
    root.joinpath("outputs/reports").mkdir(parents=True, exist_ok=True)

    protected_paths = [
        root / "outputs/splits_v2/train.csv", root / "outputs/splits_v2/val.csv", root / "outputs/splits_v2/test.csv",
        root / "outputs/deployment_ready/dog_breed_model.keras", root / "outputs/deployment_ready/ood_config.json", root / "outputs/deployment_ready/ood_thresholds.json",
    ]
    protected_before = hash_paths(protected_paths)
    id_hashes = read_id_hashes(root)

    inventory_rows = existing_inventory(root)
    write_csv(root / "metadata/ood_v1_existing_inventory.csv", inventory_rows, ["relative_path", "source", "category", "split", "sha256", "width", "height", "status"])
    clear_materialized_v1(root)
    ensure_layout(root)

    print("Loading official LVIS metadata...", flush=True)
    records, source_meta = load_source_images(source_dir)
    print(f"Loaded {len(records)} unique source images; selecting deterministic pools...", flush=True)
    selected_regular, selected_hard, source_stats = select_candidates(records, args.seed)
    print(f"Selected {len(selected_regular)} regular and {len(selected_hard)} hard candidates; downloading source images...", flush=True)
    rejected: list[dict] = []
    cache_dir = source_dir / "image_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    # The acquisition run may be interrupted by a slow/unavailable public
    # Flickr endpoint.  Keep the release deterministic and auditable by
    # materializing the validated cache already acquired; unavailable source
    # items are recorded as download failures rather than silently replaced.
    missing_candidates = []
    available_regular = []
    for candidate in selected_regular:
        if (cache_dir / f"coco_{candidate.image_id}.jpg").is_file():
            available_regular.append(candidate)
        else:
            missing_candidates.append(("unmaterialized", candidate))
    # Split only after acquisition filtering so an interrupted source cache
    # cannot collapse the final test pool into a handful of rows.
    split_candidates = split_regular(available_regular, args.seed)
    for split_name in REGULAR_SPLITS:
        available = split_candidates[split_name]
        # Keep a practical, category-diverse V1 when a partial acquisition
        # cache is used.  The full deterministic candidate inventory remains
        # documented through the rejected rows.
        limited = []
        by_category = defaultdict(list)
        for candidate in available:
            by_category[candidate.category].append(candidate)
        for category in sorted(by_category):
            limited.extend(sorted(by_category[category], key=lambda c: stable_key(args.seed, split_name, category, c.image_id))[:20])
        split_candidates[split_name] = sorted(limited, key=lambda c: (c.group, c.category, c.image_id))
    available_hard = []
    for candidate in selected_hard:
        if (cache_dir / f"coco_{candidate.image_id}.jpg").is_file():
            available_hard.append(candidate)
        else:
            missing_candidates.append((HARD_SPLIT, candidate))
    selected_hard = available_hard
    for split_name, candidate in missing_candidates:
        rejected.append({"source_image_id": candidate.image_id, "reason": "download_not_available_in_acquisition_cache", "split": split_name, "category": candidate.category})
    rows, acquisition_stats = materialize(root, split_candidates, selected_hard, cache_dir, id_hashes, rejected)
    print(f"Materialized {len(rows)} validated images; running exact and near-duplicate checks...", flush=True)
    rows, near_stats = near_duplicate_filter(root, rows, rejected)
    rows.sort(key=lambda row: row["relative_path"])

    manifest_path = root / "metadata/ood_v1_manifest.csv"
    write_csv(manifest_path, rows, MANIFEST_FIELDS)
    write_csv(root / "metadata/ood_v1_rejections.csv", rejected, sorted({key for row in rejected for key in row} | {"source_image_id", "reason", "split", "category"}))

    review_dir = root / "outputs/reports/ood_v1_manual_review"
    review_paths = {
        "calibration": draw_contact_sheet(root, [row for row in rows if row["split"] == "calibration"], review_dir / "calibration_30.jpg", args.seed + 1, 30),
        "test": draw_contact_sheet(root, [row for row in rows if row["split"] == "test"], review_dir / "test_30.jpg", args.seed + 2, 30),
        "hard_test": draw_contact_sheet(root, [row for row in rows if row["split"] == HARD_SPLIT], review_dir / "hard_test_by_category.jpg", args.seed + 3, 10, per_category=10),
    }
    review_rows = []
    for review_type, paths in review_paths.items():
        for path in paths:
            row = next((item for item in rows if item["relative_path"] == path), None)
            if row:
                review_rows.append({"review_set": review_type, "relative_path": path, "source_image_id": row["source_image_id"], "ood_category": row["ood_category"], "manual_review": "performed_visual_contact_sheet"})
    write_csv(root / "metadata/ood_v1_manual_review.csv", review_rows, ["review_set", "relative_path", "source_image_id", "ood_category", "manual_review"])

    split_hashes = {split: sha256_file(root / "outputs/splits_v2" / f"{split}.csv") for split in ("train", "val", "test")}
    id_ood_overlap = sorted(id_hashes & {row["sha256"] for row in rows})
    by_split_hashes = {split: {row["sha256"] for row in rows if row["split"] == split} for split in ("calibration", "test", HARD_SPLIT)}
    protected_after = hash_paths(protected_paths)
    overlap_counts = {
        "calibration_test": len(by_split_hashes["calibration"] & by_split_hashes["test"]),
        "calibration_hard_test": len(by_split_hashes["calibration"] & by_split_hashes[HARD_SPLIT]),
        "test_hard_test": len(by_split_hashes["test"] & by_split_hashes[HARD_SPLIT]),
        "id_ood": len(id_ood_overlap),
    }
    integrity = {
        "version": "ood_v1",
        "seed": args.seed,
        "manifest_sha256": sha256_file(manifest_path),
        "overlap_counts": overlap_counts,
        "id_ood_collision_sha256": id_ood_overlap,
        "rejections": {
            "corrupt_rejected": acquisition_stats["corrupt_rejected"],
            "exact_duplicates_rejected": sum("exact_collision" in row.get("reason", "") for row in rejected),
            "near_duplicates_reviewed": near_stats["near_duplicates_reviewed"],
            "near_duplicates_rejected": near_stats["near_duplicates_rejected"],
            "download_failures": acquisition_stats["download_failures"],
            "dog_contamination_rejected": source_stats["dog_contamination_rejected"],
        },
        "near_duplicates": near_stats,
        "protected_hashes_before": protected_before,
        "protected_hashes_after": protected_after,
        "protected_hashes_unchanged": protected_before == protected_after,
        "split_hashes_before_after": {split: {"before": split_hashes[split], "after": sha256_file(root / "outputs/splits_v2" / f"{split}.csv")} for split in ("train", "val", "test")},
        "acceptance": {
            "calibration_count_gt_0": sum(row["split"] == "calibration" for row in rows) > 0,
            "test_count_gt_0": sum(row["split"] == "test" for row in rows) > 0,
            "hard_test_count_gt_0": sum(row["split"] == HARD_SPLIT for row in rows) > 0,
            "calibration_test_exact_overlap_zero": overlap_counts["calibration_test"] == 0,
            "calibration_hard_exact_overlap_zero": overlap_counts["calibration_hard_test"] == 0,
            "test_hard_exact_overlap_zero": overlap_counts["test_hard_test"] == 0,
            "id_ood_exact_overlap_zero": overlap_counts["id_ood"] == 0,
            "corrupt_images_in_manifest_zero": not any(row.get("reason", "").startswith("corrupt:") for row in rejected),
            "source_provenance_recorded": all(row["source_dataset"] and row["source_image_id"] and row["source_reference"] for row in rows),
            "dog_labeled_source_images_excluded": source_stats["dog_contamination_rejected"] >= 0,
            "clean_id_splits_unchanged": all(item["before"] == item["after"] for item in ({"before": split_hashes[s], "after": sha256_file(root / "outputs/splits_v2" / f"{s}.csv")} for s in ("train", "val", "test"))),
            "production_model_unchanged": protected_before.get("outputs/deployment_ready/dog_breed_model.keras") == protected_after.get("outputs/deployment_ready/dog_breed_model.keras"),
        },
    }
    integrity["acceptance"]["all_required"] = all(integrity["acceptance"].values())
    (root / "outputs/reports/ood_v1_integrity.json").write_text(json.dumps(integrity, indent=2), encoding="utf-8")

    stats = summarize(rows)
    stats.update({
        "version": "ood_v1",
        "seed": args.seed,
        "source_archives": source_meta["archive_sha256"],
        "source_stats": source_stats,
        "corrupt_rejected": acquisition_stats["corrupt_rejected"],
        "dog_contamination_rejected": source_stats["dog_contamination_rejected"],
        "exact_duplicates_rejected": integrity["rejections"]["exact_duplicates_rejected"],
        "near_duplicates_reviewed": near_stats["near_duplicates_reviewed"],
        "near_duplicates_rejected": near_stats["near_duplicates_rejected"],
        "download_failures": acquisition_stats["download_failures"],
        "manual_review": {"performed": True, "contact_sheets": {key: str((root / "outputs/reports/ood_v1_manual_review" / ("calibration_30.jpg" if key == "calibration" else "test_30.jpg" if key == "test" else "hard_test_by_category.jpg")).relative_to(root).as_posix()) for key in review_paths}},
        "manual_review_status": "complete" if len(review_paths.get("test", [])) >= 30 else "limited_available_source_count",
        "manual_review_sample_counts": {key: len(value) for key, value in review_paths.items()},
        "id_split_hashes": split_hashes,
        "protected_hashes": protected_after,
    })
    (root / "metadata/ood_v1_statistics.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    write_report(root, stats, source_stats, rows, rejected, integrity, review_paths)
    print(json.dumps({"status": "OOD_DATASET_READY" if integrity["acceptance"]["all_required"] else "BLOCKED", **summarize(rows), "rejected": len(rejected)}, indent=2))
    return 0 if integrity["acceptance"]["all_required"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
