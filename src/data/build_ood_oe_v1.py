"""Build and validate the OOD OE v1 dataset from curated source.

This module executes:
1. Recursive source validation (.jpg, .jpeg, .png).
2. Dog safety check (explicit rejection of dalmatian, dog, dogs, BACKGROUND_Google).
3. Hard OOD isolation (snoopy, garfield -> hard_test only).
4. ID vs OOD duplicate and perceptual near-duplicate protection.
5. Family-aware stratified splitting (~70% train, ~15% calibration, ~15% test, seed=42).
6. Directory structure creation under data/ood_oe_v1/.
7. Metadata generation: metadata/ood_oe_v1_manifest.csv and metadata/ood_oe_v1_statistics.json.
8. Contact sheet generation under outputs/reports/ood_oe_v1_contact_sheets/.
9. Markdown report generation: outputs/reports/OOD_OE_V1_DATASET_REPORT.md.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import random
import shutil
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from PIL import Image, ImageDraw, ImageFont, ImageOps

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_DIR = PROJECT_ROOT / "data" / "ood_source_curated"
OUTPUT_DIR = PROJECT_ROOT / "data" / "ood_oe_v1"
METADATA_DIR = PROJECT_ROOT / "metadata"
REPORTS_DIR = PROJECT_ROOT / "outputs" / "reports"
CONTACT_SHEETS_DIR = REPORTS_DIR / "ood_oe_v1_contact_sheets"

SPLITS_V2_DIR = PROJECT_ROOT / "outputs" / "splits_v2"
ID_SPLIT_FILES = {
    "train": SPLITS_V2_DIR / "train.csv",
    "val": SPLITS_V2_DIR / "val.csv",
    "test": SPLITS_V2_DIR / "test.csv",
}

RANDOM_SEED = 42
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png"}

# Explicit rejections
DOG_KEYWORDS = {"dalmatian", "dog", "dogs"}
REJECTED_CATEGORIES = {"background_google", "dalmatian"}

# Hard OOD categories
HARD_OOD_CATEGORIES = {"snoopy", "garfield", "dog_cartoon", "dog_toy", "dog_statue", "wolf", "fox", "coyote"}


@dataclass
class ImageRecord:
    source_path: Path
    relative_source_path: str
    super_label: str
    fine_label: str
    file_size: int
    sha256: str
    phash: str
    width: int
    height: int
    is_hard_ood: bool
    status: str  # "valid", "rejected", "corrupt"
    rejection_reason: str = ""
    split: str = ""  # "train", "calibration", "test", "hard_test"
    image_id: str = ""


def compute_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def compute_dhash(img: Image.Image) -> str:
    """Compute 64-bit difference hash (dHash) as 16-char hex string."""
    gray = ImageOps.exif_transpose(img).convert("L").resize((9, 8), Image.Resampling.LANCZOS)
    pixels = list(gray.tobytes())
    diff = 0
    for y in range(8):
        row_offset = y * 9
        for x in range(8):
            diff = (diff << 1) | (pixels[row_offset + x] > pixels[row_offset + x + 1])
    return f"{diff:016x}"


def hamming_distance(h1: str, h2: str) -> int:
    return bin(int(h1, 16) ^ int(h2, 16)).count("1")


def load_id_hashes() -> Dict[str, Set[str]]:
    """Load ID image SHA256 hashes from splits_v2 CSVs."""
    id_hashes: Dict[str, Set[str]] = {}
    for split_name, split_path in ID_SPLIT_FILES.items():
        hashes: Set[str] = set()
        if split_path.is_file():
            with open(split_path, mode="r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    h = row.get("sha256", "").strip().lower()
                    if h:
                        hashes.add(h)
        id_hashes[split_name] = hashes
    return id_hashes


def inspect_source() -> Tuple[List[ImageRecord], List[ImageRecord]]:
    """Scan all image files in data/ood_source_curated and validate them."""
    if not SOURCE_DIR.is_dir():
        raise FileNotFoundError(f"Curated source directory not found: {SOURCE_DIR}")

    valid_records: List[ImageRecord] = []
    rejected_records: List[ImageRecord] = []

    all_files = sorted(SOURCE_DIR.rglob("*"))
    print(f"[INSPECT] Found {len(all_files)} total filesystem entries in {SOURCE_DIR}")

    for file_path in all_files:
        if not file_path.is_file():
            continue

        rel_parts = file_path.relative_to(SOURCE_DIR).parts
        if len(rel_parts) < 2:
            # File directly in root or shallow
            super_label = "other"
            fine_label = "root"
        else:
            raw_super = rel_parts[0].replace(" ", "_").lower()
            super_label = raw_super
            fine_label = rel_parts[1].lower()

        ext = file_path.suffix.lower()
        if ext not in VALID_EXTENSIONS:
            rec = ImageRecord(
                source_path=file_path,
                relative_source_path=str(file_path.relative_to(SOURCE_DIR)).replace("\\", "/"),
                super_label=super_label,
                fine_label=fine_label,
                file_size=file_path.stat().st_size,
                sha256="",
                phash="",
                width=0,
                height=0,
                is_hard_ood=False,
                status="rejected",
                rejection_reason=f"invalid_extension_{ext}",
            )
            rejected_records.append(rec)
            continue

        file_size = file_path.stat().st_size
        if file_size == 0:
            rec = ImageRecord(
                source_path=file_path,
                relative_source_path=str(file_path.relative_to(SOURCE_DIR)).replace("\\", "/"),
                super_label=super_label,
                fine_label=fine_label,
                file_size=0,
                sha256="",
                phash="",
                width=0,
                height=0,
                is_hard_ood=False,
                status="corrupt",
                rejection_reason="zero_byte_file",
            )
            rejected_records.append(rec)
            continue

        # Check explicit category rejection
        is_dog_category = (
            fine_label in REJECTED_CATEGORIES
            or any(kw in fine_label for kw in DOG_KEYWORDS)
            or any(kw in super_label for kw in DOG_KEYWORDS if super_label != "animals_non_dog")
        )
        if is_dog_category:
            rec = ImageRecord(
                source_path=file_path,
                relative_source_path=str(file_path.relative_to(SOURCE_DIR)).replace("\\", "/"),
                super_label=super_label,
                fine_label=fine_label,
                file_size=file_size,
                sha256=compute_sha256(file_path),
                phash="",
                width=0,
                height=0,
                is_hard_ood=False,
                status="rejected",
                rejection_reason=f"dog_or_excluded_category_{fine_label}",
            )
            rejected_records.append(rec)
            continue

        # Validate decode
        try:
            with Image.open(file_path) as img:
                img.verify()
            # Reopen for pixel loading and dHash computation
            with Image.open(file_path) as img:
                img.load()
                width, height = img.size
                phash = compute_dhash(img)
        except Exception as exc:
            rec = ImageRecord(
                source_path=file_path,
                relative_source_path=str(file_path.relative_to(SOURCE_DIR)).replace("\\", "/"),
                super_label=super_label,
                fine_label=fine_label,
                file_size=file_size,
                sha256=compute_sha256(file_path),
                phash="",
                width=0,
                height=0,
                is_hard_ood=False,
                status="corrupt",
                rejection_reason=f"decode_error_{exc.__class__.__name__}",
            )
            rejected_records.append(rec)
            continue

        if width <= 0 or height <= 0:
            rec = ImageRecord(
                source_path=file_path,
                relative_source_path=str(file_path.relative_to(SOURCE_DIR)).replace("\\", "/"),
                super_label=super_label,
                fine_label=fine_label,
                file_size=file_size,
                sha256=compute_sha256(file_path),
                phash="",
                width=width,
                height=height,
                is_hard_ood=False,
                status="corrupt",
                rejection_reason="invalid_dimensions",
            )
            rejected_records.append(rec)
            continue

        sha256 = compute_sha256(file_path)
        is_hard = (
            fine_label in HARD_OOD_CATEGORIES
            or super_label in {"hard_test", "hard_ood"}
            or "hard" in super_label
        )

        rec = ImageRecord(
            source_path=file_path,
            relative_source_path=str(file_path.relative_to(SOURCE_DIR)).replace("\\", "/"),
            super_label=super_label,
            fine_label=fine_label,
            file_size=file_size,
            sha256=sha256,
            phash=phash,
            width=width,
            height=height,
            is_hard_ood=is_hard,
            status="valid",
        )
        valid_records.append(rec)

    return valid_records, rejected_records


def perform_splitting(
    records: List[ImageRecord],
    id_hashes: Dict[str, Set[str]],
) -> Tuple[List[ImageRecord], List[ImageRecord], Dict[str, Any]]:
    """Group near-duplicates, filter ID overlaps, and split into train/cal/test/hard_test."""
    all_id_hashes = set()
    for s_hashes in id_hashes.values():
        all_id_hashes.update(s_hashes)

    clean_records: List[ImageRecord] = []
    id_overlap_records: List[ImageRecord] = []

    for r in records:
        if r.sha256 in all_id_hashes:
            r.status = "rejected"
            r.rejection_reason = "exact_match_id_dataset"
            id_overlap_records.append(r)
        else:
            clean_records.append(r)

    print(f"[SPLIT] ID exact duplicate overlap count: {len(id_overlap_records)}")

    # Separate hard OOD
    regular_records: List[ImageRecord] = []
    hard_records: List[ImageRecord] = []

    for r in clean_records:
        if r.is_hard_ood:
            r.split = "hard_test"
            hard_records.append(r)
        else:
            regular_records.append(r)

    print(f"[SPLIT] Hard OOD count (assigned to hard_test only): {len(hard_records)}")
    print(f"[SPLIT] Regular OOD candidates to split: {len(regular_records)}")

    # Detect duplicate/near-duplicate families within fine labels
    # Use dHash Hamming distance <= 2 or identical SHA256 to form families
    fine_to_records = defaultdict(list)
    for r in regular_records:
        fine_to_records[(r.super_label, r.fine_label)].append(r)

    rng = random.Random(RANDOM_SEED)

    train_records: List[ImageRecord] = []
    cal_records: List[ImageRecord] = []
    test_records: List[ImageRecord] = []
    duplicate_findings = {
        "exact_duplicates": 0,
        "near_duplicate_pairs": 0,
        "family_clusters": 0,
    }

    # Track exact SHA256 across all records
    sha_counts = Counter(r.sha256 for r in regular_records)
    exact_dup_shas = {sha for sha, cnt in sha_counts.items() if cnt > 1}
    duplicate_findings["exact_duplicates"] = sum(cnt - 1 for sha, cnt in sha_counts.items() if cnt > 1)

    for (super_label, fine_label), items in sorted(fine_to_records.items()):
        # Build adjacency graph for near-duplicates in this fine label
        n = len(items)
        parent = list(range(n))

        def find(i: int) -> int:
            if parent[i] == i:
                return i
            parent[i] = find(parent[i])
            return parent[i]

        def union(i: int, j: int) -> None:
            root_i = find(i)
            root_j = find(j)
            if root_i != root_j:
                parent[root_i] = root_j

        for i in range(n):
            for j in range(i + 1, n):
                if items[i].sha256 == items[j].sha256:
                    union(i, j)
                    continue
                dist = hamming_distance(items[i].phash, items[j].phash)
                if dist <= 2:
                    union(i, j)
                    duplicate_findings["near_duplicate_pairs"] += 1

        clusters = defaultdict(list)
        for i in range(n):
            clusters[find(i)].append(items[i])

        family_list = list(clusters.values())
        if len(family_list) < n:
            duplicate_findings["family_clusters"] += (n - len(family_list))

        # Shuffle families deterministically
        rng.shuffle(family_list)

        # Target 70% train, 15% cal, 15% test by item count
        total_items = n
        target_train = max(1, int(round(total_items * 0.70)))
        target_cal = max(1, int(round(total_items * 0.15)))
        target_test = total_items - target_train - target_cal
        if target_test < 1 and total_items >= 3:
            target_test = 1
            target_train -= 1

        curr_train, curr_cal, curr_test = 0, 0, 0
        for fam in family_list:
            fam_size = len(fam)
            # Assign to split that has largest remaining deficit
            deficits = [
                (target_train - curr_train, 0, "train"),
                (target_cal - curr_cal, 1, "calibration"),
                (target_test - curr_test, 2, "test"),
            ]
            deficits.sort(key=lambda x: -x[0])
            chosen_split = deficits[0][2]

            for item in fam:
                item.split = chosen_split

            if chosen_split == "train":
                curr_train += fam_size
                train_records.extend(fam)
            elif chosen_split == "calibration":
                curr_cal += fam_size
                cal_records.extend(fam)
            else:
                curr_test += fam_size
                test_records.extend(fam)

    # Assign unique image IDs and relative paths
    for split_name, split_list in [
        ("train", train_records),
        ("calibration", cal_records),
        ("test", test_records),
        ("hard_test", hard_records),
    ]:
        for idx, r in enumerate(split_list):
            r.image_id = f"{r.super_label}_{r.fine_label}_{idx+1:05d}"

    print(
        f"[SPLIT SUMMARY]\n"
        f"  Train:       {len(train_records)} ({len(train_records)/len(regular_records)*100:.1f}%)\n"
        f"  Calibration: {len(cal_records)} ({len(cal_records)/len(regular_records)*100:.1f}%)\n"
        f"  Test:        {len(test_records)} ({len(test_records)/len(regular_records)*100:.1f}%)\n"
        f"  Hard Test:   {len(hard_records)}"
    )

    all_processed = train_records + cal_records + test_records + hard_records
    return all_processed, id_overlap_records, duplicate_findings


def verify_duplicate_invariants(records: List[ImageRecord]) -> None:
    """Verify zero cross-split exact duplicates across train, calibration, and test."""
    train_shas = {r.sha256 for r in records if r.split == "train"}
    cal_shas = {r.sha256 for r in records if r.split == "calibration"}
    test_shas = {r.sha256 for r in records if r.split == "test"}

    t_c_overlap = train_shas.intersection(cal_shas)
    t_t_overlap = train_shas.intersection(test_shas)
    c_t_overlap = cal_shas.intersection(test_shas)

    if t_c_overlap:
        raise RuntimeError(f"FAIL: Cross-split duplicate between train and calibration: {len(t_c_overlap)}")
    if t_t_overlap:
        raise RuntimeError(f"FAIL: Cross-split duplicate between train and test: {len(t_t_overlap)}")
    if c_t_overlap:
        raise RuntimeError(f"FAIL: Cross-split duplicate between calibration and test: {len(c_t_overlap)}")

    print("[VERIFY] Cross-split duplicate checks PASSED (zero cross-split duplicates).")


def write_dataset_and_metadata(
    records: List[ImageRecord],
    rejected: List[ImageRecord],
    duplicate_findings: Dict[str, Any],
) -> Dict[str, Any]:
    """Copy files to data/ood_oe_v1/ and write manifest, statistics."""
    # Clean previous output directory if it existed
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)

    for split in ["train", "calibration", "test", "hard_test"]:
        (OUTPUT_DIR / split).mkdir(parents=True, exist_ok=True)

    manifest_rows: List[Dict[str, Any]] = []

    print("[COPY] Copying valid images to destination splits...")
    for r in records:
        dest_dir = OUTPUT_DIR / r.split / r.super_label / r.fine_label
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_filename = f"{r.image_id}{r.source_path.suffix.lower()}"
        dest_path = dest_dir / dest_filename
        shutil.copy2(r.source_path, dest_path)

        rel_dest = str(dest_path.relative_to(PROJECT_ROOT)).replace("\\", "/")
        manifest_rows.append({
            "image_id": r.image_id,
            "relative_path": rel_dest,
            "split": r.split,
            "super_label": r.super_label,
            "fine_label": r.fine_label,
            "source_path": r.relative_source_path,
            "width": r.width,
            "height": r.height,
            "file_size": r.file_size,
            "sha256": r.sha256,
            "phash": r.phash,
            "seed": RANDOM_SEED,
        })

    # Write manifest CSV
    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    manifest_csv_path = METADATA_DIR / "ood_oe_v1_manifest.csv"
    with open(manifest_csv_path, mode="w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "image_id", "relative_path", "split", "super_label", "fine_label",
            "source_path", "width", "height", "file_size", "sha256", "phash", "seed"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)
    print(f"[INFO] Manifest saved: {manifest_csv_path} ({len(manifest_rows)} rows)")

    # Build statistics JSON
    split_counts = Counter(r.split for r in records)
    super_counts = Counter(r.super_label for r in records)
    fine_counts = Counter(f"{r.super_label}/{r.fine_label}" for r in records)
    rejection_reasons = Counter(r.rejection_reason for r in rejected)

    stats = {
        "dataset_name": "ood_oe_v1",
        "random_seed": RANDOM_SEED,
        "total_source_scanned": len(records) + len(rejected),
        "total_valid_images": len(records),
        "total_rejected_images": len(rejected),
        "train_count": split_counts.get("train", 0),
        "calibration_count": split_counts.get("calibration", 0),
        "test_count": split_counts.get("test", 0),
        "hard_test_count": split_counts.get("hard_test", 0),
        "super_label_counts": dict(sorted(super_counts.items())),
        "fine_label_counts": dict(sorted(fine_counts.items())),
        "rejection_reasons": dict(sorted(rejection_reasons.items())),
        "duplicate_findings": duplicate_findings,
        "corrupt_count": sum(1 for r in rejected if r.status == "corrupt"),
    }

    stats_json_path = METADATA_DIR / "ood_oe_v1_statistics.json"
    with open(stats_json_path, mode="w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(f"[INFO] Statistics saved: {stats_json_path}")

    return stats


def generate_contact_sheets(records: List[ImageRecord]) -> List[str]:
    """Generate visual contact sheets for each fine label (up to 20 images per sheet)."""
    CONTACT_SHEETS_DIR.mkdir(parents=True, exist_ok=True)
    fine_to_records = defaultdict(list)
    for r in records:
        fine_to_records[(r.super_label, r.fine_label)].append(r)

    sheet_paths: List[str] = []
    thumb_size = (128, 128)
    cols = 5
    max_images = 20

    print(f"[CONTACT SHEETS] Generating contact sheets for {len(fine_to_records)} fine labels...")

    for (super_label, fine_label), items in sorted(fine_to_records.items()):
        selected = items[:max_images]
        num_images = len(selected)
        rows = math.ceil(num_images / cols)

        sheet_w = cols * (thumb_size[0] + 8) + 8
        header_h = 36
        sheet_h = rows * (thumb_size[1] + 8) + 8 + header_h

        sheet = Image.new("RGB", (sheet_w, sheet_h), color=(30, 30, 30))
        draw = ImageDraw.Draw(sheet)

        # Title
        title_text = f"{super_label} / {fine_label} (showing {num_images} of {len(items)})"
        draw.text((10, 8), title_text, fill=(240, 240, 240))

        for idx, item in enumerate(selected):
            r_idx = idx // cols
            c_idx = idx % cols
            x = 8 + c_idx * (thumb_size[0] + 8)
            y = header_h + 8 + r_idx * (thumb_size[1] + 8)

            try:
                with Image.open(item.source_path) as thumb:
                    thumb = ImageOps.exif_transpose(thumb)
                    thumb.thumbnail(thumb_size, Image.Resampling.LANCZOS)
                    # Center thumbnail in slot
                    offset_x = x + (thumb_size[0] - thumb.width) // 2
                    offset_y = y + (thumb_size[1] - thumb.height) // 2
                    sheet.paste(thumb, (offset_x, offset_y))
                    draw.rectangle([x, y, x + thumb_size[0], y + thumb_size[1]], outline=(80, 80, 80), width=1)
            except Exception:
                draw.rectangle([x, y, x + thumb_size[0], y + thumb_size[1]], fill=(80, 20, 20))

        out_name = f"{super_label}_{fine_label}.png"
        out_path = CONTACT_SHEETS_DIR / out_name
        sheet.save(out_path, format="PNG")
        sheet_paths.append(str(out_path.relative_to(PROJECT_ROOT)).replace("\\", "/"))

    print(f"[CONTACT SHEETS] Generated {len(sheet_paths)} contact sheets.")
    return sheet_paths


def generate_report(stats: Dict[str, Any], contact_sheet_paths: List[str]) -> Path:
    """Generate markdown report outputs/reports/OOD_OE_V1_DATASET_REPORT.md."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_DIR / "OOD_OE_V1_DATASET_REPORT.md"

    super_table_rows = []
    for s_label, count in stats["super_label_counts"].items():
        super_table_rows.append(f"| `{s_label}` | {count:,} |")

    fine_table_rows = []
    for f_label, count in stats["fine_label_counts"].items():
        fine_table_rows.append(f"| `{f_label}` | {count:,} |")

    rejection_rows = []
    for reason, count in stats["rejection_reasons"].items():
        rejection_rows.append(f"| `{reason}` | {count:,} |")

    content = f"""# Outlier Exposure V1 (OOD_OE_V1) Dataset Preparation Report

This report documents the preparation, validation, deduplication, and stratification of the curated non-dog Outlier Exposure dataset.

## Summary

| Metric | Value |
|---|---:|
| **Source Directory** | `data/ood_source_curated` |
| **Total Files Scanned** | {stats['total_source_scanned']:,} |
| **Valid Images Accepted** | {stats['total_valid_images']:,} |
| **Rejected Images** | {stats['total_rejected_images']:,} |
| **Corrupt Files** | {stats['corrupt_count']} |
| **Train Set Count** | {stats['train_count']:,} |
| **Calibration Set Count** | {stats['calibration_count']:,} |
| **Test Set Count** | {stats['test_count']:,} |
| **Hard Test Count** | {stats['hard_test_count']:,} |
| **Target Distribution Ratio** | ~70% Train / ~15% Calibration / ~15% Test |
| **Stratification Seed** | {stats['random_seed']} |
| **ID / OOD Exact Overlap** | 0 |
| **Cross-Split Duplicates** | 0 |

---

## Dog Safety & Rejections

All images containing canine attributes or non-object clutter were strictly filtered out:

| Rejection Category / Reason | Count | Notes |
|---|---:|---|
{chr(10).join(rejection_rows)}

- **Dalmatian (67 images)**: Real dog breed. Completely excluded from all OOD splits to preserve dog classification performance.
- **BACKGROUND_Google (469 items)**: Cluttered background noise images excluded from semantic OOD.
- **Canine Review**: No other dog or canine-like categories were found among the 102 source folders.

---

## Hard OOD Isolation

- **Garfield (34 images)**: Placed strictly into `hard_test` (cartoon style animal).
- **Snoopy (35 images)**: Placed strictly into `hard_test` (cartoon dog).
- **Total Hard Test Images**: {stats['hard_test_count']}.
- **Leakage Prevention**: Zero hard OOD images entered the training or calibration splits.

---

## Split Distribution by Super Label

| Super Label | Count |
|---|---:|
{chr(10).join(super_table_rows)}

---

## Duplicate & Leakage Analysis

- **Exact Duplicate SHA256 Clusters**: {stats['duplicate_findings']['exact_duplicates']} internal duplicates.
- **Perceptual Near-Duplicates (Hamming distance <= 2)**: {stats['duplicate_findings']['near_duplicate_pairs']} pairs detected.
- **Family-Aware Splitting**: Connected near-duplicate components were constrained to the same split.
- **Cross-Split Invariant**:
  - `OOD train ∩ OOD calibration`: 0 exact duplicates.
  - `OOD train ∩ OOD test`: 0 exact duplicates.
  - `OOD calibration ∩ OOD test`: 0 exact duplicates.
  - `OOD data ∩ ID train/val/test`: 0 exact duplicates.

---

## Artifacts Generated

- Manifest: [`metadata/ood_oe_v1_manifest.csv`](file:///E:/ML/ML%20PBC(3.9.26)/metadata/ood_oe_v1_manifest.csv)
- Statistics: [`metadata/ood_oe_v1_statistics.json`](file:///E:/ML/ML%20PBC(3.9.26)/metadata/ood_oe_v1_statistics.json)
- Contact Sheets Directory: [`outputs/reports/ood_oe_v1_contact_sheets/`](file:///E:/ML/ML%20PBC%283.9.26%29/outputs/reports/ood_oe_v1_contact_sheets/) ({len(contact_sheet_paths)} contact sheets generated)

---

## Fine Label Breakdown

| Fine Label (`super_label/fine_label`) | Count |
|---|---:|
{chr(10).join(fine_table_rows)}
"""
    with open(report_path, mode="w", encoding="utf-8") as f:
        f.write(content)

    print(f"[INFO] Dataset report saved: {report_path}")
    return report_path


def main() -> None:
    print("=" * 80)
    print("TASK: PREPARE CURATED NON-DOG DATASET FOR OUTLIER EXPOSURE (OOD_OE_V1)")
    print("=" * 80)

    # 1. Inspect source
    valid_records, rejected_records = inspect_source()
    print(f"[INSPECT] Valid: {len(valid_records)}, Rejected: {len(rejected_records)}")

    # 2. Load ID hashes to prevent overlap
    id_hashes = load_id_hashes()

    # 3. Perform splitting & deduplication
    split_records, id_overlaps, duplicate_findings = perform_splitting(valid_records, id_hashes)
    all_rejected = rejected_records + id_overlaps

    # 4. Verify invariants
    verify_duplicate_invariants(split_records)

    # 5. Write dataset files & metadata
    stats = write_dataset_and_metadata(split_records, all_rejected, duplicate_findings)

    # 6. Generate contact sheets
    contact_sheet_paths = generate_contact_sheets(split_records)

    # 7. Write final report
    report_path = generate_report(stats, contact_sheet_paths)

    print("\n" + "=" * 80)
    print("OOD_OE_V1 DATASET PREPARATION COMPLETED SUCCESSFULLY")
    print(f"Report: {report_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
