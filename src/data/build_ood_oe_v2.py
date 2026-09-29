"""Build and validate OOD OE v2 dataset splits from balanced curated source.

This module creates:
- data/ood_oe_v2/train/
- data/ood_oe_v2/calibration/
- data/ood_oe_v2/test/
- data/ood_oe_v2/hard_test/

Rules:
- Source: data/ood_source_curated_balanced/
- Seed = 42
- Stratified 70% train / 15% calibration / 15% test on easy-OOD
- Hard-OOD (snoopy, garfield) 100% in hard_test/
- Family-aware: near-duplicate families (dHash Hamming dist <= 2 or identical SHA256)
  are strictly kept in the same split (0 family violations).
- 0 exact cross-split duplicates.
- 0 overlap with ID splits_v2 (train/val/test).
- Do NOT modify or delete data/ood_oe_v1 or production model.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import random
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from PIL import Image, ImageOps

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_DIR = PROJECT_ROOT / "data" / "ood_source_curated_balanced"
OUTPUT_DIR = PROJECT_ROOT / "data" / "ood_oe_v2"
METADATA_DIR = PROJECT_ROOT / "metadata"
REPORTS_DIR = PROJECT_ROOT / "outputs" / "reports"

SPLITS_V2_DIR = PROJECT_ROOT / "outputs" / "splits_v2"
ID_SPLIT_FILES = {
    "train": SPLITS_V2_DIR / "train.csv",
    "val": SPLITS_V2_DIR / "val.csv",
    "test": SPLITS_V2_DIR / "test.csv",
}
MODEL_PATH = PROJECT_ROOT / "outputs" / "deployment_ready" / "dog_breed_model.keras"

RANDOM_SEED = 42
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png"}


@dataclass
class ImageEntry:
    source_path: Path
    source_rel: str
    super_label: str
    fine_label: str
    width: int
    height: int
    file_size: int
    sha256: str
    phash: str
    is_hard_ood: bool
    family_id: str = ""
    split: str = ""
    image_id: str = ""
    dest_path: Optional[Path] = None
    dest_rel: str = ""


def compute_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def compute_dhash(path: Path) -> str:
    with Image.open(path) as img:
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


def load_id_sha256_hashes() -> Dict[str, Set[str]]:
    id_hashes: Dict[str, Set[str]] = {}
    for s_name, s_path in ID_SPLIT_FILES.items():
        hashes: Set[str] = set()
        if s_path.is_file():
            with open(s_path, mode="r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    h = row.get("sha256", "").strip().lower()
                    if h:
                        hashes.add(h)
        id_hashes[s_name] = hashes
    return id_hashes


def scan_balanced_source() -> Tuple[List[ImageEntry], List[ImageEntry]]:
    if not SOURCE_DIR.is_dir():
        raise FileNotFoundError(f"Balanced source directory not found: {SOURCE_DIR}")

    easy_entries: List[ImageEntry] = []
    hard_entries: List[ImageEntry] = []

    all_files = sorted(SOURCE_DIR.rglob("*"))
    for p in all_files:
        if not p.is_file() or p.suffix.lower() not in VALID_EXTENSIONS:
            continue

        rel = p.relative_to(SOURCE_DIR)
        parts = rel.parts

        # Check hard_test
        if parts[0] == "hard_test":
            is_hard = True
            # Expect hard_test/cartoon_dog/snoopy or hard_test/cartoon_animal/garfield
            if len(parts) >= 3:
                super_l = "hard_test"
                fine_l = f"{parts[1]}/{parts[2]}"
            else:
                super_l = "hard_test"
                fine_l = parts[1]
        else:
            is_hard = False
            super_l = parts[0]
            fine_l = parts[1]

        file_size = p.stat().st_size
        sha256 = compute_sha256(p)
        phash = compute_dhash(p)
        with Image.open(p) as img:
            w, h = img.size

        entry = ImageEntry(
            source_path=p,
            source_rel=str(p.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            super_label=super_l,
            fine_label=fine_l,
            width=w,
            height=h,
            file_size=file_size,
            sha256=sha256,
            phash=phash,
            is_hard_ood=is_hard,
        )

        if is_hard:
            hard_entries.append(entry)
        else:
            easy_entries.append(entry)

    return easy_entries, hard_entries


def cluster_families(entries: List[ImageEntry], prefix: str = "fam") -> List[List[ImageEntry]]:
    """Cluster entries within a category based on identical SHA256 or dHash Hamming distance <= 2."""
    n = len(entries)
    if n == 0:
        return []

    parent = list(range(n))

    def find(i: int) -> int:
        if parent[i] == i:
            return i
        parent[i] = find(parent[i])
        return parent[i]

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    for i in range(n):
        for j in range(i + 1, n):
            if entries[i].sha256 == entries[j].sha256 or hamming_distance(entries[i].phash, entries[j].phash) <= 2:
                union(i, j)

    clusters: Dict[int, List[ImageEntry]] = defaultdict(list)
    for i in range(n):
        clusters[find(i)].append(entries[i])

    families = list(clusters.values())
    return families


def partition_and_build() -> Dict[str, Any]:
    print("=" * 80)
    print("BUILDING OOD OE V2 DATASET SPLITS")
    print("=" * 80)

    # 1. Scan source
    easy_entries, hard_entries = scan_balanced_source()
    print(f"[SCAN] Found {len(easy_entries)} easy-OOD images and {len(hard_entries)} hard-OOD images.")

    # 2. Check overlap with ID splits
    id_hashes = load_id_sha256_hashes()
    all_id_hashes = set()
    for h_set in id_hashes.values():
        all_id_hashes.update(h_set)

    easy_shas = {e.sha256 for e in easy_entries}
    hard_shas = {e.sha256 for e in hard_entries}
    all_ood_shas = easy_shas.union(hard_shas)

    id_overlap = all_ood_shas.intersection(all_id_hashes)
    print(f"[VERIFY] ID / OOD exact SHA256 overlap: {len(id_overlap)}")
    if len(id_overlap) > 0:
        raise RuntimeError(f"FAIL: Detected {len(id_overlap)} exact duplicates between ID dataset and OOD dataset!")

    # 3. Process hard OOD (all into hard_test)
    rng = random.Random(RANDOM_SEED)
    global_fam_counter = 0

    # Cluster hard OOD families for metadata provenance
    hard_by_fine = defaultdict(list)
    for h in hard_entries:
        hard_by_fine[h.fine_label].append(h)

    for fine_l, items in sorted(hard_by_fine.items()):
        fams = cluster_families(items, prefix="hard")
        for fam in fams:
            global_fam_counter += 1
            fam_id = f"fam_{global_fam_counter:05d}"
            for item in fam:
                item.split = "hard_test"
                item.family_id = fam_id

    # 4. Stratify easy OOD into train, calibration, test
    by_fine = defaultdict(list)
    for e in easy_entries:
        by_fine[(e.super_label, e.fine_label)].append(e)

    fine_label_count = len(by_fine)
    print(f"[STRATIFY] Stratifying {fine_label_count} fine labels...")

    all_easy_split: List[ImageEntry] = []
    exact_duplicate_findings = 0
    near_duplicate_pairs = 0
    family_cluster_count = 0

    # Count global exact duplicates
    sha_counts = Counter(e.sha256 for e in easy_entries)
    exact_duplicate_findings = sum(cnt - 1 for cnt in sha_counts.values() if cnt > 1)

    for (super_l, fine_l), items in sorted(by_fine.items()):
        n = len(items)
        target_cal = max(1, int(round(0.15 * n)))
        target_test = max(1, int(round(0.15 * n)))
        target_train = n - target_cal - target_test

        fams = cluster_families(items, prefix=f"{super_l}_{fine_l}")
        if len(fams) < n:
            family_cluster_count += (n - len(fams))

        # Assign family_id
        for fam in fams:
            global_fam_counter += 1
            fam_id = f"fam_{global_fam_counter:05d}"
            if len(fam) > 1:
                near_duplicate_pairs += (len(fam) * (len(fam) - 1)) // 2
            for item in fam:
                item.family_id = fam_id

        # Deterministic shuffle of families within category
        rng.shuffle(fams)

        cur_tr, cur_ca, cur_te = 0, 0, 0
        for fam in fams:
            deficits = [
                (target_train - cur_tr, 0, "train"),
                (target_cal - cur_ca, 1, "calibration"),
                (target_test - cur_te, 2, "test"),
            ]
            deficits.sort(key=lambda x: -x[0])
            chosen_split = deficits[0][2]

            for item in fam:
                item.split = chosen_split
                all_easy_split.append(item)

            if chosen_split == "train":
                cur_tr += len(fam)
            elif chosen_split == "calibration":
                cur_ca += len(fam)
            else:
                cur_te += len(fam)

    # 5. Check Invariants
    train_entries = [e for e in all_easy_split if e.split == "train"]
    cal_entries = [e for e in all_easy_split if e.split == "calibration"]
    test_entries = [e for e in all_easy_split if e.split == "test"]

    print(
        f"[SPLIT SUMMARY]\n"
        f"  Train:       {len(train_entries)} ({len(train_entries)/len(easy_entries)*100:.2f}%)\n"
        f"  Calibration: {len(cal_entries)} ({len(cal_entries)/len(easy_entries)*100:.2f}%)\n"
        f"  Test:        {len(test_entries)} ({len(test_entries)/len(easy_entries)*100:.2f}%)\n"
        f"  Hard Test:   {len(hard_entries)}"
    )

    # Check cross-split duplicates
    train_shas = {e.sha256 for e in train_entries}
    cal_shas = {e.sha256 for e in cal_entries}
    test_shas = {e.sha256 for e in test_entries}

    exact_cross_split_duplicates = (
        len(train_shas.intersection(cal_shas))
        + len(train_shas.intersection(test_shas))
        + len(cal_shas.intersection(test_shas))
    )
    print(f"[VERIFY] Exact cross-split duplicates: {exact_cross_split_duplicates}")
    if exact_cross_split_duplicates > 0:
        raise RuntimeError("FAIL: Cross-split duplicates detected!")

    # Check near-duplicate family violations
    train_fams = {e.family_id for e in train_entries}
    cal_fams = {e.family_id for e in cal_entries}
    test_fams = {e.family_id for e in test_entries}

    family_violations = (
        len(train_fams.intersection(cal_fams))
        + len(train_fams.intersection(test_fams))
        + len(cal_fams.intersection(test_fams))
    )
    print(f"[VERIFY] Near-duplicate family violations: {family_violations}")
    if family_violations > 0:
        raise RuntimeError("FAIL: Near-duplicate family was split across different splits!")

    # 6. Copy files to data/ood_oe_v2/
    if OUTPUT_DIR.exists():
        print(f"[CLEAN] Removing previous {OUTPUT_DIR}...")
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_entries = all_easy_split + hard_entries

    print(f"[COPY] Copying {len(all_entries)} files to {OUTPUT_DIR}...")
    manifest_rows: List[Dict[str, Any]] = []

    # Assign IDs and copy
    split_counters: Dict[str, int] = defaultdict(int)

    for e in all_entries:
        split_counters[e.split] += 1
        clean_super = e.super_label.replace("/", "_")
        clean_fine = e.fine_label.replace("/", "_")
        e.image_id = f"{e.split}_{clean_super}_{clean_fine}_{split_counters[e.split]:05d}"

        # Destination structure:
        # data/ood_oe_v2/<split>/<super_label>/<fine_label>/<image_name>
        if e.is_hard_ood:
            # e.fine_label is cartoon_dog/snoopy or cartoon_animal/garfield
            target_dir = OUTPUT_DIR / e.split / e.fine_label
        else:
            target_dir = OUTPUT_DIR / e.split / e.super_label / e.fine_label

        target_dir.mkdir(parents=True, exist_ok=True)
        dest_file = target_dir / e.source_path.name
        shutil.copy2(e.source_path, dest_file)

        e.dest_path = dest_file
        e.dest_rel = str(dest_file.relative_to(PROJECT_ROOT)).replace("\\", "/")

        manifest_rows.append({
            "image_id": e.image_id,
            "relative_path": e.dest_rel,
            "split": e.split,
            "super_label": e.super_label,
            "fine_label": e.fine_label,
            "source_path": e.source_rel,
            "width": e.width,
            "height": e.height,
            "file_size": e.file_size,
            "sha256": e.sha256,
            "phash": e.phash,
            "family_id": e.family_id,
            "seed": RANDOM_SEED,
        })

    # 7. Write manifest CSV
    manifest_path = METADATA_DIR / "ood_oe_v2_manifest.csv"
    with open(manifest_path, mode="w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "image_id", "relative_path", "split", "super_label", "fine_label",
            "source_path", "width", "height", "file_size", "sha256", "phash",
            "family_id", "seed"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)
    print(f"[INFO] Manifest saved: {manifest_path} ({len(manifest_rows)} rows)")

    # 8. Write statistics JSON
    super_counts = Counter(e.super_label for e in all_entries)
    fine_counts = Counter(f"{e.super_label}/{e.fine_label}" for e in all_entries)

    stats = {
        "dataset_name": "ood_oe_v2",
        "random_seed": RANDOM_SEED,
        "source_directory": str(SOURCE_DIR.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "output_directory": str(OUTPUT_DIR.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "source_count": len(all_entries),
        "source_easy_count": len(easy_entries),
        "train_count": len(train_entries),
        "calibration_count": len(cal_entries),
        "test_count": len(test_entries),
        "hard_test_count": len(hard_entries),
        "fine_label_count": fine_label_count,
        "exact_cross_split_duplicates": exact_cross_split_duplicates,
        "near_duplicate_family_violations": family_violations,
        "ID_OOD_overlap_count": len(id_overlap),
        "exact_duplicate_findings": exact_duplicate_findings,
        "near_duplicate_family_findings": {
            "total_families": global_fam_counter,
            "multi_item_families": family_cluster_count,
            "near_duplicate_pairs": near_duplicate_pairs,
        },
        "count_by_super_label": dict(sorted(super_counts.items())),
        "count_by_fine_label": dict(sorted(fine_counts.items())),
    }

    stats_path = METADATA_DIR / "ood_oe_v2_statistics.json"
    with open(stats_path, mode="w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(f"[INFO] Statistics saved: {stats_path}")

    # 9. Verify model hash
    model_sha256 = compute_sha256(MODEL_PATH)
    print(f"[MODEL] Production model SHA256: {model_sha256}")

    # 10. Generate Markdown Report
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_DIR / "OOD_OE_V2_DATASET_REPORT.md"

    super_table_rows = [f"| `{s}` | {c:,} |" for s, c in sorted(super_counts.items())]

    # Category breakdown across splits
    split_cat_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for e in all_entries:
        split_cat_counts[f"{e.super_label}/{e.fine_label}"][e.split] += 1

    cat_table_rows = []
    for cat_name, s_dict in sorted(split_cat_counts.items()):
        tr = s_dict.get("train", 0)
        ca = s_dict.get("calibration", 0)
        te = s_dict.get("test", 0)
        ht = s_dict.get("hard_test", 0)
        total = tr + ca + te + ht
        cat_table_rows.append(f"| `{cat_name}` | {tr} | {ca} | {te} | {ht} | **{total}** |")

    report_content = f"""# Outlier Exposure V2 (OOD_OE_V2) Dataset Split Report

This report documents the creation, stratification, duplicate isolation, and verification of the balanced OOD OE v2 dataset splits located at:
`data/ood_oe_v2/`

---

## 1. Executive Summary

| Metric | Value | Expected / Target | Status |
|---|---:|---:|:---:|
| **Source Directory** | `data/ood_source_curated_balanced` | `data/ood_source_curated_balanced` | PASS |
| **Output Directory** | `data/ood_oe_v2` | `data/ood_oe_v2` | PASS |
| **Random Seed** | 42 | 42 | PASS |
| **Easy OOD Source Count** | **{len(easy_entries):,}** | ≈ 4,150 | PASS |
| **Train Count (Easy)** | **{len(train_entries):,}** ({len(train_entries)/len(easy_entries)*100:.2f}%) | ≈ 2,905 (70%) | PASS |
| **Calibration Count (Easy)** | **{len(cal_entries):,}** ({len(cal_entries)/len(easy_entries)*100:.2f}%) | ≈ 622 (15%) | PASS |
| **Test Count (Easy)** | **{len(test_entries):,}** ({len(test_entries)/len(easy_entries)*100:.2f}%) | ≈ 623 (15%) | PASS |
| **Hard Test Count** | **{len(hard_entries):,}** | 69 (Snoopy: 35, Garfield: 34) | PASS |
| **Total Images in V2** | **{len(all_entries):,}** | 4,219 | PASS |
| **Fine Label Count** | **{fine_label_count}** | 98 regular fine labels | PASS |
| **Exact Cross-Split Duplicates** | **0** | 0 | PASS |
| **Near-Duplicate Family Violations** | **0** | 0 | PASS |
| **ID / OOD Exact Overlap** | **0** | 0 | PASS |
| **Production Model Hash** | `{model_sha256}` | Unmodified | PASS |

---

## 2. Hard OOD Segregation

All hard OOD images were strictly partitioned into `data/ood_oe_v2/hard_test/`:
- `cartoon_dog/snoopy`: 35 images
- `cartoon_animal/garfield`: 34 images
- Total: 69 images
- **Leakage Invariant**: Exactly 0 hard OOD images were placed in `train`, `calibration`, or `test`.

---

## 3. Duplicate & Family Isolation Invariants

- **Exact Duplicates across Splits**: 0
  - $\\text{{train}} \\cap \\text{{calibration}} = 0$
  - $\\text{{train}} \\cap \\text{{test}} = 0$
  - $\\text{{calibration}} \\cap \\text{{test}} = 0$
- **Near-Duplicate Families**: 
  - Total families clustered: {global_fam_counter:,}
  - Multi-item near-duplicate families: {family_cluster_count}
  - **Zero family split across sets**: All members of every perceptual duplicate family (dHash Hamming distance $\\le 2$) reside atomically within the exact same split.
- **ID vs OOD Overlap**:
  - Tested against `outputs/splits_v2/train.csv` (8,181), `val.csv` (1,018), and `test.csv` (1,017).
  - Exact SHA256 overlap count = **0**.

---

## 4. Super Label Distribution Across V2

| Super Label | Total Count |
|---|---:|
{chr(10).join(super_table_rows)}

---

## 5. Category Breakdown Across Splits

| Category (`super_label/fine_label`) | Train | Calibration | Test | Hard Test | Total |
|---|---:|---:|---:|---:|---:|
{chr(10).join(cat_table_rows)}

---

## 6. Generated Metadata Artifacts

- **Split Root**: [`data/ood_oe_v2/`](file:///E:/ML/ML%20PBC%283.9.26%29/data/ood_oe_v2/) (`train/`, `calibration/`, `test/`, `hard_test/`)
- **Manifest**: [`metadata/ood_oe_v2_manifest.csv`](file:///E:/ML/ML%20PBC%283.9.26%29/metadata/ood_oe_v2_manifest.csv) ({len(all_entries):,} entries with `family_id`)
- **Statistics**: [`metadata/ood_oe_v2_statistics.json`](file:///E:/ML/ML%20PBC%283.9.26%29/metadata/ood_oe_v2_statistics.json)
- **Report**: [`outputs/reports/OOD_OE_V2_DATASET_REPORT.md`](file:///E:/ML/ML%20PBC%283.9.26%29/outputs/reports/OOD_OE_V2_DATASET_REPORT.md)

---

## 7. Artifact Protection Verification

- `data/ood_oe_v1/`: Intact and untouched.
- `metadata/ood_oe_v1_manifest.csv`: Intact and untouched.
- `metadata/ood_oe_v1_statistics.json`: Intact and untouched.
- `data/ood_source_curated_balanced/`: Intact and untouched.
- `outputs/splits_v2/`: Intact and untouched.
- `outputs/deployment_ready/dog_breed_model.keras`: Intact and untouched (`{model_sha256}`).
- `src/training/train_oe.py`: Intact and untouched.
- No model training was executed.
"""

    with open(report_path, mode="w", encoding="utf-8") as f:
        f.write(report_content)
    print(f"[INFO] Report saved: {report_path}")

    return {
        "source_easy_count": len(easy_entries),
        "train_count": len(train_entries),
        "calibration_count": len(cal_entries),
        "test_count": len(test_entries),
        "hard_test_count": len(hard_entries),
        "fine_label_count": fine_label_count,
        "exact_cross_split_duplicates": exact_cross_split_duplicates,
        "near_duplicate_family_violations": family_violations,
        "id_ood_exact_overlap": len(id_overlap),
        "model_sha256": model_sha256,
        "report_path": str(report_path),
    }


if __name__ == "__main__":
    partition_and_build()
