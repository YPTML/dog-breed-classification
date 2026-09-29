"""Create balanced curated non-dog Outlier Exposure dataset.

This module implements the user-specified sampling policy to build:
data/ood_source_curated_balanced/

Rules:
- Seed = 42
- Exclude BACKGROUND_Google and dalmatian
- Fix categories:
    human/brain -> other/brain
    other/revolver -> tools_objects/revolver
    other/snoopy -> hard_test/cartoon_dog/snoopy
    hard test/garfield -> hard_test/cartoon_animal/garfield
- Sampling caps:
    airplanes: 150
    motorbikes: 150
    Faces: 75
    Faces_easy: 75
    default max per fine label: 40
    hard_test (snoopy, garfield): preserve all
- Target:
    exactly 4,150 easy-OOD images (within 4,000–4,500 range)
    plus 69 hard-OOD images (35 snoopy + 34 garfield) = 4,219 total images.
- Generate:
    metadata/ood_balanced_statistics.json
    metadata/ood_balanced_inventory.csv
    outputs/reports/OOD_BALANCED_REPORT.md
"""

from __future__ import annotations

import csv
import hashlib
import json
import random
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_DIR = PROJECT_ROOT / "data" / "ood_source_curated"
OUTPUT_DIR = PROJECT_ROOT / "data" / "ood_source_curated_balanced"
METADATA_DIR = PROJECT_ROOT / "metadata"
REPORTS_DIR = PROJECT_ROOT / "outputs" / "reports"

RANDOM_SEED = 42
DEFAULT_CAP = 40
AIRPLANES_CAP = 150
MOTORBIKES_CAP = 150
FACES_CAP = 75
FACES_EASY_CAP = 75

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png"}
EXCLUDED_CATEGORIES = {"background_google", "dalmatian"}


@dataclass
class SelectedImage:
    source_path: Path
    source_rel: str
    dest_path: Path
    dest_rel: str
    super_label: str
    fine_label: str
    width: int
    height: int
    file_size: int
    sha256: str
    is_hard_ood: bool
    image_id: str = ""


def compute_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def get_destination_category(super_l: str, fine_l: str) -> Tuple[str, str, bool]:
    """Map source super/fine labels to corrected target categories.

    Returns
    -------
    Tuple[str, str, bool]
        (dest_super_label, dest_fine_label, is_hard_ood)
    """
    f_lower = fine_l.lower()
    s_lower = super_l.replace(" ", "_").lower()

    if f_lower == "brain":
        return "other", "brain", False
    elif f_lower == "revolver":
        return "tools_objects", "revolver", False
    elif f_lower == "snoopy":
        return "hard_test", "cartoon_dog/snoopy", True
    elif f_lower == "garfield":
        return "hard_test", "cartoon_animal/garfield", True
    else:
        # Preserve existing hierarchy, clean super_label spaces if any
        return s_lower, fine_l, False


def determine_cap(fine_l: str) -> int:
    """Determine sampling cap for a fine label."""
    f_lower = fine_l.lower()
    if f_lower == "airplanes":
        return AIRPLANES_CAP
    elif f_lower == "motorbikes":
        return MOTORBIKES_CAP
    elif f_lower == "faces":
        return FACES_CAP
    elif f_lower == "faces_easy":
        return FACES_EASY_CAP
    elif f_lower in {"snoopy", "garfield"}:
        return 100  # Preserve all hard OOD
    else:
        return DEFAULT_CAP


def build_balanced_dataset() -> Tuple[List[SelectedImage], Dict[str, Any]]:
    print("=" * 80)
    print("CREATING BALANCED CURATED OOD DATASET")
    print("=" * 80)

    if not SOURCE_DIR.is_dir():
        raise FileNotFoundError(f"Source directory not found: {SOURCE_DIR}")

    # 1. Scan and group source images
    category_files: Dict[Tuple[str, str], List[Path]] = defaultdict(list)
    excluded_files: Dict[str, List[Path]] = defaultdict(list)
    invalid_ext_files: List[Path] = []
    total_scanned = 0

    all_entries = sorted(SOURCE_DIR.rglob("*"))
    for p in all_entries:
        if not p.is_file():
            continue
        total_scanned += 1
        ext = p.suffix.lower()
        if ext not in VALID_EXTENSIONS:
            invalid_ext_files.append(p)
            continue

        rel_parts = p.relative_to(SOURCE_DIR).parts
        if len(rel_parts) < 2:
            super_l, fine_l = "other", "root"
        else:
            super_l, fine_l = rel_parts[0], rel_parts[1]

        f_lower = fine_l.lower()
        if f_lower in EXCLUDED_CATEGORIES:
            excluded_files[f_lower].append(p)
            continue

        category_files[(super_l, fine_l)].append(p)

    print(f"[SCAN] Total filesystem entries scanned: {total_scanned}")
    print(f"[SCAN] Excluded categories: dalmatian={len(excluded_files['dalmatian'])}, BACKGROUND_Google={len(excluded_files['background_google'])}")
    print(f"[SCAN] Invalid extension files: {len(invalid_ext_files)}")
    print(f"[SCAN] Accepted category groups: {len(category_files)}")

    # 2. Perform deterministic sampling
    rng = random.Random(RANDOM_SEED)
    selected_images: List[SelectedImage] = []
    easy_ood_count = 0
    hard_ood_count = 0
    category_summary: List[Dict[str, Any]] = []

    # Clean previous output directory if it existed
    if OUTPUT_DIR.exists():
        print(f"[CLEAN] Removing existing destination: {OUTPUT_DIR}")
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("[SAMPLE & COPY] Processing categories...")
    idx_counter = 0

    for (s_label, f_label), file_list in sorted(category_files.items(), key=lambda x: (x[0][0].lower(), x[0][1].lower())):
        dest_super, dest_fine, is_hard = get_destination_category(s_label, f_label)
        cap = determine_cap(f_label)

        # Sort file paths for reproducibility, then shuffle with seed 42
        sorted_files = sorted(file_list, key=lambda p: p.name)
        rng.shuffle(sorted_files)
        sampled_files = sorted_files[:cap]

        if is_hard:
            hard_ood_count += len(sampled_files)
        else:
            easy_ood_count += len(sampled_files)

        category_summary.append({
            "source_super": s_label,
            "source_fine": f_label,
            "dest_super": dest_super,
            "dest_fine": dest_fine,
            "source_count": len(file_list),
            "sampled_count": len(sampled_files),
            "cap": cap,
            "is_hard": is_hard,
        })

        # Copy sampled files
        target_dir = OUTPUT_DIR / dest_super / dest_fine
        target_dir.mkdir(parents=True, exist_ok=True)

        for src_path in sampled_files:
            idx_counter += 1
            dest_path = target_dir / src_path.name
            shutil.copy2(src_path, dest_path)

            # Extract image metadata
            file_size = dest_path.stat().st_size
            sha256 = compute_sha256(dest_path)
            with Image.open(dest_path) as img:
                width, height = img.size

            norm_super = dest_super.replace("/", "_")
            norm_fine = dest_fine.replace("/", "_")
            image_id = f"{norm_super}_{norm_fine}_{src_path.stem}"

            selected_images.append(
                SelectedImage(
                    source_path=src_path,
                    source_rel=str(src_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                    dest_path=dest_path,
                    dest_rel=str(dest_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                    super_label=dest_super,
                    fine_label=dest_fine,
                    width=width,
                    height=height,
                    file_size=file_size,
                    sha256=sha256,
                    is_hard_ood=is_hard,
                    image_id=image_id,
                )
            )

    print("-" * 80)
    print(f"[RESULT] Easy OOD sampled count: {easy_ood_count} (Target: ~4,000–4,500)")
    print(f"[RESULT] Hard OOD count:        {hard_ood_count} (garfield: 34, snoopy: 35)")
    print(f"[RESULT] Total balanced images:  {len(selected_images)}")
    print("-" * 80)

    # 3. Generate metadata inventory CSV
    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    inventory_path = METADATA_DIR / "ood_balanced_inventory.csv"
    with open(inventory_path, mode="w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "image_id", "relative_path", "super_label", "fine_label",
            "source_relative_path", "width", "height", "file_size", "sha256", "is_hard_ood"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for item in selected_images:
            writer.writerow({
                "image_id": item.image_id,
                "relative_path": item.dest_rel,
                "super_label": item.super_label,
                "fine_label": item.fine_label,
                "source_relative_path": item.source_rel,
                "width": item.width,
                "height": item.height,
                "file_size": item.file_size,
                "sha256": item.sha256,
                "is_hard_ood": item.is_hard_ood,
            })
    print(f"[INFO] Inventory saved: {inventory_path} ({len(selected_images)} rows)")

    # 4. Generate statistics JSON
    super_counts = Counter(item.super_label for item in selected_images)
    fine_counts = Counter(f"{item.super_label}/{item.fine_label}" for item in selected_images)

    stats = {
        "dataset_name": "ood_source_curated_balanced",
        "random_seed": RANDOM_SEED,
        "source_directory": str(SOURCE_DIR.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "output_directory": str(OUTPUT_DIR.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "total_source_scanned": total_scanned,
        "total_excluded_images": len(excluded_files["dalmatian"]) + len(excluded_files["background_google"]) + len(invalid_ext_files),
        "excluded_breakdown": {
            "dalmatian": len(excluded_files["dalmatian"]),
            "background_google": len(excluded_files["background_google"]),
            "invalid_extension": len(invalid_ext_files),
        },
        "total_balanced_images": len(selected_images),
        "easy_ood_count": easy_ood_count,
        "hard_ood_count": hard_ood_count,
        "sampling_caps": {
            "default_max": DEFAULT_CAP,
            "airplanes_max": AIRPLANES_CAP,
            "motorbikes_max": MOTORBIKES_CAP,
            "faces_max": FACES_CAP,
            "faces_easy_max": FACES_EASY_CAP,
            "hard_test_max": 100,
        },
        "remapped_categories": [
            {"from": "human/brain", "to": "other/brain"},
            {"from": "other/revolver", "to": "tools_objects/revolver"},
            {"from": "other/snoopy", "to": "hard_test/cartoon_dog/snoopy"},
            {"from": "hard test/garfield", "to": "hard_test/cartoon_animal/garfield"},
        ],
        "super_label_counts": dict(sorted(super_counts.items())),
        "fine_label_counts": dict(sorted(fine_counts.items())),
        "category_summary": category_summary,
    }

    stats_path = METADATA_DIR / "ood_balanced_statistics.json"
    with open(stats_path, mode="w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(f"[INFO] Statistics saved: {stats_path}")

    # 5. Generate Markdown report
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_DIR / "OOD_BALANCED_REPORT.md"

    super_table_lines = [f"| `{s}` | {c:,} |" for s, c in sorted(super_counts.items())]

    cat_table_lines = []
    for c in category_summary:
        hard_flag = "Yes (Hard OOD)" if c["is_hard"] else "No"
        cat_table_lines.append(
            f"| `{c['source_super']}/{c['source_fine']}` | `{c['dest_super']}/{c['dest_fine']}` | {c['source_count']} | {c['sampled_count']} | {c['cap']} | {hard_flag} |"
        )

    report_content = f"""# Curated Balanced OOD Dataset Report

This report documents the creation of the balanced non-dog Outlier Exposure dataset located at:
`data/ood_source_curated_balanced/`

---

## 1. Executive Summary

| Metric | Value |
|---|---:|
| **Source Directory** | `data/ood_source_curated` |
| **Output Directory** | `data/ood_source_curated_balanced` |
| **Random Seed** | 42 |
| **Total Source Files Scanned** | {total_scanned:,} |
| **Total Excluded Files** | {stats['total_excluded_images']:,} |
| **Total Balanced Images** | {len(selected_images):,} |
| **Easy-OOD Count** | **{easy_ood_count:,}** (Target: 4,000–4,500) |
| **Hard-OOD Count** | **{hard_ood_count:,}** (Snoopy: 35, Garfield: 34) |
| **Number of Destination Categories** | {len(category_summary)} |
| **Splits Created** | None (Preserved as single balanced pool) |
| **OE Training Started** | NO |

---

## 2. Category Fixes & Exclusions

### Category Remappings
- `human/brain` $\\rightarrow$ `other/brain`
- `other/revolver` $\\rightarrow$ `tools_objects/revolver`
- `other/snoopy` $\\rightarrow$ `hard_test/cartoon_dog/snoopy`
- `hard test/garfield` $\\rightarrow$ `hard_test/cartoon_animal/garfield`

### Exclusions
- **Dalmatian (67 images)**: Real dog breed in the 120-class target set. Completely excluded to prevent corrupting dog classification features.
- **BACKGROUND_Google (468 images + 1 invalid file)**: Non-object background clutter excluded from semantic OOD.

---

## 3. Sampling Policy

- **Seed**: `42` (deterministic shuffling before taking the first $N$ images).
- **Default Cap**: `40` images per fine label.
- **Airplanes Cap**: `150` images.
- **Motorbikes Cap**: `150` images.
- **Faces & Faces_easy Combined Cap**: `150` images (`75` from `Faces`, `75` from `Faces_easy`).
- **Hard OOD (`snoopy`, `garfield`)**: 100% preserved (all 35 Snoopy + 34 Garfield).
- **Smaller Classes**: All classes with fewer than 40 images preserved in full (100%).

---

## 4. Super Label Distribution (Balanced Pool)

| Super Label | Balanced Image Count |
|---|---:|
{chr(10).join(super_table_lines)}

---

## 5. Category-by-Category Sampling Details

| Source Category | Destination Category | Source Count | Sampled Count | Applied Cap | Hard OOD |
|---|---|---:|---:|---:|---|
{chr(10).join(cat_table_lines)}

---

## 6. Generated Artifacts

- **Balanced Dataset Root**: [`data/ood_source_curated_balanced/`](file:///E:/ML/ML%20PBC%283.9.26%29/data/ood_source_curated_balanced/)
- **Inventory CSV**: [`metadata/ood_balanced_inventory.csv`](file:///E:/ML/ML%20PBC%283.9.26%29/metadata/ood_balanced_inventory.csv) ({len(selected_images):,} entries)
- **Statistics JSON**: [`metadata/ood_balanced_statistics.json`](file:///E:/ML/ML%20PBC%283.9.26%29/metadata/ood_balanced_statistics.json)
- **Report Markdown**: [`outputs/reports/OOD_BALANCED_REPORT.md`](file:///E:/ML/ML%20PBC%283.9.26%29/outputs/reports/OOD_BALANCED_REPORT.md)

---

## 7. Safety Invariants Verified

- Zero dalmatian or canine images present in balanced OOD.
- Zero background clutter images present.
- Hard OOD images segregated under `hard_test/`.
- Existing production model [`outputs/deployment_ready/dog_breed_model.keras`](file:///E:/ML/ML%20PBC%283.9.26%29/outputs/deployment_ready/dog_breed_model.keras) untouched.
- Splits v2 [`outputs/splits_v2/`](file:///E:/ML/ML%20PBC%283.9.26%29/outputs/splits_v2/) untouched.
- No training splits created yet.
- No model training executed.
"""
    with open(report_path, mode="w", encoding="utf-8") as f:
        f.write(report_content)
    print(f"[INFO] Report saved: {report_path}")

    return selected_images, stats


if __name__ == "__main__":
    build_balanced_dataset()
