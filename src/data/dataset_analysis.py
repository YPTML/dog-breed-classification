"""Read-only audit of the local Kaggle dog dataset and existing split CSVs.

Run: python -m src.data.dataset_analysis [--project-root PATH]
Only the JSON statistics and Markdown report are written. No training imports.
The legacy manifest summary API remains available for ID/OOD callers.
"""

from __future__ import annotations

import csv
import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
from pathlib import Path
import argparse
import hashlib
import statistics
from datetime import datetime, timezone
from itertools import combinations

from src.data.validate_dataset import IMAGE_EXTENSIONS, inspect_image


def summarize_manifest(manifest_path: str | Path) -> dict:
    with Path(manifest_path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    by_distribution = Counter(row.get("distribution", "") for row in rows)
    by_split = Counter(row.get("split", "") for row in rows)
    by_label = Counter(row.get("label", "") for row in rows if row.get("distribution") == "id")
    return {
        "total_images": len(rows),
        "distribution_counts": dict(by_distribution),
        "split_counts": dict(by_split),
        "id_class_count": len(by_label),
        "id_class_counts": dict(sorted(by_label.items())),
        "min_class_count": min(by_label.values()) if by_label else 0,
        "max_class_count": max(by_label.values()) if by_label else 0,
    }


def write_statistics(manifest_path: str | Path, output_path: str | Path) -> dict:
    statistics = summarize_manifest(manifest_path)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(statistics, handle, indent=2, ensure_ascii=False)
    return statistics


def numeric_summary(values) -> dict:
    values = list(values)
    return {"count": len(values), "min": min(values) if values else None,
            "max": max(values) if values else None,
            "mean": statistics.mean(values) if values else None,
            "median": statistics.median(values) if values else None}


def flat_image_files(folder: Path) -> list[Path]:
    """List a flat image pool using Windows dir, with a Python fallback."""
    if not folder.is_dir():
        return []
    try:
        output = subprocess.check_output(
            ["cmd", "/c", f'dir /b /a-d "{folder}"'],
            text=True, encoding="utf-8", errors="replace", stderr=subprocess.DEVNULL,
        )
        return sorted((folder / line.strip() for line in output.splitlines()
                       if line.strip() and Path(line.strip()).suffix.lower() in IMAGE_EXTENSIONS),
                      key=lambda item: item.name.lower())
    except (OSError, subprocess.SubprocessError):
        try:
            return sorted((Path(entry.path) for entry in os.scandir(folder)
                           if entry.is_file() and Path(entry.name).suffix.lower() in IMAGE_EXTENSIONS),
                          key=lambda item: item.name.lower())
        except OSError:
            return []


def read_csv(path: Path, required: set, issues: list) -> list:
    if not path.is_file():
        issues.append({"path": path.as_posix(), "reason": "missing_csv"})
        return []
    try:
        with path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            missing = required - set(reader.fieldnames or [])
            if missing:
                issues.append({"path": path.as_posix(), "reason": "missing_columns",
                               "columns": sorted(missing)})
                return []
            rows = list(reader)
        for number, row in enumerate(rows, 2):
            if any(not row.get(key) for key in required):
                issues.append({"path": path.as_posix(), "row": number, "reason": "empty_required_field"})
        return rows
    except (OSError, UnicodeError, csv.Error) as exc:
        issues.append({"path": path.as_posix(), "reason": "csv_read_error", "error": str(exc)})
        return []


def duplicate_values(values) -> dict:
    return {key: count for key, count in Counter(values).items() if count > 1}


def summarize_pool(records: list) -> dict:
    good = [r for r in records if r["status"] == "ok"]
    return {
        "total_images": len(records), "decoded_images": len(good),
        "cannot_open": [r for r in records if r["status"] == "cannot_open"],
        "corrupt": [r for r in records if r["status"] == "corrupt"],
        "warnings": [r for r in records if r["warnings"]],
        "resolution": {
            "width": numeric_summary(r["width"] for r in good),
            "height": numeric_summary(r["height"] for r in good),
            "pixels": numeric_summary(r["width"] * r["height"] for r in good),
            "counts": dict(sorted(Counter(f'{r["width"]}x{r["height"]}' for r in good).items())),
            "either_dimension_below_224": sum(min(r["width"], r["height"]) < 224 for r in good)},
        "aspect_ratio": numeric_summary(r["aspect_ratio"] for r in good),
        "orientation": dict(Counter("landscape" if r["width"] > r["height"] else
                                    "portrait" if r["width"] < r["height"] else "square" for r in good)),
        "formats": dict(Counter(r["format"] for r in good)),
        "modes": dict(Counter(r["mode"] for r in good)),
    }


def audit_project(root: Path, cached_train_records: list[dict] | None = None) -> dict:
    root = root.resolve()
    data = root / "data"
    issues = []
    labels_path = data / "labels.csv"
    mapping_path = root / "outputs/deployment_ready/class_names.json"
    split_dir = root / "outputs/splits"
    metadata_paths = [labels_path, mapping_path, *sorted(split_dir.rglob("*.csv"))]
    # Avoid a recursive pre-scan of the image tree: on Windows this can block
    # behind antivirus/indexing and it is not needed to inspect image content.
    def snapshot():
        paths = [p for p in metadata_paths if p.is_file()]
        return {p.relative_to(root).as_posix(): (p.stat().st_size, p.stat().st_mtime_ns)
                for p in paths}

    before = snapshot()
    def hashes():
        return {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in metadata_paths if p.is_file()}
    hashes_before = hashes()
    labels = read_csv(labels_path, {"id", "breed"}, issues)
    try:
        mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
        expected = list(mapping.values()) if isinstance(mapping, dict) else mapping
        if not isinstance(expected, list) or not all(isinstance(x, str) and x for x in expected):
            raise ValueError("Class mapping must contain nonempty class names")
        if len(expected) != 120 or len(set(expected)) != 120:
            issues.append({"reason": "expected_mapping_not_120_unique_classes"})
        expected = sorted(set(expected))
    except (OSError, ValueError, TypeError) as exc:
        expected = []
        issues.append({"reason": "expected_class_mapping_unavailable", "error": str(exc)})
    label_map = {r["id"]: r["breed"] for r in labels if r.get("id") and r.get("breed")}
    pools = {}
    all_records = {}
    # The 120-class dataset is data/train. data/test has no labels and is
    # reported as a separate file-count-only pool so it cannot affect class
    # or split statistics. A cache is accepted so a long image scan can be
    # resumed in bounded chunks without changing the source dataset.
    for name in ("train",):
        if cached_train_records is not None:
            all_records[name] = cached_train_records
            pools[name] = summarize_pool(cached_train_records)
            continue
        folder = data / name
        if not folder.is_dir():
            issues.append({"path": folder.as_posix(), "reason": "missing_image_directory"})
        files = flat_image_files(folder)
        if folder.is_dir() and not files:
            issues.append({"path": folder.as_posix(), "reason": "no_image_files_found"})
        records = []
        with ThreadPoolExecutor(max_workers=8, thread_name_prefix="dataset-audit") as executor:
            inspected = executor.map(inspect_image, files)
            for index, (path, record) in enumerate(zip(files, inspected), 1):
                record["path"] = path.relative_to(root).as_posix()
                records.append(record)
                if index % 500 == 0:
                    print(f"Scanned {name}: {index}/{len(files)}", flush=True)
        all_records[name] = records
        pools[name] = summarize_pool(records)
    test_folder = data / "test"
    test_files = flat_image_files(test_folder)
    if test_folder.is_dir() and not test_files:
        issues.append({"path": test_folder.as_posix(), "reason": "no_image_files_found"})
    pools["test"] = {
        "total_images": len(test_files), "decoded_images": None,
        "cannot_open": [], "corrupt": [], "warnings": [],
        "resolution": {key: numeric_summary([]) for key in ("width", "height", "pixels")}
        | {"counts": {}, "either_dimension_below_224": None},
        "aspect_ratio": numeric_summary([]), "orientation": {}, "formats": {}, "modes": {},
        "audit_note": "Unlabeled Kaggle test pool; files counted by extension only; no class or split metrics.",
    }
    all_records["test"] = []
    physical = {Path(r["path"]).relative_to("data/train").as_posix(): r for r in all_records["train"]}
    classes = sorted(set(expected) | set(label_map.values()))
    counts = Counter()
    valid_counts = Counter()
    for image_id, breed in label_map.items():
        record = physical.get(image_id + ".jpg")
        if record:
            counts[breed] += 1
            if record["status"] == "ok":
                valid_counts[breed] += 1
    per_class = {c: counts[c] for c in classes}
    missing_images = sorted(i + ".jpg" for i in label_map if i + ".jpg" not in physical)
    unlabeled = sorted(set(physical) - {i + ".jpg" for i in label_map})
    observed_classes = sorted(c for c, count in counts.items() if count)

    def split_summary(path):
        rows = read_csv(path, {"id", "breed", "img_name"}, issues)
        by_class = Counter(r.get("breed") for r in rows if r.get("breed"))
        ids = {r["id"] for r in rows if r.get("id")}
        filenames = [r.get("img_name") for r in rows]
        return {
            "path": path.relative_to(root).as_posix(), "exists": path.is_file(),
            "rows": len(rows), "unique_ids": len(ids),
            "percent_of_labeled_rows": 100 * len(rows) / len(labels) if labels else None,
            "class_count": len(by_class), "images_per_class": {c: by_class[c] for c in classes},
            "missing_classes": sorted(set(expected) - set(by_class)),
            "unexpected_classes": sorted(set(by_class) - set(expected)) if expected else [],
            "images_per_class_summary": numeric_summary(by_class[c] for c in classes),
            "duplicate_ids": duplicate_values(r.get("id") for r in rows),
            "duplicate_filenames": duplicate_values(filenames),
            "missing_images": sorted({f for f in filenames if f and f not in physical}),
            "invalid_images": sorted({f for f in filenames if f in physical and physical[f]["status"] != "ok"}),
            "unknown_ids": sorted(ids - set(label_map)),
            "label_mismatches": [r for r in rows if r.get("id") in label_map and r.get("breed") != label_map[r["id"]]],
            "filename_mismatches": [r for r in rows if r.get("img_name") != str(r.get("id")) + ".jpg"],
        }, ids

    def split_group(paths):
        summaries, ids = {}, {}
        for name, path in paths.items():
            summaries[name], ids[name] = split_summary(path)
        union = set().union(*ids.values())
        return {"distribution": summaries,
                "overlap_ids": {f"{a}/{b}": sorted(ids[a] & ids[b]) for a, b in combinations(ids, 2)},
                "unassigned_ids": sorted(set(label_map) - union),
                "extra_ids": sorted(union - set(label_map))}

    primary = split_group({name: split_dir / f"{filename}.csv"
                           for name, filename in (("train", "train"), ("validation", "val"), ("test", "test"))})
    kfold = {}
    for train_path in sorted((split_dir / "kfold").glob("fold_*_train.csv")):
        fold = train_path.stem.removesuffix("_train")
        kfold[fold] = split_group({"train": train_path,
                                  "validation": train_path.with_name(fold + "_val.csv"),
                                  "test": train_path.with_name("holdout_test.csv")})
    positive = [n for n in per_class.values() if n]
    mean = statistics.mean(per_class.values()) if per_class else 0
    # The active labeled pool is flat. Report known auxiliary data folders
    # without recursively walking the large image pools a second time.
    empty = []
    other_images = []
    for entry in (data / "in_distribution", data / "ood", data / "ood_hard"):
        if not entry.exists():
            continue
        try:
            children = list(entry.iterdir())
            if not children:
                empty.append(entry.relative_to(root).as_posix())
            for child in children:
                if child.is_file() and child.suffix.lower() in IMAGE_EXTENSIONS:
                    other_images.append(child.relative_to(root).as_posix())
                elif child.is_dir() and not any(child.iterdir()):
                    empty.append(child.relative_to(root).as_posix())
        except OSError as exc:
            issues.append({"path": entry.as_posix(), "reason": "directory_read_error", "error": str(exc)})
    after = snapshot()
    hashes_after = hashes()
    return {
        "schema_version": 1, "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": root.as_posix(), "scope": "Kaggle labeled data/train plus unlabeled data/test; archives not extracted",
        "expected_class_count": 120, "expected_class_source": mapping_path.relative_to(root).as_posix(),
        "expected_classes": expected,
        "total_images": pools["train"]["total_images"],
        "total_physical_image_candidates": sum(p["total_images"] for p in pools.values()),
        "unlabeled_image_count": pools["test"]["total_images"],
        "labeled_image_count": sum(per_class.values()), "class_count": len(observed_classes),
        "label_rows": len(labels), "label_class_count": len(set(label_map.values())),
        "images_per_class": per_class, "valid_images_per_class": {c: valid_counts[c] for c in classes},
        "images_per_class_summary": numeric_summary(per_class.values()),
        "missing_classes": sorted(set(expected) - set(observed_classes)),
        "classes_without_decodable_images": sorted(set(expected) - set(valid_counts)),
        "unexpected_classes": sorted(set(label_map.values()) - set(expected)) if expected else [],
        "class_imbalance": {
            "max_min_nonempty_ratio": max(positive) / min(positive) if positive else None,
            "coefficient_of_variation": statistics.pstdev(per_class.values()) / mean if mean else None,
            "zero_image_classes": [c for c, n in per_class.items() if n == 0],
            "below_half_mean": [c for c, n in per_class.items() if n < mean / 2],
            "above_twice_mean": [c for c, n in per_class.items() if n > 2 * mean]},
        "pools": {"labeled_pool": pools["train"], "unlabeled_kaggle_test": pools["test"]},
        "missing_labeled_images": missing_images, "unlabeled_files_in_train": unlabeled,
        "duplicate_label_ids": duplicate_values(r.get("id") for r in labels),
        "empty_folders": empty, "additional_images_not_in_audit_scope": other_images,
        "splits": primary, "kfold_splits": kfold, "metadata_issues": issues,
        "preservation": {"data_and_split_metadata_stats_unchanged": before == after,
                         "data_and_split_file_stats_unchanged": before == after,
                         "metadata_sha256_unchanged": hashes_before == hashes_after,
                         "metadata_sha256_before": hashes_before, "metadata_sha256_after": hashes_after,
                         "changed_paths": sorted(k for k in before.keys() | after.keys() if before.get(k) != after.get(k))}}


def render_report(result: dict) -> str:
    s = result
    lines = ["# DATASET AUDIT", "", f"Generated (UTC): {s['generated_at_utc']}", "",
             "Read-only audit; no training, image deletion, automatic repair, or split generation.", "",
             "## Scope and counting", "",
             "`data/train` is a flat labeled image pool, joined by ID to `data/labels.csv`. "
             "The existing `outputs/splits/{train,val,test}.csv` define the internal splits. "
             "`data/test` is the separate unlabeled Kaggle prediction pool; its class distribution is unknown. "
             "K-fold splits reuse images and are never added to image totals. ZIP contents are not counted.", "",
             f"- Labeled dataset image candidates: **{s['total_images']:,}**",
             f"- All physical image candidates in train + unlabeled test: **{s['total_physical_image_candidates']:,}**",
             f"- Labeled physical images joined to labels: **{s['labeled_image_count']:,}**; label rows: {s['label_rows']:,}",
             f"- Observed classes: **{s['class_count']} / {s['expected_class_count']}**",
             f"- Expected class source: `{s['expected_class_source']}`",
             f"- Images/class (min / max / mean / median): {s['images_per_class_summary']['min']} / "
             f"{s['images_per_class_summary']['max']} / {s['images_per_class_summary']['mean']} / {s['images_per_class_summary']['median']}",
             "", "Counts include corrupt candidates when their filename has a label; valid counts are recorded separately. "
             "Class summaries include expected classes with zero images.", "", "## Class imbalance", "",
             f"- Max/min among nonempty classes: {s['class_imbalance']['max_min_nonempty_ratio']}",
             f"- Population standard deviation / mean: {s['class_imbalance']['coefficient_of_variation']}",
             f"- Classes below half mean: {s['class_imbalance']['below_half_mean']}",
             f"- Classes above twice mean: {s['class_imbalance']['above_twice_mean']}",
             "", "These are descriptive indicators, not an automatic resampling recommendation.", "",
             "## Image integrity, resolution and aspect ratio", "",
             "Every image candidate is inspected with a format-aware header parser; non-JPEG formats are verified with Pillow and JPEGs are checked for their end marker. "
             "`cannot_open` means failure before header opening; `corrupt` means a format check failed. "
             "They are disjoint operational categories; an unopenable file may also be damaged. "
             "Dimensions and width/height ratios use stored pixels before EXIF rotation and only successful checks. "
             "Files are selected by the image extensions listed in the validator.", ""]
    for name, pool in s["pools"].items():
        lines += [f"### {name}", "", f"- Candidates / decoded: {pool['total_images']} / {pool['decoded_images']}",
                  f"- Cannot open / corrupt: {len(pool['cannot_open'])} / {len(pool['corrupt'])}",
                  f"- Files with Pillow warnings: {len(pool['warnings'])}"]
        for key in ("width", "height", "pixels"):
            stats = pool["resolution"][key]
            lines.append(f"- {key} (min / max / mean / median): {stats['min']} / {stats['max']} / {stats['mean']} / {stats['median']}")
        lines += [f"- Aspect ratio (width/height): {pool['aspect_ratio']}",
                  f"- Orientation: {pool['orientation']}",
                  f"- Either dimension below 224: {pool['resolution']['either_dimension_below_224']} (informational)", ""]
        for record in pool["cannot_open"] + pool["corrupt"]:
            lines.append(f"- `{record['path']}`: {record['status']} / {record['stage']}: {record['error']}")
    lines += ["## Metadata and folder findings", ""]
    for key in ("missing_classes", "classes_without_decodable_images", "unexpected_classes",
                "missing_labeled_images", "unlabeled_files_in_train", "duplicate_label_ids",
                "empty_folders", "additional_images_not_in_audit_scope", "metadata_issues"):
        lines.append(f"- {key}: {json.dumps(s[key], ensure_ascii=False)}")
    lines += ["", "Empty ID/OOD placeholder folders are reported as found; they are not missing breed folders "
              "because the active dataset uses CSV labels.", "", "## Existing split distribution", "",
              "| Split | Rows | % of labeled rows | Classes | Min/class | Max/class |", "|---|---:|---:|---:|---:|---:|"]
    def group_lines(group, prefix=""):
        out = []
        for name, info in group["distribution"].items():
            summary = info["images_per_class_summary"]
            out.append(f"| {prefix}{name} | {info['rows']} | {info['percent_of_labeled_rows']} | {info['class_count']} | {summary['min']} | {summary['max']} |")
        return out
    lines += group_lines(s["splits"])
    lines += ["", "### K-fold distribution (separate alternative split scheme)", "",
              "| Fold/split | Rows | % of labeled rows | Classes | Min/class | Max/class |", "|---|---:|---:|---:|---:|---:|"]
    for fold, group in s["kfold_splits"].items():
        lines += group_lines(group, fold + "/")
    lines += ["", "### Split integrity", ""]
    for name, group in {"primary": s["splits"], **s["kfold_splits"]}.items():
        overlaps = {key: len(value) for key, value in group["overlap_ids"].items()}
        lines.append(f"- {name}: overlap IDs {overlaps}; unassigned IDs {len(group['unassigned_ids'])}; extra IDs {len(group['extra_ids'])}.")
        for split, info in group["distribution"].items():
            findings = {k: info[k] for k in ("missing_classes", "unexpected_classes", "duplicate_ids", "duplicate_filenames", "missing_images", "invalid_images", "unknown_ids", "label_mismatches", "filename_mismatches") if info[k]}
            if findings:
                lines.append(f"  - {split}: {json.dumps(findings, ensure_ascii=False)}")
    lines += ["", "Split overlap checks compare IDs, not visual/content duplicates. Complete ID lists and per-class k-fold counts are in the JSON.", "",
              "## Images per class", "", "| Class | Physical | Decodable | Train | Validation | Test |", "|---|---:|---:|---:|---:|---:|"]
    for cls, count in s["images_per_class"].items():
        split_counts = [s["splits"]["distribution"][split]["images_per_class"][cls] for split in ("train", "validation", "test")]
        lines.append(f"| {cls} | {count} | {s['valid_images_per_class'][cls]} | {' | '.join(map(str, split_counts))} |")
    lines += ["", "## Preservation and reproduction", "",
              f"- Labels, class mapping and split file sizes/modification times unchanged: {s['preservation']['data_and_split_metadata_stats_unchanged']}",
              f"- Labels, class mapping and all split CSV SHA-256 hashes unchanged: {s['preservation']['metadata_sha256_unchanged']}",
              "- Image contents are not independently hashed; preservation evidence for images uses size and modification time.",
              "", "```powershell", ".\\.venv\\Scripts\\python.exe -m src.data.validate_dataset", "```", "",
              "Equivalent: `python -m src.data.dataset_analysis`. Outputs: `metadata/dataset_statistics.json` and `outputs/reports/DATASET_AUDIT.md`.", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--cached-train-records", type=Path, default=None,
                        help="JSON records from a bounded read-only train scan")
    args = parser.parse_args()
    cached = None
    if args.cached_train_records:
        cached = json.loads(args.cached_train_records.read_text(encoding="utf-8"))
    result = audit_project(args.project_root, cached_train_records=cached)
    for relative, content in (("metadata/dataset_statistics.json", json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n"),
                              ("outputs/reports/DATASET_AUDIT.md", render_report(result))):
        target = args.project_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        print(f"Wrote {target}", flush=True)
    print(json.dumps({key: result[key] for key in ("total_images", "labeled_image_count", "class_count", "images_per_class_summary")}, indent=2))


if __name__ == "__main__":
    main()
