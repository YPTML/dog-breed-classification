"""Reproducible quality evaluation for the deployed 120-class OE V2 model.

This script is evaluation-only.  It loads the production ``OEV2Predictor``,
keeps its model, centroids, class mapping, and OOD thresholds unchanged, and
writes all results to a caller-selected directory.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import platform
import random
import statistics
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.oe_v2_predictor import (  # noqa: E402
    EXPECTED_CENTROIDS_SHA256,
    EXPECTED_CLASS_NAMES_SHA256,
    EXPECTED_MODEL_SHA256,
    LOCKED_DETECTOR,
    LOCKED_T_NOT_DOG,
    LOCKED_T_UNCERTAIN,
    OEV2Predictor,
    compute_class_names_sha256,
    compute_file_sha256,
)


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
BOOTSTRAP_SEED = 20261016
NEAR_DUPLICATE_MAX_DHASH = 4


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def safe_package_version(name: str) -> str | None:
    try:
        return importlib_metadata.version(name)
    except importlib_metadata.PackageNotFoundError:
        return None


def percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return float("nan")
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> list[float] | None:
    if total <= 0:
        return None
    p = successes / total
    denom = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denom
    half = z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * total)) / total) / denom
    return [max(0.0, center - half), min(1.0, center + half)]


def rate_record(successes: int, total: int) -> dict[str, Any]:
    return {
        "count": int(successes),
        "denominator": int(total),
        "rate": float(successes / total) if total else None,
        "ci95_wilson": wilson_interval(successes, total),
    }


def average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    return ranks


def binary_auroc(negative_scores: np.ndarray, positive_scores: np.ndarray) -> float:
    values = np.concatenate([negative_scores, positive_scores]).astype(np.float64)
    ranks = average_ranks(values)
    n_neg = len(negative_scores)
    n_pos = len(positive_scores)
    u_pos = ranks[n_neg:].sum() - n_pos * (n_pos + 1) / 2.0
    return float(u_pos / (n_neg * n_pos))


def average_precision_ood(negative_scores: np.ndarray, positive_scores: np.ndarray) -> float:
    scores = np.concatenate([negative_scores, positive_scores]).astype(np.float64)
    labels = np.concatenate(
        [np.zeros(len(negative_scores), dtype=np.int8), np.ones(len(positive_scores), dtype=np.int8)]
    )
    order = np.argsort(-scores, kind="mergesort")
    scores = scores[order]
    labels = labels[order]
    tp = np.cumsum(labels == 1)
    fp = np.cumsum(labels == 0)
    distinct_ends = np.r_[np.flatnonzero(np.diff(scores) != 0), len(scores) - 1]
    recall = tp[distinct_ends] / max(1, len(positive_scores))
    precision = tp[distinct_ends] / np.maximum(1, tp[distinct_ends] + fp[distinct_ends])
    delta_recall = np.diff(np.r_[0.0, recall])
    return float(np.sum(delta_recall * precision))


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, top5_ok: np.ndarray, n_classes: int) -> dict[str, Any]:
    total = len(y_true)
    top1_ok = y_pred == y_true
    true_count = np.bincount(y_true, minlength=n_classes)
    valid_pred = y_pred >= 0
    pred_count = np.bincount(y_pred[valid_pred], minlength=n_classes)
    tp = np.bincount(y_true[top1_ok], minlength=n_classes)
    precision = np.divide(tp, pred_count, out=np.zeros(n_classes, dtype=float), where=pred_count > 0)
    recall = np.divide(tp, true_count, out=np.zeros(n_classes, dtype=float), where=true_count > 0)
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros(n_classes, dtype=float),
        where=(precision + recall) > 0,
    )
    return {
        "sample_count": int(total),
        "processing_error_count": int(np.sum(y_pred < 0)),
        "top1": rate_record(int(top1_ok.sum()), total),
        "top5": rate_record(int(top5_ok.sum()), total),
        "macro_precision": float(np.mean(precision)),
        "macro_recall": float(np.mean(recall)),
        "macro_f1": float(np.mean(f1)),
        "per_class_arrays": {
            "support": true_count.tolist(),
            "predicted": pred_count.tolist(),
            "tp": tp.tolist(),
            "precision": precision.tolist(),
            "recall": recall.tolist(),
            "f1": f1.tolist(),
        },
    }


def bootstrap_classification(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    top5_ok: np.ndarray,
    *,
    n_classes: int,
    reps: int,
    seed: int,
) -> dict[str, list[float]]:
    rng = np.random.default_rng(seed)
    values: dict[str, list[float]] = defaultdict(list)
    n = len(y_true)
    for _ in range(reps):
        idx = rng.integers(0, n, size=n)
        result = classification_metrics(y_true[idx], y_pred[idx], top5_ok[idx], n_classes)
        values["top1"].append(result["top1"]["rate"])
        values["top5"].append(result["top5"]["rate"])
        values["macro_precision"].append(result["macro_precision"])
        values["macro_recall"].append(result["macro_recall"])
        values["macro_f1"].append(result["macro_f1"])
    return {key: [percentile(vals, 2.5), percentile(vals, 97.5)] for key, vals in values.items()}


def bootstrap_ood(
    id_scores: np.ndarray,
    ood_scores: np.ndarray,
    *,
    reps: int,
    seed: int,
) -> dict[str, list[float]]:
    rng = np.random.default_rng(seed)
    aucs: list[float] = []
    aprs: list[float] = []
    for _ in range(reps):
        sampled_id = id_scores[rng.integers(0, len(id_scores), size=len(id_scores))]
        sampled_ood = ood_scores[rng.integers(0, len(ood_scores), size=len(ood_scores))]
        aucs.append(binary_auroc(sampled_id, sampled_ood))
        aprs.append(average_precision_ood(sampled_id, sampled_ood))
    return {
        "auroc": [percentile(aucs, 2.5), percentile(aucs, 97.5)],
        "auprc_ood_positive": [percentile(aprs, 2.5), percentile(aprs, 97.5)],
    }


def dhash_and_image_stats(data: bytes) -> tuple[int, float, float, int, int, str]:
    with Image.open(io.BytesIO(data)) as source:
        image_format = str(source.format or "")
        width, height = source.size
        gray = np.asarray(source.convert("L"), dtype=np.float32)
        tiny = np.asarray(source.convert("L").resize((9, 8), Image.Resampling.BILINEAR), dtype=np.int16)
    diff = tiny[:, 1:] > tiny[:, :-1]
    dhash = 0
    for bit in diff.ravel():
        dhash = (dhash << 1) | int(bit)
    brightness = float(np.mean(gray))
    if gray.shape[0] > 1 and gray.shape[1] > 1:
        gx = np.diff(gray, axis=1)
        gy = np.diff(gray, axis=0)
        sharpness = float((np.var(gx) + np.var(gy)) / 2.0)
    else:
        sharpness = 0.0
    return dhash, brightness, sharpness, width, height, image_format


def inventory_record(
    *,
    dataset: str,
    split: str,
    path: Path,
    label: str,
    expected_sha256: str,
    family_id: str,
    source: str,
) -> dict[str, Any]:
    return {
        "dataset": dataset,
        "split": split,
        "path": path,
        "label": label,
        "expected_sha256": expected_sha256,
        "family_id": family_id,
        "source": source,
    }


def build_inventory() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    inventory: list[dict[str, Any]] = []
    csv_sources: dict[str, Any] = {}
    for split in ("train", "val", "test"):
        csv_path = PROJECT_ROOT / "outputs" / "splits_v2" / f"{split}.csv"
        rows = read_csv(csv_path)
        csv_sources[f"id_{split}"] = {
            "path": str(csv_path.relative_to(PROJECT_ROOT)),
            "sha256": compute_file_sha256(csv_path),
            "rows": len(rows),
            "source": "Kaggle Dog Breed Identification (project metadata)",
        }
        for row in rows:
            inventory.append(
                inventory_record(
                    dataset="supported_120_breeds",
                    split=split,
                    path=PROJECT_ROOT / "data" / "train" / row["img_name"],
                    label=row["breed"],
                    expected_sha256=row["sha256"],
                    family_id=row["duplicate_group_id"],
                    source="kaggle_dog_breed_identification",
                )
            )

    for split in ("train", "val", "test"):
        csv_path = PROJECT_ROOT / "outputs" / "splits_v3_new" / f"{split}.csv"
        rows = read_csv(csv_path)
        csv_sources[f"unsupported_dog_{split}"] = {
            "path": str(csv_path.relative_to(PROJECT_ROOT)),
            "sha256": compute_file_sha256(csv_path),
            "rows": len(rows),
            "source": "Oxford-IIIT Pet",
        }
        for row in rows:
            inventory.append(
                inventory_record(
                    dataset="unsupported_dog_breeds",
                    split=split,
                    path=PROJECT_ROOT / row["relative_path"],
                    label=row["breed"],
                    expected_sha256=row["sha256"],
                    family_id=row["duplicate_family_id"],
                    source=row["source"],
                )
            )

    ood_manifest = PROJECT_ROOT / "metadata" / "ood_oe_v2_manifest.csv"
    rows = read_csv(ood_manifest)
    csv_sources["ood_manifest"] = {
        "path": str(ood_manifest.relative_to(PROJECT_ROOT)),
        "sha256": compute_file_sha256(ood_manifest),
        "rows": len(rows),
        "source": "Curated Caltech-101 / project OOD build",
    }
    for row in rows:
        inventory.append(
            inventory_record(
                dataset="ood_oe_v2",
                split=row["split"],
                path=PROJECT_ROOT / row["relative_path"],
                label=f"{row['super_label']}/{row['fine_label']}",
                expected_sha256=row["sha256"],
                family_id=row["family_id"],
                source=row["source_path"],
            )
        )
    return inventory, csv_sources


def audit_one(record: Mapping[str, Any]) -> dict[str, Any]:
    output = dict(record)
    path = Path(record["path"])
    output.update(
        {
            "relative_path": str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "exists": path.is_file(),
            "decode_ok": False,
            "hash_ok": False,
            "actual_sha256": "",
            "error": "",
            "dhash": None,
            "brightness": None,
            "sharpness_proxy": None,
            "width": None,
            "height": None,
            "format": "",
        }
    )
    if not path.is_file():
        output["error"] = "missing_file"
        return output
    try:
        data = path.read_bytes()
        actual_sha = sha256_bytes(data)
        output["actual_sha256"] = actual_sha
        output["hash_ok"] = actual_sha == record["expected_sha256"]
        dhash, brightness, sharpness, width, height, image_format = dhash_and_image_stats(data)
        output.update(
            {
                "decode_ok": True,
                "dhash": dhash,
                "brightness": brightness,
                "sharpness_proxy": sharpness,
                "width": width,
                "height": height,
                "format": image_format,
            }
        )
        if not output["hash_ok"]:
            output["error"] = "sha256_mismatch"
    except Exception as exc:  # per-image evidence must be retained
        output["error"] = f"{type(exc).__name__}: {exc}"
    return output


def audit_inventory(inventory: Sequence[Mapping[str, Any]], workers: int) -> list[dict[str, Any]]:
    print(f"[DATA] Auditing {len(inventory)} referenced images with {workers} workers...", flush=True)
    audited: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for index, result in enumerate(executor.map(audit_one, inventory), start=1):
            audited.append(result)
            if index % 1000 == 0 or index == len(inventory):
                print(f"[DATA] {index}/{len(inventory)} audited", flush=True)
    return audited


def overlap_summary(audited: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    valid = [row for row in audited if row["decode_ok"] and row["hash_ok"]]
    groups: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in valid:
        groups[(row["dataset"], row["split"])].append(row)

    def sha_set(dataset: str, split: str) -> set[str]:
        return {str(row["actual_sha256"]) for row in groups[(dataset, split)]}

    def fam_set(dataset: str, split: str) -> set[str]:
        return {str(row["family_id"]) for row in groups[(dataset, split)]}

    exact_checks = {
        "id_train_vs_val": len(sha_set("supported_120_breeds", "train") & sha_set("supported_120_breeds", "val")),
        "id_train_vs_test": len(sha_set("supported_120_breeds", "train") & sha_set("supported_120_breeds", "test")),
        "id_val_vs_test": len(sha_set("supported_120_breeds", "val") & sha_set("supported_120_breeds", "test")),
        "unsupported_train_vs_test": len(sha_set("unsupported_dog_breeds", "train") & sha_set("unsupported_dog_breeds", "test")),
        "ood_train_vs_test": len(sha_set("ood_oe_v2", "train") & sha_set("ood_oe_v2", "test")),
        "ood_calibration_vs_test": len(sha_set("ood_oe_v2", "calibration") & sha_set("ood_oe_v2", "test")),
        "supported_id_all_vs_ood_all": len(
            set.union(*(sha_set("supported_120_breeds", split) for split in ("train", "val", "test")))
            & set.union(*(sha_set("ood_oe_v2", split) for split in ("train", "calibration", "test", "hard_test")))
        ),
        "supported_id_all_vs_unsupported_dog_all": len(
            set.union(*(sha_set("supported_120_breeds", split) for split in ("train", "val", "test")))
            & set.union(*(sha_set("unsupported_dog_breeds", split) for split in ("train", "val", "test")))
        ),
    }
    family_checks = {
        "unsupported_train_family_vs_test": len(
            fam_set("unsupported_dog_breeds", "train") & fam_set("unsupported_dog_breeds", "test")
        ),
        "unsupported_val_family_vs_test": len(
            fam_set("unsupported_dog_breeds", "val") & fam_set("unsupported_dog_breeds", "test")
        ),
        "ood_train_family_vs_test": len(fam_set("ood_oe_v2", "train") & fam_set("ood_oe_v2", "test")),
        "ood_calibration_family_vs_test": len(
            fam_set("ood_oe_v2", "calibration") & fam_set("ood_oe_v2", "test")
        ),
    }

    near_candidates: list[dict[str, Any]] = []
    comparisons = [
        ("supported_id", groups[("supported_120_breeds", "test")], groups[("supported_120_breeds", "train")] + groups[("supported_120_breeds", "val")]),
        ("unsupported_dog", groups[("unsupported_dog_breeds", "test")], groups[("unsupported_dog_breeds", "train")] + groups[("unsupported_dog_breeds", "val")]),
        ("ood", groups[("ood_oe_v2", "test")] + groups[("ood_oe_v2", "hard_test")], groups[("ood_oe_v2", "train")] + groups[("ood_oe_v2", "calibration")]),
    ]
    for comparison, tests, references in comparisons:
        reference_hashes = [int(row["dhash"]) for row in references if row["dhash"] is not None]
        reference_rows = [row for row in references if row["dhash"] is not None]
        for test_row in tests:
            if test_row["dhash"] is None or not reference_hashes:
                continue
            distances = [(int(test_row["dhash"]) ^ value).bit_count() for value in reference_hashes]
            best_index = int(np.argmin(distances))
            best_distance = int(distances[best_index])
            if best_distance <= NEAR_DUPLICATE_MAX_DHASH:
                reference = reference_rows[best_index]
                near_candidates.append(
                    {
                        "comparison": comparison,
                        "test_path": test_row["relative_path"],
                        "test_label": test_row["label"],
                        "reference_path": reference["relative_path"],
                        "reference_label": reference["label"],
                        "reference_split": reference["split"],
                        "dhash_hamming_distance": best_distance,
                        "exact_sha256_equal": test_row["actual_sha256"] == reference["actual_sha256"],
                        "family_id_equal": test_row["family_id"] == reference["family_id"],
                        "interpretation": "candidate_only_requires_visual_review",
                    }
                )
    return {"exact_sha256_overlap_counts": exact_checks, "family_overlap_counts": family_checks}, near_candidates


def evaluation_records(audited: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for row in audited:
        group: str | None = None
        if row["dataset"] == "supported_120_breeds" and row["split"] == "test":
            group = "supported_dog"
        elif row["dataset"] == "unsupported_dog_breeds" and row["split"] == "test":
            group = "unsupported_dog_breed"
        elif row["dataset"] == "ood_oe_v2" and row["split"] == "test":
            group = "verified_non_dog_easy"
        elif row["dataset"] == "ood_oe_v2" and row["split"] == "hard_test":
            group = "cartoon_dog_snoopy" if "cartoon_dog/snoopy" in row["label"] else "verified_non_dog_garfield"
        if group:
            records.append({**row, "evaluation_group": group})
    return records


def predict_all(predictor: OEV2Predictor, records: Sequence[Mapping[str, Any]], class_to_index: Mapping[str, int]) -> list[dict[str, Any]]:
    print(f"[MODEL] Running deployed predictor on {len(records)} independent evaluation images...", flush=True)
    output: list[dict[str, Any]] = []
    for index, record in enumerate(records, start=1):
        started = time.perf_counter()
        row: dict[str, Any] = {
            "evaluation_group": record["evaluation_group"],
            "dataset": record["dataset"],
            "split": record["split"],
            "path": record["relative_path"],
            "source": record["source"],
            "source_id": record["family_id"],
            "sha256": record["actual_sha256"],
            "true_label": record["label"],
            "true_index": class_to_index.get(str(record["label"]), ""),
            "processing_error": record["error"] if not record["decode_ok"] else "",
            "status": "PROCESSING_ERROR" if not record["decode_ok"] else "",
            "predicted_index": "",
            "predicted_breed": "",
            "confidence": "",
            "probability_margin": "",
            "ood_score": "",
            "top5_indices": "",
            "top5_breeds": "",
            "top5_confidences": "",
            "top1_correct": False,
            "top5_correct": False,
            "latency_ms_bulk": "",
            "brightness": record["brightness"],
            "sharpness_proxy": record["sharpness_proxy"],
            "width": record["width"],
            "height": record["height"],
            "format": record["format"],
        }
        if record["decode_ok"] and record["hash_ok"]:
            try:
                result = predictor.predict(Path(record["path"]).read_bytes())
                top5 = result["top5"]
                row.update(
                    {
                        "status": result["status"],
                        "predicted_index": int(result["prediction"]["index"]),
                        "predicted_breed": result["breed"],
                        "confidence": float(result["confidence"]),
                        "probability_margin": float(result["probability_margin"]),
                        "ood_score": float(result["ood_score"]),
                        "top5_indices": json.dumps([int(item["index"]) for item in top5]),
                        "top5_breeds": json.dumps([item["breed"] for item in top5], ensure_ascii=False),
                        "top5_confidences": json.dumps([float(item["confidence"]) for item in top5]),
                    }
                )
                if row["true_index"] != "":
                    row["top1_correct"] = int(row["predicted_index"]) == int(row["true_index"])
                    row["top5_correct"] = int(row["true_index"]) in [int(item["index"]) for item in top5]
            except Exception as exc:
                row["status"] = "PROCESSING_ERROR"
                row["processing_error"] = f"{type(exc).__name__}: {exc}"
        row["latency_ms_bulk"] = (time.perf_counter() - started) * 1000.0
        output.append(row)
        if index % 100 == 0 or index == len(records):
            print(f"[MODEL] {index}/{len(records)} predicted", flush=True)
    return output


def state_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    counts = Counter(str(row["status"]) for row in rows)
    return {
        "sample_count": total,
        "DOG": rate_record(counts["DOG"], total),
        "UNCERTAIN": rate_record(counts["UNCERTAIN"], total),
        "NOT_DOG": rate_record(counts["NOT_DOG"], total),
        "PROCESSING_ERROR": rate_record(counts["PROCESSING_ERROR"], total),
    }


def compute_quality_metrics(
    predictions: Sequence[Mapping[str, Any]],
    class_names: Sequence[str],
    bootstrap_reps: int,
    seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], np.ndarray, list[dict[str, Any]]]:
    by_group: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in predictions:
        by_group[str(row["evaluation_group"])].append(row)

    id_rows = by_group["supported_dog"]
    y_true = np.asarray([int(row["true_index"]) for row in id_rows], dtype=np.int64)
    y_pred = np.asarray(
        [int(row["predicted_index"]) if row["predicted_index"] != "" else -1 for row in id_rows], dtype=np.int64
    )
    top5_ok = np.asarray([bool(row["top5_correct"]) for row in id_rows], dtype=bool)
    classification = classification_metrics(y_true, y_pred, top5_ok, len(class_names))
    classification["bootstrap_ci95"] = bootstrap_classification(
        y_true,
        y_pred,
        top5_ok,
        n_classes=len(class_names),
        reps=bootstrap_reps,
        seed=seed,
    )

    states = {group: state_summary(rows) for group, rows in sorted(by_group.items())}
    verified_non_dog = by_group["verified_non_dog_easy"] + by_group["verified_non_dog_garfield"]
    states["verified_non_dog_combined"] = state_summary(verified_non_dog)

    successful_id = [row for row in id_rows if row["status"] != "PROCESSING_ERROR"]
    dog_id = [row for row in id_rows if row["status"] == "DOG"]
    non_rejected_id = [row for row in id_rows if row["status"] in {"DOG", "UNCERTAIN"}]
    dog_and_correct = sum(bool(row["top1_correct"]) and row["status"] == "DOG" for row in id_rows)
    non_rejected_and_correct = sum(
        bool(row["top1_correct"]) and row["status"] in {"DOG", "UNCERTAIN"} for row in id_rows
    )
    final_quality = {
        "supported_dog_total": len(id_rows),
        "successful_inference": len(successful_id),
        "dog_acceptance": rate_record(len(dog_id), len(id_rows)),
        "dog_and_correct_overall": rate_record(dog_and_correct, len(id_rows)),
        "breed_accuracy_among_DOG": rate_record(sum(bool(row["top1_correct"]) for row in dog_id), len(dog_id)),
        "non_rejected_coverage_DOG_plus_UNCERTAIN": rate_record(len(non_rejected_id), len(id_rows)),
        "visible_breed_correct_overall_NOT_DOG_counted_incorrect": rate_record(non_rejected_and_correct, len(id_rows)),
        "breed_accuracy_among_non_rejected": rate_record(
            sum(bool(row["top1_correct"]) for row in non_rejected_id), len(non_rejected_id)
        ),
    }

    id_scores = np.asarray([float(row["ood_score"]) for row in id_rows if row["ood_score"] != ""], dtype=np.float64)
    ood_scores = np.asarray(
        [float(row["ood_score"]) for row in verified_non_dog if row["ood_score"] != ""], dtype=np.float64
    )
    ood_metrics = {
        "positive_class": "verified_non_dog",
        "score": "cosine distance; higher means more OOD/non-dog-like",
        "id_negative_count": len(id_scores),
        "ood_positive_count": len(ood_scores),
        "ood_prevalence": float(len(ood_scores) / (len(id_scores) + len(ood_scores))),
        "auroc": binary_auroc(id_scores, ood_scores),
        "auprc_ood_positive": average_precision_ood(id_scores, ood_scores),
    }
    ood_metrics["bootstrap_ci95"] = bootstrap_ood(
        id_scores, ood_scores, reps=bootstrap_reps, seed=seed + 1
    )

    arrays = classification.pop("per_class_arrays")
    per_class: list[dict[str, Any]] = []
    for index, name in enumerate(class_names):
        per_class.append(
            {
                "class_index": index,
                "breed": name,
                "support": arrays["support"][index],
                "predicted_count": arrays["predicted"][index],
                "true_positive": arrays["tp"][index],
                "precision": arrays["precision"][index],
                "recall": arrays["recall"][index],
                "f1": arrays["f1"][index],
            }
        )

    confusion = np.zeros((len(class_names), len(class_names)), dtype=np.int64)
    for truth, pred in zip(y_true, y_pred):
        if pred >= 0:
            confusion[truth, pred] += 1
    top_confusions: list[dict[str, Any]] = []
    for truth in range(len(class_names)):
        for pred in range(len(class_names)):
            if truth != pred and confusion[truth, pred] > 0:
                top_confusions.append(
                    {
                        "true_breed": class_names[truth],
                        "predicted_breed": class_names[pred],
                        "count": int(confusion[truth, pred]),
                        "true_support": int(np.sum(y_true == truth)),
                        "share_of_true_class": float(confusion[truth, pred] / max(1, np.sum(y_true == truth))),
                    }
                )
    top_confusions.sort(key=lambda row: (-row["count"], -row["share_of_true_class"], row["true_breed"]))

    metrics = {
        "classification_before_ood_rejection": classification,
        "final_user_result_quality": final_quality,
        "state_distribution": states,
        "ood_continuous_score": ood_metrics,
    }
    return metrics, per_class, confusion, top_confusions


def image_proxy_analysis(predictions: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = [row for row in predictions if row["evaluation_group"] == "supported_dog"]
    brightness = np.asarray([float(row["brightness"]) for row in rows], dtype=float)
    sharpness = np.asarray([float(row["sharpness_proxy"]) for row in rows], dtype=float)
    b_q = np.quantile(brightness, [0.1, 0.25, 0.75, 0.9])
    s_q = np.quantile(sharpness, [0.1, 0.25, 0.75, 0.9])
    definitions = [
        ("darkest_10pct", lambda row: float(row["brightness"]) <= b_q[0]),
        ("darkest_25pct", lambda row: float(row["brightness"]) <= b_q[1]),
        ("brightest_25pct", lambda row: float(row["brightness"]) >= b_q[2]),
        ("brightest_10pct", lambda row: float(row["brightness"]) >= b_q[3]),
        ("lowest_sharpness_10pct", lambda row: float(row["sharpness_proxy"]) <= s_q[0]),
        ("lowest_sharpness_25pct", lambda row: float(row["sharpness_proxy"]) <= s_q[1]),
        ("highest_sharpness_25pct", lambda row: float(row["sharpness_proxy"]) >= s_q[2]),
        ("highest_sharpness_10pct", lambda row: float(row["sharpness_proxy"]) >= s_q[3]),
    ]
    output: list[dict[str, Any]] = []
    for name, predicate in definitions:
        subset = [row for row in rows if predicate(row)]
        output.append(
            {
                "proxy_group": name,
                "sample_count": len(subset),
                "top1_accuracy": sum(bool(row["top1_correct"]) for row in subset) / len(subset),
                "DOG_rate": sum(row["status"] == "DOG" for row in subset) / len(subset),
                "UNCERTAIN_rate": sum(row["status"] == "UNCERTAIN" for row in subset) / len(subset),
                "NOT_DOG_rate": sum(row["status"] == "NOT_DOG" for row in subset) / len(subset),
                "mean_brightness": statistics.fmean(float(row["brightness"]) for row in subset),
                "mean_sharpness_proxy": statistics.fmean(float(row["sharpness_proxy"]) for row in subset),
            }
        )
    return output


def transformed_bytes(path: Path, transform: str) -> bytes:
    with Image.open(path) as source:
        image = source.convert("RGB")
        buffer = io.BytesIO()
        if transform == "webp_q80":
            image.save(buffer, format="WEBP", quality=80, method=6)
        elif transform == "jpeg_q35":
            image.save(buffer, format="JPEG", quality=35, optimize=True)
        elif transform == "resize_96_jpeg_q85":
            image.thumbnail((96, 96), Image.Resampling.LANCZOS)
            image.save(buffer, format="JPEG", quality=85, optimize=True)
        else:
            raise ValueError(transform)
    return buffer.getvalue()


def robustness_evaluation(
    predictor: OEV2Predictor,
    predictions: Sequence[Mapping[str, Any]],
    *,
    sample_count: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    source_rows = [row for row in predictions if row["evaluation_group"] == "supported_dog" and not row["processing_error"]]
    rng = random.Random(seed)
    by_label: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in source_rows:
        by_label[str(row["true_label"])].append(row)
    labels = sorted(by_label)
    chosen_labels = [labels[index] for index in np.linspace(0, len(labels) - 1, sample_count, dtype=int)]
    selected = [rng.choice(by_label[label]) for label in chosen_labels]
    output: list[dict[str, Any]] = []
    for source in selected:
        for transform in ("webp_q80", "jpeg_q35", "resize_96_jpeg_q85"):
            row = {
                "source_path": source["path"],
                "true_label": source["true_label"],
                "source_predicted_breed": source["predicted_breed"],
                "source_status": source["status"],
                "source_confidence": source["confidence"],
                "source_ood_score": source["ood_score"],
                "transform": transform,
                "predicted_breed": "",
                "status": "PROCESSING_ERROR",
                "confidence": "",
                "ood_score": "",
                "processing_error": "",
                "top1_agrees_with_source": False,
                "status_agrees_with_source": False,
                "top1_correct": False,
            }
            try:
                result = predictor.predict(transformed_bytes(PROJECT_ROOT / str(source["path"]), transform))
                row.update(
                    {
                        "predicted_breed": result["breed"],
                        "status": result["status"],
                        "confidence": result["confidence"],
                        "ood_score": result["ood_score"],
                        "top1_agrees_with_source": result["breed"] == source["predicted_breed"],
                        "status_agrees_with_source": result["status"] == source["status"],
                        "top1_correct": result["breed"] == source["true_label"],
                    }
                )
            except Exception as exc:
                row["processing_error"] = f"{type(exc).__name__}: {exc}"
            output.append(row)
    summary: dict[str, Any] = {
        "independent_source_images": len(selected),
        "selection": "one deterministic image from each of 30 evenly spaced supported breed labels",
        "note": "Derived variants are paired stability probes and are not counted as independent evaluation images.",
        "transforms": {},
    }
    for transform in ("webp_q80", "jpeg_q35", "resize_96_jpeg_q85"):
        rows = [row for row in output if row["transform"] == transform]
        summary["transforms"][transform] = {
            "sample_count": len(rows),
            "processing_errors": sum(bool(row["processing_error"]) for row in rows),
            "top1_agreement_with_source": sum(bool(row["top1_agrees_with_source"]) for row in rows) / len(rows),
            "status_agreement_with_source": sum(bool(row["status_agrees_with_source"]) for row in rows) / len(rows),
            "top1_accuracy": sum(bool(row["top1_correct"]) for row in rows) / len(rows),
            "mean_absolute_ood_score_change": statistics.fmean(
                abs(float(row["ood_score"]) - float(row["source_ood_score"]))
                for row in rows
                if row["ood_score"] != ""
            ),
        }
    return output, summary


def latency_evaluation(
    predictor: OEV2Predictor,
    predictions: Sequence[Mapping[str, Any]],
    *,
    measured_samples: int,
) -> dict[str, Any]:
    candidates = [row for row in predictions if row["evaluation_group"] == "supported_dog" and not row["processing_error"]]
    candidates = candidates[:measured_samples]
    payloads = [(PROJECT_ROOT / str(row["path"])).read_bytes() for row in candidates]
    for data in payloads[:10]:
        predictor.predict(data)
    end_to_end: list[float] = []
    model_only: list[float] = []
    for data in payloads:
        started = time.perf_counter()
        predictor.predict(data)
        end_to_end.append((time.perf_counter() - started) * 1000.0)
    preprocessed = [predictor.preprocess(data) for data in payloads]
    for array in preprocessed[:10]:
        predictor.predict(array)
    for array in preprocessed:
        started = time.perf_counter()
        predictor.predict(array)
        model_only.append((time.perf_counter() - started) * 1000.0)
    return {
        "batch_size": 1,
        "warmup_iterations": min(10, len(payloads)),
        "measured_samples": len(payloads),
        "measurement_scope": {
            "end_to_end_in_memory_bytes": "decode, resize, model forward, OOD score, decision",
            "model_and_decision_preprocessed": "model forward, OOD score, decision; excludes decode/resize",
        },
        "end_to_end_in_memory_bytes_ms": {
            "median": percentile(end_to_end, 50),
            "p95": percentile(end_to_end, 95),
            "min": min(end_to_end),
            "max": max(end_to_end),
        },
        "model_and_decision_preprocessed_ms": {
            "median": percentile(model_only, 50),
            "p95": percentile(model_only, 95),
            "min": min(model_only),
            "max": max(model_only),
        },
    }


def error_examples(predictions: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    supported = [row for row in predictions if row["evaluation_group"] == "supported_dog"]
    non_dog = [
        row
        for row in predictions
        if row["evaluation_group"] in {"verified_non_dog_easy", "verified_non_dog_garfield"}
    ]
    selected: list[dict[str, Any]] = []
    categories = [
        (
            "high_confidence_wrong_breed",
            sorted(
                [row for row in supported if not row["top1_correct"] and row["confidence"] != ""],
                key=lambda row: -float(row["confidence"]),
            )[:20],
        ),
        (
            "verified_non_dog_returned_DOG",
            sorted(
                [row for row in non_dog if row["status"] == "DOG"],
                key=lambda row: (float(row["ood_score"]), -float(row["confidence"])),
            )[:20],
        ),
        (
            "supported_dog_returned_NOT_DOG",
            sorted(
                [row for row in supported if row["status"] == "NOT_DOG"],
                key=lambda row: -float(row["ood_score"]),
            )[:20],
        ),
    ]
    for category, rows in categories:
        for row in rows:
            selected.append(
                {
                    "error_category": category,
                    "path": row["path"],
                    "true_label": row["true_label"],
                    "predicted_breed": row["predicted_breed"],
                    "status": row["status"],
                    "confidence": row["confidence"],
                    "probability_margin": row["probability_margin"],
                    "ood_score": row["ood_score"],
                    "top5_breeds": row["top5_breeds"],
                }
            )
    return selected


def save_confusion_csv(path: Path, confusion: np.ndarray, class_names: Sequence[str]) -> None:
    rows = []
    for index, name in enumerate(class_names):
        row: dict[str, Any] = {"true_breed": name}
        row.update({pred_name: int(confusion[index, pred]) for pred, pred_name in enumerate(class_names)})
        rows.append(row)
    write_csv(path, rows, ["true_breed", *class_names])


def generate_plots(
    output_dir: Path,
    predictions: Sequence[Mapping[str, Any]],
    per_class: Sequence[Mapping[str, Any]],
    confusion: np.ndarray,
    class_names: Sequence[str],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    row_totals = confusion.sum(axis=1, keepdims=True)
    normalized = np.divide(confusion, row_totals, out=np.zeros_like(confusion, dtype=float), where=row_totals > 0)
    fig, ax = plt.subplots(figsize=(18, 16))
    image = ax.imshow(normalized, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
    ax.set_title("Ma trận nhầm lẫn chuẩn hóa theo lớp thật — ID test 120 giống")
    ax.set_xlabel("Lớp dự đoán (chỉ số 0–119)")
    ax.set_ylabel("Lớp thật (chỉ số 0–119)")
    ticks = np.arange(0, len(class_names), 10)
    ax.set_xticks(ticks, ticks)
    ax.set_yticks(ticks, ticks)
    fig.colorbar(image, ax=ax, label="Tỷ lệ trong lớp thật")
    fig.tight_layout()
    fig.savefig(plot_dir / "confusion_matrix_normalized.png", dpi=180)
    plt.close(fig)

    by_group: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in predictions:
        by_group[str(row["evaluation_group"])].append(row)
    group_order = [
        "supported_dog",
        "verified_non_dog_easy",
        "verified_non_dog_garfield",
        "cartoon_dog_snoopy",
        "unsupported_dog_breed",
    ]
    labels = ["Supported dog", "Non-dog easy", "Garfield", "Snoopy", "Unsupported breeds"]
    statuses = ["DOG", "UNCERTAIN", "NOT_DOG", "PROCESSING_ERROR"]
    colors = ["#2f855a", "#d69e2e", "#c53030", "#718096"]
    bottom = np.zeros(len(group_order), dtype=float)
    fig, ax = plt.subplots(figsize=(11, 6))
    for status, color in zip(statuses, colors):
        values = [sum(row["status"] == status for row in by_group[group]) / len(by_group[group]) * 100 for group in group_order]
        ax.bar(labels, values, bottom=bottom, label=status, color=color)
        bottom += values
    ax.set_title("Tỷ lệ trạng thái theo nhóm dữ liệu — ngưỡng triển khai cố định")
    ax.set_xlabel("Nhóm dữ liệu")
    ax.set_ylabel("Tỷ lệ (%)")
    ax.set_ylim(0, 100)
    ax.legend()
    ax.tick_params(axis="x", rotation=18)
    fig.tight_layout()
    fig.savefig(plot_dir / "state_rates.png", dpi=180)
    plt.close(fig)

    id_scores = [float(row["ood_score"]) for row in by_group["supported_dog"] if row["ood_score"] != ""]
    easy_scores = [float(row["ood_score"]) for row in by_group["verified_non_dog_easy"] if row["ood_score"] != ""]
    hard_scores = [float(row["ood_score"]) for row in by_group["verified_non_dog_garfield"] if row["ood_score"] != ""]
    unsupported_scores = [float(row["ood_score"]) for row in by_group["unsupported_dog_breed"] if row["ood_score"] != ""]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(id_scores, bins=45, density=True, alpha=0.55, label=f"Supported dog (n={len(id_scores)})")
    ax.hist(easy_scores, bins=45, density=True, alpha=0.50, label=f"Non-dog easy (n={len(easy_scores)})")
    ax.hist(hard_scores, bins=30, density=True, alpha=0.55, label=f"Garfield (n={len(hard_scores)})")
    ax.hist(unsupported_scores, bins=30, density=True, alpha=0.45, label=f"Unsupported breeds (n={len(unsupported_scores)})")
    ax.axvline(LOCKED_T_UNCERTAIN, linestyle="--", color="black", label=f"T_uncertain={LOCKED_T_UNCERTAIN:.6f}")
    ax.axvline(LOCKED_T_NOT_DOG, linestyle=":", color="black", label=f"T_not_dog={LOCKED_T_NOT_DOG:.6f}")
    ax.set_title("Phân bố cosine OOD score trên các tập đánh giá độc lập")
    ax.set_xlabel("Cosine distance (cao hơn = OOD hơn)")
    ax.set_ylabel("Mật độ")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(plot_dir / "ood_score_distributions.png", dpi=180)
    plt.close(fig)

    weakest = sorted(per_class, key=lambda row: (float(row["f1"]), int(row["support"])))[:20]
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh([str(row["breed"]) for row in weakest][::-1], [float(row["f1"]) for row in weakest][::-1], color="#4a5568")
    ax.set_title("20 giống có F1 thấp nhất trên ID test")
    ax.set_xlabel("F1")
    ax.set_ylabel("Giống chó")
    ax.set_xlim(0, 1)
    fig.tight_layout()
    fig.savefig(plot_dir / "weakest_breeds_f1.png", dpi=180)
    plt.close(fig)


def generate_error_contact_sheet(output_path: Path, examples: Sequence[Mapping[str, Any]]) -> None:
    selected: list[Mapping[str, Any]] = []
    for category in ("high_confidence_wrong_breed", "verified_non_dog_returned_DOG", "supported_dog_returned_NOT_DOG"):
        selected.extend([row for row in examples if row["error_category"] == category][:4])
    if not selected:
        return
    cell_w, cell_h = 300, 270
    columns = 4
    rows = math.ceil(len(selected) / columns)
    sheet = Image.new("RGB", (columns * cell_w, rows * cell_h), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    for index, row in enumerate(selected):
        x = (index % columns) * cell_w
        y = (index // columns) * cell_h
        try:
            with Image.open(PROJECT_ROOT / str(row["path"])) as source:
                image = source.convert("RGB")
                image.thumbnail((cell_w - 16, 180), Image.Resampling.LANCZOS)
                sheet.paste(image, (x + (cell_w - image.width) // 2, y + 4))
        except Exception:
            pass
        lines = [
            str(row["error_category"])[:42],
            f"true: {str(row['true_label'])[:31]}",
            f"pred: {str(row['predicted_breed'])[:31]}",
            f"state={row['status']} conf={float(row['confidence']):.3f} ood={float(row['ood_score']):.3f}",
            Path(str(row["path"])).name[:38],
        ]
        draw.multiline_text((x + 8, y + 186), "\n".join(lines), fill="black", font=font, spacing=2)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)


def hardware_environment() -> dict[str, Any]:
    import tensorflow as tf

    return {
        "timestamp_utc": utc_now(),
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count_logical": os.cpu_count(),
        "tensorflow": tf.__version__,
        "numpy": np.__version__,
        "pillow": safe_package_version("Pillow"),
        "scikit_learn": safe_package_version("scikit-learn"),
        "matplotlib": safe_package_version("matplotlib"),
        "tensorflow_physical_devices": [str(device) for device in tf.config.list_physical_devices()],
        "tensorflow_gpus": [str(device) for device in tf.config.list_physical_devices("GPU")],
    }


def artifact_configuration(predictor: OEV2Predictor) -> dict[str, Any]:
    class_names = {int(index): name for index, name in predictor.class_names.items()}
    paths = {
        "model": predictor.model_path,
        "centroids": predictor.centroids_path,
        "class_names": predictor.class_names_path,
        "predictor_code": PROJECT_ROOT / "src" / "inference" / "oe_v2_predictor.py",
        "preprocessing_code": PROJECT_ROOT / "src" / "inference" / "preprocessing.py",
    }
    return {
        "model_version": predictor.model_version,
        "detector": LOCKED_DETECTOR,
        "score_formula": "1 - max cosine similarity(normalized embedding, 120 normalized class centroids)",
        "score_orientation": "higher = more OOD",
        "thresholds": {"T_uncertain": LOCKED_T_UNCERTAIN, "T_not_dog": LOCKED_T_NOT_DOG},
        "decision_rules": {
            "DOG": f"score < {LOCKED_T_UNCERTAIN}",
            "UNCERTAIN": f"{LOCKED_T_UNCERTAIN} <= score < {LOCKED_T_NOT_DOG}",
            "NOT_DOG": f"score >= {LOCKED_T_NOT_DOG}",
        },
        "class_count": len(class_names),
        "semantic_class_names_sha256": compute_class_names_sha256(class_names),
        "artifacts": {
            name: {
                "path": str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                "size_bytes": path.stat().st_size,
                "sha256": compute_file_sha256(path),
            }
            for name, path in paths.items()
        },
        "expected_integrity": {
            "model_sha256": EXPECTED_MODEL_SHA256,
            "centroids_sha256": EXPECTED_CENTROIDS_SHA256,
            "semantic_class_names_sha256": EXPECTED_CLASS_NAMES_SHA256,
        },
        "centroid_shape": list(predictor.centroids.shape),
        "model_input_shape": list(predictor.raw_model.input_shape),
        "model_output_shape": list(predictor.raw_model.output_shape),
        "preprocessing": {
            "accepted_formats": ["JPEG", "PNG", "WEBP"],
            "resize": "224x224 TensorFlow bilinear, antialias=False",
            "pixels": "float32 [0,255]; MobileNetV2 preprocessing embedded in model",
            "exif_orientation": "applied by production preprocessing",
        },
    }


def create_report(
    output_dir: Path,
    summary: Mapping[str, Any],
    per_class: Sequence[Mapping[str, Any]],
    top_confusions: Sequence[Mapping[str, Any]],
    errors: Sequence[Mapping[str, Any]],
    robustness: Mapping[str, Any],
    image_proxies: Sequence[Mapping[str, Any]],
) -> None:
    classification = summary["metrics"]["classification_before_ood_rejection"]
    final_quality = summary["metrics"]["final_user_result_quality"]
    state = summary["metrics"]["state_distribution"]
    ood = summary["metrics"]["ood_continuous_score"]
    weakest = sorted(per_class, key=lambda row: (float(row["f1"]), int(row["support"])))[:10]
    high_conf_wrong = [row for row in errors if row["error_category"] == "high_confidence_wrong_breed"][:10]
    false_dog = [row for row in errors if row["error_category"] == "verified_non_dog_returned_DOG"][:10]
    rejected_dog = [row for row in errors if row["error_category"] == "supported_dog_returned_NOT_DOG"][:10]

    def pct(value: float | None) -> str:
        return "—" if value is None else f"{value * 100:.2f}%"

    def ci(interval: Sequence[float] | None) -> str:
        return "—" if not interval else f"[{interval[0] * 100:.2f}%, {interval[1] * 100:.2f}%]"

    lines = [
        "# Đánh giá chất lượng model Dog Breed Classification",
        "",
        f"- Thời điểm UTC: `{summary['completed_at_utc']}`",
        "- Model: TensorFlow/Keras OE V2 thật; không mock, không huấn luyện lại, không đổi trọng số/centroid/ngưỡng.",
        f"- Ngưỡng cố định: `T_uncertain={LOCKED_T_UNCERTAIN}`, `T_not_dog={LOCKED_T_NOT_DOG}`.",
        f"- Bootstrap: `{summary['configuration']['bootstrap_reps']}` lần, seed `{summary['configuration']['seed']}`, lấy mẫu theo ảnh nguồn độc lập.",
        "",
        "## Phạm vi dữ liệu",
        "",
        "- Chó thuộc 120 giống hỗ trợ: 1.017 ảnh test, đủ 120 lớp, 6–13 ảnh/lớp.",
        "- Không phải chó: 619 ảnh OOD test và 34 ảnh Garfield; hai nhóm này mới được dùng làm lớp dương OOD.",
        "- Cartoon dog: 35 ảnh Snoopy, báo cáo riêng; không coi là non-dog chuẩn.",
        "- Giống chó ngoài danh sách: 120 ảnh Oxford-IIIT Pet, gồm American Bulldog, American Pit Bull Terrier, Havanese và Shiba Inu; không chấm nhãn giống theo 120 lớp và không tự coi NOT_DOG là đúng.",
        "- `data/test` của Kaggle không có nhãn nên không dùng để tính chất lượng.",
        "",
        "## Chất lượng phân loại giống trước OOD",
        "",
        f"- Top-1: **{pct(classification['top1']['rate'])}** ({classification['top1']['count']}/{classification['top1']['denominator']}), Wilson 95% {ci(classification['top1']['ci95_wilson'])}; bootstrap 95% {ci(classification['bootstrap_ci95']['top1'])}.",
        f"- Top-5: **{pct(classification['top5']['rate'])}** ({classification['top5']['count']}/{classification['top5']['denominator']}), Wilson 95% {ci(classification['top5']['ci95_wilson'])}; bootstrap 95% {ci(classification['bootstrap_ci95']['top5'])}.",
        f"- Macro precision/recall/F1: **{classification['macro_precision']:.4f} / {classification['macro_recall']:.4f} / {classification['macro_f1']:.4f}**.",
        f"- Macro F1 bootstrap 95%: `{classification['bootstrap_ci95']['macro_f1'][0]:.4f}–{classification['bootstrap_ci95']['macro_f1'][1]:.4f}`.",
        f"- Lỗi xử lý: {classification['processing_error_count']}/{classification['sample_count']}; mọi lỗi đều nằm trong mẫu số.",
        "",
        "### 10 giống yếu nhất theo F1",
        "",
        "| Giống | Support | Precision | Recall | F1 |",
        "|---|---:|---:|---:|---:|",
        *[
            f"| {row['breed']} | {row['support']} | {float(row['precision']):.3f} | {float(row['recall']):.3f} | {float(row['f1']):.3f} |"
            for row in weakest
        ],
        "",
        "### Các cặp nhầm nhiều nhất",
        "",
        "| Nhãn thật | Dự đoán | Số ảnh | Tỷ lệ trong lớp thật |",
        "|---|---|---:|---:|",
        *[
            f"| {row['true_breed']} | {row['predicted_breed']} | {row['count']} | {pct(float(row['share_of_true_class']))} |"
            for row in top_confusions[:15]
        ],
        "",
        "## Kết quả cuối cùng người dùng nhận",
        "",
        f"- Tỷ lệ ảnh chó được trả `DOG`: **{pct(final_quality['dog_acceptance']['rate'])}** ({final_quality['dog_acceptance']['count']}/{final_quality['dog_acceptance']['denominator']}).",
        f"- `DOG` đồng thời đúng giống: **{pct(final_quality['dog_and_correct_overall']['rate'])}** ({final_quality['dog_and_correct_overall']['count']}/{final_quality['dog_and_correct_overall']['denominator']}).",
        f"- Accuracy giống trên riêng nhóm `DOG`: **{pct(final_quality['breed_accuracy_among_DOG']['rate'])}** ({final_quality['breed_accuracy_among_DOG']['count']}/{final_quality['breed_accuracy_among_DOG']['denominator']}).",
        f"- Nếu tính `DOG+UNCERTAIN` là còn hiển thị gợi ý giống, độ phủ là {pct(final_quality['non_rejected_coverage_DOG_plus_UNCERTAIN']['rate'])}; accuracy trên nhóm này là {pct(final_quality['breed_accuracy_among_non_rejected']['rate'])}.",
        f"- Accuracy giống cuối cùng trên toàn bộ 1.017 ảnh, coi `NOT_DOG` là sai: **{pct(final_quality['visible_breed_correct_overall_NOT_DOG_counted_incorrect']['rate'])}**.",
        "",
        "## DOG / UNCERTAIN / NOT_DOG",
        "",
        "| Nhóm | N | DOG | UNCERTAIN | NOT_DOG | Lỗi |",
        "|---|---:|---:|---:|---:|---:|",
        *[
            f"| {group} | {state[group]['sample_count']} | {pct(state[group]['DOG']['rate'])} | {pct(state[group]['UNCERTAIN']['rate'])} | {pct(state[group]['NOT_DOG']['rate'])} | {pct(state[group]['PROCESSING_ERROR']['rate'])} |"
            for group in [
                "supported_dog",
                "verified_non_dog_easy",
                "verified_non_dog_garfield",
                "verified_non_dog_combined",
                "cartoon_dog_snoopy",
                "unsupported_dog_breed",
            ]
        ],
        "",
        f"- Non-dog bị trả `DOG`: **{pct(state['verified_non_dog_combined']['DOG']['rate'])}** ({state['verified_non_dog_combined']['DOG']['count']}/{state['verified_non_dog_combined']['DOG']['denominator']}), Wilson 95% {ci(state['verified_non_dog_combined']['DOG']['ci95_wilson'])}.",
        f"- Chó hỗ trợ bị trả `NOT_DOG`: **{pct(state['supported_dog']['NOT_DOG']['rate'])}** ({state['supported_dog']['NOT_DOG']['count']}/{state['supported_dog']['NOT_DOG']['denominator']}), Wilson 95% {ci(state['supported_dog']['NOT_DOG']['ci95_wilson'])}.",
        f"- OOD AUROC: **{ood['auroc']:.4f}**, bootstrap 95% `{ood['bootstrap_ci95']['auroc'][0]:.4f}–{ood['bootstrap_ci95']['auroc'][1]:.4f}`.",
        f"- OOD AUPRC: **{ood['auprc_ood_positive']:.4f}**, bootstrap 95% `{ood['bootstrap_ci95']['auprc_ood_positive'][0]:.4f}–{ood['bootstrap_ci95']['auprc_ood_positive'][1]:.4f}`.",
        f"- Lớp dương là non-dog, điểm cao hơn nghĩa là OOD hơn; tỷ lệ dương {ood['ood_prevalence']:.4f} ({ood['ood_positive_count']}/{ood['ood_positive_count'] + ood['id_negative_count']}).",
        "",
        "## Phân tích lỗi thực tế",
        "",
        f"- Sai giống độ tự tin cao lưu: {len(high_conf_wrong)} ví dụ đầu trong `error_examples.csv`.",
        f"- Non-dog bị nhận `DOG`: {len([row for row in errors if row['error_category'] == 'verified_non_dog_returned_DOG'])} ví dụ được lưu.",
        f"- Chó hỗ trợ bị từ chối `NOT_DOG`: {len([row for row in errors if row['error_category'] == 'supported_dog_returned_NOT_DOG'])} ví dụ được lưu.",
        "- Brightness và sharpness chỉ là proxy đo từ pixel. Không có bounding box/nhãn cảnh đáng tin cậy để kết luận về chó nhỏ, nhiều con vật, che khuất hay nền phức tạp.",
        "",
        "### Proxy ảnh",
        "",
        "| Nhóm proxy | N | Top-1 | DOG | UNCERTAIN | NOT_DOG |",
        "|---|---:|---:|---:|---:|---:|",
        *[
            f"| {row['proxy_group']} | {row['sample_count']} | {pct(row['top1_accuracy'])} | {pct(row['DOG_rate'])} | {pct(row['UNCERTAIN_rate'])} | {pct(row['NOT_DOG_rate'])} |"
            for row in image_proxies
        ],
        "",
        "## Độ ổn định định dạng/nén",
        "",
        f"- {robustness['independent_source_images']} ảnh nguồn độc lập; các biến thể không được cộng vào kích thước test.",
        "| Biến thể | N | Top-1 giữ nguyên | Trạng thái giữ nguyên | Accuracy | |Δ OOD| trung bình |",
        "|---|---:|---:|---:|---:|---:|",
        *[
            f"| {name} | {item['sample_count']} | {pct(item['top1_agreement_with_source'])} | {pct(item['status_agreement_with_source'])} | {pct(item['top1_accuracy'])} | {item['mean_absolute_ood_score_change']:.4f} |"
            for name, item in robustness["transforms"].items()
        ],
        "",
        "## Độ trễ",
        "",
        f"- Batch size 1, warm-up {summary['latency']['warmup_iterations']}, đo {summary['latency']['measured_samples']} ảnh trên thiết bị TensorFlow liệt kê trong `summary.json`.",
        f"- End-to-end từ bytes trong RAM: median **{summary['latency']['end_to_end_in_memory_bytes_ms']['median']:.2f} ms**, p95 **{summary['latency']['end_to_end_in_memory_bytes_ms']['p95']:.2f} ms**.",
        f"- Model + quyết định trên tensor đã tiền xử lý: median **{summary['latency']['model_and_decision_preprocessed_ms']['median']:.2f} ms**, p95 **{summary['latency']['model_and_decision_preprocessed_ms']['p95']:.2f} ms**.",
        "",
        "## Tính độc lập và giới hạn",
        "",
        f"- Exact-overlap checks: `{json.dumps(summary['data_audit']['overlap']['exact_sha256_overlap_counts'], ensure_ascii=False)}`.",
        f"- Family-overlap checks: `{json.dumps(summary['data_audit']['overlap']['family_overlap_counts'], ensure_ascii=False)}`.",
        f"- dHash≤{NEAR_DUPLICATE_MAX_DHASH} sinh {summary['data_audit']['near_duplicate_candidate_count']} ứng viên; đây chỉ là tín hiệu cần review, không tự động coi là trùng.",
        "- Tập ID chỉ có 6–13 ảnh/lớp nên chỉ số từng giống có phương sai lớn. Nguồn ID chủ yếu là Kaggle, OOD easy chủ yếu là Caltech-101; chưa đại diện đầy đủ ảnh điện thoại ngoài đời.",
        "- Lịch sử audit cho biết split test không dùng để train/chọn model/hiệu chỉnh ngưỡng, nhưng không thể chứng minh tuyệt đối nguồn ảnh công khai chưa từng xuất hiện trong pretraining ImageNet.",
        "- Không có tiêu chí nghiệm thu được đặt trước; báo cáo không tự gán nhãn đạt/không đạt.",
        "",
        "## Ba ưu tiên cải tiến dựa trên lỗi đo được",
        "",
        f"1. Bổ sung và rà soát dữ liệu cho các giống F1 thấp nhất, ưu tiên `{', '.join(str(row['breed']) for row in weakest[:5])}`; giữ validation/test mới tách theo nguồn ảnh.",
        f"2. Bổ sung hard-negative đúng nhóm đã lọt `DOG`, ưu tiên các nhãn nguồn trong {', '.join(sorted({str(row['true_label']) for row in false_dog[:10]})) or 'không có false-DOG trong mẫu hiện tại'}; hiệu chỉnh chỉ trên validation OOD mới, không dùng test này.",
        "3. Tăng dữ liệu/augmentation cho điều kiện suy giảm thực tế được phản ánh bởi stability và proxy pixel; thu thập nhãn bounding-box/occlusion thật trước khi kết luận nguyên nhân cảnh.",
        "",
        "## Tệp bàn giao",
        "",
        "- `predictions.csv`: dự đoán từng ảnh, gồm lỗi xử lý.",
        "- `summary.json`: chỉ số, CI, cấu hình, hash, môi trường.",
        "- `per_class_metrics.csv`, `confusion_matrix.csv`, `top_confusions.csv`.",
        "- `error_examples.csv`, `robustness.csv`, `image_proxy_analysis.csv`, `near_duplicate_candidates.csv`.",
        "- `plots/`: ma trận nhầm lẫn, trạng thái, OOD score, giống yếu và contact sheet lỗi.",
        "- `run_command.txt`: lệnh tái chạy.",
    ]
    (output_dir / "REPORT_VI.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-reps", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=BOOTSTRAP_SEED)
    parser.add_argument("--audit-workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--latency-samples", type=int, default=100)
    parser.add_argument("--robustness-samples", type=int, default=30)
    args = parser.parse_args()

    output_dir = args.output_dir if args.output_dir.is_absolute() else PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    started_at = utc_now()
    command = (
        f'"{sys.executable}" src/evaluate_deployed_quality.py '
        f'--output-dir "{output_dir.relative_to(PROJECT_ROOT)}" '
        f"--bootstrap-reps {args.bootstrap_reps} --seed {args.seed} "
        f"--audit-workers {args.audit_workers} --latency-samples {args.latency_samples} "
        f"--robustness-samples {args.robustness_samples}"
    )
    (output_dir / "run_command.txt").write_text(command + "\n", encoding="utf-8")

    print("[START] Loading deployed TensorFlow predictor with integrity checks...", flush=True)
    predictor = OEV2Predictor(verify_integrity=True)
    class_names = [predictor.class_names[index] for index in range(len(predictor.class_names))]
    class_to_index = {name: index for index, name in enumerate(class_names)}
    artifacts_before = artifact_configuration(predictor)
    environment = hardware_environment()

    inventory, csv_sources = build_inventory()
    audited = audit_inventory(inventory, args.audit_workers)
    inventory_fields = [
        "dataset", "split", "relative_path", "label", "source", "family_id", "expected_sha256",
        "actual_sha256", "exists", "decode_ok", "hash_ok", "width", "height", "format", "brightness",
        "sharpness_proxy", "dhash", "error",
    ]
    write_csv(output_dir / "data_inventory.csv", audited, inventory_fields)
    overlap, near_duplicates = overlap_summary(audited)
    write_csv(output_dir / "near_duplicate_candidates.csv", near_duplicates)

    eval_records = evaluation_records(audited)
    predictions = predict_all(predictor, eval_records, class_to_index)
    prediction_fields = [
        "evaluation_group", "dataset", "split", "path", "source", "source_id", "sha256", "true_label",
        "true_index", "status", "predicted_index", "predicted_breed", "confidence", "probability_margin",
        "ood_score", "top5_indices", "top5_breeds", "top5_confidences", "top1_correct", "top5_correct",
        "processing_error", "latency_ms_bulk", "brightness", "sharpness_proxy", "width", "height", "format",
    ]
    write_csv(output_dir / "predictions.csv", predictions, prediction_fields)

    metrics, per_class, confusion, top_confusions = compute_quality_metrics(
        predictions, class_names, args.bootstrap_reps, args.seed
    )
    write_csv(output_dir / "per_class_metrics.csv", per_class)
    save_confusion_csv(output_dir / "confusion_matrix.csv", confusion, class_names)
    write_csv(output_dir / "top_confusions.csv", top_confusions)

    errors = error_examples(predictions)
    write_csv(output_dir / "error_examples.csv", errors)
    image_proxies = image_proxy_analysis(predictions)
    write_csv(output_dir / "image_proxy_analysis.csv", image_proxies)
    robustness_rows, robustness_summary = robustness_evaluation(
        predictor, predictions, sample_count=args.robustness_samples, seed=args.seed
    )
    write_csv(output_dir / "robustness.csv", robustness_rows)
    latency = latency_evaluation(predictor, predictions, measured_samples=args.latency_samples)

    generate_plots(output_dir, predictions, per_class, confusion, class_names)
    generate_error_contact_sheet(output_dir / "plots" / "error_examples_contact_sheet.png", errors)

    data_counts: dict[str, Any] = {}
    for dataset in sorted({str(row["dataset"]) for row in audited}):
        data_counts[dataset] = {}
        for split in sorted({str(row["split"]) for row in audited if row["dataset"] == dataset}):
            rows = [row for row in audited if row["dataset"] == dataset and row["split"] == split]
            label_counts = Counter(str(row["label"]) for row in rows)
            data_counts[dataset][split] = {
                "sample_count": len(rows),
                "label_count": len(label_counts),
                "min_samples_per_label": min(label_counts.values()),
                "max_samples_per_label": max(label_counts.values()),
                "label_distribution": dict(sorted(label_counts.items())),
                "missing_files": sum(not row["exists"] for row in rows),
                "decode_errors": sum(not row["decode_ok"] for row in rows),
                "sha256_mismatches": sum(row["exists"] and not row["hash_ok"] for row in rows),
            }

    artifacts_after = artifact_configuration(predictor)
    immutable_ok = {
        name: artifacts_before["artifacts"][name]["sha256"] == artifacts_after["artifacts"][name]["sha256"]
        for name in artifacts_before["artifacts"]
        if name in {"model", "centroids", "class_names"}
    }
    summary = {
        "schema_version": 1,
        "started_at_utc": started_at,
        "completed_at_utc": utc_now(),
        "configuration": {
            "seed": args.seed,
            "bootstrap_reps": args.bootstrap_reps,
            "bootstrap_unit": "independent source image; stratified by ID/OOD for AUROC/AUPRC",
            "near_duplicate_dhash_max_distance": NEAR_DUPLICATE_MAX_DHASH,
            "latency_samples": args.latency_samples,
            "robustness_source_samples": args.robustness_samples,
        },
        "environment": environment,
        "artifacts": artifacts_before,
        "artifact_hashes_unchanged_after_evaluation": immutable_ok,
        "data_sources": csv_sources,
        "data_audit": {
            "inventory": data_counts,
            "total_referenced_images": len(audited),
            "total_missing": sum(not row["exists"] for row in audited),
            "total_decode_errors": sum(not row["decode_ok"] for row in audited),
            "total_sha256_mismatches": sum(row["exists"] and not row["hash_ok"] for row in audited),
            "overlap": overlap,
            "near_duplicate_candidate_count": len(near_duplicates),
            "near_duplicate_interpretation": "dHash candidates only; exact SHA and recorded family overlaps are authoritative checks",
            "unlabeled_kaggle_test_file_count": sum(
                1 for path in (PROJECT_ROOT / "data" / "test").iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
            ),
        },
        "evaluation_group_counts": dict(Counter(str(row["evaluation_group"]) for row in predictions)),
        "metrics": metrics,
        "robustness": robustness_summary,
        "latency": latency,
        "limitations": [
            "Only 6-13 independent ID test images per supported breed.",
            "ID source is primarily Kaggle; easy non-dog source is primarily Caltech-101.",
            "No reliable annotations for dog size in frame, multiple animals, occlusion, or background complexity.",
            "Public-source overlap with ImageNet pretraining cannot be ruled out.",
            "Unsupported breeds are reported descriptively; NOT_DOG is not treated as correct for them.",
            "No predeclared product acceptance thresholds were supplied.",
        ],
    }
    write_json(output_dir / "summary.json", summary)
    create_report(output_dir, summary, per_class, top_confusions, errors, robustness_summary, image_proxies)
    print(f"[DONE] Results written to {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
