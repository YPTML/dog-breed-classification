"""Run the locked TASK 24 evaluation.

The final ID test split is evaluated once with the production model.  OOD
metrics use the already-persisted MSP policy and never select a threshold from
the final test or OOD evaluation images.  Missing OOD fixtures are reported as
unavailable rather than converted to zero-valued metrics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import tensorflow as tf
from sklearn.metrics import f1_score

from src.data_loader import PROJECT_ROOT, TRAIN_DIR
from src.dataset_pipeline import create_dataset, load_split_csv
from src.label_encoder import fit_label_encoder
from src.ood.evaluate_ood import evaluate_ood_scores


DEFAULT_MODEL_PATH = PROJECT_ROOT / "outputs" / "deployment_ready" / "dog_breed_model.keras"
DEFAULT_THRESHOLD_PATH = PROJECT_ROOT / "outputs" / "deployment_ready" / "ood_thresholds.json"
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "outputs" / "deployment_ready" / "ood_config.json"
DEFAULT_OOD_ROOT = PROJECT_ROOT / "data" / "ood" / "test"
DEFAULT_HARD_ROOT = PROJECT_ROOT / "data" / "ood_hard"
DEFAULT_OUTPUT_JSON = PROJECT_ROOT / "outputs" / "metrics" / "final_evaluation.json"
DEFAULT_OUTPUT_REPORT = PROJECT_ROOT / "outputs" / "reports" / "FINAL_EVALUATION.md"

HARD_OOD_CATEGORIES = (
    "wolf",
    "fox",
    "coyote",
    "hyena",
    "dog_toy",
    "dog_statue",
    "dog_cartoon",
    "dog_drawing",
)
IMAGE_EXTENSIONS = frozenset({".bmp", ".jpeg", ".jpg", ".png", ".webp"})


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def load_frozen_threshold(path: str | Path = DEFAULT_THRESHOLD_PATH) -> dict[str, Any]:
    """Load the fixed runtime threshold without fitting or changing it."""

    source = Path(path).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Frozen threshold artifact not found: {source}")
    with source.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("Frozen threshold artifact must be a JSON object")

    method = str(payload.get("method", "")).strip().lower()
    score_definition = str(payload.get("score_definition", "")).strip()
    if method != "msp":
        raise ValueError(f"TASK 24 expects the frozen MSP policy, found {method!r}")
    if score_definition != "1_minus_max_softmax_probability":
        raise ValueError("Frozen threshold must be defined for 1 - max softmax probability")

    threshold = _finite(payload.get("ood_threshold"), "ood_threshold")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("ood_threshold must be in [0, 1]")
    result: dict[str, Any] = {
        "method": method,
        "score_definition": score_definition,
        "ood_threshold": threshold,
        "uncertain_threshold": (
            None
            if payload.get("uncertain_threshold") is None
            else _finite(payload.get("uncertain_threshold"), "uncertain_threshold")
        ),
        "breed_confidence_threshold": (
            None
            if payload.get("breed_confidence_threshold") is None
            else _finite(payload.get("breed_confidence_threshold"), "breed_confidence_threshold")
        ),
        "calibrated": payload.get("calibrated"),
        "calibration_date": payload.get("calibration_date"),
        "source": str(source),
        "sha256": _sha256(source),
    }
    return result


def _discover_images(root: str | Path) -> list[Path]:
    directory = Path(root)
    if not directory.is_dir():
        return []
    return sorted(
        (
            path
            for path in directory.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ),
        key=lambda path: path.relative_to(directory).as_posix().lower(),
    )


def _discover_hard_images(root: str | Path) -> dict[str, list[Path]]:
    directory = Path(root)
    return {
        category: _discover_images(directory / category)
        for category in HARD_OOD_CATEGORIES
    }


def _image_dataset(paths: Sequence[Path], batch_size: int = 64) -> tf.data.Dataset:
    def decode(path: tf.Tensor) -> tf.Tensor:
        image = tf.io.decode_image(
            tf.io.read_file(path), channels=3, expand_animations=False
        )
        image = tf.image.resize(image, (224, 224))
        image = tf.cast(image, tf.float32)
        return tf.ensure_shape(image, (224, 224, 3))

    options = tf.data.Options()
    options.deterministic = True
    return (
        tf.data.Dataset.from_tensor_slices([str(path) for path in paths])
        .map(decode, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size, drop_remainder=False)
        .with_options(options)
        .prefetch(tf.data.AUTOTUNE)
    )


def _predict_path_list(
    model: tf.keras.Model, paths: Sequence[Path], batch_size: int = 64
) -> np.ndarray:
    if not paths:
        return np.empty((0, 0), dtype=np.float64)
    batches: list[np.ndarray] = []
    for images in _image_dataset(paths, batch_size=batch_size):
        values = model(images, training=False)
        batches.append(np.asarray(values, dtype=np.float64))
    predictions = np.concatenate(batches, axis=0)
    if not np.isfinite(predictions).all():
        raise ValueError("Model predictions contain NaN or Inf")
    return predictions


def _predict_final_test(
    model: tf.keras.Model, *, batch_size: int
) -> tuple[Any, np.ndarray, np.ndarray]:
    test_frame = load_split_csv("test")
    encoder = fit_label_encoder()
    test_dataset = create_dataset(
        test_frame, encoder, batch_size=batch_size, training=False
    )
    options = tf.data.Options()
    options.deterministic = True
    test_dataset = test_dataset.with_options(options)

    probability_batches: list[np.ndarray] = []
    for images, _ in test_dataset:
        probability_batches.append(
            np.asarray(model(images, training=False), dtype=np.float64)
        )
    probabilities = np.concatenate(probability_batches, axis=0)
    y_true = np.argmax(encoder.transform(test_frame["breed"]), axis=1)
    if probabilities.shape != (len(test_frame), len(encoder.classes_)):
        raise ValueError(
            f"Unexpected final-test output shape {probabilities.shape}; "
            f"expected {(len(test_frame), len(encoder.classes_))}"
        )
    return encoder, np.asarray(y_true, dtype=np.int64), probabilities


def breed_metrics(y_true: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    """Compute the requested breed metrics without any decision threshold."""

    y_pred = np.argmax(probabilities, axis=1)
    top_k = min(5, probabilities.shape[1])
    top_indices = np.argsort(-probabilities, axis=1, kind="mergesort")[:, :top_k]
    top1 = float(np.mean(y_pred == y_true))
    top5 = float(np.mean(np.any(top_indices == y_true[:, None], axis=1)))
    macro_f1 = float(
        f1_score(
            y_true,
            y_pred,
            labels=np.arange(probabilities.shape[1]),
            average="macro",
            zero_division=0,
        )
    )
    return {
        "status": "evaluated",
        "sample_count": int(y_true.size),
        "class_count": int(probabilities.shape[1]),
        "top_1": top1,
        "top_5": top5,
        "top_1_accuracy": top1,
        "top_5_accuracy": top5,
        "macro_f1": macro_f1,
        "correct_top_1": int(np.sum(y_pred == y_true)),
    }


def _unavailable_ood(reason: str, *, id_count: int, ood_count: int) -> dict[str, Any]:
    return {
        "status": "unavailable",
        "status_reason": reason,
        "id_count": int(id_count),
        "ood_count": int(ood_count),
        "metrics": {
            "auroc": None,
            "fpr95": None,
            "aupr": None,
            "aupr_in": None,
            "aupr_out": None,
            "ood_f1": None,
            "false_acceptance_rate": None,
            "false_rejection_rate": None,
        },
    }


def evaluate_regular_ood(
    id_probabilities: np.ndarray,
    ood_probabilities: np.ndarray,
    *,
    threshold: float,
    source: str | Path,
) -> dict[str, Any]:
    """Evaluate regular OOD with a fixed OOD-oriented MSP score."""

    id_scores = 1.0 - np.max(id_probabilities, axis=1)
    ood_scores = 1.0 - np.max(ood_probabilities, axis=1)
    evaluated = evaluate_ood_scores(
        id_scores,
        ood_scores,
        higher_score_is_ood=True,
        threshold=threshold,
    )
    id_count = int(id_scores.size)
    ood_count = int(ood_scores.size)
    false_acceptance_rate = float(evaluated["false_negative"] / ood_count)
    false_rejection_rate = float(evaluated["false_positive"] / id_count)
    return {
        "status": "evaluated",
        "status_reason": None,
        "source": str(Path(source).resolve()),
        "id_count": id_count,
        "ood_count": ood_count,
        "score_definition": "1 - max(softmax_probability)",
        "higher_score_is_ood": True,
        "fixed_threshold": float(threshold),
        "threshold_source": "deployment_ready/ood_thresholds.json",
        "metrics": {
            "auroc": float(evaluated["auroc"]),
            "fpr95": float(evaluated["fpr95"]),
            "aupr": float(evaluated["aupr_out"]),
            "aupr_in": float(evaluated["aupr_in"]),
            "aupr_out": float(evaluated["aupr_out"]),
            "ood_f1": float(evaluated["ood_f1"]),
            "false_acceptance_rate": false_acceptance_rate,
            "false_rejection_rate": false_rejection_rate,
        },
        "confusion": {
            "true_positive": int(evaluated["true_positive"]),
            "true_negative": int(evaluated["true_negative"]),
            "false_positive": int(evaluated["false_positive"]),
            "false_negative": int(evaluated["false_negative"]),
        },
    }


def hard_category_metrics(
    model: tf.keras.Model,
    hard_root: str | Path,
    *,
    threshold: float,
    batch_size: int,
) -> dict[str, Any]:
    images = _discover_hard_images(hard_root)
    total = sum(len(paths) for paths in images.values())
    categories: dict[str, Any] = {}
    for category, paths in images.items():
        if not paths:
            categories[category] = {
                "status": "unavailable",
                "sample_count": 0,
                "false_acceptance_count": None,
                "false_acceptance_rate": None,
                "mean_ood_score": None,
            }
            continue
        probabilities = _predict_path_list(model, paths, batch_size=batch_size)
        scores = 1.0 - np.max(probabilities, axis=1)
        accepted = scores < threshold
        categories[category] = {
            "status": "evaluated",
            "sample_count": int(scores.size),
            "false_acceptance_count": int(np.sum(accepted)),
            "false_acceptance_rate": float(np.mean(accepted)),
            "mean_ood_score": float(np.mean(scores)),
        }
    return {
        "status": "evaluated" if total else "unavailable",
        "status_reason": None
        if total
        else "No Hard OOD images were present in the required category directories.",
        "source": str(Path(hard_root).resolve()),
        "fixed_threshold": float(threshold),
        "threshold_source": "deployment_ready/ood_thresholds.json",
        "categories": categories,
        "total_sample_count": int(total),
    }


def _load_model(path: Path) -> tf.keras.Model:
    if not path.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {path}")
    tf.keras.utils.get_custom_objects()["preprocess_input"] = (
        tf.keras.applications.mobilenet_v2.preprocess_input
    )
    return tf.keras.models.load_model(str(path), compile=False, safe_mode=True)


def run_final_evaluation(
    *,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    threshold_path: str | Path = DEFAULT_THRESHOLD_PATH,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    ood_root: str | Path = DEFAULT_OOD_ROOT,
    hard_root: str | Path = DEFAULT_HARD_ROOT,
    output_json: str | Path = DEFAULT_OUTPUT_JSON,
    output_report: str | Path = DEFAULT_OUTPUT_REPORT,
    batch_size: int = 64,
) -> dict[str, Any]:
    """Run and persist the final locked evaluation."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    model_file = Path(model_path).resolve()
    threshold = load_frozen_threshold(threshold_path)
    config_file = Path(config_path).resolve()
    config_sha256 = _sha256(config_file) if config_file.is_file() else None
    model = _load_model(model_file)

    encoder, y_true, id_probabilities = _predict_final_test(model, batch_size=batch_size)
    breed = breed_metrics(y_true, id_probabilities)

    ood_paths = _discover_images(ood_root)
    if ood_paths:
        ood_probabilities = _predict_path_list(model, ood_paths, batch_size=batch_size)
        ood = evaluate_regular_ood(
            id_probabilities,
            ood_probabilities,
            threshold=float(threshold["ood_threshold"]),
            source=ood_root,
        )
    else:
        ood = _unavailable_ood(
            "No regular OOD images were present in the final evaluation pool.",
            id_count=id_probabilities.shape[0],
            ood_count=0,
        )
        ood["source"] = str(Path(ood_root).resolve())
        ood["score_definition"] = "1 - max(softmax_probability)"
        ood["higher_score_is_ood"] = True
        ood["fixed_threshold"] = float(threshold["ood_threshold"])
        ood["threshold_source"] = "deployment_ready/ood_thresholds.json"

    hard = hard_category_metrics(
        model,
        hard_root,
        threshold=float(threshold["ood_threshold"]),
        batch_size=batch_size,
    )

    complete = ood["status"] == "evaluated" and hard["status"] == "evaluated"
    result: dict[str, Any] = {
        "schema_version": 1,
        "task": "TASK 24 — FINAL LOCKED EVALUATION",
        "status": "evaluated" if complete else "partial_missing_ood_data",
        "evaluation_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "final_test_source": str((PROJECT_ROOT / "outputs" / "splits" / "test.csv").resolve()),
            "final_test_split": "internal held-out ID test",
            "final_test_used_for_threshold_selection": False,
            "threshold_frozen_before_evaluation": True,
            "threshold_tuning_performed": False,
            "model_weights_modified": False,
            "kaggle_test_used": False,
            "batch_size": int(batch_size),
            "preprocessing": "existing dataset pipeline; raw pixels [0,255], model-owned MobileNetV2 preprocessing",
        },
        "model": {
            "path": str(model_file),
            "sha256": _sha256(model_file),
            "name": str(model.name),
            "class_count": int(len(encoder.classes_)),
        },
        "frozen_threshold": threshold,
        "deployment_config": {
            "path": str(config_file),
            "sha256": config_sha256,
        },
        "breed": breed,
        "ood": {
            "method": "msp",
            **ood,
        },
        "hard_ood": {
            "method": "msp",
            **hard,
        },
    }

    json_path = Path(output_json)
    report_path = Path(output_report)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    report_path.write_text(render_report(result), encoding="utf-8")
    return result


def _fmt(value: Any) -> str:
    return "—" if value is None else f"{float(value):.6f}"


def render_report(result: Mapping[str, Any]) -> str:
    breed = result["breed"]
    ood = result["ood"]
    hard = result["hard_ood"]
    threshold = result["frozen_threshold"]
    lines = [
        "# TASK 24 — Final Locked Evaluation",
        "",
        f"- **Status:** `{result['status']}`",
        f"- **Evaluation timestamp (UTC):** `{result['evaluation_timestamp_utc']}`",
        "- **Protocol:** final internal ID test set evaluated after threshold freeze.",
        "- **Threshold tuning:** **not performed**; no threshold or model weight was changed from the evaluation result.",
        f"- **Model:** `{result['model']['path']}`",
        f"- **Threshold source:** `{threshold['source']}`",
        f"- **Frozen MSP OOD threshold:** `{threshold['ood_threshold']:.6f}` for `1 - max(softmax)`",
        "",
        "## Breed",
        "",
        f"Final test samples: **{breed['sample_count']}**; classes: **{breed['class_count']}**.",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Top-1 | {breed['top_1']:.6f} |",
        f"| Top-5 | {breed['top_5']:.6f} |",
        f"| Macro F1 | {breed['macro_f1']:.6f} |",
        "",
        "## OOD",
        "",
        f"Source: `{ood.get('source', DEFAULT_OOD_ROOT)}`. Metrics are evaluated at the frozen threshold; AUROC/FPR95/AUPR are threshold-independent where applicable.",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| AUROC | {_fmt(ood['metrics']['auroc'])} |",
        f"| FPR95 | {_fmt(ood['metrics']['fpr95'])} |",
        f"| AUPR (OOD / AUPR-OUT) | {_fmt(ood['metrics']['aupr'])} |",
        f"| OOD F1 | {_fmt(ood['metrics']['ood_f1'])} |",
        f"| False Acceptance Rate | {_fmt(ood['metrics']['false_acceptance_rate'])} |",
        f"| False Rejection Rate | {_fmt(ood['metrics']['false_rejection_rate'])} |",
        "",
    ]
    if ood["status"] != "evaluated":
        lines.extend(
            [
                f"OOD metrics are unavailable: {ood['status_reason']} They are intentionally reported as `null`/`—`, not as zero.",
                "",
            ]
        )

    lines.extend(
        [
            "## Hard OOD — category-level results",
            "",
            f"Source: `{hard['source']}`; fixed threshold: `{hard['fixed_threshold']:.6f}`.",
            "",
            "| Category | Samples | False Acceptance | False Acceptance Rate | Mean OOD Score |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for category in HARD_OOD_CATEGORIES:
        row = hard["categories"][category]
        false_acceptance = row["false_acceptance_count"]
        lines.append(
            f"| {category} | {row['sample_count']} | "
            f"{('—' if false_acceptance is None else false_acceptance)} | "
            f"{_fmt(row['false_acceptance_rate'])} | {_fmt(row['mean_ood_score'])} |"
        )
    lines.extend(["", "## Lock and integrity checks", ""])
    lines.extend(
        [
            "- Final test set was used for reporting only; it was not used to select a threshold.",
            "- Threshold artifact and model were loaded read-only; the report records their SHA-256 fingerprints.",
            "- No training, fine-tuning, calibration, or threshold search was run during this evaluation.",
        ]
    )
    if hard["status"] != "evaluated":
        lines.extend(["", f"Hard OOD results are unavailable: {hard['status_reason']}"])
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--threshold-path", type=Path, default=DEFAULT_THRESHOLD_PATH)
    parser.add_argument("--config-path", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--ood-root", type=Path, default=DEFAULT_OOD_ROOT)
    parser.add_argument("--hard-root", type=Path, default=DEFAULT_HARD_ROOT)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-report", type=Path, default=DEFAULT_OUTPUT_REPORT)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    result = run_final_evaluation(
        model_path=args.model_path,
        threshold_path=args.threshold_path,
        config_path=args.config_path,
        ood_root=args.ood_root,
        hard_root=args.hard_root,
        output_json=args.output_json,
        output_report=args.output_report,
        batch_size=args.batch_size,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
