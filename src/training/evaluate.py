"""Evaluate the current breed classifier on the held-out ID test set.

This module is intentionally evaluation-only.  It loads an existing Keras
model, builds the existing deterministic ``test`` split, computes
classification metrics, and writes the baseline artifacts requested for OOD
work.  No training or weight updates are performed here.

Run from the project root with::

    python -m src.training.evaluate

The model can be overridden without changing the source::

    python -m src.training.evaluate --model-path path/to/model.keras
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

from src.data_loader import PROJECT_ROOT
from src.dataset_pipeline import create_dataset, load_split_csv
from src.label_encoder import fit_label_encoder


BATCH_SIZE = 64
DEFAULT_MODEL_PATH = PROJECT_ROOT / "outputs" / "deployment_ready" / "dog_breed_model.keras"
DEFAULT_METRICS_PATH = PROJECT_ROOT / "outputs" / "metrics" / "breed_baseline.json"
DEFAULT_REPORT_PATH = PROJECT_ROOT / "outputs" / "reports" / "BREED_BASELINE_REPORT.md"
DEFAULT_PLOT_PATH = PROJECT_ROOT / "outputs" / "plots" / "confusion_matrix.png"

# The model stores MobileNetV2 preprocessing internally.  Registering the
# function keeps loading compatible with older .keras artifacts as well.
tf.keras.utils.get_custom_objects()["preprocess_input"] = (
    tf.keras.applications.mobilenet_v2.preprocess_input
)


def _resolve_model_path(model_path: Path | str) -> Path:
    """Resolve a model path, treating project-relative paths as convenient CLI input."""

    path = Path(model_path)
    if not path.is_absolute() and not path.exists():
        path = PROJECT_ROOT / path
    return path.resolve()


def _project_relative(path: Path) -> str:
    """Return a stable project-relative path when possible."""

    try:
        return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _model_input_size(model: tf.keras.Model) -> List[int]:
    """Read the spatial input size from a single image-input Keras model."""

    input_shape = model.input_shape
    if isinstance(input_shape, list):
        if len(input_shape) != 1:
            raise ValueError(f"Expected one model input, got {input_shape}")
        input_shape = input_shape[0]
    if input_shape is None or len(input_shape) != 4:
        raise ValueError(f"Expected input shape (batch, height, width, channels), got {input_shape}")
    height, width = input_shape[1:3]
    if height is None or width is None:
        raise ValueError(f"Model input spatial dimensions must be concrete, got {input_shape}")
    return [int(height), int(width)]


def _model_input_shape(model: tf.keras.Model) -> List[int]:
    """Read the non-batch input shape, including the channel count."""

    input_shape = model.input_shape
    if isinstance(input_shape, list):
        if len(input_shape) != 1:
            raise ValueError(f"Expected one model input, got {input_shape}")
        input_shape = input_shape[0]
    if input_shape is None or len(input_shape) != 4 or any(value is None for value in input_shape[1:]):
        raise ValueError(f"Expected concrete input shape (batch, height, width, channels), got {input_shape}")
    return [int(value) for value in input_shape[1:]]


def _model_output_classes(model: tf.keras.Model) -> int:
    """Read the classifier output width from a single-output Keras model."""

    output_shape = model.output_shape
    if isinstance(output_shape, list):
        if len(output_shape) != 1:
            raise ValueError(f"Expected one model output, got {output_shape}")
        output_shape = output_shape[0]
    if output_shape is None or len(output_shape) != 2 or output_shape[1] is None:
        raise ValueError(f"Expected output shape (batch, classes), got {output_shape}")
    return int(output_shape[1])


def collect_predictions(
    model: tf.keras.Model,
    test_ds: tf.data.Dataset,
) -> Tuple[np.ndarray, np.ndarray]:
    """Collect integer labels and probability predictions without updating weights."""

    y_true_batches: List[np.ndarray] = []
    prediction_batches: List[np.ndarray] = []

    for images, labels in test_ds:
        # Calling the model directly makes the evaluation-only behavior clear
        # and avoids relying on whether the saved model retained a compiled
        # metric configuration.
        probabilities = model(images, training=False).numpy()
        y_true_batches.append(np.argmax(labels.numpy(), axis=1))
        prediction_batches.append(probabilities)

    if not y_true_batches:
        raise ValueError("The ID test set is empty.")

    y_true = np.concatenate(y_true_batches, axis=0).astype(np.int64, copy=False)
    probabilities = np.concatenate(prediction_batches, axis=0).astype(np.float64, copy=False)
    if probabilities.ndim != 2:
        raise ValueError(f"Expected 2-D class probabilities, got shape {probabilities.shape}")
    if not np.isfinite(probabilities).all():
        raise ValueError("Model predictions contain NaN or infinite values.")
    return y_true, probabilities


def compute_metrics(
    y_true: Sequence[int] | np.ndarray,
    probabilities: np.ndarray,
    class_names: Sequence[str],
) -> Dict[str, Any]:
    """Compute all baseline metrics required by the evaluation task.

    ``class_names`` defines the complete class order, so classes with no
    predictions still appear in the per-class metrics and confusion matrix.
    """

    y_true_array = np.asarray(y_true, dtype=np.int64)
    probabilities_array = np.asarray(probabilities, dtype=np.float64)
    names = [str(name) for name in class_names]
    number_of_classes = len(names)

    if probabilities_array.ndim != 2 or probabilities_array.shape[1] != number_of_classes:
        raise ValueError(
            "Probability shape does not match class names: "
            f"{probabilities_array.shape} vs {number_of_classes} classes"
        )
    if len(y_true_array) != len(probabilities_array):
        raise ValueError("The number of labels and predictions must match.")
    if len(y_true_array) == 0:
        raise ValueError("Cannot compute metrics for an empty test set.")

    y_pred = np.argmax(probabilities_array, axis=1).astype(np.int64, copy=False)
    labels = np.arange(number_of_classes, dtype=np.int64)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true_array,
        y_pred,
        labels=labels,
        zero_division=0,
    )
    cm = confusion_matrix(y_true_array, y_pred, labels=labels)

    top_k = min(5, number_of_classes)
    top_indices = np.argsort(-probabilities_array, axis=1, kind="stable")[:, :top_k]
    top_5 = float(np.mean(np.any(top_indices == y_true_array[:, None], axis=1)))

    per_class_precision = {
        names[index]: float(precision[index]) for index in range(number_of_classes)
    }
    per_class_recall = {
        names[index]: float(recall[index]) for index in range(number_of_classes)
    }
    per_class_f1 = {
        names[index]: float(f1[index]) for index in range(number_of_classes)
    }
    per_class_support = {
        names[index]: int(support[index]) for index in range(number_of_classes)
    }

    return {
        "top_1_accuracy": float(accuracy_score(y_true_array, y_pred)),
        "top_5_accuracy": top_5,
        "macro_precision": float(np.mean(precision)),
        "macro_recall": float(np.mean(recall)),
        "macro_f1": float(np.mean(f1)),
        "weighted_f1": float(
            np.average(f1, weights=support) if int(np.sum(support)) else 0.0
        ),
        "per_class_precision": per_class_precision,
        "per_class_recall": per_class_recall,
        "per_class_f1": per_class_f1,
        "per_class_support": per_class_support,
        "confusion_matrix": cm.astype(int).tolist(),
        "class_names": names,
    }


def save_confusion_matrix_plot(
    matrix: Sequence[Sequence[int]],
    class_names: Sequence[str],
    output_path: Path,
) -> None:
    """Save a readable raw confusion matrix image for the full class set."""

    cm = np.asarray(matrix, dtype=np.int64)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure_size = max(14.0, min(32.0, 0.22 * len(class_names) + 8.0))
    fig, ax = plt.subplots(figsize=(figure_size, figure_size))
    image = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set(
        title="Breed Classifier Baseline — ID Test Confusion Matrix",
        xlabel="Predicted class",
        ylabel="True class",
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.setp(ax.get_xticklabels(), rotation=90, ha="center", fontsize=5)
    plt.setp(ax.get_yticklabels(), fontsize=5)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _write_report(
    output_path: Path,
    model_path: Path,
    input_size: Sequence[int],
    number_of_classes: int,
    test_sample_count: int,
    metrics: Mapping[str, Any],
    plot_path: Path,
) -> None:
    """Write the human-readable Markdown baseline report."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plot_reference_text = os.path.relpath(
        plot_path.resolve(), output_path.parent.resolve()
    ).replace("\\", "/")

    rows = []
    for index, class_name in enumerate(metrics["class_names"]):
        rows.append(
            "| {index} | {name} | {precision:.6f} | {recall:.6f} | {f1:.6f} | {support} |".format(
                index=index,
                name=class_name,
                precision=metrics["per_class_precision"][class_name],
                recall=metrics["per_class_recall"][class_name],
                f1=metrics["per_class_f1"][class_name],
                support=metrics["per_class_support"][class_name],
            )
        )

    content = "\n".join(
        [
            "# Breed Classifier Baseline Report",
            "",
            "Evaluation of the current breed classifier on the held-out in-distribution (ID) test set.",
            "No training, fine-tuning, or weight updates were performed.",
            "",
            "## Evaluation setup",
            "",
            f"- **Model path:** `{model_path.resolve()}`",
            f"- **Model path (project-relative):** `{_project_relative(model_path)}`",
            f"- **Input size:** `{int(input_size[0])} x {int(input_size[1])}`",
            f"- **Number of classes:** `{number_of_classes}`",
            f"- **Test sample count:** `{test_sample_count}`",
            "- **Dataset:** ID test split (`outputs/splits/test.csv`)",
            "- **Evaluation mode:** inference-only (`training=False`)",
            "",
            "## Aggregate metrics",
            "",
            "| Metric | Value |",
            "|---|---:|",
            f"| Top-1 Accuracy | {metrics['top_1_accuracy']:.6f} |",
            f"| Top-5 Accuracy | {metrics['top_5_accuracy']:.6f} |",
            f"| Macro Precision | {metrics['macro_precision']:.6f} |",
            f"| Macro Recall | {metrics['macro_recall']:.6f} |",
            f"| Macro F1 | {metrics['macro_f1']:.6f} |",
            f"| Weighted F1 | {metrics['weighted_f1']:.6f} |",
            "",
            "## Confusion matrix",
            "",
            f"![Confusion matrix]({plot_reference_text})",
            "",
            "The complete raw confusion matrix is stored in `breed_baseline.json` under `confusion_matrix`.",
            "",
            "## Per-class metrics",
            "",
            "| Index | Class | Precision | Recall | F1 | Support |",
            "|---:|---|---:|---:|---:|---:|",
            *rows,
            "",
        ]
    )
    output_path.write_text(content, encoding="utf-8")


def evaluate_baseline(
    model_path: Path | str = DEFAULT_MODEL_PATH,
    batch_size: int = BATCH_SIZE,
    metrics_path: Path | str = DEFAULT_METRICS_PATH,
    report_path: Path | str = DEFAULT_REPORT_PATH,
    plot_path: Path | str = DEFAULT_PLOT_PATH,
) -> Dict[str, Any]:
    """Run baseline evaluation and write the three requested artifacts."""

    resolved_model_path = _resolve_model_path(model_path)
    if not resolved_model_path.is_file():
        raise FileNotFoundError(f"Model not found: {resolved_model_path}")
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")

    metrics_output = Path(metrics_path)
    report_output = Path(report_path)
    plot_output = Path(plot_path)
    if not metrics_output.is_absolute():
        metrics_output = PROJECT_ROOT / metrics_output
    if not report_output.is_absolute():
        report_output = PROJECT_ROOT / report_output
    if not plot_output.is_absolute():
        plot_output = PROJECT_ROOT / plot_output

    print(f"[INFO] Loading model: {resolved_model_path}")
    model = tf.keras.models.load_model(str(resolved_model_path), compile=False)
    input_size = _model_input_size(model)
    input_shape = _model_input_shape(model)
    model_classes = _model_output_classes(model)

    print("[INFO] Building deterministic ID test pipeline...")
    test_df = load_split_csv("test")
    encoder = fit_label_encoder()
    class_names = [str(name) for name in encoder.classes_]
    if model_classes != len(class_names):
        raise ValueError(
            f"Model output classes ({model_classes}) do not match dataset classes ({len(class_names)})."
        )
    test_ds = create_dataset(test_df, encoder, batch_size=batch_size, training=False)

    print(f"[INFO] Evaluating {len(test_df)} test samples...")
    y_true, probabilities = collect_predictions(model, test_ds)
    if len(y_true) != len(test_df):
        raise ValueError(
            f"Prediction count ({len(y_true)}) does not match test sample count ({len(test_df)})."
        )

    metrics = compute_metrics(y_true, probabilities, class_names)
    save_confusion_matrix_plot(metrics["confusion_matrix"], class_names, plot_output)

    metrics_output.parent.mkdir(parents=True, exist_ok=True)
    report_output.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {
        "task": "breed_classifier_baseline",
        "model_path": str(resolved_model_path),
        "model_path_relative": _project_relative(resolved_model_path),
        "input_size": input_size,
        "input_shape": input_shape,
        "number_of_classes": len(class_names),
        "test_sample_count": int(len(test_df)),
        "metrics": metrics,
        # Keep the required metrics directly addressable at the top level as
        # well as grouped under ``metrics`` for convenient downstream use.
        **metrics,
        "artifacts": {
            "metrics_json": _project_relative(metrics_output),
            "report_markdown": _project_relative(report_output),
            "confusion_matrix_plot": _project_relative(plot_output),
        },
    }
    metrics_output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_report(
        report_output,
        resolved_model_path,
        input_size,
        len(class_names),
        len(test_df),
        metrics,
        plot_output,
    )

    print(f"[RESULT] Top-1 Accuracy: {metrics['top_1_accuracy']:.4f}")
    print(f"[RESULT] Top-5 Accuracy: {metrics['top_5_accuracy']:.4f}")
    print(f"[RESULT] Macro F1: {metrics['macro_f1']:.4f}")
    print(f"[INFO] Metrics: {metrics_output}")
    print(f"[INFO] Report: {report_output}")
    print(f"[INFO] Confusion matrix: {plot_output}")
    return payload


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the current breed classifier on the ID test set.")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help=f"Path to an existing .keras model (default: {DEFAULT_MODEL_PATH})",
    )
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--metrics-path", type=Path, default=DEFAULT_METRICS_PATH)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--plot-path", type=Path, default=DEFAULT_PLOT_PATH)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    evaluate_baseline(
        model_path=args.model_path,
        batch_size=args.batch_size,
        metrics_path=args.metrics_path,
        report_path=args.report_path,
        plot_path=args.plot_path,
    )
