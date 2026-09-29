"""Predictive-entropy OOD scoring and calibration evaluation.

Predictive entropy is computed from the classifier's softmax probabilities::

    H(p) = -sum_i p_i log(p_i)

The scorer is deliberately independent of the MSP implementation.  The
module also provides a small evaluation CLI for the project's ID validation
split and ``data/ood/calibration`` image pool.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "outputs" / "deployment_ready" / "dog_breed_model.keras"
DEFAULT_ID_SPLIT = "val"
DEFAULT_OOD_ROOT = PROJECT_ROOT / "data" / "ood" / "calibration"
DEFAULT_METRICS_PATH = PROJECT_ROOT / "outputs" / "metrics" / "entropy_calibration.json"
DEFAULT_PLOT_PATH = PROJECT_ROOT / "outputs" / "plots" / "entropy_distribution.png"
IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}


def _probability_batch(probabilities: np.ndarray) -> tuple[np.ndarray, bool]:
    """Validate probabilities and return a 2-D float64 batch.

    A separate validator is kept here so entropy remains a standalone method;
    in particular, it does not change or depend on the MSP implementation.
    """

    values = np.asarray(probabilities, dtype=np.float64)
    one_dimensional = values.ndim == 1
    if one_dimensional:
        values = values[None, :]
    if values.ndim != 2 or values.shape[1] == 0:
        raise ValueError("probabilities must have shape (batch, classes)")
    if not np.isfinite(values).all():
        raise ValueError("probabilities contain non-finite values")
    if np.any(values < 0.0):
        raise ValueError("probabilities must be non-negative")
    sums = values.sum(axis=1)
    if not np.allclose(sums, 1.0, atol=1e-4, rtol=0.0):
        raise ValueError("each probability row must sum to 1")
    return values, one_dimensional


def predictive_entropy(
    probabilities: np.ndarray, *, normalized: bool = False
) -> np.ndarray | float:
    """Return predictive entropy ``-sum(p * log(p))`` for each sample.

    Zero probabilities are clipped only for the logarithm, using the smallest
    positive float.  Their contribution remains zero because ``0 * log(eps)``
    is zero.  With ``normalized=True`` the score is divided by
    ``log(number_of_classes)`` and is therefore in ``[0, 1]`` for a valid
    probability distribution.  A one-class distribution has zero entropy and
    consequently returns zero when normalized as well.
    """

    values, one_dimensional = _probability_batch(probabilities)
    safe = np.clip(values, np.finfo(np.float64).tiny, 1.0)
    entropy = -np.sum(values * np.log(safe), axis=1, dtype=np.float64)

    if normalized:
        number_of_classes = values.shape[1]
        if number_of_classes == 1:
            entropy = np.zeros_like(entropy)
        else:
            entropy = entropy / np.log(float(number_of_classes))
            # Guard only against round-off at the mathematical boundaries.
            entropy = np.clip(entropy, 0.0, 1.0)

    if one_dimensional:
        return float(entropy[0])
    return entropy


def entropy_ood_score(probabilities: np.ndarray) -> np.ndarray | float:
    """Return normalized predictive entropy; larger values indicate OOD."""

    return predictive_entropy(probabilities, normalized=True)


score = entropy_ood_score
ood_score = entropy_ood_score
entropy_score = entropy_ood_score


def _score_summary(scores: np.ndarray) -> dict[str, float | int]:
    """Return JSON-safe descriptive statistics for a non-empty score vector."""

    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    if values.size == 0:
        return {"count": 0}
    if not np.isfinite(values).all():
        raise ValueError("entropy scores must be finite")
    percentiles = np.percentile(values, [5, 25, 50, 75, 95])
    return {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "p05": float(percentiles[0]),
        "p25": float(percentiles[1]),
        "median": float(percentiles[2]),
        "p75": float(percentiles[3]),
        "p95": float(percentiles[4]),
        "max": float(np.max(values)),
    }


def _discover_images(root: Path) -> list[Path]:
    """Discover supported image files below an OOD calibration directory."""

    if not root.is_dir():
        return []
    return sorted(
        (
            path
            for path in root.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ),
        key=lambda path: path.as_posix().lower(),
    )


def _predict_batches(model: Any, dataset: Iterable[Any]) -> np.ndarray:
    """Run inference-only prediction over a TensorFlow dataset."""

    batches: list[np.ndarray] = []
    for batch in dataset:
        images = batch[0] if isinstance(batch, (tuple, list)) else batch
        probabilities = np.asarray(model(images, training=False).numpy(), dtype=np.float64)
        if probabilities.ndim != 2:
            raise ValueError(f"model predictions must be 2-D, got {probabilities.shape}")
        # Validate the complete model output, including finiteness and sums.
        _probability_batch(probabilities)
        batches.append(probabilities)
    if not batches:
        return np.empty((0, 0), dtype=np.float64)
    return np.concatenate(batches, axis=0)


def _load_model_and_predictions(
    model_path: Path,
    id_split: str,
    ood_images: list[Path],
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Load the production model and collect ID/OOD probabilities."""

    if not model_path.is_file():
        raise FileNotFoundError(f"Model not found: {model_path}")

    # Imports stay local so importing the pure entropy scorer does not require
    # TensorFlow or a model artifact.
    import tensorflow as tf

    from src.dataset_pipeline import create_dataset, load_split_csv
    from src.label_encoder import fit_label_encoder

    tf.keras.utils.get_custom_objects()["preprocess_input"] = (
        tf.keras.applications.mobilenet_v2.preprocess_input
    )
    model = tf.keras.models.load_model(str(model_path), compile=False)

    id_df = load_split_csv(id_split)
    encoder = fit_label_encoder()
    id_dataset = create_dataset(id_df, encoder, batch_size=batch_size, training=False)
    id_probabilities = _predict_batches(model, id_dataset)
    if len(id_probabilities) != len(id_df):
        raise ValueError(
            f"ID prediction count ({len(id_probabilities)}) does not match "
            f"{id_split} split count ({len(id_df)})"
        )

    if not ood_images:
        return id_probabilities, np.empty((0, id_probabilities.shape[1]), dtype=np.float64)

    def preprocess_ood_image(path: Any) -> Any:
        """Decode common image formats and match the model input contract."""

        image = tf.io.read_file(path)
        image = tf.image.decode_image(image, channels=3, expand_animations=False)
        image = tf.image.resize(image, (224, 224))
        image = tf.cast(image, tf.float32)
        image.set_shape((224, 224, 3))
        return image

    ood_paths = tf.data.Dataset.from_tensor_slices([str(path) for path in ood_images])
    ood_dataset = (
        ood_paths.map(preprocess_ood_image, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size, drop_remainder=False)
        .prefetch(tf.data.AUTOTUNE)
    )
    ood_probabilities = _predict_batches(model, ood_dataset)
    return id_probabilities, ood_probabilities


def plot_entropy_distribution(
    id_scores: np.ndarray,
    ood_scores: np.ndarray,
    output_path: Path | str = DEFAULT_PLOT_PATH,
) -> None:
    """Save overlaid ID/OOD normalized-entropy distributions."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    id_values = np.asarray(id_scores, dtype=np.float64).reshape(-1)
    ood_values = np.asarray(ood_scores, dtype=np.float64).reshape(-1)
    if id_values.size == 0:
        raise ValueError("at least one ID entropy score is required for the plot")

    figure, axis = plt.subplots(figsize=(9, 5.5))
    bins = np.linspace(0.0, 1.0, 31)
    axis.hist(
        id_values,
        bins=bins,
        density=True,
        alpha=0.62,
        label="ID validation",
        color="#2563eb",
    )
    if ood_values.size:
        axis.hist(
            ood_values,
            bins=bins,
            density=True,
            alpha=0.62,
            label="OOD calibration",
            color="#dc2626",
        )
    else:
        axis.text(
            0.98,
            0.95,
            "OOD calibration: no images found",
            transform=axis.transAxes,
            ha="right",
            va="top",
            fontsize=10,
            color="#991b1b",
            bbox={
                "boxstyle": "round,pad=0.35",
                "facecolor": "#fee2e2",
                "edgecolor": "#fca5a5",
            },
        )
    axis.set(
        title="Predictive Entropy Distribution",
        xlabel="Normalized entropy H(p) / log(C)",
        ylabel="Density",
        xlim=(0.0, 1.0),
    )
    axis.grid(axis="y", alpha=0.22)
    axis.legend()
    figure.tight_layout()
    figure.savefig(destination, dpi=180)
    plt.close(figure)


def evaluate_entropy_calibration(
    *,
    model_path: Path | str = DEFAULT_MODEL_PATH,
    id_split: str = DEFAULT_ID_SPLIT,
    ood_root: Path | str = DEFAULT_OOD_ROOT,
    batch_size: int = 64,
    metrics_path: Path | str = DEFAULT_METRICS_PATH,
    plot_path: Path | str = DEFAULT_PLOT_PATH,
    target_id_acceptance: float = 0.95,
) -> dict[str, Any]:
    """Evaluate entropy on ID validation and OOD calibration images.

    An empty OOD calibration directory is reported explicitly.  In that case
    OOD metrics and a threshold are ``null`` because they cannot be estimated
    honestly without OOD examples; the ID evaluation and plot are still saved.
    """

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if not 0 < target_id_acceptance <= 1:
        raise ValueError("target_id_acceptance must be in (0, 1]")

    resolved_model = Path(model_path)
    if not resolved_model.is_absolute():
        resolved_model = PROJECT_ROOT / resolved_model
    resolved_model = resolved_model.resolve()
    resolved_ood_root = Path(ood_root)
    if not resolved_ood_root.is_absolute():
        resolved_ood_root = PROJECT_ROOT / resolved_ood_root
    resolved_ood_root = resolved_ood_root.resolve()
    ood_images = _discover_images(resolved_ood_root)

    id_probabilities, ood_probabilities = _load_model_and_predictions(
        resolved_model, id_split, ood_images, batch_size
    )
    id_scores = np.asarray(entropy_ood_score(id_probabilities), dtype=np.float64).reshape(-1)
    ood_scores = np.asarray(entropy_ood_score(ood_probabilities), dtype=np.float64).reshape(-1)
    plot_entropy_distribution(id_scores, ood_scores, plot_path)

    metrics: dict[str, Any] = {
        "task": "predictive_entropy_ood",
        "method": "predictive_entropy",
        "formula": "H(p) = -sum(p * log(p))",
        "score_definition": "normalized predictive entropy H(p) / log(number_of_classes)",
        "score_orientation": "higher values indicate more likely OOD",
        "numerical_stability": "zero probabilities use the smallest positive float for log",
        "model_path": str(resolved_model),
        "id_split": id_split,
        "ood_calibration_root": str(resolved_ood_root),
        "number_of_classes": int(id_probabilities.shape[1]),
        "id_validation": _score_summary(id_scores),
        "ood_calibration": _score_summary(ood_scores),
        "artifacts": {
            "metrics_json": str(Path(metrics_path)),
            "entropy_distribution_plot": str(Path(plot_path)),
        },
    }

    if ood_scores.size:
        from src.ood.calibration import calibrate_threshold
        from src.ood.evaluate_ood import evaluate_ood_scores, rejection_error_rates

        threshold = calibrate_threshold(
            id_scores,
            ood_scores,
            target_id_acceptance=target_id_acceptance,
        )
        metrics["ood_evaluation"] = evaluate_ood_scores(
            id_scores, ood_scores, detector="entropy"
        )
        metrics["threshold_calibration"] = threshold
        metrics["threshold_errors"] = rejection_error_rates(
            id_scores,
            ood_scores,
            threshold["ood_threshold"],
            higher_score_is_ood=True,
        )
        metrics["ood_status"] = "evaluated"
    else:
        metrics["ood_evaluation"] = None
        metrics["threshold_calibration"] = None
        metrics["threshold_errors"] = None
        metrics["ood_status"] = "not_evaluated"
        metrics["ood_status_reason"] = (
            "No supported image files were found in the OOD calibration directory; "
            "provide data/ood/calibration images to compute OOD metrics."
        )

    metrics_output = Path(metrics_path)
    if not metrics_output.is_absolute():
        metrics_output = PROJECT_ROOT / metrics_output
    metrics_output.parent.mkdir(parents=True, exist_ok=True)
    metrics_output.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return metrics


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--id-split", choices=("val", "test"), default=DEFAULT_ID_SPLIT)
    parser.add_argument("--ood-root", type=Path, default=DEFAULT_OOD_ROOT)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--metrics-path", type=Path, default=DEFAULT_METRICS_PATH)
    parser.add_argument("--plot-path", type=Path, default=DEFAULT_PLOT_PATH)
    parser.add_argument("--target-id-acceptance", type=float, default=0.95)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = evaluate_entropy_calibration(
        model_path=args.model_path,
        id_split=args.id_split,
        ood_root=args.ood_root,
        batch_size=args.batch_size,
        metrics_path=args.metrics_path,
        plot_path=args.plot_path,
        target_id_acceptance=args.target_id_acceptance,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
