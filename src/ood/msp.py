"""Maximum Softmax Probability (MSP) OOD baseline.

The task-level MSP score is the maximum predicted class probability::

    MSP(x) = max_c p(c | x)

ID samples are expected to have a high score and OOD samples a low score.
Consequently :func:`predict_ood` marks a sample as OOD when its MSP is below
the supplied threshold.  The threshold is deliberately an argument; this
module never embeds a production threshold.

The command-line entry point evaluates the existing model on the ID
validation split and the local OOD calibration directory, then writes the
requested JSON statistics and distribution plot. OOD data is optional at
runtime because this repository may be checked out without the externally
provided OOD image pool; in that case the artifact records that calibration
could not be completed instead of inventing a threshold.
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path
from typing import Any, Iterable

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "outputs" / "deployment_ready" / "dog_breed_model.keras"
DEFAULT_ID_SPLIT = PROJECT_ROOT / "outputs" / "splits" / "val.csv"
DEFAULT_OOD_ROOT = PROJECT_ROOT / "data" / "ood" / "calibration"
DEFAULT_METRICS_PATH = PROJECT_ROOT / "outputs" / "metrics" / "msp_calibration.json"
DEFAULT_PLOT_PATH = PROJECT_ROOT / "outputs" / "plots" / "msp_distribution.png"
DEFAULT_BATCH_SIZE = 64


def _probability_batch(probabilities: np.ndarray) -> np.ndarray:
    """Validate probability rows and return a two-dimensional float array."""

    values = np.asarray(probabilities, dtype=np.float64)
    if values.ndim == 1:
        values = values[None, :]
    if values.ndim != 2 or values.shape[1] == 0:
        raise ValueError("probabilities must have shape (batch, classes)")
    if not np.isfinite(values).all():
        raise ValueError("probabilities contain non-finite values")
    if np.any(values < 0):
        raise ValueError("probabilities must be non-negative")
    if not np.allclose(values.sum(axis=1), 1.0, atol=1e-4):
        raise ValueError("each probability row must sum to 1")
    return values


def score(probabilities: np.ndarray) -> np.ndarray | float:
    """Return MSP, ``max(softmax_probability)``, per sample.

    A one-dimensional class-probability vector returns a Python ``float``;
    a two-dimensional batch returns a one-dimensional NumPy array.
    """

    values = _probability_batch(probabilities)
    scores = values.max(axis=1)
    return float(scores[0]) if np.asarray(probabilities).ndim == 1 else scores


def predict_ood(score: np.ndarray | float, threshold: float) -> np.ndarray | bool:
    """Return whether each MSP score should be rejected as OOD.

    MSP is ID-oriented, so a sample is OOD when ``score < threshold``. The
    threshold is required explicitly and is not selected or hard-coded here.
    """

    values = np.asarray(score, dtype=np.float64)
    threshold_value = float(threshold)
    if not np.isfinite(values).all():
        raise ValueError("score must contain only finite values")
    if not np.isfinite(threshold_value):
        raise ValueError("threshold must be finite")
    prediction = values < threshold_value
    return bool(prediction) if values.ndim == 0 else prediction


def maximum_softmax_probability(probabilities: np.ndarray) -> np.ndarray | float:
    """Backward-compatible descriptive alias for :func:`score`."""

    return score(probabilities)


def msp_ood_score(probabilities: np.ndarray) -> np.ndarray | float:
    """Return the legacy OOD-oriented transform ``1 - MSP``.

    Existing benchmark code in this repository uses the convention that
    larger values mean more likely OOD. The task-facing :func:`score` remains
    the direct MSP value.
    """

    return 1.0 - score(probabilities)


ood_score = msp_ood_score
msp_score = score


def _finite_scores(values: Iterable[float] | np.ndarray, name: str) -> np.ndarray:
    """Normalize a score collection and reject empty/non-finite inputs."""

    result = np.asarray(values, dtype=np.float64).reshape(-1)
    if result.size == 0 or not np.isfinite(result).all():
        raise ValueError(f"{name} must contain at least one finite score")
    return result


def distribution_statistics(values: Iterable[float] | np.ndarray) -> dict[str, Any]:
    """Return JSON-safe descriptive statistics for an MSP distribution.

    Empty input is represented explicitly with ``available=false`` and null
    statistics so an unavailable OOD calibration set is not mistaken for a
    measured distribution.
    """

    result = np.asarray(values, dtype=np.float64).reshape(-1)
    if result.size == 0:
        return {
            "available": False,
            "count": 0,
            "min": None,
            "max": None,
            "mean": None,
            "std": None,
            "median": None,
            "percentiles": {f"p{p:02d}": None for p in (1, 5, 25, 50, 75, 95, 99)},
        }
    if not np.isfinite(result).all():
        raise ValueError("distribution values must be finite")
    percentiles = np.percentile(result, [1, 5, 25, 50, 75, 95, 99])
    return {
        "available": True,
        "count": int(result.size),
        "min": float(result.min()),
        "max": float(result.max()),
        "mean": float(result.mean()),
        "std": float(result.std()),
        "median": float(np.median(result)),
        "percentiles": {
            f"p{p:02d}": float(value)
            for p, value in zip((1, 5, 25, 50, 75, 95, 99), percentiles)
        },
    }


def calibrate_msp_threshold(
    id_scores: Iterable[float] | np.ndarray,
    ood_scores: Iterable[float] | np.ndarray,
    *,
    target_id_acceptance: float = 0.95,
) -> dict[str, Any]:
    """Select an MSP threshold from disjoint ID/OOD calibration scores.

    The chosen threshold retains at least ``target_id_acceptance`` of ID
    samples (``MSP >= threshold``) and maximizes OOD rejection on the supplied
    calibration set. No fixed numeric threshold is used.
    """

    ids = _finite_scores(id_scores, "id_scores")
    oods = _finite_scores(ood_scores, "ood_scores")
    target = float(target_id_acceptance)
    if not 0.0 < target <= 1.0:
        raise ValueError("target_id_acceptance must be in (0, 1]")

    id_quantile = float(np.quantile(ids, 1.0 - target, method="linear"))
    candidates = np.unique(np.concatenate(([id_quantile], ids, oods)))
    eligible: list[tuple[tuple[float, float], float, float, float]] = []
    for threshold in candidates:
        id_acceptance = float(np.mean(ids >= threshold))
        if id_acceptance + 1e-12 < target:
            continue
        ood_recall = float(np.mean(oods < threshold))
        # Prefer OOD recall, then the highest admissible threshold.
        eligible.append(((ood_recall, threshold), float(threshold), id_acceptance, ood_recall))

    if eligible:
        _, threshold, id_acceptance, ood_recall = max(eligible, key=lambda item: item[0])
    else:
        threshold = id_quantile
        id_acceptance = float(np.mean(ids >= threshold))
        ood_recall = float(np.mean(oods < threshold))

    return {
        "status": "calibrated",
        "threshold": float(threshold),
        "target_id_acceptance": target,
        "id_acceptance": id_acceptance,
        "ood_recall": ood_recall,
        "calibration_date": date.today().isoformat(),
    }


def _resolve_path(path: Path | str) -> Path:
    """Resolve a path relative to the project root when needed."""

    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    return candidate.resolve()


def _project_reference(path: Path) -> str:
    """Return a stable project-relative path when possible."""

    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path)


def _image_paths(root: Path) -> list[Path]:
    """Return deterministic image paths below an OOD calibration root."""

    extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    if not root.is_dir():
        return []
    return sorted(
        (path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in extensions),
        key=lambda path: path.relative_to(root).as_posix().lower(),
    )


def _collect_model_probabilities(model: Any, dataset: Any) -> np.ndarray:
    """Run inference without changing model weights."""

    batches: list[np.ndarray] = []
    for images in dataset:
        output = model(images, training=False)
        if isinstance(output, (list, tuple)):
            if len(output) != 1:
                raise ValueError("model must produce one probability output")
            output = output[0]
        probabilities = np.asarray(output.numpy(), dtype=np.float64)
        batches.append(probabilities)
    if not batches:
        raise ValueError("inference dataset is empty")
    result = np.concatenate(batches, axis=0)
    _probability_batch(result)
    return result


def _build_ood_dataset(paths: list[Path], batch_size: int) -> Any:
    """Build a TensorFlow image dataset for JPEG/PNG/WebP OOD files."""

    import tensorflow as tf

    from src.preprocessing import IMAGE_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH

    def preprocess(path: Any) -> Any:
        image = tf.io.decode_image(
            tf.io.read_file(path), channels=IMAGE_CHANNELS, expand_animations=False
        )
        image = tf.image.resize(image, (IMAGE_HEIGHT, IMAGE_WIDTH))
        image = tf.cast(image, tf.float32)
        return tf.ensure_shape(image, (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS))

    return (
        tf.data.Dataset.from_tensor_slices([str(path) for path in paths])
        .map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size, drop_remainder=False)
        .prefetch(tf.data.AUTOTUNE)
    )


def _load_model_and_scores(
    model_path: Path,
    id_split: Path,
    ood_root: Path,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Collect ID-validation and OOD-calibration probabilities from the model."""

    import tensorflow as tf
    import pandas as pd

    from src.dataset_pipeline import create_dataset
    from src.label_encoder import fit_label_encoder

    if not model_path.is_file():
        raise FileNotFoundError(f"model not found: {model_path}")
    if not id_split.is_file():
        raise FileNotFoundError(f"ID validation split not found: {id_split}")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    # The shipped .keras artifact contains MobileNetV2 preprocessing as a
    # Lambda named ``preprocess_input``. Register it before deserialization so
    # the evaluation command works with the same artifact as the baseline
    # evaluator.
    tf.keras.utils.get_custom_objects()["preprocess_input"] = (
        tf.keras.applications.mobilenet_v2.preprocess_input
    )
    model = tf.keras.models.load_model(str(model_path), compile=False)
    id_df = pd.read_csv(id_split)
    required = {"breed", "img_name"}
    if not required.issubset(id_df.columns):
        raise ValueError(f"ID split must contain columns {sorted(required)}")
    if id_df.empty:
        raise ValueError("ID validation split is empty")
    encoder = fit_label_encoder()
    id_dataset = create_dataset(id_df, encoder, batch_size=batch_size, training=False)
    id_probabilities = _collect_model_probabilities(model, (images for images, _ in id_dataset))

    ood_paths = _image_paths(ood_root)
    if not ood_paths:
        return id_probabilities, np.empty((0, id_probabilities.shape[1]), dtype=np.float64), 0
    ood_dataset = _build_ood_dataset(ood_paths, batch_size)
    ood_probabilities = _collect_model_probabilities(model, ood_dataset)
    return id_probabilities, ood_probabilities, len(ood_paths)


def _save_distribution_plot(
    id_scores: np.ndarray,
    ood_scores: np.ndarray,
    threshold: float | None,
    output_path: Path,
) -> None:
    """Save the MSP ID/OOD distribution plot."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(9, 5.5))
    if id_scores.size:
        axis.hist(id_scores, bins=30, alpha=0.68, density=True, label=f"ID validation (n={len(id_scores)})")
    if ood_scores.size:
        axis.hist(ood_scores, bins=30, alpha=0.68, density=True, label=f"OOD calibration (n={len(ood_scores)})")
    else:
        axis.text(
            0.98,
            0.96,
            "OOD calibration set unavailable",
            transform=axis.transAxes,
            ha="right",
            va="top",
            fontsize=10,
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
        )
    if threshold is not None:
        axis.axvline(threshold, color="black", linestyle="--", linewidth=1.5, label=f"threshold={threshold:.4f}")
    axis.set(
        title="Maximum Softmax Probability (MSP) distribution",
        xlabel="MSP = max(softmax probability)",
        ylabel="Density",
        xlim=(0.0, 1.0),
    )
    axis.grid(axis="y", alpha=0.25)
    axis.legend(loc="best")
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def run_calibration(
    id_probabilities: np.ndarray,
    ood_probabilities: np.ndarray,
    *,
    metrics_path: Path | str = DEFAULT_METRICS_PATH,
    plot_path: Path | str = DEFAULT_PLOT_PATH,
    target_id_acceptance: float = 0.95,
    id_source: str = "ID validation",
    ood_source: str = "OOD calibration",
) -> dict[str, Any]:
    """Score probability arrays and write the TASK 08 artifacts."""

    id_values = _probability_batch(id_probabilities)
    ood_values = np.asarray(ood_probabilities, dtype=np.float64)
    if ood_values.size:
        ood_values = _probability_batch(ood_values)
        if ood_values.shape[1] != id_values.shape[1]:
            raise ValueError("ID and OOD probability class counts must match")
    else:
        ood_values = np.empty((0, id_values.shape[1]), dtype=np.float64)

    id_scores = np.asarray(score(id_values), dtype=np.float64).reshape(-1)
    ood_scores = (
        np.asarray(score(ood_values), dtype=np.float64).reshape(-1)
        if ood_values.size
        else np.array([], dtype=np.float64)
    )

    if ood_scores.size:
        calibration = calibrate_msp_threshold(
            id_scores,
            ood_scores,
            target_id_acceptance=target_id_acceptance,
        )
        threshold = float(calibration["threshold"])
    else:
        calibration = {
            "status": "not_calibrated",
            "threshold": None,
            "target_id_acceptance": float(target_id_acceptance),
            "id_acceptance": None,
            "ood_recall": None,
            "reason": "No OOD calibration samples were available.",
        }
        threshold = None

    metrics_output = _resolve_path(metrics_path)
    plot_output = _resolve_path(plot_path)
    _save_distribution_plot(id_scores, ood_scores, threshold, plot_output)

    payload: dict[str, Any] = {
        "task": "TASK 08 — MAXIMUM SOFTMAX PROBABILITY OOD",
        "method": "msp",
        "score_definition": "max(softmax_probability)",
        "score_orientation": "higher_is_more_in_distribution",
        "id_source": id_source,
        "ood_source": ood_source,
        "id_validation": distribution_statistics(id_scores),
        "ood_calibration": distribution_statistics(ood_scores),
        "calibration": calibration,
        "threshold": threshold,
        "production_method_selected": False,
        "artifacts": {
            "metrics_json": _project_reference(metrics_output),
            "distribution_plot": _project_reference(plot_output),
        },
    }
    metrics_output.parent.mkdir(parents=True, exist_ok=True)
    metrics_output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return payload


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the TASK 08 MSP OOD calibration.")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--id-split", type=Path, default=DEFAULT_ID_SPLIT)
    parser.add_argument("--ood-root", type=Path, default=DEFAULT_OOD_ROOT)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--target-id-acceptance", type=float, default=0.95)
    parser.add_argument("--metrics-path", type=Path, default=DEFAULT_METRICS_PATH)
    parser.add_argument("--plot-path", type=Path, default=DEFAULT_PLOT_PATH)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    model_path = _resolve_path(args.model_path)
    id_split = _resolve_path(args.id_split)
    ood_root = _resolve_path(args.ood_root)
    id_probabilities, ood_probabilities, ood_count = _load_model_and_scores(
        model_path, id_split, ood_root, args.batch_size
    )
    payload = run_calibration(
        id_probabilities,
        ood_probabilities,
        metrics_path=args.metrics_path,
        plot_path=args.plot_path,
        target_id_acceptance=args.target_id_acceptance,
        id_source=_project_reference(id_split),
        ood_source=_project_reference(ood_root),
    )
    print(json.dumps({
        "id_count": payload["id_validation"]["count"],
        "ood_count": ood_count,
        "threshold": payload["threshold"],
        "calibration_status": payload["calibration"]["status"],
        "metrics": str(_resolve_path(args.metrics_path)),
        "plot": str(_resolve_path(args.plot_path)),
    }, indent=2))


if __name__ == "__main__":
    main()
