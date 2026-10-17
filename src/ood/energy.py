"""Energy-based OOD scoring and temperature calibration.

Energy is computed only from pre-softmax logits::

    E(x) = -T * logsumexp(logits / T)

The production model currently exposes a ``Dense(..., activation="softmax")``
output.  :func:`build_logits_model` reconstructs the linear projection from
that layer's input and a copy of its kernel/bias.  The source model is never
assigned to, trained, or otherwise mutated.  In particular, probabilities
are not accepted as a fallback for energy scoring: a softmax output loses the
arbitrary additive logit offset required by the true energy score.

The module also provides the evaluation entry point used for Task 10::

    python -m src.ood.energy

It evaluates the ID validation split against ``data/ood/calibration`` for
temperatures 1, 2, 5, and 10, and writes the requested JSON and plot.  If the
OOD directory is empty, the artifacts are still written with an explicit
``missing_ood_calibration_data`` status and no separation claim.
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


DEFAULT_TEMPERATURES = (1.0, 2.0, 5.0, 10.0)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "outputs" / "deployment_ready" / "dog_breed_model.keras"
DEFAULT_OOD_CALIBRATION_DIR = PROJECT_ROOT / "data" / "ood" / "calibration"
DEFAULT_METRICS_PATH = PROJECT_ROOT / "outputs" / "metrics" / "energy_calibration.json"
DEFAULT_PLOT_PATH = PROJECT_ROOT / "outputs" / "plots" / "energy_distribution.png"
_IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".webp", ".bmp"})


def _logsumexp(values: np.ndarray, axis: int = -1) -> np.ndarray:
    """Compute log-sum-exp stably for finite NumPy values."""

    values = np.asarray(values, dtype=np.float64)
    maximum = np.max(values, axis=axis, keepdims=True)
    return np.squeeze(maximum, axis=axis) + np.log(
        np.exp(values - maximum).sum(axis=axis)
    )


def _validate_temperature(temperature: float) -> float:
    try:
        value = float(temperature)
    except (TypeError, ValueError) as exc:
        raise ValueError("temperature must be a positive finite number") from exc
    if not np.isfinite(value) or value <= 0:
        raise ValueError("temperature must be a positive finite number")
    return value


def _logit_batch(logits: np.ndarray) -> tuple[np.ndarray, bool]:
    values = np.asarray(logits, dtype=np.float64)
    one_dimensional = values.ndim == 1
    if one_dimensional:
        values = values[None, :]
    if values.ndim != 2 or values.shape[1] == 0 or not np.isfinite(values).all():
        raise ValueError("logits must have shape (batch, classes) and be finite")
    return values, one_dimensional


def energy_score(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray | float:
    """Return ``-T * logsumexp(logits / T)``.

    ``logits`` must be the pre-softmax classifier output.  This function
    intentionally has no probability-input branch; passing probabilities is a
    caller error because probabilities cannot recover the original logits.
    """

    value = _validate_temperature(temperature)
    values, one_dimensional = _logit_batch(logits)
    energy = -value * _logsumexp(values / value, axis=1)
    return float(energy[0]) if one_dimensional else energy


def energy_ood_score(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray | float:
    """Return energy with the package OOD orientation.

    Higher energy is treated as more likely OOD.  The returned value is the
    conventional energy itself; no softmax probabilities are used.
    """

    return energy_score(logits, temperature=temperature)


def _tensorflow() -> Any:
    """Import TensorFlow lazily so pure scoring remains dependency-light."""

    try:
        import tensorflow as tf
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise RuntimeError("TensorFlow is required for model logit extraction") from exc
    return tf


def _single_input(model: Any) -> Any:
    inputs = getattr(model, "inputs", None)
    if inputs is None:
        raise ValueError("model must expose one Keras input")
    if isinstance(inputs, (list, tuple)):
        if len(inputs) != 1:
            raise ValueError("logit extraction supports exactly one model input")
        return inputs[0]
    return inputs


def _single_tensor(value: Any, name: str) -> Any:
    if isinstance(value, (list, tuple)):
        if len(value) != 1:
            raise ValueError(f"{name} must be a single tensor")
        return value[0]
    return value


def _activation_name(layer: Any) -> str:
    activation = getattr(layer, "activation", None)
    return str(getattr(activation, "__name__", activation or "")).lower()


def _is_softmax_layer(layer: Any, tf: Any) -> bool:
    if isinstance(layer, tf.keras.layers.Softmax):
        return True
    if isinstance(layer, tf.keras.layers.Activation):
        return _activation_name(layer) == "softmax"
    return False


def build_logits_model(model: Any) -> Any:
    """Return a Keras model that exposes the production model's logits.

    For a final softmax Dense, the returned adapter uses the same upstream
    tensor and a new linear Dense initialized from copies of the production
    classifier weights.  ``model`` itself is not changed.  For a linear Dense
    output, the existing output tensor is already the logits.
    """

    tf = _tensorflow()
    layers = list(getattr(model, "layers", ()))
    if not layers:
        raise ValueError("model must contain a classifier output layer")

    final = layers[-1]
    model_input = _single_input(model)
    logits_tensor = None

    if isinstance(final, tf.keras.layers.Dense):
        activation = _activation_name(final)
        if activation in {"linear", "identity"}:
            logits_tensor = _single_tensor(final.output, "classifier output")
        elif activation == "softmax":
            classifier_input = _single_tensor(final.input, "classifier input")
            # This new projection receives copied values only.  No source
            # model variable is assigned or modified.
            linear = tf.keras.layers.Dense(
                int(final.units),
                activation=None,
                use_bias=bool(final.use_bias),
                dtype=final.dtype,
                name=f"{final.name}_pre_softmax_logits",
            )
            logits_tensor = linear(classifier_input)
            linear.set_weights(final.get_weights())
        else:
            raise ValueError(
                f"final Dense layer '{final.name}' uses unsupported activation '{activation}'; "
                "cannot recover pre-softmax logits"
            )
    elif _is_softmax_layer(final, tf):
        # A separate Softmax/Activation layer already receives logits.
        logits_tensor = _single_tensor(final.input, "softmax input")
    else:
        raise ValueError(
            "could not identify a final linear classifier or softmax layer; "
            "pre-softmax logits are required for energy"
        )

    shape = getattr(logits_tensor, "shape", None)
    if shape is None or len(shape) != 2 or shape[-1] is None:
        raise ValueError(f"logits tensor must have shape (batch, classes), got {shape}")
    return tf.keras.Model(
        inputs=model_input,
        outputs=logits_tensor,
        name=f"{getattr(model, 'name', 'model')}_pre_softmax_logits",
    )


def extract_logits(logits_model: Any, image_batch: Any) -> np.ndarray:
    """Run a logits adapter and validate its finite 2-D output."""

    result = np.asarray(logits_model(image_batch, training=False), dtype=np.float64)
    if result.ndim != 2 or result.shape[0] == 0 or result.shape[1] == 0:
        raise RuntimeError(f"logit output must have shape (batch, classes), got {result.shape}")
    if not np.isfinite(result).all():
        raise RuntimeError("logit output contains NaN or Inf")
    return result


def _finite_logit_matrix(values: np.ndarray, name: str) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 2 or result.shape[0] == 0 or result.shape[1] == 0:
        raise ValueError(f"{name} must have shape (samples, classes) and be non-empty")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} contains NaN or Inf")
    return result


def _summary(scores: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    if values.size == 0:
        return {"count": 0}
    return {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "median": float(np.median(values)),
    }


def _standardized_gap(id_scores: np.ndarray, ood_scores: np.ndarray) -> float:
    denominator = float(np.sqrt((np.var(id_scores) + np.var(ood_scores)) / 2.0))
    gap = float(np.mean(ood_scores) - np.mean(id_scores))
    return gap if denominator == 0.0 else gap / denominator


def evaluate_energy_grid(
    id_logits: np.ndarray,
    ood_logits: np.ndarray,
    temperatures: Iterable[float] = DEFAULT_TEMPERATURES,
    *,
    target_id_acceptance: float = 0.95,
) -> dict[str, Any]:
    """Evaluate energy metrics for every requested temperature.

    This pure-NumPy entry point is useful for tests and for callers that have
    already collected logits.  OOD scores are oriented so larger values mean
    more likely OOD, matching :func:`src.ood.evaluate_ood.evaluate_ood_scores`.
    """

    ids = _finite_logit_matrix(id_logits, "id_logits")
    oods = _finite_logit_matrix(ood_logits, "ood_logits")
    if ids.shape[1] != oods.shape[1]:
        raise ValueError("ID and OOD logits must have the same class count")

    from .calibration import calibrate_threshold
    from .evaluate_ood import evaluate_ood_scores, rejection_error_rates

    results: dict[str, Any] = {}
    for raw_temperature in temperatures:
        temperature = _validate_temperature(raw_temperature)
        id_scores = np.asarray(energy_score(ids, temperature), dtype=np.float64)
        ood_scores = np.asarray(energy_score(oods, temperature), dtype=np.float64)
        metrics = evaluate_ood_scores(id_scores, ood_scores, detector="energy")
        calibration = calibrate_threshold(
            id_scores,
            ood_scores,
            target_id_acceptance=target_id_acceptance,
        )
        metrics.update(
            {
                "temperature": temperature,
                "id_energy": _summary(id_scores),
                "ood_energy": _summary(ood_scores),
                "mean_separation": float(np.mean(ood_scores) - np.mean(id_scores)),
                "standardized_mean_separation": _standardized_gap(id_scores, ood_scores),
                "calibration": calibration,
                "threshold_errors": rejection_error_rates(
                    id_scores,
                    ood_scores,
                    calibration["ood_threshold"],
                    higher_score_is_ood=True,
                ),
            }
        )
        results[str(temperature)] = metrics

    if not results:
        raise ValueError("at least one temperature is required")

    def ranking(item: tuple[str, Mapping[str, Any]]) -> tuple[float, float, float]:
        metrics = item[1]
        auroc = float(metrics["auroc"])
        fpr95 = float(metrics["fpr95"])
        gap = float(metrics["standardized_mean_separation"])
        return (
            auroc if np.isfinite(auroc) else -np.inf,
            -(fpr95 if np.isfinite(fpr95) else np.inf),
            gap if np.isfinite(gap) else -np.inf,
        )

    best_key, best_metrics = max(results.items(), key=ranking)
    return {
        "temperatures": results,
        "best_temperature": float(best_metrics["temperature"]),
        "selection_metric": "AUROC descending, then FPR95 ascending, then standardized mean separation descending",
        "best_temperature_key": best_key,
        "production_selected": False,
    }


def plot_energy_distributions(
    scores_by_temperature: Mapping[float | str, Mapping[str, np.ndarray]],
    output_path: str | Path,
    *,
    title_suffix: str = "",
) -> None:
    """Write ID/OOD energy histograms for the temperature grid."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    items = list(scores_by_temperature.items())
    columns = 2
    rows = max(1, int(np.ceil(len(items) / columns)))
    figure, axes = plt.subplots(rows, columns, figsize=(13, 4.2 * rows), squeeze=False)
    axes_flat = axes.ravel()
    for axis, (temperature, score_sets) in zip(axes_flat, items):
        id_scores = np.asarray(score_sets["id"], dtype=np.float64).reshape(-1)
        ood_scores = np.asarray(score_sets.get("ood", []), dtype=np.float64).reshape(-1)
        if id_scores.size:
            axis.hist(id_scores, bins=30, density=True, alpha=0.62, label="ID validation")
        if ood_scores.size:
            axis.hist(ood_scores, bins=30, density=True, alpha=0.62, label="OOD calibration")
        axis.set_title(f"Temperature T={float(temperature):g}")
        axis.set_xlabel("Energy (higher = more OOD)")
        axis.set_ylabel("Density")
        axis.grid(alpha=0.2)
        if id_scores.size and ood_scores.size:
            axis.legend()
        elif id_scores.size:
            axis.legend(["ID validation"])
            axis.text(
                0.03,
                0.94,
                "OOD calibration unavailable",
                transform=axis.transAxes,
                va="top",
                fontsize=9,
                color="darkred",
            )
    for axis in axes_flat[len(items):]:
        axis.set_visible(False)
    figure.suptitle(
        "Energy-Based OOD Distributions" + (f" — {title_suffix}" if title_suffix else ""),
        fontsize=14,
    )
    figure.tight_layout()
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=160, bbox_inches="tight")
    plt.close(figure)


def _resolve_project_path(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else PROJECT_ROOT / value


def _load_model(model_path: Path) -> Any:
    tf = _tensorflow()
    if not model_path.is_file():
        raise FileNotFoundError(f"Model not found: {model_path}")
    tf.keras.utils.get_custom_objects()["preprocess_input"] = (
        tf.keras.applications.mobilenet_v2.preprocess_input
    )
    return tf.keras.models.load_model(str(model_path), compile=False, safe_mode=True)


def _id_validation_dataset(batch_size: int) -> Any:
    from src.dataset_pipeline import create_dataset, load_split_csv
    from src.label_encoder import fit_label_encoder

    validation = load_split_csv("val")
    encoder = fit_label_encoder()
    return create_dataset(validation, encoder, batch_size=batch_size, training=False), len(validation)


def _preprocess_any_image(path: Any) -> Any:
    from .image_loading import preprocess_ood_image

    return preprocess_ood_image(path)


def _ood_calibration_dataset(directory: Path, batch_size: int) -> tuple[Any | None, int]:
    tf = _tensorflow()
    paths = sorted(
        path for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in _IMAGE_EXTENSIONS
    ) if directory.is_dir() else []
    if not paths:
        return None, 0
    dataset = tf.data.Dataset.from_tensor_slices([str(path) for path in paths])
    dataset = dataset.map(_preprocess_any_image, num_parallel_calls=tf.data.AUTOTUNE)
    return dataset.batch(batch_size, drop_remainder=False).prefetch(tf.data.AUTOTUNE), len(paths)


def _collect_logits(logits_model: Any, dataset: Any, *, labelled: bool) -> np.ndarray:
    batches = []
    for batch in dataset:
        images = batch[0] if labelled else batch
        batches.append(extract_logits(logits_model, images))
    if not batches:
        raise ValueError("cannot collect logits from an empty dataset")
    return np.concatenate(batches, axis=0)


def _weight_snapshot(model: Any) -> list[np.ndarray]:
    return [np.array(weight.numpy(), copy=True) for weight in model.weights]


def _weights_unchanged(model: Any, snapshot: Sequence[np.ndarray]) -> bool:
    return len(snapshot) == len(model.weights) and all(
        np.array_equal(before, weight.numpy())
        for before, weight in zip(snapshot, model.weights)
    )


def _write_missing_ood_artifacts(
    id_logits: np.ndarray,
    temperatures: Sequence[float],
    metrics_path: Path,
    plot_path: Path,
    *,
    model_path: Path,
    ood_directory: Path,
    id_count: int,
) -> dict[str, Any]:
    score_sets: dict[float, dict[str, np.ndarray]] = {}
    temperature_payload: dict[str, Any] = {}
    for raw_temperature in temperatures:
        temperature = _validate_temperature(raw_temperature)
        id_scores = np.asarray(energy_score(id_logits, temperature), dtype=np.float64)
        score_sets[temperature] = {"id": id_scores, "ood": np.array([], dtype=np.float64)}
        temperature_payload[str(temperature)] = {
            "temperature": temperature,
            "id_energy": _summary(id_scores),
            "ood_energy": {"count": 0},
            "metrics": None,
        }
    payload = {
        "task": "energy_score_ood",
        "status": "missing_ood_calibration_data",
        "evaluation_date": date.today().isoformat(),
        "formula": "E(x) = -T * logsumexp(logits / T)",
        "logits_source": "pre-softmax classifier output",
        "softmax_used_for_energy": False,
        "model_path": str(model_path),
        "id_split": "validation",
        "id_validation_count": int(id_count),
        "ood_calibration_directory": str(ood_directory),
        "ood_calibration_count": 0,
        "temperatures": temperature_payload,
        "best_temperature": None,
        "production_selected": False,
        "notes": [
            "OOD calibration directory contains no supported images.",
            "Separation and production recommendation are intentionally unavailable until OOD calibration data is supplied.",
        ],
        "artifacts": {
            "metrics_json": str(metrics_path),
            "distribution_plot": str(plot_path),
        },
    }
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    plot_energy_distributions(
        score_sets,
        plot_path,
        title_suffix="ID validation only; OOD calibration data missing",
    )
    return payload


def evaluate_energy_calibration(
    model_path: str | Path = DEFAULT_MODEL_PATH,
    *,
    ood_directory: str | Path = DEFAULT_OOD_CALIBRATION_DIR,
    metrics_path: str | Path = DEFAULT_METRICS_PATH,
    plot_path: str | Path = DEFAULT_PLOT_PATH,
    batch_size: int = 64,
    temperatures: Sequence[float] = DEFAULT_TEMPERATURES,
    target_id_acceptance: float = 0.95,
) -> dict[str, Any]:
    """Evaluate the current model on ID validation and OOD calibration data."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    resolved_model = _resolve_project_path(model_path).resolve()
    resolved_ood = _resolve_project_path(ood_directory).resolve()
    resolved_metrics = _resolve_project_path(metrics_path).resolve()
    resolved_plot = _resolve_project_path(plot_path).resolve()

    model = _load_model(resolved_model)
    snapshot = _weight_snapshot(model)
    logits_model = build_logits_model(model)
    id_dataset, id_count = _id_validation_dataset(batch_size)
    id_logits = _collect_logits(logits_model, id_dataset, labelled=True)
    if not _weights_unchanged(model, snapshot):
        raise RuntimeError("building or running logits adapter changed production model weights")

    ood_dataset, ood_count = _ood_calibration_dataset(resolved_ood, batch_size)
    if ood_dataset is None:
        return _write_missing_ood_artifacts(
            id_logits,
            temperatures,
            resolved_metrics,
            resolved_plot,
            model_path=resolved_model,
            ood_directory=resolved_ood,
            id_count=id_count,
        )

    ood_logits = _collect_logits(logits_model, ood_dataset, labelled=False)
    grid = evaluate_energy_grid(
        id_logits,
        ood_logits,
        temperatures,
        target_id_acceptance=target_id_acceptance,
    )
    score_sets: dict[float, dict[str, np.ndarray]] = {}
    for raw_temperature in temperatures:
        temperature = _validate_temperature(raw_temperature)
        score_sets[temperature] = {
            "id": np.asarray(energy_score(id_logits, temperature), dtype=np.float64),
            "ood": np.asarray(energy_score(ood_logits, temperature), dtype=np.float64),
        }

    payload = {
        "task": "energy_score_ood",
        "status": "evaluated",
        "evaluation_date": date.today().isoformat(),
        "formula": "E(x) = -T * logsumexp(logits / T)",
        "logits_source": "pre-softmax classifier output",
        "softmax_used_for_energy": False,
        "model_path": str(resolved_model),
        "id_split": "validation",
        "id_validation_count": int(id_count),
        "ood_calibration_directory": str(resolved_ood),
        "ood_calibration_count": int(ood_count),
        **grid,
        "artifacts": {
            "metrics_json": str(resolved_metrics),
            "distribution_plot": str(resolved_plot),
        },
    }
    resolved_metrics.parent.mkdir(parents=True, exist_ok=True)
    resolved_metrics.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    plot_energy_distributions(score_sets, resolved_plot)
    return payload


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate energy-based OOD separation across temperatures.")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--ood-directory", type=Path, default=DEFAULT_OOD_CALIBRATION_DIR)
    parser.add_argument("--metrics-path", type=Path, default=DEFAULT_METRICS_PATH)
    parser.add_argument("--plot-path", type=Path, default=DEFAULT_PLOT_PATH)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--target-id-acceptance", type=float, default=0.95)
    parser.add_argument("--temperatures", nargs="+", type=float, default=list(DEFAULT_TEMPERATURES))
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    payload = evaluate_energy_calibration(
        model_path=args.model_path,
        ood_directory=args.ood_directory,
        metrics_path=args.metrics_path,
        plot_path=args.plot_path,
        batch_size=args.batch_size,
        temperatures=args.temperatures,
        target_id_acceptance=args.target_id_acceptance,
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()


# Backwards-compatible short aliases used by the other OOD modules.
score = energy_ood_score
ood_score = energy_ood_score
energy = energy_ood_score
