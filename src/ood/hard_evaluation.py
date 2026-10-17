"""Evaluate the fixed OOD detectors on the isolated Hard OOD pool.

This module is deliberately separate from calibration and from the regular
OOD benchmark.  It only consumes the already-built model/detector artifacts
and the fixed thresholds that are present in those artifacts.  In particular,
``run_hard_evaluation`` never calls a threshold-selection function and never
uses Hard OOD scores to create or update a threshold.

The default input layout is::

    data/ood_hard/{wolf,fox,coyote,hyena,dog_toy,dog_statue,
                   dog_cartoon,dog_drawing}/*.{jpg,jpeg,png,webp,bmp}

The report contains one table per detector.  Scores are OOD-oriented (larger
means more likely OOD), including the runtime MSP score ``1 - max(p)``.
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .embedding_distance import cosine_distance, load_centroid_artifacts, mahalanobis_distance
from .energy import energy_ood_score
from .entropy import entropy_ood_score
from .msp import msp_ood_score


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "outputs" / "deployment_ready" / "dog_breed_model.keras"
DEFAULT_HARD_ROOT = PROJECT_ROOT / "data" / "ood_hard"
DEFAULT_REPORT_PATH = PROJECT_ROOT / "outputs" / "reports" / "HARD_OOD_REPORT.md"
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "outputs" / "deployment_ready" / "ood_config.json"
DEFAULT_DEPLOYMENT_THRESHOLDS_PATH = (
    PROJECT_ROOT / "outputs" / "deployment_ready" / "ood_thresholds.json"
)
DEFAULT_CALIBRATION_THRESHOLDS_PATH = (
    PROJECT_ROOT / "outputs" / "calibration" / "ood_thresholds.json"
)
DEFAULT_CENTROIDS_PATH = PROJECT_ROOT / "outputs" / "calibration" / "class_centroids.npy"
DEFAULT_CENTROID_METADATA_PATH = PROJECT_ROOT / "outputs" / "calibration" / "class_centroids.json"
DEFAULT_CLASS_NAMES_PATH = PROJECT_ROOT / "outputs" / "deployment_ready" / "class_names.json"
DEFAULT_MAHALANOBIS_MEAN_PATH = PROJECT_ROOT / "outputs" / "calibration" / "mahalanobis_mean.npy"
DEFAULT_MAHALANOBIS_COVARIANCE_PATH = PROJECT_ROOT / "outputs" / "calibration" / "covariance.npy"
DEFAULT_ENERGY_TEMPERATURE = 1.0

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
SUPPORTED_DETECTORS = ("msp", "entropy", "energy", "cosine", "mahalanobis")
IMAGE_EXTENSIONS = frozenset({".bmp", ".jpeg", ".jpg", ".png", ".webp"})


def _normalise_detector(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    name = value.strip().lower().replace("-", "_")
    return {
        "maximum_softmax_probability": "msp",
        "predictive_entropy": "entropy",
        "energy_score": "energy",
        "embedding_cosine": "cosine",
        "embedding_mahalanobis": "mahalanobis",
    }.get(name, name if name in SUPPORTED_DETECTORS else None)


def discover_hard_images(root: str | Path = DEFAULT_HARD_ROOT) -> dict[str, list[Path]]:
    """Return deterministic image lists for all required Hard OOD categories."""

    root_path = Path(root)
    result: dict[str, list[Path]] = {}
    for category in HARD_OOD_CATEGORIES:
        directory = root_path / category
        result[category] = sorted(
            (
                path
                for path in directory.rglob("*")
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            ),
            key=lambda path: path.relative_to(directory).as_posix().lower(),
        ) if directory.is_dir() else []
    return result


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    return value if isinstance(value, dict) else None


def _finite_number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def load_fixed_thresholds(
    *,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    deployment_thresholds_path: str | Path = DEFAULT_DEPLOYMENT_THRESHOLDS_PATH,
    calibration_thresholds_path: str | Path = DEFAULT_CALIBRATION_THRESHOLDS_PATH,
) -> tuple[dict[str, float], dict[str, str]]:
    """Load thresholds without fitting or selecting any threshold.

    The calibrated artifact is preferred when it explicitly names one of the
    supported detectors.  Otherwise the deployment artifact is used.  The
    deployment config is a final fallback for its active runtime method.  A
    threshold missing from all fixed artifacts is intentionally omitted.
    """

    thresholds: dict[str, float] = {}
    sources: dict[str, str] = {}

    for raw_path in (calibration_thresholds_path, deployment_thresholds_path):
        path = Path(raw_path)
        payload = _read_json(path)
        if payload is None:
            continue
        detector = _normalise_detector(payload.get("detector") or payload.get("method"))
        value = _finite_number(payload.get("ood_threshold", payload.get("threshold")))
        if detector is not None and value is not None and detector not in thresholds:
            thresholds[detector] = value
            sources[detector] = path.as_posix()

    config_file = Path(config_path)
    config = _read_json(config_file)
    if config is not None:
        detector = _normalise_detector(config.get("method"))
        config_thresholds = config.get("thresholds")
        if isinstance(config_thresholds, Mapping):
            value = _finite_number(config_thresholds.get("not_dog"))
            if detector is not None and value is not None and detector not in thresholds:
                thresholds[detector] = value
                sources[detector] = config_file.as_posix()
    return thresholds, sources


def false_acceptance_metrics(
    scores: Sequence[float] | np.ndarray,
    threshold: float | None,
    *,
    higher_score_is_ood: bool = True,
) -> dict[str, int | float | None]:
    """Return false-acceptance metrics using a fixed raw-score threshold.

    ``false_acceptance`` means an OOD image was accepted by the detector as
    non-OOD.  With zero samples, count/rate are undefined and are returned as
    ``None`` instead of a misleading zero.
    """

    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    if values.size and not np.isfinite(values).all():
        raise ValueError("scores must be finite")
    if threshold is None or values.size == 0:
        return {"false_acceptance": None, "false_acceptance_rate": None}
    threshold_value = _finite_number(threshold)
    if threshold_value is None:
        raise ValueError("threshold must be finite")
    rejected = values >= threshold_value if higher_score_is_ood else values <= threshold_value
    false_acceptance = int(np.sum(~rejected))
    return {
        "false_acceptance": false_acceptance,
        "false_acceptance_rate": float(false_acceptance / values.size),
    }


def category_metric(
    scores: Sequence[float] | np.ndarray,
    *,
    threshold: float | None,
    higher_score_is_ood: bool = True,
) -> dict[str, int | float | None]:
    """Build the report metric contract for one detector/category pair."""

    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    if values.size and not np.isfinite(values).all():
        raise ValueError("scores must be finite")
    metrics: dict[str, int | float | None] = {
        "sample_count": int(values.size),
        "false_acceptance": None,
        "false_acceptance_rate": None,
        "mean_ood_score": None if values.size == 0 else float(np.mean(values)),
    }
    metrics.update(false_acceptance_metrics(values, threshold, higher_score_is_ood=higher_score_is_ood))
    return metrics


def _tensorflow() -> Any:
    try:
        import tensorflow as tf
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("TensorFlow is required when Hard OOD images are available") from exc
    return tf


def _image_dataset(tf: Any, paths: Sequence[Path], batch_size: int) -> Any:
    from .image_loading import preprocess_ood_image

    return (
        tf.data.Dataset.from_tensor_slices([str(path) for path in paths])
        .map(preprocess_ood_image, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size, drop_remainder=False)
        .prefetch(tf.data.AUTOTUNE)
    )


def _collect_outputs(
    combined_model: Any,
    tf: Any,
    images_by_category: Mapping[str, Sequence[Path]],
    *,
    batch_size: int,
) -> dict[str, dict[str, np.ndarray]]:
    outputs: dict[str, dict[str, np.ndarray]] = {}
    for category in HARD_OOD_CATEGORIES:
        paths = images_by_category[category]
        if not paths:
            outputs[category] = {
                "probabilities": np.empty((0, 0), dtype=np.float64),
                "logits": np.empty((0, 0), dtype=np.float64),
                "embeddings": np.empty((0, 0), dtype=np.float64),
            }
            continue
        probabilities: list[np.ndarray] = []
        logits: list[np.ndarray] = []
        embeddings: list[np.ndarray] = []
        for batch in _image_dataset(tf, paths, batch_size):
            values = combined_model(batch, training=False)
            values = list(values) if isinstance(values, (tuple, list)) else [values]
            if len(values) != 3:
                raise RuntimeError("combined Hard OOD model must return probabilities, logits, and embeddings")
            probabilities.append(np.asarray(values[0], dtype=np.float64))
            logits.append(np.asarray(values[1], dtype=np.float64))
            embeddings.append(np.asarray(values[2], dtype=np.float64))
        outputs[category] = {
            "probabilities": np.concatenate(probabilities, axis=0),
            "logits": np.concatenate(logits, axis=0),
            "embeddings": np.concatenate(embeddings, axis=0),
        }
    return outputs


def _load_scored_categories(
    images_by_category: Mapping[str, Sequence[Path]],
    *,
    model_path: Path,
    batch_size: int,
    energy_temperature: float,
    centroids_path: Path,
    centroid_metadata_path: Path,
    class_names_path: Path,
    mahalanobis_mean_path: Path,
    mahalanobis_covariance_path: Path,
) -> dict[str, dict[str, np.ndarray]]:
    tf = _tensorflow()
    if not model_path.is_file():
        raise FileNotFoundError(f"Model not found: {model_path}")
    tf.keras.utils.get_custom_objects()["preprocess_input"] = (
        tf.keras.applications.mobilenet_v2.preprocess_input
    )
    model = tf.keras.models.load_model(str(model_path), compile=False, safe_mode=True)
    from src.models.feature_extractor import build_feature_extractor

    from .energy import build_logits_model

    logits_model = build_logits_model(model)
    feature_model = build_feature_extractor(model)
    combined_model = tf.keras.Model(
        inputs=model.inputs,
        outputs=[model.output, logits_model.output, feature_model.output],
        name="hard_ood_read_only_outputs",
    )
    outputs = _collect_outputs(combined_model, tf, images_by_category, batch_size=batch_size)

    centroids = None
    try:
        centroids, _, _ = load_centroid_artifacts(
            centroids_path, centroid_metadata_path, class_names_path
        )
    except (FileNotFoundError, ValueError):
        pass

    mean = covariance = None
    if mahalanobis_mean_path.is_file() and mahalanobis_covariance_path.is_file():
        mean = np.load(mahalanobis_mean_path, allow_pickle=False)
        covariance = np.load(mahalanobis_covariance_path, allow_pickle=False)

    scored: dict[str, dict[str, np.ndarray]] = {}
    for category, values in outputs.items():
        probabilities = values["probabilities"]
        logits = values["logits"]
        embeddings = values["embeddings"]
        if probabilities.shape[0] == 0:
            scored[category] = {
                detector: np.empty(0, dtype=np.float64)
                for detector in SUPPORTED_DETECTORS
            }
            continue
        category_scores: dict[str, np.ndarray] = {
            "msp": np.asarray(msp_ood_score(probabilities), dtype=np.float64).reshape(-1),
            "entropy": np.asarray(entropy_ood_score(probabilities), dtype=np.float64).reshape(-1),
            "energy": np.asarray(energy_ood_score(logits, temperature=energy_temperature), dtype=np.float64).reshape(-1),
        }
        if centroids is not None:
            category_scores["cosine"] = np.asarray(cosine_distance(embeddings, centroids), dtype=np.float64).reshape(-1)
        if mean is not None and covariance is not None:
            category_scores["mahalanobis"] = np.asarray(
                mahalanobis_distance(embeddings, mean, covariance, regularization=0.0),
                dtype=np.float64,
            ).reshape(-1)
        scored[category] = category_scores
    return scored


def build_report(
    scores_by_category: Mapping[str, Mapping[str, Sequence[float] | np.ndarray]],
    *,
    thresholds: Mapping[str, float],
    threshold_sources: Mapping[str, str],
    hard_root: str | Path = DEFAULT_HARD_ROOT,
    status: str = "evaluated",
) -> str:
    """Render the complete Markdown report from already-collected scores."""

    lines = [
        "# Hard OOD Evaluation",
        "",
        f"- **Status:** `{status}`",
        f"- **Evaluation date:** `{date.today().isoformat()}`",
        f"- **Hard OOD source:** `{Path(hard_root).as_posix()}`",
        "- **Protocol:** evaluation-only; the eight categories below are not used for threshold tuning.",
        "- **Score direction:** larger `ood_score` means more likely OOD; false acceptance means an OOD image was not rejected at the fixed threshold.",
        "- **Production safety:** model and deployment thresholds were not changed.",
        "",
        "Thresholds are read from existing fixed artifacts only. `—` means the category has no images or the detector has no fixed threshold artifact.",
        "",
    ]
    for detector in SUPPORTED_DETECTORS:
        threshold = thresholds.get(detector)
        source = threshold_sources.get(detector)
        lines.extend(
            [
                f"## {detector}",
                "",
                f"- **Fixed threshold:** `{threshold:.6g}`" if threshold is not None else "- **Fixed threshold:** `—`",
                f"- **Threshold source:** `{source}`" if source else "- **Threshold source:** `—`",
                "",
                "| Category | sample_count | false_acceptance | false_acceptance_rate | mean_ood_score |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for category in HARD_OOD_CATEGORIES:
            raw_scores = scores_by_category.get(category, {}).get(detector, np.empty(0, dtype=np.float64))
            metric = category_metric(raw_scores, threshold=threshold)
            lines.append(
                "| {category} | {sample_count} | {false_acceptance} | {rate} | {mean} |".format(
                    category=category,
                    sample_count=metric["sample_count"],
                    false_acceptance="—" if metric["false_acceptance"] is None else metric["false_acceptance"],
                    rate="—" if metric["false_acceptance_rate"] is None else f"{metric['false_acceptance_rate']:.6f}",
                    mean="—" if metric["mean_ood_score"] is None else f"{metric['mean_ood_score']:.6f}",
                )
            )
        lines.append("")
    if status != "evaluated":
        lines.extend(
            [
                "## Data availability",
                "",
                "No Hard OOD images were present in the required category directories at evaluation time. The report preserves the requested category-level schema without treating missing samples as zero false acceptances.",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def run_hard_evaluation(
    *,
    hard_root: str | Path = DEFAULT_HARD_ROOT,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    report_path: str | Path = DEFAULT_REPORT_PATH,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    deployment_thresholds_path: str | Path = DEFAULT_DEPLOYMENT_THRESHOLDS_PATH,
    calibration_thresholds_path: str | Path = DEFAULT_CALIBRATION_THRESHOLDS_PATH,
    batch_size: int = 64,
    energy_temperature: float = DEFAULT_ENERGY_TEMPERATURE,
    centroids_path: str | Path = DEFAULT_CENTROIDS_PATH,
    centroid_metadata_path: str | Path = DEFAULT_CENTROID_METADATA_PATH,
    class_names_path: str | Path = DEFAULT_CLASS_NAMES_PATH,
    mahalanobis_mean_path: str | Path = DEFAULT_MAHALANOBIS_MEAN_PATH,
    mahalanobis_covariance_path: str | Path = DEFAULT_MAHALANOBIS_COVARIANCE_PATH,
) -> dict[str, Any]:
    """Run the fixed-threshold Hard OOD evaluation and write its report."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    images = discover_hard_images(hard_root)
    thresholds, threshold_sources = load_fixed_thresholds(
        config_path=config_path,
        deployment_thresholds_path=deployment_thresholds_path,
        calibration_thresholds_path=calibration_thresholds_path,
    )
    total_images = sum(len(paths) for paths in images.values())
    if total_images:
        scores = _load_scored_categories(
            images,
            model_path=Path(model_path),
            batch_size=batch_size,
            energy_temperature=energy_temperature,
            centroids_path=Path(centroids_path),
            centroid_metadata_path=Path(centroid_metadata_path),
            class_names_path=Path(class_names_path),
            mahalanobis_mean_path=Path(mahalanobis_mean_path),
            mahalanobis_covariance_path=Path(mahalanobis_covariance_path),
        )
        status = "evaluated"
    else:
        scores = {category: {} for category in HARD_OOD_CATEGORIES}
        status = "incomplete_missing_hard_ood_data"
    report = build_report(
        scores,
        thresholds=thresholds,
        threshold_sources=threshold_sources,
        hard_root=hard_root,
        status=status,
    )
    destination = Path(report_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(report, encoding="utf-8")
    return {
        "status": status,
        "hard_root": str(Path(hard_root).resolve()),
        "sample_counts": {category: len(paths) for category, paths in images.items()},
        "thresholds": dict(thresholds),
        "threshold_sources": dict(threshold_sources),
        "report_path": str(destination.resolve()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hard-root", type=Path, default=DEFAULT_HARD_ROOT)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--config-path", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--deployment-thresholds", type=Path, default=DEFAULT_DEPLOYMENT_THRESHOLDS_PATH)
    parser.add_argument("--calibration-thresholds", type=Path, default=DEFAULT_CALIBRATION_THRESHOLDS_PATH)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--energy-temperature", type=float, default=DEFAULT_ENERGY_TEMPERATURE)
    parser.add_argument("--centroids-path", type=Path, default=DEFAULT_CENTROIDS_PATH)
    parser.add_argument("--centroid-metadata-path", type=Path, default=DEFAULT_CENTROID_METADATA_PATH)
    parser.add_argument("--class-names-path", type=Path, default=DEFAULT_CLASS_NAMES_PATH)
    parser.add_argument("--mahalanobis-mean-path", type=Path, default=DEFAULT_MAHALANOBIS_MEAN_PATH)
    parser.add_argument("--mahalanobis-covariance-path", type=Path, default=DEFAULT_MAHALANOBIS_COVARIANCE_PATH)
    args = parser.parse_args()
    result = run_hard_evaluation(
        hard_root=args.hard_root,
        model_path=args.model_path,
        report_path=args.report_path,
        config_path=args.config_path,
        deployment_thresholds_path=args.deployment_thresholds,
        calibration_thresholds_path=args.calibration_thresholds,
        batch_size=args.batch_size,
        energy_temperature=args.energy_temperature,
        centroids_path=args.centroids_path,
        centroid_metadata_path=args.centroid_metadata_path,
        class_names_path=args.class_names_path,
        mahalanobis_mean_path=args.mahalanobis_mean_path,
        mahalanobis_covariance_path=args.mahalanobis_covariance_path,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()


__all__ = [
    "HARD_OOD_CATEGORIES",
    "SUPPORTED_DETECTORS",
    "discover_hard_images",
    "load_fixed_thresholds",
    "false_acceptance_metrics",
    "category_metric",
    "build_report",
    "run_hard_evaluation",
]
