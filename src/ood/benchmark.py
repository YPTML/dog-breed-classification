"""Run the full, analysis-only OOD benchmark.

The benchmark evaluates the five supported detectors on one common protocol:
ID samples are the selected validation split, OOD samples are the selected OOD
test directory, and OOD F1/FAR/FRR use the detector's FPR95 operating point.
Scores are always oriented so that larger values mean more likely OOD.

The runner accepts a score ``.npz`` for reproducible offline evaluation, or it
can collect probabilities, pre-softmax logits, and penultimate embeddings from
the read-only production model. Energy is never reconstructed from
probabilities. Cosine and Mahalanobis scores require explicit embedding
artifacts/statistics. No production inference code or deployment thresholds
are changed by this module.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from datetime import date
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

from .embedding_distance import (
    cosine_distance,
    load_centroid_artifacts,
    mahalanobis_distance,
)
from .energy import energy_score
from .entropy import entropy_ood_score
from .evaluate_ood import evaluate_ood_scores, rejection_error_rates
from .msp import score as msp_score


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "outputs" / "deployment_ready" / "dog_breed_model.keras"
DEFAULT_ID_SPLIT = "val"
DEFAULT_OOD_ROOT = PROJECT_ROOT / "data" / "ood" / "test"
DEFAULT_OUTPUT_JSON = PROJECT_ROOT / "outputs" / "metrics" / "ood_benchmark.json"
DEFAULT_OUTPUT_MARKDOWN = PROJECT_ROOT / "outputs" / "reports" / "OOD_BENCHMARK.md"
DEFAULT_CENTROIDS_PATH = PROJECT_ROOT / "outputs" / "calibration" / "class_centroids.npy"
DEFAULT_CENTROID_METADATA_PATH = PROJECT_ROOT / "outputs" / "calibration" / "class_centroids.json"
DEFAULT_CLASS_NAMES_PATH = PROJECT_ROOT / "outputs" / "deployment_ready" / "class_names.json"
DEFAULT_MAHALANOBIS_MEAN_PATH = PROJECT_ROOT / "outputs" / "calibration" / "mahalanobis_mean.npy"
DEFAULT_MAHALANOBIS_COVARIANCE_PATH = PROJECT_ROOT / "outputs" / "calibration" / "covariance.npy"
DEFAULT_TEMPERATURE = 1.0
SUPPORTED_METHODS = ("msp", "entropy", "energy", "cosine", "mahalanobis")
REQUIRED_METRICS = (
    "auroc",
    "fpr95",
    "aupr_in",
    "aupr_out",
    "ood_f1",
    "false_acceptance_rate",
    "false_rejection_rate",
)
_IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".webp", ".bmp"})


def _finite_matrix(values: Any, name: str, *, allow_empty: bool = False) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 2 or result.shape[1] == 0:
        raise ValueError(f"{name} must have shape (samples, dimensions)")
    if not allow_empty and result.shape[0] == 0:
        raise ValueError(f"{name} must contain at least one sample")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must contain only finite values")
    return result


def _score_array(function: Callable[[np.ndarray], Any], values: np.ndarray) -> np.ndarray:
    scores = np.asarray(function(values), dtype=np.float64).reshape(-1)
    if scores.size != values.shape[0] or not np.isfinite(scores).all():
        raise ValueError("score function returned an invalid score array")
    return scores


def _unavailable(
    name: str,
    *,
    reason: str,
    id_count: int,
    ood_count: int,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "method": name,
        "status": "unavailable",
        "status_reason": reason,
        "higher_score_is_ood": name != "msp",
        "id_count": int(id_count),
        "ood_count": int(ood_count),
    }
    result.update({metric: None for metric in REQUIRED_METRICS})
    result.update(
        {
            "threshold": None,
            "threshold_source": None,
            "true_positive": None,
            "true_negative": None,
            "false_positive": None,
            "false_negative": None,
            "score_latency_ms_per_image": None,
            "inference_latency_ms_per_image": None,
            "latency_ms_per_image": None,
        }
    )
    return result


def _evaluate(
    name: str,
    id_scores: np.ndarray,
    ood_scores: np.ndarray,
    *,
    score_elapsed_ms: float,
    inference_latency_ms_per_image: float | None = None,
) -> dict[str, Any]:
    if ood_scores.size == 0:
        result = _unavailable(
            name,
            reason="No OOD samples were available for the common evaluation protocol.",
            id_count=id_scores.size,
            ood_count=0,
        )
        result["score_latency_ms_per_image"] = float(score_elapsed_ms / max(id_scores.size, 1))
        result["inference_latency_ms_per_image"] = inference_latency_ms_per_image
        result["latency_ms_per_image"] = (
            None
            if inference_latency_ms_per_image is None
            else float(inference_latency_ms_per_image + result["score_latency_ms_per_image"])
        )
        return result

    metrics = dict(evaluate_ood_scores(id_scores, ood_scores, detector=name))
    metrics.update(
        rejection_error_rates(
            id_scores,
            ood_scores,
            float(metrics["threshold"]),
            higher_score_is_ood=bool(metrics["higher_score_is_ood"]),
        )
    )
    score_latency = score_elapsed_ms / (id_scores.size + ood_scores.size)
    metrics.update(
        {
            "method": name,
            "status": "evaluated",
            "score_latency_ms_per_image": float(score_latency),
            "inference_latency_ms_per_image": inference_latency_ms_per_image,
            "latency_ms_per_image": (
                None
                if inference_latency_ms_per_image is None
                else float(inference_latency_ms_per_image + score_latency)
            ),
        }
    )
    return metrics


def _run_one(
    name: str,
    scorer: Callable[[np.ndarray], Any],
    id_values: np.ndarray,
    ood_values: np.ndarray,
    *,
    inference_latency_ms_per_image: float | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    id_scores = _score_array(scorer, id_values)
    ood_scores = (
        _score_array(scorer, ood_values)
        if ood_values.shape[0]
        else np.empty(0, dtype=np.float64)
    )
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return _evaluate(
        name,
        id_scores,
        ood_scores,
        score_elapsed_ms=elapsed_ms,
        inference_latency_ms_per_image=inference_latency_ms_per_image,
    )


def benchmark_methods(
    id_probabilities: np.ndarray,
    ood_probabilities: np.ndarray,
    *,
    id_logits: np.ndarray | None = None,
    ood_logits: np.ndarray | None = None,
    id_embeddings: np.ndarray | None = None,
    ood_embeddings: np.ndarray | None = None,
    centroids: np.ndarray | None = None,
    mahalanobis_mean: np.ndarray | None = None,
    mahalanobis_covariance: np.ndarray | None = None,
    energy_temperature: float = DEFAULT_TEMPERATURE,
    inference_latency_ms_per_image: float | None = None,
) -> dict[str, dict[str, Any]]:
    """Evaluate all five detectors from explicit raw model outputs."""

    ids = _finite_matrix(id_probabilities, "id_probabilities")
    oods = _finite_matrix(ood_probabilities, "ood_probabilities", allow_empty=True)
    if ids.shape[1] != oods.shape[1]:
        raise ValueError("ID and OOD probabilities must have matching class counts")

    results: dict[str, dict[str, Any]] = {
        "msp": _run_one(
            "msp", msp_score, ids, oods,
            inference_latency_ms_per_image=inference_latency_ms_per_image,
        ),
        "entropy": _run_one(
            "entropy", entropy_ood_score, ids, oods,
            inference_latency_ms_per_image=inference_latency_ms_per_image,
        ),
    }

    def matrix_pair(
        id_values: np.ndarray | None,
        ood_values: np.ndarray | None,
        name: str,
    ) -> tuple[np.ndarray | None, np.ndarray | None, str | None]:
        if id_values is None or ood_values is None:
            return None, None, f"{name} requires explicit ID and OOD raw inputs."
        left = _finite_matrix(id_values, f"id_{name}")
        right = _finite_matrix(ood_values, f"ood_{name}", allow_empty=True)
        if left.shape[1] != right.shape[1]:
            raise ValueError(f"ID and OOD {name} dimensions must match")
        return left, right, None

    if id_logits is None or ood_logits is None:
        results["energy"] = _unavailable(
            "energy",
            reason="Energy requires pre-softmax ID and OOD logits; probabilities are not a valid substitute.",
            id_count=ids.shape[0],
            ood_count=oods.shape[0],
        )
    else:
        logit_ids, logit_oods, reason = matrix_pair(id_logits, ood_logits, "logits")
        if reason:
            results["energy"] = _unavailable(
                "energy", reason=reason, id_count=ids.shape[0], ood_count=oods.shape[0]
            )
        else:
            assert logit_ids is not None and logit_oods is not None
            results["energy"] = _run_one(
                "energy",
                lambda values: energy_score(values, temperature=energy_temperature),
                logit_ids,
                logit_oods,
                inference_latency_ms_per_image=inference_latency_ms_per_image,
            )

    embedding_ids, embedding_oods, embedding_reason = matrix_pair(
        id_embeddings, ood_embeddings, "embeddings"
    )
    if embedding_reason:
        for name in ("cosine", "mahalanobis"):
            results[name] = _unavailable(
                name, reason=embedding_reason, id_count=ids.shape[0], ood_count=oods.shape[0]
            )
    else:
        assert embedding_ids is not None and embedding_oods is not None
        if centroids is None:
            results["cosine"] = _unavailable(
                "cosine",
                reason="Cosine distance requires class centroid embeddings.",
                id_count=ids.shape[0],
                ood_count=oods.shape[0],
            )
        else:
            centroid_values = _finite_matrix(centroids, "centroids")
            results["cosine"] = _run_one(
                "cosine",
                lambda values: cosine_distance(values, centroid_values),
                embedding_ids,
                embedding_oods,
                inference_latency_ms_per_image=inference_latency_ms_per_image,
            )

        mean_values: np.ndarray | None = None
        covariance_values: np.ndarray | None = None
        if mahalanobis_mean is not None and mahalanobis_covariance is not None:
            mean_values = np.asarray(mahalanobis_mean, dtype=np.float64).reshape(-1)
            covariance_values = _finite_matrix(mahalanobis_covariance, "mahalanobis_covariance")
        if mean_values is None or covariance_values is None:
            results["mahalanobis"] = _unavailable(
                "mahalanobis",
                reason="Mahalanobis requires explicit mean and covariance artifacts.",
                id_count=ids.shape[0],
                ood_count=oods.shape[0],
            )
        else:
            if covariance_values.shape[0] != embedding_ids.shape[1] or mean_values.size != embedding_ids.shape[1]:
                raise ValueError("Mahalanobis statistics do not match embedding dimensions")
            results["mahalanobis"] = _run_one(
                "mahalanobis",
                lambda values: mahalanobis_distance(
                    values, mean_values, covariance_values, regularization=0.0
                ),
                embedding_ids,
                embedding_oods,
                inference_latency_ms_per_image=inference_latency_ms_per_image,
            )

    return {name: results[name] for name in SUPPORTED_METHODS}


def _empty_runtime() -> dict[str, Any]:
    return {
        "status": "not_measured",
        "inference_latency_ms_per_image": None,
        "id_inference_count": 0,
        "ood_inference_count": 0,
        "inference_scope": "one combined read-only model call returning probabilities, logits, and embeddings",
    }


def _artifact(
    results: Mapping[str, Mapping[str, Any]],
    *,
    protocol: Mapping[str, Any],
    runtime: Mapping[str, Any] | None = None,
    source: str,
    status: str,
) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "task": "TASK 16 — OOD BENCHMARK",
        "status": status,
        "evaluation_date": date.today().isoformat(),
        "source": source,
        "score_orientation": "larger values mean more likely OOD; MSP is oriented internally by negating raw confidence",
        "protocol": dict(protocol),
        "metrics": {name: dict(results[name]) for name in SUPPORTED_METHODS},
        "runtime": dict(runtime or _empty_runtime()),
        "production_implementation_changed": False,
        "production_recommendation_applied": False,
    }


def _format_metric(value: Any) -> str:
    return "—" if value is None else f"{float(value):.4f}"


def _write_markdown(path: Path, artifact: Mapping[str, Any]) -> None:
    protocol = artifact["protocol"]
    metrics = artifact["metrics"]
    runtime = artifact["runtime"]
    lines = [
        "# OOD Benchmark",
        "",
        f"- **Status:** `{artifact['status']}`",
        f"- **Evaluation date:** `{artifact['evaluation_date']}`",
        f"- **ID protocol:** `{protocol['id_source']}` ({protocol['id_count']} samples)",
        f"- **OOD protocol:** `{protocol['ood_source']}` ({protocol['ood_count']} samples)",
        "- **Operating point:** FPR95 threshold on the same ID/OOD evaluation pool; OOD F1, FAR, and FRR are reported at that point.",
        "- **Production safety:** this artifact is analysis-only; production inference and thresholds were not changed.",
        "",
        "## Results",
        "",
        "A dash means the method could not be evaluated honestly because a required raw input or OOD sample was unavailable.",
        "",
        "| Method | AUROC | FPR95 | AUPR-IN | AUPR-OUT | OOD F1 | FAR | FRR | Latency (ms/image) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in SUPPORTED_METHODS:
        row = metrics[name]
        lines.append(
            "| {name} | {auroc} | {fpr95} | {aupr_in} | {aupr_out} | {ood_f1} | {far} | {frr} | {latency} |".format(
                name=name,
                auroc=_format_metric(row.get("auroc")),
                fpr95=_format_metric(row.get("fpr95")),
                aupr_in=_format_metric(row.get("aupr_in")),
                aupr_out=_format_metric(row.get("aupr_out")),
                ood_f1=_format_metric(row.get("ood_f1")),
                far=_format_metric(row.get("false_acceptance_rate")),
                frr=_format_metric(row.get("false_rejection_rate")),
                latency=_format_metric(row.get("latency_ms_per_image")),
            )
        )
    lines.extend(
        [
            "",
            "## Latency",
            "",
            f"The common model-output pass was measured as `{_format_metric(runtime.get('inference_latency_ms_per_image'))}` ms/image over `{runtime.get('id_inference_count', 0) + runtime.get('ood_inference_count', 0)}` samples. Per-method latency adds the detector score calculation; it is not a production routing decision.",
            "",
            "## Method trade-offs",
            "",
            "| Method | Strengths | Costs / failure modes |",
            "|---|---|---|",
            "| MSP | Lowest implementation and runtime cost; directly available from the existing softmax output; easy to audit. | Often overconfident on near-OOD inputs; strongly correlated with entropy; no representation-level signal. |",
            "| Entropy | Uses the full class distribution and remains probability-only; cheap after the classifier forward pass. | Usually adds limited information beyond MSP; sensitive to probability calibration and class-count effects. |",
            "| Energy | Uses pre-softmax evidence and can separate low-logit OOD more effectively than softmax-only scores. | Requires a logits adapter and temperature selection; temperature changes score scale; extra adapter complexity. |",
            "| Cosine distance | Uses semantic embedding geometry and class centroids; interpretable nearest-class distance; useful for visually similar OOD. | Requires centroid artifacts and an embedding forward pass; centroid quality and feature normalization matter. |",
            "| Mahalanobis distance | Models feature covariance and can account for correlated directions; potentially strongest on structured near-OOD. | Highest statistical and memory complexity; covariance can be ill-conditioned; fit must be training-only and requires careful regularization. |",
            "",
            "## Technical recommendation",
            "",
        ]
    )
    if artifact["status"] != "evaluated":
        lines.extend(
            [
                "No method can be selected from benchmark results because the OOD evaluation pool is empty or incomplete. Supply a disjoint OOD calibration/test set and rerun this benchmark before changing production.",
                "",
                "Provisional recommendation: keep the existing MSP production policy unchanged. Treat Energy as the next low-risk candidate to compare when logits are available; consider Cosine or Mahalanobis only after verifying embedding artifacts, covariance provenance, and latency against representative near-OOD data.",
            ]
        )
    else:
        lines.extend(
            [
                "Select the method by the required operating trade-off, not AUROC alone: prioritize low FAR at an acceptable FRR, then use OOD F1/AUPR and latency as tie-breakers. Keep MSP if its separation is Pareto-competitive; otherwise Energy is the simplest logits-based upgrade, while Cosine/Mahalanobis are candidates when near-OOD representation geometry matters.",
                "",
                "No production method or threshold was changed automatically. Any adoption requires a separate review of the benchmark data, threshold calibration, and end-to-end latency.",
            ]
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _load_npz(path: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    with np.load(path, allow_pickle=False) as data:
        arrays = {key: np.asarray(data[key]) for key in data.files}
    required = {"id_probabilities", "ood_probabilities"}
    missing = sorted(required - arrays.keys())
    if missing:
        raise ValueError(f"score artifact is missing required arrays: {', '.join(missing)}")
    return arrays, {
        "id_source": str(path) + ":id_probabilities",
        "ood_source": str(path) + ":ood_probabilities",
        "id_count": int(arrays["id_probabilities"].shape[0]),
        "ood_count": int(arrays["ood_probabilities"].shape[0]),
    }


def benchmark_npz(
    scores_path: str | Path,
    *,
    energy_temperature: float = DEFAULT_TEMPERATURE,
) -> dict[str, Any]:
    """Benchmark a score artifact with all available raw tensors."""

    path = Path(scores_path).resolve()
    arrays, counts = _load_npz(path)
    started = time.perf_counter()
    results = benchmark_methods(
        arrays["id_probabilities"],
        arrays["ood_probabilities"],
        id_logits=arrays.get("id_logits"),
        ood_logits=arrays.get("ood_logits"),
        id_embeddings=arrays.get("id_embeddings"),
        ood_embeddings=arrays.get("ood_embeddings"),
        centroids=arrays.get("centroids"),
        mahalanobis_mean=arrays.get("mahalanobis_mean"),
        mahalanobis_covariance=arrays.get("mahalanobis_covariance"),
        energy_temperature=energy_temperature,
    )
    elapsed = (time.perf_counter() - started) * 1000.0
    for row in results.values():
        row["inference_latency_ms_per_image"] = None
        row["latency_ms_per_image"] = row.get("score_latency_ms_per_image")
    return _artifact(
        results,
        protocol={
            "id_source": counts["id_source"],
            "ood_source": counts["ood_source"],
            "id_count": counts["id_count"],
            "ood_count": counts["ood_count"],
            "threshold_protocol": "FPR95 operating point from the same evaluation pool",
        },
        runtime={
            "status": "score_artifact_only",
            "inference_latency_ms_per_image": None,
            "score_artifact_processing_ms": float(elapsed),
            "id_inference_count": 0,
            "ood_inference_count": 0,
            "inference_scope": "not measured; input contains precomputed model outputs",
        },
        source="npz_score_artifact",
        status="evaluated" if counts["ood_count"] > 0 else "incomplete_missing_ood_data",
    )


def _tensorflow() -> Any:
    try:
        import tensorflow as tf
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("TensorFlow is required for image-based OOD benchmarking") from exc
    return tf


def _discover_images(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    return sorted(
        path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in _IMAGE_EXTENSIONS
    )


def _ood_dataset(root: Path, batch_size: int) -> tuple[Any | None, int]:
    tf = _tensorflow()
    paths = _discover_images(root)
    if not paths:
        return None, 0

    from .image_loading import preprocess_ood_image

    dataset = tf.data.Dataset.from_tensor_slices([str(path) for path in paths])
    dataset = dataset.map(preprocess_ood_image, num_parallel_calls=tf.data.AUTOTUNE)
    return dataset.batch(batch_size, drop_remainder=False).prefetch(tf.data.AUTOTUNE), len(paths)


def _collect_combined(
    combined_model: Any,
    dataset: Any,
    *,
    count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    probabilities: list[np.ndarray] = []
    logits: list[np.ndarray] = []
    embeddings: list[np.ndarray] = []
    elapsed = 0.0
    for batch in dataset:
        images = batch[0] if isinstance(batch, (tuple, list)) else batch
        started = time.perf_counter()
        outputs = combined_model(images, training=False)
        elapsed += (time.perf_counter() - started) * 1000.0
        values = list(outputs) if isinstance(outputs, (tuple, list)) else [outputs]
        if len(values) != 3:
            raise RuntimeError("combined benchmark model must return probabilities, logits, and embeddings")
        probabilities.append(np.asarray(values[0], dtype=np.float64))
        logits.append(np.asarray(values[1], dtype=np.float64))
        embeddings.append(np.asarray(values[2], dtype=np.float64))
    if not probabilities:
        raise ValueError("cannot benchmark an empty ID dataset")
    probability_values = np.concatenate(probabilities, axis=0)
    logit_values = np.concatenate(logits, axis=0)
    embedding_values = np.concatenate(embeddings, axis=0)
    if probability_values.shape[0] != count:
        raise RuntimeError("model output count does not match dataset count")
    return probability_values, logit_values, embedding_values, elapsed


def benchmark_image_sets(
    *,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    id_split: str = DEFAULT_ID_SPLIT,
    ood_root: str | Path = DEFAULT_OOD_ROOT,
    batch_size: int = 64,
    energy_temperature: float = DEFAULT_TEMPERATURE,
    centroids_path: str | Path = DEFAULT_CENTROIDS_PATH,
    centroid_metadata_path: str | Path = DEFAULT_CENTROID_METADATA_PATH,
    class_names_path: str | Path = DEFAULT_CLASS_NAMES_PATH,
    mahalanobis_mean_path: str | Path = DEFAULT_MAHALANOBIS_MEAN_PATH,
    mahalanobis_covariance_path: str | Path = DEFAULT_MAHALANOBIS_COVARIANCE_PATH,
) -> dict[str, Any]:
    """Collect one read-only model pass and run the common image protocol."""

    # Do not initialize TensorFlow when the required OOD pool is absent. This
    # keeps the missing-data result deterministic and makes the CLI useful in
    # dataset-audit environments where only the ID split is present.
    ood_directory = Path(ood_root).resolve()
    ood_count = len(_discover_images(ood_directory))
    if ood_count == 0:
        split_path = PROJECT_ROOT / "outputs" / "splits" / f"{id_split}.csv"
        if not split_path.is_file():
            raise FileNotFoundError(f"Split file not found: {split_path}")
        with split_path.open("r", encoding="utf-8", newline="") as handle:
            id_count = max(sum(1 for _ in csv.reader(handle)) - 1, 0)
        results = {
            name: _unavailable(
                name,
                reason="No OOD samples were available for the common evaluation protocol.",
                id_count=id_count,
                ood_count=0,
            )
            for name in SUPPORTED_METHODS
        }
        return _artifact(
            results,
            protocol={
                "id_source": f"outputs/splits/{id_split}.csv",
                "ood_source": str(ood_directory),
                "id_count": int(id_count),
                "ood_count": 0,
                "threshold_protocol": "FPR95 operating point from the same evaluation pool",
                "preprocessing": "existing dataset pipeline; raw pixels [0,255], model-owned MobileNetV2 preprocessing",
                "energy_temperature": float(energy_temperature),
                "centroids_source": str(Path(centroids_path).resolve()),
                "mahalanobis_statistics_source": "not evaluated because the OOD pool is empty",
            },
            runtime={
                "status": "not_measured_no_ood_data",
                "inference_latency_ms_per_image": None,
                "id_inference_count": 0,
                "ood_inference_count": 0,
                "inference_scope": "one combined read-only model call returning probabilities, logits, and embeddings",
                "reason": "No OOD image was present; model inference was intentionally skipped.",
            },
            source="read_only_model_image_inference",
            status="incomplete_missing_ood_data",
        )

    tf = _tensorflow()
    from src.dataset_pipeline import create_dataset, load_split_csv
    from src.label_encoder import fit_label_encoder
    from src.models.feature_extractor import build_feature_extractor
    from .energy import build_logits_model

    model_file = Path(model_path).resolve()
    tf.keras.utils.get_custom_objects()["preprocess_input"] = (
        tf.keras.applications.mobilenet_v2.preprocess_input
    )
    model = tf.keras.models.load_model(str(model_file), compile=False, safe_mode=True)
    logits_model = build_logits_model(model)
    feature_model = build_feature_extractor(model)
    combined_model = tf.keras.Model(
        inputs=model.inputs,
        outputs=[model.output, logits_model.output, feature_model.output],
        name="ood_benchmark_read_only_outputs",
    )

    id_frame = load_split_csv(id_split)
    encoder = fit_label_encoder()
    id_dataset = create_dataset(id_frame, encoder, batch_size=batch_size, training=False)
    id_probabilities, id_logits, id_embeddings, id_elapsed = _collect_combined(
        combined_model, id_dataset, count=len(id_frame)
    )

    ood_dataset, ood_count = _ood_dataset(ood_directory, batch_size)
    if ood_dataset is None:
        ood_probabilities = np.empty((0, id_probabilities.shape[1]), dtype=np.float64)
        ood_logits = np.empty((0, id_logits.shape[1]), dtype=np.float64)
        ood_embeddings = np.empty((0, id_embeddings.shape[1]), dtype=np.float64)
        ood_elapsed = 0.0
    else:
        ood_probabilities, ood_logits, ood_embeddings, ood_elapsed = _collect_combined(
            combined_model, ood_dataset, count=ood_count
        )

    centroids = None
    centroid_source = None
    try:
        centroids, _, _ = load_centroid_artifacts(
            centroids_path, centroid_metadata_path, class_names_path
        )
        centroid_source = str(Path(centroids_path).resolve())
    except (FileNotFoundError, ValueError) as exc:
        centroid_source = f"unavailable: {exc}"

    maha_mean = None
    maha_covariance = None
    maha_source = None
    mean_path = Path(mahalanobis_mean_path)
    covariance_path = Path(mahalanobis_covariance_path)
    if mean_path.is_file() and covariance_path.is_file():
        maha_mean = np.load(mean_path, allow_pickle=False)
        maha_covariance = np.load(covariance_path, allow_pickle=False)
        maha_source = f"{mean_path.resolve()}, {covariance_path.resolve()}"
    else:
        maha_source = "not persisted; benchmark image mode does not fit from ID validation"

    total_count = id_frame.shape[0] + ood_count
    inference_latency = (id_elapsed + ood_elapsed) / max(total_count, 1)
    results = benchmark_methods(
        id_probabilities,
        ood_probabilities,
        id_logits=id_logits,
        ood_logits=ood_logits,
        id_embeddings=id_embeddings,
        ood_embeddings=ood_embeddings,
        centroids=centroids,
        mahalanobis_mean=maha_mean,
        mahalanobis_covariance=maha_covariance,
        energy_temperature=energy_temperature,
        inference_latency_ms_per_image=inference_latency,
    )
    return _artifact(
        results,
        protocol={
            "id_source": f"outputs/splits/{id_split}.csv",
            "ood_source": str(ood_directory),
            "id_count": int(id_frame.shape[0]),
            "ood_count": int(ood_count),
            "threshold_protocol": "FPR95 operating point from the same evaluation pool",
            "preprocessing": "existing dataset pipeline; raw pixels [0,255], model-owned MobileNetV2 preprocessing",
            "energy_temperature": float(energy_temperature),
            "centroids_source": centroid_source,
            "mahalanobis_statistics_source": maha_source,
        },
        runtime={
            "status": "measured",
            "inference_latency_ms_per_image": float(inference_latency),
            "id_inference_count": int(id_frame.shape[0]),
            "ood_inference_count": int(ood_count),
            "inference_scope": "one combined read-only model call returning probabilities, logits, and embeddings",
        },
        source="read_only_model_image_inference",
        status="evaluated" if ood_count > 0 else "incomplete_missing_ood_data",
    )


def write_artifacts(
    artifact: Mapping[str, Any],
    *,
    output_json: str | Path = DEFAULT_OUTPUT_JSON,
    output_markdown: str | Path = DEFAULT_OUTPUT_MARKDOWN,
) -> None:
    json_path = Path(output_json)
    markdown_path = Path(output_markdown)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    _write_markdown(markdown_path, artifact)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scores", nargs="?", type=Path, help="optional NPZ with precomputed model outputs")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--id-split", choices=("train", "val", "test"), default=DEFAULT_ID_SPLIT)
    parser.add_argument("--ood-root", type=Path, default=DEFAULT_OOD_ROOT)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--energy-temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-markdown", type=Path, default=DEFAULT_OUTPUT_MARKDOWN)
    parser.add_argument("--centroids-path", type=Path, default=DEFAULT_CENTROIDS_PATH)
    parser.add_argument("--centroid-metadata-path", type=Path, default=DEFAULT_CENTROID_METADATA_PATH)
    parser.add_argument("--class-names-path", type=Path, default=DEFAULT_CLASS_NAMES_PATH)
    parser.add_argument("--mahalanobis-mean-path", type=Path, default=DEFAULT_MAHALANOBIS_MEAN_PATH)
    parser.add_argument("--mahalanobis-covariance-path", type=Path, default=DEFAULT_MAHALANOBIS_COVARIANCE_PATH)
    args = parser.parse_args()

    if args.scores is not None:
        artifact = benchmark_npz(args.scores, energy_temperature=args.energy_temperature)
    else:
        artifact = benchmark_image_sets(
            model_path=args.model_path,
            id_split=args.id_split,
            ood_root=args.ood_root,
            batch_size=args.batch_size,
            energy_temperature=args.energy_temperature,
            centroids_path=args.centroids_path,
            centroid_metadata_path=args.centroid_metadata_path,
            class_names_path=args.class_names_path,
            mahalanobis_mean_path=args.mahalanobis_mean_path,
            mahalanobis_covariance_path=args.mahalanobis_covariance_path,
        )
    write_artifacts(artifact, output_json=args.output_json, output_markdown=args.output_markdown)
    print(json.dumps(artifact, indent=2))


if __name__ == "__main__":
    main()
