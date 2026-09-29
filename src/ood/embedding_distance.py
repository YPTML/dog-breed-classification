"""Embedding-distance OOD scoring and Mahalanobis calibration.

The detector treats the distance to the closest class centroid as an OOD
score. A larger score means that an embedding is less similar to every known
class. It also provides squared Mahalanobis scoring and the offline
calibration entry point for TASK 13::

    python -m src.ood.embedding_distance --method mahalanobis

Mahalanobis covariance and mean are fit from the training split only. The
covariance is symmetrized, stabilized with a scale-aware diagonal ridge, and
scored with a Moore--Penrose pseudo-inverse so singular and ill-conditioned
matrices remain safe. ID validation embeddings and images in
``data/ood/calibration`` are evaluated separately. An empty OOD directory is
represented explicitly as ``missing_ood_calibration_data`` rather than being
treated as a successful separation experiment. The legacy cosine evaluator
remains available with ``--method cosine``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "outputs" / "deployment_ready" / "dog_breed_model.keras"
DEFAULT_CENTROIDS_PATH = PROJECT_ROOT / "outputs" / "calibration" / "class_centroids.npy"
DEFAULT_CENTROID_METADATA_PATH = PROJECT_ROOT / "outputs" / "calibration" / "class_centroids.json"
DEFAULT_CLASS_NAMES_PATH = PROJECT_ROOT / "outputs" / "deployment_ready" / "class_names.json"
DEFAULT_OOD_CALIBRATION_DIR = PROJECT_ROOT / "data" / "ood" / "calibration"
DEFAULT_METRICS_PATH = PROJECT_ROOT / "outputs" / "metrics" / "cosine_ood.json"
DEFAULT_PLOT_PATH = PROJECT_ROOT / "outputs" / "plots" / "cosine_distance_distribution.png"
DEFAULT_COVARIANCE_PATH = PROJECT_ROOT / "outputs" / "calibration" / "covariance.npy"
DEFAULT_MAHALANOBIS_MEAN_PATH = PROJECT_ROOT / "outputs" / "calibration" / "mahalanobis_mean.npy"
DEFAULT_MAHALANOBIS_METADATA_PATH = PROJECT_ROOT / "outputs" / "calibration" / "mahalanobis.json"
DEFAULT_MAHALANOBIS_METRICS_PATH = PROJECT_ROOT / "outputs" / "metrics" / "mahalanobis_ood.json"
DEFAULT_MAHALANOBIS_REGULARIZATION = 1e-6
DEFAULT_MAHALANOBIS_PINV_RCOND = 1e-12
_IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".webp", ".bmp"})


def _batch(features: np.ndarray, name: str = "features") -> np.ndarray:
    """Return a validated finite matrix, accepting one embedding as 1-D."""

    values = np.asarray(features, dtype=np.float64)
    if values.ndim == 1:
        values = values[None, :]
    if (
        values.ndim != 2
        or values.shape[0] == 0
        or values.shape[1] == 0
        or not np.isfinite(values).all()
    ):
        raise ValueError(f"{name} must have shape (batch, dimensions) and be finite")
    return values


def _validate_centroid_matrix(centroids: np.ndarray) -> np.ndarray:
    return _batch(centroids, "centroids")


def cosine_similarity_matrix(features: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """Return pairwise cosine similarities between embeddings and centroids.

    Zero-norm rows are assigned similarity zero. This keeps scoring finite and
    gives such an embedding the neutral cosine distance of one.
    """

    values = _batch(features)
    centers = _validate_centroid_matrix(centroids)
    if values.shape[1] != centers.shape[1]:
        raise ValueError("feature and centroid dimensions do not match")

    feature_norm = np.linalg.norm(values, axis=1, keepdims=True)
    center_norm = np.linalg.norm(centers, axis=1, keepdims=True).T
    denominator = feature_norm * center_norm
    similarity = np.divide(
        values @ centers.T,
        denominator,
        out=np.zeros((values.shape[0], centers.shape[0]), dtype=np.float64),
        where=denominator > 0,
    )
    return np.clip(similarity, -1.0, 1.0)


def nearest_cosine_centroids(
    features: np.ndarray, centroids: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(nearest_centroid_index, nearest_cosine_distance)`` per row."""

    similarities = cosine_similarity_matrix(features, centroids)
    nearest_index = np.argmax(similarities, axis=1).astype(np.int64)
    nearest_similarity = similarities[np.arange(similarities.shape[0]), nearest_index]
    return nearest_index, 1.0 - nearest_similarity


def cosine_distance(features: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """Return ``1 - max cosine similarity`` to the closest centroid."""

    return nearest_cosine_centroids(features, centroids)[1]


def score_embeddings(
    features: np.ndarray,
    centroids: np.ndarray,
    class_names: Sequence[str] | None = None,
) -> dict[str, np.ndarray]:
    """Return per-sample ``nearest_class``, ``nearest_distance``, and ``ood_score``.

    ``nearest_class`` contains integer centroid indices unless ``class_names``
    is supplied, in which case it contains the corresponding class labels.
    ``nearest_distance`` and ``ood_score`` are identical by definition.
    """

    nearest_index, nearest_distance = nearest_cosine_centroids(features, centroids)
    if class_names is None:
        nearest_class: np.ndarray = nearest_index
    else:
        names = _validate_class_names(class_names, _validate_centroid_matrix(centroids).shape[0])
        nearest_class = np.asarray([names[index] for index in nearest_index], dtype=object)
    return {
        "nearest_class": nearest_class,
        "nearest_distance": nearest_distance,
        "ood_score": nearest_distance.copy(),
    }


def score_embedding(
    feature: np.ndarray,
    centroids: np.ndarray,
    class_names: Sequence[str] | None = None,
) -> dict[str, int | str | float]:
    """Score one embedding using the public ``nearest_*`` output contract."""

    values = score_embeddings(feature, centroids, class_names=class_names)
    nearest_class = values["nearest_class"][0]
    if isinstance(nearest_class, np.generic):
        nearest_class = nearest_class.item()
    return {
        "nearest_class": nearest_class,
        "nearest_distance": float(values["nearest_distance"][0]),
        "ood_score": float(values["ood_score"][0]),
    }


def euclidean_distance(features: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """Return Euclidean distance to the closest centroid."""

    values = _batch(features)
    centers = _validate_centroid_matrix(centroids)
    if values.shape[1] != centers.shape[1]:
        raise ValueError("feature and centroid dimensions do not match")
    distances = np.linalg.norm(values[:, None, :] - centers[None, :, :], axis=2)
    return np.min(distances, axis=1)


def _regularize_covariance_with_ridge(
    covariance: np.ndarray,
    regularization: float = DEFAULT_MAHALANOBIS_REGULARIZATION,
) -> tuple[np.ndarray, float]:
    """Symmetrize covariance and add a scale-aware diagonal ridge.

    The applied ridge is ``regularization * max(mean(diag(covariance)), 1)``.
    Scaling prevents the same setting from being either ineffective or
    dominant when embedding magnitudes change. A zero ridge is supported
    because :func:`mahalanobis_distance` still uses ``pinv`` for singular
    matrices.
    """

    cov = np.asarray(covariance, dtype=np.float64)
    if cov.ndim != 2 or cov.shape[0] == 0 or cov.shape[0] != cov.shape[1]:
        raise ValueError("covariance must be a non-empty square matrix")
    if not np.isfinite(cov).all():
        raise ValueError("covariance must be finite")
    if not np.isfinite(regularization) or regularization < 0:
        raise ValueError("regularization must be non-negative and finite")
    cov = (cov + cov.T) / 2.0
    mean_variance = float(np.mean(np.diag(cov)))
    if mean_variance < 0 or not np.isfinite(mean_variance):
        raise ValueError("covariance diagonal must be non-negative")
    ridge = float(regularization) * max(mean_variance, 1.0)
    return cov + np.eye(cov.shape[0], dtype=np.float64) * ridge, ridge


def regularize_covariance(
    covariance: np.ndarray,
    regularization: float = DEFAULT_MAHALANOBIS_REGULARIZATION,
) -> np.ndarray:
    """Return the symmetric, ridge-regularized covariance matrix."""

    result, _ = _regularize_covariance_with_ridge(covariance, regularization)
    return result


def _fit_mahalanobis_details(
    features: np.ndarray,
    *,
    regularization: float = DEFAULT_MAHALANOBIS_REGULARIZATION,
) -> dict[str, Any]:
    """Fit mean/covariance using only the supplied training embeddings.

    The covariance is the unbiased sample covariance (``ddof=1``). The
    returned ``covariance`` is the symmetrized, ridge-regularized matrix that
    should be persisted and passed to the scorer with ``regularization=0``.
    """

    values = _batch(features, "training_embeddings")
    mean = np.mean(values, axis=0, dtype=np.float64)
    centered = values - mean
    denominator = max(values.shape[0] - 1, 1)
    raw_covariance = (centered.T @ centered) / float(denominator)
    return _fit_mahalanobis_from_moments(
        mean,
        raw_covariance,
        sample_count=values.shape[0],
        regularization=regularization,
    )


def _fit_mahalanobis_from_moments(
    mean: np.ndarray,
    raw_covariance: np.ndarray,
    *,
    sample_count: int,
    regularization: float,
) -> dict[str, Any]:
    """Finalize fit metadata from a mean and unbiased raw covariance."""

    mean = np.asarray(mean, dtype=np.float64).reshape(-1)
    raw_covariance = np.asarray(raw_covariance, dtype=np.float64)
    if sample_count <= 0 or raw_covariance.shape != (mean.size, mean.size):
        raise ValueError("invalid covariance moments")
    raw_covariance = (raw_covariance + raw_covariance.T) / 2.0
    covariance, ridge = _regularize_covariance_with_ridge(
        raw_covariance, regularization=regularization
    )
    eigenvalues = np.linalg.eigvalsh(covariance)
    condition_number = float(np.linalg.cond(covariance))
    return {
        "mean": mean,
        "raw_covariance": raw_covariance,
        "covariance": covariance,
        "sample_count": int(sample_count),
        "embedding_dimension": int(mean.size),
        "regularization": float(regularization),
        "ridge_added": float(ridge),
        "min_eigenvalue": float(np.min(eigenvalues)),
        "max_eigenvalue": float(np.max(eigenvalues)),
        "condition_number": condition_number if np.isfinite(condition_number) else None,
    }


def fit_mahalanobis_statistics(
    features: np.ndarray,
    *,
    regularization: float = DEFAULT_MAHALANOBIS_REGULARIZATION,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(training_mean, regularized_covariance)``.

    This compact API is useful for callers that only need the two numerical
    artifacts. The internal calibration runner also records the additional
    provenance returned by ``_fit_mahalanobis_details``.
    """

    details = _fit_mahalanobis_details(
        features, regularization=regularization
    )
    return details["mean"], details["covariance"]


def mahalanobis_distance(
    features: np.ndarray,
    mean: np.ndarray,
    covariance: np.ndarray,
    *,
    regularization: float = DEFAULT_MAHALANOBIS_REGULARIZATION,
    pinv_rcond: float = DEFAULT_MAHALANOBIS_PINV_RCOND,
) -> np.ndarray:
    """Return squared Mahalanobis distance with singular-covariance handling.

    ``mean`` may be one global mean with shape ``(D,)`` or several means with
    shape ``(K, D)``. For multiple means this returns the minimum distance to
    any mean. The covariance is symmetrized, ridge-regularized, then inverted
    with a Moore--Penrose pseudo-inverse, so singular matrices remain usable.
    """

    values = _batch(features)
    center = np.asarray(mean, dtype=np.float64)
    if center.ndim == 1:
        center = center[None, :]
    cov = np.asarray(covariance, dtype=np.float64)
    if center.ndim != 2 or center.shape[0] == 0 or center.shape[1] != values.shape[1]:
        raise ValueError("mean/covariance dimensions do not match features")
    if cov.shape != (values.shape[1], values.shape[1]):
        raise ValueError("mean/covariance dimensions do not match features")
    if not np.isfinite(center).all() or not np.isfinite(cov).all():
        raise ValueError("mean and covariance must be finite")
    if regularization < 0 or not np.isfinite(regularization):
        raise ValueError("regularization must be non-negative and finite")
    cov = regularize_covariance(cov, regularization=regularization)
    if pinv_rcond < 0 or not np.isfinite(pinv_rcond):
        raise ValueError("pinv_rcond must be non-negative and finite")
    try:
        inverse = np.linalg.pinv(cov, rcond=float(pinv_rcond), hermitian=True)
    except TypeError:  # NumPy versions without hermitian on pinv.
        inverse = np.linalg.pinv(cov, rcond=float(pinv_rcond))
    delta = values[:, None, :] - center[None, :, :]
    distances = np.einsum("bki,ij,bkj->bk", delta, inverse, delta, optimize=True)
    return np.maximum(np.min(distances, axis=1), 0.0)


class EmbeddingDistanceDetector:
    """Small callable detector around persisted centroids/covariance."""

    def __init__(
        self,
        *,
        method: str,
        centroids: np.ndarray | None = None,
        mean: np.ndarray | None = None,
        covariance: np.ndarray | None = None,
        regularization: float = DEFAULT_MAHALANOBIS_REGULARIZATION,
        pinv_rcond: float = DEFAULT_MAHALANOBIS_PINV_RCOND,
    ):
        method = method.lower().strip()
        if method not in {"cosine", "euclidean", "mahalanobis"}:
            raise ValueError("method must be cosine, euclidean, or mahalanobis")
        self.method = method
        self.centroids = None if centroids is None else _validate_centroid_matrix(centroids)
        self.mean = None if mean is None else np.asarray(mean, dtype=np.float64)
        self.covariance = None if covariance is None else np.asarray(covariance, dtype=np.float64)
        self.regularization = float(regularization)
        self.pinv_rcond = float(pinv_rcond)

    def score(self, features: np.ndarray) -> np.ndarray:
        if self.method == "cosine":
            if self.centroids is None:
                raise ValueError("cosine detector requires centroids")
            return cosine_distance(features, self.centroids)
        if self.method == "euclidean":
            if self.centroids is None:
                raise ValueError("euclidean detector requires centroids")
            return euclidean_distance(features, self.centroids)
        if self.mean is None or self.covariance is None:
            raise ValueError("mahalanobis detector requires mean and covariance")
        return mahalanobis_distance(
            features,
            self.mean,
            self.covariance,
            regularization=self.regularization,
            pinv_rcond=self.pinv_rcond,
        )

    __call__ = score


cosine_ood_score = cosine_distance
euclidean_ood_score = euclidean_distance
mahalanobis_ood_score = mahalanobis_distance


def _validate_class_names(class_names: Sequence[str], expected_count: int) -> list[str]:
    names = [str(name) for name in class_names]
    if (
        len(names) != expected_count
        or any(not name.strip() for name in names)
        or len(set(names)) != len(names)
    ):
        raise ValueError(f"expected {expected_count} unique non-empty class names")
    return names


def _load_json_object(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact must contain an object: {path}")
    return value


def _load_class_names(path: Path) -> list[str]:
    with path.open(encoding="utf-8") as handle:
        value: Any = json.load(handle)
    if isinstance(value, dict):
        try:
            items = sorted(value.items(), key=lambda item: int(item[0]))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"class names must use numeric keys: {path}") from exc
        names = [item[1] for item in items]
    elif isinstance(value, list):
        names = value
    else:
        raise ValueError(f"class names must be a JSON object or array: {path}")
    return names


def load_centroid_artifacts(
    centroids_path: str | Path = DEFAULT_CENTROIDS_PATH,
    metadata_path: str | Path = DEFAULT_CENTROID_METADATA_PATH,
    class_names_path: str | Path = DEFAULT_CLASS_NAMES_PATH,
) -> tuple[np.ndarray, list[str], dict[str, Any]]:
    """Load centroids and class order without modifying either artifact."""

    array_path = Path(centroids_path)
    if not array_path.is_file():
        raise FileNotFoundError(f"Centroid artifact not found: {array_path}")
    centroids = _validate_centroid_matrix(np.load(array_path, allow_pickle=False))

    metadata_file = Path(metadata_path)
    metadata: dict[str, Any] = {}
    class_names_value: Any = None
    if metadata_file.is_file():
        metadata = _load_json_object(metadata_file)
        class_names_value = metadata.get("class_order")

    if class_names_value is None:
        names_file = Path(class_names_path)
        if not names_file.is_file():
            raise FileNotFoundError(
                f"Centroid metadata has no class_order and class names were not found: {names_file}"
            )
        class_names_value = _load_class_names(names_file)
    if not isinstance(class_names_value, list):
        raise ValueError("centroid class_order must be a list")
    class_names = _validate_class_names(class_names_value, centroids.shape[0])

    expected_dimension = metadata.get("embedding_dimension")
    if expected_dimension is not None and int(expected_dimension) != centroids.shape[1]:
        raise ValueError("centroid metadata embedding dimension does not match the array")
    expected_classes = metadata.get("num_classes")
    if expected_classes is not None and int(expected_classes) != centroids.shape[0]:
        raise ValueError("centroid metadata class count does not match the array")
    return centroids, class_names, metadata


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


def _class_counts(classes: np.ndarray, class_names: Sequence[str]) -> dict[str, int]:
    values = [str(value) for value in np.asarray(classes).reshape(-1)]
    counts = Counter(values)
    return {name: int(counts[name]) for name in class_names if counts[name]}


def _tensorflow() -> Any:
    try:
        import tensorflow as tf
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise RuntimeError("TensorFlow is required for embedding extraction") from exc
    return tf


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


def _id_validation_dataset(batch_size: int, split_name: str = "val") -> Any:
    from src.dataset_pipeline import create_dataset, load_split_csv
    from src.label_encoder import fit_label_encoder

    if split_name not in {"train", "val", "test"}:
        raise ValueError("split_name must be train, val, or test")
    validation = load_split_csv(split_name)
    encoder = fit_label_encoder()
    return create_dataset(validation, encoder, batch_size=batch_size, training=False), len(validation)


def _preprocess_any_image(path: Any) -> Any:
    tf = _tensorflow()
    contents = tf.io.read_file(path)
    image = tf.io.decode_image(contents, channels=3, expand_animations=False)
    image = tf.image.resize(image, (224, 224))
    image = tf.cast(image, tf.float32)
    return tf.ensure_shape(image, (224, 224, 3))


def _ood_calibration_dataset(directory: Path, batch_size: int) -> tuple[Any | None, int]:
    tf = _tensorflow()
    paths = (
        sorted(
            path
            for path in directory.rglob("*")
            if path.is_file() and path.suffix.lower() in _IMAGE_EXTENSIONS
        )
        if directory.is_dir()
        else []
    )
    if not paths:
        return None, 0
    dataset = tf.data.Dataset.from_tensor_slices([str(path) for path in paths])
    dataset = dataset.map(_preprocess_any_image, num_parallel_calls=tf.data.AUTOTUNE)
    return dataset.batch(batch_size, drop_remainder=False).prefetch(tf.data.AUTOTUNE), len(paths)


def _collect_embeddings(
    feature_model: Any,
    dataset: Any,
    *,
    labelled: bool,
) -> tuple[np.ndarray, np.ndarray | None]:
    from src.models.feature_extractor import extract_embeddings

    embedding_batches: list[np.ndarray] = []
    label_batches: list[np.ndarray] = []
    for batch_index, batch in enumerate(dataset):
        if labelled:
            images, labels = batch
            label_values = np.asarray(labels)
            if label_values.ndim != 2 or label_values.shape[1] == 0:
                raise ValueError("ID validation labels must be a non-empty one-hot matrix")
            label_batches.append(np.argmax(label_values, axis=1).astype(np.int64))
        else:
            images = batch[0] if isinstance(batch, (tuple, list)) else batch
        embedding_batches.append(
            extract_embeddings(
                feature_model,
                images,
                check_determinism=(batch_index == 0),
            ).astype(np.float64, copy=False)
        )
        del batch
    if not embedding_batches:
        raise ValueError("cannot collect embeddings from an empty dataset")
    embeddings = np.concatenate(embedding_batches, axis=0)
    labels = np.concatenate(label_batches, axis=0) if labelled else None
    return embeddings, labels


def _stream_mahalanobis_moments(
    feature_model: Any,
    dataset: Any,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Compute mean and unbiased covariance moments without retaining batches."""

    from src.models.feature_extractor import extract_embeddings

    count = 0
    mean: np.ndarray | None = None
    m2: np.ndarray | None = None
    for batch_index, batch in enumerate(dataset):
        images = batch[0] if isinstance(batch, (tuple, list)) else batch
        values = extract_embeddings(
            feature_model,
            images,
            check_determinism=(batch_index == 0),
        ).astype(np.float64, copy=False)
        batch_count = int(values.shape[0])
        batch_mean = np.mean(values, axis=0, dtype=np.float64)
        centered = values - batch_mean
        batch_m2 = centered.T @ centered
        if mean is None:
            mean = batch_mean.copy()
            m2 = batch_m2.copy()
            count = batch_count
            continue
        assert m2 is not None
        delta = batch_mean - mean
        new_count = count + batch_count
        m2 += batch_m2 + np.outer(delta, delta) * (count * batch_count / new_count)
        mean += delta * (batch_count / new_count)
        count = new_count
    if mean is None or m2 is None or count == 0:
        raise ValueError("cannot fit Mahalanobis covariance from an empty dataset")
    denominator = max(count - 1, 1)
    covariance = (m2 / float(denominator) + (m2 / float(denominator)).T) / 2.0
    return mean, covariance, count


def _validation_image_paths(class_names: Sequence[str]) -> tuple[list[Path], np.ndarray]:
    """Resolve validation images and labels without constructing a tf.data graph."""

    from src.data_loader import TRAIN_DIR
    from src.dataset_pipeline import load_split_csv

    validation = load_split_csv("val")
    class_to_index = {name: index for index, name in enumerate(class_names)}
    paths: list[Path] = []
    labels: list[int] = []
    for row in validation.itertuples(index=False):
        image_name = str(row.img_name)
        image_path = (TRAIN_DIR / image_name).resolve()
        if Path(image_name).name != image_name or not image_path.is_file():
            raise FileNotFoundError(f"validation image not found: {image_path}")
        breed = str(row.breed)
        if breed not in class_to_index:
            raise ValueError(f"validation label is not present in centroid class_order: {breed}")
        paths.append(image_path)
        labels.append(class_to_index[breed])
    if not paths:
        raise ValueError("validation split is empty")
    return paths, np.asarray(labels, dtype=np.int64)


def _iter_image_batches(paths: Sequence[Path], batch_size: int):
    """Yield raw-pixel image batches using the same 224x224 RGB contract."""

    from PIL import Image

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    for start in range(0, len(paths), batch_size):
        batch_paths = paths[start : start + batch_size]
        images = []
        for path in batch_paths:
            with Image.open(path) as image:
                image = image.convert("RGB").resize((224, 224), Image.Resampling.BILINEAR)
                values = np.asarray(image, dtype=np.float32)
            if values.shape != (224, 224, 3) or not np.isfinite(values).all():
                raise ValueError(f"invalid decoded image tensor: {path}")
            images.append(values)
        yield np.stack(images, axis=0)


def _collect_path_embeddings(
    feature_model: Any,
    paths: Sequence[Path],
    batch_size: int,
    labels: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Extract embeddings from paths with bounded host memory."""

    from src.models.feature_extractor import extract_embeddings

    if labels is not None and len(labels) != len(paths):
        raise ValueError("labels and image paths must have the same length")
    embedding_batches: list[np.ndarray] = []
    for batch_index, image_batch in enumerate(_iter_image_batches(paths, batch_size)):
        embedding_batches.append(
            extract_embeddings(
                feature_model,
                image_batch,
                check_determinism=(batch_index == 0),
            ).astype(np.float64, copy=False)
        )
    if not embedding_batches:
        raise ValueError("cannot collect embeddings from an empty path list")
    embeddings = np.concatenate(embedding_batches, axis=0)
    return embeddings, None if labels is None else np.asarray(labels, dtype=np.int64)


def _ood_image_paths(directory: Path) -> list[Path]:
    return sorted(
        path
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in _IMAGE_EXTENSIONS
    ) if directory.is_dir() else []


def _weight_snapshot(model: Any) -> list[np.ndarray]:
    return [np.array(weight.numpy(), copy=True) for weight in model.weights]


def _weights_unchanged(model: Any, snapshot: Sequence[np.ndarray]) -> bool:
    return len(snapshot) == len(model.weights) and all(
        np.array_equal(before, weight.numpy())
        for before, weight in zip(snapshot, model.weights)
    )


def _distribution_payload(
    scores: dict[str, np.ndarray],
    class_names: Sequence[str],
    labels: np.ndarray | None = None,
) -> dict[str, Any]:
    nearest_classes = scores["nearest_class"]
    payload: dict[str, Any] = {
        "nearest_distance": _summary(scores["nearest_distance"]),
        "ood_score": _summary(scores["ood_score"]),
        "nearest_class_counts": _class_counts(nearest_classes, class_names),
    }
    if labels is not None:
        expected = np.asarray(labels, dtype=np.int64).reshape(-1)
        predicted = np.asarray(nearest_classes, dtype=object).reshape(-1)
        if expected.shape != predicted.shape:
            raise ValueError("ID labels and embedding scores have different lengths")
        expected_names = np.asarray([class_names[index] for index in expected], dtype=object)
        payload["nearest_class_accuracy"] = float(np.mean(expected_names == predicted))
    return payload


def _plot_distributions(
    id_scores: np.ndarray,
    ood_scores: np.ndarray,
    output_path: str | Path,
    *,
    title_suffix: str = "",
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(10.5, 5.5))
    ids = np.asarray(id_scores, dtype=np.float64).reshape(-1)
    oods = np.asarray(ood_scores, dtype=np.float64).reshape(-1)
    if ids.size:
        axis.hist(ids, bins=30, density=True, alpha=0.65, label="ID validation")
    if oods.size:
        axis.hist(oods, bins=30, density=True, alpha=0.65, label="OOD calibration")
    axis.set_title(
        "Cosine Distance to Nearest Class Centroid"
        + (f" — {title_suffix}" if title_suffix else "")
    )
    axis.set_xlabel("Cosine distance / OOD score (higher = more OOD)")
    axis.set_ylabel("Density")
    axis.grid(alpha=0.2)
    if ids.size or oods.size:
        axis.legend()
    if not oods.size:
        axis.text(
            0.03,
            0.94,
            "OOD calibration unavailable",
            transform=axis.transAxes,
            va="top",
            fontsize=9,
            color="darkred",
        )
    figure.tight_layout()
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=160, bbox_inches="tight")
    plt.close(figure)


def _base_payload(
    *,
    status: str,
    model_path: Path,
    centroids_path: Path,
    centroid_metadata_path: Path,
    ood_directory: Path,
    id_count: int,
    ood_count: int,
    class_names: Sequence[str],
    centroid_metadata: Mapping[str, Any],
    metrics_path: Path,
    plot_path: Path,
) -> dict[str, Any]:
    return {
        "task": "cosine_distance_ood",
        "status": status,
        "evaluation_date": date.today().isoformat(),
        "distance_definition": "1 - max cosine similarity between input embedding and class centroids",
        "score_definition": "ood_score = nearest_distance",
        "embedding_source": centroid_metadata.get(
            "embedding_layer", "production classifier embedding"
        ),
        "model_path": str(model_path),
        "centroids_path": str(centroids_path),
        "centroid_metadata_path": str(centroid_metadata_path),
        "centroids_read_only": True,
        "num_classes": len(class_names),
        "embedding_dimension": int(centroid_metadata.get("embedding_dimension", 0)),
        "class_order": list(class_names),
        "id_split": "validation",
        "id_validation_count": int(id_count),
        "ood_calibration_directory": str(ood_directory),
        "ood_calibration_count": int(ood_count),
        "output_fields": {
            "nearest_class": "class label of the centroid with maximum cosine similarity",
            "nearest_distance": "1 - maximum cosine similarity",
            "ood_score": "same value as nearest_distance; higher means more OOD",
        },
        "artifacts": {
            "metrics_json": str(metrics_path),
            "distribution_plot": str(plot_path),
        },
    }


def evaluate_cosine_calibration(
    model_path: str | Path = DEFAULT_MODEL_PATH,
    *,
    centroids_path: str | Path = DEFAULT_CENTROIDS_PATH,
    centroid_metadata_path: str | Path = DEFAULT_CENTROID_METADATA_PATH,
    class_names_path: str | Path = DEFAULT_CLASS_NAMES_PATH,
    ood_directory: str | Path = DEFAULT_OOD_CALIBRATION_DIR,
    metrics_path: str | Path = DEFAULT_METRICS_PATH,
    plot_path: str | Path = DEFAULT_PLOT_PATH,
    batch_size: int = 64,
    target_id_acceptance: float = 0.95,
) -> dict[str, Any]:
    """Evaluate cosine-distance scores on ID validation and OOD calibration."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if not 0 < target_id_acceptance <= 1:
        raise ValueError("target_id_acceptance must be in (0, 1]")

    resolved_model = _resolve_project_path(model_path).resolve()
    resolved_centroids = _resolve_project_path(centroids_path).resolve()
    resolved_metadata = _resolve_project_path(centroid_metadata_path).resolve()
    resolved_class_names = _resolve_project_path(class_names_path).resolve()
    resolved_ood = _resolve_project_path(ood_directory).resolve()
    resolved_metrics = _resolve_project_path(metrics_path).resolve()
    resolved_plot = _resolve_project_path(plot_path).resolve()

    centroids, class_names, centroid_metadata = load_centroid_artifacts(
        resolved_centroids,
        resolved_metadata,
        resolved_class_names,
    )
    model = _load_model(resolved_model)
    snapshot = _weight_snapshot(model)
    from src.models.feature_extractor import build_feature_extractor

    feature_model = build_feature_extractor(model)
    id_paths, id_labels = _validation_image_paths(class_names)
    id_count = len(id_paths)
    id_embeddings, id_labels = _collect_path_embeddings(
        feature_model,
        id_paths,
        batch_size,
        labels=id_labels,
    )
    if not _weights_unchanged(model, snapshot):
        raise RuntimeError("feature extraction changed production model weights")
    if id_embeddings.shape[1] != centroids.shape[1]:
        raise ValueError(
            f"embedding dimension {id_embeddings.shape[1]} does not match centroids {centroids.shape[1]}"
        )

    id_scores = score_embeddings(id_embeddings, centroids, class_names)
    ood_paths = _ood_image_paths(resolved_ood)
    ood_count = len(ood_paths)

    payload = _base_payload(
        status="evaluated" if ood_paths else "missing_ood_calibration_data",
        model_path=resolved_model,
        centroids_path=resolved_centroids,
        centroid_metadata_path=resolved_metadata,
        ood_directory=resolved_ood,
        id_count=id_count,
        ood_count=ood_count,
        class_names=class_names,
        centroid_metadata=centroid_metadata,
        metrics_path=resolved_metrics,
        plot_path=resolved_plot,
    )
    payload["id_validation"] = _distribution_payload(id_scores, class_names, id_labels)

    if not ood_paths:
        payload["ood_calibration"] = {
            "nearest_distance": {"count": 0},
            "ood_score": {"count": 0},
            "nearest_class_counts": {},
        }
        payload["metrics"] = None
        payload["calibration"] = None
        payload["notes"] = [
            "OOD calibration directory contains no supported images.",
            "AUROC, threshold calibration, and OOD separation claims are unavailable until OOD data is supplied.",
        ]
        _write_payload_and_plot(
            payload,
            resolved_metrics,
            resolved_plot,
            id_scores["ood_score"],
            np.array([], dtype=np.float64),
            title_suffix="ID validation only; OOD calibration data missing",
        )
        return payload

    ood_embeddings, _ = _collect_path_embeddings(feature_model, ood_paths, batch_size)
    if not _weights_unchanged(model, snapshot):
        raise RuntimeError("feature extraction changed production model weights")
    if ood_embeddings.shape[1] != centroids.shape[1]:
        raise ValueError(
            f"OOD embedding dimension {ood_embeddings.shape[1]} does not match centroids {centroids.shape[1]}"
        )
    ood_scores = score_embeddings(ood_embeddings, centroids, class_names)
    from src.ood.calibration import calibrate_threshold
    from src.ood.evaluate_ood import evaluate_ood_scores, rejection_error_rates

    metrics = evaluate_ood_scores(
        id_scores["ood_score"], ood_scores["ood_score"], detector="cosine"
    )
    calibration = calibrate_threshold(
        id_scores["ood_score"],
        ood_scores["ood_score"],
        target_id_acceptance=target_id_acceptance,
    )
    payload["ood_calibration"] = _distribution_payload(ood_scores, class_names)
    payload["metrics"] = metrics
    payload["calibration"] = {
        **calibration,
        "threshold_errors": rejection_error_rates(
            id_scores["ood_score"],
            ood_scores["ood_score"],
            calibration["ood_threshold"],
            higher_score_is_ood=True,
        ),
    }
    payload["notes"] = [
        "Centroid artifacts were loaded read-only and were not recomputed or overwritten.",
        "Threshold calibration is reported for analysis; production policy is not changed automatically.",
    ]
    _write_payload_and_plot(
        payload,
        resolved_metrics,
        resolved_plot,
        id_scores["ood_score"],
        ood_scores["ood_score"],
    )
    return payload


def _write_payload_and_plot(
    payload: dict[str, Any],
    metrics_path: Path,
    plot_path: Path,
    id_scores: np.ndarray,
    ood_scores: np.ndarray,
    *,
    title_suffix: str = "",
) -> None:
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    _plot_distributions(id_scores, ood_scores, plot_path, title_suffix=title_suffix)


def _atomic_save_npy(path: Path, values: np.ndarray) -> None:
    """Persist a NumPy array without leaving a partially written artifact."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npy", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        np.save(temporary, np.asarray(values), allow_pickle=False)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(payload), indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def evaluate_mahalanobis_calibration(
    model_path: str | Path = DEFAULT_MODEL_PATH,
    *,
    train_csv: str | Path = PROJECT_ROOT / "outputs" / "splits" / "train.csv",
    covariance_path: str | Path = DEFAULT_COVARIANCE_PATH,
    mean_path: str | Path = DEFAULT_MAHALANOBIS_MEAN_PATH,
    metadata_path: str | Path = DEFAULT_MAHALANOBIS_METADATA_PATH,
    ood_directory: str | Path = DEFAULT_OOD_CALIBRATION_DIR,
    metrics_path: str | Path = DEFAULT_MAHALANOBIS_METRICS_PATH,
    batch_size: int = 64,
    regularization: float = DEFAULT_MAHALANOBIS_REGULARIZATION,
    pinv_rcond: float = DEFAULT_MAHALANOBIS_PINV_RCOND,
    target_id_acceptance: float = 0.95,
) -> dict[str, Any]:
    """Fit Mahalanobis covariance on train and evaluate validation/OOD.

    The training dataset is iterated once to create the covariance. The
    validation and OOD datasets are only scored after the fit is complete;
    neither can influence the saved covariance or mean.
    """

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if regularization < 0 or not np.isfinite(regularization):
        raise ValueError("regularization must be non-negative and finite")
    if pinv_rcond < 0 or not np.isfinite(pinv_rcond):
        raise ValueError("pinv_rcond must be non-negative and finite")
    if not 0 < target_id_acceptance <= 1:
        raise ValueError("target_id_acceptance must be in (0, 1]")

    resolved_model = _resolve_project_path(model_path).resolve()
    resolved_train_csv = _resolve_project_path(train_csv).resolve()
    resolved_covariance = _resolve_project_path(covariance_path).resolve()
    resolved_mean = _resolve_project_path(mean_path).resolve()
    resolved_metadata = _resolve_project_path(metadata_path).resolve()
    resolved_ood = _resolve_project_path(ood_directory).resolve()
    resolved_metrics = _resolve_project_path(metrics_path).resolve()
    if not resolved_train_csv.is_file():
        raise FileNotFoundError(f"Training split not found: {resolved_train_csv}")

    # The normal project pipeline maps this exact train split to data/train.
    # Refuse a differently named CSV so a validation/test file cannot silently
    # become the covariance source.
    if resolved_train_csv.name.lower() != "train.csv":
        raise ValueError("Mahalanobis covariance requires train.csv")

    model = _load_model(resolved_model)
    snapshot = _weight_snapshot(model)
    from src.models.feature_extractor import build_feature_extractor

    feature_model = build_feature_extractor(model)
    train_dataset, train_count = _id_validation_dataset(batch_size, "train")
    train_mean, train_covariance, streamed_train_count = _stream_mahalanobis_moments(
        feature_model, train_dataset
    )
    if streamed_train_count != train_count:
        raise RuntimeError(
            f"training embedding count ({streamed_train_count}) does not match "
            f"train split ({train_count})"
        )
    if not _weights_unchanged(model, snapshot):
        raise RuntimeError("feature extraction changed production model weights")

    fitted = _fit_mahalanobis_from_moments(
        train_mean,
        train_covariance,
        sample_count=streamed_train_count,
        regularization=regularization,
    )
    _atomic_save_npy(resolved_covariance, fitted["covariance"].astype(np.float64))
    _atomic_save_npy(resolved_mean, fitted["mean"].astype(np.float64))

    validation_dataset, validation_count = _id_validation_dataset(batch_size, "val")
    validation_embeddings, _ = _collect_embeddings(
        feature_model, validation_dataset, labelled=True
    )
    if validation_embeddings.shape[0] != validation_count:
        raise RuntimeError("validation embedding count does not match val split")
    if not _weights_unchanged(model, snapshot):
        raise RuntimeError("feature extraction changed production model weights")

    ood_dataset, ood_count = _ood_calibration_dataset(resolved_ood, batch_size)
    if ood_dataset is None:
        ood_embeddings = np.empty((0, train_mean.shape[0]), dtype=np.float64)
    else:
        ood_embeddings, _ = _collect_embeddings(
            feature_model, ood_dataset, labelled=False
        )
        if not _weights_unchanged(model, snapshot):
            raise RuntimeError("feature extraction changed production model weights")

    # covariance.npy already contains the ridge; do not add it twice here.
    id_scores = mahalanobis_distance(
        validation_embeddings,
        fitted["mean"],
        fitted["covariance"],
        regularization=0.0,
        pinv_rcond=pinv_rcond,
    )
    ood_scores = (
        mahalanobis_distance(
            ood_embeddings,
            fitted["mean"],
            fitted["covariance"],
            regularization=0.0,
            pinv_rcond=pinv_rcond,
        )
        if ood_count
        else np.empty(0, dtype=np.float64)
    )

    metadata: dict[str, Any] = {
        "task": "TASK 13 — MAHALANOBIS OOD",
        "method": "mahalanobis",
        "formula": "(embedding - training_mean)^T covariance_pinv (embedding - training_mean)",
        "score_definition": "squared Mahalanobis distance to the global training mean",
        "score_orientation": "higher values indicate more likely OOD",
        "training_source": "train split only",
        "training_sample_count": int(train_count),
        "embedding_dimension": int(fitted["embedding_dimension"]),
        "train_csv_sha256": hashlib.sha256(resolved_train_csv.read_bytes()).hexdigest(),
        "model_sha256": hashlib.sha256(resolved_model.read_bytes()).hexdigest(),
        "regularization": {
            "requested_multiplier": float(regularization),
            "ridge_added_to_diagonal": float(fitted["ridge_added"]),
            "description": "covariance was symmetrized and a scale-aware diagonal ridge was added",
        },
        "pseudo_inverse": {
            "method": "numpy.linalg.pinv",
            "hermitian": True,
            "rcond": float(pinv_rcond),
            "description": "singular/near-zero directions are truncated instead of inverted",
        },
        "fitted_covariance": {
            "min_eigenvalue": fitted["min_eigenvalue"],
            "max_eigenvalue": fitted["max_eigenvalue"],
            "condition_number": fitted["condition_number"],
        },
        "artifacts": {
            "covariance_npy": str(resolved_covariance),
            "mean_npy": str(resolved_mean),
            "metadata_json": str(resolved_metadata),
        },
    }
    _write_json(resolved_metadata, metadata)

    payload: dict[str, Any] = {
        **metadata,
        "id_split": "validation",
        "id_validation_count": int(validation_count),
        "ood_calibration_directory": str(resolved_ood),
        "ood_calibration_count": int(ood_count),
        "id_validation": _summary(id_scores),
        "ood_calibration": _summary(ood_scores),
        "artifacts": {**metadata["artifacts"], "metrics_json": str(resolved_metrics)},
    }
    if ood_count:
        from src.ood.calibration import calibrate_threshold
        from src.ood.evaluate_ood import evaluate_ood_scores, rejection_error_rates

        payload["metrics"] = evaluate_ood_scores(
            id_scores, ood_scores, detector="mahalanobis"
        )
        calibration = calibrate_threshold(
            id_scores,
            ood_scores,
            target_id_acceptance=target_id_acceptance,
        )
        payload["calibration"] = {
            **calibration,
            "threshold_errors": rejection_error_rates(
                id_scores,
                ood_scores,
                calibration["ood_threshold"],
                higher_score_is_ood=True,
            ),
        }
        payload["status"] = "evaluated"
        payload["notes"] = [
            "Covariance and mean were fit from training embeddings only.",
            "Threshold calibration is reported for analysis; production policy is not changed automatically.",
        ]
    else:
        payload["metrics"] = None
        payload["calibration"] = None
        payload["status"] = "missing_ood_calibration_data"
        payload["notes"] = [
            "OOD calibration directory contains no supported images.",
            "AUROC and threshold calibration are unavailable until OOD data is supplied.",
        ]
    _write_json(resolved_metrics, payload)
    return payload


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate embedding-distance OOD scores."
    )
    parser.add_argument(
        "--method",
        choices=("cosine", "mahalanobis"),
        default="cosine",
        help="embedding distance to evaluate (default: cosine for backward compatibility)",
    )
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--centroids-path", type=Path, default=DEFAULT_CENTROIDS_PATH)
    parser.add_argument("--centroid-metadata-path", type=Path, default=DEFAULT_CENTROID_METADATA_PATH)
    parser.add_argument("--class-names-path", type=Path, default=DEFAULT_CLASS_NAMES_PATH)
    parser.add_argument("--ood-directory", type=Path, default=DEFAULT_OOD_CALIBRATION_DIR)
    parser.add_argument("--metrics-path", type=Path, default=DEFAULT_METRICS_PATH)
    parser.add_argument("--plot-path", type=Path, default=DEFAULT_PLOT_PATH)
    parser.add_argument("--train-csv", type=Path, default=PROJECT_ROOT / "outputs" / "splits" / "train.csv")
    parser.add_argument("--covariance-path", type=Path, default=DEFAULT_COVARIANCE_PATH)
    parser.add_argument("--mean-path", type=Path, default=DEFAULT_MAHALANOBIS_MEAN_PATH)
    parser.add_argument("--metadata-path", type=Path, default=DEFAULT_MAHALANOBIS_METADATA_PATH)
    parser.add_argument("--regularization", type=float, default=DEFAULT_MAHALANOBIS_REGULARIZATION)
    parser.add_argument("--pinv-rcond", type=float, default=DEFAULT_MAHALANOBIS_PINV_RCOND)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--target-id-acceptance", type=float, default=0.95)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.method == "mahalanobis":
        payload = evaluate_mahalanobis_calibration(
            model_path=args.model_path,
            train_csv=args.train_csv,
            covariance_path=args.covariance_path,
            mean_path=args.mean_path,
            metadata_path=args.metadata_path,
            ood_directory=args.ood_directory,
            metrics_path=args.metrics_path
            if args.metrics_path != DEFAULT_METRICS_PATH
            else DEFAULT_MAHALANOBIS_METRICS_PATH,
            batch_size=args.batch_size,
            regularization=args.regularization,
            pinv_rcond=args.pinv_rcond,
            target_id_acceptance=args.target_id_acceptance,
        )
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return
    payload = evaluate_cosine_calibration(
        model_path=args.model_path,
        centroids_path=args.centroids_path,
        centroid_metadata_path=args.centroid_metadata_path,
        class_names_path=args.class_names_path,
        ood_directory=args.ood_directory,
        metrics_path=args.metrics_path,
        plot_path=args.plot_path,
        batch_size=args.batch_size,
        target_id_acceptance=args.target_id_acceptance,
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
