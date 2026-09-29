"""Unified, long-lived prediction pipeline.

The application should create one :class:`Predictor` during startup and reuse
it for every request. Model loading, graph adapter construction, threshold
loading and image-independent validation all happen in ``__init__``; the
request path only decodes the image and executes inference.

The production model currently exposes a 120-class softmax. ``Predictor``
also supports a model that exposes logits (which is useful for tests and for
future checkpoints), as well as a model adapter returning named ``logits``,
``probabilities`` and ``embedding`` outputs. In all cases the public result is
a JSON-safe Python mapping whose state is exactly ``DOG``, ``NOT_DOG`` or
``UNCERTAIN``.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from src.models.feature_extractor import build_feature_extractor, extract_embeddings
from src.ood.energy import build_logits_model, energy_ood_score, extract_logits
from src.ood.entropy import entropy_ood_score
from src.ood.embedding_distance import mahalanobis_distance, score_embedding
from src.ood.msp import msp_ood_score

from .decision import (
    DEFAULT_THRESHOLD_PATH,
    Decision,
    DecisionStatus,
    decide_prediction,
    load_thresholds,
)
from .preprocessing import IMAGE_SIZE, preprocess_image


DEFAULT_MODEL_PATH = (
    Path(__file__).resolve().parents[2]
    / "outputs"
    / "deployment_ready"
    / "dog_breed_model.keras"
)
DEFAULT_CLASS_NAMES_PATH = (
    Path(__file__).resolve().parents[2]
    / "outputs"
    / "deployment_ready"
    / "class_names.json"
)


def _tensorflow_model_loader(path: str | Path) -> Any:
    """Load a Keras model only when a ``Predictor`` is constructed."""

    try:
        import tensorflow as tf
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("TensorFlow is required to load the production model") from exc

    preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
    kwargs: dict[str, Any] = {
        "custom_objects": {"preprocess_input": preprocess_input},
        "compile": False,
    }
    # ``safe_mode`` is available in current Keras, but keep the fallback for
    # older compatible TensorFlow installations.
    try:
        return tf.keras.models.load_model(str(path), safe_mode=True, **kwargs)
    except TypeError as exc:
        if "safe_mode" not in str(exc):
            raise
        return tf.keras.models.load_model(str(path), **kwargs)


def _load_class_names(value: Mapping[int, str] | Sequence[str] | str | Path) -> dict[int, str]:
    """Normalize a class mapping or the deployment JSON artifact."""

    if isinstance(value, (str, Path)):
        try:
            with Path(value).open(encoding="utf-8") as handle:
                value = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Could not read class names from {value}: {exc}") from exc

    if isinstance(value, Mapping):
        result: dict[int, str] = {}
        for raw_index, raw_name in value.items():
            try:
                index = int(raw_index)
            except (TypeError, ValueError) as exc:
                raise ValueError("class name mapping keys must be integer indices") from exc
            if index in result:
                raise ValueError(f"duplicate class index: {index}")
            result[index] = str(raw_name)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        result = {index: str(name) for index, name in enumerate(value)}
    else:
        raise TypeError("class_names must be a mapping, sequence, or JSON path")

    if not result or set(result) != set(range(len(result))):
        raise ValueError("class_names must contain consecutive indices starting at 0")
    if any(not name.strip() for name in result.values()):
        raise ValueError("class_names must contain non-empty names")
    return result


def _invoke(model: Any, image_batch: np.ndarray) -> Any:
    """Invoke a Keras-like or test-double model in inference mode."""

    if callable(model):
        try:
            return model(image_batch, training=False)
        except TypeError as first_error:
            # Small test doubles often only accept the input positional arg.
            try:
                return model(image_batch)
            except TypeError:
                if not hasattr(model, "predict"):
                    raise first_error
    predict = getattr(model, "predict", None)
    if callable(predict):
        try:
            return predict(image_batch, verbose=0)
        except TypeError:
            return predict(image_batch)
    raise TypeError("model must be callable or expose predict()")


def _finite_matrix(value: Any, name: str, *, batch_size: int = 1) -> np.ndarray:
    values = np.asarray(value, dtype=np.float64)
    if values.ndim == 1:
        values = values[None, :]
    if (
        values.ndim != 2
        or values.shape[0] != batch_size
        or values.shape[1] == 0
        or not np.isfinite(values).all()
    ):
        raise RuntimeError(
            f"{name} must have shape ({batch_size}, classes) and contain finite values; "
            f"got {values.shape}"
        )
    return values


def _softmax(logits: np.ndarray) -> np.ndarray:
    """Compute a stable row-wise softmax and return float64 probabilities."""

    values = _finite_matrix(logits, "logits")
    shifted = values - np.max(values, axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    probabilities = exponentials / np.sum(exponentials, axis=1, keepdims=True)
    if not np.isfinite(probabilities).all():
        raise RuntimeError("softmax output contains NaN or Inf")
    return probabilities


def _probabilities(value: Any, *, expected_classes: int) -> np.ndarray:
    values = _finite_matrix(value, "probabilities")
    if values.shape[1] != expected_classes:
        raise RuntimeError(
            f"probability output has {values.shape[1]} classes; expected {expected_classes}"
        )
    if np.any(values < 0) or not np.isclose(values.sum(axis=1), 1.0, atol=1e-3).all():
        raise RuntimeError("model probabilities must be non-negative and sum to 1")
    return values


def _json_value(value: Any) -> Any:
    """Convert NumPy values returned by detectors into ordinary Python data."""

    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


class PredictionResult(dict):
    """Dictionary result with convenient enum-like status properties."""

    @property
    def status(self) -> DecisionStatus:
        return DecisionStatus(str(self["status"]))

    @property
    def decision(self) -> DecisionStatus:
        return self.status

    def as_dict(self) -> dict[str, Any]:
        """Return a plain copy suitable for JSON serialization."""

        return dict(self)


class Predictor:
    """Run the complete image-to-decision inference pipeline.

    ``model_loader`` is called exactly once by ``__init__`` and never from a
    request method. Optional logits and embedding graph adapters are likewise
    built once and reused.
    """

    def __init__(
        self,
        model: Any | None = None,
        class_names: Mapping[int, str] | Sequence[str] | str | Path | None = None,
        config: Mapping[str, Any] | None = None,
        *,
        model_path: str | Path | None = None,
        model_loader: Callable[..., Any] | None = None,
        thresholds: Mapping[str, Any] | str | Path | None = None,
        logits_model: Any | None = None,
        embedding_model: Any | None = None,
        ood_detector: Any | None = None,
        preprocess_fn: Callable[[bytes], np.ndarray] = preprocess_image,
        decision_fn: Callable[..., Decision] = decide_prediction,
    ) -> None:
        # Allow the natural ``Predictor(model_path, class_names=...)`` form
        # in addition to the explicit keyword form.
        if isinstance(model, (str, Path)) and model_path is None:
            model_path = model
            model = None
        if model is not None and model_path is not None:
            raise ValueError("pass either model or model_path, not both")
        if model is None:
            resolved_path = Path(model_path or DEFAULT_MODEL_PATH)
            loader = model_loader or _tensorflow_model_loader
            self.model = self._load_once(loader, resolved_path, model_loader is not None)
            self.model_path = resolved_path
        else:
            self.model = model
            self.model_path = None

        if class_names is None:
            class_names = DEFAULT_CLASS_NAMES_PATH
        self.class_names = _load_class_names(class_names)
        self.config = dict(config or {})
        self.method = str(self.config.get("method", "msp")).strip().lower()
        self.temperature = float(self.config.get("temperature", 1.0))
        if not np.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("temperature must be a positive finite number")
        self.preprocess_fn = preprocess_fn
        self.decision_fn = decision_fn
        self.ood_detector = ood_detector
        self._logits_exact = False

        # Policy and adapters are startup state. Do not read threshold JSON or
        # construct a Keras adapter for every request.
        self.thresholds = self._load_policy(thresholds)
        self.logits_model = logits_model
        self.embedding_model = embedding_model
        self._build_adapters_once()

    @staticmethod
    def _load_once(
        loader: Callable[..., Any], path: Path, custom_loader: bool
    ) -> Any:
        """Call a supplied loader once, accepting either zero or one argument."""

        if custom_loader:
            try:
                return loader(path)
            except TypeError as first_error:
                try:
                    return loader()
                except TypeError:
                    raise first_error
        return loader(path)

    def _load_policy(
        self, thresholds: Mapping[str, Any] | str | Path | None
    ) -> Mapping[str, Any] | None:
        policy = thresholds
        if policy is None:
            policy = self.config.get("thresholds")
        if policy is None:
            policy = self.config.get("threshold_path")
        if policy is None:
            # Loading the default is intentional and happens once at startup.
            policy = DEFAULT_THRESHOLD_PATH
        if isinstance(policy, (str, Path)):
            return load_thresholds(policy)
        if not isinstance(policy, Mapping):
            raise TypeError("thresholds must be a mapping or a JSON artifact path")
        return dict(policy)

    def _build_adapters_once(self) -> None:
        """Create optional logits/embedding graph adapters at startup."""

        if bool(self.config.get("auto_build_logits", True)) and self.logits_model is None:
            if getattr(self.model, "inputs", None) is not None and getattr(self.model, "layers", None):
                try:
                    self.logits_model = build_logits_model(self.model)
                except (AttributeError, ImportError, RuntimeError, TypeError, ValueError):
                    # Test doubles and future checkpoints may already expose
                    # logits directly, so an adapter is not always needed.
                    self.logits_model = None

        if bool(self.config.get("auto_build_embedding", True)) and self.embedding_model is None:
            if getattr(self.model, "inputs", None) is not None and getattr(self.model, "layers", None):
                try:
                    self.embedding_model = build_feature_extractor(self.model)
                except (AttributeError, ImportError, RuntimeError, TypeError, ValueError):
                    self.embedding_model = None

        self._requires_exact_logits = self.method in {"energy", "energy_score"}

    def load_image(
        self, image: bytes | bytearray | memoryview | str | Path | Any
    ) -> bytes | np.ndarray:
        """Load request input into bytes or a preprocessed batch."""

        if isinstance(image, np.ndarray):
            return image
        if isinstance(image, (str, Path)):
            try:
                return Path(image).read_bytes()
            except OSError as exc:
                raise ValueError(f"could not read image: {image}") from exc
        if isinstance(image, (bytes, bytearray, memoryview)):
            data = bytes(image)
            if not data:
                raise ValueError("image data is empty")
            return data
        read = getattr(image, "read", None)
        if callable(read):
            data = read()
            if not isinstance(data, (bytes, bytearray, memoryview)) or not data:
                raise ValueError("image stream did not contain image data")
            return bytes(data)
        raise TypeError("image must be bytes, a path, a file-like object, or an image batch")

    def validate(self, image_batch: Any) -> np.ndarray:
        """Validate the preprocessed request batch before model inference."""

        values = np.asarray(image_batch)
        expected = (1, int(IMAGE_SIZE[0]), int(IMAGE_SIZE[1]), 3)
        if values.shape != expected:
            raise ValueError(f"image batch must have shape {expected}, got {values.shape}")
        if values.dtype != np.float32:
            raise ValueError("image batch must use float32 pixels")
        if not np.isfinite(values).all() or np.any(values < 0) or np.any(values > 255):
            raise ValueError("image pixels must be finite values in [0, 255]")
        return values

    def validate_image(self, image: bytes | bytearray | memoryview | np.ndarray) -> None:
        """Validate the loaded request before the decode/resize stage."""

        if isinstance(image, np.ndarray):
            self.validate(image)
            return
        if not isinstance(image, (bytes, bytearray, memoryview)) or not bytes(image):
            raise ValueError("image data is empty or has an unsupported type")

    def preprocess(
        self, image: bytes | bytearray | memoryview | str | Path | Any
    ) -> np.ndarray:
        """Load, validate and preprocess one request image."""

        loaded = self.load_image(image)
        self.validate_image(loaded)
        if isinstance(loaded, np.ndarray):
            return self.validate(loaded)
        batch = self.preprocess_fn(loaded)
        return self.validate(batch)

    def _model_outputs(self, image_batch: np.ndarray) -> tuple[Any, Any, Any]:
        raw = _invoke(self.model, image_batch)
        if isinstance(raw, Mapping):
            logits = next(
                (raw[key] for key in ("logits", "breed_logits", "pre_logits") if key in raw),
                None,
            )
            probabilities = next(
                (
                    raw[key]
                    for key in (
                        "probabilities",
                        "breed_probabilities",
                        "probs",
                        "predictions",
                    )
                    if key in raw
                ),
                None,
            )
            embedding = next(
                (raw[key] for key in ("embedding", "embeddings", "features") if key in raw),
                None,
            )
            if logits is None and probabilities is None:
                if len(raw) == 1:
                    probabilities = next(iter(raw.values()))
                else:
                    raise RuntimeError("model mapping must expose logits or probabilities")
            return logits, probabilities, embedding

        if isinstance(raw, (tuple, list)):
            if len(raw) == 1:
                return None, raw[0], None
            if len(raw) == 2:
                # The documented multi-output order is logits, embedding.
                return raw[0], None, raw[1]
            raise RuntimeError("model output sequence must contain one or two values")
        return None, raw, None

    def _classify(
        self, image_batch: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, bool]:
        raw_logits, raw_probabilities, raw_embedding = self._model_outputs(image_batch)
        output_mode = str(self.config.get("output_mode", "auto")).lower()
        expected_classes = len(self.class_names)

        # Production Keras checkpoints expose a batched tensor.  Keep the
        # general predictor permissive for lightweight test doubles, while
        # allowing integrations to enforce the deployed model's exact batch
        # contract when required.
        if self.config.get("strict_batch_output", False):
            for name, value in (("logits", raw_logits), ("probabilities", raw_probabilities)):
                if value is not None and np.asarray(value).ndim != 2:
                    raise RuntimeError(f"{name} output must have shape (1, classes)")

        logits: np.ndarray | None = None
        probabilities: np.ndarray
        logits_exact = False
        if raw_logits is not None:
            logits = _finite_matrix(raw_logits, "logits")
            logits_exact = True
        if raw_probabilities is not None:
            candidate = _finite_matrix(raw_probabilities, "model output")
            if output_mode == "logits":
                logits = candidate
                logits_exact = True
            elif output_mode == "probabilities":
                probabilities = _probabilities(candidate, expected_classes=expected_classes)
            elif logits is not None:
                # A multi-output model may expose both values. Validate and
                # prefer its explicit probabilities while retaining the exact
                # logits for energy scoring.
                probabilities = _probabilities(candidate, expected_classes=expected_classes)
            elif logits is None and np.all(candidate >= 0) and np.isclose(
                candidate.sum(axis=1), 1.0, atol=1e-3
            ).all():
                probabilities = _probabilities(candidate, expected_classes=expected_classes)
            elif logits is None:
                logits = candidate
                logits_exact = True
        if logits is None:
            if "probabilities" not in locals():
                raise RuntimeError("model did not produce logits or probabilities")
            # A logits adapter is built once at startup and can recover the
            # true pre-softmax values from the deployed softmax model.
            if self.logits_model is not None:
                logits = extract_logits(self.logits_model, image_batch)
                logits_exact = True
                probabilities = _softmax(logits)
            else:
                # Useful for diagnostics and MSP/entropy, but deliberately
                # not accepted by energy OOD.
                logits = np.log(np.clip(probabilities, np.finfo(np.float64).tiny, 1.0))
        else:
            if logits.shape[1] != expected_classes:
                raise RuntimeError(
                    f"logit output has {logits.shape[1]} classes; expected {expected_classes}"
                )
            # Exact logits are the canonical source for the probability
            # distribution. This keeps the classifier and energy detector on
            # one numerically consistent output path.
            if logits_exact or "probabilities" not in locals():
                probabilities = _softmax(logits)

        if probabilities.shape[1] != expected_classes:
            raise RuntimeError(
                f"probability output has {probabilities.shape[1]} classes; expected {expected_classes}"
            )

        embedding = None if raw_embedding is None else _finite_matrix(raw_embedding, "embedding")
        if embedding is None and self.embedding_model is not None:
            embedding = extract_embeddings(
                self.embedding_model, image_batch, check_determinism=False
            )
        if embedding is not None and embedding.shape[0] != 1:
            raise RuntimeError("embedding output must contain one request embedding")
        return logits, probabilities, embedding, logits_exact

    def _embedding_ood_score(self, embedding: np.ndarray) -> tuple[float, dict[str, Any]]:
        detector = self.ood_detector
        if detector is not None:
            scored = (
                detector.score(embedding)
                if callable(getattr(detector, "score", None))
                else detector(embedding)
            )
            if isinstance(scored, Mapping):
                if "ood_score" in scored:
                    score = scored["ood_score"]
                elif "score" in scored:
                    score = scored["score"]
                else:
                    raise ValueError("embedding detector result must contain score or ood_score")
                return float(np.asarray(score).reshape(-1)[0]), dict(scored)
            return float(np.asarray(scored).reshape(-1)[0]), {}

        centroids = self.config.get("centroids")
        if self.method in {"cosine", "cosine_distance", "embedding", "embedding_distance"}:
            if centroids is None:
                raise ValueError("embedding OOD method requires centroids")
            scored = score_embedding(embedding[0], np.asarray(centroids))
            return float(scored["ood_score"]), scored
        if self.method in {"mahalanobis", "mahalanobis_distance"}:
            mean = self.config.get("mean")
            covariance = self.config.get("covariance")
            if mean is None or covariance is None:
                raise ValueError("mahalanobis OOD method requires mean and covariance")
            scored = mahalanobis_distance(
                embedding,
                np.asarray(mean),
                np.asarray(covariance),
                regularization=float(self.config.get("regularization", 1e-6)),
                pinv_rcond=float(self.config.get("pinv_rcond", 1e-12)),
            )
            return float(scored[0]), {}
        raise ValueError(f"Unsupported embedding OOD method: {self.method}")

    def _ood_score(
        self,
        probabilities: np.ndarray,
        logits: np.ndarray,
        embedding: np.ndarray | None,
        *,
        logits_exact: bool,
    ) -> tuple[float, dict[str, Any]]:
        if self.ood_detector is not None and self.method not in {
            "cosine",
            "cosine_distance",
            "embedding",
            "embedding_distance",
            "mahalanobis",
            "mahalanobis_distance",
        }:
            detector = self.ood_detector
            fn = detector.score if callable(getattr(detector, "score", None)) else detector
            try:
                scored = fn(
                    probabilities=probabilities,
                    logits=logits,
                    embedding=embedding,
                )
            except TypeError:
                scored = fn(probabilities)
            if isinstance(scored, Mapping):
                value = scored.get("ood_score", scored.get("score"))
                if value is None:
                    raise ValueError("OOD detector result must contain score or ood_score")
                return float(np.asarray(value).reshape(-1)[0]), dict(scored)
            return float(np.asarray(scored).reshape(-1)[0]), {}

        if self.method in {"msp", "maximum_softmax_probability"}:
            return float(np.asarray(msp_ood_score(probabilities)).reshape(-1)[0]), {}
        if self.method in {"entropy", "predictive_entropy"}:
            return float(np.asarray(entropy_ood_score(probabilities)).reshape(-1)[0]), {}
        if self.method in {"energy", "energy_score"}:
            if not logits_exact:
                raise RuntimeError(
                    "energy OOD scoring requires exact pre-softmax logits; "
                    "provide logits_model or a logits-output model"
                )
            return (
                float(np.asarray(energy_ood_score(logits, temperature=self.temperature)).reshape(-1)[0]),
                {},
            )
        if embedding is None:
            raise RuntimeError(f"OOD method '{self.method}' requires an embedding")
        return self._embedding_ood_score(embedding)

    def _make_decision(self, probabilities: np.ndarray, score: float) -> Decision:
        decision = self.decision_fn(
            probabilities[0],
            score,
            self.class_names,
            self.thresholds,
        )
        if isinstance(decision, Decision):
            return decision
        if isinstance(decision, Mapping):
            state = decision.get("decision", decision.get("status"))
            if state is None:
                raise RuntimeError("decision mapping must contain decision or status")
            status = state if isinstance(state, DecisionStatus) else DecisionStatus(str(state))
            return Decision(
                status,
                decision.get("is_dog"),
                decision.get("prediction", status.value.lower()),
                decision.get("message"),
            )
        status = decision if isinstance(decision, DecisionStatus) else DecisionStatus(str(decision))
        return Decision(status, None, status.value.lower())

    def predict_batch(self, image_batch: np.ndarray) -> PredictionResult:
        """Run inference on one already-preprocessed image batch."""

        batch = self.validate(image_batch)
        logits, probabilities, embedding, logits_exact = self._classify(batch)
        score, detector_details = self._ood_score(
            probabilities,
            logits,
            embedding,
            logits_exact=logits_exact,
        )
        if not np.isfinite(score):
            raise RuntimeError("OOD score must be finite")
        decision = self._make_decision(probabilities, score)

        indices = np.argsort(-probabilities[0], kind="stable")[:5]
        top5 = [
            {
                "index": int(index),
                "breed": self.class_names[int(index)],
                "confidence": float(probabilities[0, index]),
            }
            for index in indices
        ]
        result: dict[str, Any] = {
            "status": decision.status.value,
            "decision": decision.status.value,
            "is_dog": decision.is_dog,
            "prediction": _json_value(decision.prediction),
            "message": decision.message,
            "probabilities": _json_value(probabilities[0]),
            "logits": _json_value(logits[0]),
            "embedding": None if embedding is None else _json_value(embedding[0]),
            "ood_score": float(score),
            "ood": {
                "method": self.method,
                "score": float(score),
                "is_ood": (
                    True
                    if decision.status is DecisionStatus.NOT_DOG
                    else False
                    if decision.status is DecisionStatus.DOG
                    else None
                ),
                **_json_value(detector_details),
            },
            # Breed rankings are meaningful only after the decision layer
            # accepts the input as a dog.
            "top1": top5[0] if decision.status is DecisionStatus.DOG else None,
            "top5": top5 if decision.status is DecisionStatus.DOG else [],
        }
        if result["message"] is None:
            result.pop("message")
        model_name = getattr(self.model, "name", None)
        if model_name:
            result["model"] = str(model_name)
        return PredictionResult(result)

    def predict(
        self, image: bytes | bytearray | memoryview | str | Path | Any
    ) -> PredictionResult:
        """Load, validate, preprocess and predict one image request."""

        return self.predict_batch(self.preprocess(image))

    __call__ = predict
    predict_image = predict


class OODPredictor:
    """Backward-compatible classifier/OOD wrapper.

    Existing callers pass a preprocessed batch and expect the original raw
    top-5 contract. The new :class:`Predictor` is the complete request API.
    """

    def __init__(
        self,
        model: Any,
        class_names: Mapping[int, str] | Sequence[str],
        config: Mapping[str, Any] | None = None,
    ):
        self.model = model
        self.class_names = class_names
        self.config = dict(config or {})
        self.method = str(self.config.get("method", "msp")).lower()
        self.temperature = float(self.config.get("temperature", 1.0))
        self._logits_model = self.config.get("logits_model")

    def _score(self, probabilities: np.ndarray, *, logits: np.ndarray | None = None) -> float:
        if self.method in {"msp", "maximum_softmax_probability"}:
            return float(np.asarray(msp_ood_score(probabilities)).reshape(-1)[0])
        if self.method in {"entropy", "predictive_entropy"}:
            return float(np.asarray(entropy_ood_score(probabilities)).reshape(-1)[0])
        if self.method in {"energy", "energy_score"}:
            if logits is None:
                raise ValueError("energy OOD scoring requires pre-softmax logits")
            return float(np.asarray(energy_ood_score(logits, temperature=self.temperature)).reshape(-1)[0])
        raise ValueError(f"Unsupported OOD method: {self.method}")

    def predict_raw(self, image_batch: np.ndarray) -> dict[str, Any]:
        """Return validated probabilities, top-5 records, and OOD score."""

        values = np.asarray(image_batch)
        if values.ndim != 4 or values.shape[0] != 1:
            raise ValueError("image_batch must have shape (1, height, width, channels)")
        output = _invoke(self.model, values)
        probabilities = _probabilities(output, expected_classes=len(self.class_names))
        logits = None
        if self.method in {"energy", "energy_score"}:
            if self._logits_model is None:
                self._logits_model = build_logits_model(self.model)
            logits = extract_logits(self._logits_model, values)
        indices = np.argsort(-probabilities[0], kind="stable")[:5]
        top5 = [
            {
                "index": int(index),
                "breed": str(self.class_names[int(index)]),
                "confidence": float(probabilities[0, index]),
            }
            for index in indices
        ]
        return {
            "probabilities": probabilities[0],
            "top1": top5[0],
            "top5": top5,
            "ood_score": self._score(probabilities, logits=logits),
        }

    def predict(self, image_batch: np.ndarray) -> dict[str, Any]:
        return self.predict_raw(image_batch)


# Names used by integrations that prefer an explicit pipeline name.
UnifiedPredictor = Predictor
InferencePredictor = Predictor
PredictionStatus = DecisionStatus


__all__ = [
    "DEFAULT_CLASS_NAMES_PATH",
    "DEFAULT_MODEL_PATH",
    "InferencePredictor",
    "OODPredictor",
    "PredictionResult",
    "PredictionStatus",
    "Predictor",
    "UnifiedPredictor",
]
