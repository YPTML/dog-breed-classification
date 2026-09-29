"""Feature extraction from the production classifier's penultimate tensor.

The production model ends in a ``Dense`` classifier named ``predictions``.
This module exposes the tensor that is fed to that classifier as a separate
Keras model. It shares the production model's layers and variables; it never
copies, assigns, trains, or changes the source model's weights.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np


def _tensorflow() -> Any:
    """Import TensorFlow lazily so non-TensorFlow tooling can import this file."""

    try:
        import tensorflow as tf
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise RuntimeError("TensorFlow is required to build a feature extractor") from exc
    return tf


def _single_model_input(model: Any) -> Any:
    inputs = getattr(model, "inputs", None)
    if inputs is None:
        raise ValueError("model must expose a Keras input tensor")
    if isinstance(inputs, (list, tuple)):
        if len(inputs) != 1:
            raise ValueError("feature extraction supports exactly one image input")
        return inputs[0]
    return inputs


def _classifier_layer(model: Any) -> Any:
    layers = list(getattr(model, "layers", ()))
    if not layers:
        raise ValueError("model must contain a final classifier layer")
    classifier = layers[-1]
    if not hasattr(classifier, "input") or not hasattr(classifier, "units"):
        raise ValueError(
            "the final model layer must be a classifier with an input tensor and units"
        )
    return classifier


def _source_layer_name(tensor: Any) -> str | None:
    history = getattr(tensor, "_keras_history", None)
    layer = getattr(history, "operation", None)
    return getattr(layer, "name", None)


def build_feature_extractor(model: Any, layer_name: str | None = None) -> Any:
    """Build a Keras model that returns the classifier's input embeddings.

    ``layer_name`` can select an explicitly audited representation. By
    default, the input tensor of the final classifier is used, which is the
    literal layer immediately before the production classifier (``dropout_2``
    in the deployed model). Call the returned model with ``training=False``.
    """

    tf = _tensorflow()
    model_input = _single_model_input(model)
    classifier = _classifier_layer(model)

    if layer_name is None:
        embedding_tensor = classifier.input
        selected_layer = None
        if isinstance(embedding_tensor, (list, tuple)):
            raise ValueError("the final classifier must have one embedding input")
    else:
        selected_layer = model.get_layer(layer_name)
        embedding_tensor = selected_layer.output

    shape = getattr(embedding_tensor, "shape", None)
    if shape is None or len(shape) != 2 or shape[-1] is None:
        raise ValueError(
            "the embedding tensor must have shape (batch, embedding_dimension)"
        )
    embedding_dim = int(shape[-1])
    if embedding_dim <= 0:
        raise ValueError("embedding dimension must be positive")

    extractor = tf.keras.Model(
        inputs=model_input,
        outputs=embedding_tensor,
        name="dog_breed_feature_extractor",
    )
    # Informational metadata only. Changing trainable flags on shared layers
    # would mutate the source model, so this adapter leaves them untouched.
    extractor.embedding_dim = embedding_dim
    extractor.classifier_name = classifier.name
    extractor.penultimate_layer_name = (
        selected_layer.name
        if selected_layer is not None
        else _source_layer_name(embedding_tensor)
    )
    return extractor


def extract_embeddings(
    extractor: Any,
    image_batch: Any,
    *,
    check_determinism: bool = True,
) -> np.ndarray:
    """Run inference and validate the embedding contract.

    ``check_determinism`` defaults to ``True`` for the audited TASK 07
    validation path. Large offline extraction jobs may disable the duplicate
    forward pass after the extractor has already passed that validation.
    """

    values = np.asarray(image_batch)
    if values.ndim != 4 or values.shape[0] == 0:
        raise ValueError("image_batch must have shape (batch, height, width, channels)")
    result = np.asarray(extractor(values, training=False))
    expected_dim = getattr(extractor, "embedding_dim", None)
    if result.ndim != 2 or result.shape[0] != values.shape[0]:
        raise RuntimeError(
            f"embedding output must have shape (batch, dimension), got {result.shape}"
        )
    if expected_dim is not None and result.shape[1] != int(expected_dim):
        raise RuntimeError(
            f"embedding dimension changed: expected {expected_dim}, got {result.shape[1]}"
        )
    if not np.isfinite(result).all():
        raise RuntimeError("embedding output contains NaN or Inf")
    if check_determinism:
        repeat = np.asarray(extractor(values, training=False))
        if not np.array_equal(result, repeat):
            raise RuntimeError("feature extraction is not deterministic in inference mode")
    return result


def validate_feature_extractor(
    model: Any,
    image_batch: Any,
    *,
    extractor: Any | None = None,
) -> Mapping[str, Any]:
    """Run shape, finiteness, determinism, and immutability checks."""

    feature_model = extractor if extractor is not None else build_feature_extractor(model)
    before = [np.array(weight.numpy(), copy=True) for weight in model.weights]
    embeddings = extract_embeddings(feature_model, image_batch)
    unchanged = all(
        np.array_equal(initial, weight.numpy())
        for initial, weight in zip(before, model.weights)
    )
    if not unchanged:
        raise RuntimeError("feature extraction changed production model weights")
    return {
        "embedding_dimension": int(embeddings.shape[1]),
        "batch_shape": tuple(int(value) for value in embeddings.shape),
        "finite_values": bool(np.isfinite(embeddings).all()),
        "deterministic_inference": True,
        "production_weights_unchanged": True,
    }
