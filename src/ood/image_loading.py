"""Common image decoding for offline OOD datasets, including WebP."""

from __future__ import annotations

import io
from typing import Any

import numpy as np
from PIL import Image


def _decode_webp(contents: np.ndarray) -> np.ndarray:
    # Eager numpy_function uses fixed-width byte strings. item() on these
    # strips trailing NUL bytes, including WebP's required RIFF padding.
    contents = np.asarray(contents)
    payload = contents.item() if contents.dtype.kind == "O" else contents.tobytes()
    with Image.open(io.BytesIO(payload), formats=("WEBP",)) as source:
        return np.asarray(source.convert("RGB"), dtype=np.uint8)


def preprocess_ood_image(path: Any) -> Any:
    """Read RGB pixels and apply the existing TensorFlow resize contract.

    TensorFlow's decoder handles JPEG/PNG/BMP but not WebP. Select WebP by
    file signature so even files with misleading extensions decode correctly.
    Keep other formats on their existing decoder to preserve calibrated scores.
    """

    import tensorflow as tf

    contents = tf.io.read_file(str(path) if hasattr(path, "__fspath__") else path)
    is_webp = tf.logical_and(
        tf.equal(tf.strings.substr(contents, 0, 4), b"RIFF"),
        tf.equal(tf.strings.substr(contents, 8, 4), b"WEBP"),
    )
    image = tf.cond(
        is_webp,
        lambda: tf.numpy_function(_decode_webp, [contents], tf.uint8),
        lambda: tf.io.decode_image(contents, channels=3, expand_animations=False),
    )
    image = tf.ensure_shape(image, (None, None, 3))
    image = tf.image.resize(image, (224, 224))
    return tf.ensure_shape(tf.cast(image, tf.float32), (224, 224, 3))
