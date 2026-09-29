"""Shared image preprocessing contract for inference callers.

The production model contains MobileNetV2 ``preprocess_input`` internally;
this module deliberately returns raw float32 pixels in ``[0, 255]`` and never
divides by 255.
"""

from __future__ import annotations

import io
import warnings

import numpy as np
from PIL import Image, UnidentifiedImageError


IMAGE_SIZE = (224, 224)
MAX_IMAGE_PIXELS = 20_000_000
ALLOWED_IMAGE_FORMATS = ("JPEG", "PNG", "WEBP")


def preprocess_image(
    file_bytes: bytes, *, max_image_pixels: int = MAX_IMAGE_PIXELS
) -> np.ndarray:
    """Validate, decode and resize with the trained TensorFlow pixel contract.

    JPEG uses the same decoder and non-antialiased bilinear resize as
    ``src.preprocessing``. Pillow validates the upload before allocating the
    TensorFlow image and supplies WebP decoding and EXIF metadata.
    """

    if not isinstance(file_bytes, (bytes, bytearray, memoryview)) or not file_bytes:
        raise ValueError("file_bytes must contain image data")
    with warnings.catch_warnings():
        warnings.simplefilter("error", Image.DecompressionBombWarning)
        try:
            with Image.open(io.BytesIO(file_bytes), formats=ALLOWED_IMAGE_FORMATS) as source:
                if source.width * source.height > max_image_pixels:
                    raise ValueError("Image dimensions exceed the local decode limit.")
                source.verify()
            with Image.open(io.BytesIO(file_bytes), formats=ALLOWED_IMAGE_FORMATS) as source:
                source.load()
                image_format = source.format
                orientation = source.getexif().get(274, 1)
                import tensorflow as tf

                try:
                    if image_format == "JPEG":
                        pixels = tf.image.decode_jpeg(bytes(file_bytes), channels=3)
                    elif image_format == "PNG":
                        pixels = tf.image.decode_png(bytes(file_bytes), channels=3)
                    else:
                        pixels = tf.convert_to_tensor(np.asarray(source.convert("RGB")))
                except tf.errors.InvalidArgumentError as exc:
                    raise ValueError("Invalid or unsupported image") from exc
                # Orient decoded pixels without introducing a second JPEG codec.
                if orientation == 2:
                    pixels = tf.image.flip_left_right(pixels)
                elif orientation == 3:
                    pixels = tf.image.rot90(pixels, 2)
                elif orientation == 4:
                    pixels = tf.image.flip_up_down(pixels)
                elif orientation == 5:
                    pixels = tf.transpose(pixels, (1, 0, 2))
                elif orientation == 6:
                    pixels = tf.image.rot90(pixels, 3)
                elif orientation == 7:
                    pixels = tf.reverse(tf.transpose(pixels, (1, 0, 2)), (0, 1))
                elif orientation == 8:
                    pixels = tf.image.rot90(pixels, 1)
                result = tf.image.resize(pixels, IMAGE_SIZE, method="bilinear", antialias=False).numpy()[None, ...]
        except (UnidentifiedImageError, OSError, SyntaxError) as exc:
            raise ValueError("Invalid or unsupported image") from exc
    return result


preprocess_uploaded_image = preprocess_image
