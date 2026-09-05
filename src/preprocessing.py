"""Image preprocessing pipeline for Dog Breed Identification.

This module provides reusable TensorFlow-based preprocessing functions to read,
decode, resize, and prepare JPEG images into standard tensors (224, 224, 3)
with float32 dtype without normalization or data augmentation.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import tensorflow as tf

# Image dimension configurations
IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224
IMAGE_CHANNELS = 3
IMAGE_SIZE = (IMAGE_HEIGHT, IMAGE_WIDTH)


def load_image(image_path: Union[str, Path, tf.Tensor]) -> tf.Tensor:
    """Read an image file from disk and decode it as a 3-channel RGB JPEG tensor.

    Parameters
    ----------
    image_path : str, Path, or tf.Tensor
        Path to the JPEG image file.

    Returns
    -------
    tf.Tensor
        Decoded image tensor with shape (H, W, 3) and dtype tf.uint8.
    """
    if isinstance(image_path, Path):
        image_path = str(image_path)
    image_bytes = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image_bytes, channels=IMAGE_CHANNELS)
    return image


def resize_image(image: tf.Tensor) -> tf.Tensor:
    """Resize image tensor to target dimensions (224, 224, 3) and cast to tf.float32.

    Parameters
    ----------
    image : tf.Tensor
        Input image tensor with shape (H, W, 3).

    Returns
    -------
    tf.Tensor
        Resized image tensor with shape (224, 224, 3) and dtype tf.float32.
    """
    resized = tf.image.resize(image, IMAGE_SIZE)
    resized = tf.cast(resized, tf.float32)
    resized = tf.ensure_shape(resized, (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS))
    return resized


def preprocess_image(image_path: Union[str, Path, tf.Tensor]) -> tf.Tensor:
    """Preprocess an image from file path: load, decode JPEG RGB, resize to 224x224,

    and return a float32 tensor with shape (224, 224, 3) and pixel values in [0, 255].

    Parameters
    ----------
    image_path : str, Path, or tf.Tensor
        Path to the JPEG image file.

    Returns
    -------
    tf.Tensor
        Preprocessed image tensor with shape (224, 224, 3) and dtype tf.float32.
    """
    image = load_image(image_path)
    preprocessed = resize_image(image)
    return preprocessed


def run_validation(
    sample_count: int = 10,
    random_count: int = 100,
    random_state: int = 42,
) -> Dict[str, Union[bool, int, float, List[Dict]]]:
    """Execute complete validation suite for preprocessing pipeline.

    Parameters
    ----------
    sample_count : int
        Number of diverse sample images to evaluate in detail (default: 10).
    random_count : int
        Number of randomly sampled images to validate (default: 100).
    random_state : int
        Random seed for sampling (default: 42).

    Returns
    -------
    dict
        Validation results and metrics summary.
    """
    from src.data_loader import load_labels, TRAIN_DIR

    print("=" * 70)
    print("PREPROCESSING VALIDATION")
    print("=" * 70)
    print(f"Target shape: ({IMAGE_HEIGHT}, {IMAGE_WIDTH}, {IMAGE_CHANNELS})")
    print("Target dtype: float32")
    print(f"Sample images tested: {sample_count}")
    print(f"Random images tested: {random_count}\n")

    df = load_labels()

    # 1. Diverse sample validation (10 images with diverse original shapes)
    print("--- 1. Sample Images Validation (Diverse Dimensions) ---")
    sample_results = []
    seen_shapes = set()
    sample_rows = []

    # First collect images with different dimensions
    for _, row in df.iterrows():
        img_path = TRAIN_DIR / row["img_name"]
        raw = tf.io.read_file(str(img_path))
        orig_shape = tuple(tf.image.decode_jpeg(raw, channels=IMAGE_CHANNELS).shape.as_list())
        if orig_shape not in seen_shapes:
            seen_shapes.add(orig_shape)
            sample_rows.append((row["img_name"], img_path, orig_shape))
        if len(sample_rows) >= sample_count:
            break

    # If diverse shapes fewer than sample_count, fill up from df
    if len(sample_rows) < sample_count:
        existing_names = {r[0] for r in sample_rows}
        for _, row in df.iterrows():
            if row["img_name"] not in existing_names:
                img_path = TRAIN_DIR / row["img_name"]
                raw = tf.io.read_file(str(img_path))
                orig_shape = tuple(tf.image.decode_jpeg(raw, channels=IMAGE_CHANNELS).shape.as_list())
                sample_rows.append((row["img_name"], img_path, orig_shape))
            if len(sample_rows) >= sample_count:
                break

    print(f"{'File':<36} | {'Original shape':<15} | {'Output shape':<14} | {'Dtype':<8} | {'Min':>6} | {'Max':>6} | {'Status'}")
    print("-" * 105)

    sample_all_pass = True
    for img_name, img_path, orig_shape in sample_rows:
        tensor = preprocess_image(img_path)
        out_shape = tuple(tensor.shape.as_list())
        dtype_str = tensor.dtype.name
        min_val = float(tf.reduce_min(tensor).numpy())
        max_val = float(tf.reduce_max(tensor).numpy())

        is_shape_ok = out_shape == (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)
        is_dtype_ok = dtype_str == "float32"
        is_finite = not (tf.reduce_any(tf.math.is_nan(tensor)) or tf.reduce_any(tf.math.is_inf(tensor)))
        is_range_ok = 0.0 <= min_val <= max_val <= 255.0

        status = "PASS" if (is_shape_ok and is_dtype_ok and is_finite and is_range_ok) else "FAIL"
        if status == "FAIL":
            sample_all_pass = False

        sample_results.append({
            "file": img_name,
            "original_shape": orig_shape,
            "output_shape": out_shape,
            "dtype": dtype_str,
            "min": round(min_val, 1),
            "max": round(max_val, 1),
            "status": status,
        })
        print(f"{img_name:<36} | {str(orig_shape):<15} | {str(out_shape):<14} | {dtype_str:<8} | {min_val:>6.1f} | {max_val:>6.1f} | {status}")

    # 2. Random 100 images validation
    print(f"\n--- 2. Random Batch Validation ({random_count} images, seed={random_state}) ---")
    random_sample_df = df.sample(n=random_count, random_state=random_state)
    success_count = 0
    fail_count = 0
    shape_ok_count = 0
    dtype_ok_count = 0
    nan_count = 0
    inf_count = 0

    for _, row in random_sample_df.iterrows():
        img_path = TRAIN_DIR / row["img_name"]
        try:
            tensor = preprocess_image(img_path)
            out_shape = tuple(tensor.shape.as_list())
            dtype_str = tensor.dtype.name

            if out_shape == (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS):
                shape_ok_count += 1
            if dtype_str == "float32":
                dtype_ok_count += 1

            if bool(tf.reduce_any(tf.math.is_nan(tensor)).numpy()):
                nan_count += 1
            if bool(tf.reduce_any(tf.math.is_inf(tensor)).numpy()):
                inf_count += 1

            success_count += 1
            # Release tensor reference immediately
            del tensor
        except Exception as err:
            fail_count += 1
            print(f"Error processing {row['img_name']}: {err}")

    # 3. Determinism check
    print("\n--- 3. Determinism Validation ---")
    test_img_path = TRAIN_DIR / sample_rows[0][0]
    tensor_a = preprocess_image(test_img_path)
    tensor_b = preprocess_image(test_img_path)
    max_abs_diff = float(tf.reduce_max(tf.abs(tensor_a - tensor_b)).numpy())
    deterministic = bool(max_abs_diff == 0.0)
    print(f"Tested image: {sample_rows[0][0]}")
    print(f"Max absolute difference: {max_abs_diff}")
    print(f"Deterministic: {'PASS' if deterministic else 'FAIL'}")
    del tensor_a, tensor_b

    # 4. Invalid path check
    print("\n--- 4. Invalid Path Handling ---")
    invalid_path = TRAIN_DIR / "this_file_does_not_exist.jpg"
    exception_caught = False
    exception_type = ""
    try:
        _ = preprocess_image(invalid_path)
    except Exception as exc:
        exception_caught = True
        exception_type = type(exc).__name__

    invalid_path_pass = exception_caught
    print(f"Tested invalid path: {invalid_path}")
    print(f"Exception raised: {exception_type if exception_caught else 'None'}")
    print(f"Invalid path handling: {'PASS' if invalid_path_pass else 'FAIL'}")

    # Summary report
    all_shape_pass = (shape_ok_count == random_count) and sample_all_pass
    all_dtype_pass = (dtype_ok_count == random_count) and sample_all_pass
    all_finite_pass = (nan_count == 0) and (inf_count == 0)

    print("\n" + "=" * 70)
    print("PREPROCESSING VALIDATION SUMMARY")
    print("=" * 70)
    print(f"Target shape: ({IMAGE_HEIGHT}, {IMAGE_WIDTH}, {IMAGE_CHANNELS})")
    print("Target dtype: float32")
    print(f"\nSample images tested: {sample_count}")
    print(f"Random images tested: {random_count}")
    print(f"\nSuccessful: {success_count}")
    print(f"Failed: {fail_count}")
    print(f"\nShape validation: {'PASS' if all_shape_pass else 'FAIL'}")
    print(f"Dtype validation: {'PASS' if all_dtype_pass else 'FAIL'}")
    print(f"Finite values: {'PASS' if all_finite_pass else 'FAIL'}")
    print(f"Deterministic: {'PASS' if deterministic else 'FAIL'}")
    print(f"Invalid path handling: {'PASS' if invalid_path_pass else 'FAIL'}")

    overall_success = (
        success_count == random_count
        and fail_count == 0
        and all_shape_pass
        and all_dtype_pass
        and all_finite_pass
        and deterministic
        and invalid_path_pass
    )

    if overall_success:
        print("\nPreprocessing validation completed successfully.")
    else:
        print("\nPreprocessing validation FAILED.")
    print("=" * 70)

    return {
        "success_count": success_count,
        "fail_count": fail_count,
        "shape_ok_count": shape_ok_count,
        "dtype_ok_count": dtype_ok_count,
        "nan_count": nan_count,
        "inf_count": inf_count,
        "deterministic": deterministic,
        "max_abs_diff": max_abs_diff,
        "invalid_path_pass": invalid_path_pass,
        "sample_results": sample_results,
        "overall_success": overall_success,
    }


if __name__ == "__main__":
    run_validation()
