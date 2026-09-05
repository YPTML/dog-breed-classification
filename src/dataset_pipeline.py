"""TensorFlow tf.data.Dataset input pipeline for Dog Breed Identification.

This module builds optimized, reusable tf.data.Dataset pipelines for training,
validation, and test splits with on-the-fly image decoding, resizing (224, 224, 3),
and one-hot label encoding (120 classes). Normalization and data augmentation
are intentionally deferred to future tasks.
"""

from pathlib import Path
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import pandas as pd
import tensorflow as tf

from src.preprocessing import preprocess_image, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS
from src.label_encoder import fit_label_encoder, encode_labels, NUM_CLASSES
from src.data_loader import TRAIN_DIR, PROJECT_ROOT

# Pipeline configuration
BATCH_SIZE = 64
SHUFFLE_SEED = 42
AUTOTUNE = tf.data.AUTOTUNE

# Directory for split metadata CSV files
SPLITS_DIR = PROJECT_ROOT / "outputs" / "splits"
SPLIT_NAMES = ("train", "val", "test")


def load_split_csv(split_name: str) -> pd.DataFrame:
    """Load and validate split metadata CSV from outputs/splits/.

    Parameters
    ----------
    split_name : str
        Name of split: 'train', 'val', or 'test'.

    Returns
    -------
    pd.DataFrame
        Validated DataFrame with columns ['id', 'breed', 'img_name'].

    Raises
    ------
    ValueError
        If split_name is not in ('train', 'val', 'test'), if required columns
        are missing, if null values exist, or if duplicate IDs exist.
    FileNotFoundError
        If the split CSV file does not exist on disk.
    """
    if split_name not in SPLIT_NAMES:
        raise ValueError(
            f"Invalid split_name: '{split_name}'. Must be one of {list(SPLIT_NAMES)}"
        )

    split_path = SPLITS_DIR / f"{split_name}.csv"
    if not split_path.is_file():
        raise FileNotFoundError(
            f"Split file not found for '{split_name}' at: {split_path}"
        )

    df = pd.read_csv(split_path)

    # Validate exact required columns
    required_cols = ["id", "breed", "img_name"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Split '{split_name}' missing required column(s): {missing_cols}. Found: {list(df.columns)}"
        )

    # Validate no null values
    null_counts = df[required_cols].isnull().sum()
    total_nulls = int(null_counts.sum())
    if total_nulls > 0:
        raise ValueError(
            f"Validation failed: split '{split_name}' contains {total_nulls} null values: {null_counts.to_dict()}"
        )

    # Validate no duplicate IDs
    dup_id_count = int(df["id"].duplicated().sum())
    if dup_id_count > 0:
        raise ValueError(
            f"Validation failed: split '{split_name}' contains {dup_id_count} duplicate ID(s)."
        )

    # Return clean DataFrame with exact column order
    return df[required_cols].copy()


def _preprocess_mapped_element(image_path: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    """Helper mapping function to preprocess image tensor and retain label."""
    image = preprocess_image(image_path)
    return image, label


def create_dataset(
    df: pd.DataFrame,
    encoder: Any,
    batch_size: int = BATCH_SIZE,
    training: bool = False,
    shuffle_seed: int = SHUFFLE_SEED,
) -> tf.data.Dataset:
    """Create a tf.data.Dataset pipeline from metadata DataFrame and fitted label encoder.

    Parameters
    ----------
    df : pd.DataFrame
        Split metadata containing 'img_name' and 'breed' columns.
    encoder : LabelBinarizer
        Fitted label encoder covering all 120 dog breed classes.
    batch_size : int, default=64
        Batch size for grouping samples.
    training : bool, default=False
        Whether to shuffle samples for training (with reshuffle_each_iteration=True).
    shuffle_seed : int, default=42
        Random seed for reproducible shuffling.

    Returns
    -------
    tf.data.Dataset
        Batched and prefetched tf.data.Dataset yielding (image_batch, label_batch)
        with image shapes (None, 224, 224, 3) float32 and label shapes (None, 120) float32.

    Raises
    ------
    ValueError
        If df is empty, missing required columns, or contains invalid breed labels.
    """
    if df.empty:
        raise ValueError("DataFrame cannot be empty.")
    if "img_name" not in df.columns or "breed" not in df.columns:
        raise ValueError("DataFrame must contain 'img_name' and 'breed' columns.")

    # 1. Generate full image paths pointing to data/train/
    image_paths = [str(TRAIN_DIR / name) for name in df["img_name"]]

    # 2. Encode labels into one-hot float32 vectors of shape (N, 120)
    one_hot_labels = encode_labels(df["breed"], encoder)

    # 3. Create initial dataset from tensor slices (image_path, one_hot_label)
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, one_hot_labels))

    # 4. Shuffle if training (shuffling paths before map avoids buffering heavy images in RAM)
    if training:
        dataset = dataset.shuffle(
            buffer_size=len(df),
            seed=shuffle_seed,
            reshuffle_each_iteration=True,
        )

    # 5. Map image preprocessing on-the-fly (load JPEG, decode RGB, resize to 224x224, cast float32)
    dataset = dataset.map(
        _preprocess_mapped_element,
        num_parallel_calls=AUTOTUNE,
    )

    # 6. Batch samples without dropping remainders
    dataset = dataset.batch(batch_size, drop_remainder=False)

    # 7. Prefetch next batches for CPU-GPU overlap
    dataset = dataset.prefetch(AUTOTUNE)

    return dataset


def build_datasets(
    batch_size: int = BATCH_SIZE,
) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, Any]:
    """Load train, val, and test splits and build unified tf.data.Datasets.

    Uses a single globally-fitted LabelBinarizer across all splits to ensure
    identical class indexing (0..119).

    Parameters
    ----------
    batch_size : int, default=64
        Batch size for all created datasets.

    Returns
    -------
    Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, LabelBinarizer]
        (train_ds, val_ds, test_ds, encoder)
        where train_ds has training=True, and val_ds, test_ds have training=False.
    """
    train_df = load_split_csv("train")
    val_df = load_split_csv("val")
    test_df = load_split_csv("test")

    # Fit unified encoder on full 120 classes
    encoder = fit_label_encoder()
    if len(encoder.classes_) != NUM_CLASSES:
        raise ValueError(
            f"Expected {NUM_CLASSES} classes in encoder, got {len(encoder.classes_)}"
        )

    train_ds = create_dataset(train_df, encoder, batch_size=batch_size, training=True)
    val_ds = create_dataset(val_df, encoder, batch_size=batch_size, training=False)
    test_ds = create_dataset(test_df, encoder, batch_size=batch_size, training=False)

    return train_ds, val_ds, test_ds, encoder


def run_validation() -> Dict[str, Any]:
    """Execute complete validation suite for Task 2.1 and print structured summary.

    Returns
    -------
    Dict[str, Any]
        Comprehensive dictionary of validation results and metrics.
    """
    print("=" * 70)
    print("TF.DATA PIPELINE VALIDATION")
    print("=" * 70)

    # 1. Load splits
    train_df = load_split_csv("train")
    val_df = load_split_csv("val")
    test_df = load_split_csv("test")

    train_samples = len(train_df)
    val_samples = len(val_df)
    test_samples = len(test_df)
    total_samples = train_samples + val_samples + test_samples

    print(f"Batch size: {BATCH_SIZE}")
    print(f"Shuffle seed: {SHUFFLE_SEED}")
    print(f"Classes: {NUM_CLASSES}\n")
    print(f"Train samples: {train_samples}")
    print(f"Validation samples: {val_samples}")
    print(f"Test samples: {test_samples}")
    print(f"Total labeled samples: {total_samples}")

    # 2. Fit unified encoder and check classes
    encoder = fit_label_encoder()
    encoder_classes = len(encoder.classes_)
    class_idx_min = 0
    class_idx_max = encoder_classes - 1

    # 3. Verify data source paths: no Kaggle test directory referenced
    all_train_paths = [str(TRAIN_DIR / name) for name in train_df["img_name"]]
    all_val_paths = [str(TRAIN_DIR / name) for name in val_df["img_name"]]
    all_test_paths = [str(TRAIN_DIR / name) for name in test_df["img_name"]]

    kaggle_test_refs = sum(
        1 for p in (all_train_paths + all_val_paths + all_test_paths)
        if "data/test" in p.replace("\\", "/")
    )

    # 4. Build datasets
    train_ds, val_ds, test_ds, _ = build_datasets(batch_size=BATCH_SIZE)

    # 5. Expected batch counts vs cardinality
    expected_train_batches = math.ceil(train_samples / BATCH_SIZE)
    expected_val_batches = math.ceil(val_samples / BATCH_SIZE)
    expected_test_batches = math.ceil(test_samples / BATCH_SIZE)

    cardinality_train = int(tf.data.experimental.cardinality(train_ds).numpy())
    cardinality_val = int(tf.data.experimental.cardinality(val_ds).numpy())
    cardinality_test = int(tf.data.experimental.cardinality(test_ds).numpy())

    batches_match = (
        expected_train_batches == cardinality_train
        and expected_val_batches == cardinality_val
        and expected_test_batches == cardinality_test
    )

    print(f"\nTrain batches: {cardinality_train} (expected {expected_train_batches})")
    print(f"Validation batches: {cardinality_val} (expected {expected_val_batches})")
    print(f"Test batches: {cardinality_test} (expected {expected_test_batches})")

    # 6. Check Element Specs
    train_spec = train_ds.element_spec
    val_spec = val_ds.element_spec
    test_spec = test_ds.element_spec

    spec_img_shape = train_spec[0].shape.as_list()
    spec_lbl_shape = train_spec[1].shape.as_list()
    spec_img_dtype = train_spec[0].dtype.name
    spec_lbl_dtype = train_spec[1].dtype.name

    spec_valid = (
        spec_img_shape == [None, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS]
        and spec_lbl_shape == [None, NUM_CLASSES]
        and spec_img_dtype == "float32"
        and spec_lbl_dtype == "float32"
    )

    print(f"\nImage shape: {tuple(spec_img_shape)}")
    print(f"Label shape: {tuple(spec_lbl_shape)}")
    print(f"Image dtype: {spec_img_dtype}")
    print(f"Label dtype: {spec_lbl_dtype}")

    # 7. First train batch inspection
    first_train_images, first_train_labels = next(iter(train_ds))
    b1_img_shape = tuple(first_train_images.shape.as_list())
    b1_lbl_shape = tuple(first_train_labels.shape.as_list())
    b1_img_dtype = first_train_images.dtype.name
    b1_lbl_dtype = first_train_labels.dtype.name

    b1_min = float(tf.reduce_min(first_train_images).numpy())
    b1_max = float(tf.reduce_max(first_train_images).numpy())

    # One-hot integrity in first batch
    lbl_sums = tf.reduce_sum(first_train_labels, axis=1).numpy()
    one_hot_valid = bool(np.allclose(lbl_sums, 1.0))
    active_counts = (first_train_labels.numpy() == 1.0).sum(axis=1)
    exactly_one_active = bool(np.all(active_counts == 1))

    nan_first_batch = bool(
        tf.reduce_any(tf.math.is_nan(first_train_images)).numpy()
        or tf.reduce_any(tf.math.is_nan(first_train_labels)).numpy()
    )
    inf_first_batch = bool(
        tf.reduce_any(tf.math.is_inf(first_train_images)).numpy()
        or tf.reduce_any(tf.math.is_inf(first_train_labels)).numpy()
    )

    # 8. First test batch inspection
    first_test_images, first_test_labels = next(iter(test_ds))
    test_b1_img_shape = tuple(first_test_images.shape.as_list())
    test_b1_lbl_shape = tuple(first_test_labels.shape.as_list())
    test_first_batch_valid = (
        test_b1_img_shape == (BATCH_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)
        and test_b1_lbl_shape == (BATCH_SIZE, NUM_CLASSES)
    )

    # 9. Shuffle verification for train
    iter1_first_labels = next(iter(train_ds))[1].numpy()
    iter2_first_labels = next(iter(train_ds))[1].numpy()
    train_reshuffle_pass = not np.array_equal(iter1_first_labels, iter2_first_labels)

    # 10. Deterministic verification for validation
    val_iter1_labels = next(iter(val_ds))[1].numpy()
    val_iter2_labels = next(iter(val_ds))[1].numpy()
    val_deterministic_pass = np.array_equal(val_iter1_labels, val_iter2_labels)

    # 11. Full train pipeline iteration
    print("\nIterating full train pipeline (8,177 samples)...")
    train_samples_seen = 0
    train_failed_batches = 0
    train_incorrect_img_shapes = 0
    train_incorrect_lbl_shapes = 0
    train_nan_batches = 0
    train_inf_batches = 0

    for batch_idx, (img_b, lbl_b) in enumerate(train_ds):
        try:
            b_size = int(img_b.shape[0])
            train_samples_seen += b_size

            if img_b.shape[1:] != (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS):
                train_incorrect_img_shapes += 1
            if lbl_b.shape[1:] != (NUM_CLASSES,):
                train_incorrect_lbl_shapes += 1

            if bool(tf.reduce_any(tf.math.is_nan(img_b)).numpy()) or bool(tf.reduce_any(tf.math.is_nan(lbl_b)).numpy()):
                train_nan_batches += 1
            if bool(tf.reduce_any(tf.math.is_inf(img_b)).numpy()) or bool(tf.reduce_any(tf.math.is_inf(lbl_b)).numpy()):
                train_inf_batches += 1
        except Exception:
            train_failed_batches += 1
        finally:
            del img_b, lbl_b

    train_full_pass = (
        train_samples_seen == train_samples
        and train_failed_batches == 0
        and train_incorrect_img_shapes == 0
        and train_incorrect_lbl_shapes == 0
        and train_nan_batches == 0
        and train_inf_batches == 0
    )

    # 12. Full validation pipeline iteration
    print("Iterating full validation pipeline (1,022 samples)...")
    val_samples_seen = 0
    val_failed_batches = 0
    val_incorrect_shapes = 0
    val_nan_batches = 0

    for batch_idx, (img_b, lbl_b) in enumerate(val_ds):
        try:
            val_samples_seen += int(img_b.shape[0])
            if img_b.shape[1:] != (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS) or lbl_b.shape[1:] != (NUM_CLASSES,):
                val_incorrect_shapes += 1
            if bool(tf.reduce_any(tf.math.is_nan(img_b)).numpy()):
                val_nan_batches += 1
        except Exception:
            val_failed_batches += 1
        finally:
            del img_b, lbl_b

    val_full_pass = (
        val_samples_seen == val_samples
        and val_failed_batches == 0
        and val_incorrect_shapes == 0
        and val_nan_batches == 0
        and val_deterministic_pass
    )

    # 13. Summary check
    finite_values = (
        not nan_first_batch
        and not inf_first_batch
        and train_nan_batches == 0
        and train_inf_batches == 0
        and val_nan_batches == 0
    )

    print("\n" + "-" * 70)
    print(f"Train shuffle: {'PASS' if train_reshuffle_pass else 'FAIL'}")
    print(f"Validation deterministic: {'PASS' if val_deterministic_pass else 'FAIL'}")
    print(f"One-hot labels: {'PASS' if (one_hot_valid and exactly_one_active) else 'FAIL'}")
    print(f"Finite values: {'PASS' if finite_values else 'FAIL'}")
    print(f"Kaggle test used: {'NO' if kaggle_test_refs == 0 else 'YES'}")
    print("-" * 70)

    overall_pass = (
        batches_match
        and spec_valid
        and (b1_img_shape == (BATCH_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS))
        and (b1_lbl_shape == (BATCH_SIZE, NUM_CLASSES))
        and one_hot_valid
        and exactly_one_active
        and finite_values
        and train_reshuffle_pass
        and val_deterministic_pass
        and train_full_pass
        and val_full_pass
        and test_first_batch_valid
        and (kaggle_test_refs == 0)
    )

    if overall_pass:
        print("\nDataset pipeline validation completed successfully.")
    else:
        print("\nDataset pipeline validation FAILED.")
    print("=" * 70)

    return {
        "status": "PASS" if overall_pass else "FAIL",
        "batch_size": BATCH_SIZE,
        "shuffle_seed": SHUFFLE_SEED,
        "num_classes": NUM_CLASSES,
        "encoder_classes": encoder_classes,
        "class_index_range": f"{class_idx_min}..{class_idx_max}",
        "train_samples": train_samples,
        "val_samples": val_samples,
        "test_samples": test_samples,
        "train_batches": cardinality_train,
        "val_batches": cardinality_val,
        "test_batches": cardinality_test,
        "train_spec": str(train_spec),
        "val_spec": str(val_spec),
        "test_spec": str(test_spec),
        "first_train_img_shape": b1_img_shape,
        "first_train_lbl_shape": b1_lbl_shape,
        "first_train_img_dtype": b1_img_dtype,
        "first_train_lbl_dtype": b1_lbl_dtype,
        "first_train_img_min": b1_min,
        "first_train_img_max": b1_max,
        "one_hot_valid": one_hot_valid and exactly_one_active,
        "train_samples_seen": train_samples_seen,
        "train_failed_batches": train_failed_batches,
        "train_incorrect_img_shapes": train_incorrect_img_shapes,
        "train_incorrect_lbl_shapes": train_incorrect_lbl_shapes,
        "val_samples_seen": val_samples_seen,
        "val_deterministic_pass": val_deterministic_pass,
        "test_first_batch_valid": test_first_batch_valid,
        "train_reshuffle_pass": train_reshuffle_pass,
        "kaggle_test_refs": kaggle_test_refs,
        "overall_pass": overall_pass,
    }


if __name__ == "__main__":
    run_validation()
