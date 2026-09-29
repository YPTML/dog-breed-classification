"""Outlier Exposure (OE) Training Module for Dog Breed Classifier.

This module implements Outlier Exposure fine-tuning for the 120-class dog breed classifier.
Model remains strictly 120 outputs (no 121st class, no binary head).
Loss formulation:
    L_total = L_ID + lambda_oe * L_OE
where L_ID is categorical cross-entropy over 120 dog breeds and L_OE is cross-entropy
against a uniform distribution u_i = 1/120 for non-dog OOD images.
"""

from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Enable line-buffered stdout for real-time background logging
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass

# ---------------------------------------------------------------------------
# 0. Environment Setup for WSL2 / Linux GPU Library Resolution
# ---------------------------------------------------------------------------
ld_path = os.environ.get("LD_LIBRARY_PATH", "")
if sys.platform.startswith("linux") and "/usr/lib/wsl/lib" not in ld_path:
    site_packages = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(sys.executable))),
        "lib",
        f"python{sys.version_info.major}.{sys.version_info.minor}",
        "site-packages",
        "nvidia",
    )
    if os.path.isdir(site_packages):
        nv_libs = ":".join(glob.glob(os.path.join(site_packages, "*", "lib")))
        new_ld = f"{nv_libs}:/usr/lib/wsl/lib:{ld_path}" if ld_path else f"{nv_libs}:/usr/lib/wsl/lib"
        os.environ["LD_LIBRARY_PATH"] = new_ld
        try:
            os.execv(sys.executable, [sys.executable] + sys.argv)
        except OSError:
            pass

# ---------------------------------------------------------------------------
# 1. Project Root & Imports
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import tensorflow as tf
from tensorflow.keras import mixed_precision

from src.preprocessing import IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS

# Register preprocess_input function with Keras custom objects
tf.keras.utils.get_custom_objects()["preprocess_input"] = (
    tf.keras.applications.mobilenet_v2.preprocess_input
)

# ---------------------------------------------------------------------------
# Default Paths & Constants
# ---------------------------------------------------------------------------
NUM_CLASSES: int = 120
EXPECTED_PROD_MODEL_HASH: str = "4695eae55c8464c1a6be3f147828f901259fb2c2541c23c2e36e8b4eb95d5f3d"
DEFAULT_MODEL_PATH: Path = PROJECT_ROOT / "outputs" / "deployment_ready" / "dog_breed_model.keras"
CLASS_NAMES_PATH: Path = PROJECT_ROOT / "outputs" / "deployment_ready" / "class_names.json"
SPLITS_V2_DIR: Path = PROJECT_ROOT / "outputs" / "splits_v2"
TRAIN_CSV_PATH: Path = SPLITS_V2_DIR / "train.csv"
VAL_CSV_PATH: Path = SPLITS_V2_DIR / "val.csv"
DATA_TRAIN_DIR: Path = PROJECT_ROOT / "data" / "train"

# OOD V2 Paths
DEFAULT_OOD_TRAIN_DIR: Path = PROJECT_ROOT / "data" / "ood_oe_v2" / "train"
DEFAULT_OOD_CALIBRATION_DIR: Path = PROJECT_ROOT / "data" / "ood_oe_v2" / "calibration"
OOD_V2_MANIFEST_PATH: Path = PROJECT_ROOT / "metadata" / "ood_oe_v2_manifest.csv"

# Protected Final Test Paths - Strictly Prohibited During Training/Smoke-testing
PROTECTED_PATHS: Tuple[Path, ...] = (
    PROJECT_ROOT / "outputs" / "splits_v2" / "test.csv",
    PROJECT_ROOT / "data" / "ood_oe_v2" / "test",
    PROJECT_ROOT / "data" / "ood_oe_v2" / "hard_test",
)
PROTECTED_PATTERNS: Tuple[str, ...] = (
    "splits_v2/test.csv",
    "splits_v2\\test.csv",
    "ood_oe_v2/test",
    "ood_oe_v2\\test",
    "ood_oe_v2/hard_test",
    "ood_oe_v2\\hard_test",
)

# Checkpoint & Output Paths
OE_OUTPUT_DIR: Path = PROJECT_ROOT / "outputs" / "models" / "oe_v2"
OE_BEST_MODEL_PATH: Path = OE_OUTPUT_DIR / "stage_a_best.keras"
OE_LAST_MODEL_PATH: Path = OE_OUTPUT_DIR / "stage_a_last.keras"
OE_STAGE_B_BEST_MODEL_PATH: Path = OE_OUTPUT_DIR / "stage_b_best.keras"
OE_STAGE_B_LAST_MODEL_PATH: Path = OE_OUTPUT_DIR / "stage_b_last.keras"
OE_METRICS_DIR: Path = PROJECT_ROOT / "outputs" / "metrics" / "oe_v2"
OE_PLOTS_DIR: Path = PROJECT_ROOT / "outputs" / "plots" / "oe_v2"
OE_REPORTS_DIR: Path = PROJECT_ROOT / "outputs" / "reports"

EXPECTED_STAGE_A_MODEL_HASH: str = "e4508564a9389250e265cae6ea6fed0e1a0edf8497d62cd9c8a44342bdd45e61"


# ---------------------------------------------------------------------------
# 2. Runtime Safety & Protection Guards
# ---------------------------------------------------------------------------
def assert_not_protected_path(target_path: Path | str) -> None:
    """Fast, zero-IO runtime guard preventing access to protected test sets."""
    norm = str(target_path).lower().replace("/", "\\")
    for pat in PROTECTED_PATTERNS:
        if pat.lower().replace("/", "\\") in norm:
            raise PermissionError(
                f"[GUARD TRIGGERED] Access to protected test path is strictly prohibited during "
                f"training, validation, or smoke-testing: {target_path}"
            )


def get_file_sha256(filepath: Path) -> str:
    """Calculate SHA256 checksum of a file."""
    if not filepath.is_file():
        raise FileNotFoundError(f"File not found: {filepath}")
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest().lower()


def get_dataset_counts(
    ood_train_dir: Path = DEFAULT_OOD_TRAIN_DIR,
    ood_cal_dir: Path = DEFAULT_OOD_CALIBRATION_DIR,
) -> Dict[str, int]:
    """Retrieve sample counts for ID and OOD train and validation/calibration splits."""
    id_train_count = 0
    if TRAIN_CSV_PATH.is_file():
        with open(TRAIN_CSV_PATH, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            id_train_count = sum(1 for _ in reader) - 1

    id_val_count = 0
    if VAL_CSV_PATH.is_file():
        with open(VAL_CSV_PATH, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            id_val_count = sum(1 for _ in reader) - 1

    valid_exts = {".jpg", ".jpeg", ".png", ".webp"}
    ood_train_count = 0
    if ood_train_dir.is_dir():
        ood_train_count = sum(
            1 for p in ood_train_dir.rglob("*")
            if p.suffix.lower() in valid_exts and p.is_file()
        )

    ood_cal_count = 0
    if ood_cal_dir.is_dir():
        ood_cal_count = sum(
            1 for p in ood_cal_dir.rglob("*")
            if p.suffix.lower() in valid_exts and p.is_file()
        )

    return {
        "id_train_count": id_train_count,
        "id_val_count": id_val_count,
        "ood_train_count": ood_train_count,
        "ood_cal_count": ood_cal_count,
    }


# ---------------------------------------------------------------------------
# 3. GPU & Precision Configuration
# ---------------------------------------------------------------------------
def configure_gpu(allow_cpu: bool = False) -> Tuple[List[tf.config.PhysicalDevice], str]:
    """Configure GPU memory growth immediately after importing TensorFlow."""
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception as exc:
            print(f"[WARN] Failed to set memory growth for {gpu}: {exc}", flush=True)
    print("GPUs:", gpus, flush=True)

    gpu_name = "CPU"
    if gpus:
        try:
            details = tf.config.experimental.get_device_details(gpus[0])
            gpu_name = details.get("device_name", "NVIDIA GPU")
            print(f"GPU Name: {gpu_name}", flush=True)
        except Exception:
            gpu_name = "NVIDIA GPU (details unavailable)"
            print(f"GPU Name: {gpu_name}", flush=True)
    elif not allow_cpu:
        raise RuntimeError("FAIL: No GPU detected. Run under WSL2 with GPU or pass --allow-cpu.")
    else:
        print("[INFO] Running on CPU as --allow-cpu was specified.", flush=True)

    return gpus, gpu_name


def configure_mixed_precision(enabled: bool = True) -> str:
    """Configure mixed precision policy for Turing / GTX 1650."""
    if enabled:
        policy = mixed_precision.Policy("mixed_float16")
        mixed_precision.set_global_policy(policy)
        policy_name = str(mixed_precision.global_policy().name)
        print(f"Mixed precision policy: {policy_name}", flush=True)
        return policy_name
    else:
        policy = mixed_precision.Policy("float32")
        mixed_precision.set_global_policy(policy)
        policy_name = "float32"
        print("Mixed precision policy: float32", flush=True)
        return policy_name


# ---------------------------------------------------------------------------
# 4. Model Loading & Verification
# ---------------------------------------------------------------------------
def load_and_verify_model(model_path: Path) -> tf.keras.Model:
    """Load production model, verify input/output shapes and 120 breed classes."""
    if not model_path.is_file():
        raise FileNotFoundError(f"Model file not found at: {model_path}")

    print(f"[MODEL] Loading model from: {model_path}", flush=True)
    tf.keras.utils.get_custom_objects()["preprocess_input"] = (
        tf.keras.applications.mobilenet_v2.preprocess_input
    )
    model = tf.keras.models.load_model(str(model_path), compile=False)

    # Input shape validation
    input_shape = model.input_shape
    if isinstance(input_shape, list):
        input_shape = input_shape[0]
    expected_input = (None, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)
    if input_shape != expected_input:
        raise ValueError(f"Unexpected model input shape: {input_shape}, expected: {expected_input}")

    # Output shape validation
    output_shape = model.output_shape
    if isinstance(output_shape, list):
        output_shape = output_shape[0]
    expected_output = (None, NUM_CLASSES)
    if output_shape != expected_output:
        raise ValueError(
            f"Model output shape must be exactly {expected_output} (120 classes). Got: {output_shape}"
        )

    print(f"[MODEL] Verified input shape: {input_shape}, output shape: {output_shape}", flush=True)
    print(f"[MODEL] Number of classes: {NUM_CLASSES} (Strictly preserved, no class 121)", flush=True)
    return model


# ---------------------------------------------------------------------------
# 5. Class Mapping
# ---------------------------------------------------------------------------
def load_class_mapping(mapping_path: Path = CLASS_NAMES_PATH) -> Tuple[Dict[str, int], Dict[int, str]]:
    """Load canonical class mapping {0: breed_name} and reverse {breed_name: 0}."""
    if not mapping_path.is_file():
        raise FileNotFoundError(f"Class names mapping not found at: {mapping_path}")

    with open(mapping_path, "r", encoding="utf-8") as f:
        raw_mapping = json.load(f)

    breed_to_idx: Dict[str, int] = {}
    idx_to_breed: Dict[int, str] = {}
    for k, v in raw_mapping.items():
        idx = int(k)
        breed_to_idx[v] = idx
        idx_to_breed[idx] = v

    if len(breed_to_idx) != NUM_CLASSES:
        raise ValueError(f"Expected {NUM_CLASSES} classes, found {len(breed_to_idx)}")

    return breed_to_idx, idx_to_breed


# ---------------------------------------------------------------------------
# 6. Outlier Exposure Loss Functions
# ---------------------------------------------------------------------------
def compute_id_loss(
    y_true: tf.Tensor,
    y_pred: tf.Tensor,
    eps: float = 1e-7,
) -> tf.Tensor:
    """Compute categorical cross-entropy loss over 120 breeds for ID samples.

    Always executed in float32 for numerical stability.
    """
    y_pred_f32 = tf.cast(y_pred, tf.float32)
    y_true_f32 = tf.cast(y_true, tf.float32)
    p_clamped = tf.clip_by_value(y_pred_f32, eps, 1.0 - eps)
    sample_losses = -tf.reduce_sum(y_true_f32 * tf.math.log(p_clamped), axis=-1)
    return tf.reduce_mean(sample_losses)


def compute_oe_loss(
    y_pred_ood: tf.Tensor,
    eps: float = 1e-7,
) -> tf.Tensor:
    """Compute Uniform Outlier Exposure loss for OOD samples.

    Target distribution: u_i = 1 / NUM_CLASSES (uniform distribution).
    Cross-entropy CE(u, p) = - 1/C * sum(log(p_c)) = - mean(log(p_c)).
    Always executed in float32 for numerical stability.
    """
    p_ood_f32 = tf.cast(y_pred_ood, tf.float32)
    p_clamped = tf.clip_by_value(p_ood_f32, eps, 1.0 - eps)
    sample_oe_losses = -tf.reduce_mean(tf.math.log(p_clamped), axis=-1)
    return tf.reduce_mean(sample_oe_losses)


def compute_id_metrics(
    y_true: tf.Tensor,
    y_pred: tf.Tensor,
) -> Tuple[tf.Tensor, tf.Tensor]:
    """Compute Top-1 and Top-5 accuracy in float32."""
    y_pred_f32 = tf.cast(y_pred, tf.float32)
    y_true_idx = tf.cast(tf.argmax(y_true, axis=-1), tf.int32)
    y_pred_idx = tf.cast(tf.argmax(y_pred_f32, axis=-1), tf.int32)

    top1 = tf.reduce_mean(tf.cast(tf.equal(y_true_idx, y_pred_idx), tf.float32))
    in_top5 = tf.math.in_top_k(
        predictions=y_pred_f32,
        targets=y_true_idx,
        k=min(5, NUM_CLASSES),
    )
    top5 = tf.reduce_mean(tf.cast(in_top5, tf.float32))
    return top1, top5


# ---------------------------------------------------------------------------
# 7. GradientTape Training Step
# ---------------------------------------------------------------------------
@tf.function
def train_step(
    model: tf.keras.Model,
    optimizer: tf.keras.optimizers.Optimizer,
    x_id: tf.Tensor,
    y_id: tf.Tensor,
    x_ood: tf.Tensor,
    lambda_oe: float = 0.5,
) -> Dict[str, tf.Tensor]:
    """Execute one mixed ID + OOD training step with Outlier Exposure."""
    with tf.GradientTape() as tape:
        p_id = model(x_id, training=True)
        p_ood = model(x_ood, training=True)

        id_loss = compute_id_loss(y_id, p_id)
        oe_loss = compute_oe_loss(p_ood)
        total_loss = id_loss + lambda_oe * oe_loss

    trainable_vars = model.trainable_variables
    grads = tape.gradient(total_loss, trainable_vars)
    optimizer.apply_gradients(zip(grads, trainable_vars))

    top1, top5 = compute_id_metrics(y_id, p_id)
    return {
        "id_loss": id_loss,
        "oe_loss": oe_loss,
        "total_loss": total_loss,
        "id_top1": top1,
        "id_top5": top5,
    }


# ---------------------------------------------------------------------------
# 8. Data Loading & Preprocessing Utilities
# ---------------------------------------------------------------------------
def load_and_preprocess_image(image_path: Path | str) -> tf.Tensor:
    """Read and decode image (JPEG/PNG), resize to (224, 224, 3) float32."""
    image_bytes = tf.io.read_file(str(image_path))
    image = tf.image.decode_jpeg(image_bytes, channels=IMAGE_CHANNELS)
    image = tf.image.resize(image, (IMAGE_HEIGHT, IMAGE_WIDTH))
    image = tf.cast(image, tf.float32)
    image = tf.ensure_shape(image, (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS))
    return image


def load_id_samples(
    csv_path: Path,
    breed_to_idx: Dict[str, int],
    count: int = 4,
) -> Tuple[tf.Tensor, tf.Tensor]:
    """Load a small batch of ID images and one-hot labels for training/testing."""
    assert_not_protected_path(csv_path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"ID split CSV not found: {csv_path}")

    images: List[tf.Tensor] = []
    labels: List[int] = []

    with open(csv_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            img_file = DATA_TRAIN_DIR / row["img_name"]
            assert_not_protected_path(img_file)
            if not img_file.is_file():
                raise FileNotFoundError(f"ID image file not found: {img_file}")
            img_tensor = load_and_preprocess_image(img_file)
            images.append(img_tensor)
            labels.append(breed_to_idx[row["breed"]])
            if len(images) >= count:
                break

    batch_x = tf.stack(images, axis=0)
    batch_y = tf.one_hot(labels, depth=NUM_CLASSES, dtype=tf.float32)
    return batch_x, batch_y


def load_ood_samples(
    ood_dir: Path,
    count: int = 4,
) -> tf.Tensor:
    """Load a small batch of OOD images for training/testing."""
    assert_not_protected_path(ood_dir)
    if not ood_dir.is_dir():
        raise FileNotFoundError(f"OOD directory not found: {ood_dir}")

    image_paths: List[Path] = []
    valid_exts = {".jpg", ".jpeg", ".png", ".webp"}
    for p in ood_dir.rglob("*"):
        if p.suffix.lower() in valid_exts and p.is_file():
            assert_not_protected_path(p)
            image_paths.append(p)
            if len(image_paths) >= count:
                break

    if len(image_paths) < count:
        raise FileNotFoundError(
            f"Could not find at least {count} valid OOD images in: {ood_dir}"
        )

    images: List[tf.Tensor] = []
    for p in image_paths[:count]:
        img_tensor = load_and_preprocess_image(p)
        images.append(img_tensor)

    return tf.stack(images, axis=0)


# ---------------------------------------------------------------------------
# 9. Smoke Test Execution
# ---------------------------------------------------------------------------
def run_smoke_test(
    model_path: Path = DEFAULT_MODEL_PATH,
    id_batch_size: int = 4,
    ood_batch_size: int = 4,
    lambda_oe: float = 0.5,
    allow_cpu: bool = False,
    mixed_prec: bool = True,
    custom_ood_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Execute smoke test without modifying the production model."""
    print("\n" + "=" * 80, flush=True)
    print("RUNNING OUTLIER EXPOSURE SMOKE TEST (NO TRAINING ON PRODUCTION ARTIFACT)", flush=True)
    print("=" * 80, flush=True)

    # 1. Protected paths runtime guard checks
    ood_source = custom_ood_dir if custom_ood_dir is not None else DEFAULT_OOD_TRAIN_DIR
    assert_not_protected_path(ood_source)
    assert_not_protected_path(DEFAULT_OOD_CALIBRATION_DIR)
    assert_not_protected_path(TRAIN_CSV_PATH)
    assert_not_protected_path(VAL_CSV_PATH)

    # 2. Production model hash BEFORE smoke test
    hash_before = get_file_sha256(model_path)
    print(f"[MODEL HASH] Production model SHA256 before smoke test: {hash_before}", flush=True)

    # 3. Dataset Counts
    counts = get_dataset_counts(ood_train_dir=ood_source, ood_cal_dir=DEFAULT_OOD_CALIBRATION_DIR)
    print(f"[DATASETS] ID Train Count:        {counts['id_train_count']}", flush=True)
    print(f"[DATASETS] ID Val Count:          {counts['id_val_count']}", flush=True)
    print(f"[DATASETS] OOD Train Count:       {counts['ood_train_count']} ({ood_source})", flush=True)
    print(f"[DATASETS] OOD Calibration Count: {counts['ood_cal_count']} ({DEFAULT_OOD_CALIBRATION_DIR})", flush=True)

    # 4. GPU Detection
    gpus, gpu_name = configure_gpu(allow_cpu=allow_cpu)
    gpu_detected = len(gpus) > 0
    smoke_gpu_device = "/device:GPU:0" if gpu_detected else "/device:CPU:0"

    # 5. Mixed Precision
    mp_policy = configure_mixed_precision(enabled=mixed_prec)

    # 6. Model Load & Shape Verification
    prod_model = load_and_verify_model(model_path)
    model_input_shape = tuple(prod_model.input_shape)
    model_output_shape = tuple(prod_model.output_shape)

    # Clone model architecture & weights to disposable instance for safe smoke test
    print("[SMOKE TEST] Creating disposable model instance for step verification...", flush=True)
    disposable_model = tf.keras.models.clone_model(prod_model)
    disposable_model.set_weights(prod_model.get_weights())
    for layer in disposable_model.layers:
        if "mobilenetv2" in layer.name.lower() or isinstance(layer, tf.keras.Model):
            layer.trainable = False

    # 7. Load tiny ID batch
    print(f"[SMOKE TEST] Loading ID batch (size={id_batch_size}) from {TRAIN_CSV_PATH}...", flush=True)
    breed_to_idx, _ = load_class_mapping()
    x_id, y_id = load_id_samples(TRAIN_CSV_PATH, breed_to_idx, count=id_batch_size)
    print(f"[SMOKE TEST] ID batch loaded: images shape {x_id.shape}, labels shape {y_id.shape}", flush=True)

    # 8. Load tiny OOD V2 batch
    if not ood_source.is_dir():
        raise FileNotFoundError(
            f"OOD train directory does not exist: {ood_source}. Ensure data/ood_oe_v2/train is present."
        )

    print(f"[SMOKE TEST] Loading OOD batch (size={ood_batch_size}) from {ood_source}...", flush=True)
    x_ood = load_ood_samples(ood_source, count=ood_batch_size)
    print(f"[SMOKE TEST] OOD batch loaded: images shape {x_ood.shape}", flush=True)

    # 9. Execute forward pass & optimization step
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

    try:
        step_results = train_step(
            model=disposable_model,
            optimizer=optimizer,
            x_id=x_id,
            y_id=y_id,
            x_ood=x_ood,
            lambda_oe=lambda_oe,
        )
    except tf.errors.ResourceExhaustedError as oom_err:
        print("\n" + "!" * 80, flush=True)
        print(f"[OOM DETECTED] ResourceExhaustedError: GPU VRAM exhausted at ID={id_batch_size}, OOD={ood_batch_size}.", flush=True)
        print("[RECOMMENDATION] Reduce batch size to: --id-batch-size 2 --ood-batch-size 2", flush=True)
        print("!" * 80 + "\n", flush=True)
        raise oom_err

    id_loss = float(step_results["id_loss"].numpy())
    oe_loss = float(step_results["oe_loss"].numpy())
    total_loss = float(step_results["total_loss"].numpy())
    id_top1 = float(step_results["id_top1"].numpy())
    id_top5 = float(step_results["id_top5"].numpy())

    post_step_preds = disposable_model(x_id, training=False)
    pred_shape = tuple(post_step_preds.shape)
    shape_preserved = (pred_shape == (id_batch_size, NUM_CLASSES))

    losses_finite = (
        math.isfinite(id_loss)
        and math.isfinite(oe_loss)
        and math.isfinite(total_loss)
    )

    # 10. Check GPU Memory
    peak_mem_mb = None
    if gpu_detected:
        try:
            mem_info = tf.config.experimental.get_memory_info("GPU:0")
            peak_mem_mb = mem_info["peak"] / (1024 ** 2)
            print(f"[SMOKE TEST] Peak GPU memory used: {peak_mem_mb:.2f} MB", flush=True)
        except Exception:
            pass

    # Production model hash AFTER smoke test
    hash_after = get_file_sha256(model_path)
    print(f"[MODEL HASH] Production model SHA256 after smoke test:  {hash_after}", flush=True)
    model_unmodified = (hash_before == hash_after)

    print("-" * 80, flush=True)
    print(f"[SMOKE TEST] ID Loss:    {id_loss:.6f}", flush=True)
    print(f"[SMOKE TEST] OE Loss:    {oe_loss:.6f}", flush=True)
    print(f"[SMOKE TEST] Total Loss: {total_loss:.6f} (lambda_oe={lambda_oe})", flush=True)
    print(f"[SMOKE TEST] ID Top-1:   {id_top1:.4f}", flush=True)
    print(f"[SMOKE TEST] ID Top-5:   {id_top5:.4f}", flush=True)
    print(f"[SMOKE TEST] Output shape after step: {pred_shape} (Preserved: {shape_preserved})", flush=True)
    print(f"[SMOKE TEST] Losses finite: {'YES' if losses_finite else 'NO'}", flush=True)
    print(f"[SMOKE TEST] Production model modified: {'NO' if model_unmodified else 'YES (CRITICAL ERROR)'}", flush=True)
    print("=" * 80 + "\n", flush=True)

    passed = losses_finite and shape_preserved and model_unmodified
    return {
        "ood_train_path": str(ood_source),
        "ood_calibration_path": str(DEFAULT_OOD_CALIBRATION_DIR),
        "id_train_count": counts["id_train_count"],
        "id_val_count": counts["id_val_count"],
        "ood_train_count": counts["ood_train_count"],
        "ood_cal_count": counts["ood_cal_count"],
        "gpu_detected": gpu_detected,
        "gpu_name": gpu_name,
        "gpu_device": smoke_gpu_device,
        "mixed_precision": mp_policy,
        "id_batch_size": id_batch_size,
        "ood_batch_size": ood_batch_size,
        "model_input_shape": model_input_shape,
        "model_output_shape": model_output_shape,
        "id_loss": id_loss,
        "oe_loss": oe_loss,
        "total_loss": total_loss,
        "id_top1": id_top1,
        "id_top5": id_top5,
        "peak_gpu_mem_mb": peak_mem_mb,
        "hash_before": hash_before,
        "hash_after": hash_after,
        "final_test_accessed": False,
        "passed": passed,
    }


# ---------------------------------------------------------------------------
# 10. Evaluation Utilities (ID Validation & OOD Calibration Diagnostics)
# ---------------------------------------------------------------------------
@tf.function
def val_step(model: tf.keras.Model, x: tf.Tensor, y: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    """Single compiled validation step."""
    preds = model(x, training=False)
    loss = compute_id_loss(y, preds)
    top1, top5 = compute_id_metrics(y, preds)
    return preds, loss, top1, top5


def evaluate_id_validation(
    model: tf.keras.Model,
    val_ds: tf.data.Dataset,
) -> Dict[str, float]:
    """Evaluate ID validation loss, top-1, top-5 accuracy, and macro F1."""
    total_loss = 0.0
    all_y_true: List[int] = []
    all_y_pred: List[int] = []
    total_samples = 0
    top1_correct = 0.0
    top5_correct = 0.0

    for x_batch, y_batch in val_ds:
        b_size = int(tf.shape(x_batch)[0].numpy())
        preds, loss, top1, top5 = val_step(model, x_batch, y_batch)
        
        total_loss += float(loss.numpy()) * b_size
        total_samples += b_size
        
        y_true_idx = tf.argmax(y_batch, axis=-1).numpy()
        y_pred_idx = tf.argmax(preds, axis=-1).numpy()
        
        all_y_true.extend(y_true_idx)
        all_y_pred.extend(y_pred_idx)
        
        top1_correct += float(top1.numpy()) * b_size
        top5_correct += float(top5.numpy()) * b_size

    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
    avg_top1 = top1_correct / total_samples if total_samples > 0 else 0.0
    avg_top5 = top5_correct / total_samples if total_samples > 0 else 0.0

    # Macro F1 across 120 classes
    y_true_arr = np.array(all_y_true, dtype=np.int64)
    y_pred_arr = np.array(all_y_pred, dtype=np.int64)
    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    np.add.at(cm, (y_true_arr, y_pred_arr), 1)
    tp = np.diag(cm)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    denom = 2 * tp + fp + fn
    f1_per_class = np.where(denom > 0, 2 * tp / denom, 0.0)
    macro_f1 = float(np.mean(f1_per_class))

    return {
        "val_id_loss": avg_loss,
        "val_id_accuracy": avg_top1,
        "val_id_top5": avg_top5,
        "val_id_macro_f1": macro_f1,
    }


@tf.function
def ood_step(
    feat_model: tf.keras.Model,
    w: tf.Tensor,
    b: tf.Tensor,
    x: tf.Tensor,
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    """Single compiled OOD evaluation step computing OE loss, MSP, entropy, and energy."""
    feats = feat_model(x, training=False)
    logits = tf.matmul(feats, w) + b
    preds = tf.nn.softmax(logits)
    oe_loss = compute_oe_loss(preds)
    msp = tf.reduce_max(preds, axis=-1)
    p_clamped = tf.clip_by_value(tf.cast(preds, tf.float32), 1e-7, 1.0)
    entropy = -tf.reduce_sum(p_clamped * tf.math.log(p_clamped), axis=-1)
    energy = -tf.reduce_logsumexp(tf.cast(logits, tf.float32), axis=-1)
    return oe_loss, msp, entropy, energy


def evaluate_ood_calibration(
    model: tf.keras.Model,
    ood_cal_ds: tf.data.Dataset,
) -> Dict[str, float]:
    """Evaluate OOD calibration diagnostics: OE loss, MSP, entropy, and energy."""
    total_oe_loss = 0.0
    total_msp = 0.0
    total_entropy = 0.0
    total_energy = 0.0
    total_samples = 0

    dense = model.get_layer("predictions")
    w = dense.weights[0]
    b = dense.weights[1]
    feat_model = tf.keras.Model(inputs=model.inputs, outputs=dense.input)

    for x_batch in ood_cal_ds:
        b_size = int(tf.shape(x_batch)[0].numpy())
        oe_loss, msp, entropy, energy = ood_step(feat_model, w, b, x_batch)

        total_oe_loss += float(oe_loss.numpy()) * b_size
        total_msp += float(tf.reduce_sum(msp).numpy())
        total_entropy += float(tf.reduce_sum(entropy).numpy())
        total_energy += float(tf.reduce_sum(energy).numpy())
        total_samples += b_size

    return {
        "ood_cal_oe_loss": total_oe_loss / total_samples if total_samples > 0 else 0.0,
        "ood_cal_max_probability_mean": total_msp / total_samples if total_samples > 0 else 0.0,
        "ood_cal_entropy_mean": total_entropy / total_samples if total_samples > 0 else 0.0,
        "ood_cal_energy_mean": total_energy / total_samples if total_samples > 0 else 0.0,
    }


# ---------------------------------------------------------------------------
# 11. Stage A Training Pipeline
# ---------------------------------------------------------------------------
def run_stage_a(
    model_path: Path = DEFAULT_MODEL_PATH,
    id_train_csv: Path = TRAIN_CSV_PATH,
    id_val_csv: Path = VAL_CSV_PATH,
    ood_train_dir: Path = DEFAULT_OOD_TRAIN_DIR,
    ood_cal_dir: Path = DEFAULT_OOD_CALIBRATION_DIR,
    id_batch_size: int = 4,
    ood_batch_size: int = 4,
    lambda_oe: float = 0.5,
    initial_lr: float = 1e-4,
    max_epochs: int = 5,
    patience: int = 2,
    allow_cpu: bool = False,
    mixed_prec: bool = True,
) -> Dict[str, Any]:
    """Execute Stage A Outlier Exposure training."""
    print("\n" + "=" * 80, flush=True)
    print("OUTLIER EXPOSURE STAGE A TRAINING (HEAD TUNING WITH FROZEN BACKBONE)", flush=True)
    print("=" * 80, flush=True)

    # 1. Protection runtime guard checks
    assert_not_protected_path(id_train_csv)
    assert_not_protected_path(id_val_csv)
    assert_not_protected_path(ood_train_dir)
    assert_not_protected_path(ood_cal_dir)

    # Verify production model SHA256 BEFORE doing anything
    prod_hash_before = get_file_sha256(model_path)
    print(f"[SAFETY] Production model hash BEFORE: {prod_hash_before}", flush=True)
    if prod_hash_before != EXPECTED_PROD_MODEL_HASH:
        raise ValueError(
            f"Production model hash mismatch: {prod_hash_before} vs expected {EXPECTED_PROD_MODEL_HASH}"
        )

    # Create destination directories
    OE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OE_METRICS_DIR.mkdir(parents=True, exist_ok=True)
    OE_PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    OE_REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # 2. Configure GPU & Mixed Precision
    gpus, gpu_name = configure_gpu(allow_cpu=allow_cpu)
    mp_policy = configure_mixed_precision(enabled=mixed_prec)

    # 3. Load Production Model & Clone to Training Instance
    prod_model = load_and_verify_model(model_path)
    breed_to_idx, _ = load_class_mapping()

    print("[MODEL] Cloning production model architecture and weights for Stage A training...", flush=True)
    model = tf.keras.models.clone_model(prod_model)
    model.set_weights(prod_model.get_weights())

    # 4. Stage A Freezing: Freeze MobileNetV2 Backbone, Train Classifier Head Only
    base_backbone = model.get_layer("mobilenetv2_1.00_224")
    base_backbone.trainable = False

    for layer in model.layers:
        if layer.name in ["input_image", "preprocess_input", "mobilenetv2_1.00_224"]:
            layer.trainable = False
        elif layer.name == "predictions":
            layer.trainable = True
        else:
            layer.trainable = True

    total_top_layers = len(model.layers)
    nested_base_layers = len(base_backbone.layers)
    total_layers_all = total_top_layers - 1 + nested_base_layers

    trainable_params = sum(int(np.prod(w.shape)) for w in model.trainable_weights)
    frozen_params = sum(int(np.prod(w.shape)) for w in model.non_trainable_weights)
    total_params = trainable_params + frozen_params

    trainable_layer_names = [l.name for l in model.layers if l.trainable and len(l.trainable_weights) > 0]
    frozen_layer_names = [l.name for l in model.layers if not l.trainable or len(l.trainable_weights) == 0]

    print("\n--- Model Architecture & Freezing Inspection ---", flush=True)
    print(f"Top-level Layers:     {total_top_layers}", flush=True)
    print(f"Base Backbone Layers: {nested_base_layers} (inside mobilenetv2_1.00_224)", flush=True)
    print(f"Total Combined Layers:{total_layers_all}", flush=True)
    print(f"Trainable Layers:     {trainable_layer_names}", flush=True)
    print(f"Frozen Layers:        {len(frozen_layer_names)} layers ({frozen_layer_names})", flush=True)
    print(f"Trainable Parameters: {trainable_params:,}", flush=True)
    print(f"Frozen Parameters:    {frozen_params:,}", flush=True)
    print(f"Total Parameters:     {total_params:,}", flush=True)
    print(f"Output Classes:       {model.output_shape[-1]} (Strictly preserved, no class 121)", flush=True)
    print("------------------------------------------------\n", flush=True)

    # 5. Build Datasets
    print("[DATA] Preparing datasets...", flush=True)
    id_train_paths: List[str] = []
    id_train_labels: List[int] = []
    with open(id_train_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            img_p = str(DATA_TRAIN_DIR / r["img_name"])
            id_train_paths.append(img_p)
            id_train_labels.append(breed_to_idx[r["breed"]])

    valid_exts = {".jpg", ".jpeg", ".png", ".webp"}
    ood_train_paths = [
        str(p) for p in ood_train_dir.rglob("*")
        if p.suffix.lower() in valid_exts and p.is_file()
    ]

    id_val_paths: List[str] = []
    id_val_labels: List[int] = []
    with open(id_val_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            img_p = str(DATA_TRAIN_DIR / r["img_name"])
            id_val_paths.append(img_p)
            id_val_labels.append(breed_to_idx[r["breed"]])

    ood_cal_paths = [
        str(p) for p in ood_cal_dir.rglob("*")
        if p.suffix.lower() in valid_exts and p.is_file()
    ]

    print(f"[DATA] ID Train Samples:        {len(id_train_paths)}", flush=True)
    print(f"[DATA] OOD Train Samples:       {len(ood_train_paths)}", flush=True)
    print(f"[DATA] ID Validation Samples:   {len(id_val_paths)}", flush=True)
    print(f"[DATA] OOD Calibration Samples: {len(ood_cal_paths)}", flush=True)

    def _parse_id(path: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        img_raw = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img_raw, channels=IMAGE_CHANNELS)
        img = tf.image.resize(img, (IMAGE_HEIGHT, IMAGE_WIDTH))
        img = tf.cast(img, tf.float32)
        y = tf.one_hot(label, NUM_CLASSES, dtype=tf.float32)
        return img, y

    def _parse_ood(path: tf.Tensor) -> tf.Tensor:
        img_raw = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img_raw, channels=IMAGE_CHANNELS)
        img = tf.image.resize(img, (IMAGE_HEIGHT, IMAGE_WIDTH))
        img = tf.cast(img, tf.float32)
        return img

    id_train_ds = (
        tf.data.Dataset.from_tensor_slices((id_train_paths, id_train_labels))
        .shuffle(buffer_size=len(id_train_paths), seed=42, reshuffle_each_iteration=True)
        .map(_parse_id, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(id_batch_size, drop_remainder=True)
        .prefetch(tf.data.AUTOTUNE)
    )

    ood_train_ds = (
        tf.data.Dataset.from_tensor_slices(ood_train_paths)
        .shuffle(buffer_size=len(ood_train_paths), seed=42, reshuffle_each_iteration=True)
        .repeat()
        .map(_parse_ood, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(ood_batch_size, drop_remainder=True)
        .prefetch(tf.data.AUTOTUNE)
    )

    train_ds = tf.data.Dataset.zip((id_train_ds, ood_train_ds))
    steps_per_epoch = len(id_train_paths) // id_batch_size

    # Batch validation/calibration at batch_size 4 to match memory footprint
    val_ds = (
        tf.data.Dataset.from_tensor_slices((id_val_paths, id_val_labels))
        .map(_parse_id, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(id_batch_size, drop_remainder=False)
        .prefetch(tf.data.AUTOTUNE)
    )

    ood_cal_ds = (
        tf.data.Dataset.from_tensor_slices(ood_cal_paths)
        .map(_parse_ood, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(ood_batch_size, drop_remainder=False)
        .prefetch(tf.data.AUTOTUNE)
    )

    # 6. Baseline Reference Evaluation (Pre-training on Unmodified Production Model)
    print("\n" + "=" * 80, flush=True)
    print("EVALUATING BASELINE PRODUCTION MODEL PERFORMANCE (PRE-TRAINING)", flush=True)
    print("=" * 80, flush=True)
    base_id_metrics = evaluate_id_validation(prod_model, val_ds)
    base_ood_metrics = evaluate_ood_calibration(prod_model, ood_cal_ds)

    baseline_val_loss = base_id_metrics["val_id_loss"]
    baseline_val_top1 = base_id_metrics["val_id_accuracy"]
    baseline_val_top5 = base_id_metrics["val_id_top5"]
    baseline_val_macro_f1 = base_id_metrics["val_id_macro_f1"]

    baseline_ood_oe_loss = base_ood_metrics["ood_cal_oe_loss"]
    baseline_ood_msp = base_ood_metrics["ood_cal_max_probability_mean"]
    baseline_ood_entropy = base_ood_metrics["ood_cal_entropy_mean"]
    baseline_ood_energy = base_ood_metrics["ood_cal_energy_mean"]

    print(f"BASELINE ID Val Loss:      {baseline_val_loss:.6f}", flush=True)
    print(f"BASELINE ID Val Top-1:     {baseline_val_top1 * 100:.4f}%", flush=True)
    print(f"BASELINE ID Val Top-5:     {baseline_val_top5 * 100:.4f}%", flush=True)
    print(f"BASELINE ID Val Macro F1:  {baseline_val_macro_f1:.4f}", flush=True)
    print(f"BASELINE OOD Cal OE Loss:  {baseline_ood_oe_loss:.6f}", flush=True)
    print(f"BASELINE OOD Cal Mean MSP: {baseline_ood_msp:.6f}", flush=True)
    print(f"BASELINE OOD Cal Entropy:  {baseline_ood_entropy:.6f} nats", flush=True)
    print(f"BASELINE OOD Cal Energy:   {baseline_ood_energy:.6f}", flush=True)
    print("=" * 80 + "\n", flush=True)

    # 7. Optimizer, Scheduler, and Training State
    current_lr = initial_lr
    optimizer = tf.keras.optimizers.Adam(learning_rate=current_lr)

    best_val_loss = float("inf")
    best_epoch = -1
    best_weights = [w.copy() for w in model.get_weights()]
    epochs_without_improvement = 0
    reduce_lr_counter = 0

    history_records: List[Dict[str, Any]] = []
    training_start_time = time.time()

    # 8. Training Loop
    for epoch in range(1, max_epochs + 1):
        epoch_start_time = time.time()
        print(f"\n>>> Epoch {epoch}/{max_epochs} [LR: {current_lr:.2e}] (Steps: {steps_per_epoch})", flush=True)

        total_train_id_loss = 0.0
        total_train_oe_loss = 0.0
        total_train_total_loss = 0.0
        train_top1_correct = 0.0
        train_top5_correct = 0.0
        step_count = 0

        for step, ((x_id_b, y_id_b), x_ood_b) in enumerate(train_ds.take(steps_per_epoch), 1):
            try:
                step_res = train_step(
                    model=model,
                    optimizer=optimizer,
                    x_id=x_id_b,
                    y_id=y_id_b,
                    x_ood=x_ood_b,
                    lambda_oe=lambda_oe,
                )
            except tf.errors.ResourceExhaustedError as oom_err:
                print("\n" + "!" * 80, flush=True)
                print(f"[OOM DETECTED] GPU VRAM exhausted at ID={id_batch_size}, OOD={ood_batch_size}.", flush=True)
                print("[RECOMMENDATION] Reduce batch size to: --id-batch-size 2 --ood-batch-size 2", flush=True)
                print("!" * 80 + "\n", flush=True)
                raise oom_err

            total_train_id_loss += float(step_res["id_loss"].numpy())
            total_train_oe_loss += float(step_res["oe_loss"].numpy())
            total_train_total_loss += float(step_res["total_loss"].numpy())
            train_top1_correct += float(step_res["id_top1"].numpy())
            train_top5_correct += float(step_res["id_top5"].numpy())
            step_count += 1

            if step % 250 == 0 or step == steps_per_epoch:
                cur_total = total_train_total_loss / step
                cur_id = total_train_id_loss / step
                cur_oe = total_train_oe_loss / step
                cur_acc = train_top1_correct / step
                print(
                    f"  Step {step:4d}/{steps_per_epoch} | "
                    f"Total: {cur_total:.4f} (ID: {cur_id:.4f}, OE: {cur_oe:.4f}) | "
                    f"ID Top-1: {cur_acc * 100:.2f}%",
                    flush=True,
                )

        epoch_duration = time.time() - epoch_start_time

        # Epoch training metrics
        train_id_loss = total_train_id_loss / step_count
        train_oe_loss = total_train_oe_loss / step_count
        train_total_loss = total_train_total_loss / step_count
        train_id_acc = train_top1_correct / step_count
        train_id_top5 = train_top5_correct / step_count

        # Validation evaluation
        print("  Evaluating ID Validation & OOD Calibration...", flush=True)
        val_res = evaluate_id_validation(model, val_ds)
        ood_res = evaluate_ood_calibration(model, ood_cal_ds)

        val_id_loss = val_res["val_id_loss"]
        val_id_acc = val_res["val_id_accuracy"]
        val_id_top5 = val_res["val_id_top5"]
        val_id_macro_f1 = val_res["val_id_macro_f1"]

        ood_cal_oe_loss = ood_res["ood_cal_oe_loss"]
        ood_cal_msp = ood_res["ood_cal_max_probability_mean"]
        ood_cal_entropy = ood_res["ood_cal_entropy_mean"]
        ood_cal_energy = ood_res["ood_cal_energy_mean"]

        # VRAM check
        vram_mb = None
        try:
            mem_info = tf.config.experimental.get_memory_info("GPU:0")
            vram_mb = mem_info["peak"] / (1024 ** 2)
        except Exception:
            pass

        record = {
            "epoch": epoch,
            "duration_sec": epoch_duration,
            "learning_rate": current_lr,
            "train_id_loss": train_id_loss,
            "train_oe_loss": train_oe_loss,
            "train_total_loss": train_total_loss,
            "train_id_accuracy": train_id_acc,
            "train_id_top5": train_id_top5,
            "val_id_loss": val_id_loss,
            "val_id_accuracy": val_id_acc,
            "val_id_top5": val_id_top5,
            "val_id_macro_f1": val_id_macro_f1,
            "ood_cal_oe_loss": ood_cal_oe_loss,
            "ood_cal_max_probability_mean": ood_cal_msp,
            "ood_cal_entropy_mean": ood_cal_entropy,
            "ood_cal_energy_mean": ood_cal_energy,
            "peak_vram_mb": vram_mb,
        }
        history_records.append(record)

        print(
            f"  [Epoch {epoch} Summary] Time: {epoch_duration:.1f}s | "
            f"Train Loss: {train_total_loss:.4f} | "
            f"Val ID Loss: {val_id_loss:.4f} | "
            f"Val Top-1: {val_id_acc * 100:.2f}% (Base: {baseline_val_top1 * 100:.2f}%) | "
            f"OOD MSP: {ood_cal_msp:.4f} (Base: {baseline_ood_msp:.4f}) | "
            f"OOD Entropy: {ood_cal_entropy:.4f} (Base: {baseline_ood_entropy:.4f})",
            flush=True,
        )

        # Check safety degradation flag (> 2 percentage points drop in ID Val Top-1)
        if val_id_acc < (baseline_val_top1 - 0.02):
            print(
                f"  [SAFETY WARNING] ID validation Top-1 ({val_id_acc * 100:.2f}%) dropped by > 2% "
                f"below baseline ({baseline_val_top1 * 100:.2f}%)!",
                flush=True,
            )

        # Model Checkpoint Selection (Preserve breed accuracy + minimize val_id_loss)
        is_safe = (val_id_acc >= baseline_val_top1 - 0.02)
        if is_safe and val_id_loss < best_val_loss:
            print(f"  [CHECKPOINT] New best model found! Val Loss improved: {best_val_loss:.6f} -> {val_id_loss:.6f}", flush=True)
            best_val_loss = val_id_loss
            best_epoch = epoch
            best_weights = [w.copy() for w in model.get_weights()]
            epochs_without_improvement = 0
            reduce_lr_counter = 0
            model.save(str(OE_BEST_MODEL_PATH))
            print(f"  [CHECKPOINT] Saved best model to: {OE_BEST_MODEL_PATH}", flush=True)
        else:
            epochs_without_improvement += 1
            reduce_lr_counter += 1
            print(f"  [INFO] No improvement. Epochs without improvement: {epochs_without_improvement}/{patience}", flush=True)

        # ReduceLROnPlateau logic (patience 1)
        if reduce_lr_counter >= 1 and current_lr > 1e-6:
            current_lr = max(current_lr * 0.5, 1e-6)
            optimizer.learning_rate.assign(current_lr)
            reduce_lr_counter = 0
            print(f"  [LR SCHEDULER] Reduced learning rate to: {current_lr:.2e}", flush=True)

        # EarlyStopping check (patience 2)
        if epochs_without_improvement >= patience:
            print(f"\n[EARLY STOPPING] Triggered at epoch {epoch} (patience={patience}).", flush=True)
            break

    total_training_time = time.time() - training_start_time
    print(f"\n[STAGE A COMPLETE] Total training time: {total_training_time:.1f}s across {len(history_records)} epochs.", flush=True)

    # 9. Save Last Model
    model.save(str(OE_LAST_MODEL_PATH))
    print(f"[CHECKPOINT] Saved last model to: {OE_LAST_MODEL_PATH}", flush=True)

    # 10. Restore Best Weights and Evaluate Final Best Model
    print(f"[CHECKPOINT] Restoring best weights from epoch {best_epoch}...", flush=True)
    model.set_weights(best_weights)
    model.save(str(OE_BEST_MODEL_PATH))
    best_model_hash = get_file_sha256(OE_BEST_MODEL_PATH)
    print(f"[CHECKPOINT] Best model SHA256: {best_model_hash}", flush=True)

    print("[EVALUATION] Final validation of best Stage A model...", flush=True)
    best_id_eval = evaluate_id_validation(model, val_ds)
    best_ood_eval = evaluate_ood_calibration(model, ood_cal_ds)

    best_val_top1 = best_id_eval["val_id_accuracy"]
    best_val_top5 = best_id_eval["val_id_top5"]
    best_val_macro_f1 = best_id_eval["val_id_macro_f1"]
    best_val_loss = best_id_eval["val_id_loss"]

    best_ood_msp = best_ood_eval["ood_cal_max_probability_mean"]
    best_ood_entropy = best_ood_eval["ood_cal_entropy_mean"]
    best_ood_energy = best_ood_eval["ood_cal_energy_mean"]
    best_ood_oe_loss = best_ood_eval["ood_cal_oe_loss"]

    # 11. Save Metrics History JSON & CSV
    history_payload = {
        "baseline": {
            "val_id_loss": baseline_val_loss,
            "val_id_accuracy": baseline_val_top1,
            "val_id_top5": baseline_val_top5,
            "val_id_macro_f1": baseline_val_macro_f1,
            "ood_cal_oe_loss": baseline_ood_oe_loss,
            "ood_cal_max_probability_mean": baseline_ood_msp,
            "ood_cal_entropy_mean": baseline_ood_entropy,
            "ood_cal_energy_mean": baseline_ood_energy,
        },
        "best_epoch": best_epoch,
        "best_model_hash": best_model_hash,
        "epochs": history_records,
    }
    history_json_path = OE_METRICS_DIR / "stage_a_history.json"
    with open(history_json_path, "w", encoding="utf-8") as f:
        json.dump(history_payload, f, indent=2)
    print(f"[METRICS] Saved history JSON to: {history_json_path}", flush=True)

    history_csv_path = OE_METRICS_DIR / "stage_a_history.csv"
    if history_records:
        with open(history_csv_path, "w", newline="", encoding="utf-8") as f:
            fieldnames = list(history_records[0].keys())
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(history_records)
    print(f"[METRICS] Saved history CSV to: {history_csv_path}", flush=True)

    # 12. Safety Post-Training Verification of Production Artifact
    prod_hash_after = get_file_sha256(model_path)
    print(f"[SAFETY] Production model hash AFTER: {prod_hash_after}", flush=True)
    prod_unmodified = (prod_hash_before == prod_hash_after == EXPECTED_PROD_MODEL_HASH)
    if not prod_unmodified:
        raise RuntimeError(
            f"CRITICAL: Production model was modified! Before: {prod_hash_before}, After: {prod_hash_after}"
        )

    return {
        "gpu_name": gpu_name,
        "tf_version": tf.__version__,
        "mixed_precision": mp_policy,
        "id_batch_size": id_batch_size,
        "ood_batch_size": ood_batch_size,
        "lambda_oe": lambda_oe,
        "optimizer": "Adam",
        "initial_lr": initial_lr,
        "epochs_completed": len(history_records),
        "best_epoch": best_epoch,
        "training_time_sec": total_training_time,
        "total_layers": total_layers_all,
        "trainable_layers": len(trainable_layer_names),
        "frozen_layers": len(frozen_layer_names),
        "trainable_params": trainable_params,
        "frozen_params": frozen_params,
        # Baseline
        "baseline_val_top1": baseline_val_top1,
        "baseline_val_top5": baseline_val_top5,
        "baseline_val_macro_f1": baseline_val_macro_f1,
        "baseline_ood_msp": baseline_ood_msp,
        "baseline_ood_entropy": baseline_ood_entropy,
        "baseline_ood_energy": baseline_ood_energy,
        # Stage A Best
        "stage_a_val_top1": best_val_top1,
        "stage_a_val_top5": best_val_top5,
        "stage_a_val_macro_f1": best_val_macro_f1,
        "stage_a_ood_msp": best_ood_msp,
        "stage_a_ood_entropy": best_ood_entropy,
        "stage_a_ood_energy": best_ood_energy,
        # Deltas
        "delta_val_top1": best_val_top1 - baseline_val_top1,
        "delta_val_top5": best_val_top5 - baseline_val_top5,
        "delta_val_macro_f1": best_val_macro_f1 - baseline_val_macro_f1,
        "delta_ood_msp": best_ood_msp - baseline_ood_msp,
        "delta_ood_entropy": best_ood_entropy - baseline_ood_entropy,
        "delta_ood_energy": best_ood_energy - baseline_ood_energy,
        # Checkpoints & Hashes
        "best_checkpoint": str(OE_BEST_MODEL_PATH),
        "best_checkpoint_sha256": best_model_hash,
        "prod_hash_before": prod_hash_before,
        "prod_hash_after": prod_hash_after,
        "prod_unmodified": prod_unmodified,
        "final_id_test_accessed": False,
        "ood_test_accessed": False,
        "hard_ood_accessed": False,
        "passed": True,
    }


# ---------------------------------------------------------------------------
# 12. Stage B Training Pipeline (Upper Backbone Fine-Tuning)
# ---------------------------------------------------------------------------
def run_stage_b(
    stage_a_model_path: Path = OE_BEST_MODEL_PATH,
    prod_model_path: Path = DEFAULT_MODEL_PATH,
    id_train_csv: Path = TRAIN_CSV_PATH,
    id_val_csv: Path = VAL_CSV_PATH,
    ood_train_dir: Path = DEFAULT_OOD_TRAIN_DIR,
    ood_cal_dir: Path = DEFAULT_OOD_CALIBRATION_DIR,
    id_batch_size: int = 4,
    ood_batch_size: int = 4,
    lambda_oe: float = 0.5,
    initial_lr: float = 1e-5,
    max_epochs: int = 5,
    patience: int = 2,
    allow_cpu: bool = False,
    mixed_prec: bool = True,
) -> Dict[str, Any]:
    """Execute Stage B Outlier Exposure fine-tuning (upper backbone + classifier head)."""
    print("\n" + "=" * 80, flush=True)
    print("OUTLIER EXPOSURE STAGE B TRAINING (UPPER BACKBONE FINE-TUNING)", flush=True)
    print("=" * 80, flush=True)

    # 1. Protection runtime guard checks
    assert_not_protected_path(id_train_csv)
    assert_not_protected_path(id_val_csv)
    assert_not_protected_path(ood_train_dir)
    assert_not_protected_path(ood_cal_dir)

    # Verify production model SHA256 BEFORE doing anything
    prod_hash_before = get_file_sha256(prod_model_path)
    print(f"[SAFETY] Production model hash BEFORE: {prod_hash_before}", flush=True)
    if prod_hash_before != EXPECTED_PROD_MODEL_HASH:
        raise ValueError(
            f"Production model hash mismatch: {prod_hash_before} vs expected {EXPECTED_PROD_MODEL_HASH}"
        )

    # Verify Stage A best model SHA256 BEFORE doing anything
    stage_a_hash_before = get_file_sha256(stage_a_model_path)
    print(f"[SAFETY] Stage A model hash BEFORE: {stage_a_hash_before}", flush=True)
    if stage_a_hash_before != EXPECTED_STAGE_A_MODEL_HASH:
        raise ValueError(
            f"Stage A best model hash mismatch: {stage_a_hash_before} vs expected {EXPECTED_STAGE_A_MODEL_HASH}"
        )

    # Ensure output directories exist
    OE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OE_METRICS_DIR.mkdir(parents=True, exist_ok=True)
    OE_PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    OE_REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # 2. Configure GPU & Mixed Precision
    gpus, gpu_name = configure_gpu(allow_cpu=allow_cpu)
    mp_policy = configure_mixed_precision(enabled=mixed_prec)

    # 3. Load Stage A Best Model & Clone for Stage B Training
    print(f"[MODEL] Loading Stage A best model from: {stage_a_model_path}", flush=True)
    stage_a_model = load_and_verify_model(stage_a_model_path)
    breed_to_idx, _ = load_class_mapping()

    print("[MODEL] Cloning Stage A model architecture and weights for Stage B fine-tuning...", flush=True)
    model = tf.keras.models.clone_model(stage_a_model)
    model.set_weights(stage_a_model.get_weights())

    # 4. Stage B Unfreezing Strategy
    # Unfreeze upper/final MobileNetV2 blocks (from block_13_expand at index 116 onwards)
    # KEEP ALL BatchNormalization layers frozen!
    base_backbone = model.get_layer("mobilenetv2_1.00_224")
    base_backbone.trainable = True

    start_unfreeze_idx = 116  # block_13_expand
    for i, layer in enumerate(base_backbone.layers):
        is_bn = isinstance(layer, tf.keras.layers.BatchNormalization)
        if i < start_unfreeze_idx or is_bn:
            layer.trainable = False
        else:
            layer.trainable = True

    # Configure top-level layers
    for layer in model.layers:
        if layer.name in ["input_image", "preprocess_input"]:
            layer.trainable = False
        elif layer.name == "predictions":
            layer.trainable = True
        elif layer.name == "mobilenetv2_1.00_224":
            layer.trainable = True
        else:
            layer.trainable = True

    total_backbone_layers = len(base_backbone.layers)
    unfrozen_backbone_layers = sum(1 for l in base_backbone.layers if l.trainable)
    frozen_backbone_layers = sum(1 for l in base_backbone.layers if not l.trainable)
    trainable_bn_layers = sum(
        1 for l in base_backbone.layers
        if isinstance(l, tf.keras.layers.BatchNormalization) and l.trainable
    )

    trainable_params = sum(int(np.prod(w.shape)) for w in model.trainable_weights)
    frozen_params = sum(int(np.prod(w.shape)) for w in model.non_trainable_weights)
    total_params = trainable_params + frozen_params

    print("\n--- Stage B Architecture & Unfreezing Inspection ---", flush=True)
    print(f"Total Backbone Layers:        {total_backbone_layers}", flush=True)
    print(f"Unfrozen Backbone Layers:     {unfrozen_backbone_layers} ({unfrozen_backbone_layers / total_backbone_layers * 100:.1f}%)", flush=True)
    print(f"Frozen Backbone Layers:       {frozen_backbone_layers} ({frozen_backbone_layers / total_backbone_layers * 100:.1f}%)", flush=True)
    print(f"Trainable BatchNorm Layers:   {trainable_bn_layers} (Must be strictly 0)", flush=True)
    print(f"Trainable Parameters:         {trainable_params:,}", flush=True)
    print(f"Frozen Parameters:            {frozen_params:,}", flush=True)
    print(f"Total Parameters:             {total_params:,}", flush=True)
    print(f"Output Classes:               {model.output_shape[-1]} (Strictly preserved, no class 121)", flush=True)
    print("----------------------------------------------------\n", flush=True)

    if trainable_bn_layers > 0:
        raise ValueError(f"FATAL: Found {trainable_bn_layers} trainable BatchNorm layers! All BN layers must be frozen.")

    # 5. Build Datasets
    print("[DATA] Preparing datasets...", flush=True)
    id_train_paths: List[str] = []
    id_train_labels: List[int] = []
    with open(id_train_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            img_p = str(DATA_TRAIN_DIR / r["img_name"])
            id_train_paths.append(img_p)
            id_train_labels.append(breed_to_idx[r["breed"]])

    valid_exts = {".jpg", ".jpeg", ".png", ".webp"}
    ood_train_paths = [
        str(p) for p in ood_train_dir.rglob("*")
        if p.suffix.lower() in valid_exts and p.is_file()
    ]

    id_val_paths: List[str] = []
    id_val_labels: List[int] = []
    with open(id_val_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            img_p = str(DATA_TRAIN_DIR / r["img_name"])
            id_val_paths.append(img_p)
            id_val_labels.append(breed_to_idx[r["breed"]])

    ood_cal_paths = [
        str(p) for p in ood_cal_dir.rglob("*")
        if p.suffix.lower() in valid_exts and p.is_file()
    ]

    print(f"[DATA] ID Train Samples:        {len(id_train_paths)}", flush=True)
    print(f"[DATA] OOD Train Samples:       {len(ood_train_paths)}", flush=True)
    print(f"[DATA] ID Validation Samples:   {len(id_val_paths)}", flush=True)
    print(f"[DATA] OOD Calibration Samples: {len(ood_cal_paths)}", flush=True)

    def _parse_id(path: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        img_raw = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img_raw, channels=IMAGE_CHANNELS)
        img = tf.image.resize(img, (IMAGE_HEIGHT, IMAGE_WIDTH))
        img = tf.cast(img, tf.float32)
        y = tf.one_hot(label, NUM_CLASSES, dtype=tf.float32)
        return img, y

    def _parse_ood(path: tf.Tensor) -> tf.Tensor:
        img_raw = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img_raw, channels=IMAGE_CHANNELS)
        img = tf.image.resize(img, (IMAGE_HEIGHT, IMAGE_WIDTH))
        img = tf.cast(img, tf.float32)
        return img

    id_train_ds = (
        tf.data.Dataset.from_tensor_slices((id_train_paths, id_train_labels))
        .shuffle(buffer_size=len(id_train_paths), seed=42, reshuffle_each_iteration=True)
        .map(_parse_id, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(id_batch_size, drop_remainder=True)
        .prefetch(tf.data.AUTOTUNE)
    )

    ood_train_ds = (
        tf.data.Dataset.from_tensor_slices(ood_train_paths)
        .shuffle(buffer_size=len(ood_train_paths), seed=42, reshuffle_each_iteration=True)
        .repeat()
        .map(_parse_ood, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(ood_batch_size, drop_remainder=True)
        .prefetch(tf.data.AUTOTUNE)
    )

    train_ds = tf.data.Dataset.zip((id_train_ds, ood_train_ds))
    steps_per_epoch = len(id_train_paths) // id_batch_size

    val_ds = (
        tf.data.Dataset.from_tensor_slices((id_val_paths, id_val_labels))
        .map(_parse_id, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(id_batch_size, drop_remainder=False)
        .prefetch(tf.data.AUTOTUNE)
    )

    ood_cal_ds = (
        tf.data.Dataset.from_tensor_slices(ood_cal_paths)
        .map(_parse_ood, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(ood_batch_size, drop_remainder=False)
        .prefetch(tf.data.AUTOTUNE)
    )

    # 6. Reference Evaluation (Stage A Best Reference)
    print("\n" + "=" * 80, flush=True)
    print("EVALUATING STAGE A BEST REFERENCE MODEL PERFORMANCE (PRE-STAGE-B)", flush=True)
    print("=" * 80, flush=True)
    stage_a_id_metrics = evaluate_id_validation(stage_a_model, val_ds)
    stage_a_ood_metrics = evaluate_ood_calibration(stage_a_model, ood_cal_ds)

    stage_a_val_loss = stage_a_id_metrics["val_id_loss"]
    stage_a_val_top1 = stage_a_id_metrics["val_id_accuracy"]
    stage_a_val_top5 = stage_a_id_metrics["val_id_top5"]
    stage_a_val_macro_f1 = stage_a_id_metrics["val_id_macro_f1"]

    stage_a_ood_oe_loss = stage_a_ood_metrics["ood_cal_oe_loss"]
    stage_a_ood_msp = stage_a_ood_metrics["ood_cal_max_probability_mean"]
    stage_a_ood_entropy = stage_a_ood_metrics["ood_cal_entropy_mean"]
    stage_a_ood_energy = stage_a_ood_metrics["ood_cal_energy_mean"]

    print(f"STAGE A ID Val Loss:      {stage_a_val_loss:.6f}", flush=True)
    print(f"STAGE A ID Val Top-1:     {stage_a_val_top1 * 100:.4f}%", flush=True)
    print(f"STAGE A ID Val Top-5:     {stage_a_val_top5 * 100:.4f}%", flush=True)
    print(f"STAGE A ID Val Macro F1:  {stage_a_val_macro_f1:.4f}", flush=True)
    print(f"STAGE A OOD Cal OE Loss:  {stage_a_ood_oe_loss:.6f}", flush=True)
    print(f"STAGE A OOD Cal Mean MSP: {stage_a_ood_msp:.6f}", flush=True)
    print(f"STAGE A OOD Cal Entropy:  {stage_a_ood_entropy:.6f} nats", flush=True)
    print(f"STAGE A OOD Cal Energy:   {stage_a_ood_energy:.6f}", flush=True)
    print("=" * 80 + "\n", flush=True)

    # 7. Optimizer, Scheduler, and Training State (Fresh Adam at 1e-5)
    current_lr = initial_lr
    optimizer = tf.keras.optimizers.Adam(learning_rate=current_lr)

    best_val_loss = stage_a_val_loss  # Must improve or match while passing safety guards!
    best_epoch = -1
    best_weights = [w.copy() for w in model.get_weights()]
    epochs_without_improvement = 0
    reduce_lr_counter = 0

    history_records: List[Dict[str, Any]] = []
    training_start_time = time.time()

    # 8. Training Loop
    for epoch in range(1, max_epochs + 1):
        epoch_start_time = time.time()
        print(f"\n>>> Epoch {epoch}/{max_epochs} [LR: {current_lr:.2e}] (Steps: {steps_per_epoch})", flush=True)

        total_train_id_loss = 0.0
        total_train_oe_loss = 0.0
        total_train_total_loss = 0.0
        train_top1_correct = 0.0
        train_top5_correct = 0.0
        step_count = 0

        for step, ((x_id_b, y_id_b), x_ood_b) in enumerate(train_ds.take(steps_per_epoch), 1):
            try:
                step_res = train_step(
                    model=model,
                    optimizer=optimizer,
                    x_id=x_id_b,
                    y_id=y_id_b,
                    x_ood=x_ood_b,
                    lambda_oe=lambda_oe,
                )
            except tf.errors.ResourceExhaustedError as oom_err:
                print("\n" + "!" * 80, flush=True)
                print(f"[OOM DETECTED] GPU VRAM exhausted at ID={id_batch_size}, OOD={ood_batch_size}.", flush=True)
                print("!" * 80 + "\n", flush=True)
                raise oom_err

            total_train_id_loss += float(step_res["id_loss"].numpy())
            total_train_oe_loss += float(step_res["oe_loss"].numpy())
            total_train_total_loss += float(step_res["total_loss"].numpy())
            train_top1_correct += float(step_res["id_top1"].numpy())
            train_top5_correct += float(step_res["id_top5"].numpy())
            step_count += 1

            if step % 250 == 0 or step == steps_per_epoch:
                cur_total = total_train_total_loss / step
                cur_id = total_train_id_loss / step
                cur_oe = total_train_oe_loss / step
                cur_acc = train_top1_correct / step
                print(
                    f"  Step {step:4d}/{steps_per_epoch} | "
                    f"Total: {cur_total:.4f} (ID: {cur_id:.4f}, OE: {cur_oe:.4f}) | "
                    f"ID Top-1: {cur_acc * 100:.2f}%",
                    flush=True,
                )

        epoch_duration = time.time() - epoch_start_time

        # Epoch training metrics
        train_id_loss = total_train_id_loss / step_count
        train_oe_loss = total_train_oe_loss / step_count
        train_total_loss = total_train_total_loss / step_count
        train_id_acc = train_top1_correct / step_count
        train_id_top5 = train_top5_correct / step_count

        # Validation evaluation
        print("  Evaluating ID Validation & OOD Calibration...", flush=True)
        val_res = evaluate_id_validation(model, val_ds)
        ood_res = evaluate_ood_calibration(model, ood_cal_ds)

        val_id_loss = val_res["val_id_loss"]
        val_id_acc = val_res["val_id_accuracy"]
        val_id_top5 = val_res["val_id_top5"]
        val_id_macro_f1 = val_res["val_id_macro_f1"]

        ood_cal_oe_loss = ood_res["ood_cal_oe_loss"]
        ood_cal_msp = ood_res["ood_cal_max_probability_mean"]
        ood_cal_entropy = ood_res["ood_cal_entropy_mean"]
        ood_cal_energy = ood_res["ood_cal_energy_mean"]

        # VRAM check
        vram_mb = None
        try:
            mem_info = tf.config.experimental.get_memory_info("GPU:0")
            vram_mb = mem_info["peak"] / (1024 ** 2)
        except Exception:
            pass

        record = {
            "epoch": epoch,
            "duration_sec": epoch_duration,
            "learning_rate": current_lr,
            "train_id_loss": train_id_loss,
            "train_oe_loss": train_oe_loss,
            "train_total_loss": train_total_loss,
            "train_id_accuracy": train_id_acc,
            "train_id_top5": train_id_top5,
            "val_id_loss": val_id_loss,
            "val_id_accuracy": val_id_acc,
            "val_id_top5": val_id_top5,
            "val_id_macro_f1": val_id_macro_f1,
            "ood_cal_oe_loss": ood_cal_oe_loss,
            "ood_cal_max_probability_mean": ood_cal_msp,
            "ood_cal_entropy_mean": ood_cal_entropy,
            "ood_cal_energy_mean": ood_cal_energy,
            "peak_vram_mb": vram_mb,
        }
        history_records.append(record)

        print(
            f"  [Epoch {epoch} Summary] Time: {epoch_duration:.1f}s | "
            f"Train Loss: {train_total_loss:.4f} | "
            f"Val ID Loss: {val_id_loss:.4f} (Stage A: {stage_a_val_loss:.4f}) | "
            f"Val Top-1: {val_id_acc * 100:.2f}% (Stage A: {stage_a_val_top1 * 100:.2f}%) | "
            f"OOD MSP: {ood_cal_msp:.4f} (Stage A: {stage_a_ood_msp:.4f}) | "
            f"OOD Entropy: {ood_cal_entropy:.4f} (Stage A: {stage_a_ood_entropy:.4f})",
            flush=True,
        )

        # Model Selection Guard:
        # Flag if Val ID Top-1 drops > 1.0 percentage point below Stage A best
        # or Val Macro F1 materially deteriorates (> 0.015 drop)
        id_acc_drop = stage_a_val_top1 - val_id_acc
        f1_drop = stage_a_val_macro_f1 - val_id_macro_f1
        if id_acc_drop > 0.01:
            print(
                f"  [GUARD TRIGGERED] Val ID Top-1 ({val_id_acc * 100:.2f}%) dropped by {id_acc_drop * 100:.2f}% "
                f"(> 1.0 pp) below Stage A ({stage_a_val_top1 * 100:.2f}%)!",
                flush=True,
            )
        if f1_drop > 0.015:
            print(
                f"  [GUARD TRIGGERED] Val Macro F1 ({val_id_macro_f1:.4f}) dropped by {f1_drop:.4f} "
                f"below Stage A ({stage_a_val_macro_f1:.4f})!",
                flush=True,
            )

        is_safe = (id_acc_drop <= 0.01) and (f1_drop <= 0.015)
        if is_safe and val_id_loss < best_val_loss:
            print(f"  [CHECKPOINT] New best Stage B model! Val Loss improved: {best_val_loss:.6f} -> {val_id_loss:.6f}", flush=True)
            best_val_loss = val_id_loss
            best_epoch = epoch
            best_weights = [w.copy() for w in model.get_weights()]
            epochs_without_improvement = 0
            reduce_lr_counter = 0
            model.save(str(OE_STAGE_B_BEST_MODEL_PATH))
            print(f"  [CHECKPOINT] Saved best Stage B model to: {OE_STAGE_B_BEST_MODEL_PATH}", flush=True)
        else:
            epochs_without_improvement += 1
            reduce_lr_counter += 1
            print(f"  [INFO] No improvement. Epochs without improvement: {epochs_without_improvement}/{patience}", flush=True)

        # ReduceLROnPlateau logic (factor 0.5, patience 1, min_lr 1e-6)
        if reduce_lr_counter >= 1 and current_lr > 1e-6:
            current_lr = max(current_lr * 0.5, 1e-6)
            optimizer.learning_rate.assign(current_lr)
            reduce_lr_counter = 0
            print(f"  [LR SCHEDULER] Reduced learning rate to: {current_lr:.2e}", flush=True)

        # EarlyStopping check (patience 2)
        if epochs_without_improvement >= patience:
            print(f"\n[EARLY STOPPING] Triggered at epoch {epoch} (patience={patience}).", flush=True)
            break

    total_training_time = time.time() - training_start_time
    print(f"\n[STAGE B TRAINING FINISHED] Total training time: {total_training_time:.1f}s across {len(history_records)} epochs.", flush=True)

    # 9. Save Last Model
    model.save(str(OE_STAGE_B_LAST_MODEL_PATH))
    print(f"[CHECKPOINT] Saved last Stage B model to: {OE_STAGE_B_LAST_MODEL_PATH}", flush=True)

    # 10. Checkpoint Finalization & Evaluation
    stage_b_improved = (best_epoch != -1)
    if stage_b_improved:
        print(f"[CHECKPOINT] Restoring best Stage B weights from epoch {best_epoch}...", flush=True)
        model.set_weights(best_weights)
        model.save(str(OE_STAGE_B_BEST_MODEL_PATH))
        best_model_hash = get_file_sha256(OE_STAGE_B_BEST_MODEL_PATH)
        print(f"[CHECKPOINT] Best Stage B model SHA256: {best_model_hash}", flush=True)

        print("[EVALUATION] Final validation of best Stage B model...", flush=True)
        best_id_eval = evaluate_id_validation(model, val_ds)
        best_ood_eval = evaluate_ood_calibration(model, ood_cal_ds)

        best_val_top1 = best_id_eval["val_id_accuracy"]
        best_val_top5 = best_id_eval["val_id_top5"]
        best_val_macro_f1 = best_id_eval["val_id_macro_f1"]
        best_val_loss = best_id_eval["val_id_loss"]

        best_ood_msp = best_ood_eval["ood_cal_max_probability_mean"]
        best_ood_entropy = best_ood_eval["ood_cal_entropy_mean"]
        best_ood_energy = best_ood_eval["ood_cal_energy_mean"]
        best_ood_oe_loss = best_ood_eval["ood_cal_oe_loss"]
    else:
        print("[INFO] No Stage B epoch surpassed Stage A with safety constraints. Stage A remains best.", flush=True)
        best_val_top1 = stage_a_val_top1
        best_val_top5 = stage_a_val_top5
        best_val_macro_f1 = stage_a_val_macro_f1
        best_val_loss = stage_a_val_loss
        best_ood_msp = stage_a_ood_msp
        best_ood_entropy = stage_a_ood_entropy
        best_ood_energy = stage_a_ood_energy
        best_ood_oe_loss = stage_a_ood_oe_loss
        best_model_hash = stage_a_hash_before

    # 11. Save Metrics History JSON & CSV
    history_payload = {
        "stage_a_reference": {
            "val_id_loss": stage_a_val_loss,
            "val_id_accuracy": stage_a_val_top1,
            "val_id_top5": stage_a_val_top5,
            "val_id_macro_f1": stage_a_val_macro_f1,
            "ood_cal_oe_loss": stage_a_ood_oe_loss,
            "ood_cal_max_probability_mean": stage_a_ood_msp,
            "ood_cal_entropy_mean": stage_a_ood_entropy,
            "ood_cal_energy_mean": stage_a_ood_energy,
        },
        "best_epoch": best_epoch,
        "best_model_hash": best_model_hash,
        "stage_b_improved": stage_b_improved,
        "epochs": history_records,
    }
    history_json_path = OE_METRICS_DIR / "stage_b_history.json"
    with open(history_json_path, "w", encoding="utf-8") as f:
        json.dump(history_payload, f, indent=2)
    print(f"[METRICS] Saved history JSON to: {history_json_path}", flush=True)

    history_csv_path = OE_METRICS_DIR / "stage_b_history.csv"
    if history_records:
        with open(history_csv_path, "w", newline="", encoding="utf-8") as f:
            fieldnames = list(history_records[0].keys())
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(history_records)
    print(f"[METRICS] Saved history CSV to: {history_csv_path}", flush=True)

    # 12. Safety Post-Training Verification
    prod_hash_after = get_file_sha256(prod_model_path)
    print(f"[SAFETY] Production model hash AFTER: {prod_hash_after}", flush=True)
    if prod_hash_after != EXPECTED_PROD_MODEL_HASH:
        raise RuntimeError("CRITICAL: Production model hash changed during Stage B!")

    stage_a_hash_after = get_file_sha256(stage_a_model_path)
    print(f"[SAFETY] Stage A model hash AFTER: {stage_a_hash_after}", flush=True)
    if stage_a_hash_after != EXPECTED_STAGE_A_MODEL_HASH:
        raise RuntimeError("CRITICAL: Stage A model hash changed during Stage B!")

    return {
        "gpu_name": gpu_name,
        "tf_version": tf.__version__,
        "mixed_precision": mp_policy,
        "id_batch_size": id_batch_size,
        "ood_batch_size": ood_batch_size,
        "lambda_oe": lambda_oe,
        "optimizer": "Adam",
        "initial_lr": initial_lr,
        "epochs_completed": len(history_records),
        "best_epoch": best_epoch,
        "training_time_sec": total_training_time,
        "total_backbone_layers": total_backbone_layers,
        "unfrozen_backbone_layers": unfrozen_backbone_layers,
        "frozen_backbone_layers": frozen_backbone_layers,
        "trainable_bn_layers": trainable_bn_layers,
        "trainable_params": trainable_params,
        "frozen_params": frozen_params,
        # Stage A Ref
        "stage_a_val_top1": stage_a_val_top1,
        "stage_a_val_top5": stage_a_val_top5,
        "stage_a_val_macro_f1": stage_a_val_macro_f1,
        "stage_a_val_loss": stage_a_val_loss,
        "stage_a_ood_msp": stage_a_ood_msp,
        "stage_a_ood_entropy": stage_a_ood_entropy,
        "stage_a_ood_energy": stage_a_ood_energy,
        "stage_a_ood_oe_loss": stage_a_ood_oe_loss,
        # Stage B Best
        "stage_b_val_top1": best_val_top1,
        "stage_b_val_top5": best_val_top5,
        "stage_b_val_macro_f1": best_val_macro_f1,
        "stage_b_val_loss": best_val_loss,
        "stage_b_ood_msp": best_ood_msp,
        "stage_b_ood_entropy": best_ood_entropy,
        "stage_b_ood_energy": best_ood_energy,
        "stage_b_ood_oe_loss": best_ood_oe_loss,
        # Deltas
        "delta_val_top1": best_val_top1 - stage_a_val_top1,
        "delta_val_top5": best_val_top5 - stage_a_val_top5,
        "delta_val_macro_f1": best_val_macro_f1 - stage_a_val_macro_f1,
        "delta_val_loss": best_val_loss - stage_a_val_loss,
        "delta_ood_msp": best_ood_msp - stage_a_ood_msp,
        "delta_ood_entropy": best_ood_entropy - stage_a_ood_entropy,
        "delta_ood_energy": best_ood_energy - stage_a_ood_energy,
        "delta_ood_oe_loss": best_ood_oe_loss - stage_a_ood_oe_loss,
        # Checkpoints & Hashes
        "stage_a_checkpoint": str(stage_a_model_path),
        "stage_a_checkpoint_sha256": stage_a_hash_before,
        "best_checkpoint": str(OE_STAGE_B_BEST_MODEL_PATH) if stage_b_improved else str(stage_a_model_path),
        "best_checkpoint_sha256": best_model_hash,
        "stage_b_improved": stage_b_improved,
        "prod_hash_before": prod_hash_before,
        "prod_hash_after": prod_hash_after,
        "final_id_test_accessed": False,
        "ood_test_accessed": False,
        "hard_ood_accessed": False,
        "passed": True,
    }


# ---------------------------------------------------------------------------
# 13. CLI Argument Parser
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Outlier Exposure (OE) Training and Smoke Testing for Dog Breed Classifier"
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Path to starting production model (.keras)",
    )
    parser.add_argument(
        "--id-batch-size",
        type=int,
        default=4,
        help="Batch size for ID dog images (default: 4)",
    )
    parser.add_argument(
        "--ood-batch-size",
        type=int,
        default=4,
        help="Batch size for OOD non-dog images (default: 4)",
    )
    parser.add_argument(
        "--lambda-oe",
        type=float,
        default=0.5,
        help="Weight for Outlier Exposure loss (default: 0.5)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Initial learning rate for Stage A (default: 1e-4)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Maximum epochs for Stage A (default: 5)",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=2,
        help="EarlyStopping patience in epochs (default: 2)",
    )
    parser.add_argument(
        "--mixed-precision",
        action="store_true",
        default=True,
        help="Enable mixed_float16 precision policy (default: True)",
    )
    parser.add_argument(
        "--no-mixed-precision",
        dest="mixed_precision",
        action="store_false",
        help="Disable mixed precision policy (force float32)",
    )
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        default=False,
        help="Allow running on CPU if no GPU is detected",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        default=False,
        help="Run 1-step smoke test only without full training",
    )
    parser.add_argument(
        "--stage-a",
        action="store_true",
        default=False,
        help="Run Stage A Outlier Exposure training (classifier head only)",
    )
    parser.add_argument(
        "--stage-b",
        action="store_true",
        default=False,
        help="Run Stage B Outlier Exposure fine-tuning (upper backbone + classifier head)",
    )
    parser.add_argument(
        "--stage-a-model-path",
        type=Path,
        default=OE_BEST_MODEL_PATH,
        help="Path to Stage A best model checkpoint (.keras)",
    )
    parser.add_argument(
        "--ood-train-path",
        type=Path,
        default=DEFAULT_OOD_TRAIN_DIR,
        help="Path to OOD training dataset split",
    )
    parser.add_argument(
        "--ood-calibration-path",
        type=Path,
        default=DEFAULT_OOD_CALIBRATION_DIR,
        help="Path to OOD calibration dataset split",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# 14. Main Entry Point
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()

    # Guard against accidental protected test set paths from CLI flags
    assert_not_protected_path(args.ood_train_path)
    assert_not_protected_path(args.ood_calibration_path)
    assert_not_protected_path(args.stage_a_model_path)

    if args.smoke_test:
        results = run_smoke_test(
            model_path=args.model_path,
            id_batch_size=args.id_batch_size,
            ood_batch_size=args.ood_batch_size,
            lambda_oe=args.lambda_oe,
            allow_cpu=args.allow_cpu,
            mixed_prec=args.mixed_precision,
            custom_ood_dir=args.ood_train_path,
        )

        passed = results["passed"]
        mem_str = f"{results['peak_gpu_mem_mb']:.2f} MB" if results["peak_gpu_mem_mb"] is not None else "N/A"

        print("=" * 80, flush=True)
        print("OUTLIER EXPOSURE SMOKE TEST REPORT", flush=True)
        print("=" * 80, flush=True)
        print(f"OOD TRAIN PATH:               {results['ood_train_path']}", flush=True)
        print(f"OOD CALIBRATION PATH:         {results['ood_calibration_path']}", flush=True)
        print(f"ID TRAIN COUNT:               {results['id_train_count']}", flush=True)
        print(f"ID VAL COUNT:                 {results['id_val_count']}", flush=True)
        print(f"OOD TRAIN COUNT:              {results['ood_train_count']}", flush=True)
        print(f"OOD CALIBRATION COUNT:        {results['ood_cal_count']}", flush=True)
        print(f"GPU:                          {results['gpu_name']}", flush=True)
        print(f"MIXED PRECISION:              {results['mixed_precision']}", flush=True)
        print(f"ID BATCH SIZE:                {results['id_batch_size']}", flush=True)
        print(f"OOD BATCH SIZE:               {results['ood_batch_size']}", flush=True)
        print(f"MODEL INPUT:                  {results['model_input_shape']}", flush=True)
        print(f"MODEL OUTPUT:                 {results['model_output_shape']}", flush=True)
        print(f"ID LOSS:                      {results['id_loss']:.6f}", flush=True)
        print(f"OE LOSS:                      {results['oe_loss']:.6f}", flush=True)
        print(f"TOTAL LOSS:                   {results['total_loss']:.6f}", flush=True)
        print(f"GPU PEAK MEMORY:              {mem_str}", flush=True)
        print(f"PRODUCTION MODEL HASH BEFORE: {results['hash_before']}", flush=True)
        print(f"PRODUCTION MODEL HASH AFTER:  {results['hash_after']}", flush=True)
        print(f"FINAL TEST ACCESSED:          NO", flush=True)
        print(f"SMOKE TEST RESULT:            {'PASSED' if passed else 'FAILED'}", flush=True)
        print("=" * 80, flush=True)
        final_status = "READY_FOR_FULL_OE_TRAINING" if passed else "BLOCKED"
        print(f"FINAL STATUS:                 {final_status}", flush=True)
        print("=" * 80, flush=True)

        if not passed:
            sys.exit(1)
        return

    if args.stage_a:
        res = run_stage_a(
            model_path=args.model_path,
            id_train_csv=TRAIN_CSV_PATH,
            id_val_csv=VAL_CSV_PATH,
            ood_train_dir=args.ood_train_path,
            ood_cal_dir=args.ood_calibration_path,
            id_batch_size=args.id_batch_size,
            ood_batch_size=args.ood_batch_size,
            lambda_oe=args.lambda_oe,
            initial_lr=args.lr,
            max_epochs=args.epochs,
            patience=args.patience,
            allow_cpu=args.allow_cpu,
            mixed_prec=args.mixed_precision,
        )

        print("\n" + "=" * 80, flush=True)
        print("STAGE A TRAINING SUMMARY RESULTS", flush=True)
        print("=" * 80, flush=True)
        print(f"GPU:                          {res['gpu_name']}", flush=True)
        print(f"TensorFlow version:           {res['tf_version']}", flush=True)
        print(f"Mixed precision:              {res['mixed_precision']}", flush=True)
        print(f"ID batch size:                {res['id_batch_size']}", flush=True)
        print(f"OOD batch size:               {res['ood_batch_size']}", flush=True)
        print(f"Lambda OE:                    {res['lambda_oe']}", flush=True)
        print(f"Optimizer:                    {res['optimizer']}", flush=True)
        print(f"Learning rate:                {res['initial_lr']}", flush=True)
        print(f"Epochs completed:             {res['epochs_completed']}", flush=True)
        print(f"Best epoch:                   {res['best_epoch']}", flush=True)
        print(f"Training time:                {res['training_time_sec']:.1f}s", flush=True)
        print(f"Total layers:                 {res['total_layers']}", flush=True)
        print(f"Trainable layers:             {res['trainable_layers']}", flush=True)
        print(f"Frozen layers:                {res['frozen_layers']}", flush=True)
        print(f"Trainable parameters:         {res['trainable_params']:,}", flush=True)
        print(f"Frozen parameters:            {res['frozen_params']:,}", flush=True)
        print("-" * 80, flush=True)
        print(f"BASELINE ID VAL TOP1:         {res['baseline_val_top1'] * 100:.4f}%", flush=True)
        print(f"STAGE A ID VAL TOP1:          {res['stage_a_val_top1'] * 100:.4f}%", flush=True)
        print(f"DELTA:                        {res['delta_val_top1'] * 100:+.4f}%", flush=True)
        print("-" * 80, flush=True)
        print(f"BASELINE ID VAL TOP5:         {res['baseline_val_top5'] * 100:.4f}%", flush=True)
        print(f"STAGE A ID VAL TOP5:          {res['stage_a_val_top5'] * 100:.4f}%", flush=True)
        print(f"DELTA:                        {res['delta_val_top5'] * 100:+.4f}%", flush=True)
        print("-" * 80, flush=True)
        print(f"BASELINE ID MACRO F1:         {res['baseline_val_macro_f1']:.4f}", flush=True)
        print(f"STAGE A ID MACRO F1:          {res['stage_a_val_macro_f1']:.4f}", flush=True)
        print(f"DELTA:                        {res['delta_val_macro_f1']:+.4f}", flush=True)
        print("-" * 80, flush=True)
        print(f"BASELINE OOD MSP:             {res['baseline_ood_msp']:.4f}", flush=True)
        print(f"STAGE A OOD MSP:              {res['stage_a_ood_msp']:.4f}", flush=True)
        print(f"DELTA:                        {res['delta_ood_msp']:+.4f}", flush=True)
        print("-" * 80, flush=True)
        print(f"BASELINE OOD ENTROPY:         {res['baseline_ood_entropy']:.4f}", flush=True)
        print(f"STAGE A OOD ENTROPY:          {res['stage_a_ood_entropy']:.4f}", flush=True)
        print(f"DELTA:                        {res['delta_ood_entropy']:+.4f}", flush=True)
        print("-" * 80, flush=True)
        print(f"BASELINE OOD ENERGY:          {res['baseline_ood_energy']:.4f}", flush=True)
        print(f"STAGE A OOD ENERGY:           {res['stage_a_ood_energy']:.4f}", flush=True)
        print(f"DELTA:                        {res['delta_ood_energy']:+.4f}", flush=True)
        print("-" * 80, flush=True)
        print(f"BEST CHECKPOINT:              {res['best_checkpoint']}", flush=True)
        print(f"BEST CHECKPOINT SHA256:       {res['best_checkpoint_sha256']}", flush=True)
        print(f"PRODUCTION MODEL HASH BEFORE: {res['prod_hash_before']}", flush=True)
        print(f"PRODUCTION MODEL HASH AFTER:  {res['prod_hash_after']}", flush=True)
        print(f"FINAL ID TEST ACCESSED:       NO", flush=True)
        print(f"OOD TEST ACCESSED:            NO", flush=True)
        print(f"HARD OOD ACCESSED:            NO", flush=True)
        print("=" * 80, flush=True)
        status = "OE_STAGE_A_COMPLETE" if res["passed"] else "BLOCKED"
        print(f"FINAL STATUS:                 {status}", flush=True)
        print("=" * 80, flush=True)
        return

    if args.stage_b:
        stage_b_lr = args.lr if args.lr != 1e-4 else 1e-5
        res = run_stage_b(
            stage_a_model_path=args.stage_a_model_path,
            prod_model_path=args.model_path,
            id_train_csv=TRAIN_CSV_PATH,
            id_val_csv=VAL_CSV_PATH,
            ood_train_dir=args.ood_train_path,
            ood_cal_dir=args.ood_calibration_path,
            id_batch_size=args.id_batch_size,
            ood_batch_size=args.ood_batch_size,
            lambda_oe=args.lambda_oe,
            initial_lr=stage_b_lr,
            max_epochs=args.epochs,
            patience=args.patience,
            allow_cpu=args.allow_cpu,
            mixed_prec=args.mixed_precision,
        )

        print("\n" + "=" * 80, flush=True)
        print("STAGE B TRAINING SUMMARY RESULTS", flush=True)
        print("=" * 80, flush=True)
        print(f"GPU:                          {res['gpu_name']}", flush=True)
        print(f"TensorFlow version:           {res['tf_version']}", flush=True)
        print(f"Mixed precision:              {res['mixed_precision']}", flush=True)
        print(f"ID batch size:                {res['id_batch_size']}", flush=True)
        print(f"OOD batch size:               {res['ood_batch_size']}", flush=True)
        print(f"Lambda OE:                    {res['lambda_oe']}", flush=True)
        print(f"Optimizer:                    {res['optimizer']}", flush=True)
        print(f"Learning rate:                {res['initial_lr']}", flush=True)
        print(f"Epochs completed:             {res['epochs_completed']}", flush=True)
        print(f"Best epoch:                   {res['best_epoch']}", flush=True)
        print(f"Training time:                {res['training_time_sec']:.1f}s", flush=True)
        print(f"Total backbone layers:        {res['total_backbone_layers']}", flush=True)
        print(f"Unfrozen backbone layers:     {res['unfrozen_backbone_layers']}", flush=True)
        print(f"Frozen backbone layers:       {res['frozen_backbone_layers']}", flush=True)
        print(f"Trainable BatchNorm layers:   {res['trainable_bn_layers']}", flush=True)
        print(f"Trainable parameters:         {res['trainable_params']:,}", flush=True)
        print(f"Frozen parameters:            {res['frozen_params']:,}", flush=True)
        print("-" * 80, flush=True)
        print(f"STAGE A ID VAL TOP1:          {res['stage_a_val_top1'] * 100:.4f}%", flush=True)
        print(f"STAGE B ID VAL TOP1:          {res['stage_b_val_top1'] * 100:.4f}%", flush=True)
        print(f"DELTA:                        {res['delta_val_top1'] * 100:+.4f}%", flush=True)
        print("-" * 80, flush=True)
        print(f"STAGE A ID VAL TOP5:          {res['stage_a_val_top5'] * 100:.4f}%", flush=True)
        print(f"STAGE B ID VAL TOP5:          {res['stage_b_val_top5'] * 100:.4f}%", flush=True)
        print(f"DELTA:                        {res['delta_val_top5'] * 100:+.4f}%", flush=True)
        print("-" * 80, flush=True)
        print(f"STAGE A ID MACRO F1:          {res['stage_a_val_macro_f1']:.4f}", flush=True)
        print(f"STAGE B ID MACRO F1:          {res['stage_b_val_macro_f1']:.4f}", flush=True)
        print(f"DELTA:                        {res['delta_val_macro_f1']:+.4f}", flush=True)
        print("-" * 80, flush=True)
        print(f"STAGE A OOD MSP:              {res['stage_a_ood_msp']:.4f}", flush=True)
        print(f"STAGE B OOD MSP:              {res['stage_b_ood_msp']:.4f}", flush=True)
        print(f"DELTA:                        {res['delta_ood_msp']:+.4f}", flush=True)
        print("-" * 80, flush=True)
        print(f"STAGE A OOD ENTROPY:          {res['stage_a_ood_entropy']:.4f}", flush=True)
        print(f"STAGE B OOD ENTROPY:          {res['stage_b_ood_entropy']:.4f}", flush=True)
        print(f"DELTA:                        {res['delta_ood_entropy']:+.4f}", flush=True)
        print("-" * 80, flush=True)
        print(f"STAGE A OOD ENERGY:           {res['stage_a_ood_energy']:.4f}", flush=True)
        print(f"STAGE B OOD ENERGY:           {res['stage_b_ood_energy']:.4f}", flush=True)
        print(f"DELTA:                        {res['delta_ood_energy']:+.4f}", flush=True)
        print("-" * 80, flush=True)
        print(f"BEST CHECKPOINT:              {res['best_checkpoint']}", flush=True)
        print(f"BEST CHECKPOINT SHA256:       {res['best_checkpoint_sha256']}", flush=True)
        print(f"STAGE A MODEL HASH:           {res['stage_a_checkpoint_sha256']}", flush=True)
        print(f"PRODUCTION MODEL HASH BEFORE: {res['prod_hash_before']}", flush=True)
        print(f"PRODUCTION MODEL HASH AFTER:  {res['prod_hash_after']}", flush=True)
        print(f"FINAL ID TEST ACCESSED:       NO", flush=True)
        print(f"OOD TEST ACCESSED:            NO", flush=True)
        print(f"HARD OOD ACCESSED:            NO", flush=True)
        print("=" * 80, flush=True)
        status = "OE_STAGE_B_COMPLETE" if res["stage_b_improved"] else "STAGE_A_REMAINS_BEST"
        print(f"FINAL STATUS:                 {status}", flush=True)
        print("=" * 80, flush=True)
        return

    # Check if full training is requested without flag
    print("[BLOCKED] Specify --smoke-test, --stage-a, or --stage-b to run.", flush=True)
    sys.exit(2)


if __name__ == "__main__":
    main()
