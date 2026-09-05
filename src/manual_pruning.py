"""Manual Magnitude Pruning for MobileNetV2 Transfer Learning Model (Keras 3 Compatible).

This module implements Task 6.2:
- Bypasses TensorFlow Model Optimization Toolkit (TF-MOT) completely due to technical
  incompatibility with Keras 3.x and TensorFlow 2.21+.
- Loads the final selected MobileNetV2 transfer learning checkpoint in read-only mode.
- Performs one-shot manual magnitude pruning on 50% of weights with smallest |weight|
  in the Dense classifier head ('predictions.kernel').
- Preserves classifier bias, MobileNetV2 backbone, BatchNorm running statistics,
  and frozen state (trainable=False, 0 trainable variables).
- Strictly performs NO training, NO fine-tuning, NO internal test evaluation (test_ds),
  and NO Kaggle test evaluation.
- Evaluates validation accuracy on val_ds and computes accuracy drop.
- Serializes pruned model to .keras and verifies sparsity preservation upon reloading.
- Generates ZIP archives for compression comparison and writes summary artifacts.
"""

from pathlib import Path
import hashlib
import os
import zipfile
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
import keras

from src.data_loader import PROJECT_ROOT
from src.dataset_pipeline import build_datasets

# Register preprocess_input function with Keras custom objects
tf.keras.utils.get_custom_objects()["preprocess_input"] = (
    tf.keras.applications.mobilenet_v2.preprocess_input
)

# File and Directory Constants
CHECKPOINT_PATH: Path = (
    PROJECT_ROOT
    / "outputs"
    / "mobilenet_transfer"
    / "checkpoints"
    / "best_mobilenet_transfer.keras"
)
OUTPUT_DIR: Path = PROJECT_ROOT / "outputs" / "compression_manual"
PRUNED_MODEL_PATH: Path = OUTPUT_DIR / "pruned_50.keras"
ORIGINAL_ZIP_PATH: Path = OUTPUT_DIR / "original_model.zip"
PRUNED_ZIP_PATH: Path = OUTPUT_DIR / "pruned_50.zip"
SPARSITY_REPORT_PATH: Path = OUTPUT_DIR / "sparsity_report.csv"
SUMMARY_PATH: Path = OUTPUT_DIR / "compression_summary.txt"

# Pruning & Evaluation Configuration
TARGET_SPARSITY: float = 0.50
BATCH_SIZE: int = 64
BASELINE_REF_VAL_ACC: float = 0.8239  # 82.39% from Model Selection


def ensure_output_directory() -> Path:
    """Create outputs/compression_manual directory if it does not exist.

    Returns
    -------
    Path
        Path to output directory.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR


def magnitude_prune_kernel(
    kernel: np.ndarray,
    target_sparsity: float = TARGET_SPARSITY,
) -> np.ndarray:
    """Prune weights with smallest absolute magnitude to achieve target sparsity.

    Uses exact rank/index selection (via np.argsort) to ensure the exact target
    number of elements are set to zero without threshold rounding artifacts.

    Parameters
    ----------
    kernel : np.ndarray
        Weight tensor of Dense layer kernel (shape: (1280, 120)).
    target_sparsity : float, default=0.50
        Target fraction of weights to be exactly zero (0.0).

    Returns
    -------
    np.ndarray
        Pruned kernel array of identical shape with exactly round(size * target_sparsity) zeros.
    """
    flat = kernel.flatten().copy()
    total_weights = flat.size
    target_zeros = int(round(total_weights * target_sparsity))

    # Compute absolute magnitude
    abs_weights = np.abs(flat)

    # Sort indices ascending by absolute magnitude
    sorted_indices = np.argsort(abs_weights)

    # Zero out the smallest target_zeros weights
    prune_indices = sorted_indices[:target_zeros]
    flat[prune_indices] = 0.0

    return flat.reshape(kernel.shape)


def create_zip_archive(source_file: Path, zip_dest: Path) -> int:
    """Create a standalone ZIP archive containing a single file with ZIP_DEFLATED.

    Parameters
    ----------
    source_file : Path
        Path to file to compress.
    zip_dest : Path
        Destination ZIP archive path.

    Returns
    -------
    int
        Size of generated ZIP archive in bytes.
    """
    with zipfile.ZipFile(zip_dest, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(source_file, arcname=source_file.name)
    return zip_dest.stat().st_size


def run_manual_pruning() -> Dict[str, Any]:
    """Execute Task 6.2 manual magnitude pruning workflow.

    Returns
    -------
    Dict[str, Any]
        Dictionary of results and validation metrics.
    """
    print("=" * 80)
    print("TASK 6.2 — MANUAL MAGNITUDE PRUNING (KERAS 3 COMPATIBLE)")
    print("=" * 80)

    # 1. Environment verification
    tf_version = str(tf.__version__)
    keras_version = str(keras.__version__)
    print(f"\n[1/10] Active Environment:")
    print(f"  - TensorFlow: {tf_version}")
    print(f"  - Keras:      {keras_version}")
    print(f"  - TF-MOT:     NOT USED (pure TF/Keras implementation)")

    ensure_output_directory()

    # 2. Load baseline model
    print(f"\n[2/10] Loading original model checkpoint:")
    print(f"  - Path: {CHECKPOINT_PATH}")
    if not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(f"Original checkpoint not found at: {CHECKPOINT_PATH}")

    custom_objects = {
        "preprocess_input": tf.keras.applications.mobilenet_v2.preprocess_input
    }
    original_model = tf.keras.models.load_model(
        CHECKPOINT_PATH,
        custom_objects=custom_objects,
    )

    orig_total_params = int(original_model.count_params())
    orig_trainable_params = int(
        sum(np.prod(v.shape) for v in original_model.trainable_variables)
    )
    orig_file_size = CHECKPOINT_PATH.stat().st_size

    print(f"  - Model Name:       {original_model.name}")
    print(f"  - Input Shape:      {original_model.input_shape}")
    print(f"  - Output Shape:     {original_model.output_shape}")
    print(f"  - Total Params:     {orig_total_params:,}")
    print(f"  - Trainable Params: {orig_trainable_params:,}")
    print(f"  - File Size:        {orig_file_size:,} bytes ({orig_file_size / (1024**2):.2f} MB)")

    # 3. Locate MobileNetV2 backbone and Dense classifier head
    print(f"\n[3/10] Inspecting architecture components:")
    backbone: Optional[tf.keras.Model] = None
    for layer in original_model.layers:
        if isinstance(layer, tf.keras.Model) or "mobilenetv2" in layer.name.lower():
            backbone = layer
            break

    if backbone is None:
        raise RuntimeError("MobileNetV2 backbone layer not found in loaded model.")

    classifier = original_model.get_layer("predictions")
    if not isinstance(classifier, tf.keras.layers.Dense):
        raise TypeError(f"Expected Dense layer for predictions, got {type(classifier)}")

    kernel_tensor = classifier.kernel
    bias_tensor = classifier.bias
    kernel_shape = tuple(kernel_tensor.shape)
    bias_shape = tuple(bias_tensor.shape)
    kernel_total_weights = int(np.prod(kernel_shape))

    print(f"  - Backbone Name:        {backbone.name}")
    print(f"  - Backbone Trainable:   {backbone.trainable}")
    print(f"  - Backbone Trainable Vars: {len(backbone.trainable_variables)}")
    print(f"  - Classifier Layer:     {classifier.name} ({classifier.__class__.__name__})")
    print(f"  - Classifier Units:     {classifier.units}")
    print(f"  - Kernel Shape:         {kernel_shape} ({kernel_total_weights:,} weights)")
    print(f"  - Bias Shape:           {bias_shape}")

    # 4. Pre-pruning weight snapshot
    print(f"\n[4/10] Capturing pre-pruning snapshots for integrity audit:")
    backbone_weights_before = [w.numpy().copy() for w in backbone.weights]
    backbone_hashes_before = [
        hashlib.sha256(w.tobytes()).hexdigest() for w in backbone_weights_before
    ]
    kernel_before = classifier.kernel.numpy().copy()
    bias_before = classifier.bias.numpy().copy()

    kernel_zeros_before = int(np.sum(kernel_before == 0.0))
    bias_zeros_before = int(np.sum(bias_before == 0.0))
    sparsity_before = kernel_zeros_before / kernel_total_weights

    print(f"  - Backbone Tensors:     {len(backbone_weights_before)} captured")
    print(f"  - Kernel Zeros Before:  {kernel_zeros_before:,} ({sparsity_before:.2%})")
    print(f"  - Bias Zeros Before:    {bias_zeros_before}")

    # 5. Execute manual magnitude pruning on Dense kernel
    print(f"\n[5/10] Applying manual magnitude pruning (target sparsity = {TARGET_SPARSITY:.0%}):")
    pruned_kernel = magnitude_prune_kernel(kernel_before, target_sparsity=TARGET_SPARSITY)

    kernel_zeros_after = int(np.sum(pruned_kernel == 0.0))
    actual_sparsity = kernel_zeros_after / kernel_total_weights
    print(f"  - Target Zero Weights:  {int(kernel_total_weights * TARGET_SPARSITY):,}")
    print(f"  - Actual Zero Weights:  {kernel_zeros_after:,}")
    print(f"  - Actual Kernel Sparsity: {actual_sparsity:.4%}")

    # Update classifier kernel in model
    classifier.kernel.assign(pruned_kernel)

    # 6. Weight integrity audit
    print(f"\n[6/10] Verifying weight integrity post-pruning:")
    kernel_after = classifier.kernel.numpy()
    bias_after = classifier.bias.numpy()

    kernel_changed = not np.array_equal(kernel_after, kernel_before)
    kernel_diff_elements = int(np.sum(kernel_after != kernel_before))
    bias_changed = not np.array_equal(bias_after, bias_before)
    bias_max_diff = float(np.max(np.abs(bias_after - bias_before)))

    backbone_weights_after = [w.numpy() for w in backbone.weights]
    backbone_hashes_after = [
        hashlib.sha256(w.tobytes()).hexdigest() for w in backbone_weights_after
    ]
    backbone_changed = False
    for idx, (h_before, h_after) in enumerate(
        zip(backbone_hashes_before, backbone_hashes_after)
    ):
        if h_before != h_after:
            backbone_changed = True
            print(f"    WARNING: Backbone weight mismatch at index {idx}")
            break

    backbone_trainable = backbone.trainable
    backbone_trainable_vars = len(backbone.trainable_variables)

    print(f"  - Classifier Kernel Changed:  {'YES' if kernel_changed else 'NO'} ({kernel_diff_elements:,} elements modified)")
    print(f"  - Classifier Bias Changed:    {'YES' if bias_changed else 'NO'} (max diff: {bias_max_diff})")
    print(f"  - Backbone Weights Changed:   {'YES' if backbone_changed else 'NO'}")
    print(f"  - Backbone Trainable:         {backbone_trainable}")
    print(f"  - Backbone Trainable Vars:    {backbone_trainable_vars}")

    if not kernel_changed:
        raise RuntimeError("FAIL: Classifier kernel was not updated.")
    if bias_changed:
        raise RuntimeError("FAIL: Classifier bias was unexpectedly modified.")
    if backbone_changed:
        raise RuntimeError("FAIL: MobileNetV2 backbone weights were unexpectedly modified.")
    if backbone_trainable or backbone_trainable_vars != 0:
        raise RuntimeError("FAIL: MobileNetV2 backbone freezing violated.")

    # 7. Validation dataset pipeline & Forward Sanity Check
    print(f"\n[7/10] Loading validation dataset & running sanity check:")
    print("  - Calling build_datasets(batch_size=64)...")
    train_ds, val_ds, test_ds, encoder = build_datasets(batch_size=BATCH_SIZE)
    # Strictly isolate: delete train_ds and test_ds references to guarantee zero access
    del train_ds
    del test_ds

    # Sanity check forward pass on 1 validation batch
    images, labels = next(iter(val_ds))
    preds = original_model(images, training=False)
    preds_np = preds.numpy()

    output_shape = tuple(preds.shape)
    nan_count = int(np.isnan(preds_np).sum())
    inf_count = int(np.isinf(preds_np).sum())
    row_sums = np.sum(preds_np, axis=-1)
    softmax_valid = bool(np.allclose(row_sums, 1.0, atol=1e-5))

    print(f"  - Batch Images Shape: {images.shape}")
    print(f"  - Predictions Shape:  {output_shape}")
    print(f"  - NaN Count:          {nan_count}")
    print(f"  - Inf Count:          {inf_count}")
    print(f"  - Softmax Valid:      {softmax_valid}")

    if nan_count > 0 or inf_count > 0 or not softmax_valid:
        raise ValueError("Forward pass sanity check failed: invalid prediction values.")

    # 8. Validation evaluation (pruned model)
    print(f"\n[8/10] Evaluating pruned model on validation dataset:")
    pruned_eval_res = original_model.evaluate(val_ds, verbose=1)
    pruned_val_loss = float(pruned_eval_res[0])
    pruned_val_acc = float(pruned_eval_res[1])

    acc_drop = (BASELINE_REF_VAL_ACC - pruned_val_acc) * 100.0

    print(f"  - Original Val Accuracy (Ref): {BASELINE_REF_VAL_ACC * 100:.2f}%")
    print(f"  - Pruned Val Loss:             {pruned_val_loss:.4f}")
    print(f"  - Pruned Val Accuracy:         {pruned_val_acc * 100:.2f}%")
    print(f"  - Validation Accuracy Drop:    {acc_drop:+.2f} percentage points")

    # 9. Serialization and Reload Verification
    print(f"\n[9/10] Saving pruned model and verifying reload:")
    original_model.save(PRUNED_MODEL_PATH)
    pruned_file_size = PRUNED_MODEL_PATH.stat().st_size
    print(f"  - Saved to:        {PRUNED_MODEL_PATH}")
    print(f"  - Pruned File Size: {pruned_file_size:,} bytes ({pruned_file_size / (1024**2):.2f} MB)")

    # Reload model
    loaded_model = tf.keras.models.load_model(
        PRUNED_MODEL_PATH,
        custom_objects=custom_objects,
    )
    loaded_classifier = loaded_model.get_layer("predictions")
    loaded_kernel = loaded_classifier.kernel.numpy()
    loaded_total_weights = loaded_kernel.size
    loaded_zero_weights = int(np.sum(loaded_kernel == 0.0))
    loaded_sparsity = loaded_zero_weights / loaded_total_weights

    # Reload forward pass check
    reloaded_preds = loaded_model(images, training=False).numpy()
    reloaded_nan = int(np.isnan(reloaded_preds).sum())
    reloaded_inf = int(np.isinf(reloaded_preds).sum())
    reloaded_softmax = bool(np.allclose(np.sum(reloaded_preds, axis=-1), 1.0, atol=1e-5))

    print(f"  - Reload Successful:       YES")
    print(f"  - Reloaded Kernel Weights: {loaded_total_weights:,}")
    print(f"  - Reloaded Zero Weights:   {loaded_zero_weights:,}")
    print(f"  - Reloaded Sparsity:       {loaded_sparsity:.4%}")
    print(f"  - Sparsity Preserved:      {'YES' if loaded_zero_weights == kernel_zeros_after else 'NO'}")
    print(f"  - Reloaded Forward NaN:    {reloaded_nan}")
    print(f"  - Reloaded Forward Inf:    {reloaded_inf}")
    print(f"  - Reloaded Softmax Valid:  {reloaded_softmax}")

    if loaded_zero_weights != kernel_zeros_after:
        raise RuntimeError("FAIL: Sparsity was not preserved after model reload.")

    # 10. ZIP Compression and Reporting
    print(f"\n[10/10] Generating ZIP archives and reporting artifacts:")
    orig_zip_size = create_zip_archive(CHECKPOINT_PATH, ORIGINAL_ZIP_PATH)
    pruned_zip_size = create_zip_archive(PRUNED_MODEL_PATH, PRUNED_ZIP_PATH)
    zip_reduction = (orig_zip_size - pruned_zip_size) / orig_zip_size * 100.0

    print(f"  - Original ZIP Size: {orig_zip_size:,} bytes ({orig_zip_size / (1024**2):.2f} MB)")
    print(f"  - Pruned ZIP Size:   {pruned_zip_size:,} bytes ({pruned_zip_size / (1024**2):.2f} MB)")
    print(f"  - ZIP Reduction:     {zip_reduction:.2f}%")

    # Generate Sparsity Report CSV
    sparsity_rows: List[Dict[str, Any]] = []
    sparsity_rows.append({
        "layer": "predictions",
        "tensor": "kernel",
        "total_weights": kernel_total_weights,
        "zero_weights": kernel_zeros_after,
        "sparsity": f"{actual_sparsity:.4f}",
    })
    sparsity_rows.append({
        "layer": "predictions",
        "tensor": "bias",
        "total_weights": int(np.prod(bias_shape)),
        "zero_weights": int(np.sum(bias_after == 0.0)),
        "sparsity": f"{np.sum(bias_after == 0.0) / np.prod(bias_shape):.4f}",
    })

    # Record backbone summary and per-weight metrics
    backbone_total_weights = 0
    backbone_zero_weights = 0
    for w in backbone.weights:
        w_arr = w.numpy()
        w_size = w_arr.size
        w_zeros = int(np.sum(w_arr == 0.0))
        backbone_total_weights += w_size
        backbone_zero_weights += w_zeros
        sparsity_rows.append({
            "layer": backbone.name,
            "tensor": w.name,
            "total_weights": w_size,
            "zero_weights": w_zeros,
            "sparsity": f"{w_zeros / w_size:.4f}",
        })

    sparsity_rows.append({
        "layer": "mobilenetv2_backbone_total",
        "tensor": "all_tensors",
        "total_weights": backbone_total_weights,
        "zero_weights": backbone_zero_weights,
        "sparsity": f"{backbone_zero_weights / backbone_total_weights:.4f}",
    })

    sparsity_df = pd.DataFrame(sparsity_rows)
    sparsity_df.to_csv(SPARSITY_REPORT_PATH, index=False)
    print(f"  - Wrote Sparsity Report: {SPARSITY_REPORT_PATH}")

    # Generate Compression Summary TXT
    pruned_total_params = int(loaded_model.count_params())
    diff_keras_bytes = orig_file_size - pruned_file_size
    diff_keras_pct = (diff_keras_bytes / orig_file_size) * 100.0

    summary_content = f"""Method:
Manual magnitude pruning

Target sparsity:
50%

Pruned layer:
predictions Dense kernel

Original total params:
{orig_total_params:,}
Original file size:
{orig_file_size:,} bytes ({orig_file_size / (1024**2):.2f} MB)

Pruned total params:
{pruned_total_params:,}
Pruned file size:
{pruned_file_size:,} bytes ({pruned_file_size / (1024**2):.2f} MB)

Original ZIP size:
{orig_zip_size:,} bytes ({orig_zip_size / (1024**2):.2f} MB)
Pruned ZIP size:
{pruned_zip_size:,} bytes ({pruned_zip_size / (1024**2):.2f} MB)
ZIP reduction:
{zip_reduction:.2f}%

Classifier kernel total weights:
{kernel_total_weights:,}
Classifier kernel zero weights:
{kernel_zeros_after:,}
Classifier kernel sparsity:
{actual_sparsity * 100:.2f}%

Original validation accuracy:
{BASELINE_REF_VAL_ACC * 100:.2f}%
Pruned validation accuracy:
{pruned_val_acc * 100:.2f}%
Validation accuracy drop:
{acc_drop:+.2f}%

Backbone trainable:
{backbone_trainable}
Backbone weights changed:
{'YES' if backbone_changed else 'NO'}

Classifier bias changed:
{'YES' if bias_changed else 'NO'}

Training performed:
NO
Internal test evaluated:
NO
Kaggle test used:
NO
TF-MOT used:
NO
"""
    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        f.write(summary_content)
    print(f"  - Wrote Compression Summary: {SUMMARY_PATH}")

    return {
        "status": "PASS",
        "tf_version": tf_version,
        "keras_version": keras_version,
        "orig_total_params": orig_total_params,
        "orig_trainable_params": orig_trainable_params,
        "orig_file_size": orig_file_size,
        "pruned_file_size": pruned_file_size,
        "diff_keras_bytes": diff_keras_bytes,
        "diff_keras_pct": diff_keras_pct,
        "orig_zip_size": orig_zip_size,
        "pruned_zip_size": pruned_zip_size,
        "zip_reduction": zip_reduction,
        "kernel_shape": kernel_shape,
        "kernel_total_weights": kernel_total_weights,
        "kernel_zeros_before": kernel_zeros_before,
        "kernel_zeros_after": kernel_zeros_after,
        "actual_sparsity": actual_sparsity,
        "kernel_changed": kernel_changed,
        "bias_changed": bias_changed,
        "backbone_changed": backbone_changed,
        "backbone_trainable": backbone_trainable,
        "backbone_trainable_vars": backbone_trainable_vars,
        "output_shape": output_shape,
        "nan_count": nan_count,
        "inf_count": inf_count,
        "softmax_valid": softmax_valid,
        "orig_val_acc": BASELINE_REF_VAL_ACC,
        "pruned_val_loss": pruned_val_loss,
        "pruned_val_acc": pruned_val_acc,
        "acc_drop": acc_drop,
        "loaded_sparsity": loaded_sparsity,
        "loaded_zeros": loaded_zero_weights,
    }


if __name__ == "__main__":
    results = run_manual_pruning()
    print("\nTask 6.2 execution completed successfully.")
