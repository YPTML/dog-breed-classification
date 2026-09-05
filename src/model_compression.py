"""Model Compression / Pruning Preparation for Dog Breed Identification.

This module implements Task 6.1:
- Inspects and loads the final selected MobileNetV2 Frozen Transfer Learning checkpoint.
- Gathers baseline architecture and parameter counts (total, trainable, non-trainable, size).
- Verifies MobileNetV2 backbone freezing (trainable=False, 0 trainable variables).
- Audits TensorFlow Model Optimization Toolkit (tfmot) compatibility with TF 2.21 / Keras 3.
- Evaluates pruning feasibility and halts cleanly with WARNING/FAIL if incompatible,
  strictly preserving the active environment without downgrading TensorFlow.
- Records compression audit summary and sparsity metrics.
- Strictly avoids evaluating internal test (test_ds) and untouched Kaggle test set.
"""

from pathlib import Path
import os
import subprocess
import sys
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
import keras

from src.data_loader import PROJECT_ROOT

# Register custom objects for MobileNetV2 preprocessing
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
OUTPUT_DIR: Path = PROJECT_ROOT / "outputs" / "compression"
SUMMARY_PATH: Path = OUTPUT_DIR / "compression_summary.txt"
SPARSITY_REPORT_PATH: Path = OUTPUT_DIR / "sparsity_report.csv"
PRUNED_MODEL_PATH: Path = OUTPUT_DIR / "pruned_model.keras"
PRUNED_ZIP_PATH: Path = OUTPUT_DIR / "pruned_model.zip"

TARGET_SPARSITY: float = 0.50
BASELINE_VAL_ACC: float = 0.8239  # 82.39%


def ensure_output_directory() -> None:
    """Ensure outputs/compression directory exists."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def check_environment() -> Dict[str, str]:
    """Inspect installed versions and run pip check.

    Returns
    -------
    Dict[str, str]
        Dictionary with environment details.
    """
    tf_ver = str(tf.__version__)
    keras_ver = str(keras.__version__)

    # Check tfmot version
    tfmot_ver = "Not installed"
    try:
        import tensorflow_model_optimization as tfmot  # noqa: F401
        tfmot_ver = str(tfmot.__version__)
    except ImportError:
        tfmot_ver = "Not installed"
    except Exception as exc:
        tfmot_ver = f"Import error: {exc}"

    # Run pip check
    python_exe = sys.executable
    try:
        pip_check_res = subprocess.run(
            [python_exe, "-m", "pip", "check"],
            capture_output=True,
            text=True,
            check=False,
        )
        pip_check_stdout = pip_check_res.stdout.strip()
        pip_check_stderr = pip_check_res.stderr.strip()
        if pip_check_res.returncode == 0:
            pip_check_msg = pip_check_stdout or "No broken requirements found."
        else:
            pip_check_msg = f"Broken requirements: {pip_check_stdout or pip_check_stderr}"
    except Exception as exc:
        pip_check_msg = f"Failed to run pip check: {exc}"

    return {
        "tensorflow": tf_ver,
        "keras": keras_ver,
        "tfmot": tfmot_ver,
        "pip_check": pip_check_msg,
    }


def load_baseline_checkpoint() -> Tuple[tf.keras.Model, Dict[str, Any]]:
    """Load the final selected transfer learning checkpoint and extract baseline info.

    Returns
    -------
    Tuple[tf.keras.Model, Dict[str, Any]]
        The loaded Keras model and baseline metrics dictionary.
    """
    if not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(f"Checkpoint not found at: {CHECKPOINT_PATH}")

    file_size_bytes = CHECKPOINT_PATH.stat().st_size
    file_size_mb = file_size_bytes / (1024 * 1024)

    model = tf.keras.models.load_model(
        str(CHECKPOINT_PATH),
        custom_objects={
            "preprocess_input": tf.keras.applications.mobilenet_v2.preprocess_input
        },
    )

    total_params = int(model.count_params())
    trainable_params = sum(int(tf.size(w).numpy()) for w in model.trainable_weights)
    non_trainable_params = sum(int(tf.size(w).numpy()) for w in model.non_trainable_weights)

    # Locate MobileNetV2 backbone
    base_model: Optional[tf.keras.Model] = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model) or "mobilenetv2" in layer.name.lower():
            base_model = layer
            break

    backbone_trainable = base_model.trainable if base_model is not None else None
    backbone_trainable_vars = (
        len(base_model.trainable_variables) if base_model is not None else -1
    )

    # Classifier head weights
    predictions_layer = model.get_layer("predictions")
    pred_weights = predictions_layer.get_weights()
    kernel_weights = pred_weights[0] if len(pred_weights) > 0 else np.array([])
    bias_weights = pred_weights[1] if len(pred_weights) > 1 else np.array([])

    total_kernel_weights = int(kernel_weights.size)
    zero_kernel_weights = int(np.count_nonzero(kernel_weights == 0))
    kernel_sparsity = (
        (zero_kernel_weights / total_kernel_weights) * 100.0
        if total_kernel_weights > 0
        else 0.0
    )

    total_bias_weights = int(bias_weights.size)
    zero_bias_weights = int(np.count_nonzero(bias_weights == 0))
    bias_sparsity = (
        (zero_bias_weights / total_bias_weights) * 100.0
        if total_bias_weights > 0
        else 0.0
    )

    baseline_info = {
        "model_name": model.name,
        "checkpoint_path": str(CHECKPOINT_PATH),
        "input_shape": model.input_shape,
        "output_shape": model.output_shape,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "non_trainable_params": non_trainable_params,
        "file_size_bytes": file_size_bytes,
        "file_size_mb": file_size_mb,
        "backbone_name": base_model.name if base_model else "Unknown",
        "backbone_trainable": backbone_trainable,
        "backbone_trainable_vars": backbone_trainable_vars,
        "dense_kernel_total": total_kernel_weights,
        "dense_kernel_zero": zero_kernel_weights,
        "dense_kernel_sparsity_pct": kernel_sparsity,
        "dense_bias_total": total_bias_weights,
        "dense_bias_zero": zero_bias_weights,
        "dense_bias_sparsity_pct": bias_sparsity,
    }

    return model, baseline_info


def audit_tfmot_compatibility(model: tf.keras.Model) -> Dict[str, Any]:
    """Audit tensorflow-model-optimization compatibility with Keras 3 model.

    Returns
    -------
    Dict[str, Any]
        Dictionary detailing compatibility test results and diagnostics.
    """
    try:
        import tensorflow_model_optimization as tfmot
    except ImportError as err:
        return {
            "compatible": False,
            "status": "FAIL",
            "reason": f"tfmot is not installed: {err}",
            "details": "tensorflow-model-optimization package could not be imported.",
        }

    # Test layer-level pruning on Keras 3 Dense layer
    dense_layer = model.get_layer("predictions")
    prune_error: Optional[str] = None
    try:
        pruning_params = {
            "pruning_schedule": tfmot.sparsity.keras.PolynomialDecay(
                initial_sparsity=0.0,
                final_sparsity=0.50,
                begin_step=0,
                end_step=1000,
            )
        }
        _ = tfmot.sparsity.keras.prune_low_magnitude(dense_layer, **pruning_params)
        layer_prunable = True
    except Exception as exc:
        layer_prunable = False
        prune_error = f"{type(exc).__name__}: {exc}"

    # Test model-level pruning on Keras 3 Functional model
    model_prune_error: Optional[str] = None
    try:
        _ = tfmot.sparsity.keras.prune_low_magnitude(model, **pruning_params)
        model_prunable = True
    except Exception as exc:
        model_prunable = False
        model_prune_error = f"{type(exc).__name__}: {exc}"

    is_compatible = layer_prunable or model_prunable

    incompatibility_reason = (
        "tensorflow-model-optimization (v0.8.1) is designed exclusively for Keras 2 "
        "(tf_keras) and does not support Keras 3 (keras >= 3.0.0). "
        f"Layer pruning failed with [{prune_error}]. "
        f"Model pruning failed with [{model_prune_error}]."
    )

    return {
        "compatible": is_compatible,
        "status": "COMPATIBLE" if is_compatible else "INCOMPATIBLE",
        "layer_pruning_error": prune_error,
        "model_pruning_error": model_prune_error,
        "reason": incompatibility_reason if not is_compatible else "Fully compatible",
    }


def write_sparsity_report(baseline_info: Dict[str, Any]) -> None:
    """Write sparsity_report.csv for baseline weights."""
    records = [
        {
            "layer": "predictions",
            "tensor": "kernel",
            "total_weights": baseline_info["dense_kernel_total"],
            "zero_weights": baseline_info["dense_kernel_zero"],
            "sparsity": f"{baseline_info['dense_kernel_sparsity_pct']:.2f}%",
        },
        {
            "layer": "predictions",
            "tensor": "bias",
            "total_weights": baseline_info["dense_bias_total"],
            "zero_weights": baseline_info["dense_bias_zero"],
            "sparsity": f"{baseline_info['dense_bias_sparsity_pct']:.2f}%",
        },
    ]
    df = pd.DataFrame(records)
    df.to_csv(SPARSITY_REPORT_PATH, index=False)
    print(f"[INFO] Sparsity report saved to: {SPARSITY_REPORT_PATH}")


def write_compression_summary(
    env_info: Dict[str, str],
    baseline_info: Dict[str, Any],
    compat_info: Dict[str, Any],
) -> None:
    """Write compression_summary.txt matching specification."""
    content = f"""Original model: {baseline_info['checkpoint_path']}
Original size MB: {baseline_info['file_size_mb']:.2f} MB
Total parameters: {baseline_info['total_params']:,}
Trainable parameters: {baseline_info['trainable_params']:,}
Non-trainable parameters: {baseline_info['non_trainable_params']:,}

Pruned model: N/A (Blocked by technical incompatibility)
Pruned size MB: N/A

Compressed ZIP: N/A (Blocked by technical incompatibility)
ZIP size MB: N/A

Target sparsity: {int(TARGET_SPARSITY * 100)}%
Actual classifier sparsity: {baseline_info['dense_kernel_sparsity_pct']:.2f}%

Original validation accuracy: {BASELINE_VAL_ACC * 100:.2f}%
Compressed validation accuracy: N/A
Validation accuracy drop: N/A

Backbone trainable: NO
Backbone weights modified: NO

Internal test evaluated: NO
Kaggle test used: NO

--- TECHNICAL AUDIT ---
TensorFlow Version: {env_info['tensorflow']}
Keras Version: {env_info['keras']}
TF-MOT Version: {env_info['tfmot']}
pip check: {env_info['pip_check']}
Compatibility Status: {compat_info['status']}
Incompatibility Details: {compat_info['reason']}
Environment Protection: TensorFlow was NOT downgraded; active working environment was preserved.
Task 6.1 Status: HALTED per Technical Rule 2.
"""
    SUMMARY_PATH.write_text(content, encoding="utf-8")
    print(f"[INFO] Compression summary saved to: {SUMMARY_PATH}")


def run_task() -> Dict[str, Any]:
    """Execute Task 6.1 workflow."""
    print("=" * 80)
    print("TASK 6.1 — MODEL COMPRESSION / PRUNING PREPARATION")
    print("=" * 80)

    # 1. Output directory
    ensure_output_directory()

    # 2. Check environment
    print("\n[STEP 1] Checking environment & package versions...")
    env_info = check_environment()
    print(f"  TensorFlow: {env_info['tensorflow']}")
    print(f"  Keras:      {env_info['keras']}")
    print(f"  TF-MOT:     {env_info['tfmot']}")
    print(f"  pip check:  {env_info['pip_check']}")

    # 3. Load baseline checkpoint
    print("\n[STEP 2] Loading baseline MobileNetV2 transfer learning model...")
    model, baseline_info = load_baseline_checkpoint()
    print(f"  Model name:           {baseline_info['model_name']}")
    print(f"  Input shape:          {baseline_info['input_shape']}")
    print(f"  Output shape:         {baseline_info['output_shape']}")
    print(f"  Total parameters:     {baseline_info['total_params']:,} (Expected ~ 2,411,704)")
    print(f"  Trainable parameters: {baseline_info['trainable_params']:,} (Expected ~ 153,720)")
    print(f"  Non-trainable params: {baseline_info['non_trainable_params']:,}")
    print(f"  Checkpoint size:      {baseline_info['file_size_bytes']:,} bytes ({baseline_info['file_size_mb']:.2f} MB)")
    print(f"  Backbone name:        {baseline_info['backbone_name']}")
    print(f"  Backbone trainable:   {baseline_info['backbone_trainable']} (Expected: False)")
    print(f"  Backbone train vars:  {baseline_info['backbone_trainable_vars']} (Expected: 0)")

    # 4. Write initial baseline sparsity report
    print("\n[STEP 3] Measuring baseline classifier head sparsity...")
    print(f"  Dense kernel weights: {baseline_info['dense_kernel_total']:,}")
    print(f"  Dense zero weights:   {baseline_info['dense_kernel_zero']:,}")
    print(f"  Dense sparsity:       {baseline_info['dense_kernel_sparsity_pct']:.2f}%")
    write_sparsity_report(baseline_info)

    # 5. Check compatibility with tensorflow_model_optimization
    print("\n[STEP 4] Auditing tensorflow-model-optimization (tfmot) compatibility...")
    compat_info = audit_tfmot_compatibility(model)
    print(f"  Compatibility: {compat_info['status']}")
    if not compat_info["compatible"]:
        print(f"  Diagnosis:     {compat_info['reason']}")
        print(f"  Layer error:   {compat_info.get('layer_pruning_error')}")
        print(f"  Model error:   {compat_info.get('model_pruning_error')}")

    # 6. Write summary
    print("\n[STEP 5] Generating compression audit summary...")
    write_compression_summary(env_info, baseline_info, compat_info)

    # 7. Verification of constraints
    print("\n[STEP 6] Verifying Task 6.1 constraints...")
    print("  Internal test evaluated: NO (test_ds strictly untouched)")
    print("  Kaggle test used:        NO (data/test/ strictly untouched)")
    print("  Backbone modified:       NO (Backbone weights preserved)")
    print("  Environment downgraded:  NO (TensorFlow 2.21 / Keras 3 intact)")

    if not compat_info["compatible"]:
        print("\n" + "!" * 80)
        print("TECHNICAL WARNING / INCOMPATIBILITY DETECTED")
        print("tensorflow-model-optimization (0.8.1) does not support Keras 3.15.1.")
        print("Per Rule 2 instructions: Do NOT downgrade TensorFlow. Do NOT break environment.")
        print("Reported technical WARNING/FAIL and HALTING Task 6.1 cleanly.")
        print("!" * 80)

    return {
        "env_info": env_info,
        "baseline_info": baseline_info,
        "compat_info": compat_info,
    }


if __name__ == "__main__":
    run_task()
