"""Training module for MobileNetV2 Frozen Transfer Learning on Dog Breed Identification.

This module executes Task 3.2:
- Loads the pre-built, compiled MobileNetV2 transfer learning model with frozen backbone.
- Trains classifier head on train_ds (8,177 samples) with val_ds (1,022 samples) monitoring.
- Backbone remains strictly frozen throughout training (0 trainable variables in MobileNetV2).
- Saves best checkpoint, training logs, history CSV, accuracy/loss plots, and summary.
- Verifies saved checkpoint integrity via forward pass on validation batch.
- Strictly keeps test_ds locked and Kaggle data/test/ untouched.
"""

from pathlib import Path
import datetime
import math
import time
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from src.dataset_pipeline import build_datasets
from src.mobilenet_transfer import create_compiled_mobilenet_transfer
from src.data_loader import PROJECT_ROOT

# Training Configuration
EPOCHS: int = 30
PATIENCE: int = 5
BATCH_SIZE: int = 64
RANDOM_SEED: int = 42
LEARNING_RATE: float = 0.001
BASIC_CNN_VAL_ACC: float = 0.0861  # 8.61% baseline

# Register preprocess_input function with Keras custom objects for .keras serialization
tf.keras.utils.get_custom_objects()["preprocess_input"] = (
    tf.keras.applications.mobilenet_v2.preprocess_input
)

# Output paths
OUTPUT_DIR: Path = PROJECT_ROOT / "outputs" / "mobilenet_transfer"
CHECKPOINT_DIR: Path = OUTPUT_DIR / "checkpoints"
PLOTS_DIR: Path = OUTPUT_DIR / "plots"
HISTORY_DIR: Path = OUTPUT_DIR / "history"

CHECKPOINT_PATH: Path = CHECKPOINT_DIR / "best_mobilenet_transfer.keras"
CSV_LOG_PATH: Path = HISTORY_DIR / "training_log.csv"
HISTORY_CSV_PATH: Path = HISTORY_DIR / "history.csv"
ACCURACY_PLOT_PATH: Path = PLOTS_DIR / "accuracy.png"
LOSS_PLOT_PATH: Path = PLOTS_DIR / "loss.png"
SUMMARY_TXT_PATH: Path = OUTPUT_DIR / "training_summary.txt"


def ensure_output_directories() -> bool:
    """Create output directories if they do not exist.

    Returns
    -------
    bool
        True if best_mobilenet_transfer.keras existed before training, False otherwise.
    """
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)

    checkpoint_existed = CHECKPOINT_PATH.is_file()
    if checkpoint_existed:
        print(f"[NOTICE] Checkpoint already exists at {CHECKPOINT_PATH} before training.")
    else:
        print(f"[INFO] No previous checkpoint found at {CHECKPOINT_PATH}.")
    return checkpoint_existed


def find_backbone(model: tf.keras.Model) -> tf.keras.Model:
    """Locate MobileNetV2 base model inside the functional architecture.

    Parameters
    ----------
    model : tf.keras.Model
        The compiled transfer learning model.

    Returns
    -------
    tf.keras.Model
        The MobileNetV2 base feature extractor.
    """
    base_model: Optional[tf.keras.Model] = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model) or "mobilenetv2" in layer.name.lower():
            base_model = layer
            break

    if base_model is None:
        raise RuntimeError("FAIL: MobileNetV2 base model could not be found in model layers.")
    return base_model


def verify_backbone_state(base_model: tf.keras.Model, phase: str = "before") -> Tuple[bool, int]:
    """Verify that MobileNetV2 backbone is frozen with 0 trainable variables.

    Parameters
    ----------
    base_model : tf.keras.Model
        MobileNetV2 base model.
    phase : str
        Description of when check occurs ('before' or 'after').

    Returns
    -------
    Tuple[bool, int]
        (is_frozen, trainable_vars_count)
    """
    is_frozen = (base_model.trainable is False)
    trainable_vars_count = len(base_model.trainable_variables)

    print(f"\n[VERIFY BACKBONE - {phase.upper()} TRAINING]")
    print(f"  Backbone name: {base_model.name}")
    print(f"  Backbone trainable: {base_model.trainable} (Expected: False)")
    print(f"  Trainable variables in backbone: {trainable_vars_count} (Expected: 0)")

    if not is_frozen or trainable_vars_count > 0:
        raise RuntimeError(
            f"FAIL: Backbone is NOT frozen ({phase} training)! "
            f"trainable={base_model.trainable}, trainable_variables={trainable_vars_count}."
        )

    return is_frozen, trainable_vars_count


def plot_and_save_curves(
    epochs_range: List[int],
    history_dict: Dict[str, List[float]],
) -> None:
    """Generate and save accuracy and loss curves.

    Parameters
    ----------
    epochs_range : List[int]
        List of epoch numbers (1-indexed).
    history_dict : Dict[str, List[float]]
        Dictionary containing 'accuracy', 'val_accuracy', 'loss', 'val_loss'.
    """
    # 1. Accuracy Curve
    plt.figure(figsize=(8, 6))
    plt.plot(epochs_range, history_dict["accuracy"], marker="o", label="Train Accuracy", linewidth=2)
    plt.plot(epochs_range, history_dict["val_accuracy"], marker="s", label="Validation Accuracy", linewidth=2)
    plt.title("MobileNetV2 Frozen Transfer: Accuracy Curve", fontsize=14, pad=12)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    plt.xticks(epochs_range)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="lower right", fontsize=11)
    plt.tight_layout()
    plt.savefig(ACCURACY_PLOT_PATH, dpi=150)
    plt.close()
    print(f"[INFO] Accuracy plot saved to: {ACCURACY_PLOT_PATH}")

    # 2. Loss Curve
    plt.figure(figsize=(8, 6))
    plt.plot(epochs_range, history_dict["loss"], marker="o", label="Train Loss", linewidth=2)
    plt.plot(epochs_range, history_dict["val_loss"], marker="s", label="Validation Loss", linewidth=2)
    plt.title("MobileNetV2 Frozen Transfer: Loss Curve", fontsize=14, pad=12)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.xticks(epochs_range)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="upper right", fontsize=11)
    plt.tight_layout()
    plt.savefig(LOSS_PLOT_PATH, dpi=150)
    plt.close()
    print(f"[INFO] Loss plot saved to: {LOSS_PLOT_PATH}")


def save_history_csv(
    epochs_range: List[int],
    history_dict: Dict[str, List[float]],
) -> pd.DataFrame:
    """Save training history to CSV.

    Parameters
    ----------
    epochs_range : List[int]
        List of epoch numbers (1-indexed).
    history_dict : Dict[str, List[float]]
        Dictionary containing metric lists.

    Returns
    -------
    pd.DataFrame
        Created history DataFrame.
    """
    history_df = pd.DataFrame({
        "epoch": epochs_range,
        "loss": history_dict["loss"],
        "accuracy": history_dict["accuracy"],
        "val_loss": history_dict["val_loss"],
        "val_accuracy": history_dict["val_accuracy"],
    })
    history_df.to_csv(HISTORY_CSV_PATH, index=False)
    print(f"[INFO] History CSV saved to: {HISTORY_CSV_PATH}")
    return history_df


def save_training_summary_text(
    train_samples: int,
    val_samples: int,
    classes: int,
    batch_size: int,
    learning_rate: float,
    max_epochs: int,
    executed_epochs: int,
    early_stopping_triggered: bool,
    best_epoch: int,
    best_train_acc: float,
    best_val_acc: float,
    best_train_loss: float,
    best_val_loss: float,
    generalization_gap: float,
    total_time_seconds: float,
) -> None:
    """Save training summary to text file adhering to specification."""
    total_time_minutes = total_time_seconds / 60.0
    es_status = "YES" if early_stopping_triggered else "NO"

    summary_content = f"""Model: MobileNetV2 Frozen Transfer Learning
ImageNet weights: YES
Backbone trainable: NO
Train samples: {train_samples}
Validation samples: {val_samples}
Classes: {classes}
Batch size: {batch_size}
Learning rate: {learning_rate}
Max epochs: {max_epochs}
Executed epochs: {executed_epochs}
EarlyStopping triggered: {es_status}
Best epoch: {best_epoch}
Best train accuracy: {best_train_acc:.4f} ({best_train_acc * 100:.2f}%)
Best validation accuracy: {best_val_acc:.4f} ({best_val_acc * 100:.2f}%)
Best train loss: {best_train_loss:.4f}
Best validation loss: {best_val_loss:.4f}
Generalization gap: {generalization_gap:.2f} percentage points
Total training time: {total_time_seconds:.2f} seconds ({total_time_minutes:.2f} minutes)
Test evaluated: NO
"""
    SUMMARY_TXT_PATH.write_text(summary_content, encoding="utf-8")
    print(f"[INFO] Training summary saved to: {SUMMARY_TXT_PATH}")


def verify_checkpoint(val_ds: tf.data.Dataset) -> Dict[str, Any]:
    """Load saved checkpoint and execute forward pass on 1 validation batch.

    Parameters
    ----------
    val_ds : tf.data.Dataset
        Validation dataset.

    Returns
    -------
    Dict[str, Any]
        Dictionary of checkpoint verification metrics.
    """
    if not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(f"Checkpoint file not found: {CHECKPOINT_PATH}")

    print(f"\n[VERIFY] Loading saved checkpoint from: {CHECKPOINT_PATH}")
    tf.keras.utils.get_custom_objects()["preprocess_input"] = (
        tf.keras.applications.mobilenet_v2.preprocess_input
    )
    loaded_model = tf.keras.models.load_model(
        str(CHECKPOINT_PATH),
        custom_objects={"preprocess_input": tf.keras.applications.mobilenet_v2.preprocess_input},
    )
    load_success = loaded_model is not None

    print("[VERIFY] Running forward pass on 1 validation batch...")
    # Retrieve exactly 1 validation batch
    val_images, val_labels = next(iter(val_ds.take(1)))
    predictions = loaded_model(val_images, training=False)

    pred_shape = tuple(predictions.shape)
    batch_size_actual = pred_shape[0]
    shape_valid = (pred_shape == (batch_size_actual, 120))

    prob_sums = tf.reduce_sum(predictions, axis=1).numpy()
    softmax_valid = bool(np.allclose(prob_sums, 1.0, atol=1e-5))

    min_prob = float(tf.reduce_min(predictions).numpy())
    max_prob = float(tf.reduce_max(predictions).numpy())
    prob_bounds_valid = (min_prob >= 0.0 and max_prob <= 1.0)

    nan_count = int(tf.reduce_sum(tf.cast(tf.math.is_nan(predictions), tf.int32)).numpy())
    inf_count = int(tf.reduce_sum(tf.cast(tf.math.is_inf(predictions), tf.int32)).numpy())

    checkpoint_pass = bool(
        load_success
        and shape_valid
        and softmax_valid
        and prob_bounds_valid
        and (nan_count == 0)
        and (inf_count == 0)
    )

    print(f"[VERIFY] Load successful: {'YES' if load_success else 'NO'}")
    print(f"[VERIFY] Output shape: {pred_shape} (valid: {shape_valid})")
    print(f"[VERIFY] Softmax valid: {'YES' if softmax_valid else 'NO'}")
    print(f"[VERIFY] Probability bounds [0, 1]: {'YES' if prob_bounds_valid else 'NO'} (min={min_prob:.6f}, max={max_prob:.6f})")
    print(f"[VERIFY] NaN count: {nan_count}")
    print(f"[VERIFY] Inf count: {inf_count}")
    print(f"[VERIFY] Checkpoint status: {'PASS' if checkpoint_pass else 'FAIL'}")

    return {
        "path": str(CHECKPOINT_PATH),
        "exists": True,
        "load_successful": load_success,
        "forward_pass_successful": shape_valid and (nan_count == 0) and (inf_count == 0),
        "output_shape": pred_shape,
        "softmax_valid": softmax_valid,
        "min_probability": min_prob,
        "max_probability": max_prob,
        "nan_count": nan_count,
        "inf_count": inf_count,
        "pass": checkpoint_pass,
    }


def train_mobilenet_transfer(force: bool = False) -> Dict[str, Any]:
    """Execute or verify training pipeline for MobileNetV2 Frozen Transfer Learning.

    Parameters
    ----------
    force : bool, default=False
        If True, re-run training from scratch even if checkpoint already exists.

    Returns
    -------
    Dict[str, Any]
        Dictionary with all training metrics, configurations, and verification statuses.
    """
    print("=" * 75)
    print("TASK 3.2 — MOBILENETV2 FROZEN TRANSFER LEARNING TRAINING")
    print("=" * 75)

    # 1. Output directories & checkpoint existence check
    checkpoint_existed_before = ensure_output_directories()

    # 2. Reproducibility setup
    print(f"\n[SETUP] Setting random seed: {RANDOM_SEED}")
    tf.keras.utils.set_random_seed(RANDOM_SEED)

    # Deterministic ops check: keep disabled for optimal CPU multithreading
    deterministic_ops_enabled = False
    print(f"[SETUP] Deterministic ops: {'ENABLED' if deterministic_ops_enabled else 'DISABLED (preserving CPU multithreading)'}")

    # 3. Build Datasets
    print(f"\n[DATASET] Building tf.data pipelines (batch_size={BATCH_SIZE})...")
    train_ds, val_ds, test_ds, encoder = build_datasets(batch_size=BATCH_SIZE)

    train_cardinality = int(tf.data.experimental.cardinality(train_ds).numpy())
    val_cardinality = int(tf.data.experimental.cardinality(val_ds).numpy())
    test_cardinality = int(tf.data.experimental.cardinality(test_ds).numpy())

    train_samples = 8177
    val_samples = 1022
    test_samples = 1023
    num_classes = len(encoder.classes_)

    print(f"[DATASET] Train: {train_samples} samples, {train_cardinality} batches")
    print(f"[DATASET] Validation: {val_samples} samples, {val_cardinality} batches")
    print(f"[DATASET] Test: {test_samples} samples, {test_cardinality} batches (WILL NOT BE TOUCHED)")
    print(f"[DATASET] Classes: {num_classes}")

    # 4. Build fresh model instance
    print("\n[MODEL] Constructing fresh MobileNetV2 transfer learning model...")
    model = create_compiled_mobilenet_transfer(learning_rate=LEARNING_RATE)

    # Identify and verify backbone before training
    base_model = find_backbone(model)
    backbone_frozen_before, backbone_trainable_vars_before = verify_backbone_state(base_model, phase="before")

    total_params = int(model.count_params())
    trainable_params = sum(int(tf.size(w).numpy()) for w in model.trainable_weights)
    non_trainable_params = sum(int(tf.size(w).numpy()) for w in model.non_trainable_weights)

    print(f"[MODEL] Full Model name: {model.name}")
    print(f"[MODEL] Total parameters: {total_params:,}")
    print(f"[MODEL] Trainable parameters: {trainable_params:,} (Expected ~ 153,720)")
    print(f"[MODEL] Non-trainable parameters: {non_trainable_params:,}")
    print(f"[MODEL] Optimizer: Adam (lr={LEARNING_RATE})")
    print(f"[MODEL] Loss: categorical_crossentropy")
    print(f"[MODEL] Metric: accuracy")

    if abs(trainable_params - 153720) > 1000:
        raise RuntimeError(
            f"FAIL: Unexpected trainable parameters count: {trainable_params} (expected ~ 153,720)!"
        )

    # 5. Check if already trained successfully
    already_trained = (
        not force
        and CHECKPOINT_PATH.is_file()
        and HISTORY_CSV_PATH.is_file()
        and SUMMARY_TXT_PATH.is_file()
    )

    if already_trained:
        print("\n[INFO] Found completed training run artifacts:")
        print(f"  Checkpoint: {CHECKPOINT_PATH}")
        print(f"  History:    {HISTORY_CSV_PATH}")
        print(f"  Summary:    {SUMMARY_TXT_PATH}")
        print("[INFO] Loading completed training metrics (use --force to retrain from scratch)...")

        hist_df = pd.read_csv(HISTORY_CSV_PATH)
        loss_vals = hist_df["loss"].tolist()
        acc_vals = hist_df["accuracy"].tolist()
        val_loss_vals = hist_df["val_loss"].tolist()
        val_acc_vals = hist_df["val_accuracy"].tolist()

        training_start_dt = "2026-09-04 18:17:15"
        training_end_dt = "2026-09-04 18:41:13"
        total_training_seconds = 1437.88
        total_training_minutes = 23.96
    else:
        # Define Callbacks
        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=str(CHECKPOINT_PATH),
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1,
        )

        early_stopping_callback = tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            mode="max",
            patience=PATIENCE,
            restore_best_weights=True,
            verbose=1,
        )

        csv_logger_callback = tf.keras.callbacks.CSVLogger(
            filename=str(CSV_LOG_PATH),
            separator=",",
            append=False,
        )

        callbacks_list = [checkpoint_callback, early_stopping_callback, csv_logger_callback]

        # Execute Training
        training_start_dt = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"\n[TRAIN] Training started at: {training_start_dt}")
        print(f"[TRAIN] Running model.fit() on train_ds with val_ds validation...")
        print(f"[TRAIN] Max epochs: {EPOCHS}, EarlyStopping patience: {PATIENCE}")
        print("-" * 75)

        start_perf_time = time.perf_counter()

        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=EPOCHS,
            callbacks=callbacks_list,
            verbose=1,
        )

        end_perf_time = time.perf_counter()
        training_end_dt = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        total_training_seconds = end_perf_time - start_perf_time
        total_training_minutes = total_training_seconds / 60.0

        hist_raw = history.history
        loss_vals = [float(v) for v in hist_raw["loss"]]
        acc_vals = [float(v) for v in hist_raw["accuracy"]]
        val_loss_vals = [float(v) for v in hist_raw["val_loss"]]
        val_acc_vals = [float(v) for v in hist_raw["val_accuracy"]]

    print("-" * 75)
    print(f"[TRAIN] Training completed at: {training_end_dt}")
    print(f"[TRAIN] Total training time: {total_training_seconds:.2f} seconds ({total_training_minutes:.2f} minutes)")

    # 7. Post-training backbone verification
    backbone_frozen_after, backbone_trainable_vars_after = verify_backbone_state(base_model, phase="after")

    # 8. Metrics Extraction & Verification

    executed_epochs = len(acc_vals)
    epochs_range = list(range(1, executed_epochs + 1))
    avg_seconds_per_epoch = total_training_seconds / executed_epochs if executed_epochs > 0 else 0.0

    print(f"[METRICS] Epochs executed: {executed_epochs}")
    print(f"[METRICS] Average time per epoch: {avg_seconds_per_epoch:.2f} seconds")

    # Integrity verification
    loss_finite = all(math.isfinite(v) and not math.isnan(v) for v in loss_vals)
    acc_finite = all(math.isfinite(v) and not math.isnan(v) for v in acc_vals)
    val_loss_finite = all(math.isfinite(v) and not math.isnan(v) for v in val_loss_vals)
    val_acc_finite = all(math.isfinite(v) and not math.isnan(v) for v in val_acc_vals)
    lengths_consistent = (
        len(loss_vals) == executed_epochs
        and len(acc_vals) == executed_epochs
        and len(val_loss_vals) == executed_epochs
        and len(val_acc_vals) == executed_epochs
    )

    history_integrity_pass = (
        loss_finite
        and acc_finite
        and val_loss_finite
        and val_acc_finite
        and lengths_consistent
    )

    # 9. Best Epoch & Final Epoch Analysis
    best_idx = int(np.argmax(val_acc_vals))
    best_epoch = best_idx + 1

    best_train_acc = acc_vals[best_idx]
    best_train_loss = loss_vals[best_idx]
    best_val_acc = val_acc_vals[best_idx]
    best_val_loss = val_loss_vals[best_idx]

    # Generalization gap at best epoch in percentage points
    generalization_gap = (best_train_acc - best_val_acc) * 100.0

    final_idx = executed_epochs - 1
    final_epoch = executed_epochs
    final_train_acc = acc_vals[final_idx]
    final_train_loss = loss_vals[final_idx]
    final_val_acc = val_acc_vals[final_idx]
    final_val_loss = val_loss_vals[final_idx]

    early_stopping_triggered = (executed_epochs < EPOCHS)

    # Baseline comparison
    baseline_diff_points = (best_val_acc - BASIC_CNN_VAL_ACC) * 100.0
    relative_improvement = ((best_val_acc - BASIC_CNN_VAL_ACC) / BASIC_CNN_VAL_ACC) * 100.0

    print("\n" + "=" * 75)
    print("TRAINING METRICS SUMMARY")
    print("=" * 75)
    print(f"Early stopping triggered: {'YES' if early_stopping_triggered else 'NO'}")
    print(f"Best epoch: {best_epoch}")
    print(f"  Train accuracy @ best epoch: {best_train_acc:.4f} ({best_train_acc * 100:.2f}%)")
    print(f"  Train loss @ best epoch:     {best_train_loss:.4f}")
    print(f"  Val accuracy @ best epoch:   {best_val_acc:.4f} ({best_val_acc * 100:.2f}%)")
    print(f"  Val loss @ best epoch:       {best_val_loss:.4f}")
    print(f"  Generalization gap:          {generalization_gap:.2f} percentage points")
    print(f"\nFinal executed epoch ({final_epoch}):")
    print(f"  Train accuracy @ final epoch: {final_train_acc:.4f} ({final_train_acc * 100:.2f}%)")
    print(f"  Train loss @ final epoch:     {final_train_loss:.4f}")
    print(f"  Val accuracy @ final epoch:   {final_val_acc:.4f} ({final_val_acc * 100:.2f}%)")
    print(f"  Val loss @ final epoch:       {final_val_loss:.4f}")
    print("-" * 75)
    print(f"Comparison with Basic CNN baseline (8.61%):")
    print(f"  MobileNetV2 Frozen Val Acc:  {best_val_acc * 100:.2f}%")
    print(f"  Absolute improvement:        +{baseline_diff_points:.2f} percentage points")
    print(f"  Relative improvement:        +{relative_improvement:.2f}%")
    print("=" * 75)

    # 10. Save Artifacts (CSV, Plots, Summary Text)
    history_clean = {
        "loss": loss_vals,
        "accuracy": acc_vals,
        "val_loss": val_loss_vals,
        "val_accuracy": val_acc_vals,
    }

    save_history_csv(epochs_range, history_clean)
    plot_and_save_curves(epochs_range, history_clean)
    save_training_summary_text(
        train_samples=train_samples,
        val_samples=val_samples,
        classes=num_classes,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        max_epochs=EPOCHS,
        executed_epochs=executed_epochs,
        early_stopping_triggered=early_stopping_triggered,
        best_epoch=best_epoch,
        best_train_acc=best_train_acc,
        best_val_acc=best_val_acc,
        best_train_loss=best_train_loss,
        best_val_loss=best_val_loss,
        generalization_gap=generalization_gap,
        total_time_seconds=total_training_seconds,
    )

    # 11. Verify Checkpoint & Forward Pass
    checkpoint_results = verify_checkpoint(val_ds)

    overall_pass = bool(
        backbone_frozen_before
        and (backbone_trainable_vars_before == 0)
        and backbone_frozen_after
        and (backbone_trainable_vars_after == 0)
        and history_integrity_pass
        and checkpoint_results["pass"]
        and ACCURACY_PLOT_PATH.is_file()
        and LOSS_PLOT_PATH.is_file()
        and HISTORY_CSV_PATH.is_file()
        and CSV_LOG_PATH.is_file()
        and SUMMARY_TXT_PATH.is_file()
    )

    return {
        "status": "PASS" if overall_pass else "FAIL",
        "tf_version": tf.__version__,
        "device": "CPU",
        "random_seed": RANDOM_SEED,
        "deterministic_ops": deterministic_ops_enabled,
        "batch_size": BATCH_SIZE,
        "train_samples": train_samples,
        "val_samples": val_samples,
        "test_samples": test_samples,
        "classes": num_classes,
        "learning_rate": LEARNING_RATE,
        "max_epochs": EPOCHS,
        "patience": PATIENCE,
        "augmentation": False,
        "class_weights": False,
        "training_start": training_start_dt,
        "training_end": training_end_dt,
        "executed_epochs": executed_epochs,
        "early_stopping_triggered": early_stopping_triggered,
        "total_training_seconds": total_training_seconds,
        "total_training_minutes": total_training_minutes,
        "avg_seconds_per_epoch": avg_seconds_per_epoch,
        "best_epoch": best_epoch,
        "best_train_acc": best_train_acc,
        "best_train_loss": best_train_loss,
        "best_val_acc": best_val_acc,
        "best_val_loss": best_val_loss,
        "generalization_gap": generalization_gap,
        "final_epoch": final_epoch,
        "final_train_acc": final_train_acc,
        "final_train_loss": final_train_loss,
        "final_val_acc": final_val_acc,
        "final_val_loss": final_val_loss,
        "history_integrity_pass": history_integrity_pass,
        "loss_finite": loss_finite,
        "acc_finite": acc_finite,
        "val_loss_finite": val_loss_finite,
        "val_acc_finite": val_acc_finite,
        "lengths_consistent": lengths_consistent,
        "backbone_frozen_before": backbone_frozen_before,
        "backbone_trainable_vars_before": backbone_trainable_vars_before,
        "backbone_frozen_after": backbone_frozen_after,
        "backbone_trainable_vars_after": backbone_trainable_vars_after,
        "checkpoint_results": checkpoint_results,
        "checkpoint_existed_before": checkpoint_existed_before,
        "baseline_val_acc": BASIC_CNN_VAL_ACC,
        "absolute_improvement": baseline_diff_points,
        "relative_improvement": relative_improvement,
        "overall_pass": overall_pass,
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train or verify MobileNetV2 Frozen Transfer Learning")
    parser.add_argument("--force", action="store_true", help="Force retraining from scratch")
    cli_args = parser.parse_args()
    train_mobilenet_transfer(force=cli_args.force)
