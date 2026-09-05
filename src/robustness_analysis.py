"""Robustness Analysis module for Dog Breed Identification.

This module executes Task 5.4:
- Evaluates the final selected model (MobileNetV2 Frozen Transfer Learning checkpoint:
  outputs/mobilenet_transfer/checkpoints/best_mobilenet_transfer.keras)
  under corrupted test conditions:
    * Clean baseline (for sanity check vs Task 5.1)
    * Gaussian blur (mild, medium, strong)
    * Low brightness (mild, medium, strong)
- Strictly post-hoc analysis:
    * No model selection
    * No hyperparameter tuning
    * No retraining or fine-tuning
    * No changes to architecture or augmentation
- Pure in-memory transformations without touching original image files.
- Generates required CSV summaries, plots, per-class degradation reports,
  and failure examples.
"""

from pathlib import Path
import math
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from src.dataset_pipeline import build_datasets, load_split_csv
from src.data_loader import PROJECT_ROOT, TRAIN_DIR
from src.label_encoder import NUM_CLASSES

# Register preprocess_input function with Keras custom objects
tf.keras.utils.get_custom_objects()["preprocess_input"] = (
    tf.keras.applications.mobilenet_v2.preprocess_input
)

# Configuration
BATCH_SIZE: int = 64
EXPECTED_TEST_SAMPLES: int = 1023
EXPECTED_CLASSES: int = 120
CLEAN_BASELINE_ACCURACY: float = 0.8201  # 82.01% from Task 5.1

# Checkpoint path for final selected model (only this model is evaluated)
CHECKPOINT_PATH: Path = (
    PROJECT_ROOT
    / "outputs"
    / "mobilenet_transfer"
    / "checkpoints"
    / "best_mobilenet_transfer.keras"
)

# Output directory for robustness analysis artifacts
ROBUSTNESS_DIR: Path = PROJECT_ROOT / "outputs" / "robustness"


def ensure_robustness_dir() -> Path:
    """Create outputs/robustness directory if it does not exist.

    Returns
    -------
    Path
        Path to the robustness directory.
    """
    ROBUSTNESS_DIR.mkdir(parents=True, exist_ok=True)
    return ROBUSTNESS_DIR


# ==============================================================================
# SECTION A: GAUSSIAN BLUR TRANSFORMATION (PURE TENSORFLOW)
# ==============================================================================

def apply_gaussian_blur(
    images: tf.Tensor,
    kernel_size: int = 5,
    sigma: float = 1.5,
) -> tf.Tensor:
    """Apply Gaussian blur to batched RGB image tensor using depthwise convolution.

    Parameters
    ----------
    images : tf.Tensor
        Input image batch of shape (B, 224, 224, 3), float32, pixel range [0, 255].
    kernel_size : int, default=5
        Size of Gaussian blur kernel (odd integer: 3, 5, 9).
    sigma : float, default=1.5
        Standard deviation of Gaussian distribution.

    Returns
    -------
    tf.Tensor
        Blurred image batch with identical shape (B, 224, 224, 3) and dtype float32.
    """
    radius = kernel_size // 2
    coords = tf.range(-radius, radius + 1, dtype=tf.float32)
    # 1D Gaussian kernel
    g1d = tf.exp(-0.5 * tf.square(coords / sigma))
    g1d = g1d / tf.reduce_sum(g1d)
    # 2D Gaussian kernel via outer product
    g2d = tf.tensordot(g1d, g1d, axes=0)
    # Shape for tf.nn.depthwise_conv2d: [filter_height, filter_width, in_channels, channel_multiplier]
    g_kernel = tf.tile(g2d[:, :, tf.newaxis, tf.newaxis], [1, 1, 3, 1])
    blurred = tf.nn.depthwise_conv2d(
        images,
        g_kernel,
        strides=[1, 1, 1, 1],
        padding="SAME",
    )
    return blurred


# ==============================================================================
# SECTION B: LOW-LIGHT / BRIGHTNESS FACTOR TRANSFORMATION
# ==============================================================================

def apply_brightness_factor(
    images: tf.Tensor,
    factor: float,
) -> tf.Tensor:
    """Scale image pixel brightness deterministically and clip to [0.0, 255.0].

    Parameters
    ----------
    images : tf.Tensor
        Input image batch of shape (B, 224, 224, 3), float32.
    factor : float
        Multiplicative brightness factor (0.75, 0.50, 0.25).

    Returns
    -------
    tf.Tensor
        Darkened image batch with identical shape (B, 224, 224, 3) and dtype float32.
    """
    return tf.clip_by_value(images * factor, 0.0, 255.0)


# ==============================================================================
# PREDICTION AND EVALUATION HELPER
# ==============================================================================

def evaluate_transformed_test(
    model: tf.keras.Model,
    test_ds: tf.data.Dataset,
    transform_fn: Optional[Callable[[tf.Tensor], tf.Tensor]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, int, int]:
    """Evaluate model on test dataset with optional in-memory image transformation.

    Parameters
    ----------
    model : tf.keras.Model
        Loaded Keras model (frozen, inference mode).
    test_ds : tf.data.Dataset
        Batched test dataset yielding (images, labels).
    transform_fn : Optional[Callable[[tf.Tensor], tf.Tensor]], default=None
        In-memory image transformation function. If None, clean images are used.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray, float, int, int]
        - y_true: Ground truth class indices (N,)
        - y_pred: Predicted class indices (N,)
        - predictions: Softmax probability matrix (N, 120)
        - accuracy: Fraction of correct predictions [0..1]
        - correct: Count of correct predictions
        - incorrect: Count of incorrect predictions
    """
    y_true_list: List[np.ndarray] = []
    preds_list: List[np.ndarray] = []

    for images_batch, labels_batch in test_ds:
        if transform_fn is not None:
            images_transformed = transform_fn(images_batch)
        else:
            images_transformed = images_batch

        # Forward pass in inference mode (training=False)
        batch_preds = model(images_transformed, training=False)
        preds_list.append(batch_preds.numpy())
        y_true_list.append(labels_batch.numpy())

    predictions = np.concatenate(preds_list, axis=0)
    y_true_onehot = np.concatenate(y_true_list, axis=0)
    y_true = np.argmax(y_true_onehot, axis=1)
    y_pred = np.argmax(predictions, axis=1)

    # Prediction integrity validations
    if predictions.shape != (EXPECTED_TEST_SAMPLES, EXPECTED_CLASSES):
        raise ValueError(
            f"Expected predictions shape ({EXPECTED_TEST_SAMPLES}, {EXPECTED_CLASSES}), got {predictions.shape}"
        )

    nan_count = int(np.isnan(predictions).sum())
    inf_count = int(np.isinf(predictions).sum())
    if nan_count > 0 or inf_count > 0:
        raise ValueError(f"Predictions contain invalid values: NaN={nan_count}, Inf={inf_count}")

    prob_sums = np.sum(predictions, axis=1)
    if not np.allclose(prob_sums, 1.0, atol=1e-4):
        max_dev = float(np.max(np.abs(prob_sums - 1.0)))
        print(f"  [WARNING] Softmax probabilities sum deviation: max deviation={max_dev:.6f}")

    correct = int(np.sum(y_true == y_pred))
    incorrect = int(np.sum(y_true != y_pred))
    accuracy = float(np.mean(y_true == y_pred))

    return y_true, y_pred, predictions, accuracy, correct, incorrect


# ==============================================================================
# PLOTTING FUNCTIONS
# ==============================================================================

def plot_robustness_accuracy(
    summary_df: pd.DataFrame,
    clean_acc_pct: float,
    output_path: Path,
) -> None:
    """Plot test accuracy across clean and corrupted conditions.

    Parameters
    ----------
    summary_df : pd.DataFrame
        DataFrame with columns ['condition', 'accuracy', 'drop_vs_clean_pp'].
    clean_acc_pct : float
        Clean test accuracy in percentage (e.g. 82.01).
    output_path : Path
        Target image file path.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    conditions = summary_df["condition"].tolist()
    accuracies = [acc * 100 for acc in summary_df["accuracy"]]

    # Colors: blue for clean, orange shades for blur, purple shades for dark
    colors = [
        "#1f77b4",  # Clean
        "#ffbb78",  # Blur mild
        "#ff7f0e",  # Blur medium
        "#d62728",  # Blur strong
        "#c5b0d5",  # Dark mild
        "#9467bd",  # Dark medium
        "#4b0082",  # Dark strong
    ]

    bars = ax.bar(conditions, accuracies, color=colors, width=0.55, edgecolor="black", linewidth=0.8)

    # Reference line for clean baseline
    ax.axhline(
        clean_acc_pct,
        color="#1f77b4",
        linestyle="--",
        linewidth=1.5,
        alpha=0.8,
        label=f"Clean Baseline ({clean_acc_pct:.2f}%)",
    )

    # Value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f"{height:.2f}%",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    ax.set_title("Model Robustness Under Visual Corruptions (Internal Test Set)", fontsize=13, fontweight="bold", pad=12)
    ax.set_ylabel("Test Accuracy (%)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Evaluation Condition", fontsize=11, fontweight="bold")
    ax.set_ylim(0, 100)
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    ax.legend(loc="lower left", fontsize=10)

    plt.xticks(rotation=20, ha="right", fontsize=10)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"[INFO] Saved accuracy plot to: {output_path}")


def plot_accuracy_drop(
    summary_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot accuracy degradation (drop vs clean) in percentage points.

    Parameters
    ----------
    summary_df : pd.DataFrame
        DataFrame with columns ['condition', 'drop_vs_clean_pp'].
    output_path : Path
        Target image file path.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    conditions = summary_df["condition"].tolist()
    drops = summary_df["drop_vs_clean_pp"].tolist()

    colors = [
        "#7f7f7f",  # Clean (0.0 pp)
        "#ffbb78",  # Blur mild
        "#ff7f0e",  # Blur medium
        "#d62728",  # Blur strong
        "#c5b0d5",  # Dark mild
        "#9467bd",  # Dark medium
        "#4b0082",  # Dark strong
    ]

    bars = ax.bar(conditions, drops, color=colors, width=0.55, edgecolor="black", linewidth=0.8)

    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f"{height:.2f} pp",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    ax.set_title("Accuracy Degradation vs Clean Baseline (Percentage Points)", fontsize=13, fontweight="bold", pad=12)
    ax.set_ylabel("Accuracy Drop (pp)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Evaluation Condition", fontsize=11, fontweight="bold")
    max_drop = max(drops) if drops else 10.0
    ax.set_ylim(0, max(max_drop * 1.25, 10.0))
    ax.grid(axis="y", linestyle=":", alpha=0.6)

    plt.xticks(rotation=20, ha="right", fontsize=10)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"[INFO] Saved degradation plot to: {output_path}")


# ==============================================================================
# MAIN ROBUSTNESS ANALYSIS PIPELINE
# ==============================================================================

def run_robustness_analysis() -> Dict[str, Any]:
    """Execute complete Task 5.4 Robustness Analysis pipeline.

    Returns
    -------
    dict
        Comprehensive results dictionary containing metrics, conditions, and findings.
    """
    print("=" * 80)
    print("TASK 5.4: ROBUSTNESS ANALYSIS ON BLURRED AND LOW-LIGHT IMAGES")
    print("=" * 80)
    print(f"Selected Checkpoint: {CHECKPOINT_PATH}")
    print(f"Test Samples:        {EXPECTED_TEST_SAMPLES}")
    print(f"Classes:             {EXPECTED_CLASSES}")
    print(f"Expected Clean Acc:  {CLEAN_BASELINE_ACCURACY * 100:.2f}%\n")

    ensure_robustness_dir()

    # 1. Verify Checkpoint Exists
    if not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(f"Selected checkpoint not found at: {CHECKPOINT_PATH}")

    # 2. Build Datasets (batch_size=64)
    print("[INFO] Building tf.data pipelines (test_ds only)...")
    train_ds, val_ds, test_ds, encoder = build_datasets(batch_size=BATCH_SIZE)

    # Discard train_ds and val_ds explicitly to prevent any accidental use
    del train_ds, val_ds

    # Enforce deterministic order for test dataset
    options = tf.data.Options()
    options.deterministic = True
    test_ds = test_ds.with_options(options)

    class_names = [str(c) for c in encoder.classes_]
    if len(class_names) != EXPECTED_CLASSES:
        raise ValueError(f"Expected {EXPECTED_CLASSES} classes, got {len(class_names)}")

    test_df = load_split_csv("test")
    if len(test_df) != EXPECTED_TEST_SAMPLES:
        raise ValueError(f"Expected {EXPECTED_TEST_SAMPLES} test samples, got {len(test_df)}")

    # 3. Load Model Checkpoint
    print(f"[INFO] Loading model checkpoint from: {CHECKPOINT_PATH}")
    model = tf.keras.models.load_model(str(CHECKPOINT_PATH))
    print("[INFO] Model loaded successfully.")

    # Snapshot weights before evaluation to ensure immutability
    initial_weights = [w.numpy().copy() for w in model.weights]

    # Define all 7 evaluation conditions
    conditions_spec: List[Tuple[str, str, Optional[Callable[[tf.Tensor], tf.Tensor]]]] = [
        ("Clean", "None", None),
        ("Blur mild", "kernel=3, sigma=0.8", lambda x: apply_gaussian_blur(x, kernel_size=3, sigma=0.8)),
        ("Blur medium", "kernel=5, sigma=1.5", lambda x: apply_gaussian_blur(x, kernel_size=5, sigma=1.5)),
        ("Blur strong", "kernel=9, sigma=3.0", lambda x: apply_gaussian_blur(x, kernel_size=9, sigma=3.0)),
        ("Dark mild", "factor=0.75", lambda x: apply_brightness_factor(x, factor=0.75)),
        ("Dark medium", "factor=0.50", lambda x: apply_brightness_factor(x, factor=0.50)),
        ("Dark strong", "factor=0.25", lambda x: apply_brightness_factor(x, factor=0.25)),
    ]

    evaluation_results: Dict[str, Dict[str, Any]] = {}
    summary_records: List[Dict[str, Any]] = []

    print("\n" + "=" * 80)
    print("EVALUATING CONDITIONS (7 TOTAL)")
    print("=" * 80)

    clean_acc: float = 0.0

    for cond_name, param_desc, transform_fn in conditions_spec:
        print(f"\nEvaluating condition: [{cond_name}] (Parameter: {param_desc})...")
        start_t = time.time()
        y_true, y_pred, preds, acc, corr, incorr = evaluate_transformed_test(
            model=model,
            test_ds=test_ds,
            transform_fn=transform_fn,
        )
        duration = time.time() - start_t

        if cond_name == "Clean":
            clean_acc = acc
            diff = abs(acc - CLEAN_BASELINE_ACCURACY)
            print(f"  -> Accuracy: {acc * 100:.2f}% | Correct: {corr} | Incorrect: {incorr} | ({duration:.2f}s)")
            print(f"  -> Baseline Verification: Clean accuracy={acc * 100:.2f}%, Expected={CLEAN_BASELINE_ACCURACY * 100:.2f}%, Diff={diff * 100:.4f} pp")
            if diff > 1e-3:
                raise RuntimeError(
                    f"FAIL: Discrepancy between measured clean accuracy ({acc * 100:.2f}%) "
                    f"and Task 5.1 baseline ({CLEAN_BASELINE_ACCURACY * 100:.2f}%) exceeds threshold: {diff:.6f}"
                )
            drop_pp = 0.0
        else:
            drop_pp = (clean_acc - acc) * 100.0
            print(f"  -> Accuracy: {acc * 100:.2f}% | Correct: {corr} | Incorrect: {incorr} | Drop vs Clean: {drop_pp:.2f} pp | ({duration:.2f}s)")

        evaluation_results[cond_name] = {
            "parameter": param_desc,
            "y_true": y_true,
            "y_pred": y_pred,
            "predictions": preds,
            "accuracy": acc,
            "correct": corr,
            "incorrect": incorr,
            "drop_vs_clean_pp": drop_pp,
        }

        summary_records.append({
            "condition": cond_name,
            "parameter": param_desc,
            "accuracy": round(acc, 4),
            "correct": corr,
            "incorrect": incorr,
            "drop_vs_clean_pp": round(drop_pp, 2),
        })

    # Verify weights unchanged across all forward passes
    weight_changes = sum(
        1 for w_init, w_now in zip(initial_weights, model.weights)
        if not np.array_equal(w_init, w_now.numpy())
    )
    if weight_changes != 0:
        raise RuntimeError("FAIL: Model weights were modified during robustness evaluation!")

    print("\n[INFO] Model weights verified: 100% identical (NO weights modified).")

    # 4. Save outputs/robustness/robustness_summary.csv
    summary_df = pd.DataFrame(summary_records)
    summary_csv_path = ROBUSTNESS_DIR / "robustness_summary.csv"
    summary_df.to_csv(summary_csv_path, index=False)
    print(f"[INFO] Saved robustness summary CSV to: {summary_csv_path}")

    # Print summary table
    print("\n" + "=" * 80)
    print("ROBUSTNESS EVALUATION SUMMARY TABLE")
    print("=" * 80)
    print(f"{'Condition':<15} | {'Parameter':<22} | {'Accuracy':>10} | {'Correct':>8} | {'Incorrect':>10} | {'Drop vs Clean':>14}")
    print("-" * 88)
    for _, row in summary_df.iterrows():
        print(f"{row['condition']:<15} | {row['parameter']:<22} | {row['accuracy']*100:>9.2f}% | {row['correct']:>8d} | {row['incorrect']:>10d} | {row['drop_vs_clean_pp']:>11.2f} pp")

    # 5. Generate Plots
    accuracy_png_path = ROBUSTNESS_DIR / "robustness_accuracy.png"
    plot_robustness_accuracy(summary_df, clean_acc * 100, accuracy_png_path)

    degradation_png_path = ROBUSTNESS_DIR / "accuracy_drop.png"
    plot_accuracy_drop(summary_df, degradation_png_path)

    # 6. Per-class degradation analysis for Strong Blur and Strong Dark
    print("\n[INFO] Computing per-class recall degradation for Strong Blur and Strong Dark...")
    y_true_clean = evaluation_results["Clean"]["y_true"]
    y_pred_clean = evaluation_results["Clean"]["y_pred"]
    y_pred_blur_strong = evaluation_results["Blur strong"]["y_pred"]
    y_pred_dark_strong = evaluation_results["Dark strong"]["y_pred"]

    per_class_records: List[Dict[str, Any]] = []

    for target_cond, target_preds in [
        ("strong_blur", y_pred_blur_strong),
        ("strong_dark", y_pred_dark_strong),
    ]:
        cond_class_records = []
        for class_idx, breed_name in enumerate(class_names):
            class_mask = (y_true_clean == class_idx)
            support = int(np.sum(class_mask))
            if support == 0:
                continue

            clean_corr = int(np.sum(class_mask & (y_pred_clean == class_idx)))
            corr_corr = int(np.sum(class_mask & (target_preds == class_idx)))

            clean_recall = clean_corr / support
            corrupted_recall = corr_corr / support
            recall_drop = clean_recall - corrupted_recall

            cond_class_records.append({
                "condition": target_cond,
                "breed": breed_name,
                "clean_recall": round(clean_recall, 4),
                "corrupted_recall": round(corrupted_recall, 4),
                "recall_drop": round(recall_drop, 4),
                "support": support,
            })

        cond_df = pd.DataFrame(cond_class_records)
        cond_df_sorted = cond_df.sort_values(
            by=["recall_drop", "support"],
            ascending=[False, False],
        ).head(10)
        per_class_records.extend(cond_df_sorted.to_dict(orient="records"))

    per_class_df = pd.DataFrame(per_class_records)
    per_class_csv_path = ROBUSTNESS_DIR / "per_class_degradation.csv"
    per_class_df.to_csv(per_class_csv_path, index=False)
    print(f"[INFO] Saved per-class degradation CSV to: {per_class_csv_path}")

    # 7. Identify Failure Examples
    print("\n[INFO] Identifying failure examples (Clean correct -> Corrupted wrong)...")
    blur_failure_indices: List[int] = []
    dark_failure_indices: List[int] = []

    for i in range(EXPECTED_TEST_SAMPLES):
        clean_correct = (y_pred_clean[i] == y_true_clean[i])
        blur_wrong = (y_pred_blur_strong[i] != y_true_clean[i])
        dark_wrong = (y_pred_dark_strong[i] != y_true_clean[i])

        if clean_correct and blur_wrong and len(blur_failure_indices) < 10:
            blur_failure_indices.append(i)

        if clean_correct and dark_wrong and len(dark_failure_indices) < 10:
            dark_failure_indices.append(i)

    combined_indices = sorted(list(set(blur_failure_indices + dark_failure_indices)))
    failure_records: List[Dict[str, Any]] = []

    for idx in combined_indices:
        img_id = str(test_df.iloc[idx]["id"])
        img_name = str(test_df.iloc[idx]["img_name"])
        img_rel_path = f"data/train/{img_name}"
        true_breed = class_names[y_true_clean[idx]]
        clean_p = class_names[y_pred_clean[idx]]
        blur_p = class_names[y_pred_blur_strong[idx]]
        dark_p = class_names[y_pred_dark_strong[idx]]

        failure_records.append({
            "image_id": img_id,
            "image_path": img_rel_path,
            "true_breed": true_breed,
            "clean_prediction": clean_p,
            "strong_blur_prediction": blur_p,
            "strong_dark_prediction": dark_p,
        })

    failure_df = pd.DataFrame(failure_records)
    failure_csv_path = ROBUSTNESS_DIR / "failure_examples.csv"
    failure_df.to_csv(failure_csv_path, index=False)
    print(f"[INFO] Saved failure examples ({len(failure_df)} samples) to: {failure_csv_path}")

    # 8. Find largest degradation condition
    corrupted_summary = summary_df[summary_df["condition"] != "Clean"]
    max_drop_row = corrupted_summary.loc[corrupted_summary["drop_vs_clean_pp"].idxmax()]
    largest_condition = str(max_drop_row["condition"])
    largest_drop_pp = float(max_drop_row["drop_vs_clean_pp"])

    # 9. Create outputs/robustness/robustness_summary.txt
    summary_text = (
        "Selected model: MobileNetV2 Frozen Transfer Learning (best_mobilenet_transfer.keras)\n"
        f"Clean accuracy: {clean_acc * 100:.2f}%\n\n"
        f"Blur mild accuracy: {evaluation_results['Blur mild']['accuracy'] * 100:.2f}%\n"
        f"Blur medium accuracy: {evaluation_results['Blur medium']['accuracy'] * 100:.2f}%\n"
        f"Blur strong accuracy: {evaluation_results['Blur strong']['accuracy'] * 100:.2f}%\n\n"
        f"Dark mild accuracy: {evaluation_results['Dark mild']['accuracy'] * 100:.2f}%\n"
        f"Dark medium accuracy: {evaluation_results['Dark medium']['accuracy'] * 100:.2f}%\n"
        f"Dark strong accuracy: {evaluation_results['Dark strong']['accuracy'] * 100:.2f}%\n\n"
        f"Largest degradation condition: {largest_condition}\n"
        f"Largest accuracy drop: {largest_drop_pp:.2f} pp\n\n"
        "Training performed: NO\n"
        "Weights modified: NO\n"
        "Model selection performed: NO\n"
    )

    summary_txt_path = ROBUSTNESS_DIR / "robustness_summary.txt"
    summary_txt_path.write_text(summary_text, encoding="utf-8")
    print(f"[INFO] Saved robustness summary TXT to: {summary_txt_path}")

    print("\n" + "=" * 80)
    print("ROBUSTNESS ANALYSIS COMPLETE")
    print("=" * 80)

    return {
        "status": "PASS",
        "clean_accuracy": clean_acc,
        "summary_df": summary_df,
        "evaluation_results": evaluation_results,
        "largest_condition": largest_condition,
        "largest_drop_pp": largest_drop_pp,
        "per_class_df": per_class_df,
        "failure_df": failure_df,
    }


if __name__ == "__main__":
    run_robustness_analysis()
