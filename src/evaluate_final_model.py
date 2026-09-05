"""Evaluation module for Final Selected Model on Internal Held-out Test Set.

This module executes Task 5.1:
- Loads the final selected model: MobileNetV2 Frozen Transfer Learning checkpoint
  (outputs/mobilenet_transfer/checkpoints/best_mobilenet_transfer.keras).
- Evaluates strictly on the internal held-out test split (1,023 labeled samples).
- Absolutely NO training or weight updates are performed.
- Kaggle competition test (data/test/) remains completely untouched.
- Generates comprehensive evaluation metrics:
    * Test loss & Test accuracy
    * Manual accuracy verification
    * Classification report (TXT and CSV)
    * Raw and normalized Confusion Matrix (CSV and PNG plots)
    * Per-class summary with Top 10 and Bottom 10 breeds
    * Top confused breed pairs
    * Overall macro and weighted metrics
    * Validation-test generalization gap analysis
    * evaluation_summary.txt
"""

from pathlib import Path
import math
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf

from src.dataset_pipeline import build_datasets, load_split_csv
from src.label_encoder import get_class_mappings, NUM_CLASSES
from src.data_loader import PROJECT_ROOT

# Configuration
BATCH_SIZE: int = 64
EXPECTED_TEST_SAMPLES: int = 1023
EXPECTED_CLASSES: int = 120
BEST_VAL_ACCURACY: float = 0.8239  # 82.39% from validation-based model selection

# Checkpoint path for final selected model
CHECKPOINT_PATH: Path = (
    PROJECT_ROOT
    / "outputs"
    / "mobilenet_transfer"
    / "checkpoints"
    / "best_mobilenet_transfer.keras"
)

# Output directory for evaluation artifacts
EVAL_DIR: Path = PROJECT_ROOT / "outputs" / "evaluation"

# Register preprocess_input function with Keras custom objects
tf.keras.utils.get_custom_objects()["preprocess_input"] = (
    tf.keras.applications.mobilenet_v2.preprocess_input
)


def ensure_eval_dir() -> Path:
    """Create outputs/evaluation directory if it does not exist.

    Returns
    -------
    Path
        Path to the evaluation directory.
    """
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    return EVAL_DIR


def verify_model_architecture(model: tf.keras.Model) -> Dict[str, Any]:
    """Verify loaded model structure, parameters, and frozen backbone.

    Parameters
    ----------
    model : tf.keras.Model
        Loaded Keras model.

    Returns
    -------
    dict
        Dictionary containing parameter counts and verification status.
    """
    input_shape = model.input_shape
    output_shape = model.output_shape

    # Find MobileNetV2 backbone
    backbone: Optional[tf.keras.Model] = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model) or "mobilenetv2" in layer.name.lower():
            backbone = layer
            break

    if backbone is None:
        raise RuntimeError("FAIL: MobileNetV2 backbone could not be found in model layers.")

    backbone_trainable = backbone.trainable
    backbone_trainable_vars = len(backbone.trainable_variables)

    total_params = model.count_params()
    trainable_params = sum(np.prod(v.shape) for v in model.trainable_variables)
    non_trainable_params = sum(np.prod(v.shape) for v in model.non_trainable_variables)

    # Base model parameter verification
    base_params = backbone.count_params()

    info = {
        "model_name": model.name,
        "input_shape": input_shape,
        "output_shape": output_shape,
        "total_params": int(total_params),
        "trainable_params": int(trainable_params),
        "non_trainable_params": int(non_trainable_params),
        "backbone_name": backbone.name,
        "backbone_trainable": backbone_trainable,
        "backbone_trainable_vars": backbone_trainable_vars,
        "backbone_params": int(base_params),
    }
    return info


def extract_ground_truth_and_predictions(
    model: tf.keras.Model,
    test_ds: tf.data.Dataset,
    encoder: Any,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract ground truth labels and compute model predictions on test dataset.

    Parameters
    ----------
    model : tf.keras.Model
        Loaded evaluation model.
    test_ds : tf.data.Dataset
        Batched test dataset yielding (images, one_hot_labels).
    encoder : LabelBinarizer
        Fitted label encoder.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        (y_true, y_pred, raw_predictions)
    """
    # 1. Collect all true one-hot labels from test_ds
    print("\n[INFO] Extracting ground truth labels from test_ds...")
    y_true_list: List[np.ndarray] = []
    for _, labels_batch in test_ds:
        y_true_list.append(labels_batch.numpy())

    y_true_onehot = np.concatenate(y_true_list, axis=0)
    y_true = np.argmax(y_true_onehot, axis=1)

    # Cross-verify with test.csv metadata
    test_df = load_split_csv("test")
    if len(test_df) != EXPECTED_TEST_SAMPLES:
        raise ValueError(
            f"Expected {EXPECTED_TEST_SAMPLES} samples in test.csv, got {len(test_df)}"
        )

    if len(y_true) != EXPECTED_TEST_SAMPLES:
        raise ValueError(
            f"Expected {EXPECTED_TEST_SAMPLES} ground truth labels, got {len(y_true)}"
        )

    # 2. Compute model predictions
    print("[INFO] Computing model predictions on test_ds...")
    predictions = model.predict(test_ds, verbose=1)

    if predictions.shape != (EXPECTED_TEST_SAMPLES, EXPECTED_CLASSES):
        raise ValueError(
            f"Unexpected predictions shape: {predictions.shape}. "
            f"Expected ({EXPECTED_TEST_SAMPLES}, {EXPECTED_CLASSES})"
        )

    # Validate prediction integrity
    nan_count = int(np.isnan(predictions).sum())
    inf_count = int(np.isinf(predictions).sum())
    if nan_count > 0 or inf_count > 0:
        raise ValueError(
            f"Predictions contain invalid values: NaN count={nan_count}, Inf count={inf_count}"
        )

    prob_sums = np.sum(predictions, axis=1)
    if not np.allclose(prob_sums, 1.0, atol=1e-4):
        max_dev = float(np.max(np.abs(prob_sums - 1.0)))
        print(f"[WARNING] Softmax probabilities sum deviation: max deviation={max_dev:.6f}")

    y_pred = np.argmax(predictions, axis=1)

    return y_true, y_pred, predictions


def plot_and_save_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    output_path: Path,
    normalized: bool = False,
) -> None:
    """Plot high-resolution confusion matrix using matplotlib and save to disk.

    Parameters
    ----------
    cm : np.ndarray
        Confusion matrix (shape [120, 120]).
    class_names : List[str]
        List of 120 breed names in exact class index order.
    output_path : Path
        Target image file path.
    normalized : bool, default=False
        Whether cm is normalized [0, 1].
    """
    fig, ax = plt.subplots(figsize=(30, 30))
    cmap = plt.cm.Blues

    if normalized:
        im = ax.imshow(cm, interpolation="nearest", cmap=cmap, vmin=0.0, vmax=1.0)
        title = "Normalized Confusion Matrix (Row-Normalized) - MobileNetV2 Frozen (Test Set)"
    else:
        im = ax.imshow(cm, interpolation="nearest", cmap=cmap)
        title = "Confusion Matrix - MobileNetV2 Frozen Transfer Learning (Test Set)"

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        title=title,
        ylabel="True Breed Label",
        xlabel="Predicted Breed Label",
    )

    plt.setp(ax.get_xticklabels(), rotation=90, ha="right", rotation_mode="anchor", fontsize=6)
    plt.setp(ax.get_yticklabels(), fontsize=6)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Saved confusion matrix plot to: {output_path}")


def evaluate_final_model() -> Dict[str, Any]:
    """Execute complete Task 5.1 Final Evaluation pipeline.

    Returns
    -------
    dict
        Comprehensive evaluation metrics dictionary.
    """
    print("=" * 80)
    print("TASK 5.1: FINAL EVALUATION ON INTERNAL HELD-OUT TEST SET")
    print("=" * 80)
    print(f"Selected Checkpoint: {CHECKPOINT_PATH}")
    print(f"Internal Test Samples: {EXPECTED_TEST_SAMPLES}")
    print(f"Classes: {EXPECTED_CLASSES}\n")

    ensure_eval_dir()

    # 1. Verify Checkpoint Exists
    if not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(f"Selected checkpoint not found at: {CHECKPOINT_PATH}")

    # 2. Build Datasets (batch_size=64)
    print("[INFO] Building tf.data pipelines (train_ds, val_ds, test_ds)...")
    train_ds, val_ds, test_ds, encoder = build_datasets(batch_size=BATCH_SIZE)

    # Enforce deterministic order for evaluation
    options = tf.data.Options()
    options.deterministic = True
    test_ds = test_ds.with_options(options)

    class_names = [str(c) for c in encoder.classes_]
    if len(class_names) != EXPECTED_CLASSES:
        raise ValueError(
            f"Expected {EXPECTED_CLASSES} classes in encoder, got {len(class_names)}"
        )

    # 3. Load Model Checkpoint
    print(f"[INFO] Loading model checkpoint from: {CHECKPOINT_PATH}")
    model = tf.keras.models.load_model(str(CHECKPOINT_PATH))
    print("[INFO] Model loaded successfully.")

    # 4. Verify Model Architecture & Frozen Backbone
    arch_info = verify_model_architecture(model)
    print("\n--- MODEL VERIFICATION ---")
    print(f"Model Name:             {arch_info['model_name']}")
    print(f"Input Shape:            {arch_info['input_shape']}")
    print(f"Output Shape:           {arch_info['output_shape']}")
    print(f"Total Params:           {arch_info['total_params']:,}")
    print(f"Trainable Params:       {arch_info['trainable_params']:,}")
    print(f"Non-trainable Params:   {arch_info['non_trainable_params']:,}")
    print(f"Backbone Layer Name:    {arch_info['backbone_name']}")
    print(f"Backbone Trainable:     {arch_info['backbone_trainable']}")
    print(f"Backbone Trainable Vars:{arch_info['backbone_trainable_vars']}")

    if arch_info["backbone_trainable"] or arch_info["backbone_trainable_vars"] != 0:
        raise RuntimeError(
            "FAIL: Backbone is not frozen! Backbone trainable variables must be 0."
        )

    # 5. Final Test Evaluation (model.evaluate)
    print("\n" + "=" * 50)
    print("[INFO] Executing model.evaluate(test_ds)...")
    start_eval_time = time.time()
    eval_results = model.evaluate(test_ds, verbose=1)
    eval_duration = time.time() - start_eval_time

    test_loss = float(eval_results[0])
    test_accuracy = float(eval_results[1])
    print(f"[RESULT] Test Loss:     {test_loss:.4f}")
    print(f"[RESULT] Test Accuracy: {test_accuracy:.4f} ({test_accuracy * 100:.2f}%)")
    print(f"[INFO] Evaluation completed in {eval_duration:.2f} seconds.")

    # 6. Model Predictions & Ground Truth Extraction
    start_pred_time = time.time()
    y_true, y_pred, predictions = extract_ground_truth_and_predictions(
        model, test_ds, encoder
    )
    pred_duration = time.time() - start_pred_time
    print(f"[INFO] Predictions completed in {pred_duration:.2f} seconds.")

    # 7. Accuracy Consistency Verification
    manual_accuracy = float(np.mean(y_true == y_pred))
    acc_diff = abs(manual_accuracy - test_accuracy)
    correct_count = int(np.sum(y_true == y_pred))
    incorrect_count = int(np.sum(y_true != y_pred))

    print("\n--- ACCURACY CONSISTENCY CHECK ---")
    print(f"model.evaluate Test Accuracy: {test_accuracy:.6f}")
    print(f"Manual Computed Accuracy:     {manual_accuracy:.6f}")
    print(f"Absolute Difference:          {acc_diff:.6f}")
    print(f"Correct Predictions:          {correct_count} / {EXPECTED_TEST_SAMPLES}")
    print(f"Incorrect Predictions:        {incorrect_count} / {EXPECTED_TEST_SAMPLES}")
    print(f"Total Verified:               {correct_count + incorrect_count} (Expected: {EXPECTED_TEST_SAMPLES})")

    if acc_diff > 1e-3:
        raise RuntimeError(
            f"FAIL: Discrepancy between model.evaluate ({test_accuracy:.4f}) and "
            f"manual accuracy ({manual_accuracy:.4f}) exceeds threshold: {acc_diff:.6f}"
        )
    if correct_count + incorrect_count != EXPECTED_TEST_SAMPLES:
        raise RuntimeError(
            f"FAIL: Total count mismatch: {correct_count + incorrect_count} != {EXPECTED_TEST_SAMPLES}"
        )

    # 8. Classification Report Generation
    print("\n[INFO] Generating classification report...")
    report_txt = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        digits=4,
        zero_division=0,
    )
    report_dict = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    # Save classification report text
    report_txt_path = EVAL_DIR / "classification_report.txt"
    report_txt_path.write_text(report_txt, encoding="utf-8")
    print(f"[INFO] Saved classification report text to: {report_txt_path}")

    # Save classification report CSV
    report_df = pd.DataFrame(report_dict).transpose()
    report_df.index.name = "class_or_metric"
    report_csv_path = EVAL_DIR / "classification_report.csv"
    report_df.to_csv(report_csv_path)
    print(f"[INFO] Saved classification report CSV to: {report_csv_path}")

    # Extract macro and weighted metrics
    macro_precision = float(report_dict["macro avg"]["precision"])
    macro_recall = float(report_dict["macro avg"]["recall"])
    macro_f1 = float(report_dict["macro avg"]["f1-score"])

    weighted_precision = float(report_dict["weighted avg"]["precision"])
    weighted_recall = float(report_dict["weighted avg"]["recall"])
    weighted_f1 = float(report_dict["weighted avg"]["f1-score"])

    print("\n--- OVERALL CLASSIFICATION METRICS ---")
    print(f"Macro Precision:    {macro_precision:.4f} ({macro_precision * 100:.2f}%)")
    print(f"Macro Recall:       {macro_recall:.4f} ({macro_recall * 100:.2f}%)")
    print(f"Macro F1-Score:     {macro_f1:.4f} ({macro_f1 * 100:.2f}%)")
    print(f"Weighted Precision: {weighted_precision:.4f} ({weighted_precision * 100:.2f}%)")
    print(f"Weighted Recall:    {weighted_recall:.4f} ({weighted_recall * 100:.2f}%)")
    print(f"Weighted F1-Score:  {weighted_f1:.4f} ({weighted_f1 * 100:.2f}%)")

    # 9. Confusion Matrix Computation
    print("\n[INFO] Computing confusion matrices...")
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(EXPECTED_CLASSES))
    if cm.shape != (EXPECTED_CLASSES, EXPECTED_CLASSES):
        raise ValueError(
            f"Expected confusion matrix shape ({EXPECTED_CLASSES}, {EXPECTED_CLASSES}), got {cm.shape}"
        )
    if cm.sum() != EXPECTED_TEST_SAMPLES:
        raise ValueError(
            f"Confusion matrix sum ({cm.sum()}) does not match test sample count ({EXPECTED_TEST_SAMPLES})"
        )

    # Save raw confusion matrix CSV
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    cm_csv_path = EVAL_DIR / "confusion_matrix.csv"
    cm_df.to_csv(cm_csv_path)
    print(f"[INFO] Saved raw confusion matrix CSV to: {cm_csv_path}")

    # Plot raw confusion matrix
    cm_png_path = EVAL_DIR / "confusion_matrix.png"
    plot_and_save_confusion_matrix(cm, class_names, cm_png_path, normalized=False)

    # Compute row-normalized confusion matrix
    row_sums = cm.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        cm_normalized = np.divide(
            cm.astype(float),
            row_sums,
            out=np.zeros_like(cm, dtype=float),
            where=row_sums != 0,
        )

    # Save normalized confusion matrix CSV
    cm_norm_df = pd.DataFrame(cm_normalized, index=class_names, columns=class_names)
    cm_norm_csv_path = EVAL_DIR / "confusion_matrix_normalized.csv"
    cm_norm_df.to_csv(cm_norm_csv_path)
    print(f"[INFO] Saved normalized confusion matrix CSV to: {cm_norm_csv_path}")

    # Plot normalized confusion matrix
    cm_norm_png_path = EVAL_DIR / "confusion_matrix_normalized.png"
    plot_and_save_confusion_matrix(cm_normalized, class_names, cm_norm_png_path, normalized=True)

    # 10. Per-Class Analysis
    print("\n[INFO] Compiling per-class summary...")
    per_class_records: List[Dict[str, Any]] = []
    for idx, breed in enumerate(class_names):
        b_metrics = report_dict[breed]
        prec = float(b_metrics["precision"])
        rec = float(b_metrics["recall"])
        f1 = float(b_metrics["f1-score"])
        sup = int(b_metrics["support"])
        corr = int(cm[idx, idx])

        per_class_records.append({
            "breed": breed,
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "support": sup,
            "correct_predictions": corr,
        })

    per_class_df = pd.DataFrame(per_class_records)
    # Sort descending by F1 score, then recall, then support
    per_class_df_sorted = per_class_df.sort_values(
        by=["f1_score", "recall", "precision", "support"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)

    per_class_csv_path = EVAL_DIR / "per_class_summary.csv"
    per_class_df_sorted.to_csv(per_class_csv_path, index=False)
    print(f"[INFO] Saved per-class summary to: {per_class_csv_path}")

    # Top 10 and Bottom 10 Breeds by F1
    top_10_f1 = per_class_df_sorted.head(10)
    bottom_10_f1 = per_class_df_sorted.tail(10).iloc[::-1].reset_index(drop=True)

    print("\n--- TOP 10 BREEDS BY F1-SCORE ---")
    for rank, row in top_10_f1.iterrows():
        print(f"  {rank+1:2d}. {row['breed']:30s} | F1: {row['f1_score']:.4f} | Prec: {row['precision']:.4f} | Rec: {row['recall']:.4f} | Supp: {row['support']:2d} | Corr: {row['correct_predictions']:2d}")

    print("\n--- BOTTOM 10 BREEDS BY F1-SCORE ---")
    for rank, row in bottom_10_f1.iterrows():
        print(f"  {rank+1:2d}. {row['breed']:30s} | F1: {row['f1_score']:.4f} | Prec: {row['precision']:.4f} | Rec: {row['recall']:.4f} | Supp: {row['support']:2d} | Corr: {row['correct_predictions']:2d}")

    # 11. Most Confused Breed Pairs
    print("\n[INFO] Identifying most confused breed pairs...")
    cm_offdiag = cm.copy()
    np.fill_diagonal(cm_offdiag, 0)

    confused_indices = np.argwhere(cm_offdiag > 0)
    confused_records: List[Dict[str, Any]] = []
    for r, c in confused_indices:
        confused_records.append({
            "true_breed": class_names[r],
            "predicted_breed": class_names[c],
            "count": int(cm_offdiag[r, c]),
        })

    confused_df = pd.DataFrame(confused_records)
    if not confused_df.empty:
        confused_df = confused_df.sort_values(
            by=["count", "true_breed", "predicted_breed"],
            ascending=[False, True, True],
        ).reset_index(drop=True)

    top_confusions_path = EVAL_DIR / "top_confusions.csv"
    confused_df.to_csv(top_confusions_path, index=False)
    print(f"[INFO] Saved top confusions to: {top_confusions_path}")

    top_10_confusions = confused_df.head(10)
    print("\n--- TOP 10 CONFUSED BREED PAIRS ---")
    for rank, row in top_10_confusions.iterrows():
        print(f"  {rank+1:2d}. True: {row['true_breed']:25s} -> Pred: {row['predicted_breed']:25s} | Count: {row['count']}")

    # 12. Validation vs Test Comparison
    val_test_gap = (BEST_VAL_ACCURACY - test_accuracy) * 100
    print("\n--- VALIDATION VS TEST ACCURACY ---")
    print(f"Best Validation Accuracy: {BEST_VAL_ACCURACY * 100:.2f}%")
    print(f"Test Accuracy:            {test_accuracy * 100:.2f}%")
    print(f"Validation-Test Gap:      {val_test_gap:+.2f} percentage points")

    # 13. Write outputs/evaluation/evaluation_summary.txt
    summary_text = (
        "Final Selected Model: MobileNetV2 Frozen Transfer Learning\n"
        f"Checkpoint: outputs/mobilenet_transfer/checkpoints/best_mobilenet_transfer.keras\n\n"
        "Model selected using validation: YES\n"
        f"Best validation accuracy: {BEST_VAL_ACCURACY * 100:.2f}%\n\n"
        f"Internal test samples: {EXPECTED_TEST_SAMPLES}\n"
        "Internal test evaluated: YES\n"
        "Kaggle competition test used: NO\n\n"
        f"Test loss: {test_loss:.4f}\n"
        f"Test accuracy: {test_accuracy * 100:.2f}%\n\n"
        f"Correct: {correct_count}\n"
        f"Incorrect: {incorrect_count}\n\n"
        f"Macro precision: {macro_precision:.4f}\n"
        f"Macro recall: {macro_recall:.4f}\n"
        f"Macro F1: {macro_f1:.4f}\n\n"
        f"Weighted precision: {weighted_precision:.4f}\n"
        f"Weighted recall: {weighted_recall:.4f}\n"
        f"Weighted F1: {weighted_f1:.4f}\n\n"
        f"Validation-test accuracy gap: {val_test_gap:+.2f} percentage points\n\n"
        "Training performed during evaluation: NO\n"
        "Weights modified: NO\n"
    )

    summary_path = EVAL_DIR / "evaluation_summary.txt"
    summary_path.write_text(summary_text, encoding="utf-8")
    print(f"\n[INFO] Saved evaluation summary to: {summary_path}")

    # Collect complete evaluation results
    results = {
        "model_name": arch_info["model_name"],
        "checkpoint": str(CHECKPOINT_PATH),
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
        "manual_accuracy": manual_accuracy,
        "acc_diff": acc_diff,
        "correct_count": correct_count,
        "incorrect_count": incorrect_count,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "weighted_precision": weighted_precision,
        "weighted_recall": weighted_recall,
        "weighted_f1": weighted_f1,
        "val_test_gap": val_test_gap,
        "top_10_f1": top_10_f1,
        "bottom_10_f1": bottom_10_f1,
        "top_10_confusions": top_10_confusions,
        "arch_info": arch_info,
    }
    return results


if __name__ == "__main__":
    evaluate_final_model()
