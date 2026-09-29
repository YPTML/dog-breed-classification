"""One-Time Locked Final Evaluation for Outlier Exposure V2.

Artifacts Evaluated:
- Champion Model: outputs/models/oe_v2/stage_a_best.keras (SHA256: e4508564a9389250e265cae6ea6fed0e1a0edf8497d62cd9c8a44342bdd45e61)
- Frozen Centroids: outputs/calibration/oe_v2/class_centroids.npy (SHA256: d4588ee2cff5d8cbe3b670ca99507ae97f7f07b75822d1e247200e7c481e85cb)
- Locked Thresholds: T_UNCERTAIN = 0.343358, T_NOT_DOG = 0.457513

Datasets:
1. ID Final Test: outputs/splits_v2/test.csv (1,017 samples)
2. Easy OOD Final Test: data/ood_oe_v2/test (619 samples)
3. Hard OOD Final Test: data/ood_oe_v2/hard_test (69 samples: 35 Snoopy, 34 Garfield)
"""

from __future__ import annotations

import csv
import glob
import hashlib
import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Enable line-buffered stdout
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
# 1. Project Root & Locked Constants
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

CHAMPION_MODEL_PATH = PROJECT_ROOT / "outputs/models/oe_v2/stage_a_best.keras"
EXPECTED_CHAMPION_SHA256 = "e4508564a9389250e265cae6ea6fed0e1a0edf8497d62cd9c8a44342bdd45e61"

CENTROIDS_PATH = PROJECT_ROOT / "outputs/calibration/oe_v2/class_centroids.npy"
EXPECTED_CENTROIDS_SHA256 = "d4588ee2cff5d8cbe3b670ca99507ae97f7f07b75822d1e247200e7c481e85cb"
EXPECTED_CENTROIDS_SHAPE = (120, 1280)

CALIBRATION_JSON_PATH = PROJECT_ROOT / "outputs/calibration/oe_v2/ood_calibration.json"
CLASS_NAMES_PATH = PROJECT_ROOT / "outputs/deployment_ready/class_names.json"

ID_TEST_CSV_PATH = PROJECT_ROOT / "outputs/splits_v2/test.csv"
DATA_TRAIN_DIR = PROJECT_ROOT / "data/train"

OOD_TEST_DIR = PROJECT_ROOT / "data/ood_oe_v2/test"
OOD_HARD_TEST_DIR = PROJECT_ROOT / "data/ood_oe_v2/hard_test"
OOD_MANIFEST_PATH = PROJECT_ROOT / "metadata/ood_oe_v2_manifest.csv"

OUTPUT_FINAL_DIR = PROJECT_ROOT / "outputs/final/oe_v2"
OUTPUT_PLOTS_DIR = PROJECT_ROOT / "outputs/plots/oe_v2"
OUTPUT_REPORTS_DIR = PROJECT_ROOT / "outputs/reports"

# LOCKED OPERATING THRESHOLDS
LOCKED_DETECTOR = "COSINE"
LOCKED_T_UNCERTAIN = 0.343358
LOCKED_T_NOT_DOG = 0.457513


def get_file_sha256(path: Path) -> str:
    """Compute SHA256 hex digest of a file in chunks."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# 2. Pure NumPy Classification & OOD Metrics
# ---------------------------------------------------------------------------
import numpy as np


def compute_classification_metrics(y_true: np.ndarray, y_pred_prob: np.ndarray) -> Dict[str, float]:
    """Compute Top-1, Top-5, Macro Precision, Macro Recall, Macro F1, and Loss."""
    n_samples, n_classes = y_pred_prob.shape
    pred_idx = np.argmax(y_pred_prob, axis=1)

    # Top-1 Accuracy
    top1_correct = (pred_idx == y_true)
    top1_acc = float(np.mean(top1_correct))

    # Top-5 Accuracy
    top5_correct = []
    for i in range(n_samples):
        top5_cands = np.argsort(-y_pred_prob[i])[:5]
        top5_correct.append(y_true[i] in top5_cands)
    top5_correct = np.array(top5_correct, dtype=bool)
    top5_acc = float(np.mean(top5_correct))

    # Cross-entropy loss
    eps = 1e-15
    sample_losses = -np.log(y_pred_prob[np.arange(n_samples), y_true] + eps)
    ce_loss = float(np.mean(sample_losses))

    # Per-class TP, FP, FN for Macro Precision, Recall, F1
    precisions = []
    recalls = []
    f1s = []

    for c in range(n_classes):
        tp = np.sum((pred_idx == c) & (y_true == c))
        fp = np.sum((pred_idx == c) & (y_true != c))
        fn = np.sum((pred_idx != c) & (y_true == c))

        prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)

    macro_precision = float(np.mean(precisions))
    macro_recall = float(np.mean(recalls))
    macro_f1 = float(np.mean(f1s))

    return {
        "top1_accuracy": top1_acc,
        "top5_accuracy": top5_acc,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "cross_entropy_loss": ce_loss,
        "top1_correct": top1_correct,
        "top5_correct": top5_correct,
    }


def compute_auroc(id_scores: np.ndarray, ood_scores: np.ndarray) -> float:
    """Exact Mann-Whitney U AUROC (higher score = OOD)."""
    n_id = len(id_scores)
    n_ood = len(ood_scores)
    if n_id == 0 or n_ood == 0:
        return 0.0

    scores = np.concatenate([id_scores, ood_scores])
    labels = np.concatenate([np.zeros(n_id, dtype=int), np.ones(n_ood, dtype=int)])

    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)

    unique_scores, inverse, counts = np.unique(scores, return_inverse=True, return_counts=True)
    if len(unique_scores) < len(scores):
        tie_ranks = np.zeros(len(unique_scores), dtype=float)
        np.add.at(tie_ranks, inverse, ranks)
        tie_ranks /= counts
        ranks = tie_ranks[inverse]

    u_ood = np.sum(ranks[labels == 1]) - (n_ood * (n_ood + 1)) / 2.0
    return float(u_ood / (n_id * n_ood))


def compute_aupr_ood(id_scores: np.ndarray, ood_scores: np.ndarray) -> float:
    """AUPR with OOD as positive (higher score = more OOD)."""
    n_id = len(id_scores)
    n_ood = len(ood_scores)
    if n_ood == 0 or n_id == 0:
        return 0.0

    scores = np.concatenate([id_scores, ood_scores])
    labels = np.concatenate([np.zeros(n_id, dtype=int), np.ones(n_ood, dtype=int)])

    order = np.argsort(-scores)
    sorted_labels = labels[order]

    tp = np.cumsum(sorted_labels == 1)
    fp = np.cumsum(sorted_labels == 0)

    precision = tp / (tp + fp)
    recall = tp / n_ood

    rec_diff = np.diff(np.concatenate(([0.0], recall)))
    return float(np.sum(precision * rec_diff))


def compute_aupr_id(id_scores: np.ndarray, ood_scores: np.ndarray) -> float:
    """AUPR with ID as positive (lower score = more ID)."""
    n_id = len(id_scores)
    n_ood = len(ood_scores)
    if n_id == 0 or n_ood == 0:
        return 0.0

    inv_id = -id_scores
    inv_ood = -ood_scores

    scores = np.concatenate([inv_id, inv_ood])
    labels = np.concatenate([np.ones(n_id, dtype=int), np.zeros(n_ood, dtype=int)])

    order = np.argsort(-scores)
    sorted_labels = labels[order]

    tp = np.cumsum(sorted_labels == 1)
    fp = np.cumsum(sorted_labels == 0)

    precision = tp / (tp + fp)
    recall = tp / n_id

    rec_diff = np.diff(np.concatenate(([0.0], recall)))
    return float(np.sum(precision * rec_diff))


def compute_fpr95(id_scores: np.ndarray, ood_scores: np.ndarray) -> Tuple[float, float]:
    """Compute FPR at 95% ID TPR on test data."""
    tau = float(np.percentile(id_scores, 95.0))
    fpr95 = float(np.mean(ood_scores < tau))
    return fpr95, tau


def get_percentile_stats(scores: np.ndarray) -> Dict[str, float]:
    """Detailed percentiles and summary stats."""
    p_keys = [0, 5, 10, 25, 50, 75, 90, 95, 100]
    p_vals = np.percentile(scores, p_keys)
    return {
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores)),
        "median": float(p_vals[4]),
        "min": float(p_vals[0]),
        "p05": float(p_vals[1]),
        "p10": float(p_vals[2]),
        "p25": float(p_vals[3]),
        "p50": float(p_vals[4]),
        "p75": float(p_vals[5]),
        "p90": float(p_vals[6]),
        "p95": float(p_vals[7]),
        "max": float(p_vals[8]),
    }


# ---------------------------------------------------------------------------
# 3. Model Loading & Feature Extraction
# ---------------------------------------------------------------------------
def setup_tf_and_model(model_path: Path):
    """Load model, configure memory growth, extract Dense weights and build feature extractor."""
    import tensorflow as tf

    # Enable GPU memory growth
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
            print(f"[GPU] Memory growth enabled for {gpu}")
        except Exception as exc:
            print(f"[GPU] Note: {exc}")

    tf.keras.utils.get_custom_objects()["preprocess_input"] = (
        tf.keras.applications.mobilenet_v2.preprocess_input
    )
    custom_objects = {"preprocess_input": tf.keras.applications.mobilenet_v2.preprocess_input}

    model = tf.keras.models.load_model(
        str(model_path), custom_objects=custom_objects, compile=False, safe_mode=False
    )

    dense_layer = model.get_layer("predictions")
    dense_W = dense_layer.weights[0].numpy().astype(np.float32)  # (1280, 120)
    dense_b = dense_layer.weights[1].numpy().astype(np.float32)  # (120,)

    feat_tensor = dense_layer.input
    feat_model = tf.keras.Model(inputs=model.inputs, outputs=feat_tensor)
    return model, feat_model, dense_W, dense_b


def extract_features_and_logits(
    feat_model,
    dense_W: np.ndarray,
    dense_b: np.ndarray,
    centroids: np.ndarray,
    image_paths: List[Path],
    batch_size: int = 16,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract features, logits, and cosine OOD scores in batches."""
    import tensorflow as tf

    str_paths = [str(p) for p in image_paths]

    def decode_and_resize(path: tf.Tensor):
        img_bytes = tf.io.read_file(path)
        img = tf.image.decode_image(img_bytes, channels=3, expand_animations=False)
        img = tf.image.resize(img, (224, 224))
        img = tf.cast(img, tf.float32)
        img.set_shape((224, 224, 3))
        return img

    dataset = (
        tf.data.Dataset.from_tensor_slices(str_paths)
        .map(decode_and_resize, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size, drop_remainder=False)
        .prefetch(tf.data.AUTOTUNE)
    )

    all_features = []
    total_batches = (len(str_paths) + batch_size - 1) // batch_size
    print(f"  Streaming {len(str_paths)} images across {total_batches} batches...")
    for b_idx, imgs in enumerate(dataset, start=1):
        feats = feat_model(imgs, training=False).numpy()
        all_features.append(feats)
        if b_idx % 25 == 0 or b_idx == total_batches:
            print(f"    Batch {b_idx}/{total_batches} processed", flush=True)

    features = np.concatenate(all_features, axis=0).astype(np.float32)

    # 1. Logits and Softmax
    logits = features @ dense_W + dense_b
    max_logit = np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(logits - max_logit)
    probabilities = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

    # 2. Normalized features and Cosine OOD score
    feat_norms = np.linalg.norm(features, axis=1, keepdims=True)
    norm_feats = np.divide(features, feat_norms, out=np.zeros_like(features), where=feat_norms > 0)
    sims = norm_feats @ centroids.T  # (N, 120)
    max_sim = np.max(sims, axis=1)
    cosine_scores = (1.0 - max_sim).astype(np.float32)

    return probabilities, cosine_scores, features


# ---------------------------------------------------------------------------
# 4. Tri-State Decision Logic
# ---------------------------------------------------------------------------
def apply_tristate_decision(score: float, t_unc: float = LOCKED_T_UNCERTAIN, t_nd: float = LOCKED_T_NOT_DOG) -> str:
    """Return DOG, UNCERTAIN, or NOT_DOG based on locked thresholds."""
    if score < t_unc:
        return "DOG"
    elif score < t_nd:
        return "UNCERTAIN"
    else:
        return "NOT_DOG"


# ---------------------------------------------------------------------------
# 5. Diagnostic Plots
# ---------------------------------------------------------------------------
def generate_final_plots(
    id_scores: np.ndarray,
    easy_ood_scores: np.ndarray,
    hard_ood_scores: np.ndarray,
    confusion_summary: Dict[str, Any],
    output_dir: Path,
):
    """Generate final distribution and rate plots."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)

    # Plot 1: Cosine Score Distributions (ID vs Easy OOD vs Hard OOD)
    fig, ax = plt.subplots(figsize=(10, 6))
    bins = np.linspace(0.1, 0.8, 60)

    ax.hist(id_scores, bins=bins, alpha=0.55, color="#1f77b4", label=f"ID Test Dog (n={len(id_scores)})", density=True)
    ax.hist(easy_ood_scores, bins=bins, alpha=0.55, color="#d62728", label=f"Easy OOD Test (n={len(easy_ood_scores)})", density=True)
    ax.hist(hard_ood_scores, bins=bins, alpha=0.65, color="#9467bd", label=f"Hard OOD Test (n={len(hard_ood_scores)})", density=True)

    # Locked Thresholds
    ax.axvline(LOCKED_T_UNCERTAIN, color="#2ca02c", linestyle="--", linewidth=2.2, label=f"T_uncertain ({LOCKED_T_UNCERTAIN:.6f})")
    ax.axvline(LOCKED_T_NOT_DOG, color="#d62728", linestyle="-.", linewidth=2.2, label=f"T_not_dog ({LOCKED_T_NOT_DOG:.6f})")

    ax.set_title("Final Locked Evaluation: Cosine OOD Score Distributions", fontsize=13, fontweight="bold")
    ax.set_xlabel("Cosine Distance Score (1 - max sim)", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper right", fontsize=10)

    plt.tight_layout()
    p1 = output_dir / "final_cosine_distributions.png"
    plt.savefig(p1, dpi=300)
    plt.close()
    print(f"[PLOT] Saved: {p1}")

    # Plot 2: Tri-State Decision Rates Bar Chart
    fig, ax = plt.subplots(figsize=(9, 6))
    groups = ["ID Final Test", "Easy OOD Test", "Hard OOD Test"]
    dog_rates = [
        confusion_summary["id_test"]["dog_rate"] * 100,
        confusion_summary["easy_ood"]["dog_rate"] * 100,
        confusion_summary["hard_ood"]["dog_rate"] * 100,
    ]
    unc_rates = [
        confusion_summary["id_test"]["uncertain_rate"] * 100,
        confusion_summary["easy_ood"]["uncertain_rate"] * 100,
        confusion_summary["hard_ood"]["uncertain_rate"] * 100,
    ]
    not_dog_rates = [
        confusion_summary["id_test"]["not_dog_rate"] * 100,
        confusion_summary["easy_ood"]["not_dog_rate"] * 100,
        confusion_summary["hard_ood"]["not_dog_rate"] * 100,
    ]

    x = np.arange(len(groups))
    width = 0.25

    rects1 = ax.bar(x - width, dog_rates, width, label="DOG", color="#2ca02c", alpha=0.85)
    rects2 = ax.bar(x, unc_rates, width, label="UNCERTAIN", color="#ff7f0e", alpha=0.85)
    rects3 = ax.bar(x + width, not_dog_rates, width, label="NOT_DOG", color="#d62728", alpha=0.85)

    ax.set_ylabel("Rate (%)", fontsize=11)
    ax.set_title("Tri-State Decision Rates Across Final Test Splits", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(groups, fontsize=11, fontweight="bold")
    ax.set_ylim([0, 115])
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    ax.legend(loc="upper right", fontsize=10)

    for rects in [rects1, rects2, rects3]:
        for rect in rects:
            h = rect.get_height()
            ax.annotate(f"{h:.1f}%",
                        xy=(rect.get_x() + rect.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=9, fontweight="bold")

    plt.tight_layout()
    p2 = output_dir / "final_tri_state_rates.png"
    plt.savefig(p2, dpi=300)
    plt.close()
    print(f"[PLOT] Saved: {p2}")


# ---------------------------------------------------------------------------
# 6. Main Pipeline
# ---------------------------------------------------------------------------
def run_final_evaluation():
    print("=" * 80)
    print("STARTING ONE-TIME LOCKED FINAL EVALUATION FOR OUTLIER EXPOSURE V2")
    print(f"Timestamp: {datetime.now(timezone.utc).isoformat()}")
    print("=" * 80)

    # -----------------------------------------------------------------------
    # Step 1: Pre-Flight Integrity Check
    # -----------------------------------------------------------------------
    print("\n--- STEP 1: PRE-FLIGHT INTEGRITY VERIFICATION ---")

    # Check model file & SHA256
    if not CHAMPION_MODEL_PATH.is_file():
        print(f"BLOCKED: Champion model missing at {CHAMPION_MODEL_PATH}")
        return "BLOCKED"
    model_sha = get_file_sha256(CHAMPION_MODEL_PATH)
    if model_sha != EXPECTED_CHAMPION_SHA256:
        print(f"BLOCKED: Champion model SHA256 mismatch! Expected {EXPECTED_CHAMPION_SHA256}, got {model_sha}")
        return "BLOCKED"
    print(f"[OK] Model SHA256 matches locked value: {model_sha}")

    # Check centroids file & SHA256
    if not CENTROIDS_PATH.is_file():
        print(f"BLOCKED: Centroids file missing at {CENTROIDS_PATH}")
        return "BLOCKED"
    centroids_sha = get_file_sha256(CENTROIDS_PATH)
    if centroids_sha != EXPECTED_CENTROIDS_SHA256:
        print(f"BLOCKED: Centroids SHA256 mismatch! Expected {EXPECTED_CENTROIDS_SHA256}, got {centroids_sha}")
        return "BLOCKED"
    print(f"[OK] Centroids SHA256 matches locked value: {centroids_sha}")

    centroids = np.load(CENTROIDS_PATH)
    if centroids.shape != EXPECTED_CENTROIDS_SHAPE:
        print(f"BLOCKED: Centroids shape mismatch! Expected {EXPECTED_CENTROIDS_SHAPE}, got {centroids.shape}")
        return "BLOCKED"
    print(f"[OK] Centroids shape matches locked value: {centroids.shape}")

    # Check calibration JSON & thresholds
    if not CALIBRATION_JSON_PATH.is_file():
        print(f"BLOCKED: Calibration JSON missing at {CALIBRATION_JSON_PATH}")
        return "BLOCKED"
    with open(CALIBRATION_JSON_PATH, "r", encoding="utf-8") as f:
        cal_meta = json.load(f)
    if cal_meta.get("selected_detector") != LOCKED_DETECTOR:
        print(f"BLOCKED: Detector mismatch! Expected {LOCKED_DETECTOR}, got {cal_meta.get('selected_detector')}")
        return "BLOCKED"
    if abs(cal_meta["T_uncertain"] - LOCKED_T_UNCERTAIN) > 1e-5:
        print(f"BLOCKED: T_uncertain mismatch! Expected {LOCKED_T_UNCERTAIN}, got {cal_meta['T_uncertain']}")
        return "BLOCKED"
    if abs(cal_meta["T_not_dog"] - LOCKED_T_NOT_DOG) > 1e-5:
        print(f"BLOCKED: T_not_dog mismatch! Expected {LOCKED_T_NOT_DOG}, got {cal_meta['T_not_dog']}")
        return "BLOCKED"
    print(f"[OK] Locked calibration verified: Detector={LOCKED_DETECTOR}, T_unc={LOCKED_T_UNCERTAIN}, T_nd={LOCKED_T_NOT_DOG}")

    # Check class names
    with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as f:
        class_names_dict = json.load(f)
    if len(class_names_dict) != 120:
        print(f"BLOCKED: Class count mismatch! Expected 120, got {len(class_names_dict)}")
        return "BLOCKED"
    class_order = [class_names_dict[str(i)] for i in range(120)]
    class_to_idx = {name: i for i, name in enumerate(class_order)}
    print(f"[OK] Class names verified: 120 unique classes, strictly 120 outputs")

    # Load Model and verify layer dimensions
    model, feat_model, dense_W, dense_b = setup_tf_and_model(CHAMPION_MODEL_PATH)
    if int(dense_W.shape[0]) != 1280 or int(dense_W.shape[1]) != 120:
        print(f"BLOCKED: Dense layer weights shape mismatch! Expected (1280, 120), got {dense_W.shape}")
        return "BLOCKED"
    print(f"[OK] Architecture verified: 1280 feature dimension, 120 class logits (No class 121)")

    OUTPUT_FINAL_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # Step 2: ID Final Test Evaluation (1,017 samples)
    # -----------------------------------------------------------------------
    print("\n--- STEP 2: ID FINAL TEST EVALUATION ---")
    with open(ID_TEST_CSV_PATH, "r", encoding="utf-8") as f:
        id_test_rows = list(csv.DictReader(f))
    if len(id_test_rows) != 1017:
        print(f"BLOCKED: Expected 1017 ID test rows, got {len(id_test_rows)}")
        return "BLOCKED"
    print(f"[DATA] ID Final Test sample count verified: {len(id_test_rows)}")

    id_image_paths = [DATA_TRAIN_DIR / r["img_name"] for r in id_test_rows]
    id_true_indices = np.array([class_to_idx[r["breed"]] for r in id_test_rows], dtype=int)

    id_probs, id_cosine_scores, _ = extract_features_and_logits(
        feat_model, dense_W, dense_b, centroids, id_image_paths, batch_size=16
    )

    # Classification Metrics
    id_cls_metrics = compute_classification_metrics(id_true_indices, id_probs)
    print(f"  Classification Quality (120 classes):")
    print(f"    Top-1 Accuracy:   {id_cls_metrics['top1_accuracy']*100:.2f}%")
    print(f"    Top-5 Accuracy:   {id_cls_metrics['top5_accuracy']*100:.2f}%")
    print(f"    Macro Precision:  {id_cls_metrics['macro_precision']*100:.2f}%")
    print(f"    Macro Recall:     {id_cls_metrics['macro_recall']*100:.2f}%")
    print(f"    Macro F1:         {id_cls_metrics['macro_f1']*100:.2f}%")
    print(f"    Cross-Entropy:    {id_cls_metrics['cross_entropy_loss']:.4f}")

    # Tri-State Decisions for ID Test
    id_decisions = [apply_tristate_decision(s) for s in id_cosine_scores]
    id_dog_mask = np.array([d == "DOG" for d in id_decisions], dtype=bool)
    id_unc_mask = np.array([d == "UNCERTAIN" for d in id_decisions], dtype=bool)
    id_nd_mask = np.array([d == "NOT_DOG" for d in id_decisions], dtype=bool)

    id_dog_count = int(np.sum(id_dog_mask))
    id_unc_count = int(np.sum(id_unc_mask))
    id_nd_count = int(np.sum(id_nd_mask))

    id_dog_rate = float(id_dog_count / len(id_test_rows))
    id_unc_rate = float(id_unc_count / len(id_test_rows))
    id_nd_rate = float(id_nd_count / len(id_test_rows))

    print(f"  Tri-State Gating Behavior:")
    print(f"    DOG (Accepted):    {id_dog_rate*100:.2f}% ({id_dog_count}/{len(id_test_rows)})")
    print(f"    UNCERTAIN:         {id_unc_rate*100:.2f}% ({id_unc_count}/{len(id_test_rows)})")
    print(f"    False NOT_DOG:     {id_nd_rate*100:.2f}% ({id_nd_count}/{len(id_test_rows)})")

    # Accuracy on Accepted DOG subset
    if id_dog_count > 0:
        dog_top1 = float(np.mean(id_cls_metrics["top1_correct"][id_dog_mask]))
        dog_top5 = float(np.mean(id_cls_metrics["top5_correct"][id_dog_mask]))
    else:
        dog_top1, dog_top5 = 0.0, 0.0
    print(f"  Accuracy on Accepted DOG Subset (n={id_dog_count}):")
    print(f"    Top-1 Accuracy:   {dog_top1*100:.2f}%")
    print(f"    Top-5 Accuracy:   {dog_top5*100:.2f}%")

    # Rejection of correctly classified dogs
    correct_dogs_mask = id_cls_metrics["top1_correct"]
    correct_rejected_unc = int(np.sum(correct_dogs_mask & id_unc_mask))
    correct_rejected_nd = int(np.sum(correct_dogs_mask & id_nd_mask))
    print(f"  Correctly Classified Dogs Filtered Out:")
    print(f"    Sent to UNCERTAIN: {correct_rejected_unc} samples")
    print(f"    Sent to NOT_DOG:   {correct_rejected_nd} samples")

    # Save final_id_predictions.csv
    pred_idx_id = np.argmax(id_probs, axis=1)
    conf_id = np.max(id_probs, axis=1)
    id_pred_rows = []
    for i, r in enumerate(id_test_rows):
        id_pred_rows.append({
            "id": r["id"],
            "breed": r["breed"],
            "true_index": id_true_indices[i],
            "predicted_breed": class_order[pred_idx_id[i]],
            "predicted_index": int(pred_idx_id[i]),
            "confidence": round(float(conf_id[i]), 6),
            "top1_correct": bool(id_cls_metrics["top1_correct"][i]),
            "top5_correct": bool(id_cls_metrics["top5_correct"][i]),
            "cosine_score": round(float(id_cosine_scores[i]), 6),
            "decision": id_decisions[i],
        })
    id_pred_csv_path = OUTPUT_FINAL_DIR / "final_id_predictions.csv"
    with open(id_pred_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(id_pred_rows[0].keys()))
        writer.writeheader()
        writer.writerows(id_pred_rows)
    print(f"[SAVE] Saved {id_pred_csv_path}")

    # -----------------------------------------------------------------------
    # Step 3: Easy OOD Final Test Evaluation (619 samples)
    # -----------------------------------------------------------------------
    print("\n--- STEP 3: EASY OOD FINAL TEST EVALUATION ---")
    with open(OOD_MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest_rows = list(csv.DictReader(f))
    easy_ood_rows = [r for r in manifest_rows if r["split"] == "test"]
    if len(easy_ood_rows) != 619:
        print(f"BLOCKED: Expected 619 Easy OOD test rows, got {len(easy_ood_rows)}")
        return "BLOCKED"
    print(f"[DATA] Easy OOD Final Test sample count verified: {len(easy_ood_rows)}")

    easy_ood_paths = [PROJECT_ROOT / r["relative_path"] for r in easy_ood_rows]
    easy_ood_probs, easy_ood_cosine_scores, _ = extract_features_and_logits(
        feat_model, dense_W, dense_b, centroids, easy_ood_paths, batch_size=16
    )

    easy_ood_decisions = [apply_tristate_decision(s) for s in easy_ood_cosine_scores]
    easy_ood_dog_count = int(np.sum([d == "DOG" for d in easy_ood_decisions]))
    easy_ood_unc_count = int(np.sum([d == "UNCERTAIN" for d in easy_ood_decisions]))
    easy_ood_nd_count = int(np.sum([d == "NOT_DOG" for d in easy_ood_decisions]))

    easy_ood_dog_rate = float(easy_ood_dog_count / len(easy_ood_rows))
    easy_ood_unc_rate = float(easy_ood_unc_count / len(easy_ood_rows))
    easy_ood_nd_rate = float(easy_ood_nd_count / len(easy_ood_rows))

    print(f"  Tri-State Decisions (Easy OOD):")
    print(f"    False DOG:         {easy_ood_dog_rate*100:.2f}% ({easy_ood_dog_count}/{len(easy_ood_rows)})")
    print(f"    UNCERTAIN:         {easy_ood_unc_rate*100:.2f}% ({easy_ood_unc_count}/{len(easy_ood_rows)})")
    print(f"    NOT_DOG (Rej):     {easy_ood_nd_rate*100:.2f}% ({easy_ood_nd_count}/{len(easy_ood_rows)})")

    easy_ood_stats = get_percentile_stats(easy_ood_cosine_scores)
    print(f"  Cosine Score Stats: mean={easy_ood_stats['mean']:.4f}, std={easy_ood_stats['std']:.4f}, median={easy_ood_stats['median']:.4f}, min={easy_ood_stats['min']:.4f}, max={easy_ood_stats['max']:.4f}")

    # Breakdown by super_label & fine_label
    super_breakdown = defaultdict(lambda: {"total": 0, "dog": 0, "uncertain": 0, "not_dog": 0})
    fine_breakdown = defaultdict(lambda: {"super": "", "total": 0, "dog": 0, "uncertain": 0, "not_dog": 0, "scores": []})

    pred_idx_ood = np.argmax(easy_ood_probs, axis=1)
    conf_ood = np.max(easy_ood_probs, axis=1)
    easy_ood_pred_rows = []

    for i, r in enumerate(easy_ood_rows):
        sl = r["super_label"]
        fl = r["fine_label"]
        d = easy_ood_decisions[i]
        sc = float(easy_ood_cosine_scores[i])

        super_breakdown[sl]["total"] += 1
        super_breakdown[sl][d.lower()] += 1

        fine_breakdown[fl]["super"] = sl
        fine_breakdown[fl]["total"] += 1
        fine_breakdown[fl][d.lower()] += 1
        fine_breakdown[fl]["scores"].append(sc)

        easy_ood_pred_rows.append({
            "relative_path": r["relative_path"],
            "super_label": sl,
            "fine_label": fl,
            "cosine_score": round(sc, 6),
            "decision": d,
            "top_predicted_breed": class_order[pred_idx_ood[i]],
            "top_confidence": round(float(conf_ood[i]), 6),
        })

    ood_pred_csv_path = OUTPUT_FINAL_DIR / "final_ood_predictions.csv"
    with open(ood_pred_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(easy_ood_pred_rows[0].keys()))
        writer.writeheader()
        writer.writerows(easy_ood_pred_rows)
    print(f"[SAVE] Saved {ood_pred_csv_path}")

    # Identify fine labels with highest false-DOG rate
    fine_ranking = []
    for fl, data in fine_breakdown.items():
        fdr = float(data["dog"] / data["total"])
        fine_ranking.append({
            "fine_label": fl,
            "super_label": data["super"],
            "total": data["total"],
            "dog": data["dog"],
            "uncertain": data["uncertain"],
            "not_dog": data["not_dog"],
            "false_dog_rate": round(fdr, 4),
            "mean_score": round(float(np.mean(data["scores"])), 4),
        })
    fine_ranking.sort(key=lambda x: (-x["false_dog_rate"], -x["dog"], x["mean_score"]))
    print(f"  Fine Labels with Highest False-DOG Rate (Top 5):")
    for item in fine_ranking[:5]:
        print(f"    {item['fine_label']} ({item['super_label']}): {item['false_dog_rate']*100:.1f}% false dog ({item['dog']}/{item['total']}), mean score={item['mean_score']:.4f}")

    # -----------------------------------------------------------------------
    # Step 4: Hard OOD Final Test Evaluation (69 samples)
    # -----------------------------------------------------------------------
    print("\n--- STEP 4: HARD OOD FINAL TEST EVALUATION ---")
    snoopy_dir = OOD_HARD_TEST_DIR / "cartoon_dog/snoopy"
    garfield_dir = OOD_HARD_TEST_DIR / "cartoon_animal/garfield"

    snoopy_paths = sorted(list(snoopy_dir.glob("*.jpg")))
    garfield_paths = sorted(list(garfield_dir.glob("*.jpg")))

    if len(snoopy_paths) != 35 or len(garfield_paths) != 34:
        print(f"BLOCKED: Expected 35 Snoopy and 34 Garfield files, got {len(snoopy_paths)} and {len(garfield_paths)}")
        return "BLOCKED"
    print(f"[DATA] Hard OOD counts verified: Snoopy={len(snoopy_paths)}, Garfield={len(garfield_paths)} (Total={len(snoopy_paths)+len(garfield_paths)})")

    all_hard_paths = snoopy_paths + garfield_paths
    hard_probs, hard_cosine_scores, _ = extract_features_and_logits(
        feat_model, dense_W, dense_b, centroids, all_hard_paths, batch_size=16
    )

    snoopy_scores = hard_cosine_scores[:len(snoopy_paths)]
    garfield_scores = hard_cosine_scores[len(snoopy_paths):]

    snoopy_decisions = [apply_tristate_decision(s) for s in snoopy_scores]
    garfield_decisions = [apply_tristate_decision(s) for s in garfield_scores]
    all_hard_decisions = snoopy_decisions + garfield_decisions

    # Snoopy stats
    snoopy_dog = int(np.sum([d == "DOG" for d in snoopy_decisions]))
    snoopy_unc = int(np.sum([d == "UNCERTAIN" for d in snoopy_decisions]))
    snoopy_nd = int(np.sum([d == "NOT_DOG" for d in snoopy_decisions]))

    # Garfield stats
    garfield_dog = int(np.sum([d == "DOG" for d in garfield_decisions]))
    garfield_unc = int(np.sum([d == "UNCERTAIN" for d in garfield_decisions]))
    garfield_nd = int(np.sum([d == "NOT_DOG" for d in garfield_decisions]))

    # Combined Hard stats
    hard_dog = snoopy_dog + garfield_dog
    hard_unc = snoopy_unc + garfield_unc
    hard_nd = snoopy_nd + garfield_nd
    total_hard = len(all_hard_paths)

    print(f"  Snoopy (n=35):   DOG={snoopy_dog} ({snoopy_dog/35*100:.2f}%), UNCERTAIN={snoopy_unc} ({snoopy_unc/35*100:.2f}%), NOT_DOG={snoopy_nd} ({snoopy_nd/35*100:.2f}%)")
    print(f"    Scores: mean={np.mean(snoopy_scores):.4f}, min={np.min(snoopy_scores):.4f}, max={np.max(snoopy_scores):.4f}")
    print(f"  Garfield (n=34): DOG={garfield_dog} ({garfield_dog/34*100:.2f}%), UNCERTAIN={garfield_unc} ({garfield_unc/34*100:.2f}%), NOT_DOG={garfield_nd} ({garfield_nd/34*100:.2f}%)")
    print(f"    Scores: mean={np.mean(garfield_scores):.4f}, min={np.min(garfield_scores):.4f}, max={np.max(garfield_scores):.4f}")
    print(f"  Combined Hard:   DOG={hard_dog} ({hard_dog/total_hard*100:.2f}%), UNCERTAIN={hard_unc} ({hard_unc/total_hard*100:.2f}%), NOT_DOG={hard_nd} ({hard_nd/total_hard*100:.2f}%)")

    # Save final_hard_ood_predictions.csv
    pred_idx_hard = np.argmax(hard_probs, axis=1)
    conf_hard = np.max(hard_probs, axis=1)
    hard_pred_rows = []
    for i, p in enumerate(all_hard_paths):
        is_snoopy = (i < len(snoopy_paths))
        hard_pred_rows.append({
            "relative_path": str(p.relative_to(PROJECT_ROOT)),
            "category": "cartoon_dog" if is_snoopy else "cartoon_animal",
            "character": "snoopy" if is_snoopy else "garfield",
            "cosine_score": round(float(hard_cosine_scores[i]), 6),
            "decision": all_hard_decisions[i],
            "top_predicted_breed": class_order[pred_idx_hard[i]],
            "top_confidence": round(float(conf_hard[i]), 6),
        })

    hard_pred_csv_path = OUTPUT_FINAL_DIR / "final_hard_ood_predictions.csv"
    with open(hard_pred_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(hard_pred_rows[0].keys()))
        writer.writeheader()
        writer.writerows(hard_pred_rows)
    print(f"[SAVE] Saved {hard_pred_csv_path}")

    # -----------------------------------------------------------------------
    # Step 5: Detector Final Discrimination Metrics (ID Test vs Easy OOD Test)
    # -----------------------------------------------------------------------
    print("\n--- STEP 5: DETECTOR FINAL DISCRIMINATION METRICS ---")
    test_auroc = compute_auroc(id_cosine_scores, easy_ood_cosine_scores)
    test_aupr_ood = compute_aupr_ood(id_cosine_scores, easy_ood_cosine_scores)
    test_aupr_id = compute_aupr_id(id_cosine_scores, easy_ood_cosine_scores)
    test_fpr95, test_tau95 = compute_fpr95(id_cosine_scores, easy_ood_cosine_scores)

    print(f"  Locked COSINE Detector on Final Test Sets:")
    print(f"    AUROC:        {test_auroc:.6f}")
    print(f"    AUPR-OOD:     {test_aupr_ood:.6f}")
    print(f"    AUPR-ID:      {test_aupr_id:.6f}")
    print(f"    FPR@95% TPR:  {test_fpr95*100:.2f}% (tau={test_tau95:.6f})")

    # -----------------------------------------------------------------------
    # Step 6: Tri-State Confusion Summary
    # -----------------------------------------------------------------------
    print("\n--- STEP 6: TRI-STATE CONFUSION SUMMARY ---")
    confusion_summary = {
        "id_test": {
            "actual": "DOG",
            "total": len(id_test_rows),
            "dog_count": id_dog_count,
            "dog_rate": id_dog_rate,
            "uncertain_count": id_unc_count,
            "uncertain_rate": id_unc_rate,
            "not_dog_count": id_nd_count,
            "not_dog_rate": id_nd_rate,
        },
        "easy_ood": {
            "actual": "NOT_DOG",
            "total": len(easy_ood_rows),
            "dog_count": easy_ood_dog_count,
            "dog_rate": easy_ood_dog_rate,
            "uncertain_count": easy_ood_unc_count,
            "uncertain_rate": easy_ood_unc_rate,
            "not_dog_count": easy_ood_nd_count,
            "not_dog_rate": easy_ood_nd_rate,
        },
        "hard_ood": {
            "actual": "NOT_DOG",
            "total": total_hard,
            "dog_count": hard_dog,
            "dog_rate": float(hard_dog / total_hard),
            "uncertain_count": hard_unc,
            "uncertain_rate": float(hard_unc / total_hard),
            "not_dog_count": hard_nd,
            "not_dog_rate": float(hard_nd / total_hard),
            "snoopy": {
                "total": 35,
                "dog_count": snoopy_dog,
                "dog_rate": float(snoopy_dog / 35),
                "uncertain_count": snoopy_unc,
                "uncertain_rate": float(snoopy_unc / 35),
                "not_dog_count": snoopy_nd,
                "not_dog_rate": float(snoopy_nd / 35),
            },
            "garfield": {
                "total": 34,
                "dog_count": garfield_dog,
                "dog_rate": float(garfield_dog / 34),
                "uncertain_count": garfield_unc,
                "uncertain_rate": float(garfield_unc / 34),
                "not_dog_count": garfield_nd,
                "not_dog_rate": float(garfield_nd / 34),
            },
        },
    }

    conf_json_path = OUTPUT_FINAL_DIR / "final_confusion_summary.json"
    with open(conf_json_path, "w", encoding="utf-8") as f:
        json.dump(confusion_summary, f, indent=2)
    print(f"[SAVE] Saved {conf_json_path}")

    # -----------------------------------------------------------------------
    # Step 7: Post-Hoc Regression Evaluation
    # -----------------------------------------------------------------------
    print("\n--- STEP 7: POST-HOC REGRESSION CHECK ---")
    snoopy_0001_path = snoopy_dir / "image_0001.jpg"
    snoopy_0001_idx = [i for i, p in enumerate(snoopy_paths) if p.name == "image_0001.jpg"][0]
    snoopy_0001_score = float(snoopy_scores[snoopy_0001_idx])
    snoopy_0001_decision = snoopy_decisions[snoopy_0001_idx]
    if snoopy_0001_decision == "NOT_DOG":
        reg_status = "PASS"
    elif snoopy_0001_decision == "UNCERTAIN":
        reg_status = "PARTIAL PASS"
    else:
        reg_status = "REGRESSION FAIL"

    print(f"  Regression Target: {snoopy_0001_path.name}")
    print(f"    Cosine Score: {snoopy_0001_score:.6f}")
    print(f"    Decision:     {snoopy_0001_decision}")
    print(f"    Result:       {reg_status}")

    # -----------------------------------------------------------------------
    # Step 8: Save Final Metrics JSON
    # -----------------------------------------------------------------------
    final_metrics_data = {
        "champion_model_sha256": model_sha,
        "centroid_sha256": centroids_sha,
        "selected_detector": LOCKED_DETECTOR,
        "T_uncertain": LOCKED_T_UNCERTAIN,
        "T_not_dog": LOCKED_T_NOT_DOG,
        "id_test_metrics": {
            "sample_count": len(id_test_rows),
            "top1_accuracy": id_cls_metrics["top1_accuracy"],
            "top5_accuracy": id_cls_metrics["top5_accuracy"],
            "macro_precision": id_cls_metrics["macro_precision"],
            "macro_recall": id_cls_metrics["macro_recall"],
            "macro_f1": id_cls_metrics["macro_f1"],
            "cross_entropy_loss": id_cls_metrics["cross_entropy_loss"],
            "dog_accept_rate": id_dog_rate,
            "uncertain_rate": id_unc_rate,
            "false_not_dog_rate": id_nd_rate,
            "filtered_dog_top1": dog_top1,
            "filtered_dog_top5": dog_top5,
            "correct_dogs_rejected_uncertain": correct_rejected_unc,
            "correct_dogs_rejected_not_dog": correct_rejected_nd,
        },
        "easy_ood_metrics": {
            "sample_count": len(easy_ood_rows),
            "false_dog_rate": easy_ood_dog_rate,
            "uncertain_rate": easy_ood_unc_rate,
            "not_dog_rate": easy_ood_nd_rate,
            "score_stats": easy_ood_stats,
            "auroc": test_auroc,
            "aupr_ood": test_aupr_ood,
            "aupr_id": test_aupr_id,
            "fpr95": test_fpr95,
        },
        "hard_ood_metrics": {
            "snoopy": {
                "count": 35,
                "dog_rate": float(snoopy_dog / 35),
                "uncertain_rate": float(snoopy_unc / 35),
                "not_dog_rate": float(snoopy_nd / 35),
                "mean_score": float(np.mean(snoopy_scores)),
                "min_score": float(np.min(snoopy_scores)),
                "max_score": float(np.max(snoopy_scores)),
            },
            "garfield": {
                "count": 34,
                "dog_rate": float(garfield_dog / 34),
                "uncertain_rate": float(garfield_unc / 34),
                "not_dog_rate": float(garfield_nd / 34),
                "mean_score": float(np.mean(garfield_scores)),
                "min_score": float(np.min(garfield_scores)),
                "max_score": float(np.max(garfield_scores)),
            },
            "combined": {
                "count": total_hard,
                "dog_rate": float(hard_dog / total_hard),
                "uncertain_rate": float(hard_unc / total_hard),
                "not_dog_rate": float(hard_nd / total_hard),
            },
        },
        "post_hoc_regression": {
            "image": "cartoon_dog/snoopy/image_0001.jpg",
            "score": snoopy_0001_score,
            "decision": snoopy_0001_decision,
            "status": reg_status,
        },
        "integrity": {
            "model_modified": False,
            "centroids_modified": False,
            "thresholds_modified": False,
            "final_test_used_for_tuning": False,
            "production_baseline_modified": False,
        },
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
    }

    metrics_json_path = OUTPUT_FINAL_DIR / "final_metrics.json"
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(final_metrics_data, f, indent=2)
    print(f"[SAVE] Saved {metrics_json_path}")

    # -----------------------------------------------------------------------
    # Step 9: Diagnostic Plots
    # -----------------------------------------------------------------------
    print("\n--- STEP 9: GENERATING FINAL EVALUATION PLOTS ---")
    generate_final_plots(
        id_scores=id_cosine_scores,
        easy_ood_scores=easy_ood_cosine_scores,
        hard_ood_scores=hard_cosine_scores,
        confusion_summary=confusion_summary,
        output_dir=OUTPUT_PLOTS_DIR,
    )

    # -----------------------------------------------------------------------
    # Step 10: Final Evaluation Report
    # -----------------------------------------------------------------------
    print("\n--- STEP 10: WRITING FINAL EVALUATION REPORT ---")
    report_content = f"""# Outlier Exposure (OE) V2 Final Evaluation Report

**Evaluation Date:** {datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")}  
**Status:** LOCKED FINAL EVALUATION COMPLETED  
**Execution Type:** One-time evaluation on uncalibrated, sealed final test sets.  

---

## 1. Locked System Configuration & Artifact Hashes

- **MODEL SHA256:** `{model_sha}`
- **CENTROID SHA256:** `{centroids_sha}`
- **DETECTOR:** `{LOCKED_DETECTOR}`
- **T_UNCERTAIN:** `{LOCKED_T_UNCERTAIN:.6f}`
- **T_NOT_DOG:** `{LOCKED_T_NOT_DOG:.6f}`

---

## 2. ID Final Test Evaluation (`outputs/splits_v2/test.csv`)

### Classification Quality (120 Dog Breeds):
- **COUNT:** {len(id_test_rows)} samples
- **TOP1 ACCURACY:** {id_cls_metrics['top1_accuracy']*100:.2f}%
- **TOP5 ACCURACY:** {id_cls_metrics['top5_accuracy']*100:.2f}%
- **MACRO PRECISION:** {id_cls_metrics['macro_precision']*100:.2f}%
- **MACRO RECALL:** {id_cls_metrics['macro_recall']*100:.2f}%
- **MACRO F1:** {id_cls_metrics['macro_f1']*100:.2f}%
- **LOSS (Cross-Entropy):** {id_cls_metrics['cross_entropy_loss']:.4f}

### Tri-State Gating Behavior:
- **DOG ACCEPT RATE:** {id_dog_rate*100:.2f}% ({id_dog_count} / {len(id_test_rows)})
- **UNCERTAIN RATE:** {id_unc_rate*100:.2f}% ({id_unc_count} / {len(id_test_rows)})
- **FALSE NOT_DOG RATE:** {id_nd_rate*100:.2f}% ({id_nd_count} / {len(id_test_rows)})

### Filtered Accuracy on Accepted Dogs ($S < T_{{\\text{{uncertain}}}}$):
- **ACCEPTED DOG COUNT:** {id_dog_count} samples
- **ACCEPTED TOP1 ACCURACY:** {dog_top1*100:.2f}%
- **ACCEPTED TOP5 ACCURACY:** {dog_top5*100:.2f}%
- **Correct Dogs Sent to UNCERTAIN:** {correct_rejected_unc} samples
- **Correct Dogs Sent to NOT_DOG:** {correct_rejected_nd} samples

---

## 3. Easy OOD Final Test Evaluation (`data/ood_oe_v2/test`)

- **COUNT:** {len(easy_ood_rows)} samples
- **FALSE DOG RATE:** {easy_ood_dog_rate*100:.2f}% ({easy_ood_dog_count} / {len(easy_ood_rows)})
- **UNCERTAIN RATE:** {easy_ood_unc_rate*100:.2f}% ({easy_ood_unc_count} / {len(easy_ood_rows)})
- **NOT_DOG RATE (Rejection):** {easy_ood_nd_rate*100:.2f}% ({easy_ood_nd_count} / {len(easy_ood_rows)})

### Locked Detector Performance (ID Test vs Easy OOD Test):
- **AUROC:** {test_auroc:.6f}
- **AUPR-OOD:** {test_aupr_ood:.6f}
- **AUPR-ID:** {test_aupr_id:.6f}
- **FPR@95% ID TPR:** {test_fpr95*100:.2f}% (computed at ID Test 95th percentile $\\tau = {test_tau95:.6f}$)

### Score Distribution Moments (Cosine OOD Score):
- **Mean:** {easy_ood_stats['mean']:.4f}
- **Std:** {easy_ood_stats['std']:.4f}
- **Median:** {easy_ood_stats['median']:.4f}
- **Min:** {easy_ood_stats['min']:.4f}
- **p05:** {easy_ood_stats['p05']:.4f}
- **p25:** {easy_ood_stats['p25']:.4f}
- **p50:** {easy_ood_stats['p50']:.4f}
- **p75:** {easy_ood_stats['p75']:.4f}
- **p95:** {easy_ood_stats['p95']:.4f}
- **Max:** {easy_ood_stats['max']:.4f}

### Super-Label Breakdown:
| Super Label | Total | DOG (False Dog) | UNCERTAIN | NOT_DOG (Rejection) |
| :--- | :---: | :---: | :---: | :---: |
"""
    for sl, d in sorted(super_breakdown.items()):
        report_content += f"| **{sl}** | {d['total']} | {d['dog']} ({d['dog']/d['total']*100:.1f}%) | {d['uncertain']} ({d['uncertain']/d['total']*100:.1f}%) | {d['not_dog']} ({d['not_dog']/d['total']*100:.1f}%) |\n"

    report_content += f"""
### Fine-Labels with Highest False-DOG Rates:
| Fine Label | Super Label | Total | DOG (False Dog) | False-DOG Rate | Mean Score |
| :--- | :--- | :---: | :---: | :---: | :---: |
"""
    for item in fine_ranking[:8]:
        report_content += f"| **{item['fine_label']}** | {item['super_label']} | {item['total']} | {item['dog']} | {item['false_dog_rate']*100:.1f}% | {item['mean_score']:.4f} |\n"

    report_content += f"""
---

## 4. Hard OOD Final Test Evaluation (`data/ood_oe_v2/hard_test`)

Generalization stress test on out-of-distribution cartoon characters:

### Snoopy (`cartoon_dog/snoopy`):
- **SAMPLE COUNT:** 35
- **DOG COUNT / RATE:** {snoopy_dog} ({snoopy_dog/35*100:.2f}%)
- **UNCERTAIN COUNT / RATE:** {snoopy_unc} ({snoopy_unc/35*100:.2f}%)
- **NOT_DOG COUNT / RATE:** {snoopy_nd} ({snoopy_nd/35*100:.2f}%)
- **Cosine Score Mean:** {np.mean(snoopy_scores):.4f} (Min: {np.min(snoopy_scores):.4f}, Max: {np.max(snoopy_scores):.4f})

### Garfield (`cartoon_animal/garfield`):
- **SAMPLE COUNT:** 34
- **DOG COUNT / RATE:** {garfield_dog} ({garfield_dog/34*100:.2f}%)
- **UNCERTAIN COUNT / RATE:** {garfield_unc} ({garfield_unc/34*100:.2f}%)
- **NOT_DOG COUNT / RATE:** {garfield_nd} ({garfield_nd/34*100:.2f}%)
- **Cosine Score Mean:** {np.mean(garfield_scores):.4f} (Min: {np.min(garfield_scores):.4f}, Max: {np.max(garfield_scores):.4f})

### Combined Hard OOD Summary:
- **COMBINED SAMPLE COUNT:** {total_hard}
- **DOG RATE:** {hard_dog/total_hard*100:.2f}% ({hard_dog} / {total_hard})
- **UNCERTAIN RATE:** {hard_unc/total_hard*100:.2f}% ({hard_unc} / {total_hard})
- **NOT_DOG RATE:** {hard_nd/total_hard*100:.2f}% ({hard_nd} / {total_hard})

---

## 5. Tri-State Confusion Matrix Summary

| Split | Actual Class | Predicted DOG | Predicted UNCERTAIN | Predicted NOT_DOG | Total |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **ID FINAL TEST** | DOG | **{id_dog_count}** ({id_dog_rate*100:.2f}%) | {id_unc_count} ({id_unc_rate*100:.2f}%) | {id_nd_count} ({id_nd_rate*100:.2f}%) | {len(id_test_rows)} |
| **EASY OOD TEST** | NOT_DOG | {easy_ood_dog_count} ({easy_ood_dog_rate*100:.2f}%) | {easy_ood_unc_count} ({easy_ood_unc_rate*100:.2f}%) | **{easy_ood_nd_count}** ({easy_ood_nd_rate*100:.2f}%) | {len(easy_ood_rows)} |
| **HARD OOD TEST** | NOT_DOG | {hard_dog} ({hard_dog/total_hard*100:.2f}%) | {hard_unc} ({hard_unc/total_hard*100:.2f}%) | **{hard_nd}** ({hard_nd/total_hard*100:.2f}%) | {total_hard} |

---

## 6. Post-Hoc Regression Example

Evaluation on classic cartoon benchmark image: `cartoon_dog/snoopy/image_0001.jpg`:
- **Cosine Score:** `{snoopy_0001_score:.6f}`
- **Tri-State Decision:** `{snoopy_0001_decision}`
- **Regression Result:** **`{reg_status}`**

---

## 7. System Integrity & Governance Checklist

- **MODEL MODIFIED:** NO
- **CENTROIDS MODIFIED:** NO
- **THRESHOLDS MODIFIED:** NO
- **FINAL TEST USED FOR TUNING:** NO
- **PRODUCTION BASELINE MODIFIED:** NO

---

## 8. Artifacts Generated

1. `outputs/final/oe_v2/final_id_predictions.csv`
2. `outputs/final/oe_v2/final_ood_predictions.csv`
3. `outputs/final/oe_v2/final_hard_ood_predictions.csv`
4. `outputs/final/oe_v2/final_metrics.json`
5. `outputs/final/oe_v2/final_confusion_summary.json`
6. `outputs/plots/oe_v2/final_cosine_distributions.png`
7. `outputs/plots/oe_v2/final_tri_state_rates.png`
8. `outputs/reports/OE_V2_FINAL_EVALUATION_REPORT.md`

**STATUS:** `OE_V2_FINAL_EVALUATION_COMPLETE`
"""

    report_path = OUTPUT_REPORTS_DIR / "OE_V2_FINAL_EVALUATION_REPORT.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)
    print(f"[SAVE] Saved {report_path}")

    print("\n" + "=" * 80)
    print("FINAL EVALUATION COMPLETED SUCCESSFULLY")
    print("OE_V2_FINAL_EVALUATION_COMPLETE")
    print("=" * 80)
    return "OE_V2_FINAL_EVALUATION_COMPLETE"


if __name__ == "__main__":
    res = run_final_evaluation()
    if res != "OE_V2_FINAL_EVALUATION_COMPLETE":
        sys.exit(1)
