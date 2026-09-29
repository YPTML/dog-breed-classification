"""Outlier Exposure V2 Predictor.

Deployment inference module for the frozen OE V2 release candidate:
- Model: outputs/deployment_oe_v2/dog_breed_model_oe_v2.keras (or stage_a_best.keras)
- Centroids: outputs/deployment_oe_v2/class_centroids.npy (120, 1280)
- Detector: COSINE on L2-normalized penultimate Dense embedding
- Locked Thresholds: T_UNCERTAIN = 0.343358, T_NOT_DOG = 0.457513
- Single 120-class output architecture (no 121st class, no binary head)
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

import numpy as np
import tensorflow as tf

from src.inference.preprocessing import preprocess_image, MAX_IMAGE_PIXELS

logger = logging.getLogger("oe_v2_predictor")

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# LOCKED ARTIFACT PATHS (Default to staging bundle, fallback to training/calibration outputs)
STAGING_DIR = PROJECT_ROOT / "outputs" / "deployment_oe_v2"
DEFAULT_MODEL_PATH = (
    STAGING_DIR / "dog_breed_model_oe_v2.keras"
    if (STAGING_DIR / "dog_breed_model_oe_v2.keras").exists()
    else PROJECT_ROOT / "outputs" / "models" / "oe_v2" / "stage_a_best.keras"
)
DEFAULT_CENTROIDS_PATH = (
    STAGING_DIR / "class_centroids.npy"
    if (STAGING_DIR / "class_centroids.npy").exists()
    else PROJECT_ROOT / "outputs" / "calibration" / "oe_v2" / "class_centroids.npy"
)
DEFAULT_CLASS_NAMES_PATH = (
    STAGING_DIR / "class_names.json"
    if (STAGING_DIR / "class_names.json").exists()
    else PROJECT_ROOT / "outputs" / "deployment_ready" / "class_names.json"
)

# EXPECTED INTEGRITY CONSTANTS
EXPECTED_MODEL_SHA256 = "e4508564a9389250e265cae6ea6fed0e1a0edf8497d62cd9c8a44342bdd45e61"
EXPECTED_CENTROIDS_SHA256 = "d4588ee2cff5d8cbe3b670ca99507ae97f7f07b75822d1e247200e7c481e85cb"
EXPECTED_NUM_CLASSES = 120
EXPECTED_EMBEDDING_DIM = 1280
EXPECTED_CENTROID_SHAPE = (120, 1280)
EXPECTED_INPUT_SHAPE = (None, 224, 224, 3)

# LOCKED OPERATING THRESHOLDS
LOCKED_DETECTOR = "cosine"
LOCKED_T_UNCERTAIN = 0.343358
LOCKED_T_NOT_DOG = 0.457513

NOT_DOG_MESSAGE = "No dog was confidently detected."
UNCERTAIN_MESSAGE = "The image is ambiguous or outside the reliable dog-breed region."


def compute_file_sha256(path: Path) -> str:
    """Compute SHA256 hex digest of a file in 64 KiB chunks."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def decide_tri_state(
    score: float,
    t_uncertain: float = LOCKED_T_UNCERTAIN,
    t_not_dog: float = LOCKED_T_NOT_DOG,
) -> str:
    """Apply the frozen tri-state decision boundaries.

    score < 0.343358 => DOG
    0.343358 <= score < 0.457513 => UNCERTAIN
    score >= 0.457513 => NOT_DOG
    """
    if score < t_uncertain:
        return "DOG"
    elif score < t_not_dog:
        return "UNCERTAIN"
    else:
        return "NOT_DOG"


class OEV2Predictor:
    """Production predictor for the Outlier Exposure V2 pipeline."""

    def __init__(
        self,
        model_path: Path = DEFAULT_MODEL_PATH,
        centroids_path: Path = DEFAULT_CENTROIDS_PATH,
        class_names_path: Path = DEFAULT_CLASS_NAMES_PATH,
        verify_integrity: bool = True,
    ):
        self.model_path = Path(model_path)
        self.centroids_path = Path(centroids_path)
        self.class_names_path = Path(class_names_path)
        self.model_version = "oe_v2"
        self.detector = LOCKED_DETECTOR
        self.t_uncertain = LOCKED_T_UNCERTAIN
        self.t_not_dog = LOCKED_T_NOT_DOG

        if self.t_uncertain >= self.t_not_dog:
            raise RuntimeError(
                f"Integrity error: T_UNCERTAIN ({self.t_uncertain}) must be < T_NOT_DOG ({self.t_not_dog})"
            )

        # 1. Startup Integrity Check: Files Existence and Hashes
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        if not self.centroids_path.exists():
            raise FileNotFoundError(f"Centroids file not found: {self.centroids_path}")
        if not self.class_names_path.exists():
            raise FileNotFoundError(f"Class names file not found: {self.class_names_path}")

        if verify_integrity:
            actual_model_sha = compute_file_sha256(self.model_path)
            if actual_model_sha != EXPECTED_MODEL_SHA256:
                raise RuntimeError(
                    f"Model SHA256 integrity mismatch: expected {EXPECTED_MODEL_SHA256}, got {actual_model_sha}"
                )

            actual_centroids_sha = compute_file_sha256(self.centroids_path)
            if actual_centroids_sha != EXPECTED_CENTROIDS_SHA256:
                raise RuntimeError(
                    f"Centroids SHA256 integrity mismatch: expected {EXPECTED_CENTROIDS_SHA256}, got {actual_centroids_sha}"
                )

        # 2. Load Class Names
        with open(self.class_names_path, "r", encoding="utf-8") as f:
            raw_classes = json.load(f)
        if isinstance(raw_classes, dict):
            self.class_names = {int(k): str(v) for k, v in raw_classes.items()}
        elif isinstance(raw_classes, list):
            self.class_names = {i: str(v) for i, v in enumerate(raw_classes)}
        else:
            raise RuntimeError("Invalid class names format: must be dict or list")

        if len(self.class_names) != EXPECTED_NUM_CLASSES:
            raise RuntimeError(
                f"Class names count mismatch: expected {EXPECTED_NUM_CLASSES}, got {len(self.class_names)}"
            )

        # 3. Load Centroids
        self.centroids = np.load(self.centroids_path).astype(np.float32)
        if self.centroids.shape != EXPECTED_CENTROID_SHAPE:
            raise RuntimeError(
                f"Centroids shape mismatch: expected {EXPECTED_CENTROID_SHAPE}, got {self.centroids.shape}"
            )
        # Ensure centroids are L2-normalized
        c_norms = np.linalg.norm(self.centroids, axis=-1, keepdims=True)
        if np.any(np.abs(c_norms - 1.0) > 1e-4):
            logger.info("Normalizing class centroids to unit sphere.")
            self.centroids = self.centroids / np.maximum(c_norms, 1e-12)

        # 4. Load Model
        custom_objects = {
            "preprocess_input": tf.keras.applications.mobilenet_v2.preprocess_input
        }
        self.raw_model = tf.keras.models.load_model(
            str(self.model_path),
            custom_objects=custom_objects,
            compile=False,
            safe_mode=False,
        )

        # Model validation
        if self.raw_model.input_shape != EXPECTED_INPUT_SHAPE:
            raise RuntimeError(
                f"Model input shape mismatch: expected {EXPECTED_INPUT_SHAPE}, got {self.raw_model.input_shape}"
            )
        if self.raw_model.output_shape != (None, EXPECTED_NUM_CLASSES):
            raise RuntimeError(
                f"Model output shape mismatch: expected (None, {EXPECTED_NUM_CLASSES}), got {self.raw_model.output_shape}"
            )

        # 5. Build Unified Inference Model (One pass for probabilities and penultimate embeddings)
        dense_layer = self.raw_model.get_layer("predictions")
        feat_tensor = dense_layer.input
        if feat_tensor.shape[-1] != EXPECTED_EMBEDDING_DIM:
            raise RuntimeError(
                f"Embedding dimension mismatch: expected {EXPECTED_EMBEDDING_DIM}, got {feat_tensor.shape[-1]}"
            )

        self.inference_model = tf.keras.Model(
            inputs=self.raw_model.inputs,
            outputs=[self.raw_model.output, feat_tensor],
            name="oe_v2_unified_inference",
        )

        logger.info(
            f"OEV2Predictor initialized successfully. Model: {self.model_path.name}, "
            f"Classes: {len(self.class_names)}, Centroids: {self.centroids.shape}"
        )

    def preprocess(self, image_input: Union[bytes, bytearray, memoryview, np.ndarray]) -> np.ndarray:
        """Preprocess image input to batch float32 array (1, 224, 224, 3) in [0, 255]."""
        if isinstance(image_input, (bytes, bytearray, memoryview)):
            image_array = preprocess_image(image_input, max_image_pixels=MAX_IMAGE_PIXELS)
        elif isinstance(image_input, np.ndarray):
            image_array = image_input
        else:
            raise ValueError(f"Unsupported image input type: {type(image_input)}")

        if (
            image_array.ndim != 4
            or image_array.shape != (1, 224, 224, 3)
            or image_array.dtype != np.float32
        ):
            raise ValueError(
                f"Expected float32 image batch with shape (1, 224, 224, 3), got shape {image_array.shape} and dtype {image_array.dtype}"
            )
        if not np.isfinite(image_array).all() or np.any(image_array < 0) or np.any(image_array > 255):
            raise ValueError("Image pixels must be finite values in [0, 255].")

        return image_array

    def predict(
        self,
        image_input: Union[bytes, bytearray, memoryview, np.ndarray],
    ) -> Dict[str, Any]:
        """Execute single-pass inference and tri-state OOD decision.

        Returns standard response contract.
        """
        image_batch = self.preprocess(image_input)

        # Single forward pass through unified model
        probs_tensor, feats_tensor = self.inference_model([image_batch], training=False)
        probs = np.asarray(probs_tensor)[0]  # (120,)
        feats = np.asarray(feats_tensor)[0]  # (1280,)

        # Validate finiteness
        if not np.isfinite(probs).all() or not np.isfinite(feats).all():
            raise RuntimeError("Model generated non-finite probabilities or features.")

        # Compute Cosine OOD Score: S_cosine = 1.0 - max_k(norm(feat) . centroid_k)
        feat_norm = np.linalg.norm(feats)
        if feat_norm > 0:
            norm_feat = feats / feat_norm
        else:
            norm_feat = np.zeros_like(feats)

        cos_sims = norm_feat @ self.centroids.T  # (120,)
        max_sim = float(np.max(cos_sims))
        ood_score = float(1.0 - max_sim)

        # Tri-state decision
        status = decide_tri_state(ood_score, self.t_uncertain, self.t_not_dog)

        # Compute top predictions and probability margin across all states
        top5_indices = np.argsort(-probs, kind="stable")[:5]
        top1_idx = int(top5_indices[0])
        top2_idx = int(top5_indices[1])
        top1_breed = self.class_names[top1_idx]
        top1_conf = float(probs[top1_idx])
        top2_conf = float(probs[top2_idx])
        prob_margin = float(top1_conf - top2_conf)

        top5 = [
            {
                "rank": rank + 1,
                "index": int(idx),
                "breed": self.class_names[int(idx)],
                "confidence": float(probs[int(idx)]),
            }
            for rank, idx in enumerate(top5_indices)
        ]

        response = {
            "status": status,
            "final_status": status,
            "decision": status,
            "breed": top1_breed,
            "top1_breed": top1_breed,
            "confidence": top1_conf,
            "top1_confidence": top1_conf,
            "top5": top5,
            "top5_predictions": top5,
            "probability_margin": prob_margin,
            "margin": prob_margin,
            "ood_score": ood_score,
            "raw_ood_score": ood_score,
            "T_uncertain": self.t_uncertain,
            "t_uncertain": self.t_uncertain,
            "T_not_dog": self.t_not_dog,
            "t_not_dog": self.t_not_dog,
            "is_dog": True if status == "DOG" else (None if status == "UNCERTAIN" else False),
            "prediction": {
                "index": top1_idx,
                "breed": top1_breed,
                "confidence": top1_conf,
            },
        }

        if status == "UNCERTAIN":
            response["message"] = UNCERTAIN_MESSAGE
        elif status == "NOT_DOG":
            response["message"] = NOT_DOG_MESSAGE

        return response
