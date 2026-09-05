"""Deployment Artifact Finalization Module for Dog Breed Identification.

Task 6.3: Package the final selected production model (MobileNetV2 Transfer Learning),
export class mappings, generate model manifest and README documentation, and verify
integrity, parameter counts, and synthetic inference prior to Flask deployment.

Strict Constraints:
- Production model: outputs/mobilenet_transfer/checkpoints/best_mobilenet_transfer.keras
- Pruned model (outputs/compression_manual/pruned_50.keras) is retained as an experiment
  and NOT selected for production due to accuracy drop (1.96 pp) vs negligible compression (2.39%).
- Binary copy via shutil.copy2 (no re-serialization).
- SHA-256 verification between source and deployment copy.
- No training (no fit, train_on_batch, GradientTape, optimizer).
- No dataset evaluation (no build_datasets, evaluate, or predict on test_ds).
- Synthetic inference only on random uniform tensor [0, 255].
"""

import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any, Dict, Tuple

import numpy as np
import tensorflow as tf

from src.label_encoder import NUM_CLASSES, fit_label_encoder

# Project paths
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
SOURCE_MODEL_PATH: Path = (
    PROJECT_ROOT
    / "outputs"
    / "mobilenet_transfer"
    / "checkpoints"
    / "best_mobilenet_transfer.keras"
)
OUTPUT_DIR: Path = PROJECT_ROOT / "outputs" / "deployment_ready"
DEPLOY_MODEL_PATH: Path = OUTPUT_DIR / "dog_breed_model.keras"
CLASS_NAMES_PATH: Path = OUTPUT_DIR / "class_names.json"
MANIFEST_PATH: Path = OUTPUT_DIR / "model_manifest.json"
README_PATH: Path = OUTPUT_DIR / "README.txt"

# Model specifications
IMAGE_HEIGHT: int = 224
IMAGE_WIDTH: int = 224
IMAGE_CHANNELS: int = 3
INPUT_SHAPE: Tuple[int, int, int] = (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)
EXPECTED_TOTAL_PARAMS: int = 2_411_704
EXPECTED_TRAINABLE_PARAMS: int = 153_720
EXPECTED_NON_TRAINABLE_PARAMS: int = 2_257_984
VALIDATION_ACCURACY: float = 0.8239
INTERNAL_TEST_ACCURACY: float = 0.8201
PRUNED_VALIDATION_ACCURACY: float = 0.8043
PRUNED_ZIP_REDUCTION: float = 2.39
VALIDATION_DROP_PP: float = 1.96


def compute_sha256(file_path: Path) -> str:
    """Compute SHA-256 checksum of a file in binary chunks."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            sha256.update(chunk)
    return sha256.hexdigest()


def copy_and_verify_model(
    source: Path, destination: Path
) -> Dict[str, Any]:
    """Perform binary copy of production model and verify SHA-256 integrity."""
    if not source.is_file():
        raise FileNotFoundError(f"Source model not found at: {source}")

    destination.parent.mkdir(parents=True, exist_ok=True)

    # Perform binary copy preserving metadata
    shutil.copy2(source, destination)

    source_size = source.stat().st_size
    dest_size = destination.stat().st_size
    source_sha256 = compute_sha256(source)
    dest_sha256 = compute_sha256(destination)

    identical = (source_size == dest_size) and (source_sha256 == dest_sha256)
    if not identical:
        raise ValueError(
            f"Model copy verification failed! Source SHA256 ({source_sha256}) "
            f"!= Destination SHA256 ({dest_sha256})"
        )

    return {
        "source_path": str(source),
        "dest_path": str(destination),
        "source_size": source_size,
        "dest_size": dest_size,
        "source_sha256": source_sha256,
        "dest_sha256": dest_sha256,
        "identical": identical,
    }


def export_and_verify_class_names(output_path: Path) -> Dict[str, str]:
    """Extract class mapping from project label encoder, validate, and save as JSON."""
    # Reuse existing project label encoder without iterating dataset images
    encoder = fit_label_encoder()
    classes = encoder.classes_

    if len(classes) != NUM_CLASSES:
        raise ValueError(
            f"Class count mismatch: expected {NUM_CLASSES}, got {len(classes)}"
        )

    # Build index -> breed dictionary with string keys for clean JSON serialization
    class_mapping = {str(idx): str(cls_name) for idx, cls_name in enumerate(classes)}

    # Validation
    indices = [int(k) for k in class_mapping.keys()]
    breeds = list(class_mapping.values())

    is_count_valid = len(class_mapping) == NUM_CLASSES
    is_indices_unique = len(set(indices)) == NUM_CLASSES
    is_breeds_unique = len(set(breeds)) == NUM_CLASSES
    first_index = min(indices)
    last_index = max(indices)
    is_range_valid = (first_index == 0) and (last_index == NUM_CLASSES - 1)

    if not (is_count_valid and is_indices_unique and is_breeds_unique and is_range_valid):
        raise ValueError("Class mapping validation failed!")

    # Write class_names.json
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(class_mapping, f, indent=2, ensure_ascii=False)

    return class_mapping


def load_and_verify_model(
    model_path: Path, class_mapping: Dict[str, str]
) -> Dict[str, Any]:
    """Load model artifact, verify parameter counts, and execute synthetic inference."""
    # Register MobileNetV2 preprocess_input custom object for Keras deserialization
    tf.keras.utils.get_custom_objects()["preprocess_input"] = (
        tf.keras.applications.mobilenet_v2.preprocess_input
    )

    model = tf.keras.models.load_model(str(model_path))

    # Architecture verification
    model_name = model.name
    input_shape = model.input_shape
    output_shape = model.output_shape

    total_params = int(model.count_params())
    trainable_params = int(
        sum(tf.keras.backend.count_params(w) for w in model.trainable_weights)
    )
    non_trainable_params = int(
        sum(tf.keras.backend.count_params(w) for w in model.non_trainable_weights)
    )

    # Check backbone layer
    backbone_layer = None
    for layer in model.layers:
        if "mobilenetv2" in layer.name.lower():
            backbone_layer = layer
            break

    backbone_trainable = backbone_layer.trainable if backbone_layer else None
    backbone_trainable_vars = (
        len(backbone_layer.trainable_variables) if backbone_layer else None
    )

    # Assert parameter counts
    if total_params != EXPECTED_TOTAL_PARAMS:
        raise ValueError(
            f"Total params mismatch: expected {EXPECTED_TOTAL_PARAMS}, got {total_params}"
        )
    if trainable_params != EXPECTED_TRAINABLE_PARAMS:
        raise ValueError(
            f"Trainable params mismatch: expected {EXPECTED_TRAINABLE_PARAMS}, got {trainable_params}"
        )
    if non_trainable_params != EXPECTED_NON_TRAINABLE_PARAMS:
        raise ValueError(
            f"Non-trainable params mismatch: expected {EXPECTED_NON_TRAINABLE_PARAMS}, got {non_trainable_params}"
        )
    if backbone_trainable is not False or backbone_trainable_vars != 0:
        raise ValueError(
            f"Backbone frozen check failed! trainable={backbone_trainable}, trainable_vars={backbone_trainable_vars}"
        )

    # Synthetic inference test (NO dataset used)
    tf.random.set_seed(42)
    x = tf.random.uniform(
        (1, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS),
        minval=0,
        maxval=255,
        dtype=tf.float32,
        seed=42,
    )
    pred_tensor = model(x, training=False)
    pred = pred_tensor.numpy()

    # Verify predictions tensor
    pred_shape = pred.shape
    nan_count = int(np.isnan(pred).sum())
    inf_count = int(np.isinf(pred).sum())
    prob_sum = float(np.sum(pred[0]))
    softmax_valid = (nan_count == 0) and (inf_count == 0) and np.isclose(prob_sum, 1.0, atol=1e-4)

    # Prediction decoding sanity
    predicted_index = int(np.argmax(pred[0]))
    predicted_breed = class_mapping[str(predicted_index)]
    confidence = float(np.max(pred[0]))

    index_valid = 0 <= predicted_index < NUM_CLASSES
    breed_valid = predicted_breed in class_mapping.values()
    conf_valid = 0.0 <= confidence <= 1.0

    if not (softmax_valid and index_valid and breed_valid and conf_valid):
        raise ValueError("Synthetic inference sanity checks failed!")

    return {
        "model_name": model_name,
        "input_shape": input_shape,
        "output_shape": output_shape,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "non_trainable_params": non_trainable_params,
        "backbone_trainable": backbone_trainable,
        "backbone_trainable_vars": backbone_trainable_vars,
        "pred_shape": pred_shape,
        "nan_count": nan_count,
        "inf_count": inf_count,
        "prob_sum": prob_sum,
        "softmax_valid": softmax_valid,
        "predicted_index": predicted_index,
        "predicted_breed": predicted_breed,
        "confidence": confidence,
    }


def create_model_manifest(
    output_path: Path,
    sha256_hash: str,
    file_size_bytes: int,
) -> Dict[str, Any]:
    """Generate model_manifest.json containing deployment metadata and compression decision."""
    manifest_data = {
        "model_name": "mobilenet_v2_transfer",
        "task": "dog_breed_classification",
        "num_classes": NUM_CLASSES,
        "input_height": IMAGE_HEIGHT,
        "input_width": IMAGE_WIDTH,
        "input_channels": IMAGE_CHANNELS,
        "input_dtype": "float32",
        "input_pixel_range": "[0,255]",
        "preprocessing": "MobileNetV2 preprocess_input inside model",
        "output": "120-class softmax",
        "backbone": "MobileNetV2",
        "backbone_weights": "ImageNet",
        "backbone_frozen": True,
        "validation_accuracy": VALIDATION_ACCURACY,
        "internal_test_accuracy": INTERNAL_TEST_ACCURACY,
        "production_model": True,
        "compression_used_for_production": False,
        "sha256": sha256_hash,
        "file_size_bytes": file_size_bytes,
        "tensorflow_version": tf.__version__,
        "keras_version": getattr(tf.keras, "__version__", "3.x"),
        "model_selection_rationale": {
            "selected_model": "Original dense MobileNetV2 transfer model (outputs/mobilenet_transfer/checkpoints/best_mobilenet_transfer.keras)",
            "experimental_compression_evaluated": "Manual magnitude pruning at 50% classifier sparsity (outputs/compression_manual/pruned_50.keras)",
            "original_validation_accuracy": VALIDATION_ACCURACY,
            "pruned_validation_accuracy": PRUNED_VALIDATION_ACCURACY,
            "accuracy_drop_pp": VALIDATION_DROP_PP,
            "pruned_zip_size_reduction_pct": PRUNED_ZIP_REDUCTION,
            "decision": "Pruned model rejected for production because it causes a 1.96 percentage point drop in validation accuracy with negligible size reduction (2.39%). Original unpruned model selected.",
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2, ensure_ascii=False)

    return manifest_data


def create_readme(output_path: Path) -> None:
    """Generate README.txt documenting artifact usage, preprocessing rules, and model selection."""
    readme_content = f"""================================================================================
Dog Breed Recognition System - Deployment Artifact (Task 6.3)
================================================================================

Production Model:
  File: dog_breed_model.keras
  Architecture: MobileNetV2 Frozen Transfer Learning
  Backbone: MobileNetV2 (ImageNet pretrained, frozen feature extractor)
  Head: GlobalAveragePooling2D -> Dropout(0.5) -> Dropout(0.2) -> Dense(120, softmax)
  Total Parameters: {EXPECTED_TOTAL_PARAMS:,}
  Trainable Parameters: {EXPECTED_TRAINABLE_PARAMS:,} (dense classifier kernel + bias)
  Non-trainable Parameters: {EXPECTED_NON_TRAINABLE_PARAMS:,} (frozen MobileNetV2 backbone)

Performance:
  Validation Accuracy: {VALIDATION_ACCURACY * 100:.2f}%
  Internal Test Accuracy: {INTERNAL_TEST_ACCURACY * 100:.2f}%

Input Specification:
  Channels: RGB
  Dimensions: 224 x 224 pixels
  Data Type: float32
  Pixel Range: [0, 255]

CRITICAL PREPROCESSING REQUIREMENT:
  Do NOT divide image pixel values by 255.0 externally!
  The model pipeline already incorporates an internal Lambda preprocessing layer:
    tf.keras.applications.mobilenet_v2.preprocess_input
  which automatically maps input raw pixels in [0, 255] to [-1.0, 1.0].
  Dividing by 255 externally would cause double-normalization and corrupt predictions.

Output Specification:
  Tensor shape: (batch_size, 120)
  Values: 120 softmax probabilities summing to 1.0.

Class Mapping:
  File: class_names.json
  Format: JSON dictionary mapping string class index ("0".."119") to dog breed name.
  Total Classes: 120

Model Loading Requirement:
  Before invoking `tf.keras.models.load_model("dog_breed_model.keras")`, register the
  MobileNetV2 preprocess_input custom object:
  -----------------------------------------------------------------------------
  import tensorflow as tf
  tf.keras.utils.get_custom_objects()["preprocess_input"] = (
      tf.keras.applications.mobilenet_v2.preprocess_input
  )
  model = tf.keras.models.load_model("dog_breed_model.keras")
  -----------------------------------------------------------------------------

Model Choice & Compression Documentation:
  Manual magnitude pruning was experimentally evaluated in Phase 6 (outputs/compression_manual/pruned_50.keras)
  with 50% classifier sparsity.
  - Original Validation Accuracy: {VALIDATION_ACCURACY * 100:.2f}%
  - Pruned Validation Accuracy: {PRUNED_VALIDATION_ACCURACY * 100:.2f}% (Drop: {VALIDATION_DROP_PP:.2f} percentage points)
  - Pruned ZIP Size Reduction: {PRUNED_ZIP_REDUCTION:.2f}%
  Decision: The pruned model was REJECTED for production due to accuracy degradation
  with negligible size benefits. The production artifact is the original, dense,
  unpruned MobileNetV2 transfer learning model. Compression artifacts are preserved in
  outputs/compression_manual/ for audit and experimental evidence.
================================================================================
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(readme_content)


def main() -> None:
    """Execute Task 6.3 deployment artifact preparation."""
    print("=" * 80)
    print("TASK 6.3: FINALIZE MODEL ARTIFACT FOR DEPLOYMENT")
    print("=" * 80)

    # 1. Copy model and verify SHA-256 integrity
    print("\n[Step 1/5] Copying production model and computing SHA-256 integrity...")
    copy_meta = copy_and_verify_model(SOURCE_MODEL_PATH, DEPLOY_MODEL_PATH)
    print(f"  Source path:      {copy_meta['source_path']}")
    print(f"  Destination path: {copy_meta['dest_path']}")
    print(f"  Source size:      {copy_meta['source_size']:,} bytes")
    print(f"  Destination size: {copy_meta['dest_size']:,} bytes")
    print(f"  Source SHA-256:   {copy_meta['source_sha256']}")
    print(f"  Dest SHA-256:     {copy_meta['dest_sha256']}")
    print(f"  Binary identical: {'YES' if copy_meta['identical'] else 'NO'}")

    # 2. Export class mapping
    print("\n[Step 2/5] Exporting and validating class names...")
    class_mapping = export_and_verify_class_names(CLASS_NAMES_PATH)
    indices = [int(k) for k in class_mapping.keys()]
    print(f"  Class mapping path: {CLASS_NAMES_PATH}")
    print(f"  Total classes:      {len(class_mapping)}")
    print(f"  Index range:        {min(indices)}..{max(indices)}")
    print(f"  First class:        idx 0 -> '{class_mapping['0']}'")
    print(f"  Last class:         idx 119 -> '{class_mapping['119']}'")
    print("  Class mapping validation: PASS")

    # 3. Load deployment model and test synthetic inference
    print("\n[Step 3/5] Loading deployment model and running synthetic inference...")
    model_meta = load_and_verify_model(DEPLOY_MODEL_PATH, class_mapping)
    print(f"  Model loaded:         YES")
    print(f"  Model name:           {model_meta['model_name']}")
    print(f"  Input shape:          {model_meta['input_shape']}")
    print(f"  Output shape:         {model_meta['output_shape']}")
    print(f"  Total params:         {model_meta['total_params']:,}")
    print(f"  Trainable params:     {model_meta['trainable_params']:,}")
    print(f"  Non-trainable params: {model_meta['non_trainable_params']:,}")
    print(f"  Backbone trainable:   {model_meta['backbone_trainable']}")
    print(f"  Backbone train vars:  {model_meta['backbone_trainable_vars']}")
    print(f"  Synthetic input:      (1, 224, 224, 3) float32 [0, 255]")
    print(f"  Synthetic output:     {model_meta['pred_shape']}")
    print(f"  NaN count:            {model_meta['nan_count']}")
    print(f"  Inf count:            {model_meta['inf_count']}")
    print(f"  Probability sum:      {model_meta['prob_sum']:.6f} ~= 1.0")
    print(f"  Predicted index:      {model_meta['predicted_index']}")
    print(f"  Predicted breed:      {model_meta['predicted_breed']}")
    print(f"  Confidence:           {model_meta['confidence']:.4f}")
    print("  Synthetic inference:  PASS")

    # 4. Generate manifest
    print("\n[Step 4/5] Generating model_manifest.json...")
    manifest_meta = create_model_manifest(
        MANIFEST_PATH,
        sha256_hash=copy_meta["dest_sha256"],
        file_size_bytes=copy_meta["dest_size"],
    )
    print(f"  Manifest written:     {MANIFEST_PATH}")
    print(f"  Validation accuracy:  {manifest_meta['validation_accuracy']}")
    print(f"  Test accuracy:        {manifest_meta['internal_test_accuracy']}")
    print(f"  Production model:     {manifest_meta['production_model']}")
    print(f"  Compression for prod: {manifest_meta['compression_used_for_production']}")

    # 5. Generate README
    print("\n[Step 5/5] Generating README.txt...")
    create_readme(README_PATH)
    print(f"  README written:       {README_PATH}")

    # Final sanity check of artifacts
    print("\n[Verification] Checking all deployment artifacts on disk...")
    for path in (DEPLOY_MODEL_PATH, CLASS_NAMES_PATH, MANIFEST_PATH, README_PATH):
        exists = path.is_file()
        size = path.stat().st_size if exists else 0
        print(f"  {path.name:25s} -> Exists: {'YES' if exists else 'NO'} ({size:,} bytes)")

    print("\n" + "=" * 80)
    print("TASK 6.3 COMPLETED SUCCESSFULLY: PASS")
    print("=" * 80)


if __name__ == "__main__":
    main()
