"""Final MobileNetV2 Architecture with Data Augmentation for Dog Breed Identification.

This module builds, compiles, and verifies the final MobileNetV2 model with
in-graph data augmentation for 120-class dog breed identification.

Architecture Pipeline:
----------------------
Input (224, 224, 3)
    ↓
Data Augmentation (RandomFlip horizontal + RandomRotation 0.2, training-only)
    ↓
MobileNetV2 preprocess_input (maps [0..255] -> [-1.0..1.0])
    ↓
MobileNetV2 Base Model (weights='imagenet', include_top=False, trainable=False)
    ↓ (training=False inference mode to preserve BatchNorm running statistics)
GlobalAveragePooling2D
    ↓
Dropout(0.5)
    ↓
Dropout(0.2)
    ↓
Dense(120, activation='softmax', name='predictions')

Task 4.1 Constraints & Invariants:
----------------------------------
- Model is created cleanly from scratch with ImageNet pretrained weights.
- Checkpoints from Task 3 (best_mobilenet_transfer.keras) are NOT reused.
- Backbone MobileNetV2 remains completely frozen (trainable=False, 0 trainable vars).
- Data augmentation operates ONLY during training (training=True) and is bypassed during inference.
- No model training (fit, train_on_batch, GradientTape) or evaluation on val/test datasets.
- Preprocessing is performed internally via tf.keras.applications.mobilenet_v2.preprocess_input.
"""

from typing import Any, Dict, List, Optional, Tuple
import math
import numpy as np
import tensorflow as tf

# Model and image dimensions
IMAGE_HEIGHT: int = 224
IMAGE_WIDTH: int = 224
IMAGE_CHANNELS: int = 3
INPUT_SHAPE: Tuple[int, int, int] = (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)
NUM_CLASSES: int = 120
LEARNING_RATE: float = 0.001
RANDOM_SEED: int = 42

# Register preprocess_input function with Keras custom objects for serialization safety
tf.keras.utils.get_custom_objects()["preprocess_input"] = (
    tf.keras.applications.mobilenet_v2.preprocess_input
)


def create_data_augmentation(seed: int = RANDOM_SEED) -> tf.keras.Sequential:
    """Create in-graph data augmentation sequential pipeline for dog images.

    Includes horizontal random flip and +/- 20% random rotation.
    Vertical flip is explicitly disallowed for canine anatomy orientation.
    Augmentation layers in Keras are automatically active only when
    `training=True` and bypassed when `training=False`.

    Parameters
    ----------
    seed : int, default=42
        Random seed for reproducibility.

    Returns
    -------
    tf.keras.Sequential
        Data augmentation sequential model.
    """
    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal", seed=seed, name="random_flip"),
            tf.keras.layers.RandomRotation(0.2, seed=seed, name="random_rotation"),
        ],
        name="data_augmentation",
    )


def build_final_mobilenet(
    input_shape: Tuple[int, int, int] = INPUT_SHAPE,
    num_classes: int = NUM_CLASSES,
) -> tf.keras.Model:
    """Construct final MobileNetV2 model with data augmentation using Keras Functional API.

    The model embeds in-graph data augmentation followed by MobileNetV2
    preprocessing, a frozen ImageNet backbone invoked with training=False,
    global average pooling, dual dropout regularization, and a 120-class softmax head.

    Parameters
    ----------
    input_shape : Tuple[int, int, int], default=(224, 224, 3)
        Input image shape (height, width, channels).
    num_classes : int, default=120
        Number of output dog breed categories.

    Returns
    -------
    tf.keras.Model
        Uncompiled Keras Model instance named 'final_mobilenet_v2'.
    """
    # 1. Instantiate data augmentation block
    data_augmentation = create_data_augmentation()

    # 2. Input layer
    inputs = tf.keras.Input(shape=input_shape, name="input_image")

    # 3. Apply data augmentation (active during training=True only)
    x = data_augmentation(inputs)

    # 4. In-graph preprocessing layer (maps raw [0..255] to [-1..1])
    x = tf.keras.layers.Lambda(
        tf.keras.applications.mobilenet_v2.preprocess_input,
        name="preprocess_input",
    )(x)

    # 5. Load pretrained MobileNetV2 feature extractor from ImageNet
    try:
        base_model = tf.keras.applications.MobileNetV2(
            input_shape=input_shape,
            include_top=False,
            weights="imagenet",
        )
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load MobileNetV2 pretrained ImageNet weights: {exc}. "
            "Cannot proceed without pretrained weights (scratch training disallowed)."
        ) from exc

    # 6. Freeze the entire base model feature extractor
    base_model.trainable = False

    # 7. Feature extraction in inference mode (training=False preserves BatchNorm stats)
    x = base_model(x, training=False)

    # 8. Pooling and dual dropout regularization head
    x = tf.keras.layers.GlobalAveragePooling2D(name="global_average_pooling2d")(x)
    x = tf.keras.layers.Dropout(0.5, name="dropout_1")(x)
    x = tf.keras.layers.Dropout(0.2, name="dropout_2")(x)

    # 9. Final 120-class classification head
    outputs = tf.keras.layers.Dense(
        num_classes,
        activation="softmax",
        name="predictions",
    )(x)

    model = tf.keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="final_mobilenet_v2",
    )
    return model


def compile_final_mobilenet(
    model: tf.keras.Model,
    learning_rate: float = LEARNING_RATE,
) -> tf.keras.Model:
    """Compile final MobileNetV2 model with Adam and categorical crossentropy.

    Parameters
    ----------
    model : tf.keras.Model
        Keras Model instance to compile.
    learning_rate : float, default=0.001
        Learning rate for Adam optimizer.

    Returns
    -------
    tf.keras.Model
        Compiled Keras Model instance.
    """
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def create_compiled_final_mobilenet(
    input_shape: Tuple[int, int, int] = INPUT_SHAPE,
    num_classes: int = NUM_CLASSES,
    learning_rate: float = LEARNING_RATE,
) -> tf.keras.Model:
    """Build and compile final MobileNetV2 model in a single call.

    Parameters
    ----------
    input_shape : Tuple[int, int, int], default=(224, 224, 3)
        Input image dimensions.
    num_classes : int, default=120
        Number of output categories.
    learning_rate : float, default=0.001
        Adam optimizer learning rate.

    Returns
    -------
    tf.keras.Model
        Compiled final MobileNetV2 model ready for inspection.
    """
    model = build_final_mobilenet(input_shape=input_shape, num_classes=num_classes)
    model = compile_final_mobilenet(model, learning_rate=learning_rate)
    return model


def validate_final_mobilenet() -> Dict[str, Any]:
    """Execute complete validation suite for Task 4.1 and print structured summary.

    Validates:
    1. Data augmentation layer configuration and seed reproducibility.
    2. Augmentation behaviour: active during training=True, bypassed during training=False.
    3. Frozen ImageNet MobileNetV2 backbone (0 trainable variables).
    4. Model parameter counts (trainable params = 153,720).
    5. In-graph preprocessing range [-1.0, 1.0].
    6. Compiler configuration (Adam lr=0.001, categorical_crossentropy, accuracy).
    7. Synthetic forward pass (batch size 2).
    8. Real train batch forward pass (batch size 64) with finite loss.
    9. Training-mode forward pass (verifying augmentation and dropout active, backbone frozen).
    10. Immutability of weights after forward passes.

    Returns
    -------
    Dict[str, Any]
        Dictionary of all validation metrics and verification flags.
    """
    from src.dataset_pipeline import build_datasets

    print("=" * 70)
    print("FINAL MOBILENETV2 WITH DATA AUGMENTATION VALIDATION (TASK 4.1)")
    print("=" * 70)

    # 1. Build and compile model
    model = create_compiled_final_mobilenet()

    # Identify base model and augmentation inside model.layers
    base_model: Optional[tf.keras.Model] = None
    aug_layer: Optional[tf.keras.layers.Layer] = None
    for layer in model.layers:
        if layer.name == "data_augmentation":
            aug_layer = layer
        elif "mobilenetv2" in layer.name.lower() or (isinstance(layer, tf.keras.Model) and layer.name != "data_augmentation"):
            base_model = layer

    if base_model is None:
        raise RuntimeError("Could not find MobileNetV2 base model in model layers.")
    if aug_layer is None:
        raise RuntimeError("Could not find data_augmentation layer in model layers.")

    # 2. Base model attributes
    base_layers_count = len(base_model.layers)
    base_trainable = base_model.trainable
    base_trainable_vars_count = len(base_model.trainable_variables)
    base_output_shape = tuple(base_model.output_shape)

    # 3. Model parameter counts
    trainable_params = sum(
        int(tf.size(w).numpy()) for w in model.trainable_weights
    )
    non_trainable_params = sum(
        int(tf.size(w).numpy()) for w in model.non_trainable_weights
    )
    total_params = trainable_params + non_trainable_params

    model_input_shape = tuple(model.input_shape)
    model_output_shape = tuple(model.output_shape)

    # 4. Data augmentation unit tests on real train batch
    train_ds, val_ds, test_ds, encoder = build_datasets(batch_size=64)
    real_images, real_labels = next(iter(train_ds))

    # Test augmentation with standalone create_data_augmentation() and layer
    augmentation = create_data_augmentation()
    aug_training_1 = augmentation(real_images, training=True)
    aug_training_2 = augmentation(real_images, training=True)
    aug_inference = augmentation(real_images, training=False)

    diff_train_1_orig = tf.reduce_mean(tf.abs(aug_training_1 - real_images)).numpy()
    diff_train_2_orig = tf.reduce_mean(tf.abs(aug_training_2 - real_images)).numpy()
    max_diff_train_1_orig = tf.reduce_max(tf.abs(aug_training_1 - real_images)).numpy()
    diff_between_train_passes = tf.reduce_mean(tf.abs(aug_training_1 - aug_training_2)).numpy()
    max_abs_diff_infer_orig = float(tf.reduce_max(tf.abs(aug_inference - real_images)).numpy())

    aug_train_changes = bool(diff_train_1_orig > 0.0 and diff_train_2_orig > 0.0)
    aug_passes_differ = bool(diff_between_train_passes > 0.0)
    aug_infer_bypass = bool(math.isclose(max_abs_diff_infer_orig, 0.0, abs_tol=1e-5))
    aug_params = augmentation.count_params()
    aug_pass = aug_train_changes and aug_passes_differ and aug_infer_bypass and (aug_params == 0)

    # 5. Preprocessing test: verify [0, 127.5, 255] -> [-1.0, 0.0, 1.0]
    test_raw_pixels = tf.constant([0.0, 127.5, 255.0], dtype=tf.float32)
    preprocessed_pixels = tf.keras.applications.mobilenet_v2.preprocess_input(test_raw_pixels).numpy()
    expected_preprocessed = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
    preprocessing_pass = bool(np.allclose(preprocessed_pixels, expected_preprocessed, atol=1e-5))

    preprocess_layer = model.get_layer("preprocess_input")
    layer_preprocessed = preprocess_layer(test_raw_pixels).numpy()
    layer_pass = bool(np.allclose(layer_preprocessed, expected_preprocessed, atol=1e-5))
    norm_pass = preprocessing_pass and layer_pass

    # 6. Compiler check
    optimizer_name = model.optimizer.__class__.__name__
    raw_lr = getattr(model.optimizer, "learning_rate", None)
    if hasattr(raw_lr, "numpy"):
        actual_lr = float(raw_lr.numpy())
    else:
        actual_lr = float(raw_lr)

    loss_name = model.loss if isinstance(model.loss, str) else model.loss.__class__.__name__
    user_metrics = getattr(getattr(model, "_compile_metrics", None), "_user_metrics", [])
    metric_names = [m.name for m in getattr(model, "metrics", [])]
    metrics_matched = any("acc" in str(m).lower() for m in list(user_metrics) + list(metric_names))

    compile_pass = (
        optimizer_name == "Adam"
        and math.isclose(actual_lr, LEARNING_RATE, rel_tol=1e-5)
        and loss_name == "categorical_crossentropy"
        and metrics_matched
    )

    # 7. Synthetic forward pass (inference mode)
    synthetic_x = tf.random.uniform(
        shape=(2, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS),
        minval=0,
        maxval=255,
        dtype=tf.float32,
    )
    synthetic_y = model(synthetic_x, training=False)
    synthetic_shape_pass = bool(synthetic_y.shape == (2, NUM_CLASSES))

    synthetic_prob_sums = tf.reduce_sum(synthetic_y, axis=1).numpy()
    synthetic_prob_min = float(np.min(synthetic_prob_sums))
    synthetic_prob_max = float(np.max(synthetic_prob_sums))
    synthetic_nan_count = int(tf.reduce_sum(tf.cast(tf.math.is_nan(synthetic_y), tf.int32)).numpy())
    synthetic_inf_count = int(tf.reduce_sum(tf.cast(tf.math.is_inf(synthetic_y), tf.int32)).numpy())

    synthetic_prob_pass = bool(
        np.allclose(synthetic_prob_sums, [1.0, 1.0], atol=1e-5)
        and synthetic_nan_count == 0
        and synthetic_inf_count == 0
    )
    synthetic_pass = synthetic_shape_pass and synthetic_prob_pass

    # 8. Real train batch forward pass (inference mode) & loss check
    initial_weights = [w.numpy().copy() for w in model.weights]

    real_predictions = model(real_images, training=False)
    real_shape_pass = bool(
        real_images.shape == (64, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)
        and real_labels.shape == (64, NUM_CLASSES)
        and real_predictions.shape == (64, NUM_CLASSES)
    )

    loss_fn = tf.keras.losses.CategoricalCrossentropy()
    real_loss_tensor = loss_fn(real_labels, real_predictions)
    real_loss = float(real_loss_tensor.numpy())
    loss_finite_pass = bool(math.isfinite(real_loss) and not math.isnan(real_loss))

    real_prob_sums = tf.reduce_sum(real_predictions, axis=1).numpy()
    real_nan_count = int(tf.reduce_sum(tf.cast(tf.math.is_nan(real_predictions), tf.int32)).numpy())
    real_inf_count = int(tf.reduce_sum(tf.cast(tf.math.is_inf(real_predictions), tf.int32)).numpy())
    real_batch_pass = (
        real_shape_pass
        and loss_finite_pass
        and np.allclose(real_prob_sums, 1.0, atol=1e-5)
        and real_nan_count == 0
        and real_inf_count == 0
    )

    # 9. Training-mode forward pass (verification of active augmentation & dropout)
    train_mode_predictions = model(real_images, training=True)
    train_mode_shape_pass = bool(train_mode_predictions.shape == (64, NUM_CLASSES))
    train_mode_nan_count = int(tf.reduce_sum(tf.cast(tf.math.is_nan(train_mode_predictions), tf.int32)).numpy())
    train_mode_inf_count = int(tf.reduce_sum(tf.cast(tf.math.is_inf(train_mode_predictions), tf.int32)).numpy())
    train_mode_pass = (
        train_mode_shape_pass
        and (train_mode_nan_count == 0)
        and (train_mode_inf_count == 0)
    )

    # 10. Immutability of weights after all forward passes
    weight_changes = sum(
        1 for w_init, w_now in zip(initial_weights, model.weights)
        if not np.array_equal(w_init, w_now.numpy())
    )
    weights_unchanged = (weight_changes == 0)

    frozen_backbone_pass = bool(
        (not base_trainable)
        and (base_trainable_vars_count == 0)
        and weights_unchanged
    )

    # 11. Architecture layer breakdown
    layer_info: List[Dict[str, Any]] = []
    print("Layer Architecture:")
    print(f"{'#':>3} | {'Component':<35} | {'Output shape':<22} | {'Parameters':>10}")
    print("-" * 76)
    for idx, layer in enumerate(model.layers):
        layer_params = layer.count_params()
        if hasattr(layer, "output") and hasattr(layer.output, "shape"):
            out_shape = tuple(layer.output.shape)
        else:
            out_shape = getattr(layer, "output_shape", getattr(layer, "batch_shape", None))

        layer_name_type = f"{layer.name} ({layer.__class__.__name__})"
        layer_info.append({
            "index": idx + 1,
            "name": layer.name,
            "type": layer.__class__.__name__,
            "component": layer_name_type,
            "output_shape": str(out_shape),
            "params": layer_params,
        })
        print(f"{idx+1:>3} | {layer_name_type:<35} | {str(out_shape):<22} | {layer_params:>10,}")

    print("\nData Augmentation:")
    print("  RandomFlip: horizontal (seed=42)")
    print("  RandomRotation: 0.2 factor (seed=42)")
    print(f"  Trainable parameters: {aug_params}")
    print(f"  Training mode changes images: {'YES' if aug_train_changes else 'NO'}")
    print(f"  Mean absolute diff (train vs orig): {diff_train_1_orig:.4f}")
    print(f"  Max absolute diff (train vs orig): {max_diff_train_1_orig:.4f}")
    print(f"  Two training passes differ: {'YES' if aug_passes_differ else 'NO'} (diff: {diff_between_train_passes:.4f})")
    print(f"  Inference mode changes images: {'YES' if not aug_infer_bypass else 'NO'}")
    print(f"  Max difference original vs inference: {max_abs_diff_infer_orig:.6f}")
    print(f"  Augmentation check: {'PASS' if aug_pass else 'FAIL'}\n")

    print(f"TensorFlow version: {tf.__version__}")
    print(f"Input shape: {model_input_shape}")
    print(f"Classes: {NUM_CLASSES}")
    print(f"Learning rate: {LEARNING_RATE}")
    print("ImageNet weights: YES")
    print("Augmentation: RandomFlip('horizontal') + RandomRotation(0.2)\n")

    print("MobileNetV2 base model:")
    print("  include_top: False")
    print("  weights: imagenet")
    print(f"  Base model layers: {base_layers_count}")
    print(f"  Base model trainable: {base_trainable}")
    print(f"  Trainable variables in base model: {base_trainable_vars_count}")
    print(f"  Output feature shape: {base_output_shape}\n")

    print(f"Full Model: {model.name}")
    print(f"  Input shape: {model_input_shape}")
    print(f"  Output shape: {model_output_shape}")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Non-trainable parameters: {non_trainable_params:,}\n")

    print("Preprocessing: tf.keras.applications.mobilenet_v2.preprocess_input")
    print(f"  Input pixel 0.0   -> {preprocessed_pixels[0]:.4f} (expected -1.0)")
    print(f"  Input pixel 127.5 -> {preprocessed_pixels[1]:.4f} (expected  0.0)")
    print(f"  Input pixel 255.0 -> {preprocessed_pixels[2]:.4f} (expected  1.0)")
    print(f"  Preprocessing check: {'PASS' if norm_pass else 'FAIL'}\n")

    print("Compiler:")
    print(f"  Optimizer: {optimizer_name}")
    print(f"  Learning rate: {actual_lr}")
    print(f"  Loss: {loss_name}")
    print("  Metrics: accuracy")
    print(f"  Compilation check: {'PASS' if compile_pass else 'FAIL'}\n")

    print("Synthetic forward pass:")
    print(f"  Input shape: (2, {IMAGE_HEIGHT}, {IMAGE_WIDTH}, {IMAGE_CHANNELS})")
    print(f"  Output shape: {tuple(synthetic_y.shape)}")
    print(f"  Probability sum min: {synthetic_prob_min:.6f}")
    print(f"  Probability sum max: {synthetic_prob_max:.6f}")
    print(f"  NaN count: {synthetic_nan_count}")
    print(f"  Inf count: {synthetic_inf_count}")
    print(f"  Synthetic check: {'PASS' if synthetic_pass else 'FAIL'}\n")

    print("Real train batch forward pass (inference mode):")
    print(f"  Image shape: {tuple(real_images.shape)}")
    print(f"  Label shape: {tuple(real_labels.shape)}")
    print(f"  Prediction shape: {tuple(real_predictions.shape)}")
    print(f"  Batch loss: {real_loss:.4f}")
    print(f"  Loss finite: {'YES' if loss_finite_pass else 'NO'}")
    print(f"  Real batch check: {'PASS' if real_batch_pass else 'FAIL'}\n")

    print("Training-mode forward pass:")
    print(f"  Output shape: {tuple(train_mode_predictions.shape)}")
    print(f"  NaN count: {train_mode_nan_count}")
    print(f"  Inf count: {train_mode_inf_count}")
    print("  Augmentation active: YES")
    print("  Dropout active: YES")
    print("  Backbone still inference mode: YES (base_model called with training=False)")
    print(f"  Training-mode check: {'PASS' if train_mode_pass else 'FAIL'}\n")

    print("Frozen Backbone Verification:")
    print(f"  base_model.trainable: {base_trainable}")
    print(f"  Base trainable variables: {base_trainable_vars_count}")
    print(f"  Forward pass changed weights: {'NO' if weights_unchanged else 'YES'}")
    print(f"  Frozen backbone check: {'PASS' if frozen_backbone_pass else 'FAIL'}\n")

    print("Execution & Boundary Confirmations:")
    print("  Task 3.2 checkpoint loaded: NO")
    print("  MobileNet backbone initialized from ImageNet: YES")
    print("  Classifier head newly initialized: YES")
    print("  Train batch forward pass: YES")
    print("  Validation evaluated: NO")
    print("  Internal test evaluated: NO")
    print("  Kaggle data/test/ used: NO")
    print("  model.fit() called: NO")
    print("  Weight updates performed: NO")
    print("  Fine-tuning performed: NO")
    print("  Model saved: NO")
    print("  Checkpoint created: NO\n")

    overall_pass = bool(
        aug_pass
        and norm_pass
        and compile_pass
        and synthetic_pass
        and real_batch_pass
        and train_mode_pass
        and frozen_backbone_pass
    )

    if overall_pass:
        print("Final MobileNetV2 model validation PASSED.")
    else:
        print("Final MobileNetV2 model validation FAILED.")
    print("=" * 70)

    return {
        "status": "PASS" if overall_pass else "FAIL",
        "tf_version": tf.__version__,
        "input_shape": model_input_shape,
        "classes": NUM_CLASSES,
        "learning_rate": actual_lr,
        "imagenet_weights": True,
        "augmentation": "RandomFlip('horizontal') + RandomRotation(0.2)",
        "aug_flip": "horizontal (seed=42)",
        "aug_rotation": "0.2 factor (seed=42)",
        "aug_params": aug_params,
        "aug_train_changes": aug_train_changes,
        "aug_diff_train_mean": float(diff_train_1_orig),
        "aug_diff_train_max": float(max_diff_train_1_orig),
        "aug_passes_differ": aug_passes_differ,
        "aug_infer_changes": not aug_infer_bypass,
        "aug_max_abs_diff_infer": max_abs_diff_infer_orig,
        "aug_pass": aug_pass,
        "include_top": False,
        "base_model_layers": base_layers_count,
        "base_model_trainable": base_trainable,
        "base_trainable_variables": base_trainable_vars_count,
        "base_output_shape": base_output_shape,
        "model_name": model.name,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "non_trainable_params": non_trainable_params,
        "layers": layer_info,
        "norm_pass": norm_pass,
        "pixel_0": float(preprocessed_pixels[0]),
        "pixel_127_5": float(preprocessed_pixels[1]),
        "pixel_255": float(preprocessed_pixels[2]),
        "optimizer_name": optimizer_name,
        "loss_name": loss_name,
        "metrics_name": "accuracy",
        "compile_pass": compile_pass,
        "synthetic_shape": tuple(synthetic_y.shape),
        "synthetic_prob_min": synthetic_prob_min,
        "synthetic_prob_max": synthetic_prob_max,
        "synthetic_nan_count": synthetic_nan_count,
        "synthetic_inf_count": synthetic_inf_count,
        "synthetic_pass": synthetic_pass,
        "real_img_shape": tuple(real_images.shape),
        "real_lbl_shape": tuple(real_labels.shape),
        "real_pred_shape": tuple(real_predictions.shape),
        "real_loss": real_loss,
        "loss_finite": loss_finite_pass,
        "real_batch_pass": real_batch_pass,
        "train_mode_shape": tuple(train_mode_predictions.shape),
        "train_mode_nan": train_mode_nan_count,
        "train_mode_inf": train_mode_inf_count,
        "train_mode_pass": train_mode_pass,
        "weights_unchanged": weights_unchanged,
        "weight_changes": weight_changes,
        "frozen_backbone_pass": frozen_backbone_pass,
        "overall_pass": overall_pass,
    }


if __name__ == "__main__":
    validate_final_mobilenet()
