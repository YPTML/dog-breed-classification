"""MobileNetV2 Transfer Learning Model for Dog Breed Identification.

This module builds, compiles, and verifies the MobileNetV2 transfer learning
architecture for 120-class dog breed classification.

Architecture Pipeline:
----------------------
Input (224, 224, 3)
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

Constraints:
------------
- Base model is completely frozen (0 trainable parameters in base model).
- Training and fine-tuning are strictly deferred to subsequent tasks.
- No model.fit(), train_on_batch(), or gradient updates performed.
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

# Register preprocess_input function with Keras custom objects for .keras serialization
tf.keras.utils.get_custom_objects()["preprocess_input"] = (
    tf.keras.applications.mobilenet_v2.preprocess_input
)


def build_mobilenet_transfer(
    input_shape: Tuple[int, int, int] = INPUT_SHAPE,
    num_classes: int = NUM_CLASSES,
) -> tf.keras.Model:
    """Construct MobileNetV2 transfer learning model using Keras Functional API.

    The base model is loaded with ImageNet weights and frozen (trainable=False).
    In-graph preprocessing using MobileNetV2's preprocess_input transforms raw
    pixel values in [0, 255] into [-1.0, 1.0]. The base model is invoked with
    training=False to guarantee stable BatchNormalization running statistics.

    Parameters
    ----------
    input_shape : Tuple[int, int, int], default=(224, 224, 3)
        Input image shape (height, width, channels).
    num_classes : int, default=120
        Number of output dog breed categories.

    Returns
    -------
    tf.keras.Model
        Uncompiled Keras Model instance.
    """
    # 1. Load pretrained MobileNetV2 feature extractor
    try:
        base_model = tf.keras.applications.MobileNetV2(
            input_shape=input_shape,
            include_top=False,
            weights="imagenet",
        )
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load MobileNetV2 pretrained ImageNet weights: {exc}. "
            "Cannot proceed without pretrained weights (training from scratch is disallowed)."
        ) from exc

    # 2. Freeze the entire base model feature extractor
    base_model.trainable = False

    # 3. Define Functional model inputs
    inputs = tf.keras.Input(shape=input_shape, name="input_image")

    # 4. In-graph preprocessing layer (maps [0..255] to [-1..1])
    x = tf.keras.layers.Lambda(
        tf.keras.applications.mobilenet_v2.preprocess_input,
        name="preprocess_input",
    )(inputs)

    # 5. Feature extraction with base model in inference mode
    # training=False ensures BatchNorm layers use learned population statistics
    x = base_model(x, training=False)

    # 6. Pooling and regularization head
    x = tf.keras.layers.GlobalAveragePooling2D(name="global_average_pooling2d")(x)
    x = tf.keras.layers.Dropout(0.5, name="dropout_1")(x)
    x = tf.keras.layers.Dropout(0.2, name="dropout_2")(x)

    # 7. Final 120-class classification head
    outputs = tf.keras.layers.Dense(
        num_classes,
        activation="softmax",
        name="predictions",
    )(x)

    model = tf.keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="mobilenet_v2_transfer",
    )
    return model


def compile_mobilenet_transfer(
    model: tf.keras.Model,
    learning_rate: float = LEARNING_RATE,
) -> tf.keras.Model:
    """Compile MobileNetV2 transfer learning model with Adam and categorical crossentropy.

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


def create_compiled_mobilenet_transfer(
    input_shape: Tuple[int, int, int] = INPUT_SHAPE,
    num_classes: int = NUM_CLASSES,
    learning_rate: float = LEARNING_RATE,
) -> tf.keras.Model:
    """Build and compile MobileNetV2 transfer learning model in a single call.

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
        Compiled MobileNetV2 transfer learning model ready for inspection.
    """
    model = build_mobilenet_transfer(input_shape=input_shape, num_classes=num_classes)
    model = compile_mobilenet_transfer(model, learning_rate=learning_rate)
    return model


def validate_mobilenet_transfer() -> Dict[str, Any]:
    """Execute complete validation suite for Task 3.1 and print structured summary.

    Validates:
    - Model construction and compilation
    - Frozen base model parameters and variables
    - In-graph preprocessing layer mapping
    - Synthetic forward pass (batch size 2)
    - Real train batch forward pass (batch size 64)
    - Loss sanity check (finite and valid)
    - Confirmation that weights remain unmodified after forward passes

    Returns
    -------
    Dict[str, Any]
        Dictionary of all validation metrics and verification flags.
    """
    from src.dataset_pipeline import build_datasets

    print("=" * 70)
    print("MOBILENETV2 TRANSFER LEARNING VALIDATION (TASK 3.1)")
    print("=" * 70)

    # 1. Build and compile model
    model = create_compiled_mobilenet_transfer()

    # Identify base model inside model.layers
    base_model: Optional[tf.keras.Model] = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model) or "mobilenetv2" in layer.name.lower():
            base_model = layer
            break

    if base_model is None:
        raise RuntimeError("Could not find MobileNetV2 base model in model layers.")

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

    print(f"TensorFlow version: {tf.__version__}")
    print(f"Input shape: {model_input_shape}")
    print(f"Classes: {NUM_CLASSES}")
    print(f"Learning rate: {LEARNING_RATE}")
    print("ImageNet weights: YES")
    print("Augmentation: NO\n")

    print("MobileNetV2 base model:")
    print(f"  include_top: False")
    print(f"  weights: imagenet")
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

    # 4. Architecture layer breakdown
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

    # 5. Preprocessing test: verify [0, 127.5, 255] -> [-1.0, 0.0, 1.0]
    test_raw_pixels = tf.constant([0.0, 127.5, 255.0], dtype=tf.float32)
    preprocessed_pixels = tf.keras.applications.mobilenet_v2.preprocess_input(test_raw_pixels).numpy()
    expected_preprocessed = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
    preprocessing_pass = bool(np.allclose(preprocessed_pixels, expected_preprocessed, atol=1e-5))

    # Also test through model's Lambda layer directly
    preprocess_layer = model.get_layer("preprocess_input")
    layer_preprocessed = preprocess_layer(test_raw_pixels).numpy()
    layer_pass = bool(np.allclose(layer_preprocessed, expected_preprocessed, atol=1e-5))
    norm_pass = preprocessing_pass and layer_pass

    print(f"\nPreprocessing: tf.keras.applications.mobilenet_v2.preprocess_input")
    print(f"  Input pixel 0.0   -> {preprocessed_pixels[0]:.4f} (expected -1.0)")
    print(f"  Input pixel 127.5 -> {preprocessed_pixels[1]:.4f} (expected  0.0)")
    print(f"  Input pixel 255.0 -> {preprocessed_pixels[2]:.4f} (expected  1.0)")
    print(f"  Preprocessing check: {'PASS' if norm_pass else 'FAIL'}\n")

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

    print("Compiler:")
    print(f"  Optimizer: {optimizer_name}")
    print(f"  Learning rate: {actual_lr}")
    print(f"  Loss: {loss_name}")
    print(f"  Metrics: accuracy")
    print(f"  Compilation check: {'PASS' if compile_pass else 'FAIL'}\n")

    # 7. Synthetic forward pass
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

    print("Synthetic forward pass:")
    print(f"  Input shape: (2, {IMAGE_HEIGHT}, {IMAGE_WIDTH}, {IMAGE_CHANNELS})")
    print(f"  Output shape: {tuple(synthetic_y.shape)}")
    print(f"  Probability sum min: {synthetic_prob_min:.6f}")
    print(f"  Probability sum max: {synthetic_prob_max:.6f}")
    print(f"  NaN count: {synthetic_nan_count}")
    print(f"  Inf count: {synthetic_inf_count}")
    print(f"  Synthetic check: {'PASS' if synthetic_pass else 'FAIL'}\n")

    # 8. Real train batch forward pass & loss check
    # Snapshot weights before forward pass to verify immutability
    initial_weights = [w.numpy().copy() for w in model.weights]

    train_ds, val_ds, test_ds, encoder = build_datasets(batch_size=64)
    real_images, real_labels = next(iter(train_ds))

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

    print("Real train batch forward pass:")
    print(f"  Image shape: {tuple(real_images.shape)}")
    print(f"  Label shape: {tuple(real_labels.shape)}")
    print(f"  Prediction shape: {tuple(real_predictions.shape)}")
    print(f"  Batch loss: {real_loss:.4f}")
    print(f"  Loss finite: {'YES' if loss_finite_pass else 'NO'}")
    print(f"  Real batch check: {'PASS' if real_batch_pass else 'FAIL'}\n")

    # 9. Verify weights unchanged after all forward passes
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

    print("Frozen Backbone Verification:")
    print(f"  base_model.trainable: {base_trainable}")
    print(f"  Base trainable variables: {base_trainable_vars_count}")
    print(f"  Forward pass changed weights: {'NO' if weights_unchanged else 'YES'}")
    print(f"  Frozen backbone check: {'PASS' if frozen_backbone_pass else 'FAIL'}\n")

    # 10. Data usage & training boundary confirmations
    print("Execution Boundaries:")
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
        norm_pass
        and compile_pass
        and synthetic_pass
        and real_batch_pass
        and frozen_backbone_pass
    )

    if overall_pass:
        print("MobileNetV2 transfer model validation PASSED.")
    else:
        print("MobileNetV2 transfer model validation FAILED.")
    print("=" * 70)

    return {
        "status": "PASS" if overall_pass else "FAIL",
        "tf_version": tf.__version__,
        "input_shape": model_input_shape,
        "classes": NUM_CLASSES,
        "learning_rate": actual_lr,
        "imagenet_weights": True,
        "augmentation": False,
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
        "weights_unchanged": weights_unchanged,
        "weight_changes": weight_changes,
        "frozen_backbone_pass": frozen_backbone_pass,
        "overall_pass": overall_pass,
    }


if __name__ == "__main__":
    validate_mobilenet_transfer()
