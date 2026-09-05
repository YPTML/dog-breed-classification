"""Basic CNN Baseline Model for Dog Breed Identification.

This module constructs, compiles, and verifies the baseline Convolutional
Neural Network (CNN) for 120-class dog breed classification. It includes
an in-graph Rescaling layer for pixel normalization (0..255 -> 0..1),
GlobalAveragePooling2D to prevent parameter explosion, and Softmax output.
Training and evaluation are strictly deferred to subsequent tasks.
"""

from typing import Any, Dict, List, Optional, Tuple
import math
import numpy as np
import tensorflow as tf

# Model and image configurations
IMAGE_HEIGHT: int = 224
IMAGE_WIDTH: int = 224
IMAGE_CHANNELS: int = 3
NUM_CLASSES: int = 120
INPUT_SHAPE: Tuple[int, int, int] = (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)


def build_basic_cnn(
    input_shape: Tuple[int, int, int] = INPUT_SHAPE,
    num_classes: int = NUM_CLASSES,
) -> tf.keras.Model:
    """Construct the baseline CNN model using the Keras Functional API.

    Architecture:
    -------------
    - Input: shape (224, 224, 3)
    - Rescaling(1.0 / 255): internal normalization from [0, 255] to [0.0, 1.0]
    - Conv2D(32, 3x3, relu, padding='same') -> MaxPooling2D(2x2)
    - Conv2D(64, 3x3, relu, padding='same') -> MaxPooling2D(2x2)
    - Conv2D(128, 3x3, relu, padding='same') -> MaxPooling2D(2x2)
    - GlobalAveragePooling2D: reduces spatial dimensions without parameter bloat
    - Dense(128, relu)
    - Dropout(0.3)
    - Dense(num_classes, softmax, name='predictions')

    Parameters
    ----------
    input_shape : Tuple[int, int, int], default=(224, 224, 3)
        Dimensions (height, width, channels) of input image tensors.
    num_classes : int, default=120
        Number of output classification categories.

    Returns
    -------
    tf.keras.Model
        Uncompiled Keras Model instance.
    """
    inputs = tf.keras.Input(shape=input_shape, name="input_image")

    # In-graph normalization layer: keeps preprocessing reusable for MobileNetV2
    x = tf.keras.layers.Rescaling(1.0 / 255.0, name="rescaling")(inputs)

    # Block 1
    x = tf.keras.layers.Conv2D(
        filters=32,
        kernel_size=3,
        activation="relu",
        padding="same",
        name="conv2d_1",
    )(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), name="max_pooling2d_1")(x)

    # Block 2
    x = tf.keras.layers.Conv2D(
        filters=64,
        kernel_size=3,
        activation="relu",
        padding="same",
        name="conv2d_2",
    )(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), name="max_pooling2d_2")(x)

    # Block 3
    x = tf.keras.layers.Conv2D(
        filters=128,
        kernel_size=3,
        activation="relu",
        padding="same",
        name="conv2d_3",
    )(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), name="max_pooling2d_3")(x)

    # Spatial dimension reduction via pooling instead of Flatten
    x = tf.keras.layers.GlobalAveragePooling2D(name="global_average_pooling2d")(x)

    # Dense classification head
    x = tf.keras.layers.Dense(128, activation="relu", name="dense_128")(x)
    x = tf.keras.layers.Dropout(0.3, name="dropout")(x)

    # Softmax output layer
    outputs = tf.keras.layers.Dense(
        num_classes,
        activation="softmax",
        name="predictions",
    )(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="basic_cnn")
    return model


def compile_basic_cnn(
    model: tf.keras.Model,
    learning_rate: float = 0.001,
) -> tf.keras.Model:
    """Compile the baseline CNN model with Adam optimizer and categorical crossentropy loss.

    Parameters
    ----------
    model : tf.keras.Model
        Uncompiled or compiled Keras model.
    learning_rate : float, default=0.001
        Learning rate for the Adam optimizer.

    Returns
    -------
    tf.keras.Model
        Compiled Keras model instance.
    """
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def create_compiled_basic_cnn(
    input_shape: Tuple[int, int, int] = INPUT_SHAPE,
    num_classes: int = NUM_CLASSES,
    learning_rate: float = 0.001,
) -> tf.keras.Model:
    """Build and compile the baseline CNN in a single call.

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
        Compiled baseline CNN model ready for inspection or subsequent training.
    """
    model = build_basic_cnn(input_shape=input_shape, num_classes=num_classes)
    model = compile_basic_cnn(model, learning_rate=learning_rate)
    return model


def validate_basic_cnn() -> Dict[str, Any]:
    """Execute validation suite for Task 2.2 and print structured summary.

    Returns
    -------
    Dict[str, Any]
        Dictionary of all validation metrics and verification flags.
    """
    from src.dataset_pipeline import build_datasets

    print("=" * 70)
    print("BASIC CNN VALIDATION")
    print("=" * 70)

    # 1. Model construction
    model = create_compiled_basic_cnn()

    model_name = model.name
    input_shape_tuple = tuple(model.input_shape[1:]) if model.input_shape[0] is None else tuple(model.input_shape)
    output_shape_tuple = tuple(model.output_shape)
    num_classes = output_shape_tuple[-1]

    trainable_params = sum(
        int(tf.size(w).numpy()) for w in model.trainable_weights
    )
    non_trainable_params = sum(
        int(tf.size(w).numpy()) for w in model.non_trainable_weights
    )
    total_params = trainable_params + non_trainable_params

    print(f"Input shape: {input_shape_tuple}")
    print(f"Classes: {num_classes}\n")
    print(f"Model name: {model_name}")
    print(f"Output shape: {output_shape_tuple}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {non_trainable_params:,}\n")

    # 2. Layer structure verification
    layer_info: List[Dict[str, Any]] = []
    print("Layer architecture:")
    print(f"{'#':>3} | {'Layer':<24} | {'Output shape':<22} | {'Parameters':>10}")
    print("-" * 68)
    for idx, layer in enumerate(model.layers):
        layer_params = layer.count_params()
        if hasattr(layer, "output") and hasattr(layer.output, "shape"):
            out_shape = tuple(layer.output.shape)
        else:
            out_shape = getattr(layer, "output_shape", getattr(layer, "batch_shape", None))
        layer_info.append({
            "index": idx + 1,
            "name": layer.name,
            "type": layer.__class__.__name__,
            "output_shape": str(out_shape),
            "params": layer_params,
        })
        layer_display = f"{layer.name} ({layer.__class__.__name__})"
        print(f"{idx+1:>3} | {layer_display:<30} | {str(out_shape):<22} | {layer_params:>10,}")

    expected_layer_classes = [
        "InputLayer",
        "Rescaling",
        "Conv2D",
        "MaxPooling2D",
        "Conv2D",
        "MaxPooling2D",
        "Conv2D",
        "MaxPooling2D",
        "GlobalAveragePooling2D",
        "Dense",
        "Dropout",
        "Dense",
    ]
    actual_layer_classes = [layer.__class__.__name__ for layer in model.layers]
    layers_ordered = (actual_layer_classes == expected_layer_classes)

    # 3. Normalization layer verification
    rescaling_layer = model.get_layer("rescaling")
    test_pixels = tf.constant([0.0, 127.5, 255.0], dtype=tf.float32)
    normalized_pixels = rescaling_layer(test_pixels).numpy()
    expected_normalized = np.array([0.0, 0.5, 1.0], dtype=np.float32)
    norm_pass = bool(np.allclose(normalized_pixels, expected_normalized, atol=1e-5))

    print("\nNormalization: Rescaling(1/255)")
    print(f"  Pixel 0.0   -> {normalized_pixels[0]:.4f} (expected 0.0)")
    print(f"  Pixel 127.5 -> {normalized_pixels[1]:.4f} (expected 0.5)")
    print(f"  Pixel 255.0 -> {normalized_pixels[2]:.4f} (expected 1.0)")
    print(f"  Normalization check: {'PASS' if norm_pass else 'FAIL'}\n")

    # 4. Compiler check
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
        and math.isclose(actual_lr, 0.001, rel_tol=1e-5)
        and loss_name == "categorical_crossentropy"
        and metrics_matched
    )

    print(f"Optimizer: {optimizer_name}")
    print(f"Learning rate: {actual_lr}")
    print(f"Loss: {loss_name}")
    print("Metric: accuracy")
    print(f"Compilation check: {'PASS' if compile_pass else 'FAIL'}\n")

    # 5. Synthetic forward pass
    synthetic_x = tf.random.uniform(
        shape=(2, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS),
        minval=0,
        maxval=255,
        dtype=tf.float32,
    )
    synthetic_y = model(synthetic_x, training=False)
    synthetic_shape_pass = bool(synthetic_y.shape == (2, num_classes))

    synthetic_prob_sums = tf.reduce_sum(synthetic_y, axis=1).numpy()
    synthetic_min_prob = float(tf.reduce_min(synthetic_y).numpy())
    synthetic_max_prob = float(tf.reduce_max(synthetic_y).numpy())
    synthetic_nan_count = int(tf.reduce_sum(tf.cast(tf.math.is_nan(synthetic_y), tf.int32)).numpy())
    synthetic_inf_count = int(tf.reduce_sum(tf.cast(tf.math.is_inf(synthetic_y), tf.int32)).numpy())

    synthetic_prob_pass = bool(
        np.allclose(synthetic_prob_sums, [1.0, 1.0], atol=1e-5)
        and synthetic_min_prob >= 0.0
        and synthetic_max_prob <= 1.0
        and synthetic_nan_count == 0
        and synthetic_inf_count == 0
    )

    synthetic_pass = (synthetic_shape_pass and synthetic_prob_pass)

    # 6. Real train batch forward pass & non-updating loss check
    # Record initial weights to confirm no training occurs
    initial_weights = [w.numpy().copy() for w in model.weights]

    train_ds, val_ds, test_ds, encoder = build_datasets()
    real_images, real_labels = next(iter(train_ds))

    real_preds = model(real_images, training=False)

    real_batch_shape_pass = bool(
        real_images.shape == (64, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)
        and real_labels.shape == (64, num_classes)
        and real_preds.shape == (64, num_classes)
    )

    real_prob_sums = tf.reduce_sum(real_preds, axis=1).numpy()
    real_min_sum = float(np.min(real_prob_sums))
    real_max_sum = float(np.max(real_prob_sums))
    real_nan_count = int(tf.reduce_sum(tf.cast(tf.math.is_nan(real_preds), tf.int32)).numpy())
    real_inf_count = int(tf.reduce_sum(tf.cast(tf.math.is_inf(real_preds), tf.int32)).numpy())

    real_softmax_pass = bool(
        np.allclose(real_prob_sums, 1.0, atol=1e-5)
        and real_nan_count == 0
        and real_inf_count == 0
    )

    # Compute loss on batch without updating weights
    cce_loss_fn = tf.keras.losses.CategoricalCrossentropy()
    batch_loss_tensor = cce_loss_fn(real_labels, real_preds)
    batch_loss = float(batch_loss_tensor.numpy())
    loss_finite_pass = bool(math.isfinite(batch_loss) and not math.isnan(batch_loss))

    # Verify weights remain completely unmodified
    weights_unchanged = bool(all(
        np.array_equal(w_init, w_now.numpy())
        for w_init, w_now in zip(initial_weights, model.weights)
    ))

    print(f"Synthetic forward pass: {'PASS' if synthetic_pass else 'FAIL'}")
    print(f"Real train batch forward pass: {'PASS' if real_batch_shape_pass else 'FAIL'}")
    print(f"Softmax validation: {'PASS' if real_softmax_pass else 'FAIL'}")
    print(f"Finite loss: {'PASS' if loss_finite_pass else 'FAIL'}")
    print(f"Weights unchanged: {'YES' if weights_unchanged else 'NO'}\n")

    print("Training performed: NO")
    print("Test evaluation performed: NO\n")

    overall_pass = bool(
        layers_ordered
        and norm_pass
        and compile_pass
        and synthetic_shape_pass
        and synthetic_prob_pass
        and real_batch_shape_pass
        and real_softmax_pass
        and loss_finite_pass
        and weights_unchanged
    )

    if overall_pass:
        print("Basic CNN validation completed successfully.")
    else:
        print("Basic CNN validation FAILED.")
    print("=" * 70)

    return {
        "status": "PASS" if overall_pass else "FAIL",
        "model_name": model_name,
        "input_shape": input_shape_tuple,
        "output_shape": output_shape_tuple,
        "num_classes": num_classes,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "non_trainable_params": non_trainable_params,
        "layers": layer_info,
        "layers_ordered": layers_ordered,
        "norm_pass": norm_pass,
        "pixel_0": float(normalized_pixels[0]),
        "pixel_127_5": float(normalized_pixels[1]),
        "pixel_255": float(normalized_pixels[2]),
        "optimizer_name": optimizer_name,
        "learning_rate": actual_lr,
        "loss_name": loss_name,
        "metrics_name": "accuracy",
        "compile_pass": compile_pass,
        "synthetic_shape_pass": synthetic_shape_pass,
        "synthetic_prob_sums": synthetic_prob_sums.tolist(),
        "synthetic_prob_sum_min": float(np.min(synthetic_prob_sums)),
        "synthetic_prob_sum_max": float(np.max(synthetic_prob_sums)),
        "synthetic_nan_count": synthetic_nan_count,
        "synthetic_inf_count": synthetic_inf_count,
        "real_img_shape": tuple(real_images.shape),
        "real_lbl_shape": tuple(real_labels.shape),
        "real_pred_shape": tuple(real_preds.shape),
        "real_img_dtype": str(real_images.dtype),
        "real_lbl_dtype": str(real_labels.dtype),
        "real_pred_dtype": str(real_preds.dtype),
        "real_prob_sum_min": real_min_sum,
        "real_prob_sum_max": real_max_sum,
        "real_nan_count": real_nan_count,
        "real_inf_count": real_inf_count,
        "batch_loss": batch_loss,
        "loss_finite": loss_finite_pass,
        "weights_unchanged": weights_unchanged,
        "overall_pass": overall_pass,
    }


if __name__ == "__main__":
    validate_basic_cnn()
