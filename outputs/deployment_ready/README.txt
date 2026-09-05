================================================================================
Dog Breed Recognition System - Deployment Artifact (Task 6.3)
================================================================================

Production Model:
  File: dog_breed_model.keras
  Architecture: MobileNetV2 Frozen Transfer Learning
  Backbone: MobileNetV2 (ImageNet pretrained, frozen feature extractor)
  Head: GlobalAveragePooling2D -> Dropout(0.5) -> Dropout(0.2) -> Dense(120, softmax)
  Total Parameters: 2,411,704
  Trainable Parameters: 153,720 (dense classifier kernel + bias)
  Non-trainable Parameters: 2,257,984 (frozen MobileNetV2 backbone)

Performance:
  Validation Accuracy: 82.39%
  Internal Test Accuracy: 82.01%

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
  - Original Validation Accuracy: 82.39%
  - Pruned Validation Accuracy: 80.43% (Drop: 1.96 percentage points)
  - Pruned ZIP Size Reduction: 2.39%
  Decision: The pruned model was REJECTED for production due to accuracy degradation
  with negligible size benefits. The production artifact is the original, dense,
  unpruned MobileNetV2 transfer learning model. Compression artifacts are preserved in
  outputs/compression_manual/ for audit and experimental evidence.
================================================================================
