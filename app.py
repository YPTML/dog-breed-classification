import io
import json
import logging
from pathlib import Path
import warnings

from flask import Flask, Request, request, jsonify, render_template
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps, UnidentifiedImageError
from werkzeug.exceptions import RequestEntityTooLarge

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("dog_breed_backend")

# ---------------------------------------------------------------------------
# App Initialization & Configuration
# ---------------------------------------------------------------------------
class InMemoryRequest(Request):
    """Keep bounded uploads in RAM, including files above Werkzeug's spool limit."""

    def _get_file_stream(self, total_content_length, content_type,
                         filename=None, content_length=None):
        return io.BytesIO()


app = Flask(__name__)
app.request_class = InMemoryRequest
app.config.update(
    MAX_CONTENT_LENGTH=10 * 1024 * 1024,  # Entire multipart request, not just file bytes.
    MAX_FORM_MEMORY_SIZE=128 * 1024,
    MAX_FORM_PARTS=10,
    TRUSTED_HOSTS=["127.0.0.1", "localhost"],
)
MAX_IMAGE_PIXELS = 20_000_000  # Local decode budget; Pillow's own limit stays enabled.
IMAGE_SIZE = (224, 224)
NUM_CLASSES = 120
ALLOWED_IMAGE_FORMATS = ("JPEG", "PNG", "WEBP")

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "outputs" / "deployment_ready" / "dog_breed_model.keras"
CLASS_NAMES_PATH = BASE_DIR / "outputs" / "deployment_ready" / "class_names.json"

# ---------------------------------------------------------------------------
# Load Class Mapping
# ---------------------------------------------------------------------------
if not CLASS_NAMES_PATH.exists():
    raise RuntimeError(f"Class names mapping not found at {CLASS_NAMES_PATH}")

def unique_json_object(pairs):
    """Reject duplicate JSON keys rather than silently discarding a mapping entry."""
    result = {}
    for key, value in pairs:
        if key in result:
            raise RuntimeError("Duplicate class mapping index.")
        result[key] = value
    return result


with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as f:
    raw_class_mapping = json.load(f, object_pairs_hook=unique_json_object)

if isinstance(raw_class_mapping, dict):
    if set(raw_class_mapping) != {str(i) for i in range(NUM_CLASSES)}:
        raise RuntimeError("Class mapping must use canonical indices 0..119.")
    class_names = {int(k): v for k, v in raw_class_mapping.items()}
elif isinstance(raw_class_mapping, list):
    class_names = dict(enumerate(raw_class_mapping))
else:
    raise RuntimeError(
        f"Unsupported format in {CLASS_NAMES_PATH}: must be dict or list."
    )

# Validate class mapping integrity: exactly 120 classes spanning indices 0..119 and 120 unique breeds
if len(class_names) != NUM_CLASSES or set(class_names) != set(range(NUM_CLASSES)):
    raise RuntimeError(
        f"Invalid class mapping: expected 120 consecutive classes (0..119), "
        f"got {len(class_names)} classes."
    )

if any(not isinstance(name, str) or not name.strip() for name in class_names.values()):
    raise RuntimeError("Class mapping must contain non-empty breed names.")

if len(set(class_names.values())) != NUM_CLASSES:
    raise RuntimeError(
        f"Invalid class mapping: expected 120 unique breed names, "
        f"got {len(set(class_names.values()))} unique breeds."
    )

logger.info(f"Loaded {len(class_names)} classes from {CLASS_NAMES_PATH.name}")

# ---------------------------------------------------------------------------
# Load Model Once at Startup
# ---------------------------------------------------------------------------
if not MODEL_PATH.exists():
    raise RuntimeError(f"Production model file not found at {MODEL_PATH}")

logger.info("Loading production model...")
model = tf.keras.models.load_model(
    MODEL_PATH,
    custom_objects={"preprocess_input": tf.keras.applications.mobilenet_v2.preprocess_input},
    compile=False,  # Inference needs weights, not optimizer or training configuration.
    safe_mode=True,
)
logger.info("Production model successfully loaded.")

# Startup Validation
EXPECTED_MODEL_NAME = "mobilenet_v2_transfer"
EXPECTED_INPUT_SHAPE = (None, 224, 224, 3)
EXPECTED_OUTPUT_SHAPE = (None, 120)

if model.name != EXPECTED_MODEL_NAME:
    raise RuntimeError(
        f"Model name mismatch: expected '{EXPECTED_MODEL_NAME}', got '{model.name}'"
    )

if model.input_shape != EXPECTED_INPUT_SHAPE:
    raise RuntimeError(
        f"Input shape mismatch: expected {EXPECTED_INPUT_SHAPE}, got {model.input_shape}"
    )

if model.output_shape != EXPECTED_OUTPUT_SHAPE:
    raise RuntimeError(
        f"Output shape mismatch: expected {EXPECTED_OUTPUT_SHAPE}, got {model.output_shape}"
    )

if model.output_shape[-1] != len(class_names):
    raise RuntimeError(
        f"Model output dimension ({model.output_shape[-1]}) does not match "
        f"class mapping size ({len(class_names)})"
    )

logger.info(
    f"Startup validation passed: name={model.name}, "
    f"input_shape={model.input_shape}, output_shape={model.output_shape}"
)

# ---------------------------------------------------------------------------
# Image Preprocessing Helper
# ---------------------------------------------------------------------------
def preprocess_uploaded_image(file_bytes: bytes) -> np.ndarray:
    """
    Decodes raw uploaded image bytes, converts to RGB, resizes to 224x224,
    and returns a float32 NumPy array with shape (1, 224, 224, 3) in [0, 255].
    Verify structure and fully decode the first frame; reject oversized dimensions.
    
    IMPORTANT:
    MobileNetV2 preprocess_input is already embedded INSIDE the model architecture.
    Do NOT apply / 255.0 or external mobilenet_v2.preprocess_input in this function.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("error", Image.DecompressionBombWarning)
        with Image.open(io.BytesIO(file_bytes), formats=ALLOWED_IMAGE_FORMATS) as source:
            if source.width * source.height > MAX_IMAGE_PIXELS:
                raise ValueError("Image dimensions exceed the local decode limit.")
            source.verify()
        # verify() invalidates the decoder; reopening is required for full decoding.
        with Image.open(io.BytesIO(file_bytes), formats=ALLOWED_IMAGE_FORMATS) as source:
            source.load()
            ImageOps.exif_transpose(source, in_place=True)
            with source.convert("RGB") as rgb:
                with rgb.resize(IMAGE_SIZE, Image.Resampling.BILINEAR) as resized:
                    return np.asarray(resized, dtype=np.float32)[None, ...]

# ---------------------------------------------------------------------------
# Prediction Helper
# ---------------------------------------------------------------------------
def predict_image(image_array: np.ndarray):
    """
    Executes inference using the preloaded model on the batch array (1, 224, 224, 3).
    Verifies absence of NaNs/Infs and validates probability sum ~ 1.0.
    Returns (top1_dict, top5_list).
    """
    if (not isinstance(image_array, np.ndarray)
            or image_array.shape != (1, *IMAGE_SIZE, 3)
            or image_array.dtype != np.float32):
        raise ValueError("Expected float32 image batch with shape (1, 224, 224, 3).")
    if (not np.isfinite(image_array).all()
            or np.any(image_array < 0) or np.any(image_array > 255)):
        raise ValueError("Image pixels must be finite values in [0, 255].")

    predictions = np.asarray(model(image_array, training=False))
    if predictions.shape != (1, NUM_CLASSES):
        raise RuntimeError("Model prediction shape must be (1, 120).")
    if not np.isfinite(predictions).all():
        raise RuntimeError("Model predictions contain non-finite values.")
    if np.any(predictions < 0) or np.any(predictions > 1):
        raise RuntimeError("Model probabilities must lie in [0, 1].")

    prob_sum = float(np.sum(predictions[0]))
    if abs(prob_sum - 1.0) > 1e-3:
        raise RuntimeError(
            f"Invalid probability distribution: softmax sum is {prob_sum:.6f}, expected approx 1.0"
        )

    # Stable descending order makes ties deterministic and Top-1 match Top-5[0].
    top5_indices = np.argsort(-predictions[0], kind="stable")[:5]
    top5 = [
        {
            "index": int(idx),
            "breed": class_names[int(idx)],
            "confidence": float(predictions[0][idx]),
        }
        for idx in top5_indices
    ]

    return dict(top5[0]), top5

# ---------------------------------------------------------------------------
# Flask Routes & Error Handlers
# ---------------------------------------------------------------------------
@app.before_request
def require_local_origin():
    """Prevent other websites from submitting browser inference requests to localhost."""
    if request.path == "/predict" and request.method == "POST":
        origin = request.headers.get("Origin")
        if ((origin is not None and origin != request.host_url.rstrip("/"))
                or request.headers.get("Sec-Fetch-Site") == "cross-site"):
            return jsonify({"error": "Please use the local application page."}), 403


@app.after_request
def add_response_headers(response):
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; img-src 'self' blob: data:; "
        "style-src 'self'; script-src 'self'; connect-src 'self'; "
        "object-src 'none'; base-uri 'none'; frame-ancestors 'none'; form-action 'self'"
    )
    if request.path in ("/health", "/predict"):
        response.headers["Cache-Control"] = "no-store"
    return response


@app.route("/", methods=["GET"])
def index():
    """
    Renders the main web interface for dog breed prediction.
    """
    return render_template("index.html")


@app.route("/health", methods=["GET"])
def health():
    """
    Health check endpoint. Confirms model is loaded and backend is operational.
    Does not run inference.
    """
    return jsonify({
        "status": "ok",
        "model_loaded": True,
        "model_name": model.name,
        "num_classes": len(class_names),
    }), 200


@app.route("/predict", methods=["POST"])
def predict():
    """
    Inference endpoint. Receives multipart/form-data with field name 'image'.
    Performs in-memory preprocessing and model inference, returning JSON predictions.
    """
    if "image" not in request.files:
        return jsonify({"error": "Missing image file"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty image file"}), 400

    try:
        file_bytes = file.read()
        if not file_bytes:
            return jsonify({"error": "Empty image file"}), 400

        image_array = preprocess_uploaded_image(file_bytes)
    except (UnidentifiedImageError, ValueError, OSError, SyntaxError,
            Image.DecompressionBombWarning, Image.DecompressionBombError):
        logger.info("Rejected invalid, unsupported, or oversized image.")
        return jsonify({"error": "Invalid or unsupported image"}), 400

    try:
        top1, top5 = predict_image(image_array)
        return jsonify({
            "model": model.name,
            "prediction": top1,
            "top5": top5,
        }), 200
    except Exception:
        logger.exception("Inference execution failed.")
        return jsonify({"error": "Inference failed"}), 500


@app.errorhandler(RequestEntityTooLarge)
def handle_file_too_large(e):
    """Handles uploads exceeding MAX_CONTENT_LENGTH (10 MB)."""
    return jsonify({"error": "File too large"}), 413


@app.errorhandler(400)
def handle_bad_request(error):
    return jsonify({"error": "Invalid request"}), 400


@app.errorhandler(500)
def handle_internal_error(error):
    # Flask records the exception server-side; never return its text to the client.
    return jsonify({"error": "Prediction failed. Please try again."}), 500


if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False,
        threaded=False,  # One personal user; bound simultaneous decode/inference memory.
        use_reloader=False,
    )
