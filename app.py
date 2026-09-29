import io
import json
import logging
import os
from collections.abc import Mapping
from pathlib import Path

from flask import Flask, Request, request, jsonify, render_template
import numpy as np
import tensorflow as tf
from PIL import Image, UnidentifiedImageError
from werkzeug.exceptions import RequestEntityTooLarge

from src.inference.oe_v2_predictor import (
    OEV2Predictor,
    LOCKED_DETECTOR,
    LOCKED_T_UNCERTAIN,
    LOCKED_T_NOT_DOG,
    NOT_DOG_MESSAGE,
    UNCERTAIN_MESSAGE,
    EXPECTED_NUM_CLASSES,
)
from src.inference.preprocessing import preprocess_image as preprocess_inference_image

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
    # Keep the local app host-restricted, but allow the App Engine host when
    # this same package is started by the production WSGI entrypoint.
    TRUSTED_HOSTS=None if os.environ.get("GAE_ENV") else ["127.0.0.1", "localhost"],
)
MAX_IMAGE_PIXELS = 20_000_000  # Local decode budget; Pillow's own limit stays enabled.
IMAGE_SIZE = (224, 224)
NUM_CLASSES = EXPECTED_NUM_CLASSES
ALLOWED_IMAGE_FORMATS = ("JPEG", "PNG", "WEBP")

BASE_DIR = Path(__file__).resolve().parent
DEPLOYMENT_OE_V2_DIR = BASE_DIR / "outputs" / "deployment_oe_v2"
DEPLOYMENT_READY_DIR = BASE_DIR / "outputs" / "deployment_ready"

MODEL_PATH = (
    DEPLOYMENT_OE_V2_DIR / "dog_breed_model_oe_v2.keras"
    if (DEPLOYMENT_OE_V2_DIR / "dog_breed_model_oe_v2.keras").exists()
    else BASE_DIR / "outputs" / "models" / "oe_v2" / "stage_a_best.keras"
)
CENTROIDS_PATH = (
    DEPLOYMENT_OE_V2_DIR / "class_centroids.npy"
    if (DEPLOYMENT_OE_V2_DIR / "class_centroids.npy").exists()
    else BASE_DIR / "outputs" / "calibration" / "oe_v2" / "class_centroids.npy"
)
CLASS_NAMES_PATH = (
    DEPLOYMENT_OE_V2_DIR / "class_names.json"
    if (DEPLOYMENT_OE_V2_DIR / "class_names.json").exists()
    else DEPLOYMENT_READY_DIR / "class_names.json"
)

# ---------------------------------------------------------------------------
# Load Predictor at Startup
# ---------------------------------------------------------------------------
logger.info("Initializing Outlier Exposure V2 predictor...")
oe_v2_predictor = OEV2Predictor(
    model_path=MODEL_PATH,
    centroids_path=CENTROIDS_PATH,
    class_names_path=CLASS_NAMES_PATH,
    verify_integrity=True,
)
predictor = oe_v2_predictor
model = oe_v2_predictor.raw_model
class_names = oe_v2_predictor.class_names
logger.info("Outlier Exposure V2 predictor loaded successfully.")


# ---------------------------------------------------------------------------
# Image Preprocessing Helper
# ---------------------------------------------------------------------------
def preprocess_uploaded_image(file_bytes: bytes) -> np.ndarray:
    """
    Decodes raw uploaded image bytes, converts to RGB, resizes to 224x224,
    and returns a float32 NumPy array with shape (1, 224, 224, 3) in [0, 255].
    
    IMPORTANT:
    MobileNetV2 preprocess_input is already embedded INSIDE the model architecture.
    Do NOT apply / 255.0 or external mobilenet_v2.preprocess_input in this function.
    """
    return preprocess_inference_image(file_bytes, max_image_pixels=MAX_IMAGE_PIXELS)


# ---------------------------------------------------------------------------
# Prediction Helpers
# ---------------------------------------------------------------------------
def predict_image(image_array: np.ndarray):
    """Legacy inference helper returning (top1_dict, top5_list)."""
    res = predictor.predict(image_array)
    top5 = res.get("top5", [])
    if top5:
        top1 = dict(top5[0])
    else:
        top1 = {
            "index": 0,
            "breed": res.get("breed", ""),
            "confidence": float(res.get("confidence") or 0.0),
        }
    return top1, top5


def predict_image_with_ood(image_array: np.ndarray) -> dict:
    """Run model inference and tri-state decision for image array."""
    res = predictor.predict(image_array)
    return _api_prediction_response(res)


def _api_prediction_response(result: Mapping) -> dict:
    """Translate a predictor result into the public OE V2 API contract."""
    if not isinstance(result, Mapping):
        raise TypeError("predictor result must be a mapping")

    state = str(result.get("final_status", result.get("status", result.get("decision", "")))).upper()
    if state not in ("DOG", "UNCERTAIN", "NOT_DOG"):
        raise ValueError(f"unsupported predictor decision: {state}")

    ood_score = float(result.get("raw_ood_score", result.get("ood_score", 0.0)))
    t_uncertain = float(result.get("T_uncertain", result.get("t_uncertain", LOCKED_T_UNCERTAIN)))
    t_not_dog = float(result.get("T_not_dog", result.get("t_not_dog", LOCKED_T_NOT_DOG)))

    # Resolve breed
    breed = result.get("top1_breed", result.get("breed"))
    if breed is None and isinstance(result.get("prediction"), Mapping):
        breed = result["prediction"].get("breed")

    # Resolve confidence
    confidence = result.get("top1_confidence", result.get("confidence"))
    if confidence is None and isinstance(result.get("prediction"), Mapping):
        confidence = result["prediction"].get("confidence")
    confidence = float(confidence) if confidence is not None else 0.0

    # Resolve top5
    top5 = list(result.get("top5_predictions", result.get("top5", [])))

    # Resolve margin
    margin = result.get("probability_margin", result.get("margin"))
    if margin is None:
        if len(top5) >= 2:
            margin = float(top5[0].get("confidence", 0.0) - top5[1].get("confidence", 0.0))
        else:
            margin = float(confidence)
    else:
        margin = float(margin)

    payload = {
        "status": state,
        "final_status": state,
        "decision": state,
        "breed": breed,
        "top1_breed": breed,
        "confidence": confidence,
        "top1_confidence": confidence,
        "top5": top5,
        "top5_predictions": top5,
        "probability_margin": margin,
        "margin": margin,
        "ood_score": ood_score,
        "raw_ood_score": ood_score,
        "T_uncertain": t_uncertain,
        "t_uncertain": t_uncertain,
        "T_not_dog": t_not_dog,
        "t_not_dog": t_not_dog,
        "is_dog": True if state == "DOG" else (None if state == "UNCERTAIN" else False),
        "prediction": {
            "breed": breed,
            "confidence": confidence,
        },
    }

    if state == "UNCERTAIN":
        payload["message"] = result.get("message", UNCERTAIN_MESSAGE)
    elif state == "NOT_DOG":
        payload["message"] = result.get("message", NOT_DOG_MESSAGE)

    if "model" in result:
        payload["model"] = result["model"]
    return payload


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
    """Renders the main web interface for dog breed prediction."""
    return render_template("index.html")


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint. Reports operational status and OE V2 detector metadata."""
    return jsonify({
        "status": "ok",
        "model_loaded": True,
        "ood_detector_loaded": True,
        "num_classes": len(class_names),
        "detector": LOCKED_DETECTOR,
        "model_version": "oe_v2",
    }), 200


@app.route("/predict", methods=["POST"])
def predict():
    """Inference endpoint. Receives multipart/form-data with field name 'image'.

    Performs in-memory preprocessing and model inference through the OE V2
    predictor, returning the public three-state JSON contract.
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
    except (UnidentifiedImageError, ValueError, OSError, SyntaxError,
            Image.DecompressionBombWarning, Image.DecompressionBombError):
        logger.info("Rejected invalid, unsupported, or oversized image.")
        return jsonify({"error": "Invalid or unsupported image"}), 400

    try:
        result = predictor.predict(file_bytes)
    except (UnidentifiedImageError, ValueError, OSError, SyntaxError,
            Image.DecompressionBombWarning, Image.DecompressionBombError):
        logger.info("Rejected invalid, unsupported, or oversized image.")
        return jsonify({"error": "Invalid or unsupported image"}), 400
    except Exception:
        logger.exception("Inference execution failed.")
        return jsonify({"error": "Inference failed"}), 500

    try:
        return jsonify(_api_prediction_response(result)), 200
    except Exception:
        logger.exception("Prediction response formatting failed.")
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
    return jsonify({"error": "Prediction failed. Please try again."}), 500


if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False,
        threaded=False,
        use_reloader=False,
    )
