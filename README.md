# Dog Breed Classification

## Overview

This project is a Flask web application backed by a TensorFlow/Keras image classifier for 120 dog breeds. The application accepts an uploaded image, runs a 224×224 MobileNetV2-based model, and applies an out-of-distribution (OOD) detector before returning a result.

## Features

- JPEG, PNG, and WebP image upload
- Top-1 breed prediction and Top-5 candidates
- Browser-based Flask interface
- `GET /health` operational endpoint
- `POST /predict` multipart inference endpoint
- Three-state OOD handling: `DOG`, `UNCERTAIN`, or `NOT_DOG`
- In-memory uploads with a 10 MiB request limit

## Model

The active runtime bundle is `outputs/deployment_oe_v2/`. Its checked-in manifest identifies:

- model version: `oe_v2`
- classifier classes: 120
- input shape: `(None, 224, 224, 3)`
- embedding dimension: 1,280
- OOD detector: cosine distance to class centroids
- decision thresholds: `T_uncertain = 0.343358`, `T_not_dog = 0.457513`

The application verifies the model, centroid, and class-name mapping SHA-256 values recorded in the code before loading them. The model is a single 120-class classifier; OOD handling is a separate inference decision layer, not a 121st class.

### Historical baseline

`outputs/deployment_ready/model_manifest.json` describes the earlier dense MobileNetV2 transfer-learning artifact. It records a 224×224 RGB input, an ImageNet-initialized frozen MobileNetV2 backbone, a 120-class softmax output, validation accuracy of 0.8239, and internal test accuracy of 0.8201. These are historical values from the stored manifest, not newly reproduced results.

### Input preprocessing

Input pixels are resized to 224×224 and passed as `float32` values in `[0, 255]`. MobileNetV2 preprocessing is embedded in the model, so callers must not divide pixels by 255 again.

## Project Structure

```text
.
├── app.py                         # Flask entry point
├── app.yaml                       # Google App Engine configuration
├── requirements.txt               # Runtime dependencies
├── requirements-training.txt      # Offline training/evaluation dependencies
├── configs/                       # ML/OOD configuration
├── metadata/                      # Dataset metadata and reproducibility records
├── src/
│   ├── data/                      # Dataset audit/build utilities
│   ├── inference/                 # Preprocessing and runtime predictors
│   ├── models/                    # Feature-extractor helpers
│   ├── ood/                       # OOD scoring and calibration
│   └── training/                  # Training/evaluation utilities
├── tests/                         # Automated tests and a small image fixture
├── templates/                     # Flask HTML templates
├── static/                        # Browser JavaScript and CSS
└── outputs/
    ├── deployment_oe_v2/          # Active model, centroids, labels, and metadata
    └── deployment_ready/          # Baseline deployment model and metadata
```

The raw and processed datasets, virtual environments, caches, temporary audit files, checkpoints, plots, and other generated experiment outputs are intentionally excluded from Git.

## Requirements

- Python 3.11
- Git LFS for the checked-in Keras model files

Runtime versions are pinned in `requirements.txt`. Offline training and evaluation tools additionally use `requirements-training.txt`.

## Installation

```powershell
git lfs install
git lfs pull
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m pip check
```

## Run Locally

```powershell
python app.py
```

The local server listens at `http://127.0.0.1:5000/`. On Windows, `run_local.bat` starts the same application with the project virtual environment.

## Health Check

```http
GET /health
```

A healthy response reports `status: "ok"`, confirms the model and OOD detector are loaded, and reports 120 classes.

## Prediction

```http
POST /predict
Content-Type: multipart/form-data
```

Send the image in the `image` form field. A successful response includes the three-state decision, OOD score, thresholds, confidence, probability margin, and Top-5 predictions. Invalid/unsupported images return HTTP 400, oversized requests return HTTP 413, and inference failures return HTTP 500 without exposing internal exception details.

## Model Artifacts

Active OE V2 runtime files:

- `outputs/deployment_oe_v2/dog_breed_model_oe_v2.keras`
- `outputs/deployment_oe_v2/class_centroids.npy`
- `outputs/deployment_oe_v2/class_names.json`
- `outputs/deployment_oe_v2/model_manifest.json`
- `outputs/deployment_oe_v2/ood_calibration.json`

Baseline deployment files retained for provenance and compatibility:

- `outputs/deployment_ready/dog_breed_model.keras`
- `outputs/deployment_ready/class_names.json`
- `outputs/deployment_ready/model_manifest.json`
- `outputs/deployment_ready/ood_config.json`
- `outputs/deployment_ready/ood_thresholds.json`

Both class-name files map indices `0` through `119` to breed names.

## Git LFS

Keras and other common ML model formats are configured for Git LFS in `.gitattributes`. Clone with Git LFS installed so the model pointers are replaced by the real binaries.

## Dataset

The image dataset is not uploaded to this repository. Existing project metadata identifies the source as the Kaggle Dog Breed Identification dataset. See `metadata/README.md` for the manifest format. No dataset license is asserted by this repository; verify the source terms before redistributing data or model artifacts.

## Tests

The main local validation commands are:

```powershell
python -m unittest discover -s tests -p "test_*.py"
python -m src.test_deployment_smoke
python -m src.test_flask_backend
python -m src.test_flask_frontend
```

These commands do not retrain the model. Some offline data-pipeline tests may require local datasets or additional training dependencies.

## Deployment

`app.yaml` targets the Google App Engine Python 3.11 runtime and starts the service with:

```text
gunicorn -b :$PORT app:app
```

Review service sizing, artifact redistribution terms, and cloud costs before deploying. The repository is intended to remain private unless those terms have been checked.
