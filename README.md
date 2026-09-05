# Dog Breed Recognition System

A local web application for dog breed classification using MobileNetV2 transfer learning and Flask.

## Overview

The Dog Breed Recognition System is a lightweight, self-contained web application designed for personal, local use on Windows. It allows users to upload an image of a dog through an intuitive browser interface and instantly receive predictions across 120 recognized dog breeds, powered by a pre-trained MobileNetV2 deep learning model.

## Features

- **120 Dog Breeds**: Comprehensive classification across 120 breeds from the Stanford Dogs / Kaggle Dog Breed dataset.
- **MobileNetV2 Transfer Learning**: Efficient, high-accuracy convolutional neural network backbone pre-trained on ImageNet.
- **Flask Backend**: Clean, robust REST API serving endpoints for classification and health checks.
- **Local Browser UI**: Responsive web interface with interactive drag-and-drop or file selection.
- **Top-1 & Top-5 Predictions**: Clear visual display of the highest-confidence breed alongside the top 5 candidates with probability bars.
- **Image Preview**: Immediate client-side image preview before submission, with reset capabilities.
- **Format Support**: Accepts JPEG, PNG, and WEBP images (with automatic EXIF orientation correction).

## Model Performance

The production model recorded the following historical evaluation metrics:

- **Validation Accuracy**: 82.39%
- **Internal Test Accuracy**: 82.01%

*(Note: These metrics reflect evaluation on the held-out validation and test sets; training accuracy reached higher levels during training, but evaluation accuracy is reported above as the true performance benchmark.)*

## Architecture

```text
Input (224x224 RGB)
       │
       ▼
MobileNetV2 (ImageNet Backbone)
       │
       ▼
Global Average Pooling 2D
       │
       ▼
Dropout
       │
       ▼
Dense (120 classes, Softmax)
```

- **Input Shape**: `(None, 224, 224, 3)`
- **Output Shape**: `(None, 120)` with softmax probability distribution

## Preprocessing Contract

Input images are converted to RGB, resized bilinearly to 224×224, and passed to the model as `float32` values in `[0, 255]`.

> [!IMPORTANT]
> **MobileNetV2 preprocessing is already embedded inside the model.**
> **Do NOT divide input pixels by 255 externally.** Doing so will cause double-normalization and corrupt inference predictions.

## Project Structure

```text
Dog-Breed-Recognition/
│
├── app.py                      # Flask backend and local server entrypoint
├── requirements.txt            # Python dependencies for local runtime
├── run_local.bat               # Windows quick-launcher script
├── README.md                   # Project documentation
│
├── templates/
│   └── index.html              # Frontend user interface
│
├── static/
│   ├── style.css               # User interface styles
│   └── app.js                  # Frontend interaction and AJAX client
│
├── outputs/
│   └── deployment_ready/       # Production inference artifacts
│       ├── dog_breed_model.keras  # Trained Keras model (~10.94 MB)
│       ├── class_names.json       # 120-class index-to-name mapping
│       ├── model_manifest.json    # Artifact metadata and integrity specs
│       └── README.txt             # Model provenance notes
│
├── src/                        # Test suite and offline training utilities
│   ├── test_flask_backend.py   # Backend API unit tests
│   ├── test_flask_frontend.py  # Frontend integration tests
│   ├── test_deployment_smoke.py# Smoke test for runtime contract
│   ├── test_local_app.py       # Comprehensive local server validation
│   └── ...                     # Offline training & experiment scripts
│
├── DEPLOYMENT.md               # (Optional) Legacy cloud deployment notes
├── app.yaml                    # (Optional) App Engine configuration
├── .gcloudignore               # (Optional) App Engine ignore rules
└── requirements-cloud.txt      # (Optional) Cloud Gunicorn dependencies
```

## Production Model Artifact

The repository includes the trained inference model directly under `outputs/deployment_ready/`:

- **Path**: `outputs/deployment_ready/dog_breed_model.keras`
- **File Size**: ~10.94 MB (well within standard Git limits)
- **SHA256 Hash**: `4695eae55c8464c1a6be3f147828f901259fb2c2541c23c2e36e8b4eb95d5f3d`
- **Class Mapping**: `outputs/deployment_ready/class_names.json` (120 classes, indices 0–119)

No separate model download is required; the application is ready to run immediately upon cloning.

## Dataset

The training dataset is **intentionally not included** in this repository to keep the repository lightweight, fast to clone, and free of third-party training data.

- **Origin**: Trained on the [Kaggle Dog Breed Identification](https://www.kaggle.com/c/dog-breed-identification) dataset.
- **Local Reproducibility**: The included production model allows full local execution and testing without needing the dataset.

## Installation (Windows)

Prerequisites: **Python 3.11** installed on your system.

1. **Clone the repository:**
   ```powershell
   git clone https://github.com/YPTML/dog-breed-recognition.git
   cd dog-breed-recognition
   ```

2. **Create a virtual environment:**
   ```powershell
   py -3.11 -m venv .venv
   ```

3. **Activate the virtual environment:**
   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```

4. **Install dependencies:**
   ```powershell
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

5. **Verify dependencies:**
   ```powershell
   python -m pip check
   ```

## Running the Application

### Quick Run (Recommended)
Double-click `run_local.bat` in Windows Explorer. It automatically checks for `.venv` and starts the application.

### Command Line
Activate your virtual environment and run `app.py`:

```powershell
python app.py
```
or directly:
```powershell
.\.venv\Scripts\python.exe app.py
```

Then open your browser and navigate to:
```text
http://127.0.0.1:5000/
```

Press `Ctrl+C` in the console window to stop the server.

## Local-Only Configuration

The application is configured for local personal use:
- The server binds strictly to `127.0.0.1:5000`.
- Cross-origin requests from outside `127.0.0.1` and `localhost` are rejected for local security.
- Maximum upload request size is bounded at 10 MiB.
- Uploaded images stay in memory and are never written to disk.
- Cloud deployment files (`app.yaml`, `.gcloudignore`, `DEPLOYMENT.md`, `requirements-cloud.txt`) are optional historical files and are **not required** for local execution.

## Testing & Validation

Run the automated test suite to verify backend and frontend contracts without needing any training dataset:

```powershell
# Backend API & error handling tests
.\.venv\Scripts\python.exe -m src.test_flask_backend

# Frontend UI & asset serving tests
.\.venv\Scripts\python.exe -m src.test_flask_frontend

# Runtime smoke test
.\.venv\Scripts\python.exe -m src.test_deployment_smoke

# Comprehensive local app contract tests
.\.venv\Scripts\python.exe -m src.test_local_app
```

Verify model file integrity at any time:
```powershell
Get-FileHash -Algorithm SHA256 outputs/deployment_ready/dog_breed_model.keras
```
Expected output: `4695EAE55C8464C1A6BE3F147828F901259FB2C2541C23C2E36E8B4EB95D5F3D`

## Limitations & Known Boundaries

- **Closed-Set Classification**: The model classifies inputs strictly among the 120 known breeds. It does not determine whether an image is a dog (no Open Set Recognition).
- **Image Quality**: Prediction accuracy may degrade on heavily blurred images, low-light or very dark photos, extreme angles, or occluded subjects.
- **Non-Dog Images**: Submitting objects, humans, or unsupported breeds will still return probabilities distributed across the 120 breeds.

## Model Sharing & Legal Notice

- The model weights were trained using the Kaggle Dog Breed Identification dataset. Verify dataset and model redistribution terms before making the repository public.
- **License**: A formal LICENSE file is currently not included. If you plan to make this repository public, please select and add an appropriate open-source license (such as MIT or Apache-2.0).

