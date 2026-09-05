# Dog Breed Recognition System — Cloud Deployment Guide

> Historical reference only. This project now targets local personal use on Windows.
> Follow [README.md](README.md) and `run_local.bat`. Cloud configuration and old sizing
> measurements below are retained as history and are not required for local startup.
> Gunicorn has moved to the optional `requirements-cloud.txt`; no cloud action is needed.

## Deployment Target
* **Platform:** Google Cloud Platform (GCP)
* **Service:** Google App Engine (GAE)
* **Environment:** Standard Environment
* **Runtime:** `python311` (Python 3.11)
* **Instance Class:** `F2` (768 MB RAM, 1.2 GHz CPU)
  * *Rationale:* TensorFlow 2.21 + Keras 3.15 + MobileNetV2 model footprint consumes ~399.3 MB RSS on startup and ~495.6 MB RSS upon inference, exceeding the 384 MB limit of F1. F2 ensures stable operation without Out-Of-Memory (OOM) crashes.
* **Entrypoint:** `gunicorn -b :$PORT app:app`

---

## Required Runtime Artifacts
The application requires the following files at runtime:
1. `app.py` — Flask application module defining `app` and REST endpoints (`/`, `/health`, `/predict`).
2. `outputs/deployment_ready/dog_breed_model.keras` — Production MobileNetV2 model artifact (~10.94 MB, SHA256: `4695eae55c8464c1a6be3f147828f901259fb2c2541c23c2e36e8b4eb95d5f3d`).
3. `outputs/deployment_ready/class_names.json` — 120-class mapping index-to-breed.
4. `templates/index.html` — Web UI template.
5. `static/style.css` — Web UI stylesheet.
6. `static/app.js` — Client-side interaction logic.
7. `requirements.txt` — Production Python dependencies.
8. `app.yaml` — App Engine configuration.

---

## Pre-deployment Preparation
Install dependencies in the deployment environment:
```bash
pip install -r requirements-cloud.txt
```

Verify WSGI entrypoint and app initialization:
```bash
python -c "import app; assert app.app is not None; print('Entrypoint OK')"
```

---

## Future Deployment Commands (Documentation Only)
> [!IMPORTANT]
> The following commands are documented for future deployment stages (Task 8.2+).
> **DO NOT execute them during Task 8.1 preparation.**

1. Authenticate with Google Cloud:
   ```bash
   gcloud auth login
   ```

2. Set target Google Cloud project:
   ```bash
   gcloud config set project <PROJECT_ID>
   ```

3. Initialize App Engine application (if not already created in the project):
   ```bash
   gcloud app create --region=<REGION>
   ```

4. Deploy the application:
   ```bash
   gcloud app deploy app.yaml
   ```

5. View the live application:
   ```bash
   gcloud app browse
   ```

---

## Source Packaging & File Inclusion (.gcloudignore)
The `.gcloudignore` file ensures only essential production files are uploaded:
* **Included:** `app.py`, `app.yaml`, `requirements.txt`, `templates/`, `static/`, and `outputs/deployment_ready/`.
* **Excluded:** `.venv/`, `data/` (Kaggle dataset ~1.5 GB), `src/` (training/pipeline scripts), non-production checkpoints/outputs in `outputs/*`, and `.git/`.
* **Total Estimated Upload Size:** ~11.0 MB.
