# Dog Breed Classification — Deployment Guide

The deployable runtime is rooted at the repository root. Runtime paths are resolved relative to `app.py`, so the WSGI service does not depend on the caller's working directory.

## Target

- Platform: Google Cloud App Engine standard environment
- Runtime: Python 3.11
- Instance class: F2
- Entrypoint: `gunicorn -b :$PORT app:app`

The F2 setting is the current project configuration, not a production capacity guarantee. Validate memory, latency, concurrency, and cost in the target Google Cloud project before serving traffic.

## Required Runtime Files

- `app.py`
- `src/inference/` and imported `src/` modules
- `templates/index.html`
- `static/app.js` and `static/style.css`
- `requirements.txt`
- `app.yaml`
- `outputs/deployment_oe_v2/dog_breed_model_oe_v2.keras`
- `outputs/deployment_oe_v2/class_centroids.npy`
- `outputs/deployment_oe_v2/class_names.json`
- `outputs/deployment_oe_v2/model_manifest.json`
- `outputs/deployment_oe_v2/ood_calibration.json`

The active runtime uses a 120-class OE V2 model and cosine distance to 120 class centroids. Startup verifies the model and centroid SHA-256 values and rejects missing, altered, or shape-incompatible artifacts.

The `outputs/deployment_ready/` bundle is retained in the repository as the baseline deployment artifact and provenance record. It is not selected while the active OE V2 bundle is present.

## Local Pre-deployment Checks

```powershell
python -m pip install -r requirements.txt
python -m pip check
python -c "import app; assert app.app is not None; print('Entrypoint OK')"
python -m unittest tests.test_oe_v2_deployment
python -m unittest src.test_deployment_smoke src.test_flask_backend src.test_flask_frontend
```

These checks load the shipped model but do not train or modify it.

## Git LFS

The Keras files are stored through Git LFS. Before deployment from a fresh clone:

```powershell
git lfs install
git lfs pull
git lfs ls-files
```

Do not deploy unresolved LFS pointer files.

## App Engine Deployment

Authenticate and select an existing Google Cloud project:

```powershell
gcloud auth login
gcloud config set project <PROJECT_ID>
```

Create an App Engine application only if the selected project does not already have one:

```powershell
gcloud app create --region=<REGION>
```

Deploy and open the service:

```powershell
gcloud app deploy app.yaml
gcloud app browse
```

Deployment creates billable cloud resources. Review the selected account, project, region, service limits, and model redistribution terms before running these commands.

## Packaging Policy

`.gcloudignore` excludes Git metadata, virtual environments, datasets, caches, logs, temporary audit files, credentials, and non-deployment experiment outputs. It explicitly includes both deployment bundles; the application selects `outputs/deployment_oe_v2/` when present.

Uploaded images are processed in memory and are not written to disk by the Flask application.
