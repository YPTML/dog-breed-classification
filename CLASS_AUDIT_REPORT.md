# Class Inventory Audit Report (Phase 1)

**Date**: 2026-09-29  
**Audit Target**: Dog Breed Classifier Class Inventory, Dataset Assets, Split Configurations, and Deployment Manifests  
**Status**: AUDIT COMPLETE — PROCEED TO PHASE 2  

---

## 1. Executive Summary

An exhaustive inventory audit was conducted across raw data directories (`data/`), curated extensions (`data/new_breeds_v3/`), existing split configurations (`outputs/splits_v2/`, `outputs/splits_v3/`), production training pipelines (`src/training/`), and deployed runtime artifacts (`outputs/deployment_oe_v2/`, `outputs/deployment_ready/`).

The audit confirms the structural root cause of the Shiba Inu classification failure:
- The deployed production classifier (`dog_breed_model_oe_v2.keras`) is strictly a **120-class softmax classifier** mapped to the Stanford Dogs taxonomy (`0..119`).
- `shiba_inu` is **completely absent** from the production model weights, `outputs/deployment_oe_v2/class_names.json`, and the deployment manifest.
- The active training pipeline (`src/training/train_oe.py`) is hardcoded to 120 classes and consumes only `outputs/splits_v2/`, completely ignoring `data/new_breeds_v3/`.
- `data/new_breeds_v3/shiba_inu` contains **200 valid, non-corrupt RGB images** with verified deterministic splits (140 train, 30 val, 30 test), providing more than sufficient data to train and evaluate `shiba_inu` as a canonical class.

---

## 2. Core Inventory Metrics

| Audit Field | Value | Notes |
|:---|:---:|:---|
| **original_classes** | **120** | Stanford Dogs dataset classes (indices `0` to `119`) |
| **new_classes** | **4** | `american_bulldog`, `american_pit_bull_terrier`, `havanese`, `shiba_inu` |
| **expected_total_classes** | **124** | 120 historical + 4 curated new breeds |
| **deployed_classes** | **120** | Active in `outputs/deployment_oe_v2/class_names.json` |
| **missing_classes** | **4** | `['american_bulldog', 'american_pit_bull_terrier', 'havanese', 'shiba_inu']` |

---

## 3. Dataset & Split Inspection

### 3.1 Original Dataset (`data/train`, `outputs/splits_v2`)
- **Total historical samples**: 10,216 images across 120 Stanford Dogs breeds.
- **Historical split distribution**:
  - `train`: 8,181 images (~68.2 images/class)
  - `val`: 1,018 images (~8.5 images/class)
  - `test`: 1,017 images (~8.5 images/class)

### 3.2 Candidate Extension (`data/new_breeds_v3`)
Curated from non-overlapping dog classes in Oxford-IIIT Pet:
1. `american_bulldog`: 200 files (200 valid RGB, 0 corrupt)
2. `american_pit_bull_terrier`: 200 files (200 valid RGB, 0 corrupt)
3. `havanese`: 200 files (200 valid RGB, 0 corrupt)
4. `shiba_inu`: 200 files (200 valid RGB, 0 corrupt)

### 3.3 Existing V3 Splits (`outputs/splits_v3`)
- **Total samples**: 11,016 images across 124 classes.
- **Splits breakdown**:
  - `train.csv`: 8,741 images (8,181 historical + 560 new [4 breeds $\times$ 140])
  - `val.csv`: 1,138 images (1,018 historical + 120 new [4 breeds $\times$ 30])
  - `test.csv`: 1,137 images (1,017 historical + 120 new [4 breeds $\times$ 30])
- **Shiba Inu distribution**:
  - `train`: 140 images
  - `val`: 30 images
  - `test`: 30 images
  - **Total**: 200 images

---

## 4. Pipeline Consumption Analysis

- **`src/training/train_oe.py`**:
  - Hardcoded parameter: `NUM_CLASSES = 120`
  - Input split path: `SPLITS_V2_DIR = PROJECT_ROOT / "outputs/splits_v2"`
  - Consumption of `new_breeds_v3`: **FALSE (0 references)**
  - Consumption of `splits_v3`: **FALSE (0 references)**
- **Active Deployment Manifest (`outputs/deployment_oe_v2/model_manifest.json`)**:
  - `class_count`: 120
  - `embedding_dimension`: 1280
  - `class_names_file`: `class_names.json` (120 entries)
  - `centroids_file`: `class_centroids.npy` (shape: `(120, 1280)`)

**Conclusion**: The production training and deployment pipelines have never incorporated `data/new_breeds_v3` or `shiba_inu`. Previous exploration of V3 occurred strictly within temporary scratch scripts in `_audit_tmp/`.

---

## 5. Shiba Inu Data Validation & Quality Assessment

A full integrity scan of all 200 `shiba_inu` images in `data/new_breeds_v3/shiba_inu/` was performed:
- **Image integrity**: 200 / 200 images successfully verified via PIL (0 corrupt, 0 unreadable).
- **Color mode**: 100% standard 3-channel RGB.
- **Resolution**: Variable natural photography resolutions ranging from $221 \times 306$ to $500 \times 500$ (all compatible with MobileNetV2 $224 \times 224$ bicubic preprocessing).
- **Cross-split hash collision**: SHA256 hashes verified across `train`, `val`, and `test`; **zero duplicate hashes** exist across split boundaries.
- **Sample adequacy**: At 140 train / 30 val / 30 test images, `shiba_inu` has **more than double** the sample count of the average Stanford Dogs class (~68 train / ~8 val / ~8 test), guaranteeing sufficient statistical support for gradient descent and robust evaluation.

---

## 6. Audit Gate Decision

- [x] Exact count of original classes verified ($120$)
- [x] Exact count of candidate new breeds verified ($4$)
- [x] Presence of `shiba_inu` in candidate splits verified ($140 / 30 / 30$)
- [x] Production pipeline non-consumption of `new_breeds_v3` confirmed
- [x] 100% of `shiba_inu` images verified uncorrupted and valid
- [x] Hard requirement met: Sufficient data confirmed prior to training

**GATE DECISION**: **PROCEED TO PHASE 2 (DATASET INTEGRATION)**
