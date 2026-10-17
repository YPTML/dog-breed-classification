# Full-pipeline OOD fixture coverage

Result: **PASS** for test-data integrity and the unchanged runtime contract. Semantic OOD decisions are reported separately below.

Full-pipeline OOD skips: **5 before -> 0 after**. Added **32** real, visually reviewed non-dog photographs, selected before inference.

## Fixture locations

Paths are relative to this repository.

| Group | Images | Directories |
|---|---:|---|
| other animals | 6 | `data/ood/test/bird/`, `data/ood/test/cat/`, `data/ood/test/elephant/`, `data/ood/test/horse/`, `data/ood/test/rabbit/`, `data/ood/test/sheep/` |
| humans | 6 | `data/ood/test/humans/` |
| vehicles | 6 | `data/ood/test/bicycle/`, `data/ood/test/bus/`, `data/ood/test/car/`, `data/ood/test/motorcycle/` |
| household objects | 6 | `data/ood/test/bottle/`, `data/ood/test/chair/`, `data/ood/test/laptop/`, `data/ood/test/phone/` |
| hard OOD | 8 | `data/ood_hard/coyote/`, `data/ood_hard/fox/`, `data/ood_hard/hyena/`, `data/ood_hard/wolf/` |

Every filename, SHA-256, decoded-pixel hash, perceptual hash, source URL, author, license, and dimension is in [the fixture inventory](metadata/full_pipeline_ood_fixtures.csv). [The standard OOD manifest](metadata/full_pipeline_ood_manifest.csv) uses the existing builder's fields and layout validator. It is separate from the pre-existing ID/training manifest, which was preserved.

Raw datasets remain local under the repository's existing `/data/` ignore rule. Source URLs and content hashes in the inventory document restoration; no image is a link to a source or training file.

## Integrity

- Compared against 28,800 physical ID/source images (27,881 distinct SHA-256 hashes), including `data/train`, `data/test`, `data/new_breeds_v3`, and original Oxford images. Train/val/test split membership references the same physical ID pools.
- Also compared 21,290 ID metadata hashes from the dataset manifest and `outputs/splits*/*.csv`, and 8,610 hashes in both OE manifests (all splits).
- **0** fixture byte duplicates, decoded-pixel duplicates, average-hash duplicates, ID byte/pixel overlaps, and OE manifest byte overlaps.
- **0** perceptual near matches (64-bit dHash Hamming distance <= 4) within fixtures or against ID/source images. This is heuristic screening, supplemented by source and visual review.
- All fixtures passed Pillow `verify()` and full `load()`. **0** corrupt files, symlinks, ancestor reparse points, or hardlinks.
- Minimum stored width: 500 px; minimum height: 600 px; largest file: 1,115,139 bytes. No synthetic category placeholders or domestic dogs.
- Existing Caltech sources were excluded because their identities were already present in the OE datasets. A phone pictogram was replaced by a real Nokia photograph before inference; no fixture was removed for its prediction.

Full details: [integrity results](metadata/full_pipeline_ood_integrity.json).

## Requested verification

The literal `..venv/Scripts/python.exe` path does not exist here; commands used the project's existing `.venv/Scripts/python.exe`.

| Run | Result | Summary |
|---|---|---|
| `.venv/Scripts/python.exe -u -m unittest tests.test_full_pipeline -v` | PASS | Ran 11 tests in 161.190s; OK |
| `.venv/Scripts/python.exe -u -m unittest discover -s tests -p test_*.py -v` | PASS | Ran 118 tests in 257.543s; OK |
| `.venv/Scripts/python.exe -u -m unittest src.test_local_app -v` | PASS | Ran 25 tests in 20.167s; OK |
| `node --test src/test_frontend_behavior.js` | PASS | 48 tests; 48 pass; 0 fail; 0 skipped |
| `.venv/Scripts/python.exe -u outputs/ood_fixture_coverage/evaluate_all_fixtures.py` | PASS | 32 fixtures, 0 contract failures |

Exact commands, exit codes, times and log paths: [validation runs](metadata/full_pipeline_ood_validation_runs.json).

The original five OOD group tests select only the lexically first image in each group. The supplementary all-fixture run posts every inventory image to `/predict` and calls the same unchanged `FullPipelineTests.assert_valid_prediction_response` contract. No assertions were weakened or production files edited.

## Observed decisions (all fixtures)

| Group | NOT_DOG | UNCERTAIN | DOG (false acceptance) |
|---|---:|---:|---:|
| other animals | 5 | 1 | 0 |
| humans | 4 | 2 | 0 |
| vehicles | 6 | 0 | 0 |
| household objects | 6 | 0 | 0 |
| hard OOD | 0 | 8 | 0 |

All fixtures are labeled non-dog. `DOG` is a semantic false acceptance; `UNCERTAIN` is abstention rather than rejection. The existing runtime-contract tests allow all three valid states, so their PASS result is not a claim of perfect OOD detection.

| Fixture | Decision | OOD score | Top dog-breed candidate |
|---|---|---:|---|
| `data/ood/test/cat/regression_01_cat.jpg` | UNCERTAIN | 0.419295132 | pomeranian |
| `data/ood/test/humans/regression_boy_walking_contraluz.jpg` | UNCERTAIN | 0.452794671 | flat-coated_retriever |
| `data/ood/test/humans/regression_farmer_at_harvest.jpg` | UNCERTAIN | 0.421088815 | welsh_springer_spaniel |
| `data/ood_hard/coyote/regression_coyote_yellowstone.jpg` | UNCERTAIN | 0.444051743 | dhole |
| `data/ood_hard/coyote/regression_coyote_yellowstone_snow.jpg` | UNCERTAIN | 0.439921737 | siberian_husky |
| `data/ood_hard/fox/regression_fox_portrait.jpg` | UNCERTAIN | 0.452980936 | pembroke |
| `data/ood_hard/fox/regression_fox_rautas_snow.jpg` | UNCERTAIN | 0.417224348 | pembroke |
| `data/ood_hard/hyena/regression_hyena_kongkham_portrait.jpg` | UNCERTAIN | 0.346412301 | african_hunting_dog |
| `data/ood_hard/hyena/regression_hyena_lubaya_portrait.jpg` | UNCERTAIN | 0.453124106 | african_hunting_dog |
| `data/ood_hard/wolf/regression_wolf_european_portrait.jpg` | UNCERTAIN | 0.445367277 | dhole |
| `data/ood_hard/wolf/regression_wolf_gary_kramer.jpg` | UNCERTAIN | 0.405644536 | dhole |

Full responses and raw scores: [prediction records](metadata/full_pipeline_ood_predictions.json).

## Protected artifacts

**107 pre-existing files checked; 0 changed by this task.** This includes app, inference, tests, OOD code, configs, both deployment bundles, centroids, thresholds, the existing dataset manifest and `.gitignore`. Pre-existing uncommitted code changes were retained.

Deployed detector remains cosine, `T_uncertain = 0.343358`, `T_not_dog = 0.457513`. No training, fine-tuning, calibration, centroid regeneration, or model modification was performed.

| Artifact | Unchanged SHA-256 |
|---|---|
| `outputs/deployment_oe_v2/dog_breed_model_oe_v2.keras` | `e4508564a9389250e265cae6ea6fed0e1a0edf8497d62cd9c8a44342bdd45e61` |
| `outputs/deployment_oe_v2/class_centroids.npy` | `d4588ee2cff5d8cbe3b670ca99507ae97f7f07b75822d1e247200e7c481e85cb` |
| `outputs/deployment_oe_v2/ood_calibration.json` | `a334b38692d6a9d834fbe5c0aceb80ffaf617f6ef6e80be3c0b3d2ce38dc7640` |

All before/after hashes: [protected-file verification](metadata/full_pipeline_ood_protected_files.json).

Audit and materialization helpers, visual review sheets, original source metadata and test logs remain in `outputs/ood_fixture_coverage/`. No calibration fixtures were added.
