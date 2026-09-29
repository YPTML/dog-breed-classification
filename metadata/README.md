# Dataset manifest

Build from the project root (Pillow is used when installed; JPEG/PNG/GIF/BMP
header parsing provides a standard-library fallback):

```powershell
.\.venv\Scripts\python.exe -m src.data.build_manifest
.\.venv\Scripts\python.exe -m src.data.build_manifest --validate-only
.\.venv\Scripts\python.exe -m unittest tests.test_build_manifest
```

The output is `metadata/dataset_manifest.csv`. The script also supports direct
execution and `--root PATH`; defaults are independent of the working directory.
Images are opened only for reading. Validation failures exit with status 1 and
leave the existing output intact.

| Field | Meaning |
| --- | --- |
| image_id | Original Kaggle filename stem; independent of CSV row order |
| path | Project-relative path with `/` separators |
| label, breed | Breed name validated against `outputs/deployment_ready/class_names.json` |
| distribution | `id` for dog images, `ood` for OOD images |
| ood_category | Empty for ID images; supported OOD category for OOD rows |
| split | Existing `outputs/splits/{train,val,test}.csv` assignment for labeled images |
| source | `kaggle_dog_breed_identification` |
| width, height | Original stored pixel dimensions, before EXIF rotation or resizing |
| sha256 | SHA-256 of the original file bytes |

Kaggle images under `data/test` have no ground-truth labels: `label` and `breed`
are empty and `split=unlabeled_test`. They are distinct from the internal labeled
test split and must not be used as labeled evaluation data. If all split CSVs are
absent, labeled images use `unsplit`; partial, overlapping or incomplete split
metadata is rejected. Unexpected images outside this current layout are rejected
rather than silently omitted or assigned guessed metadata.

Rows are sorted by path, with fixed field order, UTF-8 encoding and LF line
endings. Unchanged images and input metadata produce byte-identical output.
Duplicate paths (resolved to detect aliases), duplicate image IDs, missing files,
unknown breeds and conflicting labels cause validation errors. Generation also
checks image headers and computes dimensions and hashes. Pillow is used for
full image verification when available; the fallback checks the image header.
`--validate-only` mode checks metadata integrity and file existence; it does not
recompute stored hashes or dimensions.

The OOD builder uses `data/ood/calibration/{category}` and
`data/ood/test/{category}` for regular OOD, and `data/ood_hard/{category}` for
hard OOD. OOD rows have empty `label`/`breed`, `distribution=ood`, and
`split=calibration`, `test`, or `hard`. Exact OOD duplicates and exact matches
to the 120-class training pool are rejected.
