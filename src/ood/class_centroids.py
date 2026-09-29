"""Build one embedding centroid for each dog-breed class.

The training split is the only source of images used here.  Class order is
read from the production model's ``class_names.json`` so row ``i`` in the
saved centroid array always corresponds to classifier class ``i``.

Run from the project root with::

    .venv\\Scripts\\python.exe -m src.ood.class_centroids
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from src.models.feature_extractor import build_feature_extractor, extract_embeddings


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "outputs/deployment_ready/dog_breed_model.keras"
DEFAULT_CLASS_NAMES_PATH = PROJECT_ROOT / "outputs/deployment_ready/class_names.json"
DEFAULT_TRAIN_CSV = PROJECT_ROOT / "outputs/splits/train.csv"
DEFAULT_TRAIN_DIR = PROJECT_ROOT / "data/train"
DEFAULT_CENTROIDS_PATH = PROJECT_ROOT / "outputs/calibration/class_centroids.npy"
DEFAULT_METADATA_PATH = PROJECT_ROOT / "outputs/calibration/class_centroids.json"
EXPECTED_NUM_CLASSES = 120


def load_class_order(path: str | Path = DEFAULT_CLASS_NAMES_PATH) -> list[str]:
    """Load the production classifier's class order from JSON."""

    source = Path(path)
    with source.open(encoding="utf-8") as handle:
        raw = json.load(handle)

    if isinstance(raw, dict):
        if set(raw) != {str(index) for index in range(EXPECTED_NUM_CLASSES)}:
            raise ValueError("class indices must be contiguous from 0 to 119")
        # The shipped artifact uses JSON object keys "0" ... "119".  Sorting
        # numerically protects the classifier order from JSON key formatting.
        try:
            ordered_items = sorted(raw.items(), key=lambda item: int(item[0]))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"class names must use numeric keys: {source}") from exc
        classes = [value for _, value in ordered_items]
    elif isinstance(raw, list):
        classes = raw
    else:
        raise ValueError(f"class names must be a JSON object or array: {source}")

    if (
        len(classes) != EXPECTED_NUM_CLASSES
        or any(not isinstance(name, str) or not name.strip() for name in classes)
        or len(set(classes)) != len(classes)
    ):
        raise ValueError(f"expected {EXPECTED_NUM_CLASSES} unique non-empty class names")
    return [str(name) for name in classes]


def load_training_rows(path: str | Path = DEFAULT_TRAIN_CSV) -> list[dict[str, str]]:
    """Read and validate the authoritative training split metadata.

    This deliberately accepts only a file named ``train.csv``.  The split
    files point into the shared ``data/train`` directory, so this guard keeps
    an accidental validation/test CSV from silently becoming calibration data.
    """

    source = Path(path)
    if source.name.lower() != "train.csv":
        raise ValueError(f"centroid extraction requires the training split train.csv, got {source.name}")
    with source.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"id", "breed", "img_name"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"training split is missing columns: {sorted(required)}")
        rows = list(reader)

    if not rows:
        raise ValueError("training split is empty")
    if any(not row["id"] or not row["breed"] or not row["img_name"] for row in rows):
        raise ValueError("training split contains an empty id, breed, or image name")
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("training split contains duplicate image ids")
    if len({row["img_name"] for row in rows}) != len(rows):
        raise ValueError("training split contains duplicate image names")
    for row in rows:
        if row["img_name"] != f"{row['id']}.jpg":
            raise ValueError("training image name must match its id")
    # Read held-out metadata only to check disjointness; never decode its images.
    train_ids = {row["id"] for row in rows}
    train_names = {row["img_name"] for row in rows}
    for split in ("val", "test"):
        with source.with_name(f"{split}.csv").open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if not {"id", "img_name"}.issubset(reader.fieldnames or []):
                raise ValueError(f"invalid {split} split metadata")
            for row in reader:
                if row["id"] in train_ids or row["img_name"] in train_names:
                    raise ValueError(f"training split overlaps {split}: {row['id']}")
    return rows


def _training_image_path(train_dir: Path, image_name: str) -> Path:
    """Resolve an image name while preventing paths outside ``data/train``."""

    candidate = (train_dir / image_name).resolve()
    root = train_dir.resolve()
    if Path(image_name).name != image_name or not candidate.is_relative_to(root):
        raise ValueError(f"training image must be a direct child of {root}: {image_name}")
    if not candidate.is_file():
        raise FileNotFoundError(f"training image not found: {candidate}")
    return candidate


def _iter_batches(
    rows: Sequence[dict[str, str]],
    train_dir: Path,
    class_to_index: dict[str, int],
    batch_size: int,
) -> Iterable[tuple[list[int], np.ndarray]]:
    """Yield class indices and decoded image tensors in deterministic order."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    # Keep TensorFlow lazy so metadata validation can run without the ML
    # runtime. Decode and resize through one tf.data pipeline instead of
    # starting separate eager read/decode/resize operations for every image.
    import tensorflow as tf

    image_paths = []
    labels = []
    for row in rows:
        if row["breed"] not in class_to_index:
            raise ValueError(f"unknown breed in training split: {row['breed']}")
        labels.append(class_to_index[row["breed"]])
        image_paths.append(str(_training_image_path(train_dir, row["img_name"])))

    dataset = tf.data.Dataset.from_tensor_slices(
        (tf.constant(image_paths), tf.constant(labels, dtype=tf.int32))
    )

    def decode_and_resize(path: Any, label: Any) -> tuple[Any, Any]:
        image = tf.io.read_file(path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, (224, 224))
        return tf.cast(image, tf.float32), label

    dataset = dataset.map(decode_and_resize, num_parallel_calls=4)
    dataset = dataset.batch(batch_size, drop_remainder=False).prefetch(1)
    for image_batch, label_batch in dataset:
        yield label_batch.numpy().tolist(), image_batch.numpy()


def compute_class_centroids(
    extractor: Any,
    rows: Sequence[dict[str, str]],
    class_order: Sequence[str],
    train_dir: str | Path,
    *,
    batch_size: int = 32,
) -> tuple[np.ndarray, list[int]]:
    """Extract training embeddings and return ``(centroids, sample_counts)``."""

    if not rows:
        raise ValueError("cannot compute centroids from an empty training split")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    classes = list(class_order)
    if not classes or len(set(classes)) != len(classes):
        raise ValueError("class_order must contain unique class names")
    class_to_index = {name: index for index, name in enumerate(classes)}
    present = {row["breed"] for row in rows}
    if present != set(classes):
        raise ValueError("training breeds must match class_order with no missing or unknown class")
    embedding_dim = getattr(extractor, "embedding_dim", None)
    if embedding_dim is None or int(embedding_dim) <= 0:
        raise ValueError("extractor must expose a positive embedding_dim")

    sums = np.zeros((len(classes), int(embedding_dim)), dtype=np.float64)
    counts = np.zeros(len(classes), dtype=np.int64)
    total_batches = (len(rows) + batch_size - 1) // batch_size
    for batch_number, (labels, image_batch) in enumerate(
        _iter_batches(rows, Path(train_dir), class_to_index, batch_size), start=1
    ):
        embeddings = extract_embeddings(
            extractor, image_batch, check_determinism=(batch_number == 1)
        )
        if embeddings.shape != (len(labels), int(embedding_dim)):
            raise RuntimeError(
                f"unexpected embedding shape: expected {(len(labels), int(embedding_dim))}, "
                f"got {embeddings.shape}"
            )
        for class_index in range(len(classes)):
            selected = embeddings[np.asarray(labels) == class_index]
            if selected.size:
                sums[class_index] += selected.sum(axis=0, dtype=np.float64)
                counts[class_index] += selected.shape[0]
        if batch_number == 1 or batch_number == total_batches or batch_number % 25 == 0:
            print(f"Embedding batches: {batch_number}/{total_batches}", flush=True)

    if np.any(counts == 0):
        missing = [classes[index] for index, count in enumerate(counts) if count == 0]
        raise ValueError(f"training split has no samples for class(es): {missing}")
    centroids = (sums / counts[:, None]).astype(np.float32)
    if not np.isfinite(centroids).all():
        raise RuntimeError("class centroids contain NaN or Inf")
    return centroids, [int(count) for count in counts]


def _atomic_save_npy(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npy", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        np.save(temporary, values, allow_pickle=False)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_save_json(path: Path, metadata: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, suffix=".json", delete=False
    ) as handle:
        temporary = Path(handle.name)
        json.dump(metadata, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def build_centroid_artifacts(
    *,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    class_names_path: str | Path = DEFAULT_CLASS_NAMES_PATH,
    train_csv: str | Path = DEFAULT_TRAIN_CSV,
    train_dir: str | Path = DEFAULT_TRAIN_DIR,
    centroids_path: str | Path = DEFAULT_CENTROIDS_PATH,
    metadata_path: str | Path = DEFAULT_METADATA_PATH,
    batch_size: int = 32,
) -> dict[str, Any]:
    """Build and persist class centroids from the training split only."""

    rows = load_training_rows(train_csv)
    class_order = load_class_order(class_names_path)
    class_to_index = {name: index for index, name in enumerate(class_order)}
    unknown = sorted({row["breed"] for row in rows} - set(class_to_index))
    if unknown:
        raise ValueError(f"training split contains unknown class(es): {unknown}")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    print(f"Validated {len(rows)} training rows; loading TensorFlow/model...", flush=True)

    import tensorflow as tf

    model = tf.keras.models.load_model(
        model_path,
        custom_objects={"preprocess_input": tf.keras.applications.mobilenet_v2.preprocess_input},
        compile=False,
        safe_mode=True,
    )
    extractor = build_feature_extractor(model)
    if int(model.output_shape[-1]) != len(class_order):
        raise ValueError("model class count does not match class_order")
    weights_before = [np.array(weight.numpy(), copy=True) for weight in model.weights]
    centroids, samples_per_class = compute_class_centroids(
        extractor,
        rows,
        class_order,
        train_dir,
        batch_size=batch_size,
    )
    if any(not np.array_equal(before, after.numpy())
           for before, after in zip(weights_before, model.weights)):
        raise RuntimeError("feature extraction changed model weights")
    metadata = {
        "embedding_dimension": int(centroids.shape[1]),
        "num_classes": int(centroids.shape[0]),
        "class_order": class_order,
        # Entry i corresponds to centroid row i and class_order[i].
        "samples_per_class": samples_per_class,
        "source_split": "train",
        "num_samples": len(rows),
        "train_csv_sha256": hashlib.sha256(Path(train_csv).read_bytes()).hexdigest(),
        "model_sha256": hashlib.sha256(Path(model_path).read_bytes()).hexdigest(),
        "embedding_layer": extractor.penultimate_layer_name,
        "aggregation": "arithmetic_mean_of_raw_embeddings",
        "dtype": str(centroids.dtype),
    }
    _atomic_save_npy(Path(centroids_path), centroids)
    _atomic_save_json(Path(metadata_path), metadata)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--class-names", type=Path, default=DEFAULT_CLASS_NAMES_PATH)
    parser.add_argument("--train-csv", type=Path, default=DEFAULT_TRAIN_CSV)
    parser.add_argument("--train-dir", type=Path, default=DEFAULT_TRAIN_DIR)
    parser.add_argument("--centroids", type=Path, default=DEFAULT_CENTROIDS_PATH)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA_PATH)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    metadata = build_centroid_artifacts(
        model_path=args.model,
        class_names_path=args.class_names,
        train_csv=args.train_csv,
        train_dir=args.train_dir,
        centroids_path=args.centroids,
        metadata_path=args.metadata,
        batch_size=args.batch_size,
    )
    centroids = np.load(args.centroids, allow_pickle=False)
    print(f"Saved {args.centroids}: shape={centroids.shape}, dtype={centroids.dtype}")
    print(f"Saved {args.metadata}: {metadata['num_classes']} classes, "
          f"embedding_dimension={metadata['embedding_dimension']}")
    print(f"Training samples used: {sum(metadata['samples_per_class'])}")


if __name__ == "__main__":
    main()
