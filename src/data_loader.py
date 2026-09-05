"""Data loading and validation module for Dog Breed Identification."""

from pathlib import Path
import os
import pandas as pd

# Define paths dynamically relative to this source file
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
LABELS_PATH = DATA_DIR / "labels.csv"
TRAIN_DIR = DATA_DIR / "train"


def validate_dataset(df: pd.DataFrame, train_dir: Path) -> int:
    """Validate dataframe structure, columns, nulls, duplicates, and image file existence.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing 'id', 'breed', and 'img_name' columns.
    train_dir : Path
        Directory where training images are stored.

    Returns
    -------
    int
        Count of missing image files (0 when all exist).

    Raises
    ------
    ValueError
        If required columns are missing, null values exist, duplicate IDs exist,
        or image filenames do not match expected format.
    FileNotFoundError
        If any referenced image file is missing from train_dir.
    """
    # 1. Check required columns exist
    required_cols = ["id", "breed"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in labels: {missing_cols}")

    # 2. Check null values
    null_id_count = int(df["id"].isnull().sum())
    if null_id_count > 0:
        raise ValueError(f"Validation failed: found {null_id_count} null value(s) in 'id' column.")

    null_breed_count = int(df["breed"].isnull().sum())
    if null_breed_count > 0:
        raise ValueError(f"Validation failed: found {null_breed_count} null value(s) in 'breed' column.")

    # 3. Check duplicate IDs
    duplicate_id_count = int(df["id"].duplicated().sum())
    if duplicate_id_count > 0:
        raise ValueError(f"Validation failed: found {duplicate_id_count} duplicate ID(s).")

    # 4. Check img_name format
    if "img_name" in df.columns:
        if not df["img_name"].str.endswith(".jpg").all():
            raise ValueError("Validation failed: all 'img_name' entries must end with '.jpg'")

        duplicate_img_count = int(df["img_name"].duplicated().sum())
        if duplicate_img_count > 0:
            raise ValueError(f"Validation failed: found {duplicate_img_count} duplicate 'img_name' value(s).")

        # 5. Check each img_name exists in train_dir
        existing_images = set(os.listdir(train_dir))
        missing_images = [name for name in df["img_name"] if name not in existing_images]
        if missing_images:
            raise FileNotFoundError(
                f"Validation failed: {len(missing_images)} image file(s) missing from {train_dir}. "
                f"First missing: {missing_images[:5]}"
            )
        return len(missing_images)

    return 0


def load_labels(labels_path: Path = LABELS_PATH, train_dir: Path = TRAIN_DIR) -> pd.DataFrame:
    """Load labels metadata from CSV, construct image filenames, validate dataset integrity,

    and return standard DataFrame with columns ['id', 'breed', 'img_name'].

    Parameters
    ----------
    labels_path : Path, optional
        Path to labels.csv (default: DATA_DIR / 'labels.csv').
    train_dir : Path, optional
        Path to train directory (default: DATA_DIR / 'train').

    Returns
    -------
    pd.DataFrame
        Standardized DataFrame with columns ['id', 'breed', 'img_name'].
    """
    labels_path = Path(labels_path)
    train_dir = Path(train_dir)

    # 1. Verify existence of labels.csv and train directory
    if not labels_path.is_file():
        raise FileNotFoundError(f"labels.csv not found at: {labels_path}")

    if not train_dir.is_dir():
        raise NotADirectoryError(f"Train directory not found at: {train_dir}")

    # 2. Read labels.csv
    df = pd.read_csv(labels_path)

    # 3. Keep only 'id' and 'breed'
    if "id" not in df.columns or "breed" not in df.columns:
        raise ValueError(f"labels.csv must contain 'id' and 'breed' columns. Found: {list(df.columns)}")

    df = df[["id", "breed"]].copy()

    # 4. Generate img_name column: img_name = id + ".jpg"
    df["img_name"] = df["id"].astype(str) + ".jpg"

    # 5. Ensure exact column order: id, breed, img_name
    df = df[["id", "breed", "img_name"]]

    # 6. Perform validation checks
    validate_dataset(df, train_dir)

    return df


if __name__ == "__main__":
    df = load_labels()

    # Calculate missing image count for display
    train_files = set(os.listdir(TRAIN_DIR))
    missing_count = sum(1 for name in df["img_name"] if name not in train_files)

    print("Dataset loaded successfully")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"Unique breeds: {df['breed'].nunique()}")
    print(f"Missing image files: {missing_count}")
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)
    print(df.head())

