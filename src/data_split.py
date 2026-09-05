"""Dataset splitting module for Dog Breed Identification.

This module splits the labeled dog breed dataset into Train (80%), Validation (10%),
and Test (10%) sets using stratified splitting to preserve breed distribution
across all 120 classes with full reproducibility.
"""

from pathlib import Path
import os
from typing import Any, Dict, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.data_loader import load_labels

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
TRAIN_IMG_DIR = DATA_DIR / "train"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
SPLITS_DIR = OUTPUTS_DIR / "splits"

# Split configuration
RANDOM_STATE = 42
TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10

# Validate ratio sum
if not np.isclose(TRAIN_RATIO + VAL_RATIO + TEST_RATIO, 1.0):
    raise ValueError(
        f"Split ratios must sum to 1.0, got {TRAIN_RATIO + VAL_RATIO + TEST_RATIO}"
    )


def split_dataset(
    df: Optional[pd.DataFrame] = None,
    random_state: int = RANDOM_STATE,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split dataset into Train (80%), Validation (10%), and Test (10%) sets.

    Uses a two-step stratified split based on the 'breed' column to ensure all
    120 breeds are well-represented across all three splits.

    Parameters
    ----------
    df : pd.DataFrame, optional
        DataFrame containing 'id', 'breed', and 'img_name' columns.
        If None, load_labels() is called to load the full labeled dataset.
    random_state : int, default=42
        Random seed for reproducibility.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        (train_df, val_df, test_df) with reset indices and exact columns
        ['id', 'breed', 'img_name'].

    Raises
    ------
    ValueError
        If required columns are missing or if dataset contains fewer classes
        than expected.
    """
    if df is None:
        df = load_labels()

    required_cols = ["id", "breed", "img_name"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"DataFrame missing required columns: {missing_cols}")

    # Ensure clean DataFrame with exactly the required columns
    df = df[required_cols].copy()

    # Step 1: Split 80% train, 20% temp (for val + test)
    temp_ratio = VAL_RATIO + TEST_RATIO
    train_df, temp_df = train_test_split(
        df,
        test_size=temp_ratio,
        stratify=df["breed"],
        random_state=random_state,
    )

    # Step 2: Split temp equally into 50% val, 50% test (10% and 10% of total)
    # val_ratio within temp = VAL_RATIO / (VAL_RATIO + TEST_RATIO) = 0.5
    test_size_in_temp = TEST_RATIO / temp_ratio
    val_df, test_df = train_test_split(
        temp_df,
        test_size=test_size_in_temp,
        stratify=temp_df["breed"],
        random_state=random_state,
    )

    # Reset indices without keeping old index
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    # Ensure column order and no extra columns
    train_df = train_df[required_cols]
    val_df = val_df[required_cols]
    test_df = test_df[required_cols]

    return train_df, val_df, test_df


def save_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: Path = SPLITS_DIR,
) -> Dict[str, Path]:
    """Save train, validation, and test metadata DataFrames to CSV files.

    Parameters
    ----------
    train_df : pd.DataFrame
        Training metadata.
    val_df : pd.DataFrame
        Validation metadata.
    test_df : pd.DataFrame
        Test metadata.
    output_dir : Path, optional
        Destination directory (default: outputs/splits/).

    Returns
    -------
    Dict[str, Path]
        Mapping from split name to created CSV file path.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "train": output_dir / "train.csv",
        "val": output_dir / "val.csv",
        "test": output_dir / "test.csv",
    }

    train_df.to_csv(paths["train"], index=False)
    val_df.to_csv(paths["val"], index=False)
    test_df.to_csv(paths["test"], index=False)

    return paths


def validate_splits(
    df: pd.DataFrame,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    train_img_dir: Path = TRAIN_IMG_DIR,
) -> Dict[str, Any]:
    """Perform thorough validation checks on the train/val/test splits.

    Parameters
    ----------
    df : pd.DataFrame
        Original labeled DataFrame.
    train_df : pd.DataFrame
        Training split.
    val_df : pd.DataFrame
        Validation split.
    test_df : pd.DataFrame
        Test split.
    train_img_dir : Path, optional
        Directory containing physical training images.

    Returns
    -------
    Dict[str, Any]
        Dictionary of validation results and metrics.
    """
    total_len = len(df)
    n_train = len(train_df)
    n_val = len(val_df)
    n_test = len(test_df)
    split_sum = n_train + n_val + n_test

    # 1. Size check
    size_pass = split_sum == total_len

    # 2. Overlap check
    train_ids = set(train_df["id"])
    val_ids = set(val_df["id"])
    test_ids = set(test_df["id"])
    orig_ids = set(df["id"])

    overlap_train_val = len(train_ids & val_ids)
    overlap_train_test = len(train_ids & test_ids)
    overlap_val_test = len(val_ids & test_ids)

    # 3. Completeness check (Union)
    union_ids = train_ids | val_ids | test_ids
    missing_ids = len(orig_ids - union_ids)
    extra_ids = len(union_ids - orig_ids)

    # Internal duplicates
    dup_train = int(train_df["id"].duplicated().sum())
    dup_val = int(val_df["id"].duplicated().sum())
    dup_test = int(test_df["id"].duplicated().sum())

    # 4. Breed coverage
    total_breeds = df["breed"].nunique()
    train_breeds = train_df["breed"].nunique()
    val_breeds = val_df["breed"].nunique()
    test_breeds = test_df["breed"].nunique()
    breed_coverage_pass = (
        train_breeds == total_breeds
        and val_breeds == total_breeds
        and test_breeds == total_breeds
    )

    # 5. Class counts min/max
    train_counts = train_df["breed"].value_counts()
    val_counts = val_df["breed"].value_counts()
    test_counts = test_df["breed"].value_counts()

    train_min_count = int(train_counts.min())
    train_max_count = int(train_counts.max())
    val_min_count = int(val_counts.min())
    val_max_count = int(val_counts.max())
    test_min_count = int(test_counts.min())
    test_max_count = int(test_counts.max())

    # 6. Stratification check (class proportion deviations)
    p_orig = df["breed"].value_counts(normalize=True)
    p_train = train_counts / n_train
    p_val = val_counts / n_val
    p_test = test_counts / n_test

    max_dev_train = float((p_train - p_orig).abs().max())
    max_dev_val = float((p_val - p_orig).abs().max())
    max_dev_test = float((p_test - p_orig).abs().max())

    # 7. Reproducibility check
    train_rep, val_rep, test_rep = split_dataset(df, random_state=RANDOM_STATE)
    reproducible_train = train_df["id"].equals(train_rep["id"])
    reproducible_val = val_df["id"].equals(val_rep["id"])
    reproducible_test = test_df["id"].equals(test_rep["id"])
    reproducibility_pass = (
        reproducible_train and reproducible_val and reproducible_test
    )

    # Seed difference check
    train_diff_seed, _, _ = split_dataset(df, random_state=123)
    diff_seed_confirmed = not train_df["id"].equals(train_diff_seed["id"])

    # 8. Physical image file validation (sample 10 images from each split)
    rng = np.random.RandomState(RANDOM_STATE)
    train_sample = rng.choice(train_df["img_name"], size=10, replace=False)
    val_sample = rng.choice(val_df["img_name"], size=10, replace=False)
    test_sample = rng.choice(test_df["img_name"], size=10, replace=False)

    train_missing_files = sum(
        1 for img in train_sample if not (train_img_dir / img).is_file()
    )
    val_missing_files = sum(
        1 for img in val_sample if not (train_img_dir / img).is_file()
    )
    test_missing_files = sum(
        1 for img in test_sample if not (train_img_dir / img).is_file()
    )

    return {
        "total_samples": total_len,
        "n_train": n_train,
        "n_val": n_val,
        "n_test": n_test,
        "train_pct": (n_train / total_len) * 100,
        "val_pct": (n_val / total_len) * 100,
        "test_pct": (n_test / total_len) * 100,
        "size_pass": size_pass,
        "overlap_train_val": overlap_train_val,
        "overlap_train_test": overlap_train_test,
        "overlap_val_test": overlap_val_test,
        "missing_ids": missing_ids,
        "extra_ids": extra_ids,
        "dup_train": dup_train,
        "dup_val": dup_val,
        "dup_test": dup_test,
        "total_breeds": total_breeds,
        "train_breeds": train_breeds,
        "val_breeds": val_breeds,
        "test_breeds": test_breeds,
        "breed_coverage_pass": breed_coverage_pass,
        "train_min_count": train_min_count,
        "train_max_count": train_max_count,
        "val_min_count": val_min_count,
        "val_max_count": val_max_count,
        "test_min_count": test_min_count,
        "test_max_count": test_max_count,
        "max_dev_train": max_dev_train,
        "max_dev_val": max_dev_val,
        "max_dev_test": max_dev_test,
        "stratification_pass": (
            max_dev_train < 0.01 and max_dev_val < 0.01 and max_dev_test < 0.01
        ),
        "reproducible_train": reproducible_train,
        "reproducible_val": reproducible_val,
        "reproducible_test": reproducible_test,
        "reproducibility_pass": reproducibility_pass,
        "diff_seed_confirmed": diff_seed_confirmed,
        "train_sample_count": len(train_sample),
        "train_missing_files": train_missing_files,
        "val_sample_count": len(val_sample),
        "val_missing_files": val_missing_files,
        "test_sample_count": len(test_sample),
        "test_missing_files": test_missing_files,
    }


def validate_csv_files(paths: Dict[str, Path]) -> Dict[str, Any]:
    """Validate saved CSV split files.

    Parameters
    ----------
    paths : Dict[str, Path]
        Paths to train.csv, val.csv, and test.csv.

    Returns
    -------
    Dict[str, Any]
        Validation metrics for saved CSV files.
    """
    results = {}
    total_csv_rows = 0
    all_correct_cols = True
    all_no_null = True
    all_no_dup = True

    for name, path in paths.items():
        exists = path.is_file()
        if not exists:
            results[name] = {"exists": False}
            continue

        df_csv = pd.read_csv(path)
        rows = len(df_csv)
        total_csv_rows += rows
        cols = list(df_csv.columns)
        correct_cols = cols == ["id", "breed", "img_name"]
        null_count = int(df_csv.isnull().sum().sum())
        dup_count = int(df_csv["id"].duplicated().sum())

        if not correct_cols:
            all_correct_cols = False
        if null_count > 0:
            all_no_null = False
        if dup_count > 0:
            all_no_dup = False

        results[name] = {
            "exists": True,
            "rows": rows,
            "cols": cols,
            "correct_cols": correct_cols,
            "null_count": null_count,
            "dup_count": dup_count,
        }

    results["total_csv_rows"] = total_csv_rows
    results["all_correct_cols"] = all_correct_cols
    results["all_no_null"] = all_no_null
    results["all_no_dup"] = all_no_dup

    return results


def main() -> None:
    """Execute dataset splitting, save outputs, and print summary report."""
    print("DATA SPLIT VALIDATION\n")

    # Load dataset
    df = load_labels()
    total_labeled = len(df)
    total_breeds = df["breed"].nunique()

    print(f"Total labeled images: {total_labeled}")
    print(f"Total breeds: {total_breeds}")
    print(f"Random state: {RANDOM_STATE}\n")

    # Perform stratified split
    train_df, val_df, test_df = split_dataset(df, random_state=RANDOM_STATE)

    # Save to outputs/splits/
    saved_paths = save_splits(train_df, val_df, test_df)

    # Validate splits
    metrics = validate_splits(df, train_df, val_df, test_df)
    csv_metrics = validate_csv_files(saved_paths)

    # Display split sizes
    print(
        f"Train: {metrics['n_train']} ({metrics['train_pct']:.2f}%)"
    )
    print(
        f"Validation: {metrics['n_val']} ({metrics['val_pct']:.2f}%)"
    )
    print(
        f"Test: {metrics['n_test']} ({metrics['test_pct']:.2f}%)\n"
    )

    # Breed counts
    print(f"Train breeds: {metrics['train_breeds']}")
    print(f"Validation breeds: {metrics['val_breeds']}")
    print(f"Test breeds: {metrics['test_breeds']}\n")

    # Overlaps
    print(f"Overlap train/val: {metrics['overlap_train_val']}")
    print(f"Overlap train/test: {metrics['overlap_train_test']}")
    print(f"Overlap val/test: {metrics['overlap_val_test']}\n")

    # Missing & Extra IDs
    print(f"Missing IDs: {metrics['missing_ids']}")
    print(f"Extra IDs: {metrics['extra_ids']}\n")

    # Stratification & Reproducibility
    strat_status = "PASS" if metrics["stratification_pass"] else "FAIL"
    rep_status = "PASS" if metrics["reproducibility_pass"] else "FAIL"
    print(f"Stratification: {strat_status}")
    print(f"Reproducibility: {rep_status}\n")

    # Split files
    print("Split files:")
    print("outputs/splits/train.csv")
    print("outputs/splits/val.csv")
    print("outputs/splits/test.csv\n")

    print("Data split completed successfully.")


if __name__ == "__main__":
    main()
