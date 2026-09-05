"""Exploratory Data Analysis (EDA) module for Dog Breed Identification.

This module computes breed distribution statistics, assesses class imbalance,
generates visualizations (bar chart and histogram), and exports class count summaries.
"""

from pathlib import Path
import sys
from typing import Any, Dict, Optional

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless execution
import matplotlib.pyplot as plt
import pandas as pd

# Ensure project root is in sys.path for both module and direct execution
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import load_labels

# Default directories
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
EDA_OUTPUT_DIR = OUTPUTS_DIR / "eda"


def compute_eda_statistics(df: pd.DataFrame) -> Dict[str, Any]:
    """Compute comprehensive EDA statistics from labels dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame loaded via load_labels() containing ['id', 'breed', 'img_name'].

    Returns
    -------
    dict
        Dictionary containing all computed EDA and imbalance metrics.
    """
    total_images = len(df)
    breed_counts = df["breed"].value_counts()
    total_breeds = int(breed_counts.count())

    # Metadata confirmation
    if total_images != 10222 or total_breeds != 120:
        raise ValueError(
            f"Validation failed: expected 10222 rows and 120 unique breeds, "
            f"found {total_images} rows and {total_breeds} breeds."
        )

    min_images = int(breed_counts.min())
    max_images = int(breed_counts.max())
    mean_images = float(breed_counts.mean())
    median_images = float(breed_counts.median())
    std_images = float(breed_counts.std())

    min_breeds = breed_counts[breed_counts == min_images].index.tolist()
    max_breeds = breed_counts[breed_counts == max_images].index.tolist()

    diff_images = max_images - min_images
    imbalance_ratio = float(max_images / min_images)

    min_percentage = float(min_images / total_images * 100)
    max_percentage = float(max_images / total_images * 100)

    # Distribution intervals
    ranges = {
        "< 70": int((breed_counts < 70).sum()),
        "70-79": int(((breed_counts >= 70) & (breed_counts <= 79)).sum()),
        "80-89": int(((breed_counts >= 80) & (breed_counts <= 89)).sum()),
        "90-99": int(((breed_counts >= 90) & (breed_counts <= 99)).sum()),
        ">= 100": int((breed_counts >= 100).sum()),
    }

    # Top 10 and Bottom 10
    top_10 = breed_counts.head(10)
    bottom_10 = breed_counts.tail(10)

    return {
        "total_images": total_images,
        "total_breeds": total_breeds,
        "breed_counts": breed_counts,
        "min_images": min_images,
        "min_breeds": min_breeds,
        "max_images": max_images,
        "max_breeds": max_breeds,
        "mean_images": mean_images,
        "median_images": median_images,
        "std_images": std_images,
        "diff_images": diff_images,
        "imbalance_ratio": imbalance_ratio,
        "min_percentage": min_percentage,
        "max_percentage": max_percentage,
        "distribution_ranges": ranges,
        "top_10": top_10,
        "bottom_10": bottom_10,
    }


def plot_breed_distribution(breed_counts: pd.Series, output_path: Path) -> Path:
    """Generate and save bar chart of image distribution across all 120 breeds.

    Parameters
    ----------
    breed_counts : pd.Series
        Series with breed as index and image count as value, sorted descending.
    output_path : Path
        Filepath to save the PNG image.

    Returns
    -------
    Path
        Saved image path.
    """
    sorted_counts = breed_counts.sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(24, 8))
    ax.bar(
        range(len(sorted_counts)),
        sorted_counts.values,
        color="#2b5c8f",
        width=0.7,
        edgecolor="#1a3b5c",
        linewidth=0.5,
    )

    ax.set_xlim(-0.8, len(sorted_counts) - 0.2)
    ax.set_xticks(range(len(sorted_counts)))
    ax.set_xticklabels(sorted_counts.index, rotation=90, ha="right", fontsize=7)
    ax.set_xlabel("Dog Breed", fontsize=12, fontweight="bold", labelpad=10)
    ax.set_ylabel("Number of Images", fontsize=12, fontweight="bold")
    ax.set_title(
        f"Distribution of Images per Dog Breed ({len(sorted_counts)} Breeds, Sorted Descending)",
        fontsize=15,
        fontweight="bold",
        pad=15,
    )
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)

    return output_path


def plot_class_count_histogram(
    breed_counts: pd.Series, mean_val: float, median_val: float, output_path: Path
) -> Path:
    """Generate and save histogram of image counts per class.

    Parameters
    ----------
    breed_counts : pd.Series
        Series with image counts per breed.
    mean_val : float
        Mean images per breed.
    median_val : float
        Median images per breed.
    output_path : Path
        Filepath to save the PNG image.

    Returns
    -------
    Path
        Saved image path.
    """
    counts = breed_counts.values
    min_c = int(counts.min())
    max_c = int(counts.max())

    # Create 5-width bins covering the min and max counts
    bin_start = (min_c // 5) * 5
    bin_end = ((max_c // 5) + 2) * 5
    bins = list(range(bin_start, bin_end, 5))

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(
        counts,
        bins=bins,
        color="#1f7a8c",
        edgecolor="#022b3a",
        linewidth=1.0,
        rwidth=0.85,
    )

    ax.axvline(
        mean_val,
        color="#d90429",
        linestyle="--",
        linewidth=2.0,
        label=f"Mean: {mean_val:.2f}",
    )
    ax.axvline(
        median_val,
        color="#f77f00",
        linestyle=":",
        linewidth=2.0,
        label=f"Median: {median_val:.1f}",
    )

    ax.set_xlabel("Number of Images per Breed", fontsize=12, fontweight="bold", labelpad=8)
    ax.set_ylabel("Number of Breeds", fontsize=12, fontweight="bold", labelpad=8)
    ax.set_title(
        "Histogram of Images per Dog Breed (Class Count Distribution)",
        fontsize=14,
        fontweight="bold",
        pad=12,
    )
    ax.legend(fontsize=11, loc="upper right")
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)

    return output_path


def save_breed_counts_csv(df: pd.DataFrame, output_path: Path) -> Path:
    """Save sorted breed counts table to CSV.

    Parameters
    ----------
    df : pd.DataFrame
        Original dataframe loaded via load_labels().
    output_path : Path
        Filepath for CSV output.

    Returns
    -------
    Path
        Saved CSV path.
    """
    counts_df = (
        df["breed"]
        .value_counts()
        .sort_index()
        .reset_index()
    )
    counts_df.columns = ["breed", "count"]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    counts_df.to_csv(output_path, index=False)
    return output_path


def verify_eda_outputs(output_dir: Path) -> Dict[str, Any]:
    """Verify integrity of all generated EDA outputs.

    Parameters
    ----------
    output_dir : Path
        Directory containing EDA outputs.

    Returns
    -------
    dict
        Verification summary results.
    """
    dist_png = output_dir / "breed_distribution.png"
    hist_png = output_dir / "class_count_histogram.png"
    csv_file = output_dir / "breed_counts.csv"

    if not dist_png.is_file() or dist_png.stat().st_size == 0:
        raise FileNotFoundError(f"Missing or empty file: {dist_png}")

    if not hist_png.is_file() or hist_png.stat().st_size == 0:
        raise FileNotFoundError(f"Missing or empty file: {hist_png}")

    if not csv_file.is_file() or csv_file.stat().st_size == 0:
        raise FileNotFoundError(f"Missing or empty file: {csv_file}")

    # Validate breed_counts.csv contents
    counts_df = pd.read_csv(csv_file)
    rows = len(counts_df)
    total_count = int(counts_df["count"].sum())
    duplicates = int(counts_df["breed"].duplicated().sum())
    null_count = int(counts_df.isnull().sum().sum())

    if rows != 120:
        raise ValueError(f"Integrity check failed: expected 120 rows in {csv_file}, got {rows}")

    if total_count != 10222:
        raise ValueError(f"Integrity check failed: expected sum count 10222 in {csv_file}, got {total_count}")

    if duplicates > 0:
        raise ValueError(f"Integrity check failed: found {duplicates} duplicate breed(s) in {csv_file}")

    if null_count > 0:
        raise ValueError(f"Integrity check failed: found {null_count} null value(s) in {csv_file}")

    return {
        "breed_distribution_png_exists": True,
        "class_count_histogram_png_exists": True,
        "breed_counts_csv_exists": True,
        "rows": rows,
        "total_count": total_count,
        "duplicates": duplicates,
        "null_count": null_count,
    }


def run_eda(
    df: Optional[pd.DataFrame] = None,
    output_dir: Path = EDA_OUTPUT_DIR,
) -> Dict[str, Any]:
    """Execute complete EDA pipeline and output generation.

    Parameters
    ----------
    df : pd.DataFrame, optional
        Dataframe to analyze. If None, loaded via load_labels().
    output_dir : Path, optional
        Target directory for outputs. Defaults to outputs/eda.

    Returns
    -------
    dict
        Dictionary containing EDA statistics and output verification.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load labels if not provided
    if df is None:
        df = load_labels()

    # 2. Compute statistics
    stats = compute_eda_statistics(df)

    # 3. Generate visualizations
    dist_png = output_dir / "breed_distribution.png"
    hist_png = output_dir / "class_count_histogram.png"
    csv_file = output_dir / "breed_counts.csv"

    plot_breed_distribution(stats["breed_counts"], dist_png)
    plot_class_count_histogram(
        stats["breed_counts"],
        stats["mean_images"],
        stats["median_images"],
        hist_png,
    )

    # 4. Save CSV
    save_breed_counts_csv(df, csv_file)

    # 5. Verify outputs
    verification = verify_eda_outputs(output_dir)

    # 6. Format and print terminal summary
    print("EDA completed successfully\n")
    print(f"Total images: {stats['total_images']}")
    print(f"Total breeds: {stats['total_breeds']}\n")
    print(f"Min images/class: {stats['min_images']}")
    print(f"Max images/class: {stats['max_images']}")
    print(f"Mean images/class: {stats['mean_images']:.2f}")
    print(f"Median images/class: {stats['median_images']:.1f}")
    print(f"Std images/class: {stats['std_images']:.2f}\n")
    print(f"Imbalance ratio: {stats['imbalance_ratio']:.4f}")
    print(f"Max-min difference: {stats['diff_images']}")
    print(f"Min class percentage: {stats['min_percentage']:.2f}%")
    print(f"Max class percentage: {stats['max_percentage']:.2f}%\n")

    print("Distribution ranges:")
    for rng, cnt in stats["distribution_ranges"].items():
        print(f"  {rng}: {cnt} breeds")

    print("\nTop 10 breeds:")
    for rank, (breed, count) in enumerate(stats["top_10"].items(), start=1):
        print(f"  {rank:2d}. {breed:25s} {count:4d}")

    print("\nBottom 10 breeds:")
    for rank, (breed, count) in enumerate(stats["bottom_10"].items(), start=1):
        print(f"  {rank:2d}. {breed:25s} {count:4d}")

    print("\nOutputs:")
    print("outputs/eda/breed_distribution.png")
    print("outputs/eda/class_count_histogram.png")
    print("outputs/eda/breed_counts.csv")

    return {
        "stats": stats,
        "verification": verification,
    }


if __name__ == "__main__":
    run_eda()
