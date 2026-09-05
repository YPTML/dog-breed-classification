"""Breed label encoding and mapping module for Dog Breed Identification.

This module provides functions to convert dog breed label strings into deterministic
integer class indices (0..119) and float32 one-hot vectors of length 120, as well as
decoding one-hot vectors back to breed names.
"""

from typing import Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelBinarizer

from src.data_loader import load_labels

# Total number of dog breeds in the dataset
NUM_CLASSES = 120


def fit_label_encoder(
    df: Optional[Union[pd.DataFrame, pd.Series, Sequence[str]]] = None,
) -> LabelBinarizer:
    """Fit a LabelBinarizer on unique dog breed labels.

    Parameters
    ----------
    df : pd.DataFrame, pd.Series, or sequence of str, optional
        Data containing breed labels. If None, load_labels() is invoked to load
        the full dataset metadata.

    Returns
    -------
    LabelBinarizer
        Fitted LabelBinarizer with exactly 120 deterministic classes.

    Raises
    ------
    ValueError
        If 'breed' column is missing from DataFrame, or if the fitted encoder
        does not have exactly 120 unique classes.
    """
    if df is None:
        raw_df = load_labels()
        breeds = raw_df["breed"]
    elif isinstance(df, pd.DataFrame):
        if "breed" not in df.columns:
            raise ValueError(f"DataFrame must contain 'breed' column. Found columns: {list(df.columns)}")
        breeds = df["breed"]
    elif isinstance(df, (pd.Series, np.ndarray, list, tuple)):
        breeds = df
    else:
        raise TypeError(f"df must be a pd.DataFrame, pd.Series, or sequence, got {type(df).__name__}")

    encoder = LabelBinarizer()
    encoder.fit(breeds)

    actual_classes = len(encoder.classes_)
    if actual_classes != NUM_CLASSES:
        raise ValueError(
            f"Validation failed: expected exactly {NUM_CLASSES} classes, but found {actual_classes}."
        )

    return encoder


def encode_labels(
    breeds: Union[pd.Series, Sequence[str], np.ndarray, str],
    encoder: LabelBinarizer,
) -> np.ndarray:
    """Encode breed labels into one-hot float32 arrays of shape (N, 120).

    Parameters
    ----------
    breeds : pd.Series, sequence of str, np.ndarray, or str
        Breed label strings to encode.
    encoder : LabelBinarizer
        Fitted LabelBinarizer instance.

    Returns
    -------
    np.ndarray
        One-hot encoded array with shape (N, 120) and dtype np.float32,
        where each row sums to 1.0 and has exactly one active class.

    Raises
    ------
    ValueError
        If any breed in breeds is not recognized by the encoder.
    TypeError
        If input type is unsupported.
    """
    if isinstance(breeds, str):
        breed_list = [breeds]
    elif isinstance(breeds, (pd.Series, np.ndarray, list, tuple)):
        breed_list = [str(b) for b in breeds]
    else:
        raise TypeError(
            f"breeds must be a pandas Series, list, NumPy array, or string, got {type(breeds).__name__}"
        )

    # Validate against known classes
    known_classes = set(encoder.classes_)
    unknown_breeds = [b for b in breed_list if b not in known_classes]
    if unknown_breeds:
        unique_unknown = sorted(set(unknown_breeds))
        raise ValueError(
            f"Unknown breed(s) not found in encoder classes: {unique_unknown[:5]} "
            f"(total unknown occurrences: {len(unknown_breeds)})"
        )

    # Transform and cast to float32
    encoded = encoder.transform(breed_list).astype(np.float32)

    return encoded


def get_class_to_index(encoder: LabelBinarizer) -> Dict[str, int]:
    """Return dictionary mapping breed name string to integer class index (0..119).

    Parameters
    ----------
    encoder : LabelBinarizer
        Fitted LabelBinarizer instance.

    Returns
    -------
    dict
        Mapping from breed name to class index.
    """
    return {str(cls_name): idx for idx, cls_name in enumerate(encoder.classes_)}


def get_index_to_class(encoder: LabelBinarizer) -> Dict[int, str]:
    """Return dictionary mapping integer class index (0..119) to breed name string.

    Parameters
    ----------
    encoder : LabelBinarizer
        Fitted LabelBinarizer instance.

    Returns
    -------
    dict
        Mapping from class index to breed name.
    """
    return {idx: str(cls_name) for idx, cls_name in enumerate(encoder.classes_)}


def get_class_mappings(encoder: LabelBinarizer) -> Tuple[Dict[str, int], Dict[int, str]]:
    """Return both class_to_index and index_to_class mappings.

    Parameters
    ----------
    encoder : LabelBinarizer
        Fitted LabelBinarizer instance.

    Returns
    -------
    tuple of (dict, dict)
        (class_to_index, index_to_class)
    """
    return get_class_to_index(encoder), get_index_to_class(encoder)


def get_class_index(breed: str, encoder: LabelBinarizer) -> int:
    """Retrieve integer class index (0..119) for a given breed name.

    Parameters
    ----------
    breed : str
        Breed name string.
    encoder : LabelBinarizer
        Fitted LabelBinarizer instance.

    Returns
    -------
    int
        Class index in range 0..119.

    Raises
    ------
    ValueError
        If breed does not exist in encoder classes.
    TypeError
        If breed is not a string.
    """
    if not isinstance(breed, str):
        raise TypeError(f"breed must be a string, got {type(breed).__name__}")

    indices = np.where(encoder.classes_ == breed)[0]
    if len(indices) == 0:
        raise ValueError(
            f"Unknown breed '{breed}'. Breed does not exist in encoder classes ({len(encoder.classes_)} classes)."
        )

    return int(indices[0])


def decode_one_hot(
    one_hot: Union[np.ndarray, Sequence[float], Sequence[Sequence[float]]],
    encoder: LabelBinarizer,
) -> Union[str, np.ndarray]:
    """Decode one-hot vector(s) back into original breed name(s).

    Parameters
    ----------
    one_hot : np.ndarray or sequence
        1D one-hot vector of length 120, or 2D array of shape (N, 120).
    encoder : LabelBinarizer
        Fitted LabelBinarizer instance.

    Returns
    -------
    str or np.ndarray
        If input is a 1D vector, returns a single breed name string.
        If input is a 2D array, returns a 1D NumPy array of breed name strings.

    Raises
    ------
    ValueError
        If dimensions do not match encoder class count or array is invalid.
    """
    arr = np.asarray(one_hot)
    num_classes = len(encoder.classes_)

    if arr.ndim == 1:
        if arr.shape[0] != num_classes:
            raise ValueError(
                f"Invalid one-hot vector length: expected {num_classes}, got {arr.shape[0]}"
            )
        idx = int(np.argmax(arr))
        return str(encoder.classes_[idx])
    elif arr.ndim == 2:
        if arr.shape[1] != num_classes:
            raise ValueError(
                f"Invalid one-hot array second dimension: expected {num_classes}, got {arr.shape[1]}"
            )
        decoded = encoder.inverse_transform(arr)
        return decoded
    else:
        raise ValueError(f"Expected 1D or 2D array for one_hot, got shape {arr.shape}")


def run_validation() -> Dict[str, Union[bool, int, str]]:
    """Execute complete validation suite for Task 1.4 and return metrics."""
    df = load_labels()

    # 1. Fit encoder
    encoder = fit_label_encoder(df)
    num_classes = len(encoder.classes_)

    # 2. Encode full dataset
    encoded = encode_labels(df["breed"], encoder)

    # 3. Check shapes and dtypes
    input_samples = len(df["breed"])
    encoded_shape = encoded.shape
    encoded_dtype = str(encoded.dtype)
    min_val = float(encoded.min())
    max_val = float(encoded.max())

    # 4. One-hot integrity
    row_sums = encoded.sum(axis=1)
    rows_sum_ne_1 = int(np.sum(~np.isclose(row_sums, 1.0)))
    active_counts = (encoded == 1.0).sum(axis=1)
    rows_multiple_active = int(np.sum(active_counts > 1))
    rows_no_active = int(np.sum(active_counts == 0))
    nan_count = int(np.isnan(encoded).sum())
    inf_count = int(np.isinf(encoded).sum())
    valid_one_hot_rows = int(np.sum(np.isclose(row_sums, 1.0) & (active_counts == 1)))
    one_hot_pass = (
        rows_sum_ne_1 == 0
        and rows_multiple_active == 0
        and rows_no_active == 0
        and nan_count == 0
        and inf_count == 0
        and valid_one_hot_rows == input_samples
    )

    # 5. Class mappings
    class_to_index, index_to_class = get_class_mappings(encoder)
    mapping_pass = (
        len(class_to_index) == NUM_CLASSES
        and len(index_to_class) == NUM_CLASSES
        and min(class_to_index.values()) == 0
        and max(class_to_index.values()) == NUM_CLASSES - 1
    )

    # 6. Round-trip validation on 20 random samples (random_state = 42)
    sample_df = df.sample(n=20, random_state=42)
    sample_breeds = sample_df["breed"].tolist()
    sample_encoded = encode_labels(sample_breeds, encoder)
    matched_samples = 0
    for i, orig_breed in enumerate(sample_breeds):
        one_hot_vec = sample_encoded[i]
        decoded_breed = decode_one_hot(one_hot_vec, encoder)
        if decoded_breed == orig_breed:
            matched_samples += 1
    round_trip_samples_pass = matched_samples == 20

    # 7. All 120 classes round-trip
    unique_breeds = encoder.classes_
    matched_all_classes = 0
    for breed in unique_breeds:
        c_idx = get_class_index(breed, encoder)
        vec = np.zeros(num_classes, dtype=np.float32)
        vec[c_idx] = 1.0
        decoded_breed = decode_one_hot(vec, encoder)
        if decoded_breed == breed:
            matched_all_classes += 1
    all_classes_pass = matched_all_classes == NUM_CLASSES

    # 8. Unknown label handling
    unknown_breed = "this_breed_does_not_exist"
    unknown_handled = False
    try:
        get_class_index(unknown_breed, encoder)
    except ValueError:
        try:
            encode_labels([unknown_breed], encoder)
        except ValueError:
            unknown_handled = True

    # 9. Class order determinism
    encoder_a = fit_label_encoder()
    encoder_b = fit_label_encoder()
    deterministic_pass = bool(np.array_equal(encoder_a.classes_, encoder_b.classes_))

    print("LABEL ENCODING VALIDATION")
    print()
    print(f"Samples: {input_samples}")
    print(f"Classes: {num_classes}")
    print(f"Encoded shape: {encoded_shape}")
    print(f"Encoded dtype: {encoded_dtype}")
    print()
    print(f"One-hot integrity: {'PASS' if one_hot_pass else 'FAIL'}")
    print(f"Round-trip samples: {'PASS' if round_trip_samples_pass else 'FAIL'}")
    print(f"All classes round-trip: {'PASS' if all_classes_pass else 'FAIL'}")
    print(f"Unknown label handling: {'PASS' if unknown_handled else 'FAIL'}")
    print(f"Class order deterministic: {'PASS' if deterministic_pass else 'FAIL'}")
    print()
    print("Label encoding validation completed successfully.")

    return {
        "input_samples": input_samples,
        "num_classes": num_classes,
        "encoded_shape": encoded_shape,
        "encoded_dtype": encoded_dtype,
        "min_val": min_val,
        "max_val": max_val,
        "rows_sum_ne_1": rows_sum_ne_1,
        "rows_multiple_active": rows_multiple_active,
        "rows_no_active": rows_no_active,
        "nan_count": nan_count,
        "inf_count": inf_count,
        "valid_one_hot_rows": valid_one_hot_rows,
        "class_to_index_count": len(class_to_index),
        "index_to_class_count": len(index_to_class),
        "index_range": f"{min(class_to_index.values())}..{max(class_to_index.values())}",
        "matched_samples": matched_samples,
        "matched_all_classes": matched_all_classes,
        "unknown_handled": unknown_handled,
        "deterministic_pass": deterministic_pass,
    }


if __name__ == "__main__":
    run_validation()
