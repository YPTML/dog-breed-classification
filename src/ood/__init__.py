"""Open-set / out-of-distribution utilities for the 120-class dog model.

The package deliberately lives beside the closed-set classifier.  OOD scores
are computed after the classifier has produced its normal 120-class output;
there is no ``not_dog`` training class.
"""

from .msp import (
    calibrate_msp_threshold,
    distribution_statistics,
    maximum_softmax_probability,
    msp_ood_score,
    predict_ood,
    score,
)
from .entropy import predictive_entropy, entropy_ood_score
from .energy import (
    build_logits_model,
    energy_score,
    energy_ood_score,
    evaluate_energy_calibration,
    evaluate_energy_grid,
    extract_logits,
)
from .calibration import (
    MAX_F1,
    MIN_WEIGHTED_ERROR,
    STRATEGIES,
    TARGET_ID_TPR,
    calibrate_strategy,
    calibrate_threshold,
    calibrate_thresholds,
    load_thresholds,
    save_thresholds,
)
from .evaluate_ood import (
    DETECTORS,
    DETECTOR_SPECS,
    OOD_DETECTORS,
    DetectorSpec,
    compute_ood_metrics,
    evaluate_detector,
    evaluate_detectors,
    evaluate_ood_scores,
    get_detector,
    rejection_error_rates,
)

_EMBEDDING_DISTANCE_EXPORTS = {
    "cosine_distance",
    "euclidean_distance",
    "mahalanobis_distance",
    "fit_mahalanobis_statistics",
    "regularize_covariance",
    "EmbeddingDistanceDetector",
}


def __getattr__(name: str):
    """Load embedding-distance APIs lazily so ``python -m`` stays clean."""

    if name in _EMBEDDING_DISTANCE_EXPORTS:
        from . import embedding_distance

        value = getattr(embedding_distance, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "maximum_softmax_probability",
    "score",
    "predict_ood",
    "msp_ood_score",
    "calibrate_msp_threshold",
    "distribution_statistics",
    "predictive_entropy",
    "entropy_ood_score",
    "energy_score",
    "energy_ood_score",
    "build_logits_model",
    "extract_logits",
    "evaluate_energy_grid",
    "evaluate_energy_calibration",
    "calibrate_threshold",
    "calibrate_strategy",
    "calibrate_thresholds",
    "save_thresholds",
    "load_thresholds",
    "MAX_F1",
    "TARGET_ID_TPR",
    "MIN_WEIGHTED_ERROR",
    "STRATEGIES",
    "evaluate_ood_scores",
    "compute_ood_metrics",
    "evaluate_detector",
    "evaluate_detectors",
    "DetectorSpec",
    "DETECTORS",
    "OOD_DETECTORS",
    "DETECTOR_SPECS",
    "get_detector",
    "rejection_error_rates",
    "cosine_distance",
    "euclidean_distance",
    "mahalanobis_distance",
    "fit_mahalanobis_statistics",
    "regularize_covariance",
    "EmbeddingDistanceDetector",
]
