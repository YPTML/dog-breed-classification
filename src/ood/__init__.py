"""Open-set / out-of-distribution utilities for the 120-class dog model.

The package deliberately lives beside the closed-set classifier.  OOD scores
are computed after the classifier has produced its normal 120-class output;
there is no ``not_dog`` training class.
"""

from importlib import import_module


_LAZY_EXPORTS = {
    "maximum_softmax_probability": ("msp", "maximum_softmax_probability"),
    "score": ("msp", "score"),
    "predict_ood": ("msp", "predict_ood"),
    "msp_ood_score": ("msp", "msp_ood_score"),
    "calibrate_msp_threshold": ("msp", "calibrate_msp_threshold"),
    "distribution_statistics": ("msp", "distribution_statistics"),
    "predictive_entropy": ("entropy", "predictive_entropy"),
    "entropy_ood_score": ("entropy", "entropy_ood_score"),
    "energy_score": ("energy", "energy_score"),
    "energy_ood_score": ("energy", "energy_ood_score"),
    "build_logits_model": ("energy", "build_logits_model"),
    "extract_logits": ("energy", "extract_logits"),
    "evaluate_energy_grid": ("energy", "evaluate_energy_grid"),
    "evaluate_energy_calibration": ("energy", "evaluate_energy_calibration"),
    "calibrate_threshold": ("calibration", "calibrate_threshold"),
    "calibrate_strategy": ("calibration", "calibrate_strategy"),
    "calibrate_thresholds": ("calibration", "calibrate_thresholds"),
    "save_thresholds": ("calibration", "save_thresholds"),
    "load_thresholds": ("calibration", "load_thresholds"),
    "MAX_F1": ("calibration", "MAX_F1"),
    "TARGET_ID_TPR": ("calibration", "TARGET_ID_TPR"),
    "MIN_WEIGHTED_ERROR": ("calibration", "MIN_WEIGHTED_ERROR"),
    "STRATEGIES": ("calibration", "STRATEGIES"),
    "evaluate_ood_scores": ("evaluate_ood", "evaluate_ood_scores"),
    "compute_ood_metrics": ("evaluate_ood", "compute_ood_metrics"),
    "evaluate_detector": ("evaluate_ood", "evaluate_detector"),
    "evaluate_detectors": ("evaluate_ood", "evaluate_detectors"),
    "DetectorSpec": ("evaluate_ood", "DetectorSpec"),
    "DETECTORS": ("evaluate_ood", "DETECTORS"),
    "OOD_DETECTORS": ("evaluate_ood", "OOD_DETECTORS"),
    "DETECTOR_SPECS": ("evaluate_ood", "DETECTOR_SPECS"),
    "get_detector": ("evaluate_ood", "get_detector"),
    "rejection_error_rates": ("evaluate_ood", "rejection_error_rates"),
    "cosine_distance": ("embedding_distance", "cosine_distance"),
    "euclidean_distance": ("embedding_distance", "euclidean_distance"),
    "mahalanobis_distance": ("embedding_distance", "mahalanobis_distance"),
    "fit_mahalanobis_statistics": ("embedding_distance", "fit_mahalanobis_statistics"),
    "regularize_covariance": ("embedding_distance", "regularize_covariance"),
    "EmbeddingDistanceDetector": ("embedding_distance", "EmbeddingDistanceDetector"),
}


def __getattr__(name: str):
    """Load public APIs lazily so ``python -m src.ood.<module>`` stays clean."""

    target = _LAZY_EXPORTS.get(name)
    if target is not None:
        module_name, attribute_name = target
        module = import_module(f".{module_name}", __name__)
        value = getattr(module, attribute_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = list(_LAZY_EXPORTS)
