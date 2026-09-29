"""Metrics for evaluating out-of-distribution (OOD) score detectors.

The metric implementation works with one-dimensional score arrays. ID
samples are treated as the negative class and OOD samples as the positive
class. ``higher_score_is_ood`` is always explicit for new evaluations, or is
resolved from :data:`DETECTORS` when a detector name is supplied.

The detector directions are deliberately kept here rather than inferred from
the metric code:

* MSP is an in-distribution confidence score, so lower values indicate OOD.
* Entropy, energy, cosine distance, and Mahalanobis distance increase toward
  OOD for the score functions used by this project.

``evaluate_ood_scores`` normalizes every detector to an internal convention in
which larger values mean OOD. This makes AUROC, AUPR, and FPR95 comparable
without hard-coding one direction into those calculations.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class DetectorSpec:
    """Configuration needed to interpret one detector's raw score."""

    name: str
    higher_score_is_ood: bool


# Keep the direction next to the detector name. A caller evaluating raw MSP
# values therefore cannot accidentally use the direction of an OOD-oriented
# transform such as ``1 - MSP``.
DETECTORS: Mapping[str, DetectorSpec] = MappingProxyType(
    {
        "msp": DetectorSpec("msp", higher_score_is_ood=False),
        "entropy": DetectorSpec("entropy", higher_score_is_ood=True),
        "energy": DetectorSpec("energy", higher_score_is_ood=True),
        "cosine": DetectorSpec("cosine", higher_score_is_ood=True),
        "mahalanobis": DetectorSpec("mahalanobis", higher_score_is_ood=True),
    }
)

# Public aliases make the registry easy to discover without duplicating the
# source of truth. The embedding-prefixed names are accepted by persisted
# configuration files used elsewhere in this repository.
OOD_DETECTORS = DETECTORS
DETECTOR_SPECS = DETECTORS
_DETECTOR_ALIASES = {
    "embedding_cosine": "cosine",
    "embedding_mahalanobis": "mahalanobis",
}
_UNSET_DIRECTION = object()


def get_detector(name: str) -> DetectorSpec:
    """Return the declared score-direction configuration for ``name``."""

    if not isinstance(name, str) or not name.strip():
        raise ValueError("detector must be a non-empty string")
    normalized = name.strip().lower().replace("-", "_")
    normalized = _DETECTOR_ALIASES.get(normalized, normalized)
    try:
        return DETECTORS[normalized]
    except KeyError as exc:
        choices = ", ".join(DETECTORS)
        raise ValueError(f"unknown detector {name!r}; expected one of {choices}") from exc


def _finite(values: Sequence[float] | np.ndarray, name: str) -> np.ndarray:
    """Normalize a score collection and reject empty/non-finite input."""

    scores = np.asarray(values, dtype=np.float64).reshape(-1)
    if scores.size == 0 or not np.isfinite(scores).all():
        raise ValueError(f"{name} must contain at least one finite value")
    return scores


def _resolve_direction(
    detector: str | DetectorSpec | None,
    higher_score_is_ood: bool | None,
) -> tuple[str | None, bool]:
    """Resolve a detector declaration without silently guessing its direction."""

    spec: DetectorSpec | None
    if detector is None:
        spec = None
    elif isinstance(detector, DetectorSpec):
        spec = detector
    else:
        spec = get_detector(detector)

    if higher_score_is_ood is not None and not isinstance(higher_score_is_ood, (bool, np.bool_)):
        raise ValueError("higher_score_is_ood must be a boolean")

    if spec is not None:
        declared = bool(spec.higher_score_is_ood)
        if higher_score_is_ood is not None and bool(higher_score_is_ood) != declared:
            raise ValueError(
                f"direction for detector {spec.name!r} conflicts with its declaration"
            )
        return spec.name, declared

    if higher_score_is_ood is None:
        raise ValueError(
            "score direction is required; pass detector=... or "
            "higher_score_is_ood=True/False"
        )
    return None, bool(higher_score_is_ood)


def _ood_oriented(scores: np.ndarray, higher_score_is_ood: bool) -> np.ndarray:
    """Return scores where larger values consistently mean more OOD."""

    return scores if higher_score_is_ood else -scores


def _rank_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Compute ROC AUC using tie-aware average ranks."""

    positives = labels == 1
    negatives = labels == 0
    n_pos = int(positives.sum())
    n_neg = int(negatives.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(sorted_scores.size, dtype=np.float64)
    start = 0
    while start < sorted_scores.size:
        end = start + 1
        while end < sorted_scores.size and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[start:end] = (start + 1 + end) / 2.0
        start = end

    original_ranks = np.empty_like(ranks)
    original_ranks[order] = ranks
    sum_positive = original_ranks[positives].sum()
    return float(
        (sum_positive - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    )


def _average_precision(labels: np.ndarray, scores: np.ndarray, positive: int) -> float:
    """Compute average precision, treating tied scores as one threshold."""

    target = labels == positive
    total = int(target.sum())
    if total == 0:
        return float("nan")

    order = np.argsort(-scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_target = target[order]
    true_positive = 0
    false_positive = 0
    previous_recall = 0.0
    result = 0.0
    start = 0
    while start < sorted_scores.size:
        end = start + 1
        while end < sorted_scores.size and sorted_scores[end] == sorted_scores[start]:
            end += 1
        hits = int(sorted_target[start:end].sum())
        true_positive += hits
        false_positive += (end - start) - hits
        precision = true_positive / (true_positive + false_positive)
        recall = true_positive / total
        result += (recall - previous_recall) * precision
        previous_recall = recall
        start = end
    return float(result)


def _fpr95_operating_point(
    id_scores: np.ndarray, ood_scores: np.ndarray
) -> tuple[float, float]:
    """Return ``(threshold, fpr)`` at the first operating point with TPR >= .95."""

    scores = np.concatenate([id_scores, ood_scores])
    thresholds = np.unique(scores)[::-1]
    id_count = id_scores.size
    for threshold in thresholds:
        predicted_ood = scores >= threshold
        true_positive_rate = float(np.mean(predicted_ood[id_count:]))
        if true_positive_rate >= 0.95:
            false_positive_rate = float(np.mean(predicted_ood[:id_count]))
            return float(threshold), false_positive_rate

    # The lowest observed score always predicts every sample as OOD, so this
    # fallback is defensive only (and keeps the function total if the logic
    # above is ever changed).
    threshold = float(np.min(scores))
    return threshold, 1.0


def _classification_metrics(
    id_scores: np.ndarray,
    ood_scores: np.ndarray,
    threshold: float,
) -> tuple[float, float, float, dict[str, int]]:
    """Calculate OOD precision, recall, F1, and confusion counts."""

    scores = np.concatenate([id_scores, ood_scores])
    labels = np.concatenate(
        [np.zeros(id_scores.size, dtype=np.int8), np.ones(ood_scores.size, dtype=np.int8)]
    )
    predicted_ood = scores >= threshold
    true_positive = int(np.sum(predicted_ood & (labels == 1)))
    false_positive = int(np.sum(predicted_ood & (labels == 0)))
    false_negative = int(np.sum(~predicted_ood & (labels == 1)))
    true_negative = int(np.sum(~predicted_ood & (labels == 0)))

    precision = (
        true_positive / (true_positive + false_positive)
        if true_positive + false_positive
        else 0.0
    )
    recall = (
        true_positive / (true_positive + false_negative)
        if true_positive + false_negative
        else 0.0
    )
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    counts = {
        "true_positive": true_positive,
        "true_negative": true_negative,
        "false_positive": false_positive,
        "false_negative": false_negative,
    }
    return float(precision), float(recall), float(f1), counts


def evaluate_ood_scores(
    id_scores: Sequence[float] | np.ndarray,
    ood_scores: Sequence[float] | np.ndarray,
    higher_score_is_ood: bool | None | object = _UNSET_DIRECTION,
    *,
    detector: str | DetectorSpec | None = None,
    threshold: float | None = None,
) -> dict[str, float | int | str | bool]:
    """Evaluate one detector on ID and OOD scores.

    ``higher_score_is_ood`` should be supplied unless ``detector`` is used;
    the detector registry provides the direction for the five supported
    scoring methods. It defaults to ``True`` only for compatibility with the
    older API, whose callers supplied already OOD-oriented scores. ``threshold``
    is in raw detector-score units. If it is omitted, the threshold at the
    FPR95 operating point is used for OOD Precision/Recall/F1.

    The returned lower-case keys are ``auroc``, ``fpr95``, ``aupr_in``,
    ``aupr_out``, ``ood_precision``, ``ood_recall``, and ``ood_f1``. The
    selected threshold and confusion counts are included as supporting fields.
    """

    # Preserve the old API for already OOD-oriented arrays when no detector is
    # named, while allowing detector declarations to supply their own direction
    # without being overridden by that compatibility default.
    if higher_score_is_ood is _UNSET_DIRECTION:
        direction_argument: bool | None = None if detector is not None else True
    else:
        direction_argument = higher_score_is_ood  # type: ignore[assignment]
    detector_name, direction = _resolve_direction(detector, direction_argument)
    ids = _finite(id_scores, "id_scores")
    oods = _finite(ood_scores, "ood_scores")
    oriented_ids = _ood_oriented(ids, direction)
    oriented_oods = _ood_oriented(oods, direction)
    scores = np.concatenate([oriented_ids, oriented_oods])
    labels = np.concatenate(
        [np.zeros(ids.size, dtype=np.int8), np.ones(oods.size, dtype=np.int8)]
    )

    fpr95_threshold, fpr95 = _fpr95_operating_point(oriented_ids, oriented_oods)
    if threshold is None:
        oriented_threshold = fpr95_threshold
        raw_threshold = oriented_threshold if direction else -oriented_threshold
        threshold_source = "fpr95"
    else:
        raw_threshold = float(threshold)
        if not np.isfinite(raw_threshold):
            raise ValueError("threshold must be finite")
        oriented_threshold = raw_threshold if direction else -raw_threshold
        threshold_source = "provided"

    precision, recall, f1, counts = _classification_metrics(
        oriented_ids, oriented_oods, oriented_threshold
    )
    metrics: dict[str, float | int | str | bool] = {
        "auroc": _rank_auc(labels, scores),
        "fpr95": float(fpr95),
        "aupr_in": _average_precision(1 - labels, -scores, 1),
        "aupr_out": _average_precision(labels, scores, 1),
        "ood_precision": precision,
        "ood_recall": recall,
        "ood_f1": f1,
        "threshold": float(raw_threshold),
        "threshold_source": threshold_source,
        "higher_score_is_ood": bool(direction),
        "id_count": int(ids.size),
        "ood_count": int(oods.size),
        **counts,
    }
    if detector_name is not None:
        metrics["detector"] = detector_name
    return metrics


def evaluate_detector(
    detector: str | DetectorSpec,
    id_scores: Sequence[float] | np.ndarray,
    ood_scores: Sequence[float] | np.ndarray,
    *,
    threshold: float | None = None,
) -> dict[str, float | int | str | bool]:
    """Convenience wrapper that evaluates a declared detector."""

    return evaluate_ood_scores(
        id_scores,
        ood_scores,
        higher_score_is_ood=None,
        detector=detector,
        threshold=threshold,
    )


def evaluate_detectors(
    score_sets: Mapping[str, tuple[Sequence[float] | np.ndarray, Sequence[float] | np.ndarray]],
    *,
    thresholds: Mapping[str, float] | None = None,
) -> dict[str, dict[str, float | int | str | bool]]:
    """Evaluate several named detectors from ``{name: (id, ood)}`` scores."""

    threshold_map = {} if thresholds is None else thresholds
    results: dict[str, dict[str, float | int | str | bool]] = {}
    for name, (ids, oods) in score_sets.items():
        results[name] = evaluate_detector(
            name,
            ids,
            oods,
            threshold=threshold_map.get(name),
        )
    return results


def rejection_error_rates(
    id_scores: Sequence[float] | np.ndarray,
    ood_scores: Sequence[float] | np.ndarray,
    threshold: float,
    *,
    higher_score_is_ood: bool = True,
) -> dict[str, float]:
    """Return false-acceptance/rejection rates at a raw score threshold.

    This is retained for the calibration modules. New code should pass the
    detector direction explicitly, especially for raw MSP scores.
    """

    ids = _finite(id_scores, "id_scores")
    oods = _finite(ood_scores, "ood_scores")
    value = float(threshold)
    if not np.isfinite(value):
        raise ValueError("threshold must be finite")
    if not isinstance(higher_score_is_ood, (bool, np.bool_)):
        raise ValueError("higher_score_is_ood must be a boolean")
    if higher_score_is_ood:
        ood_rejected = oods >= value
        id_rejected = ids >= value
    else:
        ood_rejected = oods <= value
        id_rejected = ids <= value
    return {
        "false_acceptance_rate": float(np.mean(~ood_rejected)),
        "false_rejection_rate": float(np.mean(id_rejected)),
    }


# Backward-compatible aliases used by earlier OOD modules and convenient for
# callers that prefer the word "metrics" in the function name.
compute_ood_metrics = evaluate_ood_scores
evaluate = evaluate_ood_scores


__all__ = [
    "DetectorSpec",
    "DETECTORS",
    "OOD_DETECTORS",
    "DETECTOR_SPECS",
    "get_detector",
    "evaluate_ood_scores",
    "evaluate_detector",
    "evaluate_detectors",
    "compute_ood_metrics",
    "rejection_error_rates",
]
