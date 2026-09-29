"""The public three-state decision layer for model inference.

The classifier always produces a dog-breed distribution.  This module is the
boundary between that internal output and the UI: it turns the distribution
and an OOD score into ``DOG``, ``UNCERTAIN`` or ``NOT_DOG``.

Thresholds are policy data, not model code.  By default they are loaded from
``outputs/calibration/ood_thresholds.json``.  Callers may pass an already
loaded mapping (which is useful for a long-lived application and for tests).
The loader accepts both the calibration artifact's top-level ``threshold`` /
``ood_threshold`` keys and the runtime policy's nested ``thresholds`` object.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_THRESHOLD_PATH = PROJECT_ROOT / "outputs" / "calibration" / "ood_thresholds.json"


class DecisionStatus(str, Enum):
    """The only statuses exposed by the decision layer."""

    DOG = "DOG"
    NOT_DOG = "NOT_DOG"
    UNCERTAIN = "UNCERTAIN"


@dataclass(frozen=True)
class Decision:
    """A UI-safe decision result.

    ``prediction`` is a breed record only for ``DOG``.  For the other two
    states it is a non-breed sentinel, so a caller cannot accidentally render
    a classifier guess as a breed.
    """

    status: DecisionStatus
    is_dog: bool | None
    prediction: dict[str, Any] | str
    message: str | None = None

    def as_dict(self) -> dict[str, Any]:
        """Return the decision portion of the API response."""

        result: dict[str, Any] = {
            "decision": self.status.value,
            "status": self.status.value,
            "is_dog": self.is_dog,
            "prediction": self.prediction,
        }
        if self.message:
            result["message"] = self.message
        return result

    def as_ui_dict(self, top5: Sequence[Mapping[str, Any]] | None = None) -> dict[str, Any]:
        """Return a response fragment with Top-5 hidden for non-DOG states."""

        result = self.as_dict()
        result["top5"] = (
            [dict(item) for item in top5] if self.status is DecisionStatus.DOG and top5 else []
        )
        return result


def load_thresholds(path: str | Path = DEFAULT_THRESHOLD_PATH) -> dict[str, Any]:
    """Load the calibrated OOD policy from JSON.

    The function deliberately does not provide a numeric fallback.  A missing
    or malformed calibration artifact is a deployment/configuration error and
    must not silently turn into an arbitrary OOD threshold.
    """

    source = Path(path)
    try:
        with source.open(encoding="utf-8") as handle:
            data = json.load(handle)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"OOD threshold artifact not found: {source}. "
            "Run calibration before enabling the decision layer."
        ) from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read OOD threshold artifact {source}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError("OOD threshold artifact must be a JSON object")
    return data


# Explicit name for callers that want to distinguish OOD policy loading from
# other threshold artifacts.
load_ood_thresholds = load_thresholds


def _policy_values(thresholds: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return the mapping containing decision thresholds."""

    values: Mapping[str, Any] = thresholds
    # Calibration artifacts are flat; config files may be nested under ``ood``
    # and/or ``thresholds``.  Supporting both keeps the decision layer
    # independent from the producer of the policy artifact.
    for key in ("ood", "thresholds"):
        nested = values.get(key)
        if isinstance(nested, Mapping):
            values = nested
    return values


def _first_value(values: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        if name in values and values[name] is not None:
            return values[name]
    return None


def _finite_number(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _direction(values: Mapping[str, Any]) -> bool:
    """Return whether a larger raw score means more likely OOD."""

    scopes: list[Mapping[str, Any]] = [values]
    nested_ood = values.get("ood")
    if isinstance(nested_ood, Mapping):
        scopes.append(nested_ood)
    policy = _policy_values(values)
    if all(policy is not scope for scope in scopes):
        scopes.append(policy)

    explicit = next(
        (
            candidate
            for scope in scopes
            for candidate in [_first_value(scope, "higher_score_is_ood")]
            if candidate is not None
        ),
        None,
    )
    if explicit is not None:
        if not isinstance(explicit, (bool, np.bool_)):
            raise ValueError("higher_score_is_ood must be boolean")
        return bool(explicit)

    direction = next(
        (
            candidate
            for scope in scopes
            for candidate in [_first_value(scope, "score_direction", "ood_score_direction")]
            if candidate is not None
        ),
        None,
    )
    if direction is None:
        # ``decide_prediction`` documents that its score argument is OOD
        # oriented.  This is a direction convention, not a threshold value.
        return True
    normalized = str(direction).strip().lower().replace("-", "_").replace(" ", "_")
    if normalized in {"higher_is_ood", "higher_score_is_ood", "higher", "greater", "ood_higher"}:
        return True
    if normalized in {"lower_is_ood", "lower_score_is_ood", "lower", "less", "ood_lower"}:
        return False
    raise ValueError("score_direction must be 'higher_is_ood' or 'lower_is_ood'")


def _threshold(values: Mapping[str, Any], *names: str) -> float:
    value = _first_value(values, *names)
    if value is None:
        expected = ", ".join(names)
        raise ValueError(f"OOD threshold artifact is missing one of: {expected}")
    return _finite_number(value, names[0])


def _is_ood(score: float, threshold: float, *, higher_score_is_ood: bool) -> bool:
    return score >= threshold if higher_score_is_ood else score <= threshold


def _in_uncertainty_region(
    score: float,
    strong_ood_threshold: float,
    values: Mapping[str, Any],
    *,
    higher_score_is_ood: bool,
) -> bool:
    """Check the optional calibrated OOD uncertainty band."""

    uncertainty_threshold = _first_value(
        values,
        "uncertain",
        "uncertain_threshold",
        "uncertainty_threshold",
        "uncertain_ood_threshold",
        "uncertainty_ood_threshold",
    )
    if uncertainty_threshold is None:
        return False
    uncertainty_threshold = _finite_number(uncertainty_threshold, "uncertain_threshold")

    if higher_score_is_ood:
        if uncertainty_threshold > strong_ood_threshold:
            raise ValueError("uncertain threshold must be <= the strong OOD threshold")
        return uncertainty_threshold <= score < strong_ood_threshold

    if uncertainty_threshold < strong_ood_threshold:
        raise ValueError("uncertain threshold must be >= the strong OOD threshold")
    return strong_ood_threshold < score <= uncertainty_threshold


def _breed_confidence_threshold(values: Mapping[str, Any]) -> float | None:
    value = _first_value(
        values,
        "breed_confidence",
        "breed_confidence_threshold",
        "min_breed_confidence",
    )
    if value is None:
        return None
    result = _finite_number(value, "breed_confidence_threshold")
    if not 0.0 <= result <= 1.0:
        raise ValueError("breed confidence threshold must be in [0, 1]")
    return result


def _validate_probabilities(probabilities: Sequence[float] | np.ndarray) -> np.ndarray:
    values = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    if values.size == 0 or not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("probabilities must be finite and non-negative")
    if not np.isclose(float(values.sum()), 1.0, atol=1e-4):
        raise ValueError("probabilities must sum to 1")
    return values


def decide_prediction(
    probabilities: Sequence[float] | np.ndarray,
    ood_score: float,
    class_names: Mapping[int, str] | Sequence[str],
    thresholds: Mapping[str, Any] | str | Path | None = None,
    *,
    threshold_path: str | Path = DEFAULT_THRESHOLD_PATH,
) -> Decision:
    """Apply the calibrated policy to one classifier output.

    The strong OOD check always runs first.  An optional uncertainty band in
    the artifact is checked next, followed by the optional minimum breed
    confidence policy.  Only the final DOG branch returns a breed record.
    """

    if thresholds is None:
        thresholds = load_thresholds(threshold_path)
    elif isinstance(thresholds, (str, Path)):
        thresholds = load_thresholds(thresholds)
    if not isinstance(thresholds, Mapping):  # defensive for runtime callers
        raise TypeError("thresholds must be a mapping or a JSON artifact path")

    values = _validate_probabilities(probabilities)
    score = _finite_number(ood_score, "ood_score")
    policy = _policy_values(thresholds)
    strong_ood_threshold = _threshold(
        policy,
        "not_dog",
        "strong_ood_threshold",
        "ood_threshold",
        "threshold",
    )
    higher_score_is_ood = _direction(thresholds)

    top_index = int(np.argmax(values))
    confidence = float(values[top_index])

    # Validate all configured policy fields regardless of the eventual branch.
    confidence_threshold = _breed_confidence_threshold(policy)
    is_uncertain = _in_uncertainty_region(
        score,
        strong_ood_threshold,
        policy,
        higher_score_is_ood=higher_score_is_ood,
    )
    if _is_ood(score, strong_ood_threshold, higher_score_is_ood=higher_score_is_ood):
        return Decision(
            DecisionStatus.NOT_DOG,
            False,
            "not_dog",
            "Ảnh không được nhận diện là chó",
        )
    if confidence_threshold is not None and confidence < confidence_threshold:
        is_uncertain = True
    if is_uncertain:
        return Decision(
            DecisionStatus.UNCERTAIN,
            None,
            "uncertain",
            "Không thể xác định chắc chắn",
        )

    try:
        breed = class_names[top_index]  # type: ignore[index]
    except (IndexError, KeyError, TypeError) as exc:
        raise ValueError(f"class_names has no entry for predicted class {top_index}") from exc
    top_prediction = {
        "index": top_index,
        "breed": str(breed),
        "confidence": confidence,
    }
    return Decision(DecisionStatus.DOG, True, top_prediction)


# Concise aliases retained for callers that use an imperative name.
decide = decide_prediction
make_decision = decide_prediction


__all__ = [
    "DEFAULT_THRESHOLD_PATH",
    "Decision",
    "DecisionStatus",
    "decide",
    "decide_prediction",
    "load_ood_thresholds",
    "load_thresholds",
    "make_decision",
]
