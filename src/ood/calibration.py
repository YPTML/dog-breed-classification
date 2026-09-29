"""Calibrate OOD decision thresholds from ID validation and OOD calibration scores.

This module intentionally accepts *scores*, rather than loading images or a
model.  The caller is responsible for producing two disjoint score arrays:

* ``id_scores``: scores from the ID validation split only;
* ``ood_scores``: scores from the OOD calibration split only.

The final test split is not an input to this module.  The command-line
interface accepts an ``.npz``/``.json`` score artifact containing those two
arrays and writes ``outputs/calibration/ood_thresholds.json``.

All selection is performed after orienting scores so that larger values mean
"more likely OOD".  Thresholds returned in the public result remain in the
original score units, which is important for detectors such as raw MSP where
lower values indicate OOD.
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "outputs" / "calibration" / "ood_thresholds.json"
DEFAULT_SCORE_ARTIFACT = PROJECT_ROOT / "outputs" / "calibration" / "ood_scores.npz"
DEFAULT_TARGET_TPR = 0.95
DEFAULT_FALSE_ACCEPTANCE_WEIGHT = 1.0
DEFAULT_FALSE_REJECTION_WEIGHT = 1.0

MAX_F1 = "max_ood_f1"
TARGET_ID_TPR = "target_id_tpr"
MIN_WEIGHTED_ERROR = "min_weighted_error"
STRATEGIES = (MAX_F1, TARGET_ID_TPR, MIN_WEIGHTED_ERROR)

_STRATEGY_ALIASES = {
    "max_f1": MAX_F1,
    "maximize_f1": MAX_F1,
    "max_ood_f1": MAX_F1,
    "target_tpr": TARGET_ID_TPR,
    "target_id_tpr": TARGET_ID_TPR,
    "target_id_tpr_95": TARGET_ID_TPR,
    "weighted_error": MIN_WEIGHTED_ERROR,
    "min_weighted_error": MIN_WEIGHTED_ERROR,
    "weighted_false_error": MIN_WEIGHTED_ERROR,
}


def _finite_scores(values: Iterable[float] | np.ndarray, name: str) -> np.ndarray:
    """Return a non-empty, finite one-dimensional float score array."""

    result = np.asarray(values, dtype=np.float64).reshape(-1)
    if result.size == 0:
        raise ValueError(f"{name} must contain at least one score")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must contain only finite scores")
    return result


def _resolve_direction(
    score_direction: str | bool,
    higher_score_is_ood: bool | None,
) -> tuple[str, bool]:
    """Resolve direction aliases to a canonical label and boolean."""

    if higher_score_is_ood is not None:
        if not isinstance(higher_score_is_ood, (bool, np.bool_)):
            raise ValueError("higher_score_is_ood must be a boolean")
        direction = bool(higher_score_is_ood)
    elif isinstance(score_direction, (bool, np.bool_)):
        direction = bool(score_direction)
    elif isinstance(score_direction, str):
        normalized = score_direction.strip().lower().replace("-", "_").replace(" ", "_")
        if normalized in {
            "higher_is_ood",
            "higher_score_is_ood",
            "higher",
            "greater",
            "ood_higher",
        }:
            direction = True
        elif normalized in {
            "lower_is_ood",
            "lower_score_is_ood",
            "lower",
            "less",
            "ood_lower",
        }:
            direction = False
        else:
            raise ValueError(
                "score_direction must be 'higher_is_ood' or 'lower_is_ood'"
            )
    else:
        raise ValueError("score_direction must be a direction string or boolean")
    return ("higher_is_ood" if direction else "lower_is_ood"), direction


def _resolve_strategy(method: str) -> str:
    if not isinstance(method, str) or not method.strip():
        raise ValueError("method must be a non-empty strategy name")
    normalized = method.strip().lower().replace("-", "_").replace(" ", "_")
    try:
        return _STRATEGY_ALIASES[normalized]
    except KeyError as exc:
        choices = ", ".join(STRATEGIES)
        raise ValueError(f"unknown calibration strategy {method!r}; expected {choices}") from exc


def _validate_target(target_tpr: float) -> float:
    try:
        value = float(target_tpr)
    except (TypeError, ValueError) as exc:
        raise ValueError("target_tpr must be a finite value in (0, 1]") from exc
    if not np.isfinite(value) or not 0.0 < value <= 1.0:
        raise ValueError("target_tpr must be a finite value in (0, 1]")
    return value


def _validate_weight(value: float, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite non-negative number") from exc
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be a finite non-negative number")
    return result


def _oriented_scores(values: np.ndarray, higher_score_is_ood: bool) -> np.ndarray:
    """Orient a score array so larger values always imply OOD."""

    return values if higher_score_is_ood else -values


def _threshold_candidates(id_scores: np.ndarray, ood_scores: np.ndarray) -> np.ndarray:
    """Return every distinct binary operating point induced by the scores.

    Exact scores and midpoints between adjacent scores are included so a
    threshold can either reject samples tied at that score or sit cleanly
    between two score values.  The two outer points represent rejecting none
    and rejecting every sample.
    """

    unique = np.unique(np.concatenate((id_scores, ood_scores)))
    candidates = [*unique]
    if unique.size > 1:
        # Dividing before adding avoids an intermediate overflow for very
        # large, but still finite, score values.
        candidates.extend((unique[:-1] / 2.0 + unique[1:] / 2.0).tolist())
    candidates.append(float(np.nextafter(unique[-1], np.inf)))
    candidates.append(float(np.nextafter(unique[0], -np.inf)))
    result = np.asarray(candidates, dtype=np.float64)
    result = result[np.isfinite(result)]
    return np.unique(result)


def _metrics(
    id_scores: np.ndarray,
    ood_scores: np.ndarray,
    threshold: float,
    *,
    false_acceptance_weight: float,
    false_rejection_weight: float,
) -> dict[str, float | int]:
    """Calculate operating-point metrics on oriented scores."""

    id_rejected = id_scores >= threshold
    ood_rejected = ood_scores >= threshold

    true_positive = int(np.sum(ood_rejected))
    false_positive = int(np.sum(id_rejected))
    false_negative = int(np.sum(~ood_rejected))
    true_negative = int(np.sum(~id_rejected))

    ood_precision = (
        true_positive / (true_positive + false_positive)
        if true_positive + false_positive
        else 0.0
    )
    ood_recall = true_positive / ood_scores.size
    id_tpr = true_negative / id_scores.size
    false_acceptance_rate = false_negative / ood_scores.size
    false_rejection_rate = false_positive / id_scores.size
    ood_f1 = (
        2.0 * ood_precision * ood_recall / (ood_precision + ood_recall)
        if ood_precision + ood_recall
        else 0.0
    )
    total_weight = false_acceptance_weight + false_rejection_weight
    weighted_error = (
        false_acceptance_weight * false_acceptance_rate
        + false_rejection_weight * false_rejection_rate
    )
    weighted_error_rate = weighted_error / total_weight

    return {
        "id_tpr": float(id_tpr),
        "id_acceptance_rate": float(id_tpr),
        "ood_tpr": float(ood_recall),
        "ood_recall": float(ood_recall),
        "ood_precision": float(ood_precision),
        "ood_f1": float(ood_f1),
        "false_acceptance_rate": float(false_acceptance_rate),
        "false_rejection_rate": float(false_rejection_rate),
        "weighted_error": float(weighted_error),
        "weighted_error_rate": float(weighted_error_rate),
        "true_positive": true_positive,
        "true_negative": true_negative,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "id_count": int(id_scores.size),
        "ood_count": int(ood_scores.size),
    }


def _raw_threshold(oriented_threshold: float, higher_score_is_ood: bool) -> float:
    return float(oriented_threshold if higher_score_is_ood else -oriented_threshold)


def _select_strategy(
    method: str,
    candidates: np.ndarray,
    id_scores: np.ndarray,
    ood_scores: np.ndarray,
    *,
    higher_score_is_ood: bool,
    target_tpr: float,
    false_acceptance_weight: float,
    false_rejection_weight: float,
) -> dict[str, Any]:
    """Select one threshold using one of the three supported strategies."""

    records = [
        (
            float(candidate),
            _metrics(
                id_scores,
                ood_scores,
                float(candidate),
                false_acceptance_weight=false_acceptance_weight,
                false_rejection_weight=false_rejection_weight,
            ),
        )
        for candidate in candidates
    ]

    if method == MAX_F1:
        # Ties prefer more OOD recall, then fewer ID false rejections, then a
        # lower oriented threshold for reproducibility.
        oriented_threshold, metrics = max(
            records,
            key=lambda item: (
                float(item[1]["ood_f1"]),
                float(item[1]["ood_recall"]),
                -float(item[1]["false_rejection_rate"]),
                -item[0],
            ),
        )
    elif method == TARGET_ID_TPR:
        eligible = [
            item for item in records if float(item[1]["id_tpr"]) + 1e-12 >= target_tpr
        ]
        pool = eligible or records
        # The target is an ID acceptance constraint.  OOD recall is used as a
        # tie-break so the calibration data still informs the selected point.
        oriented_threshold, metrics = max(
            pool,
            key=lambda item: (
                float(item[1]["ood_recall"]),
                -abs(float(item[1]["id_tpr"]) - target_tpr),
                float(item[1]["ood_f1"]),
                -item[0],
            ),
        )
    elif method == MIN_WEIGHTED_ERROR:
        oriented_threshold, metrics = min(
            records,
            key=lambda item: (
                float(item[1]["weighted_error"]),
                -float(item[1]["ood_f1"]),
                -float(item[1]["ood_recall"]),
                item[0],
            ),
        )
    else:  # pragma: no cover - guarded by _resolve_strategy
        raise AssertionError(f"unsupported strategy: {method}")

    return {
        "method": method,
        "threshold": _raw_threshold(oriented_threshold, higher_score_is_ood),
        "target_tpr": target_tpr,
        "target_id_tpr": target_tpr,
        "metrics": metrics,
    }


def calibrate_strategy(
    id_scores: Iterable[float] | np.ndarray,
    ood_scores: Iterable[float] | np.ndarray,
    *,
    method: str = MAX_F1,
    strategy: str | None = None,
    score_direction: str | bool = "higher_is_ood",
    target_tpr: float = DEFAULT_TARGET_TPR,
    target_id_acceptance: float | None = None,
    false_acceptance_weight: float = DEFAULT_FALSE_ACCEPTANCE_WEIGHT,
    false_rejection_weight: float = DEFAULT_FALSE_REJECTION_WEIGHT,
    higher_score_is_ood: bool | None = None,
) -> dict[str, Any]:
    """Calibrate one threshold strategy.

    ``target_id_acceptance`` is a compatibility alias for ``target_tpr``.
    For the target strategy, ID TPR means the fraction of ID validation
    samples accepted as ID.  A threshold satisfying at least the requested
    target is selected because empirical scores may not permit the exact
    fraction.
    """

    ids = _finite_scores(id_scores, "id_scores")
    oods = _finite_scores(ood_scores, "ood_scores")
    if strategy is not None:
        if method != MAX_F1 and _resolve_strategy(method) != _resolve_strategy(strategy):
            raise ValueError("method and strategy specify different calibration strategies")
        method = strategy
    method_name = _resolve_strategy(method)
    target = _validate_target(
        target_tpr if target_id_acceptance is None else target_id_acceptance
    )
    fa_weight = _validate_weight(false_acceptance_weight, "false_acceptance_weight")
    fr_weight = _validate_weight(false_rejection_weight, "false_rejection_weight")
    if fa_weight + fr_weight <= 0.0:
        raise ValueError("false_acceptance_weight + false_rejection_weight must be positive")
    direction_name, higher = _resolve_direction(score_direction, higher_score_is_ood)
    oriented_ids = _oriented_scores(ids, higher)
    oriented_oods = _oriented_scores(oods, higher)
    selected = _select_strategy(
        method_name,
        _threshold_candidates(oriented_ids, oriented_oods),
        oriented_ids,
        oriented_oods,
        higher_score_is_ood=higher,
        target_tpr=target,
        false_acceptance_weight=fa_weight,
        false_rejection_weight=fr_weight,
    )
    selected["score_direction"] = direction_name
    selected["false_acceptance_weight"] = fa_weight
    selected["false_rejection_weight"] = fr_weight
    return selected


def calibrate_thresholds(
    id_scores: Iterable[float] | np.ndarray,
    ood_scores: Iterable[float] | np.ndarray,
    *,
    method: str = MAX_F1,
    strategy: str | None = None,
    score_direction: str | bool = "higher_is_ood",
    target_tpr: float = DEFAULT_TARGET_TPR,
    target_id_acceptance: float | None = None,
    false_acceptance_weight: float = DEFAULT_FALSE_ACCEPTANCE_WEIGHT,
    false_rejection_weight: float = DEFAULT_FALSE_REJECTION_WEIGHT,
    higher_score_is_ood: bool | None = None,
    source: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Calibrate all strategies and return a JSON-ready artifact.

    The top-level ``method``/``threshold``/``metrics`` fields describe the
    requested strategy.  The ``strategies`` field retains all three strategy
    results so choosing a deployment policy does not require recalibrating.
    """

    ids = _finite_scores(id_scores, "id_scores")
    oods = _finite_scores(ood_scores, "ood_scores")
    if strategy is not None:
        if method != MAX_F1 and _resolve_strategy(method) != _resolve_strategy(strategy):
            raise ValueError("method and strategy specify different calibration strategies")
        method = strategy
    requested_method = _resolve_strategy(method)
    target = _validate_target(
        target_tpr if target_id_acceptance is None else target_id_acceptance
    )
    fa_weight = _validate_weight(false_acceptance_weight, "false_acceptance_weight")
    fr_weight = _validate_weight(false_rejection_weight, "false_rejection_weight")
    if fa_weight + fr_weight <= 0.0:
        raise ValueError("false_acceptance_weight + false_rejection_weight must be positive")
    direction_name, higher = _resolve_direction(score_direction, higher_score_is_ood)
    oriented_ids = _oriented_scores(ids, higher)
    oriented_oods = _oriented_scores(oods, higher)
    candidates = _threshold_candidates(oriented_ids, oriented_oods)
    results = {
        name: _select_strategy(
            name,
            candidates,
            oriented_ids,
            oriented_oods,
            higher_score_is_ood=higher,
            target_tpr=target,
            false_acceptance_weight=fa_weight,
            false_rejection_weight=fr_weight,
        )
        for name in STRATEGIES
    }
    selected = results[requested_method]
    artifact: dict[str, Any] = {
        "schema_version": 1,
        "status": "calibrated",
        "method": requested_method,
        "strategy": requested_method,
        "threshold": selected["threshold"],
        # Keep the legacy key used by deployment consumers as an alias.
        "ood_threshold": selected["threshold"],
        "score_direction": direction_name,
        "target_tpr": target,
        "target_id_tpr": target,
        "false_acceptance_weight": fa_weight,
        "false_rejection_weight": fr_weight,
        "metrics": selected["metrics"],
        "strategies": results,
        "data_sources": {
            "id": "ID validation",
            "ood": "OOD calibration",
            "final_test_used": False,
        },
        "calibration_date": date.today().isoformat(),
    }
    if source is not None:
        artifact["source"] = dict(source)
    return artifact


def calibrate_threshold(
    id_scores: Iterable[float] | np.ndarray,
    ood_scores: Iterable[float] | np.ndarray,
    *,
    target_id_acceptance: float = DEFAULT_TARGET_TPR,
    score_direction: str | bool = "higher_is_ood",
    method: str = TARGET_ID_TPR,
    strategy: str | None = None,
    false_acceptance_weight: float = DEFAULT_FALSE_ACCEPTANCE_WEIGHT,
    false_rejection_weight: float = DEFAULT_FALSE_REJECTION_WEIGHT,
    higher_score_is_ood: bool | None = None,
) -> dict[str, Any]:
    """Backward-compatible single-strategy threshold API.

    Existing detector modules use the ``ood_threshold``/``id_acceptance`` /
    ``ood_recall`` keys.  New callers that need all strategies should use
    :func:`calibrate_thresholds`.
    """

    result = calibrate_strategy(
        id_scores,
        ood_scores,
        method=method if strategy is None else strategy,
        score_direction=score_direction,
        target_tpr=target_id_acceptance,
        false_acceptance_weight=false_acceptance_weight,
        false_rejection_weight=false_rejection_weight,
        higher_score_is_ood=higher_score_is_ood,
    )
    metrics = result["metrics"]
    return {
        **result,
        "ood_threshold": result["threshold"],
        "target_id_acceptance": float(target_id_acceptance),
        "id_acceptance": metrics["id_tpr"],
        "ood_recall": metrics["ood_recall"],
        "calibration_date": date.today().isoformat(),
    }


def _json_vector(value: Any, name: str) -> np.ndarray:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{name} must be a JSON array")
    return _finite_scores(value, name)


def _load_vector(path: Path, name: str) -> np.ndarray:
    """Load one score vector from ``.npy``, ``.json`` or one-column text."""

    if not path.is_file():
        raise FileNotFoundError(f"score file not found: {path}")
    suffix = path.suffix.lower()
    if suffix == ".npy":
        return _finite_scores(np.load(path, allow_pickle=False), name)
    if suffix == ".json":
        with path.open(encoding="utf-8") as handle:
            return _json_vector(json.load(handle), name)
    try:
        return _finite_scores(np.loadtxt(path, delimiter=",", ndmin=1), name)
    except (OSError, ValueError) as exc:
        raise ValueError(f"could not read {name} scores from {path}") from exc


def _load_score_artifact(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load only explicitly named ID-validation and OOD-calibration arrays."""

    if not path.is_file():
        raise FileNotFoundError(f"score artifact not found: {path}")
    suffix = path.suffix.lower()
    aliases = {
        "id_scores": "id",
        "id_validation_scores": "id",
        "id_validation": "id",
        "ood_scores": "ood",
        "ood_calibration_scores": "ood",
        "ood_calibration": "ood",
    }
    values: dict[str, Any] = {}
    if suffix == ".npz":
        with np.load(path, allow_pickle=False) as data:
            values = {key: data[key] for key in data.files}
    elif suffix == ".json":
        with path.open(encoding="utf-8") as handle:
            loaded = json.load(handle)
        if not isinstance(loaded, dict):
            raise ValueError("score JSON must be an object with ID/OOD calibration arrays")
        values = loaded
    else:
        raise ValueError("combined score artifact must be .npz or .json")

    selected: dict[str, Any] = {}
    for key, value in values.items():
        normalized_key = str(key).strip().lower()
        if "test" in normalized_key or "final" in normalized_key:
            raise ValueError(
                "final test scores are not accepted; provide ID validation and "
                "OOD calibration scores only"
            )
        role = aliases.get(normalized_key)
        if role is not None:
            if role in selected:
                raise ValueError(f"score artifact contains multiple arrays for {role}")
            selected[role] = value
    if set(selected) != {"id", "ood"}:
        raise ValueError(
            "score artifact must contain exactly one ID validation array and one "
            "OOD calibration array (accepted keys: id_scores/ood_scores)"
        )
    return _finite_scores(selected["id"], "id_scores"), _finite_scores(
        selected["ood"], "ood_scores"
    )


def save_thresholds(path: str | Path, thresholds: Mapping[str, Any]) -> None:
    """Write a JSON threshold artifact, rejecting non-JSON numeric values."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(dict(thresholds), handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")


def load_thresholds(path: str | Path) -> dict[str, Any]:
    """Load and minimally validate a JSON threshold artifact."""

    source = Path(path)
    with source.open(encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError("threshold artifact must be a JSON object")
    return data


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument(
        "--scores",
        type=Path,
        default=DEFAULT_SCORE_ARTIFACT,
        help=".npz/.json with id_scores and ood_scores",
    )
    input_group.add_argument(
        "--id-validation",
        type=Path,
        help=".npy/.json/.csv score vector from ID validation",
    )
    parser.add_argument(
        "--ood-calibration",
        type=Path,
        help=".npy/.json/.csv score vector from OOD calibration",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument(
        "--method",
        default=MAX_F1,
        choices=STRATEGIES,
        help="strategy represented by the top-level threshold (all strategies are saved)",
    )
    parser.add_argument(
        "--score-direction",
        default="higher_is_ood",
        choices=("higher_is_ood", "lower_is_ood"),
    )
    parser.add_argument("--target-tpr", type=float, default=DEFAULT_TARGET_TPR)
    parser.add_argument(
        "--false-acceptance-weight",
        type=float,
        default=DEFAULT_FALSE_ACCEPTANCE_WEIGHT,
    )
    parser.add_argument(
        "--false-rejection-weight",
        type=float,
        default=DEFAULT_FALSE_REJECTION_WEIGHT,
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.id_validation is None and args.ood_calibration is None:
        id_scores, ood_scores = _load_score_artifact(args.scores)
        source = {
            "artifact": str(args.scores),
            "id_split": "validation",
            "ood_split": "calibration",
        }
    elif args.id_validation is not None and args.ood_calibration is not None:
        id_scores = _load_vector(args.id_validation, "id_scores")
        ood_scores = _load_vector(args.ood_calibration, "ood_scores")
        source = {
            "id_scores": str(args.id_validation),
            "ood_scores": str(args.ood_calibration),
            "id_split": "validation",
            "ood_split": "calibration",
        }
    else:
        raise ValueError("provide both --id-validation and --ood-calibration, or use --scores")

    artifact = calibrate_thresholds(
        id_scores,
        ood_scores,
        method=args.method,
        score_direction=args.score_direction,
        target_tpr=args.target_tpr,
        false_acceptance_weight=args.false_acceptance_weight,
        false_rejection_weight=args.false_rejection_weight,
        source=source,
    )
    save_thresholds(args.output, artifact)
    print(json.dumps(artifact, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()


__all__ = [
    "MAX_F1",
    "TARGET_ID_TPR",
    "MIN_WEIGHTED_ERROR",
    "STRATEGIES",
    "calibrate_strategy",
    "calibrate_thresholds",
    "calibrate_threshold",
    "save_thresholds",
    "load_thresholds",
]
