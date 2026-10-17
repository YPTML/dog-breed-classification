"""Build a deterministic, leakage-checked OOD image dataset.

The builder consumes a local, category-organised image pool. It never
downloads images and it never touches the 120-class training files. Regular
OOD images are copied to ``data/ood/{calibration,test}/{category}``; hard OOD
images are copied to ``data/ood_hard/{category}``.

Expected source layouts are either ``ood_source/animals/cat/image.jpg`` or
the equivalent layout with category folders directly below the source root.
Hard OOD can be supplied through ``--hard-source`` or placed in the same tree
using the eight hard category names listed in :data:`HARD_OOD_CATEGORIES`.
"""

from __future__ import annotations

import argparse
import csv
import random
import re
import shutil
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from .build_manifest import FIELDS, image_dimensions, read_csv, validate_manifest
from .validate_dataset import IMAGE_EXTENSIONS, sha256_file


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SEED = 42
DEFAULT_CALIBRATION_FRACTION = 0.5
OOD_SOURCE = "local_ood"

REGULAR_OOD_GROUPS = {
    "animals": ("cat", "horse", "cow", "sheep", "rabbit", "bird", "elephant", "tiger", "bear"),
    "vehicles": ("car", "motorcycle", "bus", "bicycle"),
    "objects": ("chair", "table", "bottle", "phone", "laptop"),
}
REGULAR_OOD_CATEGORIES = frozenset(
    (*REGULAR_OOD_GROUPS["animals"], "humans", *REGULAR_OOD_GROUPS["vehicles"],
     *REGULAR_OOD_GROUPS["objects"], "food", "landscape")
)
HARD_OOD_CATEGORIES = frozenset(
    ("wolf", "fox", "coyote", "hyena", "dog_toy", "dog_statue", "dog_cartoon", "dog_drawing")
)
ALL_OOD_CATEGORIES = REGULAR_OOD_CATEGORIES | HARD_OOD_CATEGORIES
_SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")


@dataclass(frozen=True)
class Candidate:
    """A source image after category inference and content hashing."""

    path: Path
    category: str
    digest: str
    hard: bool


def _image_files(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    return sorted(
        (path for path in root.rglob("*")
         if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS),
        key=lambda path: path.relative_to(root).as_posix().lower(),
    )


def _infer_category(path: Path, source: Path, allowed: frozenset[str], *, hard: bool) -> str:
    """Infer the deepest supported category from source directory components."""
    parts = path.relative_to(source).parts[:-1]
    matches = [part for part in parts if part in allowed]
    if not matches:
        kind = "hard OOD" if hard else "OOD"
        expected = ", ".join(sorted(allowed))
        raise ValueError(f"Cannot infer {kind} category for {path}; expected one of: {expected}")
    return matches[-1]


def _collect_candidates(source: Path, *, hard: bool) -> list[Candidate]:
    if not source.exists():
        raise FileNotFoundError(f"OOD source does not exist: {source}")
    if not source.is_dir():
        raise NotADirectoryError(f"OOD source is not a directory: {source}")
    allowed = HARD_OOD_CATEGORIES if hard else REGULAR_OOD_CATEGORIES
    return [Candidate(path, _infer_category(path, source, allowed, hard=hard), sha256_file(path), hard)
            for path in _image_files(source)]


def _collect_mixed_candidates(source: Path) -> list[Candidate]:
    """Accept one source tree containing both regular and hard categories."""
    if not source.exists():
        raise FileNotFoundError(f"OOD source does not exist: {source}")
    if not source.is_dir():
        raise NotADirectoryError(f"OOD source is not a directory: {source}")
    allowed = REGULAR_OOD_CATEGORIES | HARD_OOD_CATEGORIES
    candidates = []
    for path in _image_files(source):
        parts = path.relative_to(source).parts[:-1]
        regular = [part for part in parts if part in REGULAR_OOD_CATEGORIES]
        hard = [part for part in parts if part in HARD_OOD_CATEGORIES]
        if regular and hard:
            raise ValueError(f"Ambiguous regular/hard OOD category for {path}")
        if not regular and not hard:
            expected = ", ".join(sorted(allowed))
            raise ValueError(f"Cannot infer OOD category for {path}; expected one of: {expected}")
        category, is_hard = (hard[-1], True) if hard else (regular[-1], False)
        candidates.append(Candidate(path, category, sha256_file(path), is_hard))
    return candidates


def _deduplicate(candidates: list[Candidate]) -> list[Candidate]:
    """Drop exact duplicates and reject one image assigned to two categories."""
    by_digest: dict[str, Candidate] = {}
    for candidate in sorted(candidates, key=lambda item: (item.digest, item.category,
                                                           item.hard, item.path.as_posix().lower())):
        previous = by_digest.get(candidate.digest)
        if previous is None:
            by_digest[candidate.digest] = candidate
            continue
        if (previous.category, previous.hard) != (candidate.category, candidate.hard):
            raise ValueError(
                "The same image bytes occur in multiple OOD categories: "
                f"{previous.path} ({previous.category}) and {candidate.path} ({candidate.category})"
            )
    return sorted(by_digest.values(), key=lambda item: (item.hard, item.category, item.digest))


def _resolve_from_root(root: Path, value: str | Path | None, default: str) -> Path:
    path = Path(default if value is None else value)
    return path if path.is_absolute() else root / path


def _training_hashes(root: Path) -> set[str]:
    """Collect exact-byte identities from every known 120-class train pool."""
    hashes = set()
    for directory in (root / "data/train", root / "data/in_distribution/train"):
        hashes.update(sha256_file(path) for path in _image_files(directory))
    manifest = root / "metadata/dataset_manifest.csv"
    if manifest.is_file():
        try:
            for row in read_csv(manifest, {"path", "distribution", "split", "sha256"}):
                if row["distribution"] == "id" and row["split"] == "train" and row["sha256"]:
                    hashes.add(row["sha256"])
        except (OSError, UnicodeError, csv.Error, ValueError):
            # The physical data scan remains authoritative for a malformed
            # or stale manifest.
            pass
    return hashes


def _safe_target_name(path: Path, used: set[str], digest: str) -> str:
    stem = _SAFE_NAME.sub("_", path.stem).strip("._") or "image"
    suffix = path.suffix.lower()
    name = f"{stem}{suffix}"
    if name.casefold() in used:
        name = f"{stem}__{digest[:12]}{suffix}"
        counter = 2
        while name.casefold() in used:
            name = f"{stem}__{digest[:12]}_{counter}{suffix}"
            counter += 1
    used.add(name.casefold())
    return name


def _split_regular(candidates: list[Candidate], fraction: float, seed: int) -> dict[str, list[Candidate]]:
    regular = [candidate for candidate in candidates if not candidate.hard]
    rng = random.Random(seed)
    ordered = sorted(regular, key=lambda item: (item.category, item.digest, item.path.as_posix().lower()))
    rng.shuffle(ordered)
    calibration_count = int(len(ordered) * fraction)
    if len(ordered) > 1:
        calibration_count = min(max(calibration_count, 1), len(ordered) - 1)
    return {"calibration": ordered[:calibration_count], "test": ordered[calibration_count:]}


def _copy_candidates(
    split_candidates: dict[str, list[Candidate]],
    destination: Path,
    hard_candidates: list[Candidate],
    hard_destination: Path,
) -> list[dict]:
    """Copy candidates and return the output inventory used by the manifest."""
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "calibration").mkdir(parents=True, exist_ok=True)
    (destination / "test").mkdir(parents=True, exist_ok=True)
    hard_destination.mkdir(parents=True, exist_ok=True)
    output: list[dict] = []
    for split in ("calibration", "test"):
        used_by_category: dict[str, set[str]] = {}
        for candidate in sorted(split_candidates[split], key=lambda item: (item.category, item.digest)):
            filename = _safe_target_name(candidate.path, used_by_category.setdefault(candidate.category, set()), candidate.digest)
            target = destination / split / candidate.category / filename
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(candidate.path, target)
            output.append({"path": target, "category": candidate.category, "split": split,
                           "digest": candidate.digest, "hard": False})
    used_by_category = {}
    for candidate in sorted(hard_candidates, key=lambda item: (item.category, item.digest)):
        filename = _safe_target_name(candidate.path, used_by_category.setdefault(candidate.category, set()), candidate.digest)
        target = hard_destination / candidate.category / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(candidate.path, target)
        output.append({"path": target, "category": candidate.category, "split": "hard",
                       "digest": candidate.digest, "hard": True})
    return output


def _relative_path(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _ood_rows(root: Path, destination: Path, hard_destination: Path) -> list[dict]:
    """Create manifest rows from materialized OOD output folders."""
    rows = []
    seen_digests: dict[str, str] = {}
    for split_root, split, allowed in (
        (destination / "calibration", "calibration", REGULAR_OOD_CATEGORIES),
        (destination / "test", "test", REGULAR_OOD_CATEGORIES),
        (hard_destination, "hard", HARD_OOD_CATEGORIES),
    ):
        for path in _image_files(split_root):
            relative = path.relative_to(split_root)
            if len(relative.parts) < 2 or relative.parts[0] not in allowed:
                raise ValueError(f"Invalid OOD category layout: {_relative_path(root, path)}")
            category = relative.parts[0]
            digest = sha256_file(path)
            previous = seen_digests.get(digest)
            if previous is not None:
                raise ValueError(f"OOD image appears more than once: {previous} and {_relative_path(root, path)}")
            seen_digests[digest] = _relative_path(root, path)
            width, height = image_dimensions(path.read_bytes(), path.suffix.lower())
            rows.append(dict(zip(FIELDS, [
                f"ood-{digest[:32]}", _relative_path(root, path), "", "", "ood", category,
                split, OOD_SOURCE, str(width), str(height), digest,
            ])))
    return sorted(rows, key=lambda row: row["path"])


def _merge_manifest(
    root: Path, ood_rows: list[dict], destination: Path,
    *, ood_root: Path, ood_hard_root: Path,
) -> None:
    """Replace OOD rows while preserving the project's ID metadata."""
    existing: list[dict] = []
    if destination.is_file():
        existing = read_csv(destination, set(FIELDS))
        existing = [row for row in existing if row["distribution"] != "ood"]
    else:
        labels = root / "data/labels.csv"
        classes = root / "outputs/deployment_ready/class_names.json"
        if labels.is_file() and classes.is_file():
            from .build_manifest import build_manifest
            existing = [row for row in build_manifest(root) if row["distribution"] != "ood"]
    rows = existing + ood_rows
    classes_path = root / "outputs/deployment_ready/class_names.json"
    if classes_path.is_file():
        import json
        classes_data = json.loads(classes_path.read_text(encoding="utf-8"))
        classes = set(classes_data.values()) if isinstance(classes_data, dict) else set(classes_data)
        validate_manifest(rows, root, classes, ood_root=ood_root, ood_hard_root=ood_hard_root)
    else:
        _validate_ood_only_rows(rows, root)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", newline="", encoding="utf-8",
                                         dir=destination.parent, delete=False) as handle:
            temporary = Path(handle.name)
            writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
            writer.writeheader()
            writer.writerows(sorted(rows, key=lambda row: row["path"]))
        temporary.replace(destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _validate_ood_only_rows(rows: list[dict], root: Path) -> None:
    seen_paths, seen_ids, seen_digests = set(), set(), set()
    for row in rows:
        if row["distribution"] != "ood":
            raise ValueError("An ID row requires class metadata in class_names.json")
        if row["label"] or row["breed"] or row["ood_category"] not in ALL_OOD_CATEGORIES:
            raise ValueError(f"Invalid OOD metadata: {row}")
        path = (root / row["path"]).resolve()
        if not path.is_file() or not path.is_relative_to(root.resolve()):
            raise ValueError(f"Invalid OOD path: {row['path']}")
        if row["path"] in seen_paths or row["image_id"] in seen_ids or row["sha256"] in seen_digests:
            raise ValueError("Duplicate OOD path, image_id, or sha256")
        seen_paths.add(row["path"])
        seen_ids.add(row["image_id"])
        seen_digests.add(row["sha256"])


def _display_path(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _write_report(root: Path, path: Path, *, seed: int, fraction: float,
                  source: Path, hard_source: Path, records: list[dict],
                  candidate_count: int, duplicate_count: int) -> None:
    counts = Counter((record["split"], record["ood_category"]) for record in records)
    split_counts = Counter(record["split"] for record in records)
    lines = [
        "# OOD DATASET REPORT", "",
        "Deterministic OOD dataset build; no model training was performed.", "",
        "## Configuration", "",
        f"- Seed: `{seed}`", f"- Calibration fraction: `{fraction}`",
        f"- Regular source: `{_display_path(root, source)}`",
        f"- Hard source: `{_display_path(root, hard_source)}`", "",
        "## Output", "", "| Split | Images |", "|---|---:|",
        f"| calibration | {split_counts['calibration']} |",
        f"| test | {split_counts['test']} |", f"| hard | {split_counts['hard']} |",
        f"| total | {len(records)} |", "", "### Category counts", "",
        "| Split | Category | Images |", "|---|---|---:|",
    ]
    for (split, category), count in sorted(counts.items()):
        lines.append(f"| {split} | {category} | {count} |")
    lines += [
        "", "## Integrity checks", "",
        f"- Source image candidates: **{candidate_count}**",
        f"- Exact duplicates removed before split: **{duplicate_count}**",
        "- Calibration/test exact-byte overlap: **0**",
        "- OOD images copied into `data/train`: **0**",
        "- Manifest distribution: `ood` for every OOD row",
        "- Hard OOD is separate from calibration/test and is never an ID class.",
        "", "## Reproduction", "", "```powershell",
        f"python -m src.data.build_ood_dataset --root . --seed {seed} "
        f"--calibration-fraction {fraction}", "```", "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def build_ood_dataset(
    root: str | Path = PROJECT_ROOT,
    source: str | Path | None = None,
    *,
    hard_source: str | Path | None = None,
    destination: str | Path | None = None,
    hard_destination: str | Path | None = None,
    seed: int = DEFAULT_SEED,
    calibration_fraction: float = DEFAULT_CALIBRATION_FRACTION,
    manifest_path: str | Path | None = None,
    report_path: str | Path | None = None,
) -> dict:
    """Build OOD folders, manifest rows, and report under ``root``."""
    root = Path(root).resolve()
    if not 0 < calibration_fraction < 1:
        raise ValueError("calibration_fraction must be between 0 and 1")
    regular_source = _resolve_from_root(root, source, "data/ood_source")
    hard_input = _resolve_from_root(root, hard_source, "data/ood_hard_source")
    if source is not None and not regular_source.exists():
        raise FileNotFoundError(f"OOD source does not exist: {regular_source}")
    if hard_source is not None and not hard_input.exists():
        raise FileNotFoundError(f"Hard OOD source does not exist: {hard_input}")

    if hard_source is None and regular_source.exists():
        mixed = _collect_mixed_candidates(regular_source)
        regular = [candidate for candidate in mixed if not candidate.hard]
        hard = [candidate for candidate in mixed if candidate.hard]
        if hard_input.exists():
            hard.extend(_collect_candidates(hard_input, hard=True))
    else:
        regular = _collect_candidates(regular_source, hard=False) if regular_source.exists() else []
        hard = _collect_candidates(hard_input, hard=True) if hard_input.exists() else []
    candidates = _deduplicate(regular + hard)
    training = _training_hashes(root)
    leaked = [candidate for candidate in candidates if candidate.digest in training]
    if leaked:
        details = ", ".join(str(candidate.path) for candidate in leaked[:3])
        raise ValueError(f"OOD image is byte-identical to a 120-class training image: {details}")

    output_root = _resolve_from_root(root, destination, "data/ood")
    hard_output = _resolve_from_root(root, hard_destination, "data/ood_hard")
    for output in (output_root, hard_output):
        if not output.resolve().is_relative_to(root):
            raise ValueError(f"OOD destination must be inside the project root: {output}")
    split_candidates = _split_regular(candidates, calibration_fraction, seed)
    hard_candidates = [candidate for candidate in candidates if candidate.hard]
    _copy_candidates(split_candidates, output_root, hard_candidates, hard_output)
    ood_rows = _ood_rows(root, output_root, hard_output)
    manifest = _resolve_from_root(root, manifest_path, "metadata/dataset_manifest.csv")
    _merge_manifest(root, ood_rows, manifest, ood_root=output_root, ood_hard_root=hard_output)
    report = _resolve_from_root(root, report_path, "outputs/reports/OOD_DATASET_REPORT.md")
    _write_report(root, report, seed=seed, fraction=calibration_fraction,
                  source=regular_source, hard_source=hard_input, records=ood_rows,
                  candidate_count=len(regular) + len(hard),
                  duplicate_count=len(regular) + len(hard) - len(candidates))
    return {
        "seed": seed, "calibration_fraction": calibration_fraction,
        "candidate_count": len(regular) + len(hard),
        "duplicate_count": len(regular) + len(hard) - len(candidates),
        "calibration_count": sum(row["split"] == "calibration" for row in ood_rows),
        "test_count": sum(row["split"] == "test" for row in ood_rows),
        "hard_count": sum(row["split"] == "hard" for row in ood_rows),
        "manifest": manifest, "report": report,
    }


def split_ood_pool(source: str | Path, destination: str | Path,
                   calibration_fraction: float = DEFAULT_CALIBRATION_FRACTION,
                   seed: int = DEFAULT_SEED) -> tuple[int, int]:
    """Backward-compatible regular-pool splitter."""
    candidates = _deduplicate(_collect_candidates(Path(source), hard=False))
    splits = _split_regular(candidates, calibration_fraction, seed)
    _copy_candidates(splits, Path(destination), [], Path(destination).parent / "ood_hard")
    return len(splits["calibration"]), len(splits["test"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", nargs="?", type=Path,
                        help="regular OOD source; categories are subdirectories")
    parser.add_argument("--source", dest="source_option", type=Path,
                        help="regular OOD source (alternative to positional argument)")
    parser.add_argument("--hard-source", type=Path,
                        help="hard OOD source with wolf/fox/... subdirectories")
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--destination", type=Path, default=None)
    parser.add_argument("--hard-destination", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--calibration-fraction", type=float, default=DEFAULT_CALIBRATION_FRACTION)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args()
    if args.source is not None and args.source_option is not None:
        parser.error("provide the regular OOD source positionally or with --source, not both")
    try:
        result = build_ood_dataset(
            args.root,
            args.source_option if args.source_option is not None else args.source,
            hard_source=args.hard_source, destination=args.destination,
            hard_destination=args.hard_destination, seed=args.seed,
            calibration_fraction=args.calibration_fraction,
            manifest_path=args.manifest, report_path=args.report,
        )
    except (OSError, ValueError) as exc:
        parser.exit(1, f"OOD dataset build failed: {exc}\n")
    print(" ".join(f"{key}={value}" for key, value in result.items()
                   if key.endswith("count") or key == "duplicate_count"))


if __name__ == "__main__":
    main()
