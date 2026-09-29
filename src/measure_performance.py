"""Measure production classifier and OOD inference performance.

The benchmark intentionally uses batch size one and preprocessed validation
images so that the reported latency represents one inference request and not
an offline throughput batch.  Image decoding/resizing is outside the timed
region.  ``baseline`` calls the Keras classifier directly; ``production``
calls the long-lived :class:`src.inference.predictor.Predictor` with the
shipped MSP policy and decision layer; ``embedding`` measures the optional
penultimate-feature pass separately.
"""

from __future__ import annotations

import argparse
import ctypes
import csv
import json
import os
from ctypes import wintypes
from pathlib import Path
import subprocess
import time
from typing import Any, Callable

import numpy as np

from src.inference.preprocessing import preprocess_image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "outputs" / "deployment_ready" / "dog_breed_model.keras"
CLASS_NAMES_PATH = PROJECT_ROOT / "outputs" / "deployment_ready" / "class_names.json"
THRESHOLDS_PATH = PROJECT_ROOT / "outputs" / "deployment_ready" / "ood_config.json"
VAL_SPLIT_PATH = PROJECT_ROOT / "outputs" / "splits" / "val.csv"
IMAGE_ROOT = PROJECT_ROOT / "data" / "train"


def _memory_snapshot() -> dict[str, float | None]:
    """Return current and OS-reported peak working set in MiB on Windows."""

    if os.name != "nt":
        return {"working_set_mib": None, "peak_working_set_mib": None}

    class Counters(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    get_process = ctypes.windll.kernel32.GetCurrentProcess
    get_process.restype = wintypes.HANDLE
    get_memory = ctypes.windll.psapi.GetProcessMemoryInfo
    get_memory.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(Counters),
        wintypes.DWORD,
    ]
    get_memory.restype = wintypes.BOOL
    counters = Counters()
    counters.cb = ctypes.sizeof(counters)
    if not get_memory(get_process(), ctypes.byref(counters), counters.cb):
        raise ctypes.WinError()
    mib = 1024.0**2
    return {
        "working_set_mib": round(float(counters.WorkingSetSize) / mib, 2),
        "peak_working_set_mib": round(float(counters.PeakWorkingSetSize) / mib, 2),
    }


def _tensorflow() -> Any:
    import tensorflow as tf

    return tf


def _load_model(tf: Any) -> Any:
    preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
    kwargs = {
        "custom_objects": {"preprocess_input": preprocess_input},
        "compile": False,
    }
    try:
        return tf.keras.models.load_model(str(MODEL_PATH), safe_mode=True, **kwargs)
    except TypeError as exc:
        if "safe_mode" not in str(exc):
            raise
        return tf.keras.models.load_model(str(MODEL_PATH), **kwargs)


def _load_batches(sample_count: int) -> list[np.ndarray]:
    if not VAL_SPLIT_PATH.exists():
        raise FileNotFoundError(f"validation split not found: {VAL_SPLIT_PATH}")
    rows: list[dict[str, str]] = []
    with VAL_SPLIT_PATH.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if sample_count <= 0:
        raise ValueError("sample_count must be positive")
    batches: list[np.ndarray] = []
    for row in rows[:sample_count]:
        image_name = row.get("img_name", "")
        path = IMAGE_ROOT / image_name
        if not path.exists():
            raise FileNotFoundError(f"validation image not found: {path}")
        batches.append(preprocess_image(path.read_bytes()))
    if not batches:
        raise ValueError("validation split contains no images")
    return batches


def _summary(values_ms: list[float]) -> dict[str, float]:
    values = np.asarray(values_ms, dtype=np.float64)
    return {
        "count": int(values.size),
        "mean_ms": round(float(np.mean(values)), 4),
        "median_ms": round(float(np.median(values)), 4),
        "p95_ms": round(float(np.percentile(values, 95)), 4),
        "min_ms": round(float(np.min(values)), 4),
        "max_ms": round(float(np.max(values)), 4),
    }


def _time_call(call: Callable[[np.ndarray], Any], batches: list[np.ndarray]) -> dict[str, Any]:
    first_started = time.perf_counter()
    first_result = call(batches[0])
    first_ms = (time.perf_counter() - first_started) * 1000.0

    for index in range(4):
        call(batches[index % len(batches)])

    timings: list[float] = []
    for index in range(len(batches) * 4):
        started = time.perf_counter()
        call(batches[index % len(batches)])
        timings.append((time.perf_counter() - started) * 1000.0)
    return {
        "first_inference_ms": round(first_ms, 4),
        "warm_inference": _summary(timings),
        "result_type": type(first_result).__name__,
    }


def _gpu_info(tf: Any) -> dict[str, Any]:
    devices = [device.name for device in tf.config.list_physical_devices("GPU")]
    result: dict[str, Any] = {
        "tensorflow_gpu_devices": devices,
        "tensorflow_gpu_memory_info": None,
    }
    if devices:
        try:
            result["tensorflow_gpu_memory_info"] = tf.config.experimental.get_memory_info(
                "GPU:0"
            )
        except (RuntimeError, ValueError):
            result["tensorflow_gpu_memory_info"] = "unavailable"
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.used",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
        result["nvidia_smi"] = completed.stdout.strip() or None
    except (FileNotFoundError, subprocess.SubprocessError, OSError):
        result["nvidia_smi"] = None
    return result


def run(mode: str, sample_count: int) -> dict[str, Any]:
    tf = _tensorflow()
    initial_memory = _memory_snapshot()
    load_started = time.perf_counter()
    model = _load_model(tf)
    model_load_ms = (time.perf_counter() - load_started) * 1000.0
    after_model_load = _memory_snapshot()

    batches = _load_batches(sample_count)
    class_names = json.loads(CLASS_NAMES_PATH.read_text(encoding="utf-8"))
    config = json.loads(THRESHOLDS_PATH.read_text(encoding="utf-8"))
    common: dict[str, Any] = {
        "mode": mode,
        "model_path": str(MODEL_PATH),
        "model_load_ms": round(model_load_ms, 4),
        "model_memory_after_load": after_model_load,
        "initial_memory": initial_memory,
        "sample_count": len(batches),
        "batch_shape": list(batches[0].shape),
        "batch_size": 1,
        "latency_excludes_decode_resize": True,
        "tensorflow_version": str(tf.__version__),
        "gpu": _gpu_info(tf),
    }

    if mode == "baseline":
        timing = _time_call(lambda batch: model(batch, training=False), batches)
        common["inference"] = timing
    elif mode == "production":
        from src.inference.predictor import Predictor

        predictor_started = time.perf_counter()
        predictor = Predictor(
            model=model,
            class_names=class_names,
            config={
                "method": str(config.get("method", "msp")),
                "temperature": float(config.get("temperature", 1.0)),
                "output_mode": "probabilities",
                "strict_batch_output": True,
                "auto_build_logits": False,
                "auto_build_embedding": False,
            },
            thresholds=config,
        )
        predictor_setup_ms = (time.perf_counter() - predictor_started) * 1000.0
        common["production_ood"] = {
            "method": predictor.method,
            "predictor_setup_ms": round(predictor_setup_ms, 4),
            "embedding_adapter_enabled": False,
            "logits_adapter_enabled": False,
        }
        common["inference"] = _time_call(predictor.predict_batch, batches)
    elif mode == "embedding":
        from src.models.feature_extractor import build_feature_extractor

        extractor_started = time.perf_counter()
        extractor = build_feature_extractor(model)
        extractor_setup_ms = (time.perf_counter() - extractor_started) * 1000.0
        embedding_timing = _time_call(
            lambda batch: extractor(batch, training=False), batches
        )

        def classifier_then_embedding(batch: np.ndarray) -> Any:
            output = model(batch, training=False)
            embedding = extractor(batch, training=False)
            return output, embedding

        combined_timing = _time_call(classifier_then_embedding, batches)
        common["embedding"] = {
            "embedding_adapter_setup_ms": round(extractor_setup_ms, 4),
            "embedding_dimension": int(getattr(extractor, "embedding_dim", 0)),
            "inference": embedding_timing,
            "classifier_plus_embedding_inference": combined_timing,
        }
    else:
        raise ValueError(f"unsupported mode: {mode}")

    common["memory_after_benchmark"] = _memory_snapshot()
    common["peak_memory_during_process"] = common["memory_after_benchmark"][
        "peak_working_set_mib"
    ]
    return common


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("baseline", "production", "embedding"), required=True)
    parser.add_argument("--sample-count", type=int, default=16)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run(args.mode, args.sample_count)
    payload = json.dumps(result, indent=2, sort_keys=True)
    print(payload)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
