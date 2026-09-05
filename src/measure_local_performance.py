"""Measure local import and synthetic inference only; never open a dataset."""

import argparse
import ctypes
from ctypes import wintypes
import importlib
import json
import os
from pathlib import Path
import time


def working_set_mib():
    """Return Windows process working set without an extra dependency."""
    if os.name != "nt":
        return None

    class Counters(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
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
    get_memory.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
    get_memory.restype = wintypes.BOOL
    counters = Counters()
    counters.cb = ctypes.sizeof(counters)
    if not get_memory(get_process(), ctypes.byref(counters), counters.cb):
        raise ctypes.WinError()
    return round(counters.WorkingSetSize / (1024 ** 2), 2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    start = time.perf_counter()
    backend = importlib.import_module("app")
    result = {"import_seconds": round(time.perf_counter() - start, 4),
              "memory_after_import_mib": working_set_mib()}
    pixels = backend.np.random.default_rng(42).integers(
        0, 256, (1, 224, 224, 3), dtype=backend.np.uint8
    ).astype(backend.np.float32)
    timings = []
    for _ in range(3):
        start = time.perf_counter()
        backend.predict_image(pixels)
        timings.append(round(time.perf_counter() - start, 4))
    result.update(inference_seconds=timings,
                  memory_after_inference_mib=working_set_mib(),
                  synthetic_only=True,
                  note="Single local run; import excludes interpreter launch; not an accuracy benchmark.")
    payload = json.dumps(result, indent=2)
    print(payload)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
