"""Utilisation sampling, an independent energy reading and the machine description.

Nothing here records a hostname, user name or file path, because the bundles support a
paper under anonymous review.
"""

from __future__ import annotations

import ctypes
import platform
import sys
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from functools import lru_cache
from typing import Any

import pandas as pd
import psutil
import torch

_PACKAGE_COUNTER = r"\Energy Meter(RAPL_Package0_PKG)\Energy"
_PICOWATT_HOURS_PER_KWH = 1e15
_PDH_FMT_LARGE = 0x00000400
_PDH_FMT_NOCAP100 = 0x00008000
_WINDOWS_11_BUILD = 22000
_ES_CONTINUOUS = 0x80000000
_ES_SYSTEM_REQUIRED = 0x00000001


def _gpu_handle() -> Any | None:
    try:
        import pynvml  # noqa: PLC0415 - optional, absent on machines without NVIDIA

        pynvml.nvmlInit()
        return pynvml.nvmlDeviceGetHandleByIndex(0)
    except Exception:  # pynvml raises many library-specific errors when no GPU exists.
        return None


class UsageSampler:
    """Sample CPU and GPU utilisation in a background thread."""

    def __init__(self, period_s: float = 0.25) -> None:
        self._period_s = period_s
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._rows: list[dict[str, float | None]] = []
        self._handle = _gpu_handle()

    def _gpu_percent(self) -> float | None:
        if self._handle is None:
            return None
        try:
            import pynvml  # noqa: PLC0415

            return float(pynvml.nvmlDeviceGetUtilizationRates(self._handle).gpu)
        except Exception:  # A failed sample becomes a gap and must not stop training.
            return None

    def _run(self) -> None:
        psutil.cpu_percent(interval=None)  # The first reading is always meaningless.
        process = psutil.Process()
        process.cpu_percent(interval=None)
        cores = psutil.cpu_count(logical=True) or 1
        while not self._stop.wait(self._period_s):
            self._rows.append(
                {
                    "timestamp": time.time(),
                    "cpu_util_pct": psutil.cpu_percent(interval=None),
                    "own_cpu_pct": process.cpu_percent(interval=None) / cores,
                    "gpu_util_pct": self._gpu_percent(),
                }
            )

    def __enter__(self) -> UsageSampler:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)

    def frame(self) -> pd.DataFrame:
        columns = ["timestamp", "cpu_util_pct", "own_cpu_pct", "gpu_util_pct"]
        table = pd.DataFrame(self._rows, columns=columns)
        return table if table["gpu_util_pct"].notna().any() else table.drop(columns="gpu_util_pct")


class _PdhValue(ctypes.Union):
    _fields_ = (
        ("longValue", ctypes.c_long),
        ("doubleValue", ctypes.c_double),
        ("largeValue", ctypes.c_longlong),
    )


class _PdhFormatted(ctypes.Structure):
    _fields_ = (("CStatus", ctypes.c_ulong), ("value", _PdhValue))


class _PackageCounter:
    """Direct, sub-millisecond read of the CPU package energy through Windows PDH.

    It uses a different route from CodeCarbon, which reads the device interface, so the
    two can check each other. Reads are fast enough to bracket each epoch tightly; a
    slower reader widens the window and makes correct energy values look wrong.
    """

    def __init__(self) -> None:
        self._pdh = ctypes.WinDLL("pdh")
        self._query = ctypes.c_void_p()
        self._counter = ctypes.c_void_p()
        status = self._pdh.PdhOpenQueryW(None, None, ctypes.byref(self._query))
        if status == 0:
            status = self._pdh.PdhAddEnglishCounterW(
                self._query, _PACKAGE_COUNTER, None, ctypes.byref(self._counter)
            )
        if status != 0:
            raise OSError(f"PDH could not open the energy counter: {status:#x}")

    def read_kwh(self) -> float:
        if self._pdh.PdhCollectQueryData(self._query) != 0:
            raise OSError("PDH could not collect the energy counter.")
        output = _PdhFormatted()
        status = self._pdh.PdhGetFormattedCounterValue(
            self._counter, _PDH_FMT_LARGE | _PDH_FMT_NOCAP100, None, ctypes.byref(output)
        )
        if status != 0:
            raise OSError(f"PDH could not format the energy counter: {status:#x}")
        return float(output.value.largeValue) / _PICOWATT_HOURS_PER_KWH


@lru_cache(maxsize=1)
def _package_counter() -> _PackageCounter | None:
    if sys.platform != "win32":
        return None
    try:
        return _PackageCounter()
    except OSError:
        return None


def read_package_counter_kwh() -> float | None:
    """Read the CPU package energy counter, or None where it is not available."""
    counter = _package_counter()
    if counter is None:
        return None
    try:
        return counter.read_kwh()
    except OSError:
        return None


def _os_name() -> str:
    """Name the OS. Python can report Windows 11 as release 10, so the build decides."""
    if sys.platform == "win32" and sys.getwindowsversion().build >= _WINDOWS_11_BUILD:
        return "Windows 11"
    return f"{platform.system()} {platform.release()}"


@contextmanager
def keep_awake() -> Iterator[None]:
    """Stop Windows sleeping during a run, for this process only. Settings do not change."""
    if sys.platform != "win32":
        yield
        return
    ctypes.windll.kernel32.SetThreadExecutionState(_ES_CONTINUOUS | _ES_SYSTEM_REQUIRED)
    try:
        yield
    finally:
        ctypes.windll.kernel32.SetThreadExecutionState(_ES_CONTINUOUS)


def power_plugged() -> bool | None:
    """Return True on mains power, False on battery, None when there is no battery."""
    try:
        battery = psutil.sensors_battery()
    except (AttributeError, OSError):  # Servers and cloud machines may have no power data.
        return None
    return None if battery is None else bool(battery.power_plugged)


def describe_machine(device: str) -> dict[str, Any]:
    """Describe hardware and software, without anything that identifies the owner."""
    memory_gb = psutil.virtual_memory().total / 1024**3
    machine: dict[str, Any] = {
        "os": _os_name(),
        "cpu_logical": psutil.cpu_count(logical=True),
        "cpu_physical": psutil.cpu_count(logical=False),
        "ram_gb": round(memory_gb, 1),
        "torch_threads": torch.get_num_threads(),
        "on_mains_power": power_plugged(),
    }
    if device == "cuda":
        machine["gpu"] = torch.cuda.get_device_name(0)
    return machine
