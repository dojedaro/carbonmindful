"""Read, validate and write run bundles.

A bundle is a zip holding summary.json, epochs.csv and, optionally, samples.csv.
Bundles are untrusted input: every value shown or calculated from one is checked here,
so the rest of the application can rely on it. See README.md for the schema.
"""

from __future__ import annotations

import io
import json
import math
import re
import zipfile
import zlib
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import Any

import numpy as np
import pandas as pd

SCHEMA_VERSION = 1
MAX_MEMBER_BYTES = 64 * 1024 * 1024
EPOCH_ENERGY_TOLERANCE = 0.05

_REQUIRED_TEXT = ("run_id", "model", "dataset", "device")
_OPTIONAL_TEXT = ("platform", "energy_source", "tracker_country_iso", "cpu_tracking_mode")
# Plain labels only. Excludes markup and path separators' parent form, because these
# values reach Markdown reports, table cells and download file names.
_SAFE_TEXT = re.compile(r"[\w .:()+,#=/-]{1,120}")
_SAFE_RUN_ID = re.compile(r"[\w.-]{1,120}")

_REQUIRED_EPOCH_COLUMNS = ("epoch", "duration_s", "val_acc")
_OPTIONAL_EPOCH_COLUMNS = (
    "samples_per_s",
    "train_s",
    "train_loss",
    "train_acc",
    "val_loss",
    "cpu_energy_kwh",
    "ram_energy_kwh",
    "gpu_energy_kwh",
    "package_counter_kwh",
)
_OPTIONAL_ENERGY_FIELDS = (
    "cpu_energy_kwh",
    "ram_energy_kwh",
    "gpu_energy_kwh",
    "package_counter_kwh",
)
_REQUIRED_SAMPLE_COLUMNS = ("timestamp", "cpu_util_pct")
_OPTIONAL_SAMPLE_COLUMNS = ("gpu_util_pct", "own_cpu_pct")


class BundleError(ValueError):
    """The bundle cannot be used. The message is safe to show to the user."""


@dataclass(frozen=True)
class RunBundle:
    """A validated run. ``energy_it_kwh`` always excludes PUE."""

    summary: dict[str, Any]
    epochs: pd.DataFrame
    samples: pd.DataFrame
    energy_it_kwh: float
    warnings: tuple[str, ...] = field(default=())

    @property
    def run_id(self) -> str:
        return str(self.summary["run_id"])

    @property
    def synthetic(self) -> bool:
        return self.summary.get("synthetic") is True

    @property
    def platform(self) -> str:
        return str(self.summary.get("platform", "unspecified"))

    @property
    def accuracy(self) -> float:
        """Test accuracy if recorded, else the last validation accuracy."""
        test = self.summary.get("test_accuracy")
        if test is not None:
            return float(test)
        return float(self.epochs.sort_values("epoch")["val_acc"].iloc[-1])

    @property
    def accuracy_label(self) -> str:
        if self.summary.get("test_accuracy") is not None:
            return "Test accuracy"
        return "Validation accuracy"


def _read_member(archive: zipfile.ZipFile, name: str, required: bool = True) -> bytes | None:
    matches = [
        info
        for info in archive.infolist()
        if not info.is_dir() and PurePosixPath(info.filename).name == name
    ]
    if not matches:
        if required:
            raise BundleError(f"The bundle has no {name}.")
        return None
    if len(matches) > 1:
        raise BundleError(f"The bundle has more than one {name}.")
    limit = f"{MAX_MEMBER_BYTES // 1024**2} MB"
    info = matches[0]
    if info.file_size > MAX_MEMBER_BYTES:
        raise BundleError(f"{name} is larger than the {limit} limit.")
    try:
        with archive.open(info) as handle:
            # The size in the header can be forged, so the read itself is capped too.
            data = handle.read(MAX_MEMBER_BYTES + 1)
    except (RuntimeError, NotImplementedError) as exc:
        raise BundleError(f"{name} is encrypted or uses an unsupported format.") from exc
    except (zipfile.BadZipFile, zlib.error, EOFError) as exc:
        raise BundleError(f"{name} is corrupt.") from exc
    if len(data) > MAX_MEMBER_BYTES:
        raise BundleError(f"{name} is larger than the {limit} limit.")
    return data


def _is_number(value: Any) -> bool:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:  # A JSON integer too large for a float.
        return False


def _validate_text(summary: Mapping[str, Any]) -> None:
    for key in (*_REQUIRED_TEXT, *_OPTIONAL_TEXT):
        if key not in summary:
            if key in _REQUIRED_TEXT:
                raise BundleError(f"summary.json is missing '{key}'.")
            continue
        value = summary[key]
        pattern = _SAFE_RUN_ID if key == "run_id" else _SAFE_TEXT
        if not isinstance(value, str) or not pattern.fullmatch(value) or ".." in value:
            raise BundleError(f"summary.json has an invalid value for '{key}'.")


def _validate_numbers(summary: Mapping[str, Any]) -> None:
    minimums = {"duration_s": 0.0, "energy_kwh": 0.0, "tracker_pue": 1.0}
    for key, minimum in minimums.items():
        if key not in summary:
            raise BundleError(f"summary.json is missing '{key}'.")
        if not _is_number(summary[key]) or summary[key] < minimum:
            raise BundleError(f"summary.json has an invalid value for '{key}'.")
    for key in ("tracker_emissions_kg", *_OPTIONAL_ENERGY_FIELDS):
        value = summary.get(key)
        if value is not None and (not _is_number(value) or value < 0):
            raise BundleError(f"summary.json has an invalid value for '{key}'.")
    test = summary.get("test_accuracy")
    if test is not None and (not _is_number(test) or not 0 <= test <= 1):
        raise BundleError("test_accuracy must be a fraction between 0 and 1.")


def _validate_summary(summary: Any) -> dict[str, Any]:
    if not isinstance(summary, dict):
        raise BundleError("summary.json must contain a JSON object.")
    for key in ("schema_version", "seed"):
        if key not in summary:
            raise BundleError(f"summary.json is missing '{key}'.")
        if isinstance(summary[key], bool) or not isinstance(summary[key], int):
            raise BundleError(f"summary.json has an invalid value for '{key}'.")
    if not 0 <= summary["seed"] < 2**63:
        raise BundleError("summary.json has an invalid value for 'seed'.")
    if summary["schema_version"] != SCHEMA_VERSION:
        raise BundleError(
            f"Unsupported schema_version {summary['schema_version']}; expected {SCHEMA_VERSION}."
        )
    _validate_text(summary)
    _validate_numbers(summary)
    return summary


def _read_csv(
    data: bytes, name: str, required: tuple[str, ...], optional: tuple[str, ...]
) -> pd.DataFrame:
    try:
        frame = pd.read_csv(io.BytesIO(data))
    except (ValueError, UnicodeDecodeError, pd.errors.ParserError) as exc:
        raise BundleError(f"{name} is not a readable CSV file.") from exc
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise BundleError(f"{name} is missing columns: {', '.join(missing)}.")
    for column in required:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
        if frame[column].isna().any():
            raise BundleError(f"{name} has non-numeric values in '{column}'.")
        if not np.isfinite(frame[column].to_numpy(dtype=float)).all():
            raise BundleError(f"{name} has values in '{column}' that are not finite.")
    for column in optional:
        if column in frame:
            numbers = pd.to_numeric(frame[column], errors="coerce")
            frame[column] = numbers.where(np.isfinite(numbers.to_numpy(dtype=float)))
    return frame


def _validate_epochs(epochs: pd.DataFrame) -> None:
    if epochs.empty:
        raise BundleError("epochs.csv has no rows.")
    if (epochs["epoch"] % 1 != 0).any():
        raise BundleError("epochs.csv must number epochs with whole numbers.")
    if epochs["epoch"].duplicated().any():
        raise BundleError("epochs.csv has duplicate epoch numbers.")
    if (epochs["duration_s"] < 0).any():
        raise BundleError("epochs.csv has negative durations.")
    if not epochs["val_acc"].between(0, 1).all():
        raise BundleError("val_acc in epochs.csv must be a fraction between 0 and 1.")


def parse_bundle(data: bytes) -> RunBundle:
    """Parse zip bytes into a validated bundle, or raise :class:`BundleError`."""
    try:
        archive = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile as exc:
        raise BundleError("The file is not a valid zip archive.") from exc
    with archive:
        raw_summary = _read_member(archive, "summary.json")
        raw_epochs = _read_member(archive, "epochs.csv")
        raw_samples = _read_member(archive, "samples.csv", required=False)

    if raw_summary is None or raw_epochs is None:  # pragma: no cover - required above
        raise BundleError("The bundle is incomplete.")
    try:
        summary = _validate_summary(json.loads(raw_summary))
    except (json.JSONDecodeError, UnicodeDecodeError, RecursionError) as exc:
        raise BundleError("summary.json is not valid JSON.") from exc

    epochs = _read_csv(raw_epochs, "epochs.csv", _REQUIRED_EPOCH_COLUMNS, _OPTIONAL_EPOCH_COLUMNS)
    _validate_epochs(epochs)
    samples = (
        _read_csv(raw_samples, "samples.csv", _REQUIRED_SAMPLE_COLUMNS, _OPTIONAL_SAMPLE_COLUMNS)
        if raw_samples is not None
        else pd.DataFrame(columns=list(_REQUIRED_SAMPLE_COLUMNS))
    )

    warnings: list[str] = []
    tracker_pue = float(summary["tracker_pue"])
    if tracker_pue != 1.0:
        warnings.append(
            f"The bundle records a tracker PUE of {tracker_pue:g}. Its energy is assumed to "
            "include that PUE and was divided by it to recover IT energy, so PUE is not "
            "counted twice. Verify this for the tracker version used, or record with PUE 1.0."
        )
    if "energy_kwh" in epochs:
        energy = pd.to_numeric(epochs["energy_kwh"], errors="coerce")
        values = energy.to_numpy(dtype=float)
        if not np.isfinite(values).all() or (values < 0).any():
            raise BundleError("energy_kwh in epochs.csv must be finite, non-negative numbers.")
        epochs = epochs.assign(energy_kwh=energy, energy_it_kwh=energy / tracker_pue)
        total = float(summary["energy_kwh"])
        if total > 0 and abs(float(energy.sum()) - total) / total > EPOCH_ENERGY_TOLERANCE:
            warnings.append(
                f"Per-epoch energy sums to {float(energy.sum()):.6g} kWh but the run total is "
                f"{total:.6g} kWh. The gap is over {EPOCH_ENERGY_TOLERANCE:.0%}; it may reflect "
                "work outside the epochs, such as evaluation or setup."
            )
    else:
        warnings.append("epochs.csv has no per-epoch energy, so the trade-off view is unavailable.")

    return RunBundle(
        summary=summary,
        epochs=epochs,
        samples=samples,
        energy_it_kwh=float(summary["energy_kwh"]) / tracker_pue,
        warnings=tuple(warnings),
    )


def build_bundle(
    summary: Mapping[str, Any], epochs: pd.DataFrame, samples: pd.DataFrame | None = None
) -> bytes:
    """Write a bundle as zip bytes. The experiment scripts and the demo both use this."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("summary.json", json.dumps(summary, indent=2, allow_nan=False))
        archive.writestr("epochs.csv", epochs.to_csv(index=False))
        if samples is not None:
            archive.writestr("samples.csv", samples.to_csv(index=False))
    data = buffer.getvalue()
    parse_bundle(data)  # A bundle that cannot be read back must fail when it is written.
    return data
